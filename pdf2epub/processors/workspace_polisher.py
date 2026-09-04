"""Run Codex directly over a complete OCR Markdown workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
import tiktoken

from ..core.persistence import ResultPersistence
from ..core.phase import PartBasedLoader
from ..core.tracking import AttemptRecord, ProcessingTracker
from ..utils.safety import ProcessLockError, exclusive_process_lock
from .utils.image_restore import extract_images_from_markdown


class WorkspacePolishError(RuntimeError):
    """Raised when the Codex process itself cannot complete."""


@dataclass(frozen=True)
class WorkspacePolishResult:
    total: int
    completed: int
    model: str
    state_path: Path
    validation_path: Path


_TOKENIZER = tiktoken.get_encoding("o200k_base")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+\t")
_PATH_LINE_RE = re.compile(r"^(?:[^:\n]+\.md:)\d+:")
_PLAIN_LINE_NUMBER_RE = re.compile(r"^\d+:")
_UNIFIED_DIFF_OUTPUT_RE = re.compile(
    r"(?m)^--- [^\n]+\n\+\+\+ [^\n]+\n@@ "
)
_APPLY_PATCH_OUTPUT_RE = re.compile(
    r"(?ms)^\*\*\* Begin Patch\s*$.*?^\*\*\* "
    r"(?:Update|Add|Delete) File: [^\n]+$.*?^@@(?: [^\n]*)?$.*?"
    r"^\*\*\* End Patch\s*$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspacePolishError(f"Cannot read Codex state at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkspacePolishError(f"Invalid Codex state at {path}")
    return value


def _prompt(content_type: str, *, resume: bool) -> str:
    kind = {
        "academic": "academic",
        "japanese": "Japanese",
        "general": "general",
    }.get(content_type, "book")
    continuation = "Continue the work already in this folder." if resume else ""
    return f"""Polish this entire {kind} OCR book. {continuation}

First analyze the complete folder and the book as a whole. Choose your own
reading order, tools, searches, scripts, working method, and context compaction.
Then inspect every Markdown file and fix every issue you can find in every file:
OCR line wraps, broken or wrongly joined paragraphs, split words, hyphenation,
punctuation, spacing, malformed Markdown, page/column order, grammar, and prose
damaged by OCR. Repair cross-file and cross-page continuity wherever needed.
Preserve the author's content and meaning; this is cleanup, not translation or
summary.

Preserve genuine Markdown footnotes: `[^key]` is an inline reference and
`[^key]: text` is its definition. Keep every genuine key unchanged and do not
turn either form into ordinary prose or a raw superscript. Repair footnote
layout when needed: join a clearly identified continuation to its definition,
even when page or column ordering has displaced that continuation elsewhere in
the file; use semantic continuity rather than mere adjacency to reconnect it,
and remove exact duplicate references or definitions only when they clearly
repeat the same physical-page material across adjacent files.
Never invent a footnote or key.

Make sure the actual text of every Markdown file enters your context; seeing a
filename or metadata is not the same as reading the file. Compact normally and
continue as often as needed.

Image references and printed captions appear only as protected comments. Do not
rewrite those comments, but move them if needed. Work directly on the Markdown
files. Do not stop after sampling or fixing representative files. Continue until
you have examined the whole folder and completed every repair you can make.
"""


def _protect_fragments(
    directory: Path, fragments: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    files = list(directory.glob("*.md"))
    unique = sorted({value for value in fragments if value}, key=len, reverse=True)
    for index, fragment in enumerate(unique, start=1):
        placeholder = f"<!--PDF2EPUB_PROTECTED_FRAGMENT_{index:06d}-->"
        occurrences = 0
        for path in files:
            content = path.read_text(encoding="utf-8")
            count = content.count(fragment)
            if count:
                path.write_text(content.replace(fragment, placeholder), encoding="utf-8")
                occurrences += count
        if occurrences:
            records.append(
                {
                    "placeholder": placeholder,
                    "content": fragment,
                    "occurrences": occurrences,
                }
            )
    return records


def _restore_fragments(directory: Path, records: list[dict[str, Any]]) -> None:
    """Restore protected comments wherever Codex left or moved them."""
    for record in reversed(records):
        for path in directory.glob("*.md"):
            content = path.read_text(encoding="utf-8")
            if record["placeholder"] in content:
                path.write_text(
                    content.replace(record["placeholder"], record["content"]),
                    encoding="utf-8",
                )


def _protected_fragment_mismatches(
    directory: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return count mismatches without exposing protected caption/image text."""
    contents = [
        path.read_text(encoding="utf-8")
        for path in directory.glob("*.md")
        if path.is_file()
    ]
    mismatches = []
    for record in records:
        actual = sum(content.count(record["placeholder"]) for content in contents)
        expected = int(record["occurrences"])
        if actual != expected:
            mismatches.append(
                {
                    "placeholder": record["placeholder"],
                    "expected": expected,
                    "actual": actual,
                }
            )
    return mismatches


def _hide_fragments(directory: Path, records: list[dict[str, Any]]) -> None:
    """Mechanically re-mask fragments before every Codex turn."""
    for record in records:
        for path in directory.glob("*.md"):
            content = path.read_text(encoding="utf-8")
            if record["content"] in content:
                path.write_text(
                    content.replace(record["content"], record["placeholder"]),
                    encoding="utf-8",
                )


def _initialize_workspace(
    *,
    source_dir: Path,
    run_dir: Path,
    snapshot_dir: Path,
    protected_fragments: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    git = shutil.which("git")
    if not git:
        raise WorkspacePolishError("git is required by the Codex workspace")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise WorkspacePolishError(f"Codex workspace is not empty: {run_dir}")
    source_files = sorted(path for path in source_dir.glob("*.md") if path.is_file())
    for source in source_files:
        shutil.copy2(source, run_dir / source.name)
    fragments = list(protected_fragments)
    for path in run_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        fragments.extend(raw for raw, _start, _end in extract_images_from_markdown(text))
    records = _protect_fragments(run_dir, fragments)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in run_dir.glob("*.md"):
        shutil.copy2(path, snapshot_dir / path.name)
    subprocess.run(
        [git, "init", "--quiet", str(run_dir)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return [path.name for path in source_files], records


def _snapshot_input(
    *, source_dir: Path, snapshot_dir: Path, records: list[dict[str, Any]]
) -> None:
    """Create the immutable, mechanically masked input used by the audit."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if any(snapshot_dir.iterdir()):
        raise WorkspacePolishError(f"Input snapshot is not empty: {snapshot_dir}")
    for source in source_dir.glob("*.md"):
        if source.is_file():
            shutil.copy2(source, snapshot_dir / source.name)
    _hide_fragments(snapshot_dir, records)


def _session_command_outputs(
    events_path: Path, session_id: str
) -> list[tuple[str, str]]:
    outputs: list[tuple[str, str]] = []
    active_session = False
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            active_session = event.get("thread_id") == session_id
        if not active_session or event.get("type") != "item.completed":
            continue
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        output = item.get("aggregated_output")
        if item.get("type") == "command_execution" and isinstance(output, str):
            command = item.get("command")
            outputs.append((command if isinstance(command, str) else "", output))
    return outputs


def _canonical_content_line(line: str) -> str:
    """Remove display-only differences while retaining the line's content."""
    line = _ANSI_ESCAPE_RE.sub("", line)
    return " ".join(unicodedata.normalize("NFC", line).split())


def _displayed_content_line(
    command: str, line: str, *, diff_output: bool = False
) -> str:
    """Undo common decorations added while displaying a file in a terminal."""
    uses_numbered_output = bool(
        re.search(r"(?:^|[^\w-])nl(?:\s|<)", command)
        or re.search(
            r"(?:^|[^\w-])cat\s+(?:-\w*n\w*|--number(?:-nonblank)?)(?:\s|$)",
            command,
        )
    )
    uses_numbered_search = bool(
        re.search(r"(?:^|[^\w-])(?:rg|grep)\s", command)
        and re.search(r"(?:^|\s)(?:-\w*n\w*|--line-number)(?:\s|$)", command)
    )
    if uses_numbered_output and _NUMBERED_LINE_RE.match(line):
        return _NUMBERED_LINE_RE.sub("", line, count=1)
    if uses_numbered_search and _PATH_LINE_RE.match(line):
        return _PATH_LINE_RE.sub("", line, count=1)
    if uses_numbered_search and _PLAIN_LINE_NUMBER_RE.match(line):
        return _PLAIN_LINE_NUMBER_RE.sub("", line, count=1)
    uses_unified_diff = bool(
        diff_output
        or re.search(r"(?:^|[^\w-])git(?:\s+\S+)*?\s+diff(?:\s|$)", command)
        or re.search(
            r"(?:^|[^\w-])diff\s+(?:-\w*u\w*|--unified(?:=\d+)?)(?:\s|$)",
            command,
        )
    )
    if uses_unified_diff and line[:1] in {" ", "+", "-"}:
        if not line.startswith(("+++", "---")):
            return line[1:]
    return line


def _input_token_coverage(
    *, snapshot_dir: Path, events_path: Path, session_id: str
) -> dict[str, Any]:
    """Measure source content that actually entered Codex as command output.

    Matching is exact after Unicode/whitespace normalization and removal of
    terminal display decorations such as ``nl -ba`` line numbers. Each output
    line occurrence is consumed at most once across the whole book, so repeated
    source prose cannot be credited from a single displayed occurrence.
    """
    outputs = _session_command_outputs(events_path, session_id)
    displayed_lines: Counter[str] = Counter()
    output_characters = 0
    for command, output in outputs:
        output_characters += len(output)
        diff_output = bool(
            _UNIFIED_DIFF_OUTPUT_RE.search(output)
            or _APPLY_PATCH_OUTPUT_RE.search(output)
        )
        for line in output.splitlines():
            canonical = _canonical_content_line(
                _displayed_content_line(
                    command, line, diff_output=diff_output
                )
            )
            if canonical:
                displayed_lines[canonical] += 1

    files: dict[str, dict[str, Any]] = {}
    unread_files: list[str] = []
    total_tokens = 0
    covered_tokens = 0
    for path in sorted(snapshot_dir.glob("*.md")):
        source_tokens = 0
        matched_tokens = 0
        unmatched_lines = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            canonical = _canonical_content_line(line)
            if not canonical:
                continue
            line_tokens = len(_TOKENIZER.encode(canonical))
            source_tokens += line_tokens
            if displayed_lines[canonical] > 0:
                displayed_lines[canonical] -= 1
                matched_tokens += line_tokens
            else:
                unmatched_lines += 1
        ratio = matched_tokens / source_tokens if source_tokens else 1.0
        files[path.name] = {
            "source_tokens": source_tokens,
            "covered_tokens": matched_tokens,
            "coverage": ratio,
            "unmatched_content_lines": unmatched_lines,
        }
        total_tokens += source_tokens
        covered_tokens += matched_tokens
        if matched_tokens != source_tokens:
            unread_files.append(path.name)
    return {
        "encoding": "o200k_base",
        "measurement": "occurrence_aware_canonical_content_lines",
        "required_coverage": 1.0,
        "command_outputs": len(outputs),
        "command_output_characters": output_characters,
        "source_tokens": total_tokens,
        "covered_tokens": covered_tokens,
        "coverage": covered_tokens / total_tokens if total_tokens else 1.0,
        "files": files,
        "below_threshold": unread_files,
        "complete": not unread_files,
        "audited_at": time.time(),
    }


def _coverage_followup_prompt(files: list[str]) -> str:
    names = "\n".join(f"- {name}" for name in files)
    return f"""Continue the same complete-book polish. The input-token audit shows
that the actual text of these files has not yet appeared sufficiently in your
tool-output context:

{names}

Read the complete actual contents of every listed file into your context, not
just filenames or metadata. Fix every remaining OCR, line-break, paragraph,
punctuation, Markdown, grammar, and continuity issue you find. Use your own
method, order, tools, and normal context compaction. Also finish anything else
you judge incomplete. Image and caption comments are protected; leave them
intact. Complete only when the remaining work is done.
"""


def _protected_fragment_followup_prompt(
    mismatches: list[dict[str, Any]], snapshot_dir: Path
) -> str:
    counts = "\n".join(
        f"- {item['placeholder']}: expected {item['expected']}, found {item['actual']}"
        for item in mismatches
    )
    return f"""Continue the same complete-book polish. Some protected image or
printed-caption placeholder comments were lost or duplicated:

{counts}

Compare the Markdown files with the immutable masked input snapshot at
{snapshot_dir}. Restore each listed placeholder comment to the correct semantic
location and exact expected count without changing the placeholder. Preserve all
useful polishing already completed. Then inspect the workspace for anything else
that remains unfinished and complete it.
"""


def _pending_followup_prompt(
    pending: dict[str, Any], *, snapshot_dir: Path
) -> str:
    kind = pending.get("kind")
    if kind == "coverage":
        return _coverage_followup_prompt(list(pending.get("files") or []))
    if kind == "file_set":
        return _file_set_followup_prompt(
            missing=list(pending.get("missing") or []),
            unexpected=list(pending.get("unexpected") or []),
        )
    if kind == "protected":
        return _protected_fragment_followup_prompt(
            list(pending.get("mismatches") or []), snapshot_dir
        )
    raise WorkspacePolishError(f"Unknown pending Codex follow-up: {kind}")


def _workspace_file_set(
    directory: Path, expected_files: list[str]
) -> tuple[list[str], list[str]]:
    """Return missing and unexpected Markdown names for the EPUB namespace."""
    expected = set(expected_files)
    actual = {
        path.name for path in Path(directory).glob("*.md") if path.is_file()
    }
    return sorted(expected - actual), sorted(actual - expected)


def _file_set_followup_prompt(
    *, missing: list[str], unexpected: list[str]
) -> str:
    missing_names = "\n".join(f"- {name}" for name in missing) or "- none"
    unexpected_names = "\n".join(f"- {name}" for name in unexpected) or "- none"
    return f"""Continue the same complete-book polish. The EPUB structure consumes
the original Markdown filename namespace, but your last turn changed that
namespace. The framework has restored every missing file from the immutable
input snapshot so no source content is lost.

Missing names that were restored:
{missing_names}

Unexpected names:
{unexpected_names}

Inspect the restored files and the whole workspace. Polish all restored content
and reconcile any true overlap without deleting or renaming any expected file.
If an unexpected file exists, preserve all useful content while moving it into
the appropriate expected files, then remove only that unexpected filename.
The contents and page order may move wherever the book requires; only the
original set of Markdown filenames must remain. Image and caption comments are
protected. Finish every remaining repair before completing.
"""


def _command(
    *, codex: str, model: str, run_dir: Path, prompt: str, session_id: str | None
) -> list[str]:
    if session_id:
        return [codex, "exec", "resume", "-m", model, "--json", session_id, prompt]
    return [
        codex,
        "exec",
        "-m",
        model,
        "-C",
        str(run_dir),
        "--sandbox",
        "workspace-write",
        "--json",
        prompt,
    ]


def _resolve_codex_binary(codex_binary: str) -> str:
    """Resolve Codex only when another model turn is actually required."""
    has_path_separator = any(
        separator and separator in codex_binary
        for separator in (os.sep, os.altsep)
    )
    if has_path_separator:
        codex = str(Path(codex_binary).resolve())
    else:
        codex = shutil.which(codex_binary)
    if not codex or not Path(codex).is_file():
        raise WorkspacePolishError(f"Codex CLI is not executable: {codex_binary}")
    return codex


def _run_codex(
    *,
    command: list[str],
    run_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    events_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    state["status"] = "running"
    state["attempts"] = int(state.get("attempts", 0)) + 1
    state["last_started_at"] = time.time()
    _atomic_write_json(state_path, state)
    saw_completion = False
    process: subprocess.Popen[str] | None = None
    try:
        with (
            events_path.open("a", encoding="utf-8") as events,
            stdout_path.open("a", encoding="utf-8") as stdout_log,
            stderr_path.open("a", encoding="utf-8") as stderr_log,
        ):
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_log,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stdout_log.write(line)
                stdout_log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                if event.get("type") == "thread.started" and event.get("thread_id"):
                    state["session_id"] = event["thread_id"]
                    _atomic_write_json(state_path, state)
                elif event.get("type") == "turn.completed":
                    saw_completion = True
                    if isinstance(event.get("usage"), dict):
                        state.setdefault("usage", []).append(event["usage"])
                elif event.get("type") == "turn.failed":
                    state["last_error"] = str(event.get("error", "Codex turn failed"))
            return_code = process.wait()
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "runner_failed"
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        state["last_finished_at"] = time.time()
        _atomic_write_json(state_path, state)
        raise
    state["last_finished_at"] = time.time()
    if return_code != 0 or not saw_completion:
        state["status"] = "codex_failed"
        _atomic_write_json(state_path, state)
        raise WorkspacePolishError(f"Codex failed; state retained at {state_path}")
    state["status"] = "codex_complete"
    state.pop("last_error", None)
    _atomic_write_json(state_path, state)


def _publish(
    *, output_dir: Path, state_dir: Path, run_dir: Path, model: str, created_at: float
) -> list[str]:
    files = sorted(path for path in run_dir.glob("*.md") if path.is_file())
    staging = state_dir / f"publish_staging_{time.time_ns()}"
    persistence = ResultPersistence(staging)
    for path in files:
        persistence.save_raw(path.stem, path.read_text(encoding="utf-8"))
        persistence.promote_to_validated(path.stem)

    existing = [
        output_dir / "raw",
        output_dir / "validated",
        output_dir / "processing_tracker.json",
    ]
    if any(path.exists() for path in existing):
        history = state_dir / "published_history" / str(time.time_ns())
        history.mkdir(parents=True)
        for path in existing:
            if path.exists():
                os.replace(path, history / path.name)
    os.replace(persistence.raw_dir, output_dir / "raw")
    os.replace(persistence.validated_dir, output_dir / "validated")

    tracker = ProcessingTracker(
        output_dir / "processing_tracker.json",
        "CodexWorkspacePolisher",
        file_checker=ResultPersistence(output_dir).has_validated,
    )
    duration = max(0.0, time.time() - created_at)
    for index, path in enumerate(files):
        if tracker.is_unit_complete(path.stem):
            continue
        tracker.record_attempt(
            path.stem,
            AttemptRecord(
                timestamp=time.time(),
                status="completed",
                model=f"codex-cli:{model}",
                duration_seconds=duration if index == 0 else 0.0,
            ),
        )
    return [path.name for path in files]


def run_workspace_polish(
    *,
    input_dir: Path,
    output_dir: Path,
    model: str = "gpt-5.6-terra",
    content_type: str = "auto",
    resume: bool = False,
    codex_binary: str = "codex",
    protected_fragments: list[str] | None = None,
    run_root: Path | None = None,
) -> WorkspacePolishResult:
    """Hand the entire folder to one ordinary, resumable Codex session."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    state_dir = output_dir / "workspace_agent"
    state_path = state_dir / "state.json"
    validation_path = state_dir / "validation.json"
    fragments = list(protected_fragments or [])

    try:
        lock = exclusive_process_lock(output_dir / ".polish.lock", "polish")
        lock.__enter__()
    except ProcessLockError as exc:
        raise WorkspacePolishError(str(exc)) from exc
    try:
        units = PartBasedLoader().load_units(input_dir, "*.md")
        if not units:
            raise WorkspacePolishError(f"No Markdown files found in {input_dir}")

        if resume:
            if not state_path.is_file():
                raise WorkspacePolishError(f"No Codex state to resume at {state_path}")
            state = _load_json(state_path)
            if state.get("version") not in {2, 3}:
                raise WorkspacePolishError(
                    "This state predates the direct Codex workspace run"
                )
        else:
            if state_path.exists():
                raise WorkspacePolishError(
                    f"Codex state already exists; use --resume: {state_path}"
                )
            run_dir = state_dir / "direct_workspace" if run_root is None else Path(run_root).resolve()
            expected, records = _initialize_workspace(
                source_dir=input_dir,
                run_dir=run_dir,
                snapshot_dir=state_dir / "input_snapshot",
                protected_fragments=fragments,
            )
            state = {
                "version": 3,
                "status": "initialized",
                "model": model,
                "content_type": content_type,
                "run_dir": str(run_dir),
                "input_snapshot_dir": str(state_dir / "input_snapshot"),
                "expected_files": expected,
                "protected_fragments": records,
                "created_at": time.time(),
                "attempts": 0,
            }
            _atomic_write_json(state_path, state)

        if state.get("model") != model:
            raise WorkspacePolishError(
                f"Resume model mismatch: {state.get('model')} != {model}"
            )
        if state.get("status") == "published":
            published = list((output_dir / "validated").glob("*.md"))
            missing, unexpected = _workspace_file_set(
                output_dir / "validated", state.get("expected_files") or []
            )
            if not missing and not unexpected:
                return WorkspacePolishResult(
                    len(published), len(published), model, state_path, validation_path
                )
            # Older runs could publish an incomplete namespace. Resume their
            # retained session and workspace instead of accepting that output.
            state["status"] = "codex_complete"
            _atomic_write_json(state_path, state)

        run_dir = Path(state["run_dir"])
        snapshot_value = state.get("input_snapshot_dir")
        if snapshot_value:
            snapshot_dir = Path(snapshot_value)
        else:
            snapshot_dir = state_dir / "input_snapshot"
            _snapshot_input(
                source_dir=input_dir,
                snapshot_dir=snapshot_dir,
                records=state.get("protected_fragments") or [],
            )
            state["input_snapshot_dir"] = str(snapshot_dir)
            state["version"] = 3
            _atomic_write_json(state_path, state)
        pending_followup = state.get("pending_followup")
        pending_coverage_files = state.get("pending_coverage_files") or []
        if not pending_followup and pending_coverage_files:
            pending_followup = {
                "kind": "coverage",
                "files": pending_coverage_files,
            }
            state["pending_followup"] = pending_followup
            _atomic_write_json(state_path, state)

        # An invalid retained workspace may have been repaired manually between
        # runs. Recheck it before spending another model turn; otherwise resume
        # the same targeted repair rather than falling back to a generic prompt.
        if state.get("status") == "invalid_file_set":
            missing, unexpected = _workspace_file_set(
                run_dir, state.get("expected_files") or []
            )
            if not missing and not unexpected:
                state["status"] = "codex_complete"
                state.pop("file_set_mismatch", None)
                state.pop("pending_followup", None)
                pending_followup = None
            else:
                pending_followup = {
                    "kind": "file_set",
                    "missing": missing,
                    "unexpected": unexpected,
                }
                state["pending_followup"] = pending_followup
            _atomic_write_json(state_path, state)
        elif state.get("status") == "invalid_protected_fragments":
            _hide_fragments(run_dir, state.get("protected_fragments") or [])
            protected_mismatches = _protected_fragment_mismatches(
                run_dir, state.get("protected_fragments") or []
            )
            if not protected_mismatches:
                state["status"] = "codex_complete"
                state.pop("protected_fragment_mismatches", None)
                state.pop("pending_followup", None)
                pending_followup = None
            else:
                pending_followup = {
                    "kind": "protected",
                    "mismatches": protected_mismatches,
                }
                state["pending_followup"] = pending_followup
            _atomic_write_json(state_path, state)

        if state.get("status") != "codex_complete":
            _hide_fragments(run_dir, state.get("protected_fragments") or [])
            _run_codex(
                command=_command(
                    codex=_resolve_codex_binary(codex_binary),
                    model=model,
                    run_dir=run_dir,
                    prompt=(
                        _pending_followup_prompt(
                            pending_followup, snapshot_dir=snapshot_dir
                        )
                        if pending_followup
                        else _prompt(content_type, resume=bool(state.get("session_id")))
                    ),
                    session_id=state.get("session_id"),
                ),
                run_dir=run_dir,
                state=state,
                state_path=state_path,
                events_path=state_dir / "codex_events.jsonl",
                stdout_path=state_dir / "codex_stdout.log",
                stderr_path=state_dir / "codex_stderr.log",
            )
        state = _load_json(state_path)
        if state.get("status") == "codex_complete" and state.get(
            "pending_followup"
        ):
            state.pop("pending_followup", None)
            state.pop("pending_coverage_files", None)
            _atomic_write_json(state_path, state)
        session_id = state.get("session_id")
        if not session_id:
            raise WorkspacePolishError("Codex completed without a resumable session id")
        missing, unexpected = _workspace_file_set(
            run_dir, state.get("expected_files") or []
        )
        if missing or unexpected:
            if int(state.get("file_set_followups", 0)) >= 1:
                state["status"] = "invalid_file_set"
                state["file_set_mismatch"] = {
                    "missing": missing,
                    "unexpected": unexpected,
                }
                _atomic_write_json(state_path, state)
                _atomic_write_json(
                    validation_path,
                    {
                        "valid": False,
                        "reason": "Markdown filename set changed",
                        "missing": missing,
                        "unexpected": unexpected,
                        "validated_at": time.time(),
                    },
                )
                raise WorkspacePolishError(
                    "Codex changed the required Markdown filename set after repair; "
                    f"state retained at {state_path}"
                )
            for name in missing:
                source = snapshot_dir / name
                if not source.is_file():
                    raise WorkspacePolishError(
                        f"Missing immutable snapshot file required for repair: {name}"
                    )
                shutil.copy2(source, run_dir / name)
            state["file_set_followups"] = 1
            state["file_set_mismatch"] = {
                "missing": missing,
                "unexpected": unexpected,
            }
            state["pending_followup"] = {
                "kind": "file_set",
                "missing": missing,
                "unexpected": unexpected,
            }
            _atomic_write_json(state_path, state)
            _hide_fragments(run_dir, state.get("protected_fragments") or [])
            _run_codex(
                command=_command(
                    codex=_resolve_codex_binary(codex_binary),
                    model=model,
                    run_dir=run_dir,
                    prompt=_file_set_followup_prompt(
                        missing=missing, unexpected=unexpected
                    ),
                    session_id=session_id,
                ),
                run_dir=run_dir,
                state=state,
                state_path=state_path,
                events_path=state_dir / "codex_events.jsonl",
                stdout_path=state_dir / "codex_stdout.log",
                stderr_path=state_dir / "codex_stderr.log",
            )
            state = _load_json(state_path)
            state.pop("pending_followup", None)
            _atomic_write_json(state_path, state)
            missing, unexpected = _workspace_file_set(
                run_dir, state.get("expected_files") or []
            )
            if missing or unexpected:
                state["status"] = "invalid_file_set"
                state["file_set_mismatch"] = {
                    "missing": missing,
                    "unexpected": unexpected,
                }
                _atomic_write_json(state_path, state)
                _atomic_write_json(
                    validation_path,
                    {
                        "valid": False,
                        "reason": "Markdown filename set changed",
                        "missing": missing,
                        "unexpected": unexpected,
                        "validated_at": time.time(),
                    },
                )
                raise WorkspacePolishError(
                    "Codex changed the required Markdown filename set after repair; "
                    f"state retained at {state_path}"
                )
            state.pop("file_set_mismatch", None)
            _atomic_write_json(state_path, state)
        coverage_report = _input_token_coverage(
            snapshot_dir=snapshot_dir,
            events_path=state_dir / "codex_events.jsonl",
            session_id=session_id,
        )
        if (
            coverage_report["below_threshold"]
            and int(state.get("coverage_followups", 0)) < 1
        ):
            state["coverage_followups"] = 1
            state["pending_coverage_files"] = coverage_report["below_threshold"]
            state["pending_followup"] = {
                "kind": "coverage",
                "files": coverage_report["below_threshold"],
            }
            _atomic_write_json(state_path, state)
            logger.info(
                "Input-token coverage follow-up for "
                f"{len(coverage_report['below_threshold'])} Markdown files"
            )
            _hide_fragments(run_dir, state.get("protected_fragments") or [])
            _run_codex(
                command=_command(
                    codex=_resolve_codex_binary(codex_binary),
                    model=model,
                    run_dir=run_dir,
                    prompt=_coverage_followup_prompt(
                        coverage_report["below_threshold"]
                    ),
                    session_id=session_id,
                ),
                run_dir=run_dir,
                state=state,
                state_path=state_path,
                events_path=state_dir / "codex_events.jsonl",
                stdout_path=state_dir / "codex_stdout.log",
                stderr_path=state_dir / "codex_stderr.log",
            )
            state = _load_json(state_path)
            state.pop("pending_followup", None)
            state.pop("pending_coverage_files", None)
            _atomic_write_json(state_path, state)
            coverage_report = _input_token_coverage(
                snapshot_dir=snapshot_dir,
                events_path=state_dir / "codex_events.jsonl",
                session_id=session_id,
            )
        state.pop("pending_coverage_files", None)
        state["input_token_coverage_complete"] = coverage_report["complete"]
        _atomic_write_json(state_dir / "input_token_coverage.json", coverage_report)
        _atomic_write_json(state_path, state)

        # A prior publish attempt may have restored the real fragments before
        # failing. Normalize them back to placeholders before count validation.
        _hide_fragments(run_dir, state.get("protected_fragments") or [])
        protected_mismatches = _protected_fragment_mismatches(
            run_dir, state.get("protected_fragments") or []
        )
        if protected_mismatches and int(
            state.get("protected_fragment_followups", 0)
        ) < 1:
            state["protected_fragment_followups"] = 1
            state["protected_fragment_mismatches"] = protected_mismatches
            state["pending_followup"] = {
                "kind": "protected",
                "mismatches": protected_mismatches,
            }
            _atomic_write_json(state_path, state)
            _run_codex(
                command=_command(
                    codex=_resolve_codex_binary(codex_binary),
                    model=model,
                    run_dir=run_dir,
                    prompt=_protected_fragment_followup_prompt(
                        protected_mismatches, snapshot_dir
                    ),
                    session_id=session_id,
                ),
                run_dir=run_dir,
                state=state,
                state_path=state_path,
                events_path=state_dir / "codex_events.jsonl",
                stdout_path=state_dir / "codex_stdout.log",
                stderr_path=state_dir / "codex_stderr.log",
            )
            state = _load_json(state_path)
            state.pop("pending_followup", None)
            _atomic_write_json(state_path, state)
            protected_mismatches = _protected_fragment_mismatches(
                run_dir, state.get("protected_fragments") or []
            )
        if protected_mismatches:
            state["status"] = "invalid_protected_fragments"
            state["protected_fragment_mismatches"] = protected_mismatches
            _atomic_write_json(state_path, state)
            _atomic_write_json(
                validation_path,
                {
                    "valid": False,
                    "reason": "Protected image/caption placeholder count changed",
                    "mismatches": protected_mismatches,
                    "validated_at": time.time(),
                },
            )
            raise WorkspacePolishError(
                "Codex did not preserve protected image/caption placeholders; "
                f"state retained at {state_path}"
            )
        state.pop("protected_fragment_mismatches", None)
        missing, unexpected = _workspace_file_set(
            run_dir, state.get("expected_files") or []
        )
        if missing or unexpected:
            state["status"] = "invalid_file_set"
            state["file_set_mismatch"] = {
                "missing": missing,
                "unexpected": unexpected,
            }
            _atomic_write_json(state_path, state)
            raise WorkspacePolishError(
                "Codex changed the Markdown filename set while restoring protected "
                f"content; state retained at {state_path}"
            )
        _restore_fragments(run_dir, state.get("protected_fragments") or [])
        published_files = _publish(
            output_dir=output_dir,
            state_dir=state_dir,
            run_dir=run_dir,
            model=model,
            created_at=float(state.get("created_at", time.time())),
        )
        _atomic_write_json(
            validation_path,
            {
                "valid": True,
                "files": len(published_files),
                "input_token_coverage": {
                    "complete": coverage_report["complete"],
                    "below_threshold": coverage_report["below_threshold"],
                },
                "validated_at": time.time(),
            },
        )
        state["candidate_sha256"] = {
            name: _sha256(output_dir / "validated" / name) for name in published_files
        }
        state["status"] = "published"
        state["published_at"] = time.time()
        _atomic_write_json(state_path, state)
        logger.success(f"Direct Codex polish published {len(published_files)} files with {model}")
        return WorkspacePolishResult(
            len(published_files), len(published_files), model, state_path, validation_path
        )
    finally:
        lock.__exit__(None, None, None)


__all__ = ["WorkspacePolishError", "WorkspacePolishResult", "run_workspace_polish"]
