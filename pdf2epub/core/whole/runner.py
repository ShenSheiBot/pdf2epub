"""
Agent-assisted generation loop with continuation support.

The core primitive: generate → agent inspects → Decision(continue/complete).
"""

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Literal, Optional

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from .sandbox import Sandbox
from .tools import register_tools

# Strip markdown fences (```json ... ```) and BOM from LLM output.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def _strip_fences_and_bom(text: str) -> str:
    """Strip BOM, markdown code fences, and leading/trailing whitespace."""
    text = text.lstrip("\ufeff")
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    text = text.strip()
    # Handle unclosed fences
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:]).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


_AGENT_RUN_MAX_RETRIES = 3

_TRANSIENT_KEYWORDS = frozenset([
    'timeout', 'timed out', 'connection', 'disconnected', 'payload',
    'content_length', 'contentlength', 'incomplete',
    '500', '502', '503', '504', '429',
    'unavailable', 'overloaded', 'rate_limit',
])

_NON_TRANSIENT_KEYWORDS = frozenset([
    'safety', 'blocked', 'prohibited', 'harmful', 'permission',
])


def _is_transient_agent_error(exc: Exception) -> bool:
    """Check if an agent error is transient (network/server) and worth retrying."""
    err = str(exc).lower()
    if any(k in err for k in _NON_TRANSIENT_KEYWORDS):
        return False
    return any(k in err for k in _TRANSIENT_KEYWORDS)


class Decision(BaseModel):
    """Agent's decision after inspecting the work directory."""

    action: Literal["continue", "complete"]
    file_path: str


def _save_agent_trace(result, round_num: int, workspace_dir: Path) -> None:
    """Save the full agent message history as JSON for debugging."""
    try:
        trace_bytes = result.all_messages_json()
        trace_path = workspace_dir / f"agent_trace_round_{round_num:03d}.json"
        trace_path.write_bytes(trace_bytes)
        logger.debug(f"[agent-loop] Saved agent trace: {trace_path.name} ({len(trace_bytes)} bytes)")
    except Exception as e:
        logger.warning(f"[agent-loop] Failed to save agent trace: {e}")


def _extract_tool_stats(result) -> dict:
    """Extract tool call statistics from an agent run result. Returns structured dict."""
    from collections import Counter

    tool_calls = Counter()
    tool_errors = Counter()

    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if hasattr(part, "part_kind"):
                if part.part_kind == "tool-call":
                    tool_calls[part.tool_name] += 1
                elif part.part_kind == "tool-return":
                    content = str(getattr(part, "content", ""))
                    if content.startswith("ERROR:"):
                        tool_errors[part.tool_name] += 1

    usage = result.usage()
    return {
        "tool_calls": dict(tool_calls),
        "tool_errors": dict(tool_errors),
        "total_calls": sum(tool_calls.values()),
        "total_errors": sum(tool_errors.values()),
        "request_tokens": getattr(usage, "request_tokens", 0) or 0,
        "response_tokens": getattr(usage, "response_tokens", 0) or 0,
    }


def _log_tool_usage(stats: dict, round_num: int) -> None:
    """Log tool call statistics from extracted stats dict."""
    total = stats["total_calls"]
    errors = stats["total_errors"]
    breakdown = ", ".join(
        f"{name}={count}" for name, count in
        sorted(stats["tool_calls"].items(), key=lambda x: -x[1])
    )
    error_detail = ""
    if errors:
        error_detail = " | errors: " + ", ".join(
            f"{name}={count}" for name, count in stats["tool_errors"].items()
        )
    tokens_info = (
        f" | tokens: {stats['request_tokens']} in, "
        f"{stats['response_tokens']} out"
    )

    logger.info(
        f"[agent-loop] Round {round_num} tools: {total} calls ({breakdown}){error_detail}{tokens_info}"
    )


def _save_round_metrics(
    workspace_dir: Path,
    round_num: int,
    duration_sec: float,
    status: str,
    decision_action: str | None,
    decision_file: str | None,
    stats: dict | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Append structured round metrics to agent_round_metrics.jsonl."""
    entry = {
        "round": round_num,
        "ts": round(time.time(), 3),
        "duration_sec": round(duration_sec, 2),
        "status": status,
        "decision_action": decision_action,
        "decision_file": decision_file,
        "error_type": error_type,
        "error_message": error_message,
    }
    if stats:
        entry.update({
            "tool_calls": stats["tool_calls"],
            "tool_errors": stats["tool_errors"],
            "total_calls": stats["total_calls"],
            "total_errors": stats["total_errors"],
            "request_tokens": stats["request_tokens"],
            "response_tokens": stats["response_tokens"],
        })
    try:
        metrics_path = workspace_dir / "_agent_round_metrics.jsonl"
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


class AgentLoopExhausted(Exception):
    """Raised when max_continuations is exceeded without completion."""

    pass


def _validate_json_content(content: str) -> Optional[str]:
    """Default content validator: checks for valid JSON. Returns error message or None."""
    try:
        json.loads(content, strict=False)
    except json.JSONDecodeError as e:
        hint = ""
        if "```" in content:
            hint = " Remove markdown fences (```...```) and any preamble."
        return f"Cannot complete: file is not valid JSON: {e}.{hint} Fix the file and try again."
    return None


# Type alias for content validators: takes file content, returns error message or None.
ContentValidator = Callable[[str], Optional[str]]


def _create_agent(
    system_prompt: str,
    model: Model,
    sandbox: Sandbox,
    content_validator: Optional[ContentValidator] = _validate_json_content,
) -> Agent:
    """Create a pydantic-ai agent with the Decision output type and validators.

    Args:
        content_validator: Optional callable that validates completed file content.
            Takes file content string, returns error message string or None if valid.
            Default: JSON validation. Pass None to skip content validation.
    """
    agent = Agent(
        model,
        output_type=Decision,
        system_prompt=system_prompt,
        retries=3,
        output_retries=3,
    )

    @agent.output_validator
    def _validate_decision(ctx, decision: Decision) -> Decision:
        """Enforce workspace path and content validity before accepting a decision."""
        work_dir = sandbox.work_dir
        # Resolve path
        p = Path(decision.file_path)
        if not p.is_absolute():
            p = work_dir / p
        resolved = p.resolve()

        # Path must be within work_dir (both actions)
        if not sandbox.is_within_work_dir(resolved):
            raise ModelRetry(
                f"file_path must be inside the work directory. "
                f"You returned: {decision.file_path}. "
                f"Use relative paths like workspace/output.txt."
            )

        # Must exist and be a file (not a directory)
        if not resolved.exists():
            raise ModelRetry(
                f"File not found: {decision.file_path}. "
                f"Create the file in workspace/ first."
            )
        if not resolved.is_file():
            raise ModelRetry(
                f"Path is a directory, not a file: {decision.file_path}. "
                f"Specify the actual file path."
            )

        # For 'complete', must be in workspace/ and pass content validation
        if decision.action == "complete":
            if not sandbox.is_writable_path(resolved):
                raise ModelRetry(
                    f"file_path must point inside workspace/ for 'complete'. "
                    f"You returned: {decision.file_path}. "
                    f"Copy/repair into workspace/ first."
                )
            content = resolved.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                raise ModelRetry(
                    "Selected file is empty. Write content to workspace/ first."
                )
            if content_validator:
                error = content_validator(content)
                if error:
                    raise ModelRetry(error)

        return decision

    return agent


async def run_agent_loop(
    generate_fn: Callable[..., str],
    system_prompt: str,
    agent_model: Model,
    max_continuations: int = 5,
    request_limit: int = 100,
    work_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    content_validator: Optional[ContentValidator] = _validate_json_content,
    extra_originals: Optional[dict[str, str]] = None,
) -> str:
    """
    Universal agent-assisted generation loop.

    1. Call generate_fn() to get initial output
    2. Save to originals/raw_output.txt in a temporary work directory
    3. Run a pydantic-ai agent with standard tools (bash, read, edit, write, glob, grep)
    4. Agent returns Decision:
       - complete(file_path) → read file, return content
       - continue(file_path) → read file as prefix, call generate_fn(prefix=...),
         save as continuation_NNN.txt, run agent again (fresh run)
    5. If max_continuations exceeded → raise AgentLoopExhausted

    Args:
        generate_fn: Generation function. Signature: generate_fn(prefix=None) -> str.
                     Caller is responsible for constructing multi-turn messages from prefix.
        system_prompt: Agent system prompt.
        agent_model: pydantic-ai Model instance.
        max_continuations: Maximum continuation rounds before giving up.
        request_limit: Max tool calls per agent round.
        work_dir: Work directory (default: auto-create temp directory).
        artifacts_dir: If provided, copy originals/ and workspace/ here before cleanup.
        content_validator: Optional callable that validates completed file content.
            Default: JSON validation. Pass None to skip content validation.
        extra_originals: Optional dict of {filename: content} to write into originals/.
            Use this to provide reference files the agent needs (e.g., source text, mapping).

    Returns:
        Final content (from the file the agent marked as complete).

    Raises:
        AgentLoopExhausted: max_continuations exceeded without completion.
    """
    loop_start = time.perf_counter()
    own_work_dir = work_dir is None
    if own_work_dir:
        work_dir = Path(tempfile.mkdtemp(prefix="agent_work_"))

    originals_dir = work_dir / "originals"
    workspace_dir = work_dir / "workspace"

    try:
        originals_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Write pre-built utilities to workspace
        from .workspace_utils import WORKSPACE_UTILS_CODE
        (workspace_dir / "_utils.py").write_text(WORKSPACE_UTILS_CODE, encoding="utf-8")

        # Write extra reference files into originals/
        if extra_originals:
            for fname, fcontent in extra_originals.items():
                (originals_dir / fname).write_text(fcontent, encoding="utf-8")
                logger.debug(f"[agent-loop] Wrote extra original: {fname}")

        # Step 1: Generate initial output
        logger.info("[agent-loop] Calling generate_fn for initial output...")
        raw_output = generate_fn()
        if not isinstance(raw_output, str):
            raw_output = str(raw_output) if raw_output is not None else ""
        # Strip markdown fences and BOM before saving
        raw_output = _strip_fences_and_bom(raw_output)
        raw_output_path = originals_dir / "raw_output.txt"
        raw_output_path.write_text(raw_output, encoding="utf-8")
        logger.info(
            f"[agent-loop] Initial output: {len(raw_output)} chars → {raw_output_path.name}"
        )

        # Step 2-5: Agent inspection + continuation loop
        sandbox = Sandbox(work_dir)
        continuation_count = 0
        empty_continuation_streak = 0

        while True:
            # Create a fresh agent each round (with output validator)
            agent = _create_agent(system_prompt, agent_model, sandbox, content_validator)
            register_tools(agent, sandbox)

            # Build user prompt describing current state
            originals_files = sorted(originals_dir.iterdir())
            file_list = "\n".join(f"  - {f.name}" for f in originals_files)
            user_prompt = (
                f"Work directory contents:\n"
                f"originals/:\n{file_list}\n"
                f"workspace/: (your writable area)\n\n"
                f"Inspect the files and produce your decision."
            )

            round_num = continuation_count + 1
            logger.info(
                f"[agent-loop] Running agent (round {round_num}, "
                f"request_limit={request_limit})..."
            )

            round_start = time.perf_counter()
            result = None
            stats = None
            last_agent_err = None
            for attempt in range(_AGENT_RUN_MAX_RETRIES):
                try:
                    result = await agent.run(
                        user_prompt,
                        usage_limits=UsageLimits(
                            request_limit=request_limit,
                        ),
                    )
                    last_agent_err = None
                    break
                except Exception as agent_err:
                    last_agent_err = agent_err
                    if _is_transient_agent_error(agent_err) and attempt < _AGENT_RUN_MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            f"[agent-loop] Agent transient error in round {round_num} "
                            f"(attempt {attempt + 1}/{_AGENT_RUN_MAX_RETRIES}): "
                            f"{type(agent_err).__name__}: {str(agent_err)[:200]}. "
                            f"Retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
                        # Recreate agent (fresh state) for retry
                        agent = _create_agent(system_prompt, agent_model, sandbox, content_validator)
                        register_tools(agent, sandbox)
                        continue
                    else:
                        round_dur = time.perf_counter() - round_start
                        if result is not None:
                            _save_agent_trace(result, round_num, workspace_dir)
                        _save_round_metrics(
                            workspace_dir, round_num, round_dur,
                            status="error", decision_action=None, decision_file=None,
                            stats=None,
                            error_type=type(agent_err).__name__,
                            error_message=str(agent_err)[:500],
                        )
                        logger.error(f"[agent-loop] Agent crashed in round {round_num}: {agent_err}")
                        raise

            round_dur = time.perf_counter() - round_start
            decision = result.output

            # Extract stats, log, save trace and metrics
            stats = _extract_tool_stats(result)
            _log_tool_usage(stats, round_num)
            _save_agent_trace(result, round_num, workspace_dir)
            _save_round_metrics(
                workspace_dir, round_num, round_dur,
                status="ok", decision_action=decision.action,
                decision_file=decision.file_path, stats=stats,
            )

            logger.info(
                f"[agent-loop] Agent decision: action={decision.action}, "
                f"file_path={decision.file_path} ({round_dur:.1f}s)"
            )

            try:
                resolved_path = _resolve_decision_path(decision.file_path, work_dir)
            except (ValueError, FileNotFoundError) as e:
                _save_round_metrics(
                    workspace_dir, round_num, round_dur,
                    status="error", decision_action=decision.action,
                    decision_file=decision.file_path, stats=stats,
                    error_type="invalid_path", error_message=str(e)[:500],
                )
                logger.warning(f"[agent-loop] Invalid decision path: {e}")
                raise

            if decision.action == "complete":
                # Read the completed file and return
                content = resolved_path.read_text(encoding="utf-8")
                loop_dur = time.perf_counter() - loop_start
                logger.info(
                    f"[agent-loop] Complete. Final output: {len(content)} chars, "
                    f"total duration: {loop_dur:.1f}s, rounds: {round_num}"
                )
                return content

            elif decision.action == "continue":
                continuation_count += 1
                if continuation_count > max_continuations:
                    raise AgentLoopExhausted(
                        f"Agent requested {continuation_count} continuations "
                        f"(max: {max_continuations}). Giving up."
                    )

                # Read the prefix file
                prefix = resolved_path.read_text(encoding="utf-8")
                logger.info(
                    f"[agent-loop] Continuation {continuation_count}/{max_continuations}: "
                    f"prefix={len(prefix)} chars"
                )

                # Call generate_fn with prefix for continuation
                continuation_output = generate_fn(prefix=prefix)
                if not isinstance(continuation_output, str):
                    continuation_output = str(continuation_output) if continuation_output is not None else ""
                # Strip markdown fences and BOM
                continuation_output = _strip_fences_and_bom(continuation_output)

                # Always write the artifact (even if empty) for observability
                continuation_path = originals_dir / f"continuation_{continuation_count:03d}.txt"
                continuation_path.write_text(continuation_output, encoding="utf-8")

                if not continuation_output.strip():
                    empty_continuation_streak += 1
                    logger.warning(
                        f"[agent-loop] Empty continuation output "
                        f"(streak: {empty_continuation_streak}/2)"
                    )
                    if empty_continuation_streak >= 2:
                        raise AgentLoopExhausted(
                            "Continuation model returned empty output twice in a row. "
                            "Aborting to prevent infinite loop."
                        )
                    continue
                else:
                    empty_continuation_streak = 0

                logger.info(
                    f"[agent-loop] Continuation output: {len(continuation_output)} chars "
                    f"→ {continuation_path.name}"
                )
                # Loop back to run agent again on updated workspace

    finally:
        # Preserve artifacts for observability
        artifacts_saved = False
        if artifacts_dir and work_dir.exists():
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                for subdir in ("originals", "workspace"):
                    src = work_dir / subdir
                    if src.exists():
                        dst = artifacts_dir / subdir
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                artifacts_saved = True
                logger.debug(f"[agent-loop] Saved artifacts to {artifacts_dir}")
            except OSError as e:
                logger.warning(f"[agent-loop] Failed to save artifacts: {e}")

        # Clean up temp dir policy:
        # - artifacts_dir provided and saved OK → clean up
        # - artifacts_dir provided but save failed → keep for diagnosis
        # - no artifacts_dir → clean up (caller chose not to save artifacts)
        if own_work_dir and work_dir.exists():
            should_keep = artifacts_dir and not artifacts_saved
            if should_keep:
                logger.warning(
                    f"[agent-loop] Keeping work dir for diagnosis: {work_dir}"
                )
            else:
                try:
                    shutil.rmtree(work_dir)
                    logger.debug(f"[agent-loop] Cleaned up work dir: {work_dir}")
                except OSError as e:
                    logger.warning(f"[agent-loop] Failed to clean up work dir: {e}")


def _resolve_decision_path(file_path: str, work_dir: Path) -> Path:
    """Resolve a file path from a Decision, relative to work_dir.

    Validates that the path is non-empty, within work_dir, exists, and is a file.
    """
    if not file_path or not file_path.strip():
        raise ValueError("Agent returned empty file_path in Decision.")
    p = Path(file_path)
    if not p.is_absolute():
        p = work_dir / p
    # Prevent path traversal — resolved path must be inside work_dir
    resolved = p.resolve()
    work_resolved = work_dir.resolve()
    if not (resolved == work_resolved or str(resolved).startswith(str(work_resolved) + "/")):
        raise ValueError(
            f"Agent referenced path outside work directory: {file_path} "
            f"(resolved to {resolved})"
        )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Agent referenced non-existent file: {file_path} "
            f"(resolved to {resolved})"
        )
    if not resolved.is_file():
        raise ValueError(
            f"Agent referenced a directory, not a file: {file_path} "
            f"(resolved to {resolved})"
        )
    return resolved


def run_agent_loop_sync(
    generate_fn: Callable[..., str],
    system_prompt: str,
    agent_model: Model,
    max_continuations: int = 5,
    request_limit: int = 100,
    work_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    content_validator: Optional[ContentValidator] = _validate_json_content,
    extra_originals: Optional[dict[str, str]] = None,
) -> str:
    """Synchronous wrapper for run_agent_loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    kwargs = dict(
        generate_fn=generate_fn,
        system_prompt=system_prompt,
        agent_model=agent_model,
        max_continuations=max_continuations,
        request_limit=request_limit,
        work_dir=work_dir,
        artifacts_dir=artifacts_dir,
        content_validator=content_validator,
        extra_originals=extra_originals,
    )

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, run_agent_loop(**kwargs))
            return future.result()
    else:
        return asyncio.run(run_agent_loop(**kwargs))
