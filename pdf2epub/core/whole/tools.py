"""
Pydantic-AI tool definitions for the agent runner.

Six standard tools: bash, read, edit, write, glob, grep.
All file operations are restricted to the work directory.
Write operations are restricted to the workspace/ subdirectory.

Every tool call is logged to workspace/_tool_trace.jsonl for observability.
"""

import json as _json
import re
import time
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent

from .sandbox import Sandbox


def register_tools(agent: Agent, sandbox: Sandbox) -> None:
    """Register all six standard tools on a pydantic-ai agent."""

    work_dir = sandbox.work_dir
    _trace_path = sandbox.workspace_dir / "_tool_trace.jsonl"

    def _trace(tool_name: str, args: dict, result: str, elapsed: float, extra: dict | None = None) -> None:
        """Append a tool call record to the trace file. No truncation."""
        entry = {
            "ts": round(time.time(), 3),
            "tool": tool_name,
            "args": {k: str(v) for k, v in args.items()},
            "result": result,
            "is_error": result.startswith("ERROR:"),
            "elapsed_ms": round(elapsed * 1000),
        }
        if extra:
            entry.update(extra)
        try:
            with open(_trace_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Don't let tracing failures break the agent

    def _resolve_path(path_str: str) -> Path:
        """Resolve a path relative to work_dir."""
        p = Path(path_str)
        if not p.is_absolute():
            p = work_dir / p
        return p

    def _check_readable(path: Path) -> Optional[str]:
        """Return error string if path is not readable, None if OK."""
        if not sandbox.is_within_work_dir(path):
            return f"ERROR: Read not permitted: path must be inside the work directory ({work_dir})"
        if not path.exists():
            return f"ERROR: File not found: {path}"
        return None

    def _check_writable(path: Path) -> Optional[str]:
        """Return error string if path is not writable, None if OK."""
        if not sandbox.is_writable_path(path):
            return f"ERROR: Write not permitted: path must be inside workspace/ ({sandbox.workspace_dir})"
        return None

    @agent.tool_plain
    def bash(command: str) -> str:
        """Execute a bash command in the sandbox work directory.

        The work directory contains:
        - originals/ — read-only raw LLM outputs
        - workspace/ — your writable work area

        Use this for running python3, cat, grep, wc, and other shell commands.
        """
        t0 = time.time()
        result = sandbox.execute(command)
        # Infer exit code from result text
        import re as _re
        exit_match = _re.search(r'\[exit code: (\d+)\]', result)
        exit_code = int(exit_match.group(1)) if exit_match else (1 if result.startswith("ERROR:") else 0)
        _trace("bash", {"command": command}, result, time.time() - t0, extra={
            "exit_code": exit_code,
        })
        return result

    @agent.tool_plain
    def read(path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read lines from a file with optional offset and limit.

        Args:
            path: File path (relative to work directory or absolute).
            offset: Starting line number (0-indexed).
            limit: Maximum number of lines to return (default 2000).
        """
        t0 = time.time()
        resolved = _resolve_path(path)
        err = _check_readable(resolved)
        if err:
            _trace("read", {"path": path, "offset": offset, "limit": limit}, err, time.time() - t0)
            return err
        if resolved.is_dir():
            result = f"ERROR: Cannot read directory: {resolved}. Use glob or bash ls instead."
            _trace("read", {"path": path, "offset": offset, "limit": limit}, result, time.time() - t0)
            return result
        # Clamp to non-negative values
        offset = max(0, offset)
        limit = max(1, limit)
        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            selected = lines[offset : offset + limit]
            if not selected:
                if offset >= len(lines):
                    result = f"Offset {offset} is beyond end of file ({len(lines)} lines)."
                    _trace("read", {"path": path, "offset": offset, "limit": limit}, result, time.time() - t0)
                    return result
                result = "(empty file)"
                _trace("read", {"path": path, "offset": offset, "limit": limit}, result, time.time() - t0)
                return result
            numbered = []
            for i, line in enumerate(selected, start=offset + 1):
                numbered.append(f"{i:6d}\t{line.rstrip()}")
            result = "\n".join(numbered)
            if offset + limit < len(lines):
                result += f"\n\n... ({len(lines) - offset - limit} more lines)"
            _trace("read", {"path": path, "offset": offset, "limit": limit}, result, time.time() - t0)
            return result
        except Exception as e:
            result = f"ERROR: reading file: {e}"
            _trace("read", {"path": path, "offset": offset, "limit": limit}, result, time.time() - t0)
            return result

    @agent.tool_plain
    def edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace text in a file. Only works in workspace/.

        Args:
            path: File path (must be inside workspace/).
            old_string: The exact text to find and replace.
            new_string: The replacement text.
            replace_all: If True, replace all occurrences. If False, replace only the first.
        """
        t0 = time.time()
        resolved = _resolve_path(path)
        err = _check_writable(resolved)
        if err:
            _trace("edit", {"path": path}, err, time.time() - t0)
            return err
        err = _check_readable(resolved)
        if err:
            _trace("edit", {"path": path}, err, time.time() - t0)
            return err
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            if old_string not in content:
                result = f"ERROR: old_string not found in {path}. No changes made."
                _trace("edit", {"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all}, result, time.time() - t0)
                return result
            if replace_all:
                new_content = content.replace(old_string, new_string)
                count = content.count(old_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
                count = 1
            resolved.write_text(new_content, encoding="utf-8")
            result = f"Replaced {count} occurrence(s) in {path}."
            _trace("edit", {"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all}, result, time.time() - t0)
            return result
        except Exception as e:
            result = f"ERROR: editing file: {e}"
            _trace("edit", {"path": path}, result, time.time() - t0)
            return result

    @agent.tool_plain
    def write(path: str, content: str) -> str:
        """Write content to a file, creating it if needed. Only works in workspace/.

        Args:
            path: File path (must be inside workspace/).
            content: The full content to write to the file.
        """
        t0 = time.time()
        resolved = _resolve_path(path)
        err = _check_writable(resolved)
        if err:
            _trace("write", {"path": path, "content_len": len(content)}, err, time.time() - t0)
            return err
        try:
            import hashlib as _hl
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            result = f"Wrote {len(content)} characters to {path}."
            _trace("write", {"path": path, "content_len": len(content)}, result, time.time() - t0, extra={
                "content_sha256": _hl.sha256(content.encode()).hexdigest(),
                "content_lines": content.count("\n") + 1 if content else 0,
            })
            return result
        except Exception as e:
            result = f"ERROR: writing file: {e}"
            _trace("write", {"path": path, "content_len": len(content)}, result, time.time() - t0)
            return result

    @agent.tool_plain
    def glob(pattern: str) -> str:
        """Find files matching a glob pattern within the work directory.

        Args:
            pattern: Glob pattern (e.g., "originals/*.txt", "workspace/**/*.json").
        """
        t0 = time.time()
        try:
            matches = sorted(work_dir.glob(pattern))
            # Filter to only files within work_dir
            matches = [m for m in matches if sandbox.is_within_work_dir(m)]
            if not matches:
                result = f"No files matching '{pattern}'."
                _trace("glob", {"pattern": pattern}, result, time.time() - t0)
                return result
            # Show paths relative to work_dir
            lines = [str(m.relative_to(work_dir)) for m in matches]
            result = "\n".join(lines)
            _trace("glob", {"pattern": pattern}, result, time.time() - t0)
            return result
        except Exception as e:
            result = f"ERROR: glob: {e}"
            _trace("glob", {"pattern": pattern}, result, time.time() - t0)
            return result

    @agent.tool_plain
    def grep(pattern: str, path: str = ".") -> str:
        """Search file contents for a regex pattern within the work directory.

        Args:
            pattern: Regular expression pattern to search for.
            path: File or directory to search in (relative to work directory, default ".").
        """
        t0 = time.time()
        resolved = _resolve_path(path)
        err = _check_readable(resolved)
        if err:
            _trace("grep", {"pattern": pattern, "path": path}, err, time.time() - t0)
            return err
        try:
            results = []
            if resolved.is_file():
                files = [resolved]
            else:
                files = sorted(resolved.rglob("*"))
                files = [f for f in files if f.is_file() and sandbox.is_within_work_dir(f)]

            regex = re.compile(pattern)
            for f in files:
                try:
                    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            rel = f.relative_to(work_dir)
                            results.append(f"{rel}:{i}: {line}")
                except (OSError, UnicodeDecodeError):
                    continue

            if not results:
                result = f"No matches for pattern '{pattern}' in {path}."
                _trace("grep", {"pattern": pattern, "path": path}, result, time.time() - t0)
                return result

            output = "\n".join(results)
            if len(output) > 32 * 1024:
                output = output[: 32 * 1024] + "\n\n[output truncated — narrow your search]"
            _trace("grep", {"pattern": pattern, "path": path}, output, time.time() - t0)
            return output
        except re.error as e:
            result = f"ERROR: invalid regex pattern: {e}"
            _trace("grep", {"pattern": pattern, "path": path}, result, time.time() - t0)
            return result
        except Exception as e:
            result = f"ERROR: grep: {e}"
            _trace("grep", {"pattern": pattern, "path": path}, result, time.time() - t0)
            return result
