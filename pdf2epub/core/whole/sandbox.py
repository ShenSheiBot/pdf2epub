"""
Sandbox for agent bash command execution.

Phase 1: subprocess with cwd isolation, timeout, and output truncation.
Phase 2: optional srt CLI wrapping for OS-level sandboxing.
"""

import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

BASH_TIMEOUT_SECONDS = 30
STDOUT_MAX_BYTES = 32 * 1024  # 32KB

# Reject commands containing absolute paths, path traversal, or home expansion.
# These patterns catch paths at token boundaries (after whitespace, redirects, or
# at command start) while avoiding false positives on relative paths and sed patterns.
_ABS_PATH_RE = re.compile(r'(?:^|[\s;|&()<>])(?<!<)/[a-zA-Z0-9_.]')
_TRAVERSAL_RE = re.compile(r'(?:^|[\s/])\.\.(?:[\s/]|$)')
_HOME_RE = re.compile(r'(?:^|[\s;|&()<>])~/')


class Sandbox:
    """Execute bash commands within an isolated work directory."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.workspace_dir = work_dir / "workspace"

    @staticmethod
    def _check_command(command: str) -> Optional[str]:
        """Reject commands with absolute paths, traversal, or home expansion."""
        if _ABS_PATH_RE.search(command):
            return (
                "ERROR: Absolute paths are not allowed. "
                "Use relative paths (e.g., workspace/file.txt, originals/raw_output.txt)."
            )
        if _TRAVERSAL_RE.search(command):
            return (
                "ERROR: Path traversal (..) is not allowed. "
                "Stay within the work directory."
            )
        if _HOME_RE.search(command):
            return (
                "ERROR: Home directory expansion (~/) is not allowed. "
                "Use relative paths."
            )
        return None

    def execute(self, command: str) -> str:
        """
        Execute a bash command in the sandbox.

        Uses incremental reads to avoid OOM from commands producing huge output.
        Rejects commands containing absolute paths, path traversal (..), or ~/
        to prevent filesystem escape.

        Args:
            command: Shell command string to execute.

        Returns:
            Combined stdout/stderr output, truncated at 32KB if needed.
        """
        err = self._check_command(command)
        if err:
            return err

        try:
            read_limit = STDOUT_MAX_BYTES + 4096
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=str(self.work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )

            # Read stdout and stderr concurrently to avoid deadlock.
            # Each thread reads up to read_limit bytes.
            stdout_bytes = b""
            stderr_bytes = b""

            def _read_stdout():
                nonlocal stdout_bytes
                stdout_bytes = proc.stdout.read(read_limit)

            def _read_stderr():
                nonlocal stderr_bytes
                stderr_bytes = proc.stderr.read(read_limit)

            t_out = threading.Thread(target=_read_stdout, daemon=True)
            t_err = threading.Thread(target=_read_stderr, daemon=True)
            t_out.start()
            t_err.start()
            t_out.join(timeout=BASH_TIMEOUT_SECONDS)
            t_err.join(timeout=max(1, BASH_TIMEOUT_SECONDS - 1))

            try:
                proc.wait(timeout=max(1, BASH_TIMEOUT_SECONDS))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return (
                    f"ERROR: Command timed out after {BASH_TIMEOUT_SECONDS}s. "
                    f"Use a simpler command or read the file directly."
                )

            output = stdout_bytes[:STDOUT_MAX_BYTES].decode(
                "utf-8", errors="replace"
            )
            stderr_str = stderr_bytes[:STDOUT_MAX_BYTES].decode(
                "utf-8", errors="replace"
            )

            if proc.returncode != 0:
                output += f"\n[exit code: {proc.returncode}]"
                if stderr_str:
                    output += f"\n[stderr]\n{stderr_str}"

            if len(stdout_bytes) > STDOUT_MAX_BYTES:
                output += "\n\n[output truncated at 32KB — use read tool to see full content]"

            return output

        except Exception as e:
            return f"ERROR: Command failed: {type(e).__name__}: {e}"

    def is_writable_path(self, path: Path) -> bool:
        """Check if a path is within the writable workspace directory."""
        try:
            resolved = path.resolve()
            workspace_resolved = self.workspace_dir.resolve()
            return resolved == workspace_resolved or str(resolved).startswith(
                str(workspace_resolved) + "/"
            )
        except (OSError, ValueError):
            return False

    def is_within_work_dir(self, path: Path) -> bool:
        """Check if a path is within the work directory (readable area)."""
        try:
            resolved = path.resolve()
            work_resolved = self.work_dir.resolve()
            return resolved == work_resolved or str(resolved).startswith(
                str(work_resolved) + "/"
            )
        except (OSError, ValueError):
            return False
