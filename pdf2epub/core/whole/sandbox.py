"""
Sandbox for agent bash command execution.

Uses Anthropic's sandbox-runtime (srt) for OS-level isolation via
sandbox-exec (macOS) or bubblewrap (Linux). Falls back to subprocess
with cwd isolation if srt is not available.
"""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

BASH_TIMEOUT_SECONDS = 30
STDOUT_MAX_BYTES = 32 * 1024  # 32KB

_srt_path = shutil.which("srt")


class Sandbox:
    """Execute bash commands within an isolated work directory."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir.resolve()
        self.workspace_dir = self.work_dir / "workspace"
        self._srt_settings_path = self.work_dir / ".srt-settings.json"

        # Write srt settings: allow write only within work_dir, no network
        if _srt_path:
            self._srt_settings_path.write_text(json.dumps({
                "filesystem": {
                    "denyRead": [],
                    "allowWrite": [str(self.work_dir)],
                    "denyWrite": [],
                },
                "network": {
                    "allowedDomains": [],
                    "deniedDomains": [],
                },
            }))

    def execute(self, command: str) -> str:
        """
        Execute a bash command in the sandbox.

        If srt is available, uses OS-level sandboxing (sandbox-exec on macOS,
        bubblewrap on Linux). Otherwise falls back to subprocess with cwd isolation.

        Args:
            command: Shell command string to execute.

        Returns:
            Combined stdout/stderr output, truncated at 32KB if needed.
        """
        if _srt_path:
            return self._execute_srt(command)
        else:
            return self._execute_fallback(command)

    def _execute_srt(self, command: str) -> str:
        """Execute via srt sandbox."""
        try:
            read_limit = STDOUT_MAX_BYTES + 4096
            env = {**os.environ, "PYTHONPATH": str(self.work_dir)}
            proc = subprocess.Popen(
                [_srt_path, "-s", str(self._srt_settings_path), "-c", command],
                cwd=str(self.work_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
            return self._collect_output(proc, read_limit)
        except Exception as e:
            return f"ERROR: Command failed: {type(e).__name__}: {e}"

    def _execute_fallback(self, command: str) -> str:
        """Fallback: subprocess with cwd isolation (no OS sandbox)."""
        try:
            read_limit = STDOUT_MAX_BYTES + 4096
            env = {**os.environ, "PYTHONPATH": str(self.work_dir)}
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=str(self.work_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
            return self._collect_output(proc, read_limit)
        except Exception as e:
            return f"ERROR: Command failed: {type(e).__name__}: {e}"

    def _collect_output(self, proc, read_limit: int) -> str:
        """Read stdout/stderr from process with timeout."""
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

        output = stdout_bytes[:STDOUT_MAX_BYTES].decode("utf-8", errors="replace")
        stderr_str = stderr_bytes[:STDOUT_MAX_BYTES].decode("utf-8", errors="replace")

        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
            if stderr_str:
                output += f"\n[stderr]\n{stderr_str}"

        if len(stdout_bytes) > STDOUT_MAX_BYTES:
            output += "\n\n[output truncated at 32KB — use read tool to see full content]"

        return output

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
