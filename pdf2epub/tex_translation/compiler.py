"""XeLaTeX compilation as the hard validation boundary."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileResult:
    success: bool
    returncode: int
    duration_seconds: float
    command: tuple[str, ...]
    log_path: Path
    pdf_path: Path | None

    def tail(self, lines: int = 80) -> str:
        if not self.log_path.exists():
            return ""
        return "\n".join(
            self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -lines:
            ]
        )


class TexCompiler:
    """Compile a project with latexmk and XeLaTeX."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 180,
        latexmk: str = "latexmk",
        shell_escape: bool = False,
    ):
        self.timeout_seconds = timeout_seconds
        self.latexmk = latexmk
        self.shell_escape = shell_escape

    def ensure_available(self) -> None:
        if shutil.which(self.latexmk) is None:
            raise RuntimeError(
                f"{self.latexmk!r} is required for TeX validation but was not found"
            )

    def compile(
        self,
        project_dir: Path,
        main_tex: str,
        log_path: Path,
    ) -> CompileResult:
        self.ensure_available()
        project_dir = project_dir.resolve()
        main_path = (project_dir / main_tex).resolve()
        try:
            main_path.relative_to(project_dir)
        except ValueError as exc:
            raise ValueError("main_tex must stay inside project_dir") from exc
        if not main_path.is_file():
            raise FileNotFoundError(f"Main TeX file not found: {main_path}")

        command = [
            self.latexmk,
            "-g",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-shell-escape" if self.shell_escape else "-no-shell-escape",
            main_tex,
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
            output = process.stdout
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            output = (
                stdout.decode("utf-8", errors="replace")
                if isinstance(stdout, bytes)
                else stdout
            )
            output += f"\nTIMED OUT after {self.timeout_seconds} seconds\n"
            returncode = 124

        duration = time.perf_counter() - started
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        pdf_path = main_path.with_suffix(".pdf")
        return CompileResult(
            success=returncode == 0 and pdf_path.is_file(),
            returncode=returncode,
            duration_seconds=duration,
            command=tuple(command),
            log_path=log_path,
            pdf_path=pdf_path if pdf_path.is_file() else None,
        )
