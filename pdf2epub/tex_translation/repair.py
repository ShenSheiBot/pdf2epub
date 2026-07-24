"""Whole-mode repair adapter for compile-failing translated TeX units."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from pdf2epub.core.whole.model_factory import create_configured_model
from pdf2epub.core.whole.runner import run_agent_loop_sync

from .compiler import TexCompiler
from .document import TranslationUnit
from .prompts import REPAIR_SYSTEM


class TexRepairAgent:
    """Invoke the existing whole-mode tool agent only after compilation fails."""

    def __init__(
        self,
        *,
        config: dict,
        compiler: TexCompiler,
        control_dir: Path,
        provider_name: str,
        model_name: str,
        request_limit: int = 60,
    ):
        self.config = config
        self.compiler = compiler
        self.control_dir = control_dir
        self.provider_name = provider_name
        self.model_name = model_name
        self.request_limit = request_limit
        self._model = None

    def repair(
        self,
        *,
        unit: TranslationUnit,
        candidate_project: Path,
        main_tex: str,
        raw_translation: str,
        marker_begin: str,
        marker_end: str,
        compile_log: str,
    ) -> str:
        work_dir = self.control_dir / "repair_work" / unit.id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        relative_path = unit.relative_path

        candidate_file = candidate_project / relative_path
        candidate_text = candidate_file.read_text(encoding="utf-8")
        expected_prefix, _, expected_suffix = _split_marked(
            candidate_text, marker_begin, marker_end
        )

        def generate_fn(prefix=None):
            if prefix is not None:
                raise RuntimeError(
                    "TeX compile repair does not request translation continuation"
                )
            return raw_translation

        def prefill(_originals_dir: Path, workspace_dir: Path) -> list:
            project = workspace_dir / "project"
            if project.exists():
                shutil.rmtree(project)
            shutil.copytree(candidate_project, project)
            return []

        def validate_repaired_file(selected_content: str) -> str | None:
            try:
                prefix, _, suffix = _split_marked(
                    selected_content, marker_begin, marker_end
                )
            except ValueError as exc:
                return str(exc)
            if prefix != expected_prefix or suffix != expected_suffix:
                return (
                    "Changes outside the current PDF2EPUB unit are not allowed. "
                    "Restore all text before the BEGIN marker and after the END marker."
                )
            repair_project = work_dir / "workspace" / "project"
            target = repair_project / relative_path
            target.write_text(selected_content, encoding="utf-8")
            result = self.compiler.compile(
                repair_project,
                main_tex,
                work_dir / "workspace" / "validator_compile.log",
            )
            if result.success:
                return None
            return (
                "The complete project still fails XeLaTeX. Inspect "
                "workspace/validator_compile.log and the TeX log, then repair only "
                f"the marked unit.\n\n{result.tail()}"
            )

        compile_command = (
            "latexmk -g -xelatex -interaction=nonstopmode -halt-on-error "
            f"-no-shell-escape {main_tex}"
        )
        instructions = f"""\
The candidate project is in `workspace/project`.
The only file you may edit is `workspace/project/{relative_path}` and only the
text between these exact comment markers:

`{marker_begin}`
`{marker_end}`

The original fragment is `originals/source_fragment.tex`; the translation
model's response is `originals/raw_output.txt`; the first failed compile tail is
`originals/failed_compile.log`.

Compile from the project root with:

`cd workspace/project && {compile_command}`

When compilation succeeds, return `complete` with
`file_path="workspace/project/{relative_path}"`. Do not return `continue`.
"""
        final_file = run_agent_loop_sync(
            generate_fn=generate_fn,
            system_prompt=REPAIR_SYSTEM,
            agent_model=self._get_model(),
            max_continuations=0,
            request_limit=self.request_limit,
            work_dir=work_dir,
            artifacts_dir=self.control_dir / "repair_artifacts" / unit.id,
            content_validator=validate_repaired_file,
            extra_originals={
                "source_fragment.tex": unit.source_text,
                "failed_compile.log": compile_log,
            },
            user_instructions=instructions,
            prefill_fn=prefill,
        )
        _, repaired, _ = _split_marked(final_file, marker_begin, marker_end)
        return repaired

    def _get_model(self):
        if self._model is not None:
            return self._model
        providers = self.config.get("credentials", {}).get("providers", {})
        provider = providers.get(self.provider_name)
        if not provider:
            raise ValueError(
                f"Repair provider {self.provider_name!r} is not configured"
            )
        self._model = create_configured_model(
            self.model_name,
            provider_name=self.provider_name,
            provider_config=provider,
        )
        return self._model


def unit_markers(unit_id: str) -> tuple[str, str]:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", unit_id)
    return (
        f"% PDF2EPUB-{safe}-BEGIN",
        f"% PDF2EPUB-{safe}-END",
    )


def mark_translation(unit_id: str, translation: str) -> tuple[str, str, str]:
    begin, end = unit_markers(unit_id)
    marked = f"{begin}\n{translation.rstrip()}\n{end}\n"
    return begin, end, marked


def _split_marked(text: str, begin: str, end: str) -> tuple[str, str, str]:
    begin_token = begin + "\n"
    end_token = "\n" + end
    if text.count(begin_token) != 1 or text.count(end_token) != 1:
        raise ValueError("The current PDF2EPUB unit markers are missing or duplicated")
    prefix, remainder = text.split(begin_token, 1)
    body, suffix = remainder.split(end_token, 1)
    suffix = suffix.removeprefix("\n")
    return prefix, body, suffix
