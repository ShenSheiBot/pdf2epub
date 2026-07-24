"""Transactional whole-mode translation of complete TeX projects."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from pdf2epub.utils.llm_client import LLMClient

from .arxiv import ArxivSourceResolver
from .cache import TranslationCache
from .compiler import TexCompiler
from .document import (
    TexProjectDocument,
    TranslationUnit,
    discover_main_tex,
    scan_project,
)
from .prompts import TRANSLATION_PROMPT_VERSION, build_translation_messages
from .repair import TexRepairAgent, mark_translation
from .state import TranslationState, atomic_write_sources


@dataclass(frozen=True)
class TexTranslationOptions:
    """Behavioral settings that do not depend on a particular source tree."""

    provider: str = "gemini"
    model: str = "gemini-3.1-pro-preview"
    source_language: str = "English"
    target_language: str = "Simplified Chinese"
    unit_chars: int = 12_000
    max_retries: int = 2
    use_local_cache: bool = True
    repair_enabled: bool = True
    repair_provider: str = "codex"
    repair_model: str = "gpt-5.6-luna"
    retry_fallbacks: bool = False


@dataclass(frozen=True)
class TexTranslationResult:
    run_dir: Path
    source_dir: Path
    project_dir: Path
    main_tex: str
    pdf_path: Path
    summary: dict[str, int]
    warnings: tuple[str, ...]


class TexTranslationPipeline:
    """Translate one unit at a time and commit only compile-safe replacements."""

    def __init__(
        self,
        *,
        config: dict,
        options: TexTranslationOptions | None = None,
        llm_client: LLMClient | None = None,
        compiler: TexCompiler | None = None,
        source_resolver: ArxivSourceResolver | None = None,
        repair_agent: TexRepairAgent | None = None,
    ):
        self.config = config
        self.options = options or TexTranslationOptions()
        self.llm_client = llm_client or LLMClient(config)
        self.compiler = compiler or TexCompiler()
        self.source_resolver = source_resolver or ArxivSourceResolver()
        self._repair_agent = repair_agent

    def run(
        self,
        source: str | Path,
        *,
        run_dir: Path,
        main_tex: str | None = None,
        limit: int | None = None,
    ) -> TexTranslationResult:
        run_dir = run_dir.resolve()
        source_dir = run_dir / "source"
        project_dir = run_dir / "project"
        control_dir = run_dir / ".pdf2epub"
        logs_dir = control_dir / "logs"
        resolved = self.source_resolver.materialize(source, source_dir)

        selected_main = discover_main_tex(
            source_dir,
            main_tex or resolved.suggested_main_tex,
        )
        document = scan_project(
            source_dir,
            selected_main,
            unit_chars=self.options.unit_chars,
            target_language=self.options.target_language,
        )
        for warning in document.warnings:
            logger.warning(f"[tex-translate] {warning}")

        state = TranslationState(control_dir)
        state.initialize(
            source_id=resolved.source_id,
            source_fingerprint=document.source_fingerprint,
            layout_fingerprint=document.layout_fingerprint,
            main_tex=document.main_tex,
            units=[unit.manifest_entry() for unit in document.units],
            translation_spec={
                "source_language": self.options.source_language,
                "target_language": self.options.target_language,
                "prompt_version": TRANSLATION_PROMPT_VERSION,
            },
        )
        cache = TranslationCache(control_dir / "cache")

        self._prepare_project(source_dir, project_dir)
        translations = state.completed_translations()
        atomic_write_sources(project_dir, document.render(translations))

        preflight = self.compiler.compile(
            project_dir,
            document.main_tex,
            logs_dir / "preflight.log",
        )
        if not preflight.success:
            raise RuntimeError(
                "The source project does not compile with the injected CJK/XeLaTeX "
                "setup, before any new translation was attempted.\n\n"
                f"{preflight.tail()}"
            )
        logger.info(
            f"[tex-translate] Preflight compile passed in "
            f"{preflight.duration_seconds:.2f}s"
        )

        units_by_id = {unit.id: unit for unit in document.units}
        pending = state.pending_ids(retry_fallbacks=self.options.retry_fallbacks)
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            pending = pending[:limit]

        for index, unit_id in enumerate(pending, 1):
            unit = units_by_id[unit_id]
            logger.info(
                f"[tex-translate] Unit {index}/{len(pending)} "
                f"{unit.id} ({unit.relative_path}, {len(unit.source_text)} chars)"
            )
            self._process_unit(
                document=document,
                unit=unit,
                project_dir=project_dir,
                control_dir=control_dir,
                logs_dir=logs_dir,
                cache=cache,
                state=state,
                translations=translations,
            )

        atomic_write_sources(project_dir, document.render(translations))
        final = self.compiler.compile(
            project_dir,
            document.main_tex,
            logs_dir / "final.log",
        )
        if not final.success or final.pdf_path is None:
            raise RuntimeError(
                f"Final translated project failed to compile.\n\n{final.tail()}"
            )

        summary = state.summary()
        logger.success(
            "[tex-translate] Complete: "
            f"{summary.get('translated', 0)} translated, "
            f"{summary.get('repaired', 0)} repaired, "
            f"{summary.get('fallback_original', 0)} kept in source language, "
            f"{summary.get('pending', 0)} pending"
        )
        return TexTranslationResult(
            run_dir=run_dir,
            source_dir=source_dir,
            project_dir=project_dir,
            main_tex=document.main_tex,
            pdf_path=final.pdf_path,
            summary=summary,
            warnings=document.warnings,
        )

    def _process_unit(
        self,
        *,
        document: TexProjectDocument,
        unit: TranslationUnit,
        project_dir: Path,
        control_dir: Path,
        logs_dir: Path,
        cache: TranslationCache,
        state: TranslationState,
        translations: dict[str, str],
    ) -> None:
        messages = build_translation_messages(
            unit.source_text,
            source_language=self.options.source_language,
            target_language=self.options.target_language,
        )
        cache_key = cache.key(
            provider=self.options.provider,
            model=self.options.model,
            messages=messages,
            prompt_version=TRANSLATION_PROMPT_VERSION,
        )
        previous_status = state.data["units"][unit.id].get("status")
        bypass_failed_cache = (
            self.options.retry_fallbacks and previous_status == "fallback_original"
        )
        cached = (
            cache.get(cache_key)
            if self.options.use_local_cache and not bypass_failed_cache
            else None
        )
        provider_usage: dict = {}
        if cached is not None:
            raw_translation = cached.content
            provider_usage = cached.metadata.get("provider_usage") or {}
            local_cache_hit = True
            logger.info(f"[tex-translate] Local cache hit for {unit.id}")
        else:
            raw_translation = self.llm_client.generate(
                prompt=messages,
                model_configs=[
                    {
                        "provider": self.options.provider,
                        "model": self.options.model,
                        "max_retries": self.options.max_retries,
                    }
                ],
                operation_name=f"Translate TeX {unit.id}",
                enable_cache=True,
            )
            provider_usage = self._last_provider_usage()
            local_cache_hit = False
            if provider_usage:
                logger.info(
                    f"[tex-translate] Provider cache audit for {unit.id}: "
                    f"input={provider_usage.get('input_tokens', 0)}, "
                    f"cache_read={provider_usage.get('cache_read_tokens', 0)} tokens"
                )
            if self.options.use_local_cache:
                cache.put(
                    cache_key,
                    raw_translation,
                    {
                        "provider": self.options.provider,
                        "model": self.options.model,
                        "provider_usage": provider_usage,
                    },
                )

        translation = _preserve_boundary_whitespace(
            unit.source_text,
            _strip_response_wrappers(raw_translation),
        )
        if not translation:
            self._fallback_unit(
                unit=unit,
                document=document,
                project_dir=project_dir,
                state=state,
                translations=translations,
                logs_dir=logs_dir,
                record={
                    "reason": "empty_translation",
                    "provider": self.options.provider,
                    "model": self.options.model,
                    "cache_key": cache_key,
                    "local_cache_hit": local_cache_hit,
                    "provider_usage": provider_usage,
                },
            )
            return

        marker_begin, marker_end, marked = mark_translation(unit.id, translation)
        candidate_translations = {**translations, unit.id: marked}
        atomic_write_sources(project_dir, document.render(candidate_translations))
        candidate_compile = self.compiler.compile(
            project_dir,
            document.main_tex,
            logs_dir / f"{unit.id}-candidate.log",
        )

        final_translation = translation
        status = "translated"
        repair_error: str | None = None
        final_compile = candidate_compile
        if not candidate_compile.success and self.options.repair_enabled:
            logger.warning(
                f"[tex-translate] {unit.id} failed compilation; invoking repair agent"
            )
            try:
                final_translation = self._get_repair_agent(control_dir).repair(
                    unit=unit,
                    candidate_project=project_dir,
                    main_tex=document.main_tex,
                    raw_translation=raw_translation,
                    marker_begin=marker_begin,
                    marker_end=marker_end,
                    compile_log=candidate_compile.tail(),
                )
                final_translation = _preserve_boundary_whitespace(
                    unit.source_text,
                    final_translation,
                )
                repaired_translations = {
                    **translations,
                    unit.id: final_translation,
                }
                atomic_write_sources(
                    project_dir,
                    document.render(repaired_translations),
                )
                final_compile = self.compiler.compile(
                    project_dir,
                    document.main_tex,
                    logs_dir / f"{unit.id}-repaired.log",
                )
                if final_compile.success:
                    status = "repaired"
                else:
                    repair_error = final_compile.tail()
            # Any repair-provider/tool failure is contained by the source rollback.
            except Exception as exc:  # noqa: BLE001
                repair_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    f"[tex-translate] Repair failed for {unit.id}: {repair_error}"
                )

        if not final_compile.success:
            self._fallback_unit(
                unit=unit,
                document=document,
                project_dir=project_dir,
                state=state,
                translations=translations,
                logs_dir=logs_dir,
                record={
                    "reason": "compile_failed",
                    "provider": self.options.provider,
                    "model": self.options.model,
                    "cache_key": cache_key,
                    "local_cache_hit": local_cache_hit,
                    "provider_usage": provider_usage,
                    "candidate_returncode": candidate_compile.returncode,
                    "repair_error": repair_error,
                },
            )
            return

        translations[unit.id] = final_translation
        state.commit_translation(
            unit.id,
            final_translation,
            {
                "status": status,
                "provider": self.options.provider,
                "model": self.options.model,
                "cache_key": cache_key,
                "local_cache_hit": local_cache_hit,
                "provider_usage": provider_usage,
                "candidate_returncode": candidate_compile.returncode,
                "final_returncode": final_compile.returncode,
                "translation_sha256": hashlib.sha256(
                    final_translation.encode("utf-8")
                ).hexdigest(),
            },
        )
        atomic_write_sources(project_dir, document.render(translations))

    def _fallback_unit(
        self,
        *,
        unit: TranslationUnit,
        document: TexProjectDocument,
        project_dir: Path,
        state: TranslationState,
        translations: dict[str, str],
        logs_dir: Path,
        record: dict,
    ) -> None:
        translations.pop(unit.id, None)
        atomic_write_sources(project_dir, document.render(translations))
        restored = self.compiler.compile(
            project_dir,
            document.main_tex,
            logs_dir / f"{unit.id}-fallback.log",
        )
        if not restored.success:
            raise RuntimeError(
                f"Restoring the original source for {unit.id} did not recover "
                f"a compilable project.\n\n{restored.tail()}"
            )
        state.commit_fallback(unit.id, record)
        logger.warning(
            f"[tex-translate] Kept original source for {unit.id}; "
            "the project remains compile-safe"
        )

    def _get_repair_agent(self, control_dir: Path) -> TexRepairAgent:
        if self._repair_agent is None:
            self._repair_agent = TexRepairAgent(
                config=self.config,
                compiler=self.compiler,
                control_dir=control_dir,
                provider_name=self.options.repair_provider,
                model_name=self.options.repair_model,
            )
        return self._repair_agent

    def _last_provider_usage(self) -> dict:
        getter = getattr(self.llm_client, "get_last_usage", None)
        if not getter:
            return {}
        usage = getter(self.options.provider)
        return usage if isinstance(usage, dict) else {}

    @staticmethod
    def _prepare_project(source_dir: Path, project_dir: Path) -> None:
        project_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, project_dir, dirs_exist_ok=True)


def _strip_response_wrappers(text: str) -> str:
    text = str(text or "").lstrip("\ufeff").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {
            "```",
            "```tex",
            "```latex",
        }:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _preserve_boundary_whitespace(source: str, translation: str) -> str:
    """Keep source span separators while replacing only its semantic content."""
    if not translation:
        return ""
    leading_length = len(source) - len(source.lstrip())
    trailing_length = len(source) - len(source.rstrip())
    leading = source[:leading_length]
    trailing = source[len(source) - trailing_length :] if trailing_length else ""
    return leading + translation.strip() + trailing
