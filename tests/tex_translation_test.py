import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdf2epub.core.whole.model_factory import _load_codex_openai_provider
from pdf2epub.tex_translation.arxiv import (
    ArxivSourceResolver,
    normalize_arxiv_id,
)
from pdf2epub.tex_translation.cache import TranslationCache
from pdf2epub.tex_translation.compiler import CompileResult
from pdf2epub.tex_translation.document import (
    discover_main_tex,
    inject_cjk_support,
    scan_project,
)
from pdf2epub.tex_translation.pipeline import (
    TexTranslationOptions,
    TexTranslationPipeline,
)
from pdf2epub.tex_translation.prompts import (
    TRANSLATION_PROMPT_VERSION,
    build_translation_messages,
)
from pdf2epub.tex_translation.state import TranslationState
from pdf2epub.utils.network_utils import (
    AnthropicClient,
    StreamingHallucinationError,
    _detect_streaming_hallucination,
    is_transient_anthropic_error,
    is_transient_gemini_error,
)


class FakeLLMClient:
    def __init__(self, transform=None):
        self.transform = transform or (lambda text: text.replace("English", "中文"))
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        source = prompt[0]["content"].split("SOURCE FRAGMENT:\n\n", 1)[1]
        return self.transform(source)

    def get_last_usage(self, _provider):
        return {
            "input_tokens": 120,
            "output_tokens": 40,
            "cache_read_tokens": 100,
        }


class FakeCompiler:
    def __init__(self):
        self.calls = []

    def compile(self, project_dir: Path, main_tex: str, log_path: Path):
        self.calls.append((project_dir, main_tex, log_path))
        all_tex = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(project_dir.rglob("*.tex"))
        )
        success = "BROKEN_TRANSLATION" not in all_tex
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok" if success else "synthetic compile failure")
        pdf_path = project_dir / Path(main_tex).with_suffix(".pdf")
        if success:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-fake")
        else:
            pdf_path.unlink(missing_ok=True)
        return CompileResult(
            success=success,
            returncode=0 if success else 1,
            duration_seconds=0.01,
            command=("fake-latexmk",),
            log_path=log_path,
            pdf_path=pdf_path if success else None,
        )


class FailIfCalledRepairAgent:
    def repair(self, **_kwargs):
        raise AssertionError("repair must not run for a compile-clean candidate")


def _write_project(root: Path) -> None:
    root.mkdir()
    (root / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{English introduction}\n"
        "English text in the main file.\n\n"
        "\\input{sections/body}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (root / "sections").mkdir()
    (root / "sections" / "body.tex").write_text(
        "\\subsection{English details}\n"
        "More English prose with $x+y=z$ and \\cite{example}.\n",
        encoding="utf-8",
    )


def test_arxiv_identifier_normalization():
    assert normalize_arxiv_id("arXiv:2503.01800v1") == "2503.01800v1"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2503.01800.pdf") == "2503.01800"
    assert normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"


def test_gemini_cancelled_request_is_retryable():
    assert is_transient_gemini_error(
        RuntimeError(
            "499 CANCELLED. The operation was cancelled."
        )
    )


def test_streaming_guard_ignores_whitespace_but_retries_real_loops():
    assert _detect_streaming_hallucination("prefix" + " " * 400) is None
    loop = _detect_streaming_hallucination("prefix" + "abc" * 100)
    assert loop is not None
    assert is_transient_gemini_error(StreamingHallucinationError(loop))


def test_tex_streaming_guard_allows_multichar_table_cycles_only():
    table_cycle = " & value \\\\\n"
    assert _detect_streaming_hallucination(
        "prefix" + table_cycle * 40,
        max_period=1,
    ) is None
    single_character_loop = _detect_streaming_hallucination(
        "prefix" + "-" * 400,
        max_period=1,
    )
    assert single_character_loop is not None
    assert is_transient_anthropic_error(
        StreamingHallucinationError(single_character_loop)
    )


def test_anthropic_guard_discards_a_single_character_partial_response():
    class FakeMessages:
        @staticmethod
        def create(**_kwargs):
            event = SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="-" * 20),
            )
            return [event] * 20

    client = object.__new__(AnthropicClient)
    client.client = SimpleNamespace(messages=FakeMessages())
    client.num_retries = 1
    client.max_backoff_seconds = 0
    client.tokenizer = SimpleNamespace(encode=lambda text: list(text))

    with pytest.raises(StreamingHallucinationError):
        client.generate_content(
            "prompt",
            model="test-model",
            operation_name="test TeX stream",
            streaming_repetition_max_period=1,
        )


def test_codex_provider_resolves_active_openai_compatible_credentials(
    tmp_path: Path,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'model_provider = "test-proxy"\n'
        '[model_providers.test-proxy]\n'
        'base_url = "https://example.invalid/v1"\n'
        'experimental_bearer_token = "test-token"\n',
        encoding="utf-8",
    )

    resolved = _load_codex_openai_provider(
        {
            "type": "codex",
            "config_path": str(config_path),
        }
    )

    assert resolved == {
        "api_key": "test-token",
        "base_url": "https://example.invalid/v1",
    }


def test_source_archive_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        payload = b"private"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe path"):
        ArxivSourceResolver().materialize(archive, tmp_path / "source")
    assert not (tmp_path / "outside.txt").exists()


def test_source_archive_rejects_oversized_expanded_content(tmp_path: Path):
    archive = tmp_path / "oversized.tar"
    with tarfile.open(archive, "w") as tar:
        payload = b"12345"
        info = tarfile.TarInfo("main.tex")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="Expanded source exceeds"):
        ArxivSourceResolver(max_download_bytes=4).materialize(
            archive,
            destination,
        )
    assert not destination.exists()


def test_source_copy_rejects_output_nested_inside_source(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text("\\documentclass{article}")

    with pytest.raises(ValueError, match="must not be nested"):
        ArxivSourceResolver().materialize(source, source / "output" / "source")


def test_cjk_injection_is_idempotent():
    source = (
        "\\pdfoutput=1\n"
        "\\documentclass{article}\n"
        "\\usepackage[latin1]{inputenc}\n"
        "\\usepackage{microtype}\n"
        "\\UseMicrotypeSet[protrusion]{basicmath}\n"
        "\\newtheorem{theorem}{Theorem}\n"
        "\\newtheorem{proposition}[theorem]{Proposition}\n"
        "\\begin{document}Hello\\end{document}\n"
    )
    prepared = inject_cjk_support(source)
    assert "\\ifdefined\\pdfoutput\\pdfoutput=1\\fi" in prepared
    assert "\\usepackage[scheme=plain,fontset=fandol]{ctex}" in prepared
    assert "\\IfFontExistsTF{Noto Serif CJK SC}" in prepared
    assert "\\setCJKmainfont{Noto Serif CJK SC}" in prepared
    assert "\\setCJKmainfont{Source Han Serif SC}" in prepared
    assert "\\setCJKmainfont{Arial Unicode MS}" in prepared
    assert "\\usepackage[latin1]{inputenc}" not in prepared
    assert "\\usepackage{microtype}" not in prepared
    assert "\\UseMicrotypeSet" not in prepared
    assert "legacy Type1 fonts can break microtype" in prepared
    assert "\\renewcommand{\\abstractname}{摘要}" in prepared
    assert "\\patchcmd{\\abstract}{Abstract}{摘要}{}{}" in prepared
    assert "\\renewcommand{\\proofname}{证明}" in prepared
    assert "\\newtheorem{theorem}{定理}" in prepared
    assert "\\newtheorem{proposition}[theorem]{命题}" in prepared
    assert inject_cjk_support(prepared) == prepared


def test_legacy_cjk_wrappers_are_migrated_to_native_xelatex_support():
    source = (
        "\\documentclass{article}\n"
        "\\usepackage{CJKutf8}\n"
        "\\begin{document}\n"
        "\\begin{CJK*}{UTF8}{gbsn}中文\\end{CJK*}\n"
        "\\end{document}\n"
    )

    prepared = inject_cjk_support(source)

    assert "\\usepackage{CJKutf8}" not in prepared
    assert "\\begin{CJK" not in prepared
    assert "\\end{CJK" not in prepared
    assert "中文" in prepared
    assert "\\usepackage[scheme=plain,fontset=fandol]{ctex}" in prepared


def test_non_chinese_target_does_not_localize_tex_labels():
    source = (
        "\\documentclass{article}\n"
        "\\newtheorem{theorem}{Theorem}\n"
        "\\begin{document}Hello\\end{document}\n"
    )

    prepared = inject_cjk_support(source, target_language="Japanese")

    assert "\\usepackage[scheme=plain,fontset=fandol]{ctex}" in prepared
    assert "\\newtheorem{theorem}{Theorem}" in prepared
    assert "\\renewcommand{\\abstractname}{摘要}" not in prepared


def test_project_scanner_follows_body_includes_and_renders_exactly(tmp_path: Path):
    source = tmp_path / "source"
    _write_project(source)
    main = discover_main_tex(source)
    document = scan_project(source, main, unit_chars=1_000)

    assert main == "main.tex"
    assert set(document.sources) == {"main.tex", "sections/body.tex"}
    assert {unit.relative_path for unit in document.units} == {
        "main.tex",
        "sections/body.tex",
    }
    rendered = document.render({})
    assert rendered == document.sources
    assert "\\usepackage[scheme=plain,fontset=fandol]{ctex}" in rendered["main.tex"]


def test_project_scanner_translates_balanced_front_matter_without_the_preamble(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\newcommand{\\internalname}{Do not translate the preamble}\n"
        "\\title{An {English} title with $x$}\n"
        "\\begin{document}\n"
        "English body.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    document = scan_project(source, "main.tex", unit_chars=1_000)

    assert len(document.units) == 2
    assert document.units[0].source_text.strip() == "English body."
    assert (
        document.units[1].source_text
        == "\\title{An {English} title with $x$}"
    )
    assert all("\\newcommand" not in unit.source_text for unit in document.units)


def test_translation_prompt_keeps_an_exact_cacheable_prefix():
    initial = build_translation_messages(
        "English $x$.",
        source_language="English",
        target_language="Simplified Chinese",
    )
    continued = build_translation_messages(
        "English $x$.",
        source_language="English",
        target_language="Simplified Chinese",
        prefix="中文 $x$。",
    )
    assert initial == continued[:1]

    key1 = TranslationCache.key(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        messages=initial,
        prompt_version=TRANSLATION_PROMPT_VERSION,
    )
    key2 = TranslationCache.key(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        messages=list(initial),
        prompt_version=TRANSLATION_PROMPT_VERSION,
    )
    assert key1 == key2


def test_translation_cache_rejects_corrupted_content(tmp_path: Path):
    cache = TranslationCache(tmp_path)
    cache.put("abc", "valid")
    path = tmp_path / "abc.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.get("abc") is None


def test_translation_state_rebases_layout_when_source_units_are_identical(
    tmp_path: Path,
):
    state = TranslationState(tmp_path)
    original = {
        "id": "unit-00001",
        "relative_path": "main.tex",
        "start": 100,
        "end": 200,
        "source_sha256": "same-source",
    }
    state.initialize(
        source_id="paper",
        source_fingerprint="old-source-layout",
        layout_fingerprint="old-layout",
        main_tex="main.tex",
        units=[original],
    )
    state.data["units"]["unit-00001"]["status"] = "fallback_original"
    state.save()

    shifted = {
        **original,
        "start": 180,
        "end": 280,
    }
    state.initialize(
        source_id="paper",
        source_fingerprint="new-source-layout",
        layout_fingerprint="new-layout",
        main_tex="main.tex",
        units=[shifted],
    )

    assert state.data["source_fingerprint"] == "new-source-layout"
    assert state.data["layout_fingerprint"] == "new-layout"
    assert state.data["units"]["unit-00001"]["start"] == 180
    assert state.data["units"]["unit-00001"]["status"] == "fallback_original"


def test_translation_state_appends_new_units_without_losing_completed_work(
    tmp_path: Path,
):
    state = TranslationState(tmp_path)
    body = {
        "id": "unit-00001",
        "relative_path": "main.tex",
        "start": 100,
        "end": 200,
        "source_sha256": "same-body",
    }
    state.initialize(
        source_id="paper",
        source_fingerprint="body-only",
        layout_fingerprint="old-layout",
        main_tex="main.tex",
        units=[body],
    )
    state.data["units"]["unit-00001"]["status"] = "translated"
    state.save()
    title = {
        "id": "unit-00002",
        "relative_path": "main.tex",
        "start": 20,
        "end": 60,
        "source_sha256": "new-title",
    }

    state.initialize(
        source_id="paper",
        source_fingerprint="body-and-title",
        layout_fingerprint="new-layout",
        main_tex="main.tex",
        units=[{**body, "start": 120, "end": 220}, title],
    )

    assert state.data["units"]["unit-00001"]["status"] == "translated"
    assert state.data["units"]["unit-00002"]["status"] == "pending"
    assert state.pending_ids() == ["unit-00002"]


def test_translation_state_rejects_rebase_when_a_source_unit_changed(
    tmp_path: Path,
):
    state = TranslationState(tmp_path)
    original = {
        "id": "unit-00001",
        "relative_path": "main.tex",
        "start": 100,
        "end": 200,
        "source_sha256": "old-source",
    }
    state.initialize(
        source_id="paper",
        source_fingerprint="old-tree",
        layout_fingerprint="old-layout",
        main_tex="main.tex",
        units=[original],
    )
    state.data["units"]["unit-00001"]["status"] = "fallback_original"
    state.save()

    with pytest.raises(ValueError, match="Source tree changed"):
        state.initialize(
            source_id="paper",
            source_fingerprint="new-tree",
            layout_fingerprint="new-layout",
            main_tex="main.tex",
            units=[{**original, "source_sha256": "changed-source"}],
        )


def test_translation_state_resets_changed_units_while_still_pristine(
    tmp_path: Path,
):
    state = TranslationState(tmp_path)
    original = {
        "id": "unit-00001",
        "relative_path": "main.tex",
        "start": 100,
        "end": 200,
        "source_sha256": "old-source",
    }
    state.initialize(
        source_id="paper",
        source_fingerprint="old-tree",
        layout_fingerprint="old-layout",
        main_tex="main.tex",
        units=[original],
    )
    changed = {
        **original,
        "end": 180,
        "source_sha256": "normalized-source",
    }

    state.initialize(
        source_id="paper",
        source_fingerprint="normalized-tree",
        layout_fingerprint="normalized-layout",
        main_tex="main.tex",
        units=[changed],
    )

    assert state.data["source_fingerprint"] == "normalized-tree"
    assert state.data["units"]["unit-00001"]["end"] == 180
    assert state.data["units"]["unit-00001"]["status"] == "pending"


def test_translation_state_rejects_a_changed_language_contract(tmp_path: Path):
    state = TranslationState(tmp_path)
    unit = {
        "id": "unit-00001",
        "relative_path": "main.tex",
        "start": 100,
        "end": 200,
        "source_sha256": "same-source",
    }
    state.initialize(
        source_id="paper",
        source_fingerprint="source-tree",
        layout_fingerprint="layout",
        main_tex="main.tex",
        units=[unit],
        translation_spec={
            "source_language": "English",
            "target_language": "Simplified Chinese",
        },
    )

    with pytest.raises(ValueError, match="target language"):
        state.initialize(
            source_id="paper",
            source_fingerprint="source-tree",
            layout_fingerprint="layout",
            main_tex="main.tex",
            units=[unit],
            translation_spec={
                "source_language": "English",
                "target_language": "Japanese",
            },
        )


def test_failed_legacy_state_migration_does_not_write_a_new_contract(
    tmp_path: Path,
):
    state = TranslationState(tmp_path)
    unit = {
        "id": "unit-00001",
        "relative_path": "main.tex",
        "start": 100,
        "end": 200,
        "source_sha256": "same-source",
    }
    state.initialize(
        source_id="original-paper",
        source_fingerprint="source-tree",
        layout_fingerprint="layout",
        main_tex="main.tex",
        units=[unit],
    )
    persisted = json.loads(state.path.read_text(encoding="utf-8"))
    persisted.pop("translation_spec")
    state.path.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(ValueError, match="requested source differs"):
        state.initialize(
            source_id="different-paper",
            source_fingerprint="source-tree",
            layout_fingerprint="layout",
            main_tex="main.tex",
            units=[unit],
            translation_spec={"target_language": "Japanese"},
        )

    reloaded = json.loads(state.path.read_text(encoding="utf-8"))
    assert "translation_spec" not in reloaded


def test_pipeline_commits_compile_clean_units_and_resumes_without_llm(
    tmp_path: Path,
):
    source = tmp_path / "paper"
    _write_project(source)
    run_dir = tmp_path / "run"
    llm = FakeLLMClient()
    compiler = FakeCompiler()
    pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(
            unit_chars=1_000,
            repair_enabled=True,
        ),
        llm_client=llm,
        compiler=compiler,
        repair_agent=FailIfCalledRepairAgent(),
    )

    first = pipeline.run(source, run_dir=run_dir)
    first_call_count = len(llm.calls)
    assert first_call_count == 2
    assert all(
        kwargs["streaming_repetition_max_period"] == 1
        for _prompt, kwargs in llm.calls
    )
    assert first.summary["translated"] == 2
    assert first.summary["fallback_original"] == 0
    assert "中文" in (first.project_dir / "main.tex").read_text(encoding="utf-8")
    assert "中文" in (first.project_dir / "sections" / "body.tex").read_text(
        encoding="utf-8"
    )
    assert "PDF2EPUB-unit" not in (first.project_dir / "main.tex").read_text(
        encoding="utf-8"
    )

    second = pipeline.run(source, run_dir=run_dir)
    assert len(llm.calls) == first_call_count
    assert second.summary == first.summary

    state = json.loads(
        (run_dir / ".pdf2epub" / "state.json").read_text(encoding="utf-8")
    )
    for record in state["units"].values():
        assert record["provider_usage"]["cache_read_tokens"] == 100


def test_content_cache_survives_lost_transaction_state(tmp_path: Path):
    source = tmp_path / "paper"
    _write_project(source)
    run_dir = tmp_path / "run"
    first_llm = FakeLLMClient()
    first_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(unit_chars=1_000, repair_enabled=False),
        llm_client=first_llm,
        compiler=FakeCompiler(),
    )
    first_pipeline.run(source, run_dir=run_dir)
    assert len(first_llm.calls) == 2

    control_dir = run_dir / ".pdf2epub"
    (control_dir / "state.json").unlink()
    shutil.rmtree(control_dir / "units")

    second_llm = FakeLLMClient()
    second_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(unit_chars=1_000, repair_enabled=False),
        llm_client=second_llm,
        compiler=FakeCompiler(),
    )
    second = second_pipeline.run(source, run_dir=run_dir)

    assert second_llm.calls == []
    assert second.summary["translated"] == 2
    state = json.loads((control_dir / "state.json").read_text(encoding="utf-8"))
    assert all(record["local_cache_hit"] for record in state["units"].values())


def test_pipeline_restores_original_unit_after_compile_failure(tmp_path: Path):
    source = tmp_path / "paper"
    _write_project(source)
    pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(
            unit_chars=1_000,
            repair_enabled=False,
        ),
        llm_client=FakeLLMClient(lambda _source: "BROKEN_TRANSLATION"),
        compiler=FakeCompiler(),
    )

    result = pipeline.run(source, run_dir=tmp_path / "run")

    assert result.summary["fallback_original"] == 2
    assert "English introduction" in (result.project_dir / "main.tex").read_text(
        encoding="utf-8"
    )
    assert "BROKEN_TRANSLATION" not in (result.project_dir / "main.tex").read_text(
        encoding="utf-8"
    )


def test_retry_fallbacks_bypasses_the_previous_failed_response_cache(
    tmp_path: Path,
):
    source = tmp_path / "paper"
    _write_project(source)
    run_dir = tmp_path / "run"
    broken_llm = FakeLLMClient(lambda _source: "BROKEN_TRANSLATION")
    first_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(
            unit_chars=1_000,
            repair_enabled=False,
        ),
        llm_client=broken_llm,
        compiler=FakeCompiler(),
    )
    first = first_pipeline.run(source, run_dir=run_dir)
    assert first.summary["fallback_original"] == 2

    working_llm = FakeLLMClient()
    retry_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(
            unit_chars=1_000,
            repair_enabled=False,
            retry_fallbacks=True,
        ),
        llm_client=working_llm,
        compiler=FakeCompiler(),
    )
    second = retry_pipeline.run(source, run_dir=run_dir)

    assert len(working_llm.calls) == 2
    assert second.summary["translated"] == 2
    assert second.summary["fallback_original"] == 0


def test_retry_repaired_bypasses_cache_and_replaces_the_previous_unit(
    tmp_path: Path,
):
    source = tmp_path / "paper"
    _write_project(source)
    run_dir = tmp_path / "run"
    first_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(unit_chars=1_000, repair_enabled=False),
        llm_client=FakeLLMClient(),
        compiler=FakeCompiler(),
    )
    first_pipeline.run(source, run_dir=run_dir)

    state_path = run_dir / ".pdf2epub" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for record in state["units"].values():
        record["status"] = "repaired"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    retry_llm = FakeLLMClient(
        lambda text: text.replace("English", "重试译文")
    )
    retry_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(
            unit_chars=1_000,
            repair_enabled=False,
            retry_repaired=True,
        ),
        llm_client=retry_llm,
        compiler=FakeCompiler(),
    )
    result = retry_pipeline.run(source, run_dir=run_dir)

    assert len(retry_llm.calls) == 2
    assert result.summary["translated"] == 2
    assert result.summary["repaired"] == 0
    assert "重试译文" in (result.project_dir / "main.tex").read_text(
        encoding="utf-8"
    )


def test_failed_retry_repaired_keeps_the_previous_compile_safe_translation(
    tmp_path: Path,
):
    source = tmp_path / "paper"
    _write_project(source)
    run_dir = tmp_path / "run"
    first_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(unit_chars=1_000, repair_enabled=False),
        llm_client=FakeLLMClient(),
        compiler=FakeCompiler(),
    )
    first_pipeline.run(source, run_dir=run_dir)

    state_path = run_dir / ".pdf2epub" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    original_hashes = {
        unit_id: record["translation_sha256"]
        for unit_id, record in state["units"].items()
    }
    for record in state["units"].values():
        record["status"] = "repaired"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    retry_llm = FakeLLMClient(
        lambda text: f"BROKEN_TRANSLATION\n{text}"
    )
    retry_pipeline = TexTranslationPipeline(
        config={},
        options=TexTranslationOptions(
            unit_chars=1_000,
            repair_enabled=False,
            retry_repaired=True,
        ),
        llm_client=retry_llm,
        compiler=FakeCompiler(),
    )
    result = retry_pipeline.run(source, run_dir=run_dir)

    assert len(retry_llm.calls) == 2
    assert result.summary["repaired"] == 2
    project_text = (result.project_dir / "main.tex").read_text(encoding="utf-8")
    assert "中文" in project_text
    assert "BROKEN_TRANSLATION" not in project_text
    retained = json.loads(state_path.read_text(encoding="utf-8"))
    assert {
        unit_id: record["translation_sha256"]
        for unit_id, record in retained["units"].items()
    } == original_hashes
