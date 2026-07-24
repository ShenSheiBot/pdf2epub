from pathlib import Path
from types import SimpleNamespace

from pdf2epub.core.executor._protocol import ChainEntry
from pdf2epub.core.executor.state import UnitState
from pdf2epub.core.factory_v2 import create_model_chain_from_config
from pdf2epub.core.hooks import ErrorType
from pdf2epub.html_translation.builder import (
    BuildConfig,
    HTMLEpubBuilder,
    _make_json_safe,
)
from pdf2epub.refine import structure_analyzer
from pdf2epub.refine.structure_analyzer import StructureAnalyzer
from pdf2epub.utils.network_utils import OpenAIClient


def test_unit_state_retry_uses_total_quota_and_chain_not_error_quota():
    state = UnitState(
        chain=[ChainEntry(provider="gemini", model="model-a", mode="online")],
        total_quota=1,
        quotas={ErrorType.VALIDATION: 0},
    )

    assert state.can_retry(ErrorType.VALIDATION) is True


def test_create_model_chain_filters_disabled_vertex_and_falls_back():
    chain = create_model_chain_from_config(
        {
            "use_vertex": False,
            "translation": {
                "models": [
                    {"provider": "vertex", "model": "gemini-batch"},
                    {"provider": "gemini", "model": "gemini-online"},
                ]
            },
        },
        task_type="translate",
    )

    assert [(entry.provider, entry.model) for entry in chain] == [
        ("gemini", "gemini-online")
    ]

    fallback_chain = create_model_chain_from_config(
        {
            "use_vertex": False,
            "translation": {
                "models": [{"provider": "vertex", "model": "gemini-batch"}]
            },
        },
        task_type="translate",
    )

    assert [(entry.provider, entry.model) for entry in fallback_chain] == [
        ("gemini", "gemini-2.0-flash")
    ]


def test_create_model_chain_supports_legacy_polish_models():
    chain = create_model_chain_from_config(
        {
            "polish_models": [
                {
                    "provider": "vertex",
                    "model": "gemini-2.5-flash",
                    "mode": "batch",
                }
            ]
        },
        task_type="polish",
    )

    assert [(entry.provider, entry.model, entry.mode) for entry in chain] == [
        ("vertex", "gemini-2.5-flash", "batch")
    ]


def test_html_builder_replaces_htm_files(tmp_path: Path):
    translated_dir = tmp_path / "translated"
    translated_dir.mkdir()
    (translated_dir / "chapter.htm").write_text("translated", encoding="utf-8")

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    target = extract_dir / "chapter.htm"
    target.write_text("original", encoding="utf-8")

    builder = HTMLEpubBuilder(
        BuildConfig(
            original_epub=tmp_path / "input.epub",
            translated_dir=translated_dir,
            output_path=tmp_path / "output.epub",
            book_title="Book",
        )
    )

    assert builder._replace_xhtml_files(extract_dir) == 1
    assert target.read_text(encoding="utf-8") == "translated"


def test_html_builder_normalizes_inline_block_wrapper_css():
    css = (
        ".bodytext {\n    font-size: 0.77419em\n    }\n"
        ".caption {\n    font-size: 0.8em\n    }\n"
    )

    normalized = HTMLEpubBuilder._normalize_css_content(css, {"bodytext"})

    assert "display: block" in normalized
    assert "font-size: 0.77419em" in normalized
    assert ".caption {\n    font-size: 0.8em" in normalized


def test_html_builder_does_not_duplicate_wrapper_display():
    css = ".bodytext {\n    display: block;\n    font-size: 0.77419em\n    }\n"

    normalized = HTMLEpubBuilder._normalize_css_content(css, {"bodytext"})

    assert normalized.count("display: block") == 1


def test_html_builder_detects_inline_block_wrapper_class(tmp_path: Path):
    (tmp_path / "chapter.htm").write_text(
        '<span class="bodytext"><div class="para">Text</div></span>',
        encoding="utf-8",
    )

    assert HTMLEpubBuilder._classes_used_as_inline_block_wrappers(tmp_path) == {"bodytext"}


def test_html_builder_ignores_inline_span_class(tmp_path: Path):
    (tmp_path / "chapter.htm").write_text(
        '<p>Before <span class="bodytext">small inline text</span> after.</p>',
        encoding="utf-8",
    )

    assert HTMLEpubBuilder._classes_used_as_inline_block_wrappers(tmp_path) == set()


def test_html_builder_skips_mixed_inline_and_block_wrapper_class(tmp_path: Path):
    (tmp_path / "chapter.htm").write_text(
        '<span class="bodytext"><div class="para">Text</div></span>'
        '<p>Before <span class="bodytext">small inline text</span> after.</p>',
        encoding="utf-8",
    )

    assert HTMLEpubBuilder._classes_used_as_inline_block_wrappers(tmp_path) == set()


def test_html_builder_skips_class_used_on_nonspan_elements(tmp_path: Path):
    (tmp_path / "chapter.htm").write_text(
        '<span class="bodytext"><div class="para">Text</div></span>'
        '<div class="bodytext">Already block</div>',
        encoding="utf-8",
    )

    assert HTMLEpubBuilder._classes_used_as_inline_block_wrappers(tmp_path) == set()


def test_html_builder_ignores_self_closing_span_before_block(tmp_path: Path):
    (tmp_path / "chapter.htm").write_text(
        '<span class="bodytext"/><div class="para">Text</div>',
        encoding="utf-8",
    )

    assert HTMLEpubBuilder._classes_used_as_inline_block_wrappers(tmp_path) == set()


def test_make_json_safe_converts_non_serializable_values():
    class Token:
        def __str__(self):
            return "token-value"

    result = _make_json_safe({"node": Token(), "items": (Token(), 3)})

    assert result == {"node": "token-value", "items": ["token-value", 3]}


def test_toc_completeness_warns_for_under_extracted_toc(monkeypatch):
    messages = {"warning": []}
    fake_logger = SimpleNamespace(
        info=lambda message: None,
        warning=lambda message: messages["warning"].append(message),
    )
    monkeypatch.setattr(structure_analyzer, "logger", fake_logger)

    StructureAnalyzer._validate_toc_completeness(
        chapters=[{"title": "1 First"}],
        toc_reference="\n".join(["1 First", "2 Second", "3 Third"]),
        total_pages=300,
    )

    assert any("TOC COMPLETENESS ISSUE" in msg for msg in messages["warning"])
    assert any("Less than half" in msg for msg in messages["warning"])


def test_openai_streaming_skips_empty_choice_chunks():
    class FakeTokenizer:
        def encode(self, text):
            return text.split()

    class FakeCompletions:
        def create(self, **_kwargs):
            return iter(
                [
                    SimpleNamespace(choices=[]),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="hello "),
                                finish_reason=None,
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="world"),
                                finish_reason="stop",
                            )
                        ]
                    ),
                ]
            )

    client = object.__new__(OpenAIClient)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    client.default_model = "test-model"
    client.tokenizer = FakeTokenizer()

    assert client.generate_content("prompt", operation_name="test") == "hello world"
    assert client._last_finish_reason == "stop"
