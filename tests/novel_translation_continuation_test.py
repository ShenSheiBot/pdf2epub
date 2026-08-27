from types import SimpleNamespace

from pdf2epub.html_translation.novel_translator import NovelState, NovelTranslator
from pdf2epub.html_translation.novel_verifier import verify_translation


def test_truncated_novel_translation_continues_same_conversation(monkeypatch):
    translator = object.__new__(NovelTranslator)
    translator._embedding_provider = None
    translator._embedding_model = "unused"
    translator._hallucination_threshold = 0.75
    translator._strip_model_configs = None
    translator._get_model_configs = lambda: [
        {"provider": "anthropic", "model": "test-model"}
    ]
    translator._get_llm_client = lambda: object()

    model_outputs = iter(["译文一\n译文二", "译文三\n译文四"])
    requests = []

    def fake_stream(*, messages, system_text, max_output_tokens, operation_name):
        requests.append(messages.copy())
        return next(model_outputs)

    translator._stream_with_token_cutoff = fake_stream

    verifier_inputs = []

    def fake_verify_translation(*, translated_text, **kwargs):
        verifier_inputs.append(translated_text)
        if len(verifier_inputs) == 1:
            return translated_text, "continue"
        return translated_text, "complete"

    monkeypatch.setattr(
        "pdf2epub.html_translation.novel_verifier.verify_translation",
        fake_verify_translation,
    )

    def reject_chunked_fallback(**kwargs):
        raise AssertionError("a progressing continuation must not enter chunked fallback")

    monkeypatch.setattr(
        "pdf2epub.html_translation.chunked_translator.translate_remaining",
        reject_chunked_fallback,
    )

    translated, exhausted = translator._run_translation(
        SimpleNamespace(file_name="chapter.txt"),
        "原文一\n原文二\n原文三\n原文四",
        "",
    )

    assert translated == "译文一\n译文二\n译文三\n译文四"
    assert exhausted is False
    assert verifier_inputs == [
        "译文一\n译文二",
        "译文一\n译文二\n译文三\n译文四",
    ]
    assert [message["role"] for message in requests[1]] == [
        "user",
        "assistant",
        "user",
    ]
    assert requests[1][0]["content"] == (
        "请翻译：\n原文一\n原文二\n原文三\n原文四"
    )
    assert requests[1][1]["content"] == "译文一\n译文二"


def test_verifier_does_not_accept_a_translation_missing_one_line(monkeypatch):
    monkeypatch.setattr(
        "pdf2epub.html_translation.novel_verifier._check_preamble",
        lambda *args, **kwargs: "translation",
    )
    monkeypatch.setattr(
        "pdf2epub.html_translation.novel_verifier._check_alignment",
        lambda *args, **kwargs: "A",
    )

    fixed, action = verify_translation(
        source_text="原文一\n原文二\n原文三\n原文四\n原文五\n原文六",
        translated_text="译文一\n译文二\n译文三\n译文四\n译文五",
        llm_client=object(),
        model_configs=[],
    )

    assert fixed == "译文一\n译文二\n译文三\n译文四\n译文五"
    assert action == "continue"


def test_existing_incomplete_translation_continues_as_assistant_prefix(monkeypatch):
    translator = object.__new__(NovelTranslator)
    translator._embedding_provider = None
    translator._embedding_model = "unused"
    translator._hallucination_threshold = 0.75
    translator._strip_model_configs = None
    translator._get_model_configs = lambda: [
        {"provider": "anthropic", "model": "test-model"}
    ]
    translator._get_llm_client = lambda: object()

    requests = []

    def fake_stream(*, messages, system_text, max_output_tokens, operation_name):
        requests.append(messages.copy())
        return "译文四"

    translator._stream_with_token_cutoff = fake_stream
    monkeypatch.setattr(
        "pdf2epub.html_translation.novel_verifier.verify_translation",
        lambda **kwargs: (kwargs["translated_text"], "complete"),
    )

    translated, exhausted = translator._run_translation(
        SimpleNamespace(file_name="chapter.txt"),
        "原文一\n原文二\n原文三\n原文四",
        "",
        existing_translation="译文一\n译文二\n译文三",
    )

    assert translated == "译文一\n译文二\n译文三\n译文四"
    assert exhausted is False
    assert [message["role"] for message in requests[0]] == [
        "user",
        "assistant",
        "user",
    ]
    assert requests[0][0]["content"] == "请翻译：\n原文一\n原文二\n原文三\n原文四"
    assert requests[0][1]["content"] == "译文一\n译文二\n译文三"


def test_resume_reopens_completed_unit_when_translation_is_short(tmp_path):
    translator = object.__new__(NovelTranslator)
    translator.resume = True
    translator.state_path = tmp_path / "novel_state.json"
    translator.translated_dir = tmp_path / "translated_novel"
    translator.translated_dir.mkdir()
    translator.glossary_manager = None

    source_path = tmp_path / "034_chapter.txt"
    source_path.write_text("原文一\n原文二", encoding="utf-8")
    destination = translator.translated_dir / source_path.name
    destination.write_text("译文一", encoding="utf-8")
    NovelState(current_unit_index=1, completed_units=[0]).save(translator.state_path)
    unit = SimpleNamespace(
        has_content=True,
        text_path=source_path,
        file_name="chapter",
    )

    calls = []

    def fake_translate_chapter(target, existing_translation=None):
        calls.append(existing_translation)
        destination.write_text("译文一\n译文二", encoding="utf-8")
        return False

    translator._translate_chapter = fake_translate_chapter

    summary = translator.translate_all([unit])

    assert calls == ["译文一"]
    assert destination.read_text(encoding="utf-8") == "译文一\n译文二"
    assert summary["translated"] == 1
