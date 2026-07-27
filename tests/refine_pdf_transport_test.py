import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, BadRequestError, OpenAI

from pdf2epub.refine.adaptive_pdf_call import (
    AdaptivePdfCall,
    BatchValidationError,
    DirectAnalysisCall,
    MergeValidationError,
    PdfPageLimitLearner,
    run_adaptive_batches,
)
from pdf2epub.refine import adaptive_pdf_call, boundary_agent, structure_analyzer
from pdf2epub.refine.pdf_transport import (
    GeminiPdfTransport,
    OpenAIResponsesPdfTransport,
    PdfContextLimitError,
    PdfPayloadTooLargeError,
    create_pdf_transport,
)
from pdf2epub.refine.structure_analyzer import StructureAnalyzer
from pdf2epub.utils.llm_client import LLMGenerateConfig
from pdf2epub.refine.toc_tree import TOCNode


class _CaptureResponses:
    def __init__(self, response=None, error=None):
        self.requests = []
        self.response = response or SimpleNamespace(output_text='{"chapters": []}')
        self.error = error

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


class _CaptureClient:
    def __init__(self, response=None, error=None):
        self.responses = _CaptureResponses(response=response, error=error)


def _transport(client):
    return OpenAIResponsesPdfTransport(
        {"api_key": "test-key", "base_url": "http://proxy.local/v1/responses"},
        client=client,
    )


def test_openai_responses_pdf_transport_sends_data_url_and_validated_prefix():
    client = _CaptureClient()
    transport = _transport(client)

    result = transport.generate_pdf(
        model="gpt-test",
        prompt="Find chapters.",
        pdf_data=b"%PDF-test",
        config=LLMGenerateConfig(temperature=0.1, max_tokens=4096),
        operation_name="test PDF batch",
        prefix='{"chapters": [',
    )

    assert result == '{"chapters": []}'
    assert transport.base_url == "http://proxy.local/v1"

    request = client.responses.requests[0]
    assert request["model"] == "gpt-test"
    assert request["stream"] is False
    assert "max_output_tokens" not in request
    assert "temperature" not in request
    assert request["store"] is False
    assert "previous_response_id" not in request

    content = request["input"][0]["content"]
    file_item = next(item for item in content if item["type"] == "input_file")
    assert file_item["filename"] == "refine-batch.pdf"
    assert file_item["file_data"] == "data:application/pdf;base64,JVBERi10ZXN0"
    assert any(
        item["type"] == "input_text" and '{"chapters": [' in item["text"]
        for item in content
    )


def test_openai_sdk_posts_pdf_request_to_responses_endpoint():
    observed = {}

    def handler(request):
        observed["url"] = str(request.url)
        observed["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": '{"ok": true}'}],
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    sdk_client = OpenAI(
        api_key="test-key",
        base_url="http://proxy.local/v1",
        http_client=http_client,
    )
    transport = OpenAIResponsesPdfTransport(
        {"api_key": "test-key", "base_url": "http://proxy.local/v1"},
        client=sdk_client,
    )

    try:
        assert transport.generate_pdf(
            model="gpt-test",
            prompt="Find chapters.",
            pdf_data=b"%PDF-test",
            config=LLMGenerateConfig(),
            operation_name="test PDF batch",
        ) == '{"ok": true}'
    finally:
        sdk_client.close()

    assert observed["url"] == "http://proxy.local/v1/responses"
    assert observed["json"]["store"] is False
    file_item = observed["json"]["input"][0]["content"][1]
    assert file_item == {
        "type": "input_file",
        "filename": "refine-batch.pdf",
        "file_data": "data:application/pdf;base64,JVBERi10ZXN0",
    }


def test_openai_responses_context_limit_fails_closed():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://proxy.local/v1/responses"),
    )
    error = BadRequestError(
        "context_length_exceeded: request is too large",
        response=response,
        body={"error": {"message": "context_length_exceeded"}},
    )
    transport = _transport(_CaptureClient(error=error))

    with pytest.raises(PdfContextLimitError, match="No automatic batch reduction"):
        transport.generate_pdf(
            model="gpt-test",
            prompt="Find chapters.",
            pdf_data=b"%PDF-test",
            config=LLMGenerateConfig(),
            operation_name="test PDF batch",
        )


def test_openai_responses_payload_limit_is_adaptively_split():
    response = httpx.Response(
        413,
        request=httpx.Request("POST", "http://proxy.local/v1/responses"),
    )
    error = APIStatusError(
        "request entity too large",
        response=response,
        body={"error": {"message": "request entity too large"}},
    )
    transport = _transport(_CaptureClient(error=error))

    with pytest.raises(PdfPayloadTooLargeError, match="fewer pages"):
        transport.generate_pdf(
            model="gpt-test",
            prompt="Find chapters.",
            pdf_data=b"%PDF-test",
            config=LLMGenerateConfig(),
            operation_name="test PDF batch",
        )

    learner = PdfPageLimitLearner(initial_limit=50, min_limit=10)
    attempts = []

    def process_batch(pages, *_args):
        attempts.append(list(pages))
        if len(pages) > 25:
            raise PdfPayloadTooLargeError("payload too large")
        return list(pages)

    results = run_adaptive_batches(
        list(range(1, 51)),
        process_batch,
        learner,
        is_503_fn=adaptive_pdf_call.is_503_error,
        operation_name="payload test",
    )

    assert len(results) == 2
    assert [len(pages) for pages in attempts] == [50, 25, 25]
    assert learner.limit == 25


def test_context_limit_does_not_trigger_adaptive_batch_splitting():
    learner = PdfPageLimitLearner(initial_limit=50, min_limit=20)
    attempted_batches = []

    def process_batch(pages, *_args):
        attempted_batches.append(pages)
        raise PdfContextLimitError("context_length_exceeded")

    with pytest.raises(PdfContextLimitError):
        run_adaptive_batches(
            list(range(1, 51)),
            process_batch,
            learner,
            is_503_fn=lambda _error: False,
            operation_name="test",
        )

    assert attempted_batches == [list(range(1, 51))]
    assert learner.limit == 50


def test_toc_context_limit_is_not_misclassified_as_missing_toc(monkeypatch, tmp_path):
    class ContextLimitTocCall:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            raise PdfContextLimitError("context_length_exceeded")

    monkeypatch.setattr(structure_analyzer, "TocDetectionCall", ContextLimitTocCall)
    analyzer = StructureAnalyzer(
        structure_client=object(),
        structure_model="test",
        toc_model="test",
        analysis_client=object(),
        analysis_model="test",
        config={},
    )
    batch_ctx = SimpleNamespace(total_pages=1, page_limit=50, toc_sample_pages=10)

    with pytest.raises(PdfContextLimitError):
        analyzer.detect_toc_location(tmp_path / "book.pdf", batch_ctx)


def test_toc_auth_or_repair_error_is_not_misclassified_as_missing_toc(monkeypatch, tmp_path):
    class FailingTocCall:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            raise RuntimeError("JSON repair agent returned 401")

    monkeypatch.setattr(structure_analyzer, "TocDetectionCall", FailingTocCall)
    analyzer = StructureAnalyzer(
        structure_client=object(),
        structure_model="test",
        toc_model="test",
        analysis_client=object(),
        analysis_model="test",
        config={},
    )
    batch_ctx = SimpleNamespace(total_pages=1, page_limit=50, toc_sample_pages=10)

    with pytest.raises(RuntimeError, match="401"):
        analyzer.detect_toc_location(tmp_path / "book.pdf", batch_ctx)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"toc_start": 1, "toc_end": 2}',
        '{"has_toc": true, "toc_start": null, "toc_end": 2}',
        '{"has_toc": true, "toc_start": 3, "toc_end": 2}',
    ],
)
def test_malformed_toc_detection_never_becomes_no_toc(payload):
    call = structure_analyzer.TocDetectionCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
    )

    with pytest.raises(ValueError):
        call.parse_result(payload)


def test_toc_detection_rejects_pages_not_observed_in_its_batch():
    call = structure_analyzer.TocDetectionCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
    )
    call.build_prompt([1, 2, 3], 0, 1)

    assert call.validate_batch_result(
        {
            "has_toc": True,
            "toc_start": 2,
            "toc_end": 3,
        },
        0,
        1,
    ) == []
    assert call.validate_batch_result(
        {
            "has_toc": True,
            "toc_start": 20,
            "toc_end": 21,
        },
        0,
        1,
    )
    call.build_prompt([1, 2, 99, 100], 0, 1)
    assert call.validate_batch_result(
        {
            "has_toc": True,
            "toc_start": 2,
            "toc_end": 99,
        },
        0,
        1,
    )


def test_non_google_structure_provider_requires_explicit_pdf_transport():
    config = {
        "credentials": {
            "providers": {
                "openai_pdf": {
                    "type": "openai",
                    "api_key": "test-key",
                    "base_url": "http://proxy.local/v1",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="cannot use Gemini PDF parts"):
        create_pdf_transport(
            config=config,
            structure_provider="openai_pdf",
            structure_client=object(),
        )
    with pytest.raises(ValueError, match="cannot use Gemini PDF parts"):
        create_pdf_transport(
            config=config,
            structure_provider="openai_pdf",
            structure_client=object(),
            transport_config="gemini",
        )

    for provider_name in ("deepseek", "anthropic"):
        inferred_config = {
            "credentials": {
                "providers": {
                    provider_name: {
                        "api_key": "test-key",
                    }
                }
            }
        }
        with pytest.raises(ValueError, match="cannot use Gemini PDF parts"):
            create_pdf_transport(
                config=inferred_config,
                structure_provider=provider_name,
                structure_client=object(),
                transport_config="gemini",
            )


def test_google_default_and_direct_analysis_overlap_remain_configurable():
    transport = create_pdf_transport(
        config={"credentials": {"providers": {"gemini": {"type": "google"}}}},
        structure_provider="gemini",
        structure_client=object(),
    )
    assert isinstance(transport, GeminiPdfTransport)

    runtime_config = {"refine": {"agent_request_limit": 17}}
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(initial_limit=50, min_limit=20),
        book_title="Test",
        overlap_pages=5,
        runtime_config=runtime_config,
    )
    assert call.overlap == 5
    assert call._runtime_config is runtime_config


def test_boundary_agent_model_limit_uses_explicit_runtime_config(monkeypatch):
    observed = {}

    def fake_get_model_and_limits(runtime_config):
        observed["config"] = runtime_config
        return object(), "test-model", 321

    monkeypatch.setattr(boundary_agent, "get_model_and_limits", fake_get_model_and_limits)
    monkeypatch.setattr(boundary_agent, "_model_max_tokens", None)
    monkeypatch.setattr(boundary_agent, "_model_limit_config_key", None)
    runtime_config = {"credentials": {"providers": {}}}

    assert boundary_agent.get_model_max_tokens(runtime_config) == 321
    assert observed["config"] is runtime_config


def test_boundary_agent_honors_explicit_refine_agent(monkeypatch):
    import pydantic_ai.models.openai
    import pydantic_ai.providers.openai

    observed = {}
    monkeypatch.setattr(
        pydantic_ai.providers.openai,
        "OpenAIProvider",
        lambda **kwargs: ("provider", kwargs),
    )

    def fake_model(model_name, provider):
        observed["model"] = model_name
        observed["provider"] = provider
        return "explicit-boundary-model"

    monkeypatch.setattr(
        pydantic_ai.models.openai,
        "OpenAIChatModel",
        fake_model,
    )
    monkeypatch.setattr(boundary_agent, "OpenAIProvider", lambda **kwargs: ("provider", kwargs))
    monkeypatch.setattr(boundary_agent, "OpenAIChatModel", fake_model)

    model, model_name, max_tokens = boundary_agent.get_model_and_limits(
        {
            "credentials": {
                "providers": {
                    "anthropic": {
                        "type": "anthropic",
                        "api_key": "legacy-key",
                    },
                    "repair": {
                        "type": "openai",
                        "api_key": "explicit-key",
                        "base_url": "https://repair.example/v1",
                    },
                }
            },
            "refine": {
                "agent": {
                    "provider": "repair",
                    "model": "explicit-repair-model",
                }
            },
            "model_output_limits": {
                "_default": 4000,
                "explicit-repair-model": 12345,
            },
        }
    )

    assert model == "explicit-boundary-model"
    assert model_name == "explicit-repair-model"
    assert max_tokens == 12345
    assert observed["provider"][1]["api_key"] == "explicit-key"


def test_official_deepseek_refine_agents_use_deepseek_capability_profile():
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    runtime_config = {
        "credentials": {
            "providers": {
                "repair": {
                    "type": "openai",
                    "api_key": "test-key",
                    "base_url": "https://api.deepseek.com/v1",
                }
            }
        },
        "refine": {
            "agent": {
                "provider": "repair",
                "model": "deepseek-v4-flash",
            }
        },
    }

    adaptive_call = AdaptivePdfCall(
        client=object(),
        model="structure-model",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        runtime_config=runtime_config,
    )
    adaptive_model = adaptive_call._get_agent_model()
    boundary_model, _, _ = boundary_agent.get_model_and_limits(runtime_config)

    for model in (adaptive_model, boundary_model):
        profile = OpenAIModelProfile.from_profile(model.profile)
        assert profile.openai_supports_tool_choice_required is False
        assert profile.openai_chat_thinking_field == "reasoning_content"
        assert profile.openai_chat_send_back_thinking_parts == "field"


def test_anthropic_refine_agent_prefers_process_api_key(monkeypatch):
    observed = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "process-key")
    monkeypatch.setattr(
        boundary_agent,
        "AnthropicProvider",
        lambda **kwargs: observed.update(kwargs) or ("provider", kwargs),
    )
    monkeypatch.setattr(
        boundary_agent,
        "AnthropicModel",
        lambda model_name, provider: ("model", model_name, provider),
    )

    model, model_name, _ = boundary_agent.get_model_and_limits(
        {
            "credentials": {
                "providers": {
                    "anthropic": {
                        "type": "anthropic",
                        "api_key": "stale-config-key",
                        "base_url": "https://anthropic.example",
                    }
                }
            },
            "refine": {
                "agent": {
                    "provider": "anthropic",
                    "model": "test-haiku",
                }
            },
        }
    )

    assert model_name == "test-haiku"
    assert model[0] == "model"
    assert observed["api_key"] == "process-key"


def test_explicit_google_agent_uses_google_provider_module(monkeypatch):
    import google.genai
    import pydantic_ai.models.google
    import pydantic_ai.providers.google

    observed = []
    monkeypatch.setattr(
        google.genai,
        "Client",
        lambda **kwargs: ("client", kwargs),
    )
    monkeypatch.setattr(
        pydantic_ai.providers.google,
        "GoogleProvider",
        lambda **kwargs: ("provider", kwargs),
    )

    def fake_model(model_name, provider):
        observed.append((model_name, provider))
        return "google-agent"

    monkeypatch.setattr(
        pydantic_ai.models.google,
        "GoogleModel",
        fake_model,
    )
    runtime_config = {
        "credentials": {
            "providers": {
                "repair": {
                    "type": "google",
                    "api_key": "test-key",
                }
            }
        },
        "refine": {
            "agent": {
                "provider": "repair",
                "model": "gemini-test",
            }
        },
    }

    call = AdaptivePdfCall(
        client=object(),
        model="structure-model",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        runtime_config=runtime_config,
    )
    assert call._get_agent_model() == "google-agent"

    model, model_name, _ = boundary_agent.get_model_and_limits(
        runtime_config
    )
    assert model == "google-agent"
    assert model_name == "gemini-test"
    assert len(observed) == 2


def test_explicit_refine_agent_beats_legacy_anthropic_default(monkeypatch):
    import pydantic_ai.models.openai
    import pydantic_ai.providers.openai

    observed = {}

    monkeypatch.setattr(
        pydantic_ai.providers.openai,
        "OpenAIProvider",
        lambda **kwargs: ("provider", kwargs),
    )

    def fake_model(model_name, provider):
        observed["model"] = model_name
        observed["provider"] = provider
        return "explicit-model"

    monkeypatch.setattr(
        pydantic_ai.models.openai,
        "OpenAIChatModel",
        fake_model,
    )

    call = AdaptivePdfCall(
        client=object(),
        model="structure-model",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        runtime_config={
            "credentials": {
                "providers": {
                    "anthropic": {
                        "type": "anthropic",
                        "api_key": "legacy-key",
                    },
                    "repair": {
                        "type": "openai",
                        "api_key": "explicit-key",
                        "base_url": "https://repair.example/v1",
                    },
                }
            },
            "refine": {
                "agent": {
                    "provider": "repair",
                    "model": "explicit-repair-model",
                }
            },
        },
    )

    assert call._get_agent_model() == "explicit-model"
    assert observed["model"] == "explicit-repair-model"
    assert observed["provider"][1]["api_key"] == "explicit-key"


def test_invalid_merge_fails_closed_after_configured_retries(monkeypatch):
    class InvalidMergeCall(AdaptivePdfCall):
        operation_name = "test merge"
        merge_max_retries = 2

        def build_merge_prompt(self, _results):
            return "merge these results"

        def validate_merge(self, _merged, _original_results):
            return False

    attempts = []

    def fake_agent_loop(**_kwargs):
        attempts.append(True)
        return '{"chapters": []}'

    monkeypatch.setattr(
        "pdf2epub.refine.adaptive_pdf_call.run_agent_loop_sync",
        fake_agent_loop,
    )

    call = InvalidMergeCall(
        client=SimpleNamespace(
            get_default_config=lambda **_kwargs: LLMGenerateConfig(),
        ),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
    )
    monkeypatch.setattr(call, "_get_agent_model", lambda: object())

    with pytest.raises(MergeValidationError, match="refusing to continue"):
        call.merge_results([{"chapters": []}, {"chapters": []}])

    assert len(attempts) == 3


def test_invalid_batch_fails_closed_after_configured_retries(
    monkeypatch,
    tmp_path,
):
    class InvalidBatchCall(AdaptivePdfCall):
        batch_validation_retries = 2

        def build_prompt(self, *_args):
            return "analyze"

        def validate_batch_result(self, *_args):
            return ["No chapters extracted"]

    attempts = []

    def fake_agent_loop(**_kwargs):
        attempts.append(True)
        return '{"chapters": []}'

    monkeypatch.setattr(
        adaptive_pdf_call,
        "run_agent_loop_sync",
        fake_agent_loop,
    )
    call = InvalidBatchCall(
        client=SimpleNamespace(
            get_default_config=lambda **_kwargs: LLMGenerateConfig(),
        ),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
    )
    monkeypatch.setattr(call, "_get_agent_model", lambda: object())

    with pytest.raises(BatchValidationError, match="refusing to continue"):
        call.run(tmp_path / "book.pdf", [1])

    assert len(attempts) == 3


def test_validated_pdf_batches_are_reused_for_a_fresh_merge(
    monkeypatch,
    tmp_path,
):
    class CacheableCall(AdaptivePdfCall):
        def build_prompt(self, batch_pages, batch_idx, total_batches):
            return (
                f"analyze pages {batch_pages} "
                f"batch {batch_idx + 1}/{total_batches}"
            )

        def validate_batch_result(self, result, *_args):
            return [] if result.get("chapters") else ["missing chapters"]

    attempts = []

    def fake_agent_loop(**_kwargs):
        attempts.append(True)
        return (
            '{"chapters":[{"title":"Chapter","start_page":1,'
            '"end_page":1}]}'
        )

    monkeypatch.setattr(
        adaptive_pdf_call,
        "run_agent_loop_sync",
        fake_agent_loop,
    )
    client = SimpleNamespace(
        get_default_config=lambda **_kwargs: LLMGenerateConfig(),
    )

    def run_once(model="test"):
        call = CacheableCall(
            client=client,
            model=model,
            prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
            learner=PdfPageLimitLearner(),
        )
        monkeypatch.setattr(call, "_get_agent_model", lambda: object())
        return call.run(
            tmp_path / "book.pdf",
            [1],
            artifacts_dir=tmp_path / "artifacts",
        )

    first = run_once()
    second = run_once()

    assert first == second
    assert len(attempts) == 1

    run_once(model="changed-model")
    assert len(attempts) == 2


def test_batch_cache_is_bound_to_responses_endpoint_and_credentials():
    first_transport = OpenAIResponsesPdfTransport(
        {
            "api_key": "first-key",
            "base_url": "https://first.example/v1",
        },
        client=_CaptureClient(),
    )
    second_transport = OpenAIResponsesPdfTransport(
        {
            "api_key": "second-key",
            "base_url": "https://second.example/v1",
        },
        client=_CaptureClient(),
    )

    def fingerprint(transport):
        call = AdaptivePdfCall(
            client=object(),
            model="test",
            prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
            learner=PdfPageLimitLearner(),
            pdf_transport=transport,
        )
        return call._batch_cache_fingerprint(
            prompt="analyze",
            pdf_data=b"%PDF-test",
            batch_pages=[1],
        )

    assert fingerprint(first_transport) != fingerprint(second_transport)


def test_direct_merge_prompt_carries_observed_page_evidence():
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        book_title="Test",
    )

    prompt = call.build_merge_prompt(
        [{"chapters": []}, {"chapters": []}],
        batch_pages=[[1, 2, 4], [5, 6]],
    )

    assert "OBSERVED PDF PAGES: 1-2, 4" in prompt
    assert "OBSERVED PDF PAGES: 5-6" in prompt
    assert "can support a start_page or end_page only" in prompt
    assert "never override an in-range" in prompt


def test_direct_merge_validator_rejects_recursive_omissions_and_inventions():
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        book_title="Test",
    )
    source = [
        {
            "chapters": [
                {
                    "title": "Part One",
                    "start_page": 1,
                    "end_page": 10,
                    "children": [
                        {
                            "title": "Section A",
                            "start_page": 1,
                            "end_page": 10,
                        }
                    ],
                }
            ]
        },
        {
            "chapters": [
                {
                    "title": "Part Two",
                    "start_page": 11,
                    "end_page": 20,
                }
            ]
        },
    ]
    merged = {
        "chapters": [
            {
                "title": "Part One",
                "start_page": 1,
                "end_page": 10,
            },
            {
                "title": "Invented",
                "start_page": 900,
                "end_page": 999,
            },
        ]
    }

    issues = call.get_merge_validation_issues(merged, source)

    assert any("omitted" in issue for issue in issues)
    assert any("invented" in issue for issue in issues)


def test_direct_merge_validator_requires_observed_source_page_claims():
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        book_title="Test",
    )
    source = [
        {
            "chapters": [
                {
                    "title": "Chapter",
                    "start_page": 1,
                    "end_page": 10,
                }
            ]
        },
        {
            "chapters": [
                {
                    "title": "Chapter",
                    "start_page": 1,
                    "end_page": 10,
                }
            ]
        },
    ]
    observed = [list(range(1, 6)), list(range(6, 11))]

    assert call.get_merge_evidence_issues(
        {
            "chapters": [
                {
                    "title": "Chapter",
                    "start_page": 1,
                    "end_page": 10,
                }
            ]
        },
        source,
        observed,
    ) == []
    assert call.get_merge_evidence_issues(
        {
            "chapters": [
                {
                    "title": "Chapter",
                    "start_page": 1,
                    "end_page": 999,
                }
            ]
        },
        source,
        observed,
    )


def test_direct_merge_validator_preserves_counts_and_parent_paths():
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        book_title="Test",
    )
    source_chapters = [
        {
            "title": "Part A",
            "start_page": 1,
            "end_page": 10,
            "children": [
                {
                    "title": "Introduction",
                    "start_page": 1,
                    "end_page": 5,
                },
                {
                    "title": "Section A",
                    "start_page": 6,
                    "end_page": 10,
                },
            ],
        },
        {
            "title": "Part B",
            "start_page": 11,
            "end_page": 20,
            "children": [
                {
                    "title": "Introduction",
                    "start_page": 11,
                    "end_page": 15,
                },
                {
                    "title": "Section B",
                    "start_page": 16,
                    "end_page": 20,
                },
            ],
        },
    ]
    source = [{"chapters": source_chapters}]
    missing_duplicate = {
        "chapters": [
            source_chapters[0],
            {
                **source_chapters[1],
                "children": [source_chapters[1]["children"][1]],
            },
        ]
    }
    swapped = {
        "chapters": [
            {
                **source_chapters[0],
                "children": [
                    source_chapters[0]["children"][0],
                    source_chapters[1]["children"][1],
                ],
            },
            {
                **source_chapters[1],
                "children": [
                    source_chapters[1]["children"][0],
                    source_chapters[0]["children"][1],
                ],
            },
        ]
    }

    assert any(
        "omitted" in issue
        for issue in call.get_merge_validation_issues(
            missing_duplicate,
            source,
        )
    )
    assert any(
        "unsupported parent path" in issue
        for issue in call.get_merge_validation_issues(swapped, source)
    )

    split_source = [
        {"chapters": [source_chapters[0]]},
        {"chapters": [source_chapters[1]]},
    ]
    omitted_split_intro = {
        "chapters": [
            source_chapters[0],
            {
                **source_chapters[1],
                "children": [source_chapters[1]["children"][1]],
            },
        ]
    }
    evidence_issues = call.get_merge_evidence_issues(
        omitted_split_intro,
        split_source,
        [list(range(1, 11)), list(range(11, 21))],
    )
    assert any("occurrence" in issue for issue in evidence_issues)


def test_direct_merge_occurrences_cannot_collapse_or_mix_bounds():
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        book_title="Test",
    )
    source = [
        {
            "chapters": [
                {
                    "title": "Part",
                    "start_page": 1,
                    "end_page": 100,
                    "children": [
                        {
                            "title": "Introduction",
                            "start_page": 2,
                            "end_page": 3,
                        }
                    ],
                }
            ]
        },
        {
            "chapters": [
                {
                    "title": "Part",
                    "start_page": 51,
                    "end_page": 150,
                    "children": [
                        {
                            "title": "Introduction",
                            "start_page": 120,
                            "end_page": 121,
                        }
                    ],
                }
            ]
        },
    ]
    observed = [
        list(range(1, 101)),
        list(range(51, 151)),
    ]
    collapsed = {
        "chapters": [
            {
                "title": "Part",
                "start_page": 1,
                "end_page": 150,
                "children": [
                    {
                        "title": "Introduction",
                        "start_page": 2,
                        "end_page": 3,
                    }
                ],
            }
        ]
    }
    mixed = {
        "chapters": [
            {
                "title": "Part",
                "start_page": 1,
                "end_page": 150,
                "children": [
                    {
                        "title": "Introduction",
                        "start_page": 2,
                        "end_page": 121,
                    },
                    {
                        "title": "Introduction",
                        "start_page": 120,
                        "end_page": 121,
                    },
                ],
            }
        ]
    }

    assert call.get_merge_evidence_issues(
        collapsed,
        source,
        observed,
    )
    assert call.get_merge_evidence_issues(
        mixed,
        source,
        observed,
    )

    unobserved_bridge_source = [
        source[0],
        {
            "chapters": [
                {
                    "title": "Part",
                    "start_page": 51,
                    "end_page": 150,
                    "children": [
                        {
                            "title": "Introduction",
                            "start_page": 2,
                            "end_page": 121,
                        }
                    ],
                }
            ]
        },
    ]
    bridged = {
        "chapters": [
            {
                "title": "Part",
                "start_page": 1,
                "end_page": 150,
                "children": [
                    {
                        "title": "Introduction",
                        "start_page": 2,
                        "end_page": 121,
                    }
                ],
            }
        ]
    }
    assert call.get_merge_evidence_issues(
        bridged,
        unobserved_bridge_source,
        observed,
    )

    parent_only_source = [
        {
            "chapters": [
                {
                    "title": "Part",
                    "start_page": 1,
                    "end_page": 100,
                }
            ]
        },
        {
            "chapters": [
                {
                    "title": "Part",
                    "start_page": 51,
                    "end_page": 150,
                }
            ]
        },
    ]
    assert call.get_merge_evidence_issues(
        {
            "chapters": [
                {
                    "title": "Part",
                    "start_page": 1,
                    "end_page": 150,
                }
            ]
        },
        parent_only_source,
        observed,
    ) == []


def test_direct_single_batch_requires_every_claim_to_be_observed():
    call = DirectAnalysisCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        book_title="Test",
    )
    result = {
        "chapters": [
            {
                "title": "Invented",
                "start_page": 900,
                "end_page": 999,
            }
        ]
    }

    with pytest.raises(MergeValidationError, match="Single-batch"):
        call.merge_results(
            [result],
            batch_pages=[[1, 2, 3]],
        )


def test_structure_validation_rejects_bool_pages_and_child_escape():
    bool_issues = adaptive_pdf_call.validate_chapter_structure(
        [
            {
                "title": "Chapter",
                "start_page": True,
                "end_page": 2,
            }
        ]
    )
    assert any("Invalid start_page" in issue for issue in bool_issues)

    containment_issues = adaptive_pdf_call.validate_chapter_structure(
        [
            {
                "title": "Parent",
                "start_page": 10,
                "end_page": 20,
                "children": [
                    {
                        "title": "Escaped child",
                        "start_page": 1,
                        "end_page": 2,
                    }
                ],
            }
        ]
    )
    assert any(
        "escapes parent" in issue
        for issue in containment_issues
    )


def test_merge_retry_receives_validator_feedback(monkeypatch):
    class CaptureTransport:
        def __init__(self):
            self.prompts = []

        def generate_text(self, **kwargs):
            self.prompts.append(kwargs["prompt"])
            return '{"chapters": []}'

    class RetryingMergeCall(AdaptivePdfCall):
        merge_max_retries = 2

        def build_merge_prompt(self, _results):
            return "BASE MERGE PROMPT"

        def get_merge_validation_issues(self, _merged, _original_results):
            checks = getattr(self, "_checks", 0)
            self._checks = checks + 1
            if checks < 2:
                return ["Overlap: chapter A ends at p10 but chapter B starts at p9"]
            return []

    transport = CaptureTransport()
    call = RetryingMergeCall(
        client=SimpleNamespace(
            get_default_config=lambda **_kwargs: LLMGenerateConfig(),
        ),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
        pdf_transport=transport,
    )
    monkeypatch.setattr(call, "_get_agent_model", lambda: object())
    monkeypatch.setattr(
        adaptive_pdf_call,
        "run_agent_loop_sync",
        lambda **kwargs: kwargs["generate_fn"](),
    )

    assert call.merge_results([{"chapters": []}, {"chapters": []}]) == {"chapters": []}
    assert len(transport.prompts) == 3
    assert transport.prompts[0] == "BASE MERGE PROMPT"
    assert transport.prompts[1].startswith("BASE MERGE PROMPT")
    assert transport.prompts[2].startswith("BASE MERGE PROMPT")
    assert "Validator findings:" in transport.prompts[1]
    assert "Overlap: chapter A ends at p10" in transport.prompts[1]
    assert transport.prompts[1].count("YOUR PREVIOUS MERGE FAILED") == 1
    assert transport.prompts[2].count("YOUR PREVIOUS MERGE FAILED") == 1


def test_run_forwards_successful_adaptive_batch_pages_to_merge(monkeypatch, tmp_path):
    class CaptureMergeCall(AdaptivePdfCall):
        def merge_results(self, results, artifacts_dir=None, batch_pages=None):
            return {"results": results, "batch_pages": batch_pages}

    expected_results = [{"chapters": []}, {"chapters": []}]
    expected_pages = [[1, 2], [3, 4]]
    monkeypatch.setattr(
        adaptive_pdf_call,
        "run_adaptive_batches",
        lambda *_args, **_kwargs: (expected_results, expected_pages),
    )

    call = CaptureMergeCall(
        client=object(),
        model="test",
        prepare_pdf=lambda *_args, **_kwargs: b"%PDF-test",
        learner=PdfPageLimitLearner(),
    )
    merged = call.run(tmp_path / "unused.pdf", [1, 2, 3, 4])

    assert merged == {"results": expected_results, "batch_pages": expected_pages}


def test_boundary_verification_passes_parent_and_siblings_as_insert_guards(monkeypatch, tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for page in range(1, 5):
        (pages_dir / f"page_{page:03d}.md").write_text("enough content for tokens", encoding="utf-8")

    parent = TOCNode(
        title="Parent Heading",
        level=1,
        start_page=1,
        end_page=2,
        children=[TOCNode(title="Child", level=2, start_page=1, end_page=2)],
    )
    sibling = TOCNode(title="Sibling Heading", level=1, start_page=3, end_page=4)
    observed = {}

    async def fake_verify(node, *_args, forbidden_insert_titles=None, **_kwargs):
        observed[node.title] = forbidden_insert_titles or []

    monkeypatch.setattr(boundary_agent, "verify_node_boundaries", fake_verify)
    asyncio.run(
        boundary_agent.verify_toc_recursive(
            [parent, sibling],
            pages_dir,
            total_pages=4,
            max_tokens=1,
        )
    )

    assert "Parent Heading" in observed["Parent Heading"]
    assert "Sibling Heading" in observed["Parent Heading"]
    assert boundary_agent._titles_equivalent(
        "4. A Generic Section",
        "4 A Generic Section",
    )
    assert not boundary_agent._titles_equivalent("1. Theory", "2. Theory")
    assert boundary_agent._title_supported_as_standalone_line(
        "A Generic Section",
        "Body paragraph\nA Generic Section\nMore body",
    )
    assert boundary_agent._title_supported_as_standalone_line(
        "A Generic Section",
        "## A Generic Section\nMore body",
    )
    assert not boundary_agent._title_supported_as_standalone_line(
        "A Generic Section",
        "This paragraph discusses A Generic Section in prose.",
    )
