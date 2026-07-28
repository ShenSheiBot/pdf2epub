import google.genai
import pytest

from pdf2epub import breakdown
from pdf2epub.utils import llm_client as llm_client_module
from pdf2epub.utils import network_utils


ADC_VERTEX_PROVIDER = {
    "type": "google",
    "vertexai": True,
    "project": "test-project",
    "location": "global",
}


def test_gemini_client_uses_adc_vertex_project_and_location_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        google.genai,
        "Client",
        lambda **kwargs: observed.append(kwargs) or object(),
    )

    network_utils.GeminiClient(
        api_key=None,
        vertexai=True,
        project="test-project",
        location="global",
    )

    assert observed == [
        {
            "vertexai": True,
            "project": "test-project",
            "location": "global",
        }
    ]


def test_llm_client_passes_adc_vertex_provider_fields_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        llm_client_module,
        "GeminiClient",
        lambda **kwargs: observed.append(kwargs) or object(),
    )
    client = llm_client_module.LLMClient(
        {
            "credentials": {
                "providers": {"vertex-online": ADC_VERTEX_PROVIDER}
            }
        }
    )

    assert client._get_client("vertex-online") is not None
    assert observed == [
        {
            "api_key": None,
            "base_url": None,
            "vertexai": True,
            "project": "test-project",
            "location": "global",
            "extra_headers": None,
            "num_retries": 3,
            "max_backoff_seconds": 30,
        }
    ]


def test_create_gemini_client_from_config_passes_adc_vertex_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        network_utils,
        "GeminiClient",
        lambda **kwargs: observed.append(kwargs) or object(),
    )

    client = network_utils.create_gemini_client_from_config(
        {
            "credentials": {
                "providers": {"vertex-online": ADC_VERTEX_PROVIDER}
            }
        },
        provider_name="vertex-online",
    )

    assert client is not None
    assert observed == [
        {
            "api_key": None,
            "base_url": None,
            "vertexai": True,
            "project": "test-project",
            "location": "global",
            "extra_headers": None,
            "num_retries": 3,
            "max_backoff_seconds": 30,
        }
    ]


def test_legacy_breakdown_entry_uses_shared_adc_client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        network_utils,
        "GeminiClient",
        lambda **kwargs: observed.append(kwargs) or object(),
    )

    client = breakdown.create_breakdown_client(
        {
            "credentials": {
                "providers": {"vertex-online": ADC_VERTEX_PROVIDER}
            },
            "breakdown": {"provider": "vertex-online"},
        }
    )

    assert client is not None
    assert observed == [
        {
            "api_key": None,
            "base_url": None,
            "vertexai": True,
            "project": "test-project",
            "location": "global",
            "extra_headers": None,
            "num_retries": 3,
            "max_backoff_seconds": 30,
        }
    ]


def test_gemini_vertex_express_mode_keeps_using_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        google.genai,
        "Client",
        lambda **kwargs: observed.append(kwargs) or object(),
    )

    network_utils.GeminiClient(
        api_key="express-key",
        vertexai=True,
    )

    assert observed == [{"vertexai": True, "api_key": "express-key"}]


def test_gemini_proxy_mode_keeps_api_key_and_http_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        google.genai,
        "Client",
        lambda **kwargs: observed.append(kwargs) or object(),
    )

    network_utils.GeminiClient(
        api_key="proxy-key",
        base_url="https://proxy.example",
        vertexai=True,
        project="ignored-project",
        location="ignored-location",
        extra_headers={"X-Test": "yes"},
    )

    assert observed == [
        {
            "api_key": "proxy-key",
            "http_options": {
                "base_url": "https://proxy.example",
                "headers": {"X-Test": "yes"},
                "timeout": 86400000,
            },
        }
    ]
