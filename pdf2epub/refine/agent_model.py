"""Provider-aware model construction for refine agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit


def _is_official_deepseek_endpoint(base_url: Any) -> bool:
    if not isinstance(base_url, str):
        return False
    return urlsplit(base_url.strip()).hostname == "api.deepseek.com"


def build_openai_agent_model(
    model_name: str,
    provider_config: Mapping[str, Any],
):
    """Build an OpenAI-compatible agent model with endpoint capabilities."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if provider_config.get("type") == "codex":
        from pdf2epub.core.whole.model_factory import _load_codex_openai_provider

        provider_config = _load_codex_openai_provider(dict(provider_config))

    base_url = provider_config.get("base_url")
    provider = OpenAIProvider(
        api_key=provider_config.get("api_key"),
        base_url=base_url,
    )

    if _is_official_deepseek_endpoint(base_url):
        from pydantic_ai.providers.deepseek import DeepSeekProvider

        return OpenAIChatModel(
            model_name,
            provider=provider,
            profile=DeepSeekProvider.model_profile(model_name),
        )

    return OpenAIChatModel(model_name, provider=provider)
