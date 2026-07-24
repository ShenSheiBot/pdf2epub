"""Factories for pydantic-ai models used by whole-mode agents."""

import tomllib
from pathlib import Path

from pydantic_ai.models import Model


def create_anthropic_model(
    model_name: str, *, api_key: str, base_url: str | None = None
) -> Model:
    """Create an Anthropic model with prompt caching enabled."""
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .cached_anthropic_model import CachedAnthropicModel

    provider = AnthropicProvider(api_key=api_key, base_url=base_url)
    return CachedAnthropicModel(model_name, provider=provider)


def create_configured_model(
    model_name: str,
    *,
    provider_name: str,
    provider_config: dict,
) -> Model:
    """Create a pydantic-ai model from a repository provider configuration."""
    provider_type = provider_config.get("type")
    if not provider_type:
        lowered = provider_name.lower()
        if "anthropic" in lowered or "claude" in lowered:
            provider_type = "anthropic"
        elif "gemini" in lowered or "google" in lowered or "vertex" in lowered:
            provider_type = "google"
        else:
            provider_type = "openai"

    if provider_type == "codex":
        provider_config = _load_codex_openai_provider(provider_config)
        provider_type = "openai"

    api_key = provider_config.get("api_key")
    if not api_key:
        raise ValueError(f"Provider {provider_name!r} has no api_key")
    base_url = provider_config.get("base_url")

    if provider_type == "anthropic":
        return create_anthropic_model(
            model_name,
            api_key=api_key,
            base_url=base_url,
        )

    if provider_type == "google":
        from google.genai import Client
        from google.genai.types import HttpOptions
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        http_options = HttpOptions(base_url=base_url) if base_url else None
        client = Client(api_key=api_key, http_options=http_options)
        return GoogleModel(model_name, provider=GoogleProvider(client=client))

    if provider_type == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        return OpenAIChatModel(model_name, provider=provider)

    raise ValueError(
        f"Unsupported whole-mode provider type {provider_type!r} for {provider_name!r}"
    )


def _load_codex_openai_provider(provider_config: dict) -> dict:
    """Resolve the active OpenAI-compatible provider from local Codex settings."""
    config_path = Path(
        provider_config.get("config_path", "~/.codex/config.toml")
    ).expanduser()
    try:
        codex_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Cannot read Codex provider settings: {config_path}") from exc

    selected = (
        provider_config.get("model_provider")
        or codex_config.get("model_provider")
    )
    providers = codex_config.get("model_providers") or {}
    resolved = providers.get(selected) if selected else None
    if not isinstance(resolved, dict):
        raise TypeError(
            f"Codex model provider {selected!r} is not configured in {config_path}"
        )

    api_key = resolved.get("experimental_bearer_token")
    base_url = resolved.get("base_url")
    if not api_key or not base_url:
        raise ValueError(
            "The selected Codex model provider needs both "
            "experimental_bearer_token and base_url"
        )
    return {
        "api_key": api_key,
        "base_url": base_url,
    }
