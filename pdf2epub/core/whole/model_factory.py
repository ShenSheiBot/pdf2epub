"""
Factory for creating pydantic-ai Model instances.

Automatically uses CachedAnthropicModel for Anthropic providers.
"""

from pydantic_ai.models import Model


def create_anthropic_model(model_name: str, *, api_key: str, base_url: str | None = None) -> Model:
    """Create an Anthropic model with prompt caching enabled."""
    from .cached_anthropic_model import CachedAnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key=api_key, base_url=base_url)
    return CachedAnthropicModel(model_name, provider=provider)
