"""AnthropicModel subclass with intermediate cache breakpoints.

Addresses the 20-block lookback limitation in Anthropic's prompt caching.
See docs/research/prompt-caching-analysis.md for full analysis.

Standard pydantic-ai only places cache_control on the LAST message.
For conversations with 100+ content blocks, the 20-block lookback from the
last breakpoint can't reach the cached prefix from previous calls. This
subclass adds intermediate breakpoints to bridge the gap:

  BP1: tools (handled by pydantic-ai)         — 1h TTL
  BP2: first user message (system prompt)     — 1h TTL (this class)
  BP3: resume boundary (settled history end)  — 1h TTL (this class)
  BP4: last message (handled by pydantic-ai)  — 1h TTL
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.anthropic import (
    AnthropicModel,
    AnthropicModelSettings,
    BetaMessageParam,
    BetaTextBlockParam,
)

logger = logging.getLogger(__name__)

# Context variable for the resume boundary index.
# Set by orchestration.py before run_stream; read by _map_message.
# This is async-safe (ContextVar is per-task in asyncio).
backoffice_resume_boundary: ContextVar[int | None] = ContextVar(
    "backoffice_resume_boundary", default=None
)

# Content block types that support cache_control (from Anthropic docs)
_CACHEABLE_TYPES = frozenset({"text", "tool_use", "server_tool_use", "image", "tool_result", "document"})


@dataclass(init=False)
class CachedAnthropicModel(AnthropicModel):
    """AnthropicModel with intermediate cache breakpoints for long conversations.

    Adds BP2 (system prompt in first user message) and BP3 (resume boundary)
    in addition to pydantic-ai's standard BP1 (tools) and BP4 (last message).
    """

    async def _map_message(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[str | list[BetaTextBlockParam], list[BetaMessageParam]]:
        """Map messages and add intermediate cache breakpoints."""
        system, anthropic_messages = await super()._map_message(
            messages, model_request_parameters, model_settings
        )

        if len(anthropic_messages) < 3:
            return system, anthropic_messages

        cache_1h: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}

        # BP2: system prompt (first user message) — stable across all calls
        _set_cache_on_last_block(anthropic_messages[0], cache_1h)

        # BP3: resume boundary — end of settled history from previous ticks
        resume_idx = backoffice_resume_boundary.get()
        if resume_idx is not None and 0 < resume_idx < len(anthropic_messages) - 1:
            _set_cache_on_last_block(anthropic_messages[resume_idx], cache_1h)

        # BP1 (tools) and BP4 (last message) are already set by super()._map_message
        # _limit_cache_points (called by _messages_create after us) enforces max 4

        return system, anthropic_messages


def _set_cache_on_last_block(msg: BetaMessageParam, cache_control: dict[str, Any]) -> None:
    """Add cache_control to the last cacheable content block of a message."""
    content = msg.get("content", [])
    if not isinstance(content, list):
        return
    for block in reversed(content):
        block_dict = cast(dict[str, Any], block)
        if isinstance(block_dict, dict) and block_dict.get("type") in _CACHEABLE_TYPES:
            block_dict["cache_control"] = cache_control
            return
