"""PDF-capable transports used by the refine structure-analysis calls.

The refine pipeline deliberately keeps PDF transport separate from the
provider-agnostic text client.  Gemini accepts PDF ``Part`` objects, whereas
the OpenAI Responses API accepts a PDF as an ``input_file`` data URL.  Both
transports implement the same small interface so batching, JSON repair, and
TOC merging remain transport-independent.

``openai_responses`` is an explicit opt-in transport.  It is not a fallback
for Gemini and it never changes the selected model after a failed request.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping, Optional, Protocol

from google.genai.types import Content, Part
from loguru import logger
from openai import APIStatusError, OpenAI

from ..utils.llm_client import BoundLLMClient, LLMGenerateConfig


class PdfTransportError(RuntimeError):
    """A request made through a PDF transport failed."""


class PdfContextLimitError(PdfTransportError):
    """The selected model rejected a PDF request because it exceeds context."""


class PdfPayloadTooLargeError(PdfTransportError):
    """The transport rejected the encoded PDF request before model execution."""


class PdfTransport(Protocol):
    """Minimal interface required by :class:`AdaptivePdfCall`."""

    def generate_pdf(
        self,
        *,
        model: str,
        prompt: str,
        pdf_data: bytes,
        config: LLMGenerateConfig,
        operation_name: str,
        prefix: Optional[str] = None,
    ) -> str:
        """Generate text using a prompt and an in-request PDF."""

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        config: LLMGenerateConfig,
        operation_name: str,
        prefix: Optional[str] = None,
    ) -> str:
        """Generate text without a PDF (used for batch-result merging)."""


class GeminiPdfTransport:
    """Existing Gemini ``Part`` transport, retained as the default path."""

    def __init__(self, client: BoundLLMClient):
        self._client = client

    def cache_identity(self) -> Mapping[str, Any]:
        """Return non-secret provider identity for batch-cache binding."""
        provider = getattr(self._client, "provider", None)
        llm_client = getattr(self._client, "llm_client", None)
        runtime_config = getattr(llm_client, "config", {})
        provider_config = (
            runtime_config.get("credentials", {})
            .get("providers", {})
            .get(provider, {})
            if isinstance(runtime_config, Mapping)
            else {}
        )
        api_key = provider_config.get("api_key") or ""
        return {
            "type": "gemini",
            "provider": provider,
            "base_url": provider_config.get("base_url"),
            "vertexai": provider_config.get("vertexai", False),
            "credential_sha256": (
                hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()
                if api_key
                else None
            ),
        }

    @staticmethod
    def _continuation_contents(
        prompt: str,
        prefix: str,
        pdf_data: bytes | None = None,
    ) -> list[Content]:
        original_parts = [Part(text=prompt)]
        if pdf_data is not None:
            original_parts.append(
                Part.from_bytes(data=pdf_data, mime_type="application/pdf")
            )
        return [
            Content(role="user", parts=original_parts),
            Content(role="model", parts=[Part(text=prefix)]),
            Content(
                role="user",
                parts=[Part(text="Continue from where you left off. Output only the remaining JSON content, no preamble.")],
            ),
        ]

    def generate_pdf(
        self,
        *,
        model: str,
        prompt: str,
        pdf_data: bytes,
        config: LLMGenerateConfig,
        operation_name: str,
        prefix: Optional[str] = None,
    ) -> str:
        contents: Any
        if prefix is None:
            contents = [
                prompt,
                Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
            ]
        else:
            contents = self._continuation_contents(prompt, prefix, pdf_data)
        return self._client.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
            operation_name=operation_name,
        )

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        config: LLMGenerateConfig,
        operation_name: str,
        prefix: Optional[str] = None,
    ) -> str:
        contents: Any = [prompt] if prefix is None else self._continuation_contents(prompt, prefix)
        return self._client.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
            operation_name=operation_name,
        )


class OpenAIResponsesPdfTransport:
    """OpenAI Responses transport using in-request base64 PDF data.

    The adapter is intentionally stateless: each continuation includes the
    repaired prefix that the JSON agent produced.  This makes the continuation
    independent of server-side response retention and works with compatible
    proxy endpoints that do not implement ``previous_response_id``.
    """

    _CONTINUATION_INSTRUCTION = (
        "The text below is the already validated prefix of your prior JSON output. "
        "Continue exactly after it. Output only the remaining JSON content, with no preamble.\n\n"
    )

    def __init__(
        self,
        provider_config: Mapping[str, Any],
        *,
        timeout_seconds: float = 300,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        client: Any = None,
    ):
        api_key = provider_config.get("api_key")
        base_url = provider_config.get("base_url")
        if not api_key:
            raise ValueError(
                "OpenAI Responses PDF transport requires "
                "credentials.providers.<name>.api_key"
            )
        if not base_url:
            raise ValueError(
                "OpenAI Responses PDF transport requires "
                "credentials.providers.<name>.base_url"
            )

        self.base_url = self._normalise_base_url(str(base_url))
        self._cache_identity = {
            "type": "openai_responses",
            "base_url": self.base_url,
            "credential_sha256": hashlib.sha256(
                str(api_key).encode("utf-8")
            ).hexdigest(),
        }
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    def cache_identity(self) -> Mapping[str, Any]:
        """Return endpoint identity without exposing the configured key."""
        return dict(self._cache_identity)

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        """Return an SDK base URL ending in ``/v1``.

        Accepting both a proxy root and a full Responses endpoint makes the
        adapter usable with the local proxy probe without leaking endpoint
        details into the rest of refine.
        """
        normalised = base_url.rstrip("/")
        if normalised.endswith("/responses"):
            normalised = normalised.removesuffix("/responses")
        if not normalised.endswith("/v1"):
            normalised = f"{normalised}/v1"
        return normalised

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Extract all textual message content from a Responses API result."""
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = getattr(response, "output", None)
        if output is None and isinstance(response, Mapping):
            output = response.get("output")

        fragments: list[str] = []
        for item in output or []:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, Mapping):
                item_type = item.get("type")
            if item_type != "message":
                continue

            content = getattr(item, "content", None)
            if content is None and isinstance(item, Mapping):
                content = item.get("content")
            for part in content or []:
                part_type = getattr(part, "type", None)
                part_text = getattr(part, "text", None)
                if isinstance(part, Mapping):
                    part_type = part_type or part.get("type")
                    part_text = part_text or part.get("text")
                if part_type == "output_text" and isinstance(part_text, str):
                    fragments.append(part_text)

        result = "".join(fragments).strip()
        if not result:
            raise PdfTransportError(
                "OpenAI Responses returned no output_text message content."
            )
        return result

    @staticmethod
    def _is_context_limit_error(error: APIStatusError) -> bool:
        message = str(error).lower()
        return error.status_code == 400 and any(
            marker in message
            for marker in ("context_length_exceeded", "context length", "maximum context")
        )

    def _create_response(
        self,
        *,
        model: str,
        input_items: list[dict[str, Any]],
        config: LLMGenerateConfig,
        operation_name: str,
    ) -> str:
        # Keep the default request shape deliberately minimal. It matches the
        # locally verified proxy request and leaves truncation disabled by the
        # Responses API default, so over-context inputs fail closed. Optional
        # sampling/output controls are explicit transport configuration rather
        # than inherited accidentally from the Gemini-oriented client config.
        request: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": False,
            # Responses are reconstructed from the validated prefix instead of
            # server-side response IDs, so retaining application state is
            # unnecessary.
            "store": False,
        }
        if self._max_output_tokens is not None:
            request["max_output_tokens"] = self._max_output_tokens
        if self._temperature is not None:
            request["temperature"] = self._temperature
        try:
            response = self._client.responses.create(**request)
        except APIStatusError as error:
            if self._is_context_limit_error(error):
                raise PdfContextLimitError(
                    f"{operation_name}: OpenAI Responses rejected the request because "
                    "it exceeds the selected model's context window. No automatic "
                    "batch reduction or model fallback was attempted."
                ) from error
            if error.status_code == 413:
                raise PdfPayloadTooLargeError(
                    f"{operation_name}: OpenAI Responses rejected the encoded "
                    "PDF payload as too large. Adaptive PDF batching may retry "
                    "with fewer pages; the selected model will not change."
                ) from error
            raise PdfTransportError(
                f"{operation_name}: OpenAI Responses request failed "
                f"(HTTP {error.status_code}): {error}"
            ) from error
        return self._extract_output_text(response)

    @classmethod
    def _user_item(
        cls,
        prompt: str,
        *,
        pdf_data: bytes | None = None,
        prefix: Optional[str] = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if pdf_data is not None:
            encoded_pdf = base64.b64encode(pdf_data).decode("ascii")
            content.append(
                {
                    "type": "input_file",
                    "filename": "refine-batch.pdf",
                    "file_data": f"data:application/pdf;base64,{encoded_pdf}",
                }
            )
        if prefix is not None:
            content.append(
                {
                    "type": "input_text",
                    "text": cls._CONTINUATION_INSTRUCTION + prefix,
                }
            )
        return {"role": "user", "content": content}

    def generate_pdf(
        self,
        *,
        model: str,
        prompt: str,
        pdf_data: bytes,
        config: LLMGenerateConfig,
        operation_name: str,
        prefix: Optional[str] = None,
    ) -> str:
        return self._create_response(
            model=model,
            input_items=[self._user_item(prompt, pdf_data=pdf_data, prefix=prefix)],
            config=config,
            operation_name=operation_name,
        )

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        config: LLMGenerateConfig,
        operation_name: str,
        prefix: Optional[str] = None,
    ) -> str:
        return self._create_response(
            model=model,
            input_items=[self._user_item(prompt, prefix=prefix)],
            config=config,
            operation_name=operation_name,
        )


def create_pdf_transport(
    *,
    config: Mapping[str, Any],
    structure_provider: str,
    structure_client: BoundLLMClient,
    transport_config: Any = None,
) -> PdfTransport:
    """Construct the explicitly configured PDF transport for refine.

    Existing Google providers retain their Gemini ``Part`` path when no
    transport is configured.  Non-Google providers must name a transport so a
    model change cannot silently alter PDF request semantics.
    """
    if transport_config is None:
        transport_type = None
        options: Mapping[str, Any] = {}
    elif isinstance(transport_config, str):
        transport_type = transport_config
        options = {}
    elif isinstance(transport_config, Mapping):
        transport_type = transport_config.get("type")
        options = transport_config
    else:
        raise ValueError(
            "refine.structure.pdf_transport must be a transport name or mapping"
        )

    providers = config.get("credentials", {}).get("providers", {})
    provider_config = providers.get(structure_provider, {})
    provider_type = provider_config.get("type")
    if not provider_type:
        provider_name = structure_provider.casefold()
        if "gemini" in provider_name or "vertex" in provider_name:
            provider_type = "google"
        elif "anthropic" in provider_name or "claude" in provider_name:
            provider_type = "anthropic"
        else:
            provider_type = "openai"

    if transport_type in (None, "gemini"):
        if provider_type != "google":
            raise ValueError(
                f"refine.structure.provider '{structure_provider}' is type "
                f"'{provider_type}' and cannot use Gemini PDF parts. "
                "For Responses API use {type: openai_responses}."
            )
        return GeminiPdfTransport(structure_client)

    if transport_type == "openai_responses":
        if provider_type != "openai":
            raise ValueError(
                "refine.structure.pdf_transport.type=openai_responses requires an "
                "OpenAI-compatible structure provider (type: openai)."
            )
        timeout_seconds = float(options.get("timeout_seconds", 300))
        max_output_tokens = options.get("max_output_tokens")
        if max_output_tokens is not None:
            max_output_tokens = int(max_output_tokens)
        temperature = options.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
        logger.info(
            "[refine] Using explicitly configured OpenAI Responses PDF transport "
            f"for provider '{structure_provider}'"
        )
        return OpenAIResponsesPdfTransport(
            provider_config,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported refine PDF transport '{transport_type}'. "
        "Supported values: gemini, openai_responses."
    )
