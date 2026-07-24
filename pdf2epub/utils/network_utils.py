"""
Refactored network utilities using tenacity for cleaner retry logic.
"""

import base64
import fcntl
import httpx
import json as _json
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any, Union, List
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    ThinkingConfig
)
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception, RetryCallState
from tenacity.stop import stop_base
from tenacity.wait import wait_base
import tiktoken


# ─── Unified LLM Trace ───

_trace_path: Optional[Path] = None


def set_llm_trace_path(path: Optional[Path]):
    """Set the global LLM trace output path. Call once at startup."""
    global _trace_path
    _trace_path = path


def _write_trace(entry: dict):
    """Append a trace entry with file locking. Never raises."""
    if _trace_path is None:
        return
    try:
        _trace_path.parent.mkdir(parents=True, exist_ok=True)
        line = _json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with open(_trace_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
    except Exception:
        pass  # Never break pipeline for logging


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    """Convert provider usage objects into plain dictionaries."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "__dict__"):
        return dict(vars(usage))
    if isinstance(usage, dict):
        return usage
    return {}


def _merge_usage_dict(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge streamed usage chunks, keeping cumulative numeric maxima."""
    for key, value in incoming.items():
        if value is None:
            continue
        if isinstance(value, (int, float)) and isinstance(base.get(key), (int, float)):
            base[key] = max(base[key], value)
        else:
            base[key] = value
    return base


def extract_anthropic_stream_usage(events: List[Any]) -> tuple[Dict[str, Any], str]:
    """Extract cumulative usage from Anthropic stream events.

    Anthropic reports input tokens on message_start and output tokens on
    message_delta. Taking the first usage object records output_tokens=0 for
    successful long streams, so we merge all usage-bearing events instead.
    """
    usage: Dict[str, Any] = {}
    saw_usage = False

    for event in events:
        candidates = [
            getattr(event, "usage", None),
            getattr(getattr(event, "message", None), "usage", None),
        ]
        for candidate in candidates:
            usage_dict = _usage_to_dict(candidate)
            if not usage_dict:
                continue
            saw_usage = True
            _merge_usage_dict(usage, usage_dict)

    return usage, "anthropic_stream" if saw_usage else "missing"


def usage_status(usage: Dict[str, Any], response_text: str) -> str:
    """Classify whether trace usage is trustworthy enough for reporting."""
    if not usage:
        return "missing"
    if response_text and not usage.get("output_tokens"):
        return "incomplete"
    return "ok"


def _create_method_before_sleep(operation_name_param: str = "operation_name") -> callable:
    """
    Create a before_sleep callback for class methods that logs exception details.

    This extracts operation_name from method kwargs and includes exception info.
    """
    def before_sleep(retry_state: RetryCallState) -> None:
        # Get operation_name from kwargs
        operation_name = retry_state.kwargs.get(operation_name_param, "API call")
        exc = retry_state.outcome.exception() if retry_state.outcome else None

        if exc:
            exc_msg = str(exc).replace('\n', ' ')[:500]
        else:
            exc_msg = "unknown error"

        logger.warning(
            f"Retry {retry_state.attempt_number} for {operation_name}: "
            f"waiting {retry_state.next_action.sleep:.1f}s | {exc_msg}"
        )
    return before_sleep


class stop_after_self_retries(stop_base):
    """Stop after `self.num_retries` attempts (defaults to 3 if missing)."""
    def __init__(self, attr: str = "num_retries", default: int = 3):
        self.attr = attr
        self.default = default

    def __call__(self, retry_state):
        # For bound instance methods, args[0] is `self`
        obj = retry_state.args[0] if retry_state.args else None
        num = getattr(obj, self.attr, self.default)
        return retry_state.attempt_number >= int(num)


class wait_exponential_with_self_max(wait_base):
    """Use Tenacity's wait_random_exponential with self.max_backoff_seconds."""
    def __init__(self, multiplier: float = 1, attr: str = "max_backoff_seconds", default: int = 30):
        self.multiplier = multiplier
        self.attr = attr
        self.default = default

    def __call__(self, retry_state):
        # For bound instance methods, args[0] is `self`
        obj = retry_state.args[0] if retry_state.args else None
        max_seconds = getattr(obj, self.attr, self.default)
        
        # Create and use Tenacity's wait_random_exponential
        wait_strategy = wait_random_exponential(multiplier=self.multiplier, max=max_seconds)
        return wait_strategy(retry_state)


# Thread-local context for retry behavior control.
# When skip_503=True, 503 errors are NOT retried by tenacity —
# they bubble up to run_adaptive_batches() which handles them by splitting.
_retry_context = threading.local()


class StreamingHallucinationError(RuntimeError):
    """A retryable repetition loop detected before a stream completed."""


# Define transient errors that should trigger retries
def is_transient_gemini_error(exception: Exception) -> bool:
    """Check if a Gemini API error is transient and should be retried."""
    if isinstance(exception, StreamingHallucinationError):
        return True
    # Don't retry content safety blocks - these should fail fast
    error_str = str(exception).lower()
    if any(term in error_str for term in ['prohibited', 'safety', 'blocked', 'harmful']):
        return False

    # Fail fast on Cloudflare 524 proxy timeouts — retrying won't help,
    # the proxy cannot handle large PDF requests
    if '524' in error_str and 'timeout' in error_str:
        return False

    # When inside adaptive batching, don't retry 503 at tenacity level —
    # let run_adaptive_batches() handle it by splitting into smaller batches
    if getattr(_retry_context, 'skip_503', False):
        if '503' in error_str or 'unavailable' in error_str:
            return False

    # Retry network and rate limit errors
    if isinstance(exception, (httpx.TimeoutException, httpx.ConnectError, ConnectionError)):
        return True
    
    # Check for specific error codes
    transient_keywords = [
        'rate_limit', '429', 'quota',
        'timeout', 'unavailable', '503',
        'internal', '500', '502', '504',
        'resource_exhausted', 'overloaded',
        'disconnected', 'connection',
        'cancelled', 'canceled', '499',
    ]

    return any(keyword in error_str for keyword in transient_keywords)


def is_transient_anthropic_error(exception: Exception) -> bool:
    """Check if an Anthropic API error is transient and should be retried."""
    error_str = str(exception).lower()

    # Don't retry content blocks
    if any(term in error_str for term in ['content_policy', 'unsafe', 'violation']):
        return False

    # Retry rate limits and server errors
    if '429' in error_str or 'rate' in error_str:
        return True

    if any(code in error_str for code in ['500', '502', '503', '504']):
        return True

    if isinstance(exception, (TimeoutError, ConnectionError, httpx.TimeoutException)):
        return True

    # Network disruption keywords
    if any(kw in error_str for kw in ['disconnect', 'broken pipe', 'reset by peer',
                                       'connection', 'timeout', 'remote protocol']):
        return True

    return False


def is_transient_openai_error(exception: Exception) -> bool:
    """Check if an OpenAI API error is transient and should be retried."""
    error_str = str(exception).lower()

    # Don't retry content policy violations
    if any(term in error_str for term in ['content_policy', 'violation', 'refused']):
        return False

    # Retry rate limits and server errors
    if '429' in error_str or 'rate' in error_str:
        return True

    if any(code in error_str for code in ['500', '502', '503', '504']):
        return True

    if isinstance(exception, (TimeoutError, ConnectionError, httpx.TimeoutException)):
        return True

    # Network disruption keywords
    if any(kw in error_str for kw in ['disconnect', 'broken pipe', 'reset by peer',
                                       'connection', 'timeout', 'remote protocol']):
        return True

    return False


# --- Streaming hallucination detection ---
# Detect repetition loops and abnormally long lines during streaming.



def _detect_streaming_hallucination(text: str) -> Optional[str]:
    """Check if streaming output shows hallucination patterns.

    Returns a reason string if hallucination detected, None otherwise.
    Only detects repetition loops — non-repeating long output is normal.

    Uses multi-period detection: checks if the tail of the text contains
    any repeating pattern with period 1-100 chars, repeated 3+ times.
    This catches both single-char loops (啊啊啊...) and multi-token
    cycles (啊呀啊呀...) regardless of alignment.
    """
    if len(text) < 300:
        return None

    tail = text[-200:]
    for period in range(1, 101):
        if len(tail) < period * 3:
            break
        pat = tail[-period:]
        if not pat.strip():
            continue
        if tail[-period * 3:-period * 2] == pat and tail[-period * 2:-period] == pat:
            return f"repetition loop detected (period={period}, pattern={pat[:30]!r})"

    return None


class GeminiClient:
    """Wrapper for Gemini API with smart retry logic."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        vertexai: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        num_retries: int = 3,
        max_backoff_seconds: int = 30
    ):
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key
            base_url: Custom API endpoint (e.g., 'google.shenshei.fans')
            vertexai: Use Vertex AI express mode (official, not via proxy)
            extra_headers: Extra HTTP headers (e.g., {'X-Use-Vertex': 'true'} for proxy)
            num_retries: Number of retries for transient errors
            max_backoff_seconds: Maximum backoff time between retries
        """
        from google import genai

        if vertexai and not base_url:
            # Official Vertex AI express mode
            self.client = genai.Client(vertexai=True, api_key=api_key)
        else:
            # Build http_options for proxy or default
            http_options = {}
            if base_url:
                http_options['base_url'] = base_url
            if extra_headers:
                http_options['headers'] = extra_headers

            # Set longer timeout for large PDF uploads (especially for slow networks)
            # Default Google SDK timeout is too short for 30MB+ PDFs on slow connections
            # NOTE: Google SDK expects timeout in MILLISECONDS, not seconds
            http_options['timeout'] = 86400000  # 24 hours — SDK-level timeout should not be the bottleneck

            if http_options:
                self.client = genai.Client(api_key=api_key, http_options=http_options)
            else:
                self.client = genai.Client(api_key=api_key)

        self.num_retries = num_retries
        self.max_backoff_seconds = max_backoff_seconds
        self._last_usage_metadata = None
        self._last_stream_events: Dict[str, Any] = {}
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @retry(
        retry=retry_if_exception(is_transient_gemini_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
        reraise=True,
        before_sleep=_create_method_before_sleep("operation_name")
    )
    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[GenerateContentConfig] = None,
        operation_name: str = "Gemini API call"
    ) -> Any:
        """Generate content with automatic retry for transient errors."""
        logger.info(f"Calling Gemini API for {operation_name}")

        if config is None:
            config = self.get_default_config()

        _trace_start = _time.monotonic()
        _trace_error = None
        response_text = ""
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            if not response or not response.text:
                raise ValueError(f"Empty response from Gemini for {operation_name}")

            response_text = response.text or ""
            return response
        except Exception as e:
            _trace_error = str(e)
            raise
        finally:
            _write_trace({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation_name,
                "provider": "gemini",
                "model": model,
                "duration_ms": int((_time.monotonic() - _trace_start) * 1000),
                "input_tokens": 0,
                "output_tokens": 0,
                "raw_request": {"contents": contents, "config": str(config)},
                "raw_response": response_text,
                "error": _trace_error,
                "partial_response_length": len(response_text) if _trace_error else 0,
            })

    @retry(
        retry=retry_if_exception(is_transient_gemini_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
        reraise=True,
        before_sleep=_create_method_before_sleep("operation_name")
    )
    def generate_content_stream(
        self,
        model: str,
        contents: Any,
        config: Optional[GenerateContentConfig] = None,
        operation_name: str = "Gemini API call"
    ) -> str:
        """Generate content with streaming and automatic retry."""
        logger.info(f"Streaming from Gemini API for {operation_name}")

        if config is None:
            config = self.get_default_config()

        _trace_start = _time.monotonic()
        _trace_error = None

        # Stream and aggregate response
        stream_response = self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config
        )

        aggregated_text = ""
        chunk_count = 0
        last_log_length = 0
        halted_early = False
        hallucination_reason = None
        finish_reason = None
        _usage_dict = {}

        try:
            for chunk in stream_response:
                chunk_count += 1

                # Handle Gemini 3 response format with potential thought parts
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        # Check for early termination
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                            reason = str(candidate.finish_reason)
                            if any(term in reason for term in ['PROHIBITED', 'SAFETY', 'BLOCKED']):
                                raise ValueError(f"Content blocked: {reason}")

                        # Extract text from content parts, filtering out thoughts
                        if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                # Skip thought parts (Gemini 3 thinking mode)
                                if hasattr(part, 'thought') and part.thought:
                                    continue
                                if hasattr(part, 'text') and part.text:
                                    aggregated_text += part.text
                elif hasattr(chunk, 'text') and chunk.text:
                    # Fallback for simpler response format
                    aggregated_text += chunk.text

                # --- Hallucination guard: detect repetition loops ---
                # Check every 20 chunks (~600 chars) for early detection
                if chunk_count % 20 == 0:
                    hallucination_reason = _detect_streaming_hallucination(
                        aggregated_text
                    )
                    if hallucination_reason:
                        logger.warning(
                            f"Hallucination detected for {operation_name}: "
                            f"{hallucination_reason}. "
                            f"Halting stream at {len(aggregated_text)} chars."
                        )
                        halted_early = True
                        break

                # Log progress periodically - every 500 tokens
                current_tokens = len(self.tokenizer.encode(aggregated_text))
                if current_tokens - last_log_length >= 500:
                    logger.debug(f"Streaming {operation_name}: {current_tokens} tokens")
                    last_log_length = current_tokens

            # Log finish reason from last chunk
            if halted_early:
                finish_reason = "HALLUCINATION_HALT"
                logger.warning(
                    f"Stream halted early (hallucination) for {operation_name}: "
                    "discarding the partial response and retrying"
                )
                raise StreamingHallucinationError(
                    f"Streaming hallucination for {operation_name}: "
                    f"{hallucination_reason}"
                )
            elif chunk_count > 0 and hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        finish_reason = str(candidate.finish_reason)
                        logger.debug(f"Stream finished with reason: {finish_reason}")

            if not aggregated_text:
                logger.error(f"Empty stream: {chunk_count} chunks received for {operation_name}")
                if finish_reason:
                    logger.error(f"Finish reason: {finish_reason}")
                raise ValueError(f"Empty stream response for {operation_name}")

            # Get final token count
            final_tokens = len(self.tokenizer.encode(aggregated_text))

            # Log truncation warning — actual retry is handled by caller
            if finish_reason and 'MAX_TOKENS' in finish_reason.upper():
                logger.warning(
                    f"Response truncated (MAX_TOKENS) for {operation_name}: "
                    f"{final_tokens} tokens generated before cutoff"
                )

            # Capture and store full usage metadata
            usage_meta = None
            if chunk_count > 0 and hasattr(chunk, 'usage_metadata'):
                usage_meta = chunk.usage_metadata
            self._last_usage_metadata = usage_meta
            _usage_dict = usage_meta.model_dump() if usage_meta else {}
            self._last_stream_events = _usage_dict

            logger.info(f"Streamed {final_tokens} tokens ({chunk_count} chunks) for {operation_name}")
            return aggregated_text
        except Exception as e:
            _trace_error = str(e)
            raise
        finally:
            _write_trace({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation_name,
                "provider": "gemini",
                "model": model,
                "duration_ms": int((_time.monotonic() - _trace_start) * 1000),
                "input_tokens": _usage_dict.get("prompt_token_count", 0),
                "output_tokens": _usage_dict.get("candidates_token_count", 0),
                "cache_read_tokens": _usage_dict.get(
                    "cached_content_token_count", 0
                ),
                "thought_tokens": _usage_dict.get("thoughts_token_count", 0),
                "raw_request": {"contents": contents, "config": str(config)},
                "raw_response": aggregated_text,
                "finish_reason": finish_reason,
                "error": _trace_error,
                "partial_response_length": len(aggregated_text) if _trace_error else 0,
            })

    def get_last_usage(self) -> Dict[str, Any]:
        """Return normalized usage for the most recent successful stream."""
        usage = dict(self._last_stream_events or {})
        return {
            "input_tokens": usage.get("prompt_token_count", 0) or 0,
            "output_tokens": usage.get("candidates_token_count", 0) or 0,
            "cache_read_tokens": usage.get("cached_content_token_count", 0) or 0,
            "thought_tokens": usage.get("thoughts_token_count", 0) or 0,
            "total_tokens": usage.get("total_token_count", 0) or 0,
        }

    @staticmethod
    def get_default_config(temperature: float = 0.1) -> GenerateContentConfig:
        """Get default generation config."""
        return GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            candidate_count=1,
            max_output_tokens=65536,
            # Configure thinking for Gemini 3 models (ignored by older models)
            # Note: Gemini 3 requires thinking mode, cannot disable it
            thinking_config=ThinkingConfig(
                thinking_budget=1024,  # Minimal thinking budget
                include_thoughts=False  # Don't include thought summaries in output
            ),
            stop_sequences=None,
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
        )


    def embed_content(self, texts: List[str], model: str = "gemini-embedding-001") -> List[List[float]]:
        """Embed a list of texts using Gemini embedding model.

        Returns list of embedding vectors. Calls are sequential (one per text)
        because the Gemini SDK embed_content doesn't support batch in v1beta.
        """
        results = []
        for text in texts:
            resp = self.client.models.embed_content(model=model, contents=text)
            results.append(resp.embeddings[0].values)
        return results


class AnthropicClient:
    """Wrapper for Anthropic API with smart retry logic."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, num_retries: int = 3, max_backoff_seconds: int = 30):
        """Initialize Anthropic client."""
        import anthropic
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.num_retries = num_retries
        self.max_backoff_seconds = max_backoff_seconds
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @retry(
        retry=retry_if_exception(is_transient_anthropic_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
        reraise=True,
        before_sleep=_create_method_before_sleep("operation_name")
    )
    def generate_content(
        self,
        prompt: Union[str, List[Dict]],
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        operation_name: str = "Anthropic API call",
        json_mode: bool = False,
        enable_cache: bool = False,
    ) -> str:
        """Generate content with automatic retry for transient errors.

        Args:
            prompt: Can be:
                - str: Simple text prompt
                - List[Dict]: Either content blocks or messages with roles
                  - Content blocks: [{"type": "text", "text": "..."}]
                  - Messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        logger.info(f"Calling Anthropic API for {operation_name}")

        # Check if prompt is a list of messages with roles
        messages = None
        system_text = None
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
            # It's a conversation history with roles
            # Extract system messages (Anthropic requires top-level system param)
            system_msgs = [m for m in prompt if m.get("role") == "system"]
            messages = [m for m in prompt if m.get("role") != "system"]
            if system_msgs:
                system_text = "\n\n".join(m.get("content", "") for m in system_msgs)
        else:
            # Process content for images and create single user message
            content = self._process_content(prompt)
            messages = [{"role": "user", "content": content}]

        # Add cache_control markers only when explicitly requested
        if enable_cache and system_text:
            system_param = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        elif system_text:
            system_param = system_text
        else:
            system_param = None

        if enable_cache:
            # Only mark first user message for caching (hits translation call's cache)
            # Don't mark assistant messages — cache_write ($3.75/MTok) is more expensive
            # than regular input ($3/MTok), and downstream cache_read (dedup) rarely triggers
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 100:
                        msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                    break

        # Build request kwargs
        request_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True
        }
        # Opus 4.7+ and fable models deprecate temperature parameter.
        if not any(tag in model for tag in ("opus-4-7", "opus-4-8", "fable")):
            request_kwargs["temperature"] = temperature
        if system_param:
            request_kwargs["system"] = system_param

        # Anthropic doesn't have native JSON mode, use system prompt instead
        if json_mode:
            request_kwargs["system"] = "You must respond with valid JSON only. No explanations, no markdown code blocks, just raw JSON."

        # Create message with streaming
        _trace_start = _time.monotonic()
        _trace_error = None
        response_text = ""
        finish_reason = None
        _usage = {}
        _usage_source = "missing"

        try:
            stream = self.client.messages.create(**request_kwargs)

            # Aggregate streamed response and capture all events
            chunk_count = 0
            last_log_length = 0
            self._last_stream_events = []

            halted_early = False
            for event in stream:
                if event.type in ("message_start", "message_delta"):
                    self._last_stream_events.append(event)
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        response_text += event.delta.text
                        chunk_count += 1

                        # Hallucination guard (same as Gemini)
                        if chunk_count % 20 == 0:
                            halt_reason = _detect_streaming_hallucination(response_text)
                            if halt_reason:
                                logger.warning(
                                    f"Hallucination detected for {operation_name}: {halt_reason}. "
                                    f"Halting stream at {len(response_text)} chars."
                                )
                                halted_early = True
                                last_nl = response_text.rfind('\n')
                                if last_nl > 0:
                                    response_text = response_text[:last_nl]
                                break

                        current_tokens = len(self.tokenizer.encode(response_text))
                        if current_tokens - last_log_length >= 500:
                            logger.debug(f"Streaming {operation_name}: {current_tokens} tokens")
                            last_log_length = current_tokens

            if halted_early:
                finish_reason = "HALLUCINATION_HALT"

            if not response_text:
                raise ValueError(f"Empty response from Anthropic for {operation_name}")

            # Log complete usage metadata.
            _usage, _usage_source = extract_anthropic_stream_usage(self._last_stream_events)
            _status = usage_status(_usage, response_text)
            if _usage:
                logger.info(f"Anthropic {operation_name} usage ({_status}): {_usage}")
            else:
                logger.warning(f"Anthropic {operation_name} usage unavailable")
            return response_text
        except Exception as e:
            _trace_error = str(e)
            raise
        finally:
            _write_trace({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation_name,
                "provider": "anthropic",
                "model": model,
                "duration_ms": int((_time.monotonic() - _trace_start) * 1000),
                "input_tokens": _usage.get("input_tokens", 0),
                "output_tokens": _usage.get("output_tokens", 0),
                "cache_read_tokens": _usage.get("cache_read_input_tokens", 0),
                "cache_write_tokens": _usage.get("cache_creation_input_tokens", 0),
                "usage_source": _usage_source,
                "usage_status": usage_status(_usage, response_text),
                "raw_request": messages,
                "raw_response": response_text,
                "finish_reason": finish_reason,
                "error": _trace_error,
                "partial_response_length": len(response_text) if _trace_error else 0,
            })

    def _process_content(self, prompt: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
        """Process content to handle images properly."""
        if isinstance(prompt, str):
            return prompt
        
        if isinstance(prompt, list):
            processed = []
            for part in prompt:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        processed.append(part)
                    elif part.get("type") == "image":
                        # Check if already in Anthropic format
                        if "source" in part and isinstance(part["source"], dict):
                            # Already formatted correctly, just pass through
                            processed.append(part)
                        else:
                            # Convert image bytes to base64
                            image_data = part.get("data")
                            mime_type = part.get("mime_type", "image/png")
                            
                            if isinstance(image_data, bytes):
                                base64_data = base64.b64encode(image_data).decode('utf-8')
                            else:
                                base64_data = image_data
                            
                            processed.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_data
                                }
                            })
                    else:
                        processed.append(part)
                else:
                    processed.append(part)
            return processed
        
        return prompt


class OpenAIClient:
    """Wrapper for OpenAI API with smart retry logic."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None, num_retries: int = 3, max_backoff_seconds: int = 30):
        """Initialize OpenAI client."""
        from openai import OpenAI

        # Set up client with optional base URL
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        # Store default model
        self.default_model = model or "gpt-4o"
        self.num_retries = num_retries
        self.max_backoff_seconds = max_backoff_seconds

        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def embed_content(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Embed texts using OpenAI-compatible embeddings endpoint.

        Sends all texts in a single batch request.
        """
        resp = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in resp.data]

    @retry(
        retry=retry_if_exception(is_transient_openai_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"OpenAI API retry {retry_state.attempt_number} for "
            f"{retry_state.args[0] if retry_state.args else 'unknown operation'}: "
            f"{retry_state.outcome.exception()}"
        )
    )
    def generate_content(
        self,
        prompt: Union[str, List[Dict]],
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.1,
        operation_name: str = "OpenAI API call",
        json_mode: bool = False,
        extra_body: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
    ) -> str:
        """Generate content with automatic retry for transient errors.

        Args:
            prompt: Text or multi-part content (ignored if messages is provided)
            messages: Pre-formatted messages list (overrides prompt if provided)
            extra_body: Extra parameters to pass to the API (e.g. chat_template_kwargs)
        """
        logger.info(f"Calling OpenAI API for {operation_name}")

        # Use provided model or default
        model_to_use = model or self.default_model

        # Process content for messages format
        if messages is not None:
            formatted_messages = messages
        else:
            formatted_messages = self._format_messages(prompt)

        # Build request kwargs
        request_kwargs = {
            "model": model_to_use,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        _trace_start = _time.monotonic()
        _trace_error = None
        response_text = ""
        finish_reason = None

        try:
            # Create chat completion with streaming
            stream = self.client.chat.completions.create(**request_kwargs)

            # Aggregate streamed response
            chunk_count = 0
            last_log_tokens = 0
            for chunk in stream:
                if not chunk.choices:
                    continue
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    chunk_count += 1

                    # Log progress periodically - every 500 tokens
                    current_tokens = len(self.tokenizer.encode(response_text))
                    if current_tokens - last_log_tokens >= 500:
                        logger.debug(f"Streaming {operation_name}: {current_tokens} tokens")
                        last_log_tokens = current_tokens

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if not response_text:
                raise ValueError(f"Empty response from OpenAI for {operation_name}")

            # Store finish_reason for caller inspection
            self._last_finish_reason = finish_reason

            # Get final token count
            final_tokens = len(self.tokenizer.encode(response_text))
            logger.info(f"Streamed {final_tokens} tokens ({chunk_count} chunks) from OpenAI for {operation_name} [finish={finish_reason}]")
            return response_text
        except Exception as e:
            _trace_error = str(e)
            raise
        finally:
            _write_trace({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation_name,
                "provider": "openai",
                "model": model_to_use,
                "duration_ms": int((_time.monotonic() - _trace_start) * 1000),
                "input_tokens": 0,
                "output_tokens": 0,
                "raw_request": formatted_messages,
                "raw_response": response_text,
                "finish_reason": finish_reason,
                "error": _trace_error,
                "partial_response_length": len(response_text) if _trace_error else 0,
            })

    def _format_messages(self, prompt: Union[str, List[Dict]]) -> List[Dict]:
        """Format prompt into OpenAI messages format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        if isinstance(prompt, list):
            # Check if this is already a conversation messages list
            if prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                # Already formatted as messages — pass through, converting
                # system messages to "system" role (OpenAI/DeepSeek compatible)
                return [{"role": m["role"], "content": m["content"]} for m in prompt]

            # Handle multi-part content (image + text blocks)
            content_parts = []
            for part in prompt:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif part.get("type") == "image":
                        # Handle image data
                        if "source" in part and isinstance(part["source"], dict):
                            # Already in structured format
                            base64_data = part["source"].get("data")
                            mime_type = part["source"].get("media_type", "image/png")
                        else:
                            # Simple format
                            image_data = part.get("data")
                            mime_type = part.get("mime_type", "image/png")
                            
                            if isinstance(image_data, bytes):
                                base64_data = base64.b64encode(image_data).decode('utf-8')
                            else:
                                base64_data = image_data
                        
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}"
                            }
                        })
                    else:
                        # Pass through other types
                        content_parts.append(part)
                else:
                    # String content
                    content_parts.append({
                        "type": "text",
                        "text": str(part)
                    })
            
            return [{"role": "user", "content": content_parts}]
        
        # Default case
        return [{"role": "user", "content": str(prompt)}]




def create_gemini_client_from_config(
    config: Dict[str, Any],
    provider_name: str = "gemini",
    num_retries: int = 3,
    max_backoff_seconds: int = 30
) -> GeminiClient:
    """
    Create a GeminiClient from config with full provider support.

    Args:
        config: Full config dict
        provider_name: Provider name (e.g., 'gemini', 'vertex', 'gemini-vertex')
        num_retries: Number of retries for transient errors
        max_backoff_seconds: Maximum backoff time between retries

    Returns:
        Configured GeminiClient instance

    Raises:
        ValueError: If API key not found
    """
    providers = config.get("credentials", {}).get("providers", {})
    provider_config = providers.get(provider_name, {})

    # Get provider settings - only fallback to legacy if provider not found
    if provider_config:
        api_key = provider_config.get("api_key")
        base_url = provider_config.get("base_url")
        vertexai = provider_config.get("vertexai", False)
        extra_headers = provider_config.get("extra_headers")
    else:
        # Legacy fallback for backward compatibility
        api_key = config.get("google_api_key")
        base_url = config.get("google_base_url")
        vertexai = False
        extra_headers = None

    if not api_key:
        raise ValueError(f"API key not found for provider '{provider_name}'")

    return GeminiClient(
        api_key=api_key,
        base_url=base_url,
        vertexai=vertexai,
        extra_headers=extra_headers,
        num_retries=num_retries,
        max_backoff_seconds=max_backoff_seconds
    )
