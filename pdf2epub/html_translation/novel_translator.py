"""
Novel Translator v4: Long-context translation with deterministic verification.

Translates chapters using Sonnet with token budget streaming. Deterministic
verifier checks line count alignment and detects hallucination. When initial
translation is incomplete, falls back to chunked_translator for remaining lines.
"""

import re
import json
import time
import signal
import logging
import tiktoken
from pathlib import Path
from typing import List, Optional

from .novel_extractor import NovelUnit
from .glossary_manager import GlossaryManager

logger = logging.getLogger(__name__)

# Maximum input tokens before fail-fast (user should split the chapter)
MAX_CHAPTER_TOKENS = 50_000

# Output token budget per chunk — Haiku's attention degrades above ~7500 tokens
CHUNK_OUTPUT_TOKEN_BUDGET = 20000

IMAGE_RE = re.compile(r'\[Image:\s*[^\]]+\]')
_MARKDOWN_HEADING_RE = re.compile(r'^#+\s+')

NOVEL_TRANSLATE_PROMPT = (
    "你是一位精通二次元文化的资深轻小说翻译家。"
    "请将日文文本翻译成流畅、优美的中文。\n\n"
    "**核心要求：**\n"
    "1. **信达雅：** 译文需符合中文轻小说阅读习惯，还原原作的沉浸感与文学性。\n"
    "2. **段落对应：** 保持每行翻译对应原文一行，不要合并或拆分段落。\n"
    "3. **保留标记：** [Image: xxx] 等标记原样保留，不要翻译或删除。\n"
    "4. **引号格式：** 对话使用直角引号「」和『』，不要替换为""或其他引号。"
)


# ─── State ───

class NovelState:
    """Persistent state for novel translation (resume support)."""

    def __init__(self, current_unit_index: int = 0, completed_units: List[int] = None):
        self.current_unit_index = current_unit_index
        self.completed_units = completed_units or []

    def save(self, path: Path):
        path.write_text(json.dumps({
            "current_unit_index": self.current_unit_index,
            "completed_units": self.completed_units,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NovelState":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            current_unit_index=data.get("current_unit_index", 0),
            completed_units=data.get("completed_units", []),
        )




# ─── Image Repair ───

def strip_spurious_headings(translated_text: str, source_text: str) -> str:
    """Remove markdown heading prefixes (# ## etc.) that don't exist in source.

    Haiku sometimes adds # to the first line or after section breaks.
    If the source line at the same position doesn't start with #, strip it.
    """
    trans_lines = translated_text.splitlines()
    src_lines = source_text.splitlines()

    for i, line in enumerate(trans_lines):
        if _MARKDOWN_HEADING_RE.match(line):
            # Check if source at same position also has #
            src_line = src_lines[i] if i < len(src_lines) else ""
            if not src_line.strip().startswith('#'):
                trans_lines[i] = _MARKDOWN_HEADING_RE.sub('', line)

    return '\n'.join(trans_lines)


def repair_images(source_text: str, translated_text: str) -> str:
    """Ensure image placeholders match source exactly.

    - Remove hallucinated image placeholders (not in source)
    - Re-insert missing image placeholders at original line positions
    """
    source_images = {}
    for i, line in enumerate(source_text.splitlines()):
        m = IMAGE_RE.search(line)
        if m:
            source_images[i] = m.group(0)

    translated_lines = translated_text.splitlines()
    source_filenames = set(source_images.values())

    # Remove hallucinated image placeholders
    for i, line in enumerate(translated_lines):
        m = IMAGE_RE.search(line)
        if m and m.group(0) not in source_filenames:
            translated_lines[i] = IMAGE_RE.sub('', line).strip()

    # Find which images survived
    translated_images = set()
    for line in translated_lines:
        m = IMAGE_RE.search(line)
        if m:
            translated_images.add(m.group(0))

    # Re-insert missing images at original positions
    for line_idx, placeholder in sorted(source_images.items()):
        if placeholder not in translated_images:
            insert_at = min(line_idx, len(translated_lines))
            translated_lines.insert(insert_at, placeholder)

    return '\n'.join(translated_lines)


# ─── Translator ───

class NovelTranslator:
    """Translate light novel chapters with deterministic verification and glossary memory."""

    def __init__(
        self,
        config: dict,
        book_title: str,
        source_language: str = "Japanese",
        target_language: str = "Chinese",
        glossary_manager: Optional[GlossaryManager] = None,
        resume: bool = False,
        output_dir: Optional[Path] = None,
    ):
        self.config = config
        self.book_title = book_title
        self.source_language = source_language
        self.target_language = target_language
        self.glossary_manager = glossary_manager
        self.resume = resume

        self.output_dir = output_dir or Path(f"output/{book_title}")
        self.translated_dir = self.output_dir / "translated_novel"
        self.translated_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.output_dir / "novel_state.json"

        # Tokenizer for input length guard
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # LLM client (lazy init)
        self._llm_client = None
        self._model_configs = None

        # Embedding config for verifier/position alignment (optional)
        novel_config = config.get("novel", {})
        self._embedding_provider = novel_config.get("embedding_provider")
        self._embedding_model = novel_config.get("embedding_model", "gemini-embedding-001")
        self._hallucination_threshold = novel_config.get("hallucination_threshold", 0.75)

        # Cheap model for bilingual stripping (optional)
        strip_model = novel_config.get("strip_model")
        if strip_model:
            self._strip_model_configs = [{
                "provider": strip_model.get("provider"),
                "model": strip_model.get("model"),
            }]
        else:
            self._strip_model_configs = None

    def _get_llm_client(self):
        if self._llm_client is None:
            from ..utils.llm_client import LLMClient
            self._llm_client = LLMClient(self.config)
        return self._llm_client

    def _get_model_configs(self):
        if self._model_configs is None:
            translation_config = self.config.get("translation", {})
            self._model_configs = translation_config.get("models", [
                {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
            ])
        return self._model_configs

    # NOTE: _get_agent_model removed — agent-based verification replaced by
    # deterministic verifier (novel_verifier.py). Config "novel.agent_model" no longer used.

    def _get_anthropic_client(self):
        """Get the raw Anthropic client for streaming calls."""
        if not hasattr(self, '_anthropic_client') or self._anthropic_client is None:
            from ..utils.network_utils import AnthropicClient
            providers = self.config.get("credentials", {}).get("providers", {})
            anthropic_cfg = providers.get("anthropic", {})
            self._anthropic_client = AnthropicClient(
                api_key=anthropic_cfg.get("api_key", ""),
                base_url=anthropic_cfg.get("base_url"),
            )
        return self._anthropic_client

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _stream_with_token_cutoff(
        self,
        messages: list,
        system_text: str,
        max_output_tokens: int,
        operation_name: str,
    ) -> str:
        """Stream from Anthropic, stop when output reaches token budget.

        After hitting the budget, continues streaming until the next newline
        to avoid cutting a line in half.
        """
        client = self._get_anthropic_client()
        model_configs = self._get_model_configs()
        # Find the anthropic model from config, or fall back to sonnet
        anthropic_config = next(
            (m for m in model_configs if m.get("provider", "") == "anthropic"),
            None,
        )
        model = anthropic_config.get("model") if anthropic_config else "claude-sonnet-4-6"

        # Use cache_control on system prompt and first user message
        # so source text is cached across continuation rounds
        system_with_cache = [
            {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}},
        ]

        # Add cache_control to first user message and last assistant message
        # so the entire conversation prefix is cached across continuation rounds
        cached_messages = []
        for i, msg in enumerate(messages):
            if i == 0 and msg.get("role") == "user":
                cached_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}},
                    ],
                })
            else:
                cached_messages.append(msg)

        # Mark the last assistant message for caching (if exists)
        for i in range(len(cached_messages) - 1, -1, -1):
            if cached_messages[i].get("role") == "assistant":
                content = cached_messages[i].get("content", "")
                if isinstance(content, str):
                    cached_messages[i] = {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
                        ],
                    }
                break

        request_kwargs = {
            "model": model,
            "messages": cached_messages,
            "max_tokens": 64000,
            "stream": True,
            "system": system_with_cache,
        }
        # Opus 4.7+ deprecates temperature; only set for older models.
        if not any(tag in model for tag in ("opus-4-7", "opus-4-8")):
            request_kwargs["temperature"] = 0.1

        import time as _time
        from pdf2epub.utils.network_utils import (
            _write_trace, _detect_streaming_hallucination,
            is_transient_anthropic_error, extract_anthropic_stream_usage,
            usage_status,
        )

        _trace_start = _time.monotonic()
        _trace_error = None
        response_text = ""
        stopped_early = False
        _usage = {}
        _usage_source = "missing"

        max_retries = client.num_retries
        last_error = None

        try:
            for attempt in range(1 + max_retries):
                if attempt > 0:
                    wait = min(2 ** attempt, client.max_backoff_seconds)
                    logger.warning(
                        f"Retry {attempt}/{max_retries} for {operation_name}: "
                        f"waiting {wait}s | {last_error}"
                    )
                    _time.sleep(wait)
                    # Reset state for retry
                    response_text = ""
                    stopped_early = False
                    _usage = {}
                    _usage_source = "missing"

                try:
                    stream = client.client.messages.create(**request_kwargs)

                    budget_hit = False
                    stream_events = []
                    chunk_count = 0

                    for event in stream:
                        if event.type in ("message_start", "message_delta"):
                            stream_events.append(event)
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                chunk = event.delta.text
                                response_text += chunk
                                chunk_count += 1

                                # Hallucination guard: detect repetition loops
                                if chunk_count % 20 == 0:
                                    halt_reason = _detect_streaming_hallucination(response_text)
                                    if halt_reason:
                                        logger.warning(
                                            f"Hallucination detected for {operation_name}: {halt_reason}. "
                                            f"Halting stream at {len(response_text)} chars."
                                        )
                                        last_nl = response_text.rfind('\n')
                                        if last_nl > 0:
                                            response_text = response_text[:last_nl]
                                        stopped_early = True
                                        stream.close()
                                        break

                                if budget_hit:
                                    # We hit the budget — finish the current line
                                    if '\n' in chunk:
                                        last_nl = response_text.rfind('\n')
                                        if last_nl > 0:
                                            response_text = response_text[:last_nl]
                                        stopped_early = True
                                        stream.close()
                                        break
                                elif '\n' in chunk:
                                    # Check token count at line boundaries
                                    current_tokens = self._count_tokens(response_text)
                                    if current_tokens >= max_output_tokens:
                                        budget_hit = True
                                        last_nl = response_text.rfind('\n')
                                        if last_nl > 0:
                                            response_text = response_text[:last_nl]
                                        stopped_early = True
                                        stream.close()
                                        break
                    # Extract cumulative usage from all stream events.
                    _usage, _usage_source = extract_anthropic_stream_usage(stream_events)
                    _status = usage_status(_usage, response_text)
                    if _usage:
                        logger.info(f"Anthropic {operation_name} usage ({_status}): {_usage}")
                    else:
                        logger.warning(f"Anthropic {operation_name} usage unavailable")

                    # Clean: remove empty lines
                    cleaned = '\n'.join(l for l in response_text.splitlines() if l.strip())
                    return cleaned

                except Exception as e:
                    last_error = str(e)
                    if is_transient_anthropic_error(e) and attempt < max_retries:
                        continue  # retry
                    _trace_error = str(e)
                    raise
        finally:
            _write_trace({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
                "raw_request": cached_messages,
                "raw_response": response_text,
                "stopped_early": stopped_early,
                "error": _trace_error,
                "partial_response_length": len(response_text) if _trace_error else 0,
            })

    # ─── Main Loop ───

    def translate_all(self, units: List[NovelUnit]) -> dict:
        """Translate all units sequentially."""
        if self.resume and self.state_path.exists():
            state = NovelState.load(self.state_path)
            logger.info(f"Resuming from unit {state.current_unit_index}")
        else:
            state = NovelState()

        HALLUCINATION_GRACE_CHAPTERS = 5  # Don't check ratio for first N chapters
        HALLUCINATION_MAX_RATIO = 0.2    # Abort if hallucination rate exceeds this

        start_time = time.time()
        translated_count = 0
        skipped_count = 0
        hallucinated_count = 0

        _prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            logger.warning("SIGTERM received — saving state")
            state.save(self.state_path)
            if self.glossary_manager:
                self.glossary_manager.save()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _sigterm_handler)

        try:
            for idx, unit in enumerate(units):
                already_completed = (
                    idx < state.current_unit_index
                    or idx in state.completed_units
                )
                existing_translation = None
                if already_completed:
                    if self.resume and unit.has_content and unit.text_path:
                        dest = self.translated_dir / unit.text_path.name
                        if dest.exists():
                            source_text = unit.text_path.read_text(encoding="utf-8")
                            candidate = dest.read_text(encoding="utf-8")
                            src_lines = len([l for l in source_text.splitlines() if l.strip()])
                            tl_lines = len([l for l in candidate.splitlines() if l.strip()])
                            if tl_lines < src_lines:
                                existing_translation = candidate
                                logger.warning(
                                    f"  Reopening incomplete completed unit {unit.text_path.name}: "
                                    f"{tl_lines}/{src_lines} lines; continuing existing translation"
                                )

                    if existing_translation is None:
                        skipped_count += 1
                        continue

                if not unit.has_content or not unit.text_path:
                    if unit.text_path and unit.text_path.exists():
                        dest = self.translated_dir / unit.text_path.name
                        dest.write_text(
                            unit.text_path.read_text(encoding="utf-8"),
                            encoding="utf-8",
                        )
                    skipped_count += 1
                    state.completed_units.append(idx)
                    state.current_unit_index = idx + 1
                    state.save(self.state_path)
                    continue

                if existing_translation is None:
                    logger.info(f"Translating [{idx + 1}/{len(units)}]: {unit.text_path.name}")
                chapter_failed = self._translate_chapter(
                    unit,
                    existing_translation=existing_translation,
                )
                translated_count += 1

                if chapter_failed:
                    hallucinated_count += 1
                    logger.warning(
                        f"  Chapter marked as hallucinated "
                        f"({hallucinated_count}/{translated_count} = "
                        f"{hallucinated_count/translated_count:.0%})"
                    )

                    # Check ratio after grace period
                    if translated_count > HALLUCINATION_GRACE_CHAPTERS:
                        ratio = hallucinated_count / translated_count
                        if ratio > HALLUCINATION_MAX_RATIO:
                            logger.error(
                                f"Hallucination rate {ratio:.0%} exceeds "
                                f"{HALLUCINATION_MAX_RATIO:.0%} threshold after "
                                f"{translated_count} chapters. Aborting translation."
                            )
                            # Mark current chapter completed (best-effort saved)
                            state.completed_units.append(idx)
                            state.current_unit_index = idx + 1
                            state.save(self.state_path)
                            return {
                                "translated": translated_count,
                                "skipped": skipped_count,
                                "hallucinated": hallucinated_count,
                                "aborted": True,
                                "abort_reason": f"hallucination rate {ratio:.0%}",
                            }

                # Always mark completed — hallucinated chapters save best-effort output
                # and can be fixed later with --retranslate
                if idx not in state.completed_units:
                    state.completed_units.append(idx)
                state.current_unit_index = max(state.current_unit_index, idx + 1)
                state.save(self.state_path)

        except BaseException:
            logger.warning("Interrupted — saving state")
            state.save(self.state_path)
            if self.glossary_manager:
                self.glossary_manager.save()
            raise
        finally:
            signal.signal(signal.SIGTERM, _prev_sigterm)

        elapsed = time.time() - start_time
        logger.info(
            f"Translation complete: {translated_count} translated, "
            f"{skipped_count} skipped in {elapsed:.1f}s"
        )
        return {
            "translated": translated_count,
            "skipped": skipped_count,
            "hallucinated": hallucinated_count,
            "elapsed": elapsed,
        }

    # ─── Chapter Translation ───

    def _translate_chapter(
        self,
        unit: NovelUnit,
        existing_translation: Optional[str] = None,
    ) -> bool:
        """Translate a single chapter: recall glossary → translate → extract glossary.

        Returns True if the chapter hallucinated (exhausted retries), False if OK.
        """
        source_text_raw = unit.text_path.read_text(encoding="utf-8")

        # Degeneration guard: truncate repetitive kana sequences in source
        from .chunked_translator import compress_repetitive_source
        source_text = compress_repetitive_source(source_text_raw)

        # Input length guard
        tokens = self._count_tokens(source_text)
        if tokens > MAX_CHAPTER_TOKENS:
            raise ValueError(
                f"Chapter {unit.file_name} is {tokens} tokens — too long. "
                f"Please split the chapter before translating."
            )

        # Step 1: Recall glossary
        glossary_prompt = ""
        if self.glossary_manager:
            glossary_prompt = self.glossary_manager.recall(source_text)

        # Step 2: Translate
        translated, exhausted = self._run_translation(
            unit,
            source_text,
            glossary_prompt,
            existing_translation=existing_translation,
        )

        # Step 3: Repair images (using raw source for correct placeholders)
        translated = repair_images(source_text_raw, translated)

        dest = self.translated_dir / unit.text_path.name
        src_lines = len([l for l in source_text.splitlines() if l.strip()])
        tl_lines = len([l for l in translated.splitlines() if l.strip()])
        logger.info(f"  Lines: {src_lines} → {tl_lines} (diff={tl_lines - src_lines:+d})")

        # A resume repair must never replace the recoverable prefix with another
        # structurally incomplete attempt.
        if existing_translation is not None and tl_lines != src_lines:
            logger.warning(
                f"  Continuation repair remains incomplete for {unit.file_name}: "
                f"{tl_lines}/{src_lines} lines; preserving existing translation"
            )
            return True

        # Step 4: Save fresh translations immediately. Resume repairs are saved
        # after their glossary entry has been replaced below.
        if existing_translation is None:
            dest.write_text(translated, encoding="utf-8")

        # Step 6: Extract glossary (cache hit on translation's source prefix)
        if self.glossary_manager:
            chapter_id = unit.text_path.stem
            if existing_translation is not None:
                self.glossary_manager.rollback_chapter(chapter_id)
            # Build same system prompt as translation for cache sharing
            system_prompt = NOVEL_TRANSLATE_PROMPT
            if glossary_prompt:
                system_prompt = f"{NOVEL_TRANSLATE_PROMPT}\n\n{glossary_prompt}"
            self.glossary_manager.extract_and_update(
                source_text, translated, chapter_id,
                translation_system_prompt=system_prompt,
            )

        if existing_translation is not None:
            dest.write_text(translated, encoding="utf-8")

        return exhausted

    def _run_translation(
        self,
        unit: NovelUnit,
        source_text: str,
        glossary_prompt: str,
        existing_translation: Optional[str] = None,
    ) -> tuple:
        """Run translation with deterministic verification.

        Returns (translated_text, exhausted_retries) where exhausted_retries
        is True if all retry attempts were used without clean completion.
        """
        from .novel_verifier import verify_translation

        # Build system prompt with glossary
        system_prompt = NOVEL_TRANSLATE_PROMPT
        if glossary_prompt:
            system_prompt = f"{NOVEL_TRANSLATE_PROMPT}\n\n{glossary_prompt}"

        # Stateful conversation history for cache-friendly continuations
        conversation_history = []
        prev_translated_lines = 0
        last_cont_output_lines = 0
        max_continuations = 10
        max_retries = 2

        def generate_fn(prefix=None):
            nonlocal conversation_history, prev_translated_lines, last_cont_output_lines

            if prefix is None:
                conversation_history = [
                    {"role": "user", "content": f"请翻译：\n{source_text}"},
                ]
                prev_translated_lines = 0
                op_name = f"Novel translate {unit.file_name}"
            else:
                prefix_lines = [l for l in prefix.splitlines() if l.strip()]
                current_translated_lines = len(prefix_lines)
                last_line = prefix_lines[-1] if prefix_lines else ""

                # Detect hallucination: if verifier truncated many lines from
                # last continuation, fall back to prefix mode (rebuild history)
                if last_cont_output_lines > 0:
                    actual_added = current_translated_lines - prev_translated_lines
                    lines_removed = last_cont_output_lines - actual_added
                    if lines_removed >= 10:
                        logger.warning(
                            f"  Hallucination detected: continuation had {last_cont_output_lines} lines, "
                            f"but only {actual_added} were kept ({lines_removed} removed). "
                            f"Falling back to prefix mode (cache partially lost)."
                        )
                        conversation_history = [
                            {"role": "user", "content": f"请翻译：\n{source_text}"},
                            {"role": "assistant", "content": prefix},
                        ]

                conversation_history.append({
                    "role": "user",
                    "content": f"继续翻译，只输出中文译文。上一行译文：\n{last_line}",
                })
                op_name = f"Novel continue {unit.file_name} from line {current_translated_lines}"

            result = self._stream_with_token_cutoff(
                messages=conversation_history,
                system_text=system_prompt,
                max_output_tokens=CHUNK_OUTPUT_TOKEN_BUDGET,
                operation_name=op_name,
            )
            result = strip_spurious_headings(result, source_text)

            conversation_history.append({"role": "assistant", "content": result})
            last_cont_output_lines = len([l for l in result.splitlines() if l.strip()])
            if prefix is not None:
                prev_translated_lines = len([l for l in prefix.splitlines() if l.strip()])
            else:
                prev_translated_lines = 0

            return result

        # Verification model — same as translation model for now
        verify_model_configs = self._get_model_configs()

        # Main loop: generate → verify → continue/complete
        translated = None
        for attempt in range(max_retries):
            if attempt == 0 and existing_translation is not None:
                translated = existing_translation.strip()
                conversation_history = [
                    {"role": "user", "content": f"请翻译：\n{source_text}"},
                    {"role": "assistant", "content": translated},
                ]
                prev_translated_lines = len(
                    [l for l in translated.splitlines() if l.strip()]
                )
                last_cont_output_lines = 0
                continuation = generate_fn(prefix=translated)
                if continuation:
                    translated = translated.rstrip("\n") + "\n" + continuation
                logger.info(
                    f"  Resumed existing translation: {prev_translated_lines} + "
                    f"{last_cont_output_lines} lines"
                )
            else:
                # Initial translation
                raw_output = generate_fn(prefix=None)
                translated = raw_output
                init_lines = len([l for l in raw_output.splitlines() if l.strip()])
                logger.info(f"  Initial translation: {init_lines} lines (attempt {attempt + 1})")

            # Verify + continuation loop
            for cont_round in range(max_continuations + 1):
                fixed_text, action = verify_translation(
                    source_text=source_text,
                    translated_text=translated,
                    llm_client=self._get_llm_client(),
                    model_configs=verify_model_configs,
                    is_first_chunk=(cont_round == 0),
                    embedding_provider=self._embedding_provider,
                    embedding_model=self._embedding_model,
                    hallucination_threshold=self._hallucination_threshold,
                    strip_model_configs=self._strip_model_configs,
                )

                if action == "complete":
                    src_n = len([l for l in source_text.splitlines() if l.strip()])
                    tl_n = len([l for l in fixed_text.splitlines() if l.strip()])
                    logger.info(f"  Lines: {src_n} source → {tl_n} translated (diff={tl_n - src_n:+d})")
                    return fixed_text, False

                elif action == "continue":
                    translated = fixed_text
                    prefix_lines = len([l for l in translated.splitlines() if l.strip()])
                    continuation = generate_fn(prefix=translated)
                    continuation_lines = len(
                        [l for l in continuation.splitlines() if l.strip()]
                    )
                    if continuation:
                        translated = translated.rstrip("\n") + "\n" + continuation
                    total_lines = len([l for l in translated.splitlines() if l.strip()])
                    logger.info(
                        f"  Continuation {cont_round + 1}: "
                        f"+{continuation_lines} lines → {total_lines} total "
                        f"(was {prefix_lines})"
                    )

                elif action == "retry":
                    logger.warning(f"  Verifier requested retry (attempt {attempt + 1})")
                    break  # Break continuation loop, retry from scratch

            else:
                # Exhausted continuations without completing
                tl_n = len([l for l in (translated or "").splitlines() if l.strip()])
                logger.warning(f"  Exhausted {max_continuations} continuations ({tl_n} lines)")
                if translated:
                    return translated, True

        # Exhausted retries — return whatever we have
        tl_n = len([l for l in (translated or "").splitlines() if l.strip()])
        logger.warning(f"  Exhausted {max_retries} retries, returning best effort ({tl_n} lines)")
        return translated or "", True
