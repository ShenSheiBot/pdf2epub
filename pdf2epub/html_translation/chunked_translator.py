"""
Chunked translator module for fallback translation of remaining lines.

When initial long-context translation doesn't cover the full chapter
(token budget cutoff or model skip), this module translates the remaining
lines in short-context chunks that are experimentally proven stable.

Each chunk is translated independently with sliding window context
(prev chunk's JP + CN). Chunks are pre-split by token budget before
any translation begins, so there is no drift or seam issue.
"""

import logging
import tiktoken
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Forced model for position alignment (cheap classification task)
_HAIKU_MODEL_CONFIGS = [{"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}]


def compress_repetitive_source(text: str, max_keep: int = 5, min_run: int = 10) -> str:
    """Truncate long repetitive kana sequences to prevent degeneration.

    Detects runs where ≤2 unique characters alternate for ≥min_run chars
    and truncates to max_keep chars. Handles patterns like:
    アアアァアァアアア... → アアァアァ (5 chars)
    """
    result = []
    i = 0
    while i < len(text):
        if i + min_run <= len(text):
            chars = set()
            j = i
            while j < len(text):
                new_chars = chars | {text[j]}
                if len(new_chars) > 2:
                    break
                chars = new_chars
                j += 1
            run_len = j - i
            if run_len >= min_run:
                result.append(text[i:i + max_keep])
                i = j
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)


def _find_source_position(
    source_text: str,
    translated_prefix: str,
    llm_client,
    embedding_provider: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
) -> int:
    """Find which source line the last translated line corresponds to.

    Primary: embedding-based argmax (cosine similarity against 9 candidates).
    Fallback: haiku LLM classification.
    Last resort: line count estimate.

    Returns 0-indexed source line number.
    """
    from .embedding_utils import find_position_embedding

    src_lines = [l for l in source_text.splitlines() if l.strip()]
    tl_lines = [l for l in translated_prefix.splitlines() if l.strip()]

    if not tl_lines:
        return 0

    # Estimate: translated line count ≈ source position
    estimate = min(len(tl_lines), len(src_lines)) - 1

    # Take a window of 9 source lines around estimate
    start = max(0, estimate - 4)
    end = min(len(src_lines), estimate + 5)
    window = src_lines[start:end]

    if not window:
        return estimate

    last_tl = tl_lines[-1]

    # Primary: embedding-based alignment
    if embedding_provider:
        idx = find_position_embedding(
            last_tl, window, llm_client,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        if idx is not None:
            aligned = start + idx
            logger.info(f"  Position alignment (embedding): line {aligned + 1} (window {start+1}-{end}, idx={idx})")
            return aligned
        logger.info("  Embedding position alignment unavailable, falling back to LLM")

    # Fallback: haiku LLM classification
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(window))
    prompt = (
        f"以下是{len(window)}行日语原文，编号1-{len(window)}：\n{numbered}\n\n"
        f"以下中文译文最接近哪一行原文？只回答数字。\n{last_tl}"
    )

    for attempt in range(3):
        try:
            result = llm_client.generate(
                prompt=prompt,
                model_configs=_HAIKU_MODEL_CONFIGS,
                operation_name=f"Chunk position alignment (attempt {attempt + 1})",
            )
            match_num = int(''.join(c for c in result.strip() if c.isdigit()) or '0')
            if 1 <= match_num <= len(window):
                aligned = start + match_num - 1
                logger.info(f"  Position alignment (LLM): line {aligned + 1} (window {start+1}-{end}, match={match_num})")
                return aligned
        except Exception as e:
            logger.warning(f"  Position alignment attempt {attempt + 1} failed: {e}")
            continue

    # All attempts failed — fallback to line count estimate (0-indexed)
    fallback = min(len(tl_lines) - 1, len(src_lines) - 1)
    logger.warning(f"  Position alignment failed, fallback to line {fallback + 1}")
    return fallback


def _split_into_chunks(
    lines: List[str],
    token_budget: int,
    tokenizer,
) -> List[List[str]]:
    """Split lines into chunks by token budget.

    Each chunk contains as many lines as fit within token_budget.
    Never splits mid-line. A single line exceeding the budget
    gets its own chunk (never dropped).
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer.encode(line))
        if current_tokens + line_tokens > token_budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _validate_chunk_output(input_lines: int, output_text: str) -> Optional[str]:
    """Validate chunk translation output.

    Returns None if valid, error string if invalid.
    """
    output_lines = len([l for l in output_text.splitlines() if l.strip()])

    if output_lines == 0:
        return "Empty output"

    diff = abs(output_lines - input_lines)
    if diff > 3:
        return f"Line count mismatch: {output_lines} output vs {input_lines} input (diff={diff})"

    return None


def translate_remaining(
    source_text: str,
    translated_prefix: str,
    system_prompt: str,
    llm_client,
    model_configs,
    chunk_token_budget: int = 2000,
    embedding_provider: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
) -> Tuple[str, int, int]:
    """Translate remaining source lines using short-context chunks.

    Model retry and fallback is handled by llm_client.generate_with_validation,
    which tries each model in model_configs with its configured retry counts.

    Returns (remaining_text, hallucination_count, aligned_pos) where:
    - remaining_text: translated text for lines after aligned_pos
    - hallucination_count: chunks that exhausted all models/retries
    - aligned_pos: 0-indexed source line that the prefix's last line maps to.
      Caller should truncate prefix to (aligned_pos + 1) non-empty lines
      before concatenating, to avoid overlap.
    """
    from .novel_translator import strip_spurious_headings

    # Step 1: Position alignment
    src_lines = [l for l in source_text.splitlines() if l.strip()]
    pos = _find_source_position(
        source_text, translated_prefix, llm_client,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    remaining_lines = src_lines[pos + 1:]

    if not remaining_lines:
        logger.info("  Chunked translator: no remaining lines after alignment")
        return "", 0, pos

    logger.info(
        f"  Chunked translator: {len(remaining_lines)} lines remaining "
        f"(source lines {pos + 2}-{len(src_lines)})"
    )

    # Step 2: Pre-split into chunks by token budget
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = _split_into_chunks(remaining_lines, chunk_token_budget, tokenizer)
    logger.info(f"  Chunked translator: {len(chunks)} chunks (budget={chunk_token_budget} tokens)")

    # Step 3: Translate each chunk
    # Initialize sliding context from the prefix
    tl_prefix_lines = [l for l in translated_prefix.splitlines() if l.strip()]
    prev_jp = "\n".join(src_lines[max(0, pos - 2):pos + 1])
    prev_cn = "\n".join(tl_prefix_lines[-3:]) if len(tl_prefix_lines) >= 3 else "\n".join(tl_prefix_lines)

    hallucination_count = 0
    chunk_translations = []

    for chunk_idx, chunk_lines in enumerate(chunks):
        # Apply source text compression to prevent kana degeneration
        compressed_lines = [compress_repetitive_source(l) for l in chunk_lines]
        chunk_text = "\n".join(compressed_lines)
        n_input = len(chunk_lines)

        # Build prompt with sliding context
        user_content = (
            f"上文原文：\n{prev_jp}\n\n"
            f"上文译文：\n{prev_cn}\n\n"
            f"请翻译以下内容，每行对应一行，只输出中文译文：\n{chunk_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Line count validator for generate_with_validation
        def chunk_validator(response: str):
            cleaned = strip_spurious_headings(response, "\n".join(chunk_lines))
            error = _validate_chunk_output(n_input, cleaned)
            if error is None:
                return True, ""
            return False, error

        op_start = pos + 2 + sum(len(c) for c in chunks[:chunk_idx])
        op_end = pos + 1 + sum(len(c) for c in chunks[:chunk_idx + 1])
        op_name = f"Chunk {chunk_idx + 1}/{len(chunks)} (lines {op_start}-{op_end})"

        # Translate with multi-model retry chain
        try:
            result = llm_client.generate_with_validation(
                prompt=messages,
                model_configs=model_configs,
                validator=chunk_validator,
                operation_name=op_name,
                enable_cache=True,
            )
            result = strip_spurious_headings(result, "\n".join(chunk_lines))
            chunk_translations.append(result)
            # Update sliding context
            prev_jp = "\n".join(chunk_lines[-3:])
            result_lines = [l for l in result.splitlines() if l.strip()]
            prev_cn = "\n".join(result_lines[-3:]) if len(result_lines) >= 3 else "\n".join(result_lines)
        except Exception as e:
            # All models and retries exhausted — JP fallback
            logger.warning(
                f"  Chunk {chunk_idx + 1} failed after all models/retries: {e}"
            )
            fallback = "\n".join(f"[未翻译] {line}" for line in chunk_lines)
            chunk_translations.append(fallback)
            hallucination_count += 1
            prev_jp = "\n".join(chunk_lines[-3:])
            prev_cn = "\n".join(chunk_lines[-3:])

        succeeded = chunk_translations[-1] if chunk_translations else ""
        is_fallback = succeeded.startswith("[未翻译]") if succeeded else True
        logger.info(
            f"  Chunk {chunk_idx + 1}/{len(chunks)}: "
            f"{len(chunk_lines)} lines → "
            f"{'FALLBACK' if is_fallback else 'OK'}"
        )

    assembled = "\n".join(chunk_translations)
    logger.info(
        f"  Chunked translator complete: {len(chunks)} chunks, "
        f"{hallucination_count} failures"
    )
    return assembled, hallucination_count, pos
