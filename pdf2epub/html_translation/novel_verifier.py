"""
Deterministic translation verifier for novel translation.

Replaces the agent-based verification with:
1. Preamble detection via LLM classification (translation/meta-comment)
2. Hallucination detection via embedding similarity (primary) or A/B/C/D alignment check (fallback)

Embedding uses gemini-embedding-001 cross-lingual cosine similarity.
Falls back to haiku LLM classification if embedding API is unavailable.
"""

import logging
from typing import List, Optional, Tuple

from .embedding_utils import check_alignment_embedding

logger = logging.getLogger(__name__)

BILINGUAL_STRIP_PROMPT = """\
以下文本是一份日中双语对照翻译，每行日语原文后面紧跟着对应的中文译文。
请只保留中文译文行，删除所有日语原文行。直接输出结果，不要添加任何说明。"""

PREAMBLE_CHECK_PROMPT = """\
以下译文的第一行，是原文第一行的翻译，还是"以下是翻译"之类的说明文字？
只回答：translation 或 meta-comment

原文第一行：{source_line}
译文第一行：{translated_line}"""

ALIGNMENT_CHECK_PROMPT = """\
以下是原文（日语）和译文（中文）各若干行。请判断译文和原文的对应关系：

A: 译文逐行对应原文，翻译准确
B: 译文逐行对应原文，但翻译有偏差
C: 译文和原文存在错位（如原文第2行对应译文第4行），但内容相关
D: 译文和原文完全无关

只回答一个字母（A、B、C或D）。

原文：
{source}

译文：
{translated}"""

# The novel rebuild requires one translated line per source line. Accepting even
# a one-line shortfall here creates a state that deterministically fails later.
LINE_COUNT_TOLERANCE = 0


def _check_preamble(source_line: str, translated_line: str, llm_client, model_configs) -> str:
    """Check if translated_line is a translation or meta-comment.

    Returns 'translation' or 'meta-comment'.
    """
    prompt = PREAMBLE_CHECK_PROMPT.format(
        source_line=source_line,
        translated_line=translated_line,
    )
    try:
        result = llm_client.generate(
            prompt=prompt,
            model_configs=model_configs,
            operation_name="Verifier preamble check",
        )
        answer = result.strip().lower()
        verdict = "meta-comment" if "meta" in answer else "translation"
        return verdict
    except Exception as e:
        logger.warning(f"Preamble check failed: {e}")
        return "translation"  # Fail-open: don't delete lines on error


def _check_alignment(source_window: List[str], translated_window: List[str],
                     llm_client, model_configs,
                     embedding_provider: Optional[str] = None,
                     embedding_model: str = "gemini-embedding-001",
                     hallucination_threshold: float = 0.75) -> str:
    """Check alignment between source and translated windows.

    Tries embedding-based check first (if embedding_provider configured),
    falls back to LLM classification.

    Returns 'A', 'B', 'C', or 'D'.
    """
    # Primary: embedding-based check
    if embedding_provider:
        result = check_alignment_embedding(
            source_window, translated_window, llm_client,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            hallucination_threshold=hallucination_threshold,
        )
        if result is not None:
            return result
        logger.info("  Embedding alignment unavailable, falling back to LLM")

    # Fallback: LLM classification
    prompt = ALIGNMENT_CHECK_PROMPT.format(
        source="\n".join(source_window),
        translated="\n".join(translated_window),
    )
    try:
        result = llm_client.generate(
            prompt=prompt,
            model_configs=model_configs,
            operation_name="Verifier alignment check",
        )
        answer = result.strip().upper()
        if answer and answer[0] in "ABCD":
            verdict = answer[0]
        else:
            logger.warning(f"Unexpected alignment result: {answer!r}, defaulting to A")
            verdict = "A"
        return verdict
    except Exception as e:
        logger.warning(f"Alignment check failed: {e}")
        return "A"  # Fail-open


def remove_preamble(
    source_lines: List[str],
    translated_lines: List[str],
    llm_client,
    model_configs,
) -> Optional[List[str]]:
    """Remove preamble lines from translated text.

    Tries deleting 0, 1, or 2 lines from the start.
    Returns cleaned lines, or None if all attempts fail (needs retry).
    """
    if not translated_lines or not source_lines:
        return translated_lines

    # Check as-is
    result = _check_preamble(source_lines[0], translated_lines[0], llm_client, model_configs)
    if result == "translation":
        return translated_lines

    logger.info(f"  Verifier: preamble detected, line 1 = {translated_lines[0][:60]!r}")

    # Try removing 1 line
    if len(translated_lines) > 1:
        result = _check_preamble(source_lines[0], translated_lines[1], llm_client, model_configs)
        if result == "translation":
            logger.info("  Verifier: removed 1 preamble line")
            return translated_lines[1:]

    # Try removing 2 lines
    if len(translated_lines) > 2:
        result = _check_preamble(source_lines[0], translated_lines[2], llm_client, model_configs)
        if result == "translation":
            logger.info("  Verifier: removed 2 preamble lines")
            return translated_lines[2:]

    logger.warning("  Verifier: could not remove preamble (all attempts failed)")
    return None  # Signal: needs retry


def strip_bilingual(
    tl_lines: List[str],
    llm_client,
    strip_model_configs,
) -> List[str]:
    """Strip source-language lines from bilingual output using a cheap model.

    When the translation model outputs bilingual (source + translation interleaved),
    use a cheap model to keep only the target-language lines.
    """
    text = "\n".join(tl_lines)
    try:
        result = llm_client.generate(
            prompt=f"{BILINGUAL_STRIP_PROMPT}\n\n{text}",
            model_configs=strip_model_configs,
            operation_name="Verifier bilingual strip",
        )
        stripped = [l for l in result.splitlines() if l.strip()]
        if not stripped:
            logger.warning("  Verifier: bilingual strip returned empty, keeping original")
            return tl_lines
        return stripped
    except Exception as e:
        logger.warning(f"  Verifier: bilingual strip failed ({e}), keeping original")
        return tl_lines


def find_hallucination_boundary(
    source_lines: List[str],
    translated_lines: List[str],
    llm_client,
    model_configs,
    window_size: int = 5,
    embedding_provider: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
    hallucination_threshold: float = 0.75,
) -> int:
    """Binary search for where hallucination starts.

    Returns the last good line index (conservative: first line of last good window).
    """
    if len(translated_lines) < window_size:
        return 0

    lo = 0
    hi = len(translated_lines) - window_size
    last_good = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        src_end = min(mid + window_size, len(source_lines))
        src_window = source_lines[mid:src_end]
        tl_window = translated_lines[mid:mid + window_size]

        # If source window is shorter than translated window, pad check
        if len(src_window) < len(tl_window):
            # Past the end of source — this position shouldn't be checked
            hi = mid - 1
            continue

        result = _check_alignment(
            src_window, tl_window, llm_client, model_configs,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            hallucination_threshold=hallucination_threshold,
        )
        logger.debug(f"  Verifier: binary search pos={mid}, result={result}")

        if result != "D":
            last_good = mid
            lo = mid + 1
        else:
            hi = mid - 1

    logger.info(f"  Verifier: hallucination boundary at line {last_good} "
                f"(keeping {last_good + 1}/{len(translated_lines)} lines)")
    return last_good


def verify_translation(
    source_text: str,
    translated_text: str,
    llm_client,
    model_configs,
    is_first_chunk: bool = True,
    embedding_provider: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
    hallucination_threshold: float = 0.75,
    strip_model_configs: Optional[list] = None,
) -> Tuple[Optional[str], str]:
    """Verify and fix translated text.

    Args:
        source_text: Full source text.
        translated_text: Raw translated text to verify.
        llm_client: LLMClient for verification calls.
        model_configs: Model configs for verification calls.
        is_first_chunk: Whether this is the first chunk (run preamble check).
        embedding_provider: Provider name for embedding (e.g. "gemini"). None = LLM only.
        embedding_model: Embedding model name.
        hallucination_threshold: Cosine similarity threshold for hallucination detection.
        strip_model_configs: Model configs for bilingual stripping (cheap model). None = retry instead.

    Returns:
        (fixed_text, action) where action is:
        - "complete": translation is good, done
        - "continue": translation is truncated, need continuation
        - "retry": translation is fundamentally broken, retry from scratch
    """
    src_lines = [l for l in source_text.splitlines() if l.strip()]
    tl_lines = [l for l in translated_text.splitlines() if l.strip()]


    if not tl_lines:
        return None, "retry"

    # Step 1: Preamble check (first chunk only)
    if is_first_chunk:
        cleaned = remove_preamble(src_lines, tl_lines, llm_client, model_configs)
        if cleaned is None:
            return None, "retry"
        tl_lines = cleaned

    # Step 2: Tail check
    n = len(tl_lines)
    window = min(5, n, len(src_lines))

    if window < 2:
        # Too short to check meaningfully
        if n < len(src_lines) - LINE_COUNT_TOLERANCE:
            return "\n".join(tl_lines), "continue"
        return "\n".join(tl_lines), "complete"

    tail_result = _check_alignment(
        src_lines[n - window:n],
        tl_lines[n - window:n],
        llm_client,
        model_configs,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        hallucination_threshold=hallucination_threshold,
    )

    if tail_result != "D":
        # Tail is valid translation
        if abs(n - len(src_lines)) <= LINE_COUNT_TOLERANCE:
            logger.info(f"  Verifier: complete ({n} vs {len(src_lines)} source lines, tail={tail_result})")
            return "\n".join(tl_lines), "complete"
        elif n < len(src_lines) - LINE_COUNT_TOLERANCE:
            logger.info(f"  Verifier: truncated ({n} vs {len(src_lines)} source lines, tail={tail_result})")
            return "\n".join(tl_lines), "continue"
        else:
            # Too many lines — likely bilingual output or duplication
            if n >= len(src_lines) * 1.8 and strip_model_configs:
                # Bilingual output: use cheap model to strip source lines
                logger.info(f"  Verifier: bilingual detected ({n} vs {len(src_lines)} lines, ratio={n/len(src_lines):.1f}x), stripping")
                stripped = strip_bilingual(tl_lines, llm_client, strip_model_configs)
                n_stripped = len(stripped)
                logger.info(f"  Verifier: stripped {n} → {n_stripped} lines")
                # Re-check with stripped lines (standard verification)
                if abs(n_stripped - len(src_lines)) <= LINE_COUNT_TOLERANCE:
                    # Re-verify tail alignment on stripped output
                    sw = min(5, n_stripped, len(src_lines))
                    if sw >= 2:
                        tail2 = _check_alignment(
                            src_lines[n_stripped - sw:n_stripped],
                            stripped[n_stripped - sw:n_stripped],
                            llm_client, model_configs,
                            embedding_provider=embedding_provider,
                            embedding_model=embedding_model,
                            hallucination_threshold=hallucination_threshold,
                        )
                        if tail2 != "D":
                            logger.info(f"  Verifier: complete after strip ({n_stripped} vs {len(src_lines)}, tail={tail2})")
                            return "\n".join(stripped), "complete"
                        logger.warning(f"  Verifier: tail hallucination after strip (tail={tail2}), retry")
                        return "\n".join(stripped), "retry"
                    return "\n".join(stripped), "complete"
                elif n_stripped < len(src_lines) - LINE_COUNT_TOLERANCE:
                    logger.info(f"  Verifier: truncated after strip ({n_stripped} vs {len(src_lines)})")
                    return "\n".join(stripped), "continue"
                else:
                    logger.warning(f"  Verifier: still too many after strip ({n_stripped} vs {len(src_lines)}), retry")
                    return "\n".join(stripped), "retry"
            else:
                logger.warning(f"  Verifier: too many lines ({n} vs {len(src_lines)}, +{n - len(src_lines)}), retry")
                return "\n".join(tl_lines), "retry"
    else:
        # Tail is hallucination — binary search
        logger.warning(f"  Verifier: hallucination detected at tail ({n} lines, tail={tail_result})")
        boundary = find_hallucination_boundary(
            src_lines, tl_lines, llm_client, model_configs,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            hallucination_threshold=hallucination_threshold,
        )
        truncated = tl_lines[:boundary + 1]
        logger.info(f"  Verifier: truncated to {len(truncated)} lines")
        if len(truncated) < len(src_lines) - LINE_COUNT_TOLERANCE:
            return "\n".join(truncated), "continue"
        else:
            return "\n".join(truncated), "complete"
