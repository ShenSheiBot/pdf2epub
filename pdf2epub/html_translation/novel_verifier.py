"""
Deterministic translation verifier for novel translation.

Replaces the agent-based verification with:
1. Preamble detection via LLM classification (translation/meta-comment)
2. Hallucination detection via A/B/C/D alignment check + binary search

All LLM calls are simple classification prompts (single letter/word response).
No agent, no sandbox, no tool use.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

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

# Tolerance for line count difference
LINE_COUNT_TOLERANCE = 5


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
                     llm_client, model_configs) -> str:
    """Check alignment between source and translated windows.

    Returns 'A', 'B', 'C', or 'D'.
    """
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


def find_hallucination_boundary(
    source_lines: List[str],
    translated_lines: List[str],
    llm_client,
    model_configs,
    window_size: int = 5,
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

        result = _check_alignment(src_window, tl_window, llm_client, model_configs)
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
) -> Tuple[Optional[str], str]:
    """Verify and fix translated text.

    Args:
        source_text: Full source text.
        translated_text: Raw translated text to verify.
        llm_client: LLMClient for verification calls.
        model_configs: Model configs for verification calls.
        is_first_chunk: Whether this is the first chunk (run preamble check).

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
            # More lines than source — accept anyway since tail is valid
            logger.info(f"  Verifier: complete with extra lines ({n} vs {len(src_lines)}, tail={tail_result})")
            return "\n".join(tl_lines), "complete"
    else:
        # Tail is hallucination — binary search
        logger.warning(f"  Verifier: hallucination detected at tail ({n} lines, tail={tail_result})")
        boundary = find_hallucination_boundary(src_lines, tl_lines, llm_client, model_configs)
        truncated = tl_lines[:boundary + 1]
        logger.info(f"  Verifier: truncated to {len(truncated)} lines")
        if len(truncated) < len(src_lines) - LINE_COUNT_TOLERANCE:
            return "\n".join(truncated), "continue"
        else:
            return "\n".join(truncated), "complete"
