"""
Fast image restoration using difflib position mapping.
Optimized replacement for the slow fuzzy-search approach.
"""

import re
import bisect
from typing import List, Tuple, Optional, NamedTuple

# Use rapidfuzz for faster diff (C++ implementation, 10-100x faster than difflib)
try:
    from rapidfuzz.distance import Indel as _RapidFuzzIndel
    _USE_RAPIDFUZZ = True
except ImportError:
    _USE_RAPIDFUZZ = False
    import difflib


class MatchBlock(NamedTuple):
    """Compatible with difflib.Match."""
    a: int  # start in sequence a
    b: int  # start in sequence b
    size: int  # length of match


def get_matching_blocks(a: str, b: str) -> List[MatchBlock]:
    """
    Get matching blocks between two strings.
    Uses rapidfuzz if available (10-100x faster), otherwise falls back to difflib.
    """
    if _USE_RAPIDFUZZ:
        # Extract 'equal' opcodes and convert to matching blocks
        blocks = []
        for op in _RapidFuzzIndel.opcodes(a, b):
            if op.tag == 'equal':
                blocks.append(MatchBlock(op.src_start, op.dest_start, op.src_end - op.src_start))
        # Add terminal block (difflib convention)
        blocks.append(MatchBlock(len(a), len(b), 0))
        return blocks
    else:
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        return [MatchBlock(m.a, m.b, m.size) for m in sm.get_matching_blocks()]
from loguru import logger

# Compile regex once for performance
# Markdown image pattern: ![alt](src)
MD_IMG_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
# HTML image pattern: <img src="..." ... /> or <img src="..." ...>
# Also match surrounding div if present
HTML_IMG_PATTERN = re.compile(
    r'(?:<div[^>]*>)?\s*<img\s+[^>]*src="([^"]+)"[^>]*/?\s*>\s*(?:</div>)?',
    re.IGNORECASE
)


def extract_images_from_markdown(content: str) -> List[Tuple[str, int, int]]:
    """Extract all image references from markdown and HTML content."""
    results = []

    # Extract markdown images
    for m in MD_IMG_PATTERN.finditer(content):
        results.append((m.group(0), m.start(), m.end()))

    # Extract HTML images
    for m in HTML_IMG_PATTERN.finditer(content):
        results.append((m.group(0), m.start(), m.end()))

    # Sort by position
    results.sort(key=lambda x: x[1])
    return results


def _prefix_removed_lengths(
    spans: List[Tuple[int, int]],
) -> Tuple[List[int], List[int]]:
    """
    Given sorted spans [(start, end), ...], return:
      - ends: list of end indices
      - prefix_lens: prefix sum of removed lengths up to each span (inclusive)
    """
    ends, prefix = [], []
    total = 0
    for s, e in spans:
        L = e - s
        ends.append(e)
        total += L
        prefix.append(total)
    return ends, prefix


def _removed_chars_before(idx: int, ends: List[int], prefix: List[int]) -> int:
    """How many chars were removed in original before 'idx'?"""
    # Count spans whose end <= idx
    k = bisect.bisect_right(ends, idx)
    return 0 if k == 0 else prefix[k - 1]


def _remove_images(text: str, images: List[Tuple[str, int, int]]) -> str:
    """Return text with all image segments removed (fast concat)."""
    if not images:
        return text
    parts = []
    prev = 0
    for _, s, e in images:
        parts.append(text[prev:s])
        prev = e
    parts.append(text[prev:])
    return "".join(parts)


def _nearest_mapped_pos(original_noimg_pos: int, blocks: List[MatchBlock]) -> int:
    """
    Map a char index from original_no_images -> polished using matching blocks.
    If it's inside a block: use direct offset.
    If it's between blocks: snap to the end of the last block (good heuristic).
    """
    # Binary search by original index i in blocks
    lo, hi = 0, len(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        b = blocks[mid]
        if original_noimg_pos < b.a:
            hi = mid
        elif original_noimg_pos >= b.a + b.size:
            lo = mid + 1
        else:
            # Inside block
            return blocks[mid].b + (original_noimg_pos - blocks[mid].a)

    # Not inside a block; use the closest preceding block
    idx = lo - 1
    if idx >= 0:
        b = blocks[idx]
        return b.b + b.size
    # Otherwise before the first block: map to start
    return 0


def _local_exact_probe(polished: str, before_ctx: str, after_ctx: str, center: int = -1) -> Optional[int]:
    """
    Try to refine insertion using tiny local exact matches:
    - if we can find 'before_ctx' -> insert after it
    - else if we can find 'after_ctx' -> insert before it

    When center >= 0, find the match closest to center (the difflib-mapped position).
    This prevents picking the wrong occurrence when text appears multiple times.

    Returns polished index or None.
    """
    if before_ctx:
        # Find all occurrences and pick the one closest to center
        matches = []
        start = 0
        while True:
            j = polished.find(before_ctx, start)
            if j == -1:
                break
            insert_pos = j + len(before_ctx)
            matches.append(insert_pos)
            start = j + 1

        if matches:
            if center >= 0:
                # Pick the match closest to center
                return min(matches, key=lambda p: abs(p - center))
            else:
                # Fallback: pick the last one (old behavior)
                return matches[-1]

    if after_ctx:
        # Find all occurrences and pick the one closest to center
        matches = []
        start = 0
        while True:
            j = polished.find(after_ctx, start)
            if j == -1:
                break
            matches.append(j)
            start = j + 1

        if matches:
            if center >= 0:
                return min(matches, key=lambda p: abs(p - center))
            else:
                return matches[0]

    return None


def clean_context_for_matching(text: str) -> str:
    """
    Clean context text by removing elements that are likely to be changed during polishing.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines, headers, and separators
        if (
            not line
            or line.startswith("#")
            or line == "---"
            or line.startswith("```")
            or all(c in "-=_*" for c in line)
        ):  # Skip separator lines
            continue
        cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


def restore_lost_images_fast(
    original_content: str, polished_content: str, max_edits: int = 10
) -> str:
    """
    Fast path: map original->polished via difflib on texts with images removed.
    Then insert missing images using mapped offsets. Only if mapping is weak,
    probe locally with tiny (exact) context windows. No global fuzzy scans.

    Args:
        original_content: Original OCR content with images
        polished_content: Polished content that may be missing images
        max_edits: Unused in this fast implementation (kept for API compatibility)

    Returns:
        Polished content with restored images
    """
    original_images = extract_images_from_markdown(original_content)
    polished_images = extract_images_from_markdown(polished_content)

    orig_set = {img for img, _, _ in original_images}
    pol_set = {img for img, _, _ in polished_images}
    lost = list(orig_set - pol_set)

    if not lost:
        return polished_content

    logger.info(f"Detected {len(lost)} lost images, attempting fast restoration...")

    # Sort original image spans by start; build helpers
    original_images_sorted = sorted(original_images, key=lambda t: t[1])
    spans = [(s, e) for _, s, e in original_images_sorted]
    ends, prefix = _prefix_removed_lengths(spans)

    # Build texts without images
    original_noimg = _remove_images(original_content, original_images_sorted)
    polished_noimg = _remove_images(polished_content, polished_images)

    # Diff once and reuse (rapidfuzz is 10-100x faster than difflib)
    blocks = get_matching_blocks(original_noimg, polished_noimg)

    # Index: image markdown -> (start,end) in original
    pos_by_img = {img: (s, e) for img, s, e in original_images_sorted}

    # Plan insertions: list of (polished_offset, img_markdown)
    planned: List[Tuple[int, str]] = []

    for img_md in lost:
        s, e = pos_by_img[img_md]

        # Check if image is at the end of original content
        # (within 50 chars of the end, accounting for trailing whitespace)
        is_at_end = (len(original_content) - e) < 50

        if is_at_end:
            # Image is at the end of original - put it at the end of polished
            # This avoids difflib mapping errors when LLM modifies/removes trailing content
            mapped = len(polished_content)
            logger.info(f"Image at end of original, placing at end of polished")
        else:
            # Prefer to map just *after* the image (so it appears in roughly that spot)
            # Map original index (image end) into "no-image" space
            noimg_pos = e - _removed_chars_before(e, ends, prefix)

            # Primary: mapped position from difflib blocks
            mapped = _nearest_mapped_pos(noimg_pos, blocks)

            # Optional micro refinement: use tiny local exact contexts near the image
            # Build 60-char context (exact search is very fast compared to fuzzy)
            ctx_window = 60
            before_raw = original_content[max(0, s - 180) : s]
            after_raw = original_content[e : min(len(original_content), e + 180)]

            # Clean and pick a short slice from the tail/head
            before_ctx = clean_context_for_matching(before_raw)
            after_ctx = clean_context_for_matching(after_raw)
            before_ctx = before_ctx[-ctx_window:] if before_ctx else ""
            after_ctx = after_ctx[:ctx_window] if after_ctx else ""

            # Try to refine within the neighborhood of `mapped`
            # Search in a small window around mapped to keep it fast
            local_radius = 800  # chars; tweak as needed
            left = max(0, mapped - local_radius)
            right = min(len(polished_content), mapped + local_radius)

            # Center is where difflib mapped the position, relative to window start
            window_center = mapped - left
            local_insert = _local_exact_probe(
                polished_content[left:right], before_ctx, after_ctx, center=window_center
            )
            if local_insert is not None:
                mapped = left + local_insert  # relocate within the window
                logger.info(f"Refined position for {img_md} using local context")
            else:
                logger.debug(f"Using difflib mapping for {img_md}")

        planned.append((mapped, img_md))

    # Apply all insertions right-to-left to avoid index shifts
    planned.sort(key=lambda t: t[0], reverse=True)

    out = polished_content
    for pos, img_md in planned:
        # Insert with spacing
        insert_text = f"\n\n{img_md}\n\n"
        out = out[:pos] + insert_text + out[pos:]
        logger.info(f"Restored image: {img_md} at position {pos}")

    # Remove any remaining [illustration] markers
    out = re.sub(r"\[illustration\]", "", out)
    # Clean up any resulting excessive blank lines
    out = re.sub(r"\n{4,}", "\n\n\n", out)

    return out


# Alias for backward compatibility
restore_lost_images = restore_lost_images_fast


# For compatibility with old API
def extract_images(text: str) -> List[Tuple[str, str]]:
    """
    Extract all image references from markdown text.
    Old API compatibility - returns list of (alt_text, image_path) tuples.
    """
    # Pattern to match markdown images: ![alt_text](path)
    pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
    matches = re.findall(pattern, text)
    return matches


def find_best_insertion_point(
    text: str, context_before: str, context_after: str, image_markdown: str
) -> int:
    """
    Old API compatibility - kept for backward compatibility.
    Uses the fast restoration approach internally.
    """
    # Try exact match first
    if context_before:
        j = text.rfind(context_before)
        if j != -1:
            return j + len(context_before)
    if context_after:
        j = text.find(context_after)
        if j != -1:
            return j
    return -1
