"""Shared validators for compressed HTML translation units."""

import re
from typing import List


def nonempty_lines(text: str) -> List[str]:
    """Return non-empty logical translation lines."""
    return [line for line in text.splitlines() if line.strip()]


def tag_sequence(text: str) -> List[str]:
    """Extract the simple HTML tag sequence used by compressed validation."""
    return [tag.lower() for tag in re.findall(r'<(/?[a-zA-Z0-9]+)', text)]


def tag_mismatch_count(source_lines: List[str], translated_lines: List[str]) -> int:
    """Count tag-sequence mismatches over aligned source/translation lines."""
    return sum(
        1
        for source, translated in zip(source_lines, translated_lines)
        if tag_sequence(source) != tag_sequence(translated)
    )

