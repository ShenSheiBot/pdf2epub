"""
HTML Content Splitter.

Splits HTML content at valid tag boundaries for translation processing.
"""

import re
from typing import List, Tuple, Optional
from loguru import logger
import tiktoken

from pdf2epub.processors.utils.splitter_strategies import ContentSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")


class HTMLSplitter(ContentSplitter):
    """
    Split HTML content at valid tag boundaries.

    Handles edge cases:
    - Files with only <br> tags (no block elements)
    - Deeply nested structures
    - Content without any splitting points

    Priority for split points:
    1. End of block-level tags (</p>, </div>, </section>, etc.)
    2. <br/> or <br> tags
    3. Forced split at text boundaries (last resort)
    """

    # Block-level tags where it's safe to split after
    BLOCK_TAGS = {
        'p', 'div', 'section', 'article', 'aside', 'header', 'footer',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        'blockquote', 'pre', 'figure', 'figcaption',
        'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
        'nav', 'main', 'address', 'form', 'fieldset', 'legend',
        'details', 'summary', 'dialog',
    }

    # Self-closing tags that are safe split points
    SPLIT_SAFE_TAGS = {'br', 'hr'}

    # Pattern to find closing block tags and br/hr
    SPLIT_PATTERN = re.compile(
        r'(</(?:' + '|'.join(BLOCK_TAGS) + r')>|<(?:br|hr)\s*/?>)',
        re.IGNORECASE
    )

    def __init__(self, min_part_tokens: int = 200):
        """
        Initialize the splitter.

        Args:
            min_part_tokens: Minimum tokens per part (avoid tiny fragments)
        """
        self.min_part_tokens = min_part_tokens

    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Split HTML content into parts under max_tokens.

        Args:
            content: Raw HTML/XHTML content
            max_tokens: Maximum tokens per part

        Returns:
            List of HTML fragments
        """
        total_tokens = len(tokenizer.encode(content))

        # If already under limit, return as-is
        if total_tokens <= max_tokens:
            return [content]

        # Find all valid split positions
        split_positions = self._find_split_positions(content)

        if not split_positions:
            # No valid split points found, try force splitting
            logger.warning("No valid HTML split points found, forcing text-based split")
            return self._force_split(content, max_tokens)

        # Split at positions that keep parts under max_tokens
        parts = self._split_at_positions(content, split_positions, max_tokens)

        # Merge any undersized parts
        parts = self._merge_small_parts(parts)

        return parts

    def _find_split_positions(self, content: str) -> List[Tuple[int, str]]:
        """
        Find all valid positions to cut the HTML.

        Args:
            content: HTML string

        Returns:
            List of (position, tag_type) tuples where position is char index
            after the tag, tag_type is 'block' or 'br'
        """
        positions = []

        for match in self.SPLIT_PATTERN.finditer(content):
            tag_text = match.group(1).lower()
            end_pos = match.end()

            # Determine tag type
            if '<br' in tag_text or '<hr' in tag_text:
                tag_type = 'br'
            else:
                tag_type = 'block'

            positions.append((end_pos, tag_type))

        return positions

    def _split_at_positions(
        self,
        content: str,
        positions: List[Tuple[int, str]],
        max_tokens: int
    ) -> List[str]:
        """
        Split content at given positions, respecting max_tokens.

        Prioritizes block tag boundaries over br tags.
        """
        parts = []
        current_start = 0

        # Sort positions by preference (block first, then br)
        # But we need to process in order, so just use the list as-is
        # and track what's available

        i = 0
        while current_start < len(content):
            # Find the furthest position we can go while staying under max_tokens
            best_split = None

            for pos, tag_type in positions:
                if pos <= current_start:
                    continue

                chunk = content[current_start:pos]
                chunk_tokens = len(tokenizer.encode(chunk))

                if chunk_tokens <= max_tokens:
                    # This position works, prefer block tags
                    if best_split is None or tag_type == 'block':
                        best_split = pos
                elif best_split is not None:
                    # We've exceeded max_tokens, use the last valid position
                    break

            if best_split is None:
                # No valid split found in range, need to force split
                remaining = content[current_start:]
                if len(tokenizer.encode(remaining)) <= max_tokens:
                    # Remaining content fits
                    parts.append(remaining)
                    break
                else:
                    # Force split the remaining content
                    forced_parts = self._force_split(remaining, max_tokens)
                    parts.extend(forced_parts)
                    break
            else:
                parts.append(content[current_start:best_split])
                current_start = best_split

        return parts

    def _force_split(self, content: str, max_tokens: int) -> List[str]:
        """
        Force split content when no HTML split points are available.

        Tries to split at text boundaries (spaces, newlines) without
        breaking HTML tags.
        """
        parts = []
        remaining = content

        while remaining:
            tokens = tokenizer.encode(remaining)
            if len(tokens) <= max_tokens:
                parts.append(remaining)
                break

            # Find approximate character position for max_tokens
            # Start from a safe estimate and adjust
            approx_chars = int(len(remaining) * max_tokens / len(tokens))

            # Find a safe split point (not inside a tag)
            split_pos = self._find_safe_text_split(remaining, approx_chars)

            if split_pos <= 0 or split_pos >= len(remaining):
                # Can't find safe split, just take what we can
                logger.warning("Forcing split at arbitrary position")
                split_pos = approx_chars

            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:]

        return parts

    def _find_safe_text_split(self, content: str, target_pos: int) -> int:
        """
        Find a safe position to split text without breaking HTML tags.

        Looks for whitespace near the target position that's not inside a tag.
        """
        # Define search window
        window_start = max(0, target_pos - 500)
        window_end = min(len(content), target_pos + 100)

        # Find all whitespace positions in window
        whitespace_positions = []
        in_tag = False

        for i in range(window_start, window_end):
            char = content[i]
            if char == '<':
                in_tag = True
            elif char == '>':
                in_tag = False
            elif not in_tag and char in ' \n\r\t':
                whitespace_positions.append(i)

        if not whitespace_positions:
            # No safe whitespace found, look for > as tag boundary
            for i in range(target_pos, window_start, -1):
                if content[i] == '>':
                    return i + 1
            return target_pos

        # Find whitespace closest to target
        closest = min(whitespace_positions, key=lambda x: abs(x - target_pos))
        return closest + 1  # Split after the whitespace

    def _merge_small_parts(self, parts: List[str]) -> List[str]:
        """
        Merge parts that are smaller than min_part_tokens.
        """
        if len(parts) <= 1:
            return parts

        merged = []
        current = parts[0]

        for part in parts[1:]:
            current_tokens = len(tokenizer.encode(current))

            if current_tokens < self.min_part_tokens:
                # Merge with next part
                current = current + part
            else:
                merged.append(current)
                current = part

        # Don't forget the last part
        if current:
            merged.append(current)

        return merged

    def validate_fragment(self, fragment: str) -> bool:
        """
        Check if an HTML fragment is reasonably well-formed.

        Returns:
            True if fragment appears valid
        """
        # Count opening and closing tags
        # This is a simple heuristic, not full validation
        open_tags = len(re.findall(r'<(?!/)[^>]+>', fragment))
        close_tags = len(re.findall(r'</[^>]+>', fragment))

        # Allow some imbalance (fragments may be partial)
        return abs(open_tags - close_tags) <= 5


class CompressedSplitter(ContentSplitter):
    """
    Split compressed HTML content at line boundaries.

    Much simpler than HTMLSplitter since compressed format has one
    translation unit per line. Never splits in the middle of a line.
    """

    def __init__(self, min_part_tokens: int = 200):
        """
        Initialize the splitter.

        Args:
            min_part_tokens: Minimum tokens per part (avoid tiny fragments)
        """
        self.min_part_tokens = min_part_tokens

    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Split compressed content into parts under max_tokens.

        Args:
            content: Compressed content (one unit per line)
            max_tokens: Maximum tokens per part

        Returns:
            List of content parts (each preserving line structure)
        """
        total_tokens = len(tokenizer.encode(content))

        if total_tokens <= max_tokens:
            return [content]

        lines = content.split('\n')
        parts = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = len(tokenizer.encode(line + '\n'))

            if current_tokens + line_tokens > max_tokens and current_lines:
                # Save current part
                parts.append('\n'.join(current_lines))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Add remaining lines
        if current_lines:
            parts.append('\n'.join(current_lines))

        # Merge small parts
        parts = self._merge_small_parts(parts)

        return parts

    def _merge_small_parts(self, parts: List[str]) -> List[str]:
        """Merge parts smaller than min_part_tokens."""
        if len(parts) <= 1:
            return parts

        merged = []
        current = parts[0]

        for part in parts[1:]:
            current_tokens = len(tokenizer.encode(current))

            if current_tokens < self.min_part_tokens:
                current = current + '\n' + part
            else:
                merged.append(current)
                current = part

        if current:
            merged.append(current)

        return merged
