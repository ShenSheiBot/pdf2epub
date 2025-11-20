"""
Content splitting utilities for markdown processors.

This module provides content splitting for long texts that exceed model token limits.
LLM-based splitting has been removed in favor of simpler, more reliable strategies.
"""

import regex
from typing import List, Optional, Tuple
from loguru import logger
import tiktoken

from pdf2epub.processors.utils.splitter_strategies import (
    ContentSplitter,
    SimpleSplitter,
    MarkdownStructureSplitter,
)

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def fuzzy_find_sentence(
    haystack: str,
    needle: str,
    max_edits: int = 3,
) -> Optional[Tuple[int, int, str]]:
    """
    Find a sentence in text with fuzzy matching, allowing for small differences.

    Args:
        haystack: The text to search in
        needle: The sentence to find
        max_edits: Maximum number of character edits allowed

    Returns:
        Tuple of (start_pos, end_pos, matched_text) or None if not found
    """
    # First try exact match
    exact_pos = haystack.find(needle)
    if exact_pos != -1:
        return (exact_pos, exact_pos + len(needle), needle)

    # Try fuzzy match with regex library
    try:
        # Allow up to max_edits character differences
        pattern = f"(?b)({regex.escape(needle)}){{e<={max_edits}}}"
        match = regex.search(pattern, haystack)
        if match:
            return (match.start(), match.end(), match.group(0))
    except Exception as e:
        logger.debug(f"Fuzzy matching failed: {e}")

    # Try to find with common escape variations
    variations = [
        needle.replace("&", r"\\") ,  # Escaped ampersand
        needle.replace(r"\\&", "&"),  # Unescaped ampersand
        needle.replace('"', r'\"'),  # Escaped quotes
        needle.replace(r'\"', '"'),  # Unescaped quotes
        needle.replace("'", r"'\'"),  # Escaped single quotes
        needle.replace(r"'\"", "'"),  # Unescaped single quotes
    ]

    for variant in variations:
        pos = haystack.find(variant)
        if pos != -1:
            return (pos, pos + len(variant), variant)

    return None


def get_splitter(strategy: str = "markdown") -> ContentSplitter:
    """
    Factory function to get a content splitter.

    Args:
        strategy: The splitting strategy to use:
            - "markdown": MarkdownStructureSplitter (markdown-aware, default)
            - "simple": SimpleSplitter (paragraph-based fallback)

    Returns:
        A content splitter instance.
    """
    if strategy == "simple":
        return SimpleSplitter()
    else:
        # Default to markdown-aware splitting
        return MarkdownStructureSplitter()


def split_content(
    content: str,
    max_tokens: int,
    llm_client=None,  # Deprecated, kept for backward compatibility
    model_configs=None,  # Deprecated, kept for backward compatibility
    strategy: str = "markdown",
) -> List[str]:
    """
    Splits content using a specified strategy.

    Args:
        content: The content to split.
        max_tokens: The maximum number of tokens per part.
        llm_client: Deprecated, ignored. Kept for backward compatibility.
        model_configs: Deprecated, ignored. Kept for backward compatibility.
        strategy: The splitting strategy to use:
            - "markdown": MarkdownStructureSplitter (markdown-aware, default)
            - "simple": SimpleSplitter (paragraph-based fallback)
            - "auto", "general", "japanese", "academic": All mapped to "markdown"

    Returns:
        A list of content parts.
    """
    # Map old LLM strategies to markdown splitter (structure-aware default)
    if strategy in ("auto", "general", "japanese", "academic", "paragraph"):
        if strategy not in ("markdown", "auto"):
            logger.debug(f"Strategy '{strategy}' mapped to 'markdown' (LLM splitting removed)")
        strategy = "markdown"

    # Log deprecation warning if llm_client is passed
    if llm_client is not None:
        logger.debug("llm_client parameter is deprecated and ignored")

    splitter = get_splitter(strategy)
    return splitter.split(content, max_tokens)
