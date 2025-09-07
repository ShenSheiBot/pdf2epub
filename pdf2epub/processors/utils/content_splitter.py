"""
Content splitting utilities for markdown processors.

This module provides intelligent content splitting for long texts,
particularly useful for processing large chapters that exceed model token limits.
"""

import json
import regex
from typing import List, Optional, Tuple, Dict
from loguru import logger
import tiktoken
from abc import ABC, abstractmethod

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


class BaseLLMSplitter(ContentSplitter):
    """Base class for LLM-based content splitters."""

    def __init__(self, llm_client, model_configs: Optional[List[Dict]] = None):
        self.llm_client = llm_client
        self.model_configs = model_configs or [
            {"provider": "gemini", "model": "gemini-1.5-flash", "max_retries": 2}
        ]

    @abstractmethod
    def get_prompt(self, content: str, max_tokens: int, num_parts: int) -> str:
        """
        Returns the prompt for the LLM.

        Args:
            content: The content to split.
            max_tokens: The maximum number of tokens per part.
            num_parts: The desired number of parts.

        Returns:
            The prompt string.
        """
        pass

    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Use LLM to intelligently split content at natural boundaries.

        Args:
            content: The content to split
            max_tokens: Maximum tokens per part (used as guideline)

        Returns:
            List of content parts
        """
        actual_tokens = len(tokenizer.encode(content))

        if actual_tokens <= max_tokens:
            return [content]

        num_parts = max(2, (actual_tokens // max_tokens) + 1)
        logger.info(
            f"Content has {actual_tokens:,} tokens, splitting into {num_parts} parts"
        )

        split_prompt = self.get_prompt(content, max_tokens, num_parts)
        split_prompt += f"\n\nHere is the content to analyze:\n\n{content}"

        try:
            response = self.llm_client.generate(
                prompt=split_prompt,
                model_configs=self.model_configs,
                operation_name="Split chapter",
            )
            if not response or not response.strip():
                logger.warning(
                    "Failed to get LLM split suggestions, falling back to simple split"
                )
                return SimpleSplitter().split(content, max_tokens)

            response = response.strip()
            json_str = None

            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                if response[start : start + 1] == "\n":
                    start += 1
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
            elif "[" in response:
                start = response.find("[")
                end = response.rfind("]")
                if start != -1 and end != -1 and end > start:
                    json_str = response[start : end + 1]

            if not json_str:
                json_str = response

            logger.debug(f"Parsing split response: {json_str[:200]}...")
            try:
                split_sentences = json.loads(json_str.strip())
                if not isinstance(split_sentences, list):
                    raise ValueError("Response is not a list")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse LLM split response: {e}, falling back to simple split"
                )
                return SimpleSplitter().split(content, max_tokens)

            parts = []
            split_positions = [0]

            for split_sentence in split_sentences:
                cleaned_sentence = split_sentence
                if cleaned_sentence.endswith("..."):
                    cleaned_sentence = cleaned_sentence[:-3].rstrip()
                if cleaned_sentence.endswith("…"):
                    cleaned_sentence = cleaned_sentence[:-1].rstrip()

                match_result = fuzzy_find_sentence(content, cleaned_sentence)

                if match_result:
                    split_pos = match_result[0]
                    split_positions.append(split_pos)
                    logger.debug(
                        f"Found split point at beginning of: '{match_result[2][:50]}...'"
                    )
                else:
                    logger.warning(
                        f"Could not find split sentence: '{split_sentence[:50]}...'"
                    )

            split_positions.append(len(content))

            for i in range(len(split_positions) - 1):
                part = content[split_positions[i] : split_positions[i + 1]].strip()
                if part:
                    # Ensure Notes sections use ### instead of ## to avoid appearing as chapters
                    part = regex.sub(r'^## (Notes|References|Bibliography)$', r'### \1', part, flags=regex.MULTILINE)
                    parts.append(part)

            if len(parts) < 2:
                logger.warning(
                    "LLM split resulted in too few parts, falling back to simple split"
                )
                return SimpleSplitter().split(content, max_tokens)

            for i, part in enumerate(parts, 1):
                part_tokens = len(tokenizer.encode(part))
                logger.info(f"Part {i}/{len(parts)}: {part_tokens:,} tokens")

            return parts

        except Exception as e:
            logger.warning(
                f"Intelligent split failed: {e}, falling back to simple split"
            )
            return SimpleSplitter().split(content, max_tokens)


class GeneralLLMSplitter(BaseLLMSplitter):
    """LLM splitter for general content."""

    def get_prompt(self, content: str, max_tokens: int, num_parts: int) -> str:
        total_tokens = len(tokenizer.encode(content))
        return f"""You are helping split a long document into smaller parts for processing.

The document has approximately {total_tokens:,} tokens. We need to split it into roughly {num_parts} parts.
Each part should ideally be under {max_tokens:,} tokens.

SPLITTING RULES:

1. **Maintain Semantic Coherence**:
   - Split at natural boundaries (section breaks, topic changes)
   - Keep related paragraphs together
   - Preserve the logical flow of ideas

2. **Respect Document Structure**:
   - Split at heading boundaries when possible (##, ###)
   - Keep lists and enumerations intact
   - Don't split in the middle of examples or code blocks
   - Notes sections should use ### (not ##) to avoid appearing as chapters

3. **Balance Part Sizes**:
   - Aim for roughly equal-sized parts
   - It's OK if some parts are larger to maintain coherence

Return a JSON array of the EXACT first sentences that mark the beginning of each new part (starting from the SECOND part).
The first part always starts at the beginning, so we only need markers for part 2 onwards.
Choose split points at natural breaks - typically the first sentence of a new section, new topic, or after a clear transition.

Example response:
["## Advanced Techniques", "Now that we understand the basics, let's explore more complex scenarios."]"""


class JapaneseLLMSplitter(BaseLLMSplitter):
    """LLM splitter for Japanese content."""

    def get_prompt(self, content: str, max_tokens: int, num_parts: int) -> str:
        total_tokens = len(tokenizer.encode(content))
        return f"""You are helping split a long Japanese novel/light novel chapter into smaller parts for processing.

The chapter has approximately {total_tokens:,} tokens. We'd prefer parts under {max_tokens:,} tokens. 

Return a JSON array of the EXACT first sentences that mark the beginning of each new part (starting from the SECOND part).
The first part always starts at the beginning, so we only need markers for part 2 onwards.

Each part should ideally start with a new section, scene change, or natural break point.
Ensure no sentences are split and important scenes remain intact."""


class AcademicLLMSplitter(BaseLLMSplitter):
    """LLM splitter for academic content."""

    def get_prompt(self, content: str, max_tokens: int, num_parts: int) -> str:
        total_tokens = len(tokenizer.encode(content))
        return f"""You are helping split a long academic chapter into smaller parts for processing.

The chapter has approximately {total_tokens:,} tokens. While we'd prefer parts under {max_tokens:,} tokens, 
the MOST IMPORTANT criteria are semantic completeness and avoiding citation conflicts.

CRITICAL SPLITTING RULES (in order of priority):

1. **Keep Citations with their Notes/References**:
   - IMPORTANT: Footnotes can appear in two ways:
     a) **Inline footnotes**: Definition appears immediately after citation in the text flow
     b) **End-of-section footnotes**: Definitions collected at the end under "Notes" or "References"
   - For inline footnotes: NEVER split between a citation and its nearby definition
   - For end-of-section footnotes: Keep the entire section WITH its Notes/References in the same part
   - If footnotes are inline, do NOT use them as split points - keep reading until you find a section boundary
   - Split BEFORE a new section starts (not after the previous one ends)

2. **No Duplicate Citations**:
   - Each footnote number (e.g., [^1], $^1$, ¹) must appear ONLY in one part
   - Both the citation [^1] and its definition [^1]: must be in the SAME part
   - Never split between a citation and its corresponding footnote definition
   - If footnotes are inline (definition immediately follows citation), keep them together as a unit
   - If you see patterns like [^1], [^2], [^3] in text, ensure ALL of them and their definitions stay together

3. **Section Integrity**:
   - Split at major section boundaries (look for ## or ### headings)
   - Keep entire sections together when possible
   - If footnotes are inline within a section, the entire section must stay together
   - Only split at points where NO citations span across the boundary

4. **No Cross-References Between Parts**:
   - A citation in one part should NEVER refer to a footnote definition in another part
   - For inline footnotes, this means keeping the citation and its immediate definition together
   - For end-of-section footnotes, this means keeping the entire section with its footnotes
   - Never have orphaned citations or orphaned footnote definitions

5. **Token Limits** (lowest priority):
   - Aim for parts under {max_tokens:,} tokens if possible
   - But it's OK to exceed this if needed to maintain semantic integrity
   - Aim for roughly {num_parts} parts, but adjust based on content structure

Scan the chapter and identify:
- Whether footnotes are inline (definitions immediately after citations) or collected at section ends
- If inline: Find section boundaries where no footnotes are actively being defined
- If at section ends: Identify which sections have citations and where their Notes/References are
- Natural boundaries where no citations span across
- Major section boundaries that don't break citation-reference pairs

IMPORTANT: If you detect inline footnotes (e.g., [^1] followed shortly by [^1]: definition in the main text flow),
do NOT split near these footnotes. Instead, find major section breaks or topic changes as split points.

Return a JSON array of the EXACT first sentences that mark the beginning of each new part (starting from the SECOND part).
The first part always starts at the beginning, so we only need markers for part 2 onwards.
Choose split points that respect the above priorities - typically the first sentence of a new section or major topic.

Example response:
["## Chapter 3: Advanced Methods", "The theoretical framework we have established leads us to consider..."]"""


def get_splitter(
    strategy: str,
    llm_client,
    model_configs: Optional[List[Dict]] = None,
) -> ContentSplitter:
    """
    Factory function to get a content splitter.

    Args:
        strategy: The splitting strategy to use.
        llm_client: The LLM client.
        model_configs: The model configurations.

    Returns:
        A content splitter instance.
    """
    if strategy == "simple":
        return MarkdownStructureSplitter()
    elif strategy == "general":
        return GeneralLLMSplitter(llm_client, model_configs)
    elif strategy == "japanese":
        return JapaneseLLMSplitter(llm_client, model_configs)
    elif strategy == "academic":
        return AcademicLLMSplitter(llm_client, model_configs)
    else:
        raise ValueError(f"Unknown splitter strategy: {strategy}")


def split_content(
    content: str,
    max_tokens: int,
    llm_client,
    model_configs: Optional[List[Dict]] = None,
    strategy: str = "auto",
) -> List[str]:
    """
    Splits content using a specified strategy.

    Args:
        content: The content to split.
        max_tokens: The maximum number of tokens per part.
        llm_client: The LLM client.
        model_configs: The model configurations.
        strategy: The splitting strategy to use.

    Returns:
        A list of content parts.
    """
    if strategy == "auto":
        import re

        japanese_chars = re.findall(
            r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", content[:5000]
        )
        if len(japanese_chars) > 500:
            strategy = "japanese"
            logger.info("Auto-detected content type: Japanese")
        else:
            academic_indicators = [
                r"[\^\d+]",
                r"References\s*\n",
                r"Bibliography\s*\n",
            ]
            if any(
                re.search(pattern, content[:5000]) for pattern in academic_indicators
            ):
                strategy = "academic"
                logger.info("Auto-detected content type: Academic")
            else:
                strategy = "general"
                logger.info("Auto-detected content type: General")

    splitter = get_splitter(strategy, llm_client, model_configs)
    return splitter.split(content, max_tokens)
