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
from pdf2epub.processors.utils.document_parser import (
    analyze_document_structure,
    find_split_positions,
    find_split_positions_by_indices,
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
    def get_prompt(self, structural_map: List[Dict], max_tokens: int, num_parts: int) -> str:
        """
        Returns the prompt for the LLM.

        Args:
            structural_map: The structural analysis of the document.
            max_tokens: The maximum number of tokens per part.
            num_parts: The desired number of parts.

        Returns:
            The prompt string.
        """
        pass
    
    @abstractmethod
    def get_content_type(self) -> str:
        """
        Returns the content type for document analysis.
        
        Returns:
            The content type string ("general", "academic", "japanese")
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
        
        # Analyze document structure locally
        content_type = self.get_content_type()
        structural_map = analyze_document_structure(content, content_type)
        
        # Create lightweight prompt with structural map
        split_prompt = self.get_prompt(structural_map, max_tokens, num_parts)
        split_prompt += f"\n\nDocument structure:\n{json.dumps(structural_map, indent=2)}"

        # --- TEMP MODIFICATION: Record the prompt ---
        with open("last_split_prompt.txt", "w", encoding="utf-8") as f:
            f.write(split_prompt)
        # --- END TEMP MODIFICATION ---

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
            
            # Try to parse as indices first (new format)
            split_indices = None
            try:
                parsed_response = json.loads(json_str.strip())
                if not isinstance(parsed_response, list):
                    raise ValueError("Response is not a list")
                
                # Check if response contains integers (indices) or strings (old format)
                if parsed_response and all(isinstance(x, int) for x in parsed_response):
                    # New format: section indices
                    split_indices = parsed_response
                    logger.debug(f"Using section indices: {split_indices}")
                else:
                    # Old format: string markers (for backward compatibility)
                    split_markers = [str(marker) for marker in parsed_response if marker]
                    logger.debug(f"Using string markers (legacy): {len(split_markers)} markers")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse LLM split response: {e}, falling back to simple split"
                )
                return SimpleSplitter().split(content, max_tokens)

            # Find split positions using either indices or markers
            if split_indices is not None:
                # Use the new index-based approach
                split_positions = find_split_positions_by_indices(
                    content, split_indices, structural_map
                )
            else:
                # Use the old marker-based approach (backward compatibility)
                split_positions = find_split_positions(
                    content, split_markers, structural_map
                )

            parts = []
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

            # Validate that no part exceeds the max_tokens limit (with 25% tolerance)
            for i, part in enumerate(parts, 1):
                part_tokens = len(tokenizer.encode(part))
                logger.info(f"Part {i}/{len(parts)}: {part_tokens:,} tokens")
                if part_tokens > max_tokens * 1.25:  # Allow 25% tolerance
                    logger.warning(
                        f"Part {i} has {part_tokens:,} tokens, which exceeds the limit of {max_tokens:,} by more than 25%. "
                        "Falling back to simple split."
                    )
                    return SimpleSplitter().split(content, max_tokens)

            return parts

        except Exception as e:
            logger.warning(
                f"Intelligent split failed: {e}, falling back to simple split"
            )
            return SimpleSplitter().split(content, max_tokens)


class GeneralLLMSplitter(BaseLLMSplitter):
    """LLM splitter for general content."""
    
    def get_content_type(self) -> str:
        return "general"

    def get_prompt(self, structural_map: List[Dict], max_tokens: int, num_parts: int) -> str:
        total_tokens = structural_map[-1]["cumulative_tokens"] if structural_map else 0
        return f"""You are a document planner. Based on the following structural analysis of a document, 
group the sections into parts that are under {max_tokens:,} tokens each.

The document has {total_tokens:,} tokens total. Aim for approximately {num_parts} parts.

RULES:
1. Target size: Each part should be under {max_tokens:,} tokens
   - Strongly prefer staying under {max_tokens:,} tokens
   - Acceptable if slightly over (up to {int(max_tokens * 1.25):,} tokens) when necessary
   - Use cumulative_tokens field to calculate part sizes
2. Keep related sections together when possible
3. Split at natural boundaries (section headings)
4. Create the fewest parts necessary

Return a JSON array of section indices where each new part should begin (starting from the SECOND part).
The first part always starts at section 0.

Example: [5, 12, 18] means part 2 starts at section index 5, part 3 at section 12, part 4 at section 18.

IMPORTANT: Use the cumulative_tokens field. Aim for {max_tokens:,} tokens per part."""


class JapaneseLLMSplitter(BaseLLMSplitter):
    """LLM splitter for Japanese content."""
    
    def get_content_type(self) -> str:
        return "japanese"

    def get_prompt(self, structural_map: List[Dict], max_tokens: int, num_parts: int) -> str:
        total_tokens = structural_map[-1]["cumulative_tokens"] if structural_map else 0
        return f"""You are a document planner for Japanese content. Based on the following structural analysis, 
group the sections into parts that are under {max_tokens:,} tokens each.

The document has {total_tokens:,} tokens total.

RULES:
1. Target size: Each part should be under {max_tokens:,} tokens
   - Strongly prefer staying under {max_tokens:,} tokens
   - Acceptable if slightly over (up to {int(max_tokens * 1.25):,} tokens) when necessary
   - Use cumulative_tokens field to calculate part sizes
2. Keep scene breaks and natural story flow intact
3. Split at section boundaries when possible
4. Create the fewest parts necessary

Return a JSON array of section indices where each new part should begin (starting from the SECOND part).

Example: [5, 12] means part 2 starts at section 5, part 3 at section 12.

IMPORTANT: Use the cumulative_tokens field. Aim for {max_tokens:,} tokens per part."""


class AcademicLLMSplitter(BaseLLMSplitter):
    """LLM splitter for academic content."""
    
    def get_content_type(self) -> str:
        return "academic"

    def get_prompt(self, structural_map: List[Dict], max_tokens: int, num_parts: int) -> str:
        total_tokens = structural_map[-1]["cumulative_tokens"] if structural_map else 0
        ideal_part_size = total_tokens // num_parts
        return f"""You are a document planner for academic texts. Below is a structural analysis of a chapter.

The document has {total_tokens:,} tokens total and should be split into {num_parts} parts.
The ideal size for each part is approximately {ideal_part_size:,} tokens.

CRITICAL RULES:

1. **Token Limit** (HIGHEST PRIORITY):
   - Target: Keep each part under {max_tokens:,} tokens
   - Hard limit: Never exceed {int(max_tokens * 1.25):,} tokens (25% tolerance)
   - Use cumulative_tokens to calculate part sizes
   - Example: If you split at section 10 (cumulative: 25,000) and section 15 (cumulative: 45,000), 
     the part contains 45,000 - 25,000 = 20,000 tokens

2. **Balanced Parts**:
   - Create parts close to the ideal size ({ideal_part_size:,} tokens)
   - It's better to have slightly larger parts than very unbalanced ones
   - Avoid creating tiny parts (< {int(ideal_part_size * 0.5):,} tokens) unless necessary

3. **Citation Integrity**:
   - Check citations_formatted field for citation ranges (e.g., "1-10, 15, 20-25")
   - Try to keep sections with many citations together when possible
   - Footnote definitions are less common in this document

4. **Section Order**:
   - Sections must remain in order

Return a JSON array of section indices where new parts begin (starting from the SECOND part).

Example: [7, 14, 21] means:
- Part 1: sections 0-6
- Part 2: sections 7-13  
- Part 3: sections 14-20
- Part 4: sections 21-end

Use the cumulative_tokens field to ensure parts don't exceed {max_tokens:,} tokens."""


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
                r"[\\^\\d+",
                r"References\\s*\n",
                r"Bibliography\\s*\n",
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
