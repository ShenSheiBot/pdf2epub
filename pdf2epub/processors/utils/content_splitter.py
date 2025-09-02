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

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def fuzzy_find_sentence(
    haystack: str,
    needle: str,
    max_edits: int = 3
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
        needle.replace("&", r"\&"),  # Escaped ampersand
        needle.replace(r"\&", "&"),  # Unescaped ampersand
        needle.replace('"', r"\""),  # Escaped quotes
        needle.replace(r"\"", '"'),  # Unescaped quotes
        needle.replace("'", r"\'"),  # Escaped single quotes
        needle.replace(r"\'", "'"),  # Unescaped single quotes
    ]

    for variant in variations:
        pos = haystack.find(variant)
        if pos != -1:
            return (pos, pos + len(variant), variant)

    return None


def split_content_simple(content: str, max_tokens: int) -> List[str]:
    """
    Simple content splitter that divides content into roughly equal parts.

    Args:
        content: The content to split
        max_tokens: Maximum tokens per part

    Returns:
        List of content parts
    """
    # Use proper token counting with tiktoken
    actual_tokens = len(tokenizer.encode(content))

    if actual_tokens <= max_tokens:
        return [content]

    # Calculate number of parts needed
    num_parts = (actual_tokens // max_tokens) + 1

    # Split by paragraphs
    paragraphs = content.split("\n\n")

    # If we don't have enough paragraphs, split by lines
    if len(paragraphs) < num_parts * 2:
        paragraphs = content.split("\n")

    # Distribute paragraphs to achieve roughly equal token counts per part
    parts = []
    target_tokens_per_part = actual_tokens // num_parts

    current_part = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))

        # If adding this paragraph would exceed target and we have content, start new part
        if current_tokens + para_tokens > target_tokens_per_part * 1.2 and current_part:
            parts.append(
                "\n\n".join(current_part)
                if "\n\n" in content
                else "\n".join(current_part)
            )
            current_part = [para]
            current_tokens = para_tokens
        else:
            current_part.append(para)
            current_tokens += para_tokens

    # Add the last part
    if current_part:
        parts.append(
            "\n\n".join(current_part) if "\n\n" in content else "\n".join(current_part)
        )

    # Log the actual token counts for each part
    for i, part in enumerate(parts):
        part_tokens = len(tokenizer.encode(part))
        logger.debug(f"Part {i + 1}/{len(parts)}: {part_tokens:,} tokens")

    return parts if parts else [content]


def split_content_intelligently(
    content: str,
    max_tokens: int,
    llm_client,
    model_configs: Optional[List[Dict]] = None,
    content_type: str = "auto",
) -> List[str]:
    """
    Use LLM to intelligently split content at natural boundaries.

    Args:
        content: The content to split
        max_tokens: Maximum tokens per part (used as guideline)
        llm_client: LLM client for split detection (should support generate method)
        model_configs: Optional model configurations to use for splitting
        content_type: Type of content ("academic", "japanese", "general", "auto")

    Returns:
        List of content parts
    """
    # Use proper token counting with tiktoken
    actual_tokens = len(tokenizer.encode(content))

    if actual_tokens <= max_tokens:
        return [content]

    # Calculate number of parts needed
    num_parts = max(2, (actual_tokens // max_tokens) + 1)

    logger.info(
        f"Content has {actual_tokens:,} tokens, splitting into {num_parts} parts"
    )

    # Auto-detect content type if needed
    if content_type == "auto":
        import re

        # Check for Japanese characters
        japanese_chars = re.findall(
            r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", content[:5000]
        )
        if len(japanese_chars) > 500:  # Significant Japanese content
            content_type = "japanese"
            logger.info("Auto-detected content type: Japanese")
        else:
            # Check for academic indicators
            academic_indicators = [
                r"\[\^\d+\]",  # Markdown footnotes
                r"References\s*\n",
                r"Bibliography\s*\n",
            ]
            if any(
                re.search(pattern, content[:5000]) for pattern in academic_indicators
            ):
                content_type = "academic"
                logger.info("Auto-detected content type: Academic")
            else:
                content_type = "general"
                logger.info("Auto-detected content type: General")

    # Create appropriate split prompt based on content type
    total_tokens = actual_tokens

    if content_type == "japanese":
        split_prompt = f"""You are helping split a long Japanese novel/light novel chapter into smaller parts for processing.

The chapter has approximately {total_tokens:,} tokens. We'd prefer parts under {max_tokens:,} tokens. Return a JSON array of the EXACT final sentences that mark the end of each part (except the last one). Part ideally start with a subsection and does not contain any incomplete sentences or break a important scene."""

    elif content_type == "academic":
        split_prompt = f"""You are helping split a long academic chapter into smaller parts for processing.

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
   - Split AFTER a complete section with all its footnotes (whether inline or at end)

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

Return a JSON array of the EXACT final sentences that mark the end of each part (except the last one).
Choose split points that respect the above priorities. For inline footnotes, split at section boundaries.
For end-of-section footnotes, split AFTER the Notes/References section ends.

Example response:
["[^5]: Johnson, 2019, p. 45.", "This concludes the historical overview."]"""

    else:  # general content
        split_prompt = f"""You are helping split a long document into smaller parts for processing.

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

3. **Balance Part Sizes**:
   - Aim for roughly equal-sized parts
   - It's OK if some parts are larger to maintain coherence

Return a JSON array of the EXACT final sentences that mark the end of each part (except the last one).
Choose split points at natural breaks like section endings, topic transitions, or clear paragraph boundaries.

Example response:
["This concludes our discussion of the basic concepts.", "The next section explores advanced techniques."]"""

    # Add the content to analyze
    split_prompt += f"""

Here is the content to analyze:

{content}"""

    try:
        # Generate split suggestions
        response = llm_client.generate(
            prompt=split_prompt,
            model_configs=[
                {"provider": "gemini", "model": "gemini-2.5-flash", "max_retries": 2}
            ],
            operation_name="Split chapter"
        )
        if not response or not response.strip():
            logger.warning(
                "Failed to get LLM split suggestions, falling back to simple split"
            )
            return split_content_simple(content, max_tokens)

        # Extract JSON from the response, handling various formats
        response = response.strip()

        # Try to find JSON array in the response
        json_str = None

        # Method 1: Look for ```json code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
        # Method 2: Look for ``` code block
        elif "```" in response:
            start = response.find("```") + 3
            # Skip to next line if ``` is on its own line
            if response[start : start + 1] == "\n":
                start += 1
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
        # Method 3: Look for JSON array directly (starts with [)
        elif "[" in response:
            # Find the first [ and last ]
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                json_str = response[start : end + 1]

        if not json_str:
            # Fallback: try the whole response
            json_str = response

        # Parse the JSON response
        logger.debug(f"Parsing split response: {json_str[:200]}...")
        try:
            split_sentences = json.loads(json_str.strip())
            if not isinstance(split_sentences, list):
                raise ValueError("Response is not a list")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse LLM split response: {e}, falling back to simple split"
            )
            return split_content_simple(content, max_tokens)

        # Find the split points in the content
        parts = []
        start_pos = 0

        for split_sentence in split_sentences:
            # Clean up the split sentence
            # Remove ellipsis that LLMs often add to indicate continuation
            cleaned_sentence = split_sentence
            if cleaned_sentence.endswith('...'):
                cleaned_sentence = cleaned_sentence[:-3].rstrip()
            if cleaned_sentence.endswith('…'):  # Unicode ellipsis
                cleaned_sentence = cleaned_sentence[:-1].rstrip()
            
            # Use fuzzy matching to find the sentence
            match_result = fuzzy_find_sentence(content, cleaned_sentence)

            if match_result:
                end_pos = match_result[1]
                # Include the sentence in the current part
                part = content[start_pos:end_pos].strip()
                if part:
                    parts.append(part)
                start_pos = end_pos
                logger.debug(f"Found split point: '{match_result[2][:50]}...'")
            else:
                logger.warning(
                    f"Could not find split sentence: '{split_sentence[:50]}...'"
                )

        # Add the remaining content as the last part
        if start_pos < len(content):
            last_part = content[start_pos:].strip()
            if last_part:
                parts.append(last_part)

        # Validate the split
        if len(parts) < 2:
            logger.warning(
                "LLM split resulted in too few parts, falling back to simple split"
            )
            return split_content_simple(content, max_tokens)

        # Log part sizes
        for i, part in enumerate(parts, 1):
            part_tokens = len(tokenizer.encode(part))
            logger.info(f"Part {i}/{len(parts)}: {part_tokens:,} tokens")

        return parts

    except Exception as e:
        logger.warning(f"Intelligent split failed: {e}, falling back to simple split")
        return split_content_simple(content, max_tokens)
