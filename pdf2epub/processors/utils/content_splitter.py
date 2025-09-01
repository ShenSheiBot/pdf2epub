"""
Content splitting utilities for markdown processors.

This module provides intelligent content splitting for long texts,
particularly useful for processing large chapters that exceed model token limits.
"""

import json
import regex
from typing import List, Optional, Tuple
from loguru import logger


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
        pattern = f'(?b)({regex.escape(needle)}){{e<={max_edits}}}'
        match = regex.search(pattern, haystack)
        if match:
            return (match.start(), match.end(), match.group(0))
    except Exception as e:
        logger.debug(f"Fuzzy matching failed: {e}")
    
    # Try to find with common escape variations
    variations = [
        needle.replace('&', r'\&'),  # Escaped ampersand
        needle.replace(r'\&', '&'),  # Unescaped ampersand
        needle.replace('"', r'\"'),  # Escaped quotes
        needle.replace(r'\"', '"'),  # Unescaped quotes
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
    # Estimate total tokens (rough approximation: 1 token ≈ 4 chars)
    estimated_tokens = len(content) // 4
    
    if estimated_tokens <= max_tokens:
        return [content]
    
    # Calculate number of parts needed
    num_parts = (estimated_tokens // max_tokens) + 1
    
    # Split by paragraphs
    paragraphs = content.split('\n\n')
    
    # Distribute paragraphs evenly
    parts = []
    paras_per_part = len(paragraphs) // num_parts
    
    for i in range(num_parts):
        start_idx = i * paras_per_part
        if i == num_parts - 1:
            # Last part gets all remaining paragraphs
            part = '\n\n'.join(paragraphs[start_idx:])
        else:
            end_idx = start_idx + paras_per_part
            part = '\n\n'.join(paragraphs[start_idx:end_idx])
        
        if part:
            parts.append(part)
    
    return parts if parts else [content]


def split_content_intelligently(
    content: str,
    max_tokens: int,
    llm_client
) -> List[str]:
    """
    Use LLM to intelligently split content at natural boundaries.
    
    Args:
        content: The content to split
        max_tokens: Maximum tokens per part (used as guideline)
        llm_client: LLM client for split detection (should support generate method)
    
    Returns:
        List of content parts
    """
    # Estimate total tokens
    estimated_tokens = len(content) // 4
    
    if estimated_tokens <= max_tokens:
        return [content]
    
    # Calculate number of parts needed
    num_parts = max(2, (estimated_tokens // max_tokens) + 1)
    
    logger.info(f"Content has ~{estimated_tokens:,} tokens, splitting into {num_parts} parts")
    
    # Ask LLM to identify good split points
    total_tokens = estimated_tokens
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
["[^5]: Johnson, 2019, p. 45.", "This concludes the historical overview."]

Here is the content to analyze:

{content}"""

    try:
        # Generate split suggestions
        response = llm_client.generate(
            prompt=split_prompt,
            model_configs=[
                {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2}
            ],
            operation_name="Split chapter"
        )
        
        if not response or not response.strip():
            logger.warning("Failed to get LLM split suggestions, falling back to simple split")
            return split_content_simple(content, max_tokens)
        
        # Clean response if wrapped in code blocks
        if response.startswith('```'):
            lines = response.split('\n')
            # Find the actual JSON content
            json_start = 1 if lines[0].startswith('```') else 0
            json_end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '```':
                    json_end = i
                    break
            response = '\n'.join(lines[json_start:json_end])
        
        # Parse the JSON response
        try:
            split_sentences = json.loads(response)
            if not isinstance(split_sentences, list):
                raise ValueError("Response is not a list")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM split response: {e}, falling back to simple split")
            return split_content_simple(content, max_tokens)
        
        # Find the split points in the content
        parts = []
        start_pos = 0
        
        for split_sentence in split_sentences:
            # Use fuzzy matching to find the sentence
            match_result = fuzzy_find_sentence(content, split_sentence)
            
            if match_result:
                end_pos = match_result[1]
                # Include the sentence in the current part
                part = content[start_pos:end_pos].strip()
                if part:
                    parts.append(part)
                start_pos = end_pos
                logger.debug(f"Found split point: '{match_result[2][:50]}...'")
            else:
                logger.warning(f"Could not find split sentence: '{split_sentence[:50]}...'")
        
        # Add the remaining content as the last part
        if start_pos < len(content):
            last_part = content[start_pos:].strip()
            if last_part:
                parts.append(last_part)
        
        # Validate the split
        if len(parts) < 2:
            logger.warning("LLM split resulted in too few parts, falling back to simple split")
            return split_content_simple(content, max_tokens)
        
        # Log part sizes
        for i, part in enumerate(parts, 1):
            part_tokens = len(part) // 4
            logger.info(f"Part {i}/{len(parts)}: ~{part_tokens:,} tokens")
        
        return parts
        
    except Exception as e:
        logger.warning(f"Intelligent split failed: {e}, falling back to simple split")
        return split_content_simple(content, max_tokens)
