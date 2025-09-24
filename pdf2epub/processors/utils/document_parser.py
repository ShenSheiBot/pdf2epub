"""
Document parsing utilities for content analysis and splitting.

This module provides functions to analyze document structure, identify sections,
find split positions, and extract information useful for intelligent content splitting.
"""

import re
import json
from typing import List, Optional, Tuple, Dict, Any
from loguru import logger
import tiktoken

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def find_split_positions(
    content: str, split_markers: List[str], structural_map: List[Dict]
) -> List[int]:
    """
    Find the actual character positions in content where splits should occur.

    Args:
        content: The original content
        split_markers: List of text markers where splits should happen
        structural_map: The document's structural analysis

    Returns:
        List of character positions for splitting (including 0 and len(content))
    """
    positions = [0]

    for marker in split_markers:
        if not marker or not marker.strip():
            continue

        # First try exact match
        pos = content.find(marker)

        # If not found, try fuzzy search
        if pos == -1:
            # Try without extra whitespace
            clean_marker = " ".join(marker.split())
            pos = content.find(clean_marker)

        # If still not found, try to find in structural map
        if pos == -1 and structural_map:
            for section in structural_map:
                # Check if this section's header matches
                if section.get("header", "").strip() in marker or marker in section.get(
                    "header", ""
                ):
                    # Use the section's position
                    if "position" in section:
                        pos = section["position"]["start"]
                        break

        if pos != -1 and pos not in positions:
            positions.append(pos)

    positions.append(len(content))
    return sorted(set(positions))


def map_preprocessed_position_to_original(
    preprocessed_content: str,
    original_content: str,
    preprocessed_pos: int,
    context_size: int = 100
) -> int:
    """
    Map a position in preprocessed content to the corresponding position in original content.

    Args:
        preprocessed_content: The preprocessed content
        original_content: The original content
        preprocessed_pos: Position in preprocessed content
        context_size: Size of context to use for matching (default: 100 chars)

    Returns:
        Corresponding position in original content
    """
    # Get context around the position in preprocessed content
    start_context = max(0, preprocessed_pos - context_size)
    end_context = min(len(preprocessed_content), preprocessed_pos + context_size)

    # Extract the context string
    context_str = preprocessed_content[start_context:end_context]

    if not context_str:
        return preprocessed_pos  # Fallback to same position

    # Calculate approximate position in original (as a starting point for search)
    # Use ratio of positions as initial guess
    if len(preprocessed_content) > 0:
        ratio = preprocessed_pos / len(preprocessed_content)
        approx_pos = int(ratio * len(original_content))
    else:
        approx_pos = 0

    # Search for the context string in original content, starting near approximate position
    # First try exact match near the approximate position
    search_start = max(0, approx_pos - len(original_content) // 10)  # Search within 10% range
    search_end = min(len(original_content), approx_pos + len(original_content) // 10)

    search_region = original_content[search_start:search_end]
    local_pos = search_region.find(context_str)

    if local_pos != -1:
        # Found it in the local region
        actual_pos = search_start + local_pos + (preprocessed_pos - start_context)
        return actual_pos

    # If not found locally, search the entire content
    global_pos = original_content.find(context_str)
    if global_pos != -1:
        actual_pos = global_pos + (preprocessed_pos - start_context)
        return actual_pos

    # If exact match fails, try to find a smaller portion
    smaller_context = preprocessed_content[max(0, preprocessed_pos - 20):min(len(preprocessed_content), preprocessed_pos + 20)]
    if smaller_context:
        pos = original_content.find(smaller_context, max(0, approx_pos - 1000))
        if pos != -1:
            return pos + (preprocessed_pos - max(0, preprocessed_pos - 20))

    # Fallback: use the approximate position
    return approx_pos


def find_split_positions_by_indices(
    content: str, split_indices: List[int], structural_map: List[Dict]
) -> List[int]:
    """
    Convert section indices to character positions in content.

    Args:
        content: The original content
        split_indices: List of section indices where splits should happen
        structural_map: The document's structural analysis

    Returns:
        List of character positions for splitting
    """
    positions = [0]

    for idx in split_indices:
        if 0 <= idx < len(structural_map):
            section = structural_map[idx]
            if "position" in section:
                pos = section["position"]["start"]
                if pos not in positions:
                    positions.append(pos)

    positions.append(len(content))
    return sorted(set(positions))


def _is_markdown_header(line: str) -> bool:
    """Check if a line is a markdown header."""
    return bool(re.match(r"^#{1,6}\s+\S", line))


def _extract_header_text(line: str) -> str:
    """Extract the text content from a markdown header."""
    match = re.match(r"^#{1,6}\s+(.+)$", line)
    return match.group(1) if match else line


def _is_potential_list_item(line: str) -> bool:
    """Check if a line looks like a list item."""
    # Check for numbered lists (1. or 1) format)
    if re.match(r"^\s*\d+[\.)]\s+\S", line):
        return True
    # Check for bullet lists
    if re.match(r"^\s*[-*+]\s+\S", line):
        return True
    # Check for lettered lists
    if re.match(r"^\s*[a-z][\.)]\s+\S", line):
        return True
    return False


def _detect_footnote_definitions(
    lines: List[str], section_start: int, section_end: int
) -> List[int]:
    """
    Detect footnote definitions in a section.

    Footnotes are typically marked as [^1], [^2], etc. at the start of a line.

    Args:
        lines: All document lines
        section_start: Starting line index of the section
        section_end: Ending line index of the section

    Returns:
        List of footnote numbers found as definitions
    """
    footnote_pattern = re.compile(r"^\[\^(\d+)\]:\s*")
    footnotes = []

    for i in range(section_start, min(section_end, len(lines))):
        match = footnote_pattern.match(lines[i])
        if match:
            footnotes.append(int(match.group(1)))

    return footnotes


def _detect_citations(text: str) -> List[int]:
    """
    Detect citation references in text.

    Common patterns:
    - [1] or [1,2,3] or [1-5]
    - (Smith 2020) or (Smith, 2020)
    - ^1 or ^1,2,3
    - Various footnote styles

    Args:
        text: The text to search for citations

    Returns:
        List of citation/reference numbers found
    """
    citations = set()

    # Pattern 1: Square brackets with numbers [1] [1,2,3] [1-5]
    bracket_pattern = re.compile(r"\[(\d+(?:[-,]\d+)*)\]")
    for match in bracket_pattern.finditer(text):
        citation_str = match.group(1)
        # Parse ranges and lists
        for part in citation_str.split(","):
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    citations.update(range(start, end + 1))
                except ValueError:
                    pass
            else:
                try:
                    citations.add(int(part))
                except ValueError:
                    pass

    # Pattern 2: Superscript style ^1 ^1,2,3
    superscript_pattern = re.compile(r"\^(\d+(?:,\d+)*)")
    for match in superscript_pattern.finditer(text):
        for num in match.group(1).split(","):
            try:
                citations.add(int(num))
            except ValueError:
                pass

    # Pattern 3: Footnote references [^1]
    footnote_pattern = re.compile(
        r"\[\^(\d+)\](?!:)"
    )  # Negative lookahead to exclude definitions
    for match in footnote_pattern.finditer(text):
        try:
            citations.add(int(match.group(1)))
        except ValueError:
            pass

    return sorted(citations)


def _format_citation_ranges(citations: List[int]) -> str:
    """
    Format a list of citation numbers into a compact range string.

    Example: [1,2,3,5,6,8] -> "1-3, 5-6, 8"

    Args:
        citations: List of citation numbers

    Returns:
        Formatted string representation
    """
    if not citations:
        return "none"

    ranges = []
    start = citations[0]
    end = citations[0]

    for i in range(1, len(citations)):
        if citations[i] == end + 1:
            end = citations[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = citations[i]
            end = citations[i]

    # Add the last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def analyze_document_structure(
    content: str, content_type: str = "general"
) -> List[Dict]:
    """
    Analyze the structure of a document and return section information.

    This function identifies document sections based on markdown headers and
    other structural elements, providing token counts and metadata for each section.

    Args:
        content: The document content to analyze
        content_type: Type of content ("general", "academic", "japanese")

    Returns:
        List of dictionaries containing section information:
        - header: The section header text
        - level: Header level (1-6 for markdown headers)
        - tokens: Token count for this section only
        - cumulative_tokens: Total tokens up to and including this section
        - position: Dict with 'start' and 'end' character positions
        - preview: Truncated preview of section content
        - citations_formatted: (academic only) Citation references found
        - footnote_definitions: (academic only) Footnote definitions found
    """
    logger.info(f"Analyzing document structure for {content_type} content")

    lines = content.split("\n")
    sections = []
    cumulative_tokens = 0

    # Find all headers and their positions
    for i, line in enumerate(lines):
        if _is_markdown_header(line):
            # Calculate character position
            char_pos = sum(len(lines[j]) + 1 for j in range(i))  # +1 for newline

            # Get header level
            level = len(re.match(r"^(#{1,6})\s", line).group(1))
            header_text = _extract_header_text(line)

            sections.append(
                {
                    "line_index": i,
                    "char_position": char_pos,
                    "header": header_text,
                    "level": level,
                    "tokens": 0,
                    "cumulative_tokens": 0,
                    "position": {"start": char_pos, "end": char_pos + len(line)},
                }
            )

    # If no headers found, treat entire content as one section
    if not sections:
        total_tokens = len(tokenizer.encode(content))
        return [
            {
                "header": "Document",
                "level": 1,
                "tokens": total_tokens,
                "cumulative_tokens": total_tokens,
                "position": {"start": 0, "end": len(content)},
                "preview": content[:200] + "..." if len(content) > 200 else content,
            }
        ]

    # Calculate tokens and positions for each section
    for i, section in enumerate(sections):
        # Determine section boundaries
        start_line = section["line_index"]
        if i < len(sections) - 1:
            end_line = sections[i + 1]["line_index"]
            end_char = sections[i + 1]["char_position"]
        else:
            end_line = len(lines)
            end_char = len(content)

        # Extract section content
        section_lines = lines[start_line:end_line]
        section_text = "\n".join(section_lines)

        # Calculate tokens
        section_tokens = len(tokenizer.encode(section_text))
        cumulative_tokens += section_tokens

        # Update section info
        section["tokens"] = section_tokens
        section["cumulative_tokens"] = cumulative_tokens
        section["position"]["end"] = end_char

        # Add preview
        preview_text = section_text.strip()
        if len(preview_text) > 200:
            section["preview"] = preview_text[:100] + "..." + preview_text[-97:]
        else:
            section["preview"] = preview_text

        # Academic content analysis
        if content_type == "academic":
            # Detect citations
            citations = _detect_citations(section_text)
            section["citations_formatted"] = _format_citation_ranges(citations)

            # Detect footnote definitions
            footnotes = _detect_footnote_definitions(lines, start_line, end_line)
            section["footnote_definitions"] = (
                _format_citation_ranges(footnotes) if footnotes else "none"
            )

        # Log section details for debugging
        if content_type == "academic":
            logger.debug(
                f"Section {i}: '{section['header']}': "
                f"citations: {section.get('citations_formatted', 'none')}, "
                f"footnotes: {section.get('footnote_definitions', 'none')}, "
                f"cumulative tokens: {section['cumulative_tokens']:,}"
            )
        else:
            logger.debug(
                f"Section {i}: '{section['header']}': "
                f"tokens: {section['tokens']:,}, "
                f"cumulative: {section['cumulative_tokens']:,}"
            )

    logger.info(f"Document analysis complete: {len(sections)} sections identified")

    if sections:
        logger.info(f"Total document tokens: {sections[-1]['cumulative_tokens']:,}")

    # Academic-specific summary
    if content_type == "academic" and sections:
        total_citations = 0
        total_definitions = 0
        for section in sections:
            if section.get("citations_formatted") != "none":
                # Count actual citations
                for part in section["citations_formatted"].split(", "):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        total_citations += end - start + 1
                    else:
                        total_citations += 1
            if section.get("footnote_definitions") != "none":
                for part in section["footnote_definitions"].split(", "):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        total_definitions += end - start + 1
                    else:
                        total_definitions += 1
        logger.info(
            f"Academic analysis: {total_citations} citations, {total_definitions} definitions found"
        )

    return sections


def analyze_paragraph_structure(
    content: str, max_preview_length: int = 50, min_paragraph_tokens: int = 20
) -> List[Dict]:
    """
    Analyze document at the paragraph level for intelligent splitting.

    Combines very short paragraphs to reduce token usage in structural map.

    Args:
        content: The document content to analyze
        max_preview_length: Maximum length for paragraph preview (default: 50)
        min_paragraph_tokens: Minimum tokens to keep as separate paragraph (default: 20)

    Returns:
        List of dictionaries with paragraph information:
        - paragraph_index: Sequential index starting from 0
        - tokens: Token count of the paragraph(s)
        - cumulative_tokens: Running total of tokens up to and including this paragraph
        - preview: Truncated preview of content
        - position: Dict with 'start' and 'end' character positions
    """
    # Find paragraphs using multi-level detection
    paragraphs_with_positions = _extract_paragraphs_with_positions(content)

    paragraph_map = []
    cumulative_tokens = 0

    # Buffer for combining short paragraphs
    combined_text = ""
    combined_start = None
    combined_end = None
    combined_tokens = 0

    for i, (para_text, start_pos, end_pos) in enumerate(paragraphs_with_positions):
        if not para_text.strip():
            continue

        # Calculate token count
        token_count = len(tokenizer.encode(para_text))

        # Combine very short paragraphs to reduce map size
        if token_count < min_paragraph_tokens:
            if combined_text:
                combined_text += " " + para_text
                combined_tokens += token_count
                combined_end = end_pos
            else:
                combined_text = para_text
                combined_start = start_pos
                combined_end = end_pos
                combined_tokens = token_count

            # If combined paragraphs are now large enough, add them
            if combined_tokens >= min_paragraph_tokens:
                cumulative_tokens += combined_tokens

                # Create preview
                if len(combined_text) <= max_preview_length:
                    preview = combined_text
                else:
                    preview = combined_text[: max_preview_length - 3] + "..."
                preview = preview.replace("\n", " ").strip()

                paragraph_info = {
                    "paragraph_index": len(paragraph_map),
                    "tokens": combined_tokens,
                    "cumulative_tokens": cumulative_tokens,
                    "preview": preview,
                    "position": {"start": combined_start, "end": combined_end},
                }
                paragraph_map.append(paragraph_info)

                # Reset buffer
                combined_text = ""
                combined_start = None
                combined_end = None
                combined_tokens = 0
        else:
            # First flush any buffered short paragraphs
            if combined_text:
                cumulative_tokens += combined_tokens

                # Create preview
                if len(combined_text) <= max_preview_length:
                    preview = combined_text
                else:
                    preview = combined_text[: max_preview_length - 3] + "..."
                preview = preview.replace("\n", " ").strip()

                paragraph_info = {
                    "paragraph_index": len(paragraph_map),
                    "tokens": combined_tokens,
                    "cumulative_tokens": cumulative_tokens,
                    "preview": preview,
                    "position": {"start": combined_start, "end": combined_end},
                }
                paragraph_map.append(paragraph_info)

                combined_text = ""
                combined_start = None
                combined_end = None
                combined_tokens = 0

            # Add this normal-sized paragraph
            cumulative_tokens += token_count

            # Create preview
            if len(para_text) <= max_preview_length:
                preview = para_text
            else:
                preview = para_text[: max_preview_length - 3] + "..."
            preview = preview.replace("\n", " ").strip()

            paragraph_info = {
                "paragraph_index": len(paragraph_map),
                "tokens": token_count,
                "cumulative_tokens": cumulative_tokens,
                "preview": preview,
                "position": {"start": start_pos, "end": end_pos},
            }
            paragraph_map.append(paragraph_info)

    # Don't forget remaining combined paragraphs
    if combined_text:
        cumulative_tokens += combined_tokens

        # Create preview
        if len(combined_text) <= max_preview_length:
            preview = combined_text
        else:
            preview = combined_text[: max_preview_length - 3] + "..."
        preview = preview.replace("\n", " ").strip()

        paragraph_info = {
            "paragraph_index": len(paragraph_map),
            "tokens": combined_tokens,
            "cumulative_tokens": cumulative_tokens,
            "preview": preview,
            "position": {"start": combined_start, "end": combined_end},
        }
        paragraph_map.append(paragraph_info)

    return paragraph_map


def _preprocess_ocr_page_breaks(content: str) -> Tuple[str, List[int]]:
    """
    Preprocess OCR content to handle page break markers that split sentences.
    Only merges when --- clearly interrupts a sentence.

    This is used ONLY for content splitting, not for polishing.

    Args:
        content: Original OCR content with page breaks

    Returns:
        Tuple of:
        - Content with sentence-interrupting page breaks removed
        - List where index i maps preprocessed position i to original position
    """
    result = []  # Characters in preprocessed content
    position_map = []  # Maps each char in preprocessed to position in original

    lines = content.split('\n')
    original_pos = 0  # Current position in original content

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a page break marker that should be removed
        if line.strip() == '---':
            # Analyze context to decide if we should remove this ---
            should_remove = _should_remove_page_break(result, lines, i)

            if should_remove:
                # Skip the --- line (don't add it to result)
                original_pos += len(line) + 1  # +1 for the newline
                i += 1

                # Skip any empty lines after ---
                while i < len(lines) and not lines[i].strip():
                    original_pos += len(lines[i]) + 1
                    i += 1

                # If we have a next line, join it with the previous content
                if i < len(lines):
                    next_line = lines[i]

                    # Find the last non-empty content in result to join with
                    # Remove trailing newlines from result
                    while result and result[-1] == '\n':
                        result.pop()
                        position_map.pop()  # Also remove from position map

                    # Add a space to join the sentences
                    if result and result[-1] not in ' \n':
                        result.append(' ')
                        position_map.append(original_pos)  # Map space to start of next line

                    # Add the next line
                    for char in next_line:
                        result.append(char)
                        position_map.append(original_pos)
                        original_pos += 1

                    # Add newline after the line
                    result.append('\n')
                    position_map.append(original_pos)
                    original_pos += 1  # For the newline

                    i += 1

                    # Check if we need to preserve paragraph break
                    # (If there were multiple empty lines around ---)
                    if i < len(lines) and not lines[i].strip():
                        # Add back one empty line for paragraph break
                        result.append('\n')
                        position_map.append(original_pos)
                continue

        # Normal line - add it to result with position mapping
        for char in line:
            result.append(char)
            position_map.append(original_pos)
            original_pos += 1

        # Add newline (except for last line)
        if i < len(lines) - 1:
            result.append('\n')
            position_map.append(original_pos)
            original_pos += 1  # For the newline

        i += 1

    return ''.join(result), position_map


def _should_remove_page_break(current_result: List[str], lines: List[str], break_idx: int) -> bool:
    """
    Determine if a --- page break should be removed (i.e., it interrupts a sentence).

    Args:
        current_result: Characters accumulated so far in preprocessing
        lines: All lines in the document
        break_idx: Index of the --- line

    Returns:
        True if the --- should be removed, False if it should be kept
    """
    # Get the last non-empty line from what we've processed so far
    temp_content = ''.join(current_result).rstrip()
    prev_lines = temp_content.split('\n') if temp_content else []
    prev_line = ""
    for line in reversed(prev_lines):
        if line.strip():
            prev_line = line.strip()
            break

    # Get the next non-empty line after ---
    next_idx = break_idx + 1
    while next_idx < len(lines) and not lines[next_idx].strip():
        next_idx += 1

    next_line = ""
    if next_idx < len(lines):
        next_line = lines[next_idx].strip()

    # Don't remove if next line is a markdown header, footnote, etc.
    if next_line:
        # Check if it's a markdown header (keep the ---)
        if re.match(r'^#{1,6}\s+\S', next_line):
            return False
        # Or is clearly a footnote/citation (keep the ---)
        elif re.match(r'^[\*\^¹²³⁴⁵⁶⁷⁸⁹⁰\[\$]\s*\w', next_line):
            return False
        # Or starts with a number followed by space (footnote reference)
        elif re.match(r'^\d+\s+\w', next_line):
            return False
        # Check if previous line ends mid-sentence
        elif prev_line:
            # No sentence-ending punctuation
            if prev_line[-1] not in '.!?。！？」』】）);:':
                return True
            # Or ends with comma (clear continuation)
            elif prev_line[-1] in ',，、':
                return True
            # Or next line starts with lowercase (continuation)
            elif next_line[0].islower():
                return True

    return False


def _extract_paragraphs_with_positions(content: str) -> List[Tuple[str, int, int]]:
    """
    Extract paragraphs with their character positions in the original content.

    Uses intelligent paragraph detection that handles OCR artifacts and
    respects both Western and CJK punctuation.

    Returns:
        List of tuples: (paragraph_text, start_position_in_original, end_position_in_original)
    """
    # Preprocess content to handle page breaks that interrupt sentences
    preprocessed, position_map = _preprocess_ocr_page_breaks(content)

    paragraphs_with_positions = []

    # Try triple newlines first (clear section breaks)
    if "\n\n\n" in preprocessed:
        parts = preprocessed.split("\n\n\n")
        current_pos = 0

        for part in parts:
            if part.strip():
                # Find the part in preprocessed content
                start_preprocessed = preprocessed.find(part, current_pos)
                if start_preprocessed == -1:
                    continue
                end_preprocessed = start_preprocessed + len(part)

                # Map positions back to original content
                # Find first non-whitespace character positions
                part_stripped = part.strip()
                start_stripped = part.find(part_stripped)
                end_stripped = start_stripped + len(part_stripped)

                # Map the actual content boundaries to original positions
                start_original = position_map[start_preprocessed + start_stripped] if start_preprocessed + start_stripped < len(position_map) else len(content)
                end_original = position_map[min(start_preprocessed + end_stripped - 1, len(position_map) - 1)] + 1 if position_map else len(content)

                # Get the actual text from the original content
                original_text = content[start_original:end_original].strip()

                paragraphs_with_positions.append((original_text, start_original, end_original))
                current_pos = end_preprocessed

        if len(paragraphs_with_positions) >= 10:
            return paragraphs_with_positions

    # Try double newlines (standard paragraph breaks)
    if "\n\n" in preprocessed:
        parts = preprocessed.split("\n\n")
        current_pos = 0

        for part in parts:
            if part.strip():
                # Find the part in preprocessed content
                start_preprocessed = preprocessed.find(part, current_pos)
                if start_preprocessed == -1:
                    continue
                end_preprocessed = start_preprocessed + len(part)

                # Map positions back to original content
                part_stripped = part.strip()
                start_stripped = part.find(part_stripped)
                end_stripped = start_stripped + len(part_stripped)

                # Map the actual content boundaries to original positions
                start_original = position_map[start_preprocessed + start_stripped] if start_preprocessed + start_stripped < len(position_map) else len(content)
                end_original = position_map[min(start_preprocessed + end_stripped - 1, len(position_map) - 1)] + 1 if position_map else len(content)

                # Get the actual text from the original content
                original_text = content[start_original:end_original].strip()

                paragraphs_with_positions.append((original_text, start_original, end_original))
                current_pos = end_preprocessed

        # Check if we have reasonable paragraphs
        if len(paragraphs_with_positions) >= 10:
            # Check if paragraphs are reasonably sized using token counts
            token_sizes = [len(tokenizer.encode(p[0])) for p in paragraphs_with_positions]
            max_tokens = max(token_sizes) if token_sizes else 0
            median_tokens = sorted(token_sizes)[len(token_sizes)//2] if token_sizes else 0

            # Use token-based limits for consistency across languages
            # 150% of 4000 tokens = 6000 tokens max for a single paragraph
            max_acceptable_tokens = 6000

            # If the largest paragraph is too big, continue to single newlines
            # Or if median is too small (many tiny fragments - less than ~50 tokens)
            if max_tokens <= max_acceptable_tokens and median_tokens > 50:
                return paragraphs_with_positions

    # Try single newlines with intelligent filtering
    # This handles cases where double newlines created too many tiny fragments or too few good paragraphs
    if "\n" in preprocessed:
        lines = preprocessed.split("\n")
        current_para = []
        para_start_pos = 0
        current_pos = 0

        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)
            current_pos = line_end + 1  # +1 for the newline character

            line_text = line.strip()

            if not line_text:
                # Empty line might be a paragraph break
                if current_para:
                    para_text = " ".join(current_para)
                    paragraphs_with_positions.append(
                        (para_text, para_start_pos, line_start - 1)
                    )
                    current_para = []
                continue

            if not current_para:
                para_start_pos = line_start
                current_para = [line_text]
            else:
                # Check if this line ends a paragraph
                # Western and CJK sentence-ending punctuation
                sentence_endings = ".!?。！？」』】）)」』"

                if current_para[-1] and current_para[-1][-1] in sentence_endings:
                    # Check if next line starts a new sentence
                    first_char = line_text[0] if line_text else ""

                    # Check for CJK characters
                    is_cjk = (
                        "\u4e00" <= first_char <= "\u9fff"  # Chinese
                        or "\u3040" <= first_char <= "\u309f"  # Hiragana
                        or "\u30a0" <= first_char <= "\u30ff"  # Katakana
                        or "\uac00" <= first_char <= "\ud7af"
                    )  # Korean

                    if first_char.isupper() or first_char.isdigit() or is_cjk:
                        # This looks like a real paragraph ending
                        para_text = " ".join(current_para)
                        paragraphs_with_positions.append(
                            (para_text, para_start_pos, line_start - 1)
                        )
                        current_para = [line_text]
                        para_start_pos = line_start
                    else:
                        current_para.append(line_text)
                else:
                    current_para.append(line_text)

        # Don't forget the last paragraph
        if current_para:
            para_text = " ".join(current_para)
            paragraphs_with_positions.append((para_text, para_start_pos, len(content)))

    # If we still don't have enough paragraphs, fall back to sentence boundaries
    if len(paragraphs_with_positions) < 10:
        # Split by sentence-ending punctuation
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])|"  # Western sentences
            r"(?<=[。！？」』】）])",  # CJK sentence endings
            content,
        )

        if len(sentences) > 20:
            # Group sentences into paragraphs (5 sentences each)
            paragraphs_with_positions = []
            sentences_per_para = 5
            current_pos = 0

            for i in range(0, len(sentences), sentences_per_para):
                para_sentences = sentences[i : i + sentences_per_para]
                para_text = " ".join(para_sentences).strip()
                if para_text:
                    start = content.find(para_text, current_pos)
                    if start == -1:
                        start = current_pos
                    end = start + len(para_text)
                    paragraphs_with_positions.append((para_text, start, end))
                    current_pos = end

    return (
        paragraphs_with_positions
        if paragraphs_with_positions
        else [(content, 0, len(content))]
    )
