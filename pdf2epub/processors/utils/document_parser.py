"""
Document parsing utilities for content analysis and splitting.

This module provides functions to find split positions and extract paragraphs
for content splitting operations.
"""

import re
from typing import List, Tuple, Dict
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


def extract_paragraphs_with_positions(content: str) -> List[Tuple[str, int, int]]:
    """
    Extract paragraphs with their character positions in the original content.

    Uses intelligent paragraph detection that handles OCR artifacts and
    respects both Western and CJK punctuation.

    Returns:
        List of tuples: (paragraph_text, start_position, end_position)
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
            max_acceptable_tokens = 6000

            # If the largest paragraph is too big, continue to single newlines
            if max_tokens <= max_acceptable_tokens and median_tokens > 50:
                return paragraphs_with_positions

    # Try single newlines with intelligent filtering
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
