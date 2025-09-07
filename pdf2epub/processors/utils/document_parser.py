"""
Document parsing utilities for structural analysis.

This module provides functions to analyze markdown documents and extract
structural information including sections, token counts, and citation/footnote
relationships for intelligent content splitting.
"""

import re
import regex
from typing import List, Dict, Set, Tuple, Optional
from loguru import logger
import tiktoken

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def format_citation_ranges(citations: List[str]) -> str:
    """
    Format a list of citations into compact ranges.
    
    Args:
        citations: List of citation numbers as strings
    
    Returns:
        Formatted string with ranges, e.g., "1-5, 7, 10-15"
    """
    if not citations:
        return ""
    
    # Try to convert to integers, keeping non-numeric ones separate
    numeric_citations = []
    non_numeric = []
    
    for c in citations:
        try:
            numeric_citations.append(int(c))
        except ValueError:
            non_numeric.append(c)
    
    # Sort numeric citations
    numeric_citations.sort()
    
    # Build ranges for numeric citations
    ranges = []
    if numeric_citations:
        start = numeric_citations[0]
        end = numeric_citations[0]
        
        for num in numeric_citations[1:]:
            if num == end + 1:
                end = num
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = num
                end = num
        
        # Add the last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
    
    # Combine numeric ranges and non-numeric citations
    all_formatted = ranges + non_numeric
    return ", ".join(all_formatted)

# Regex patterns for citations and footnotes
# Citation patterns (found inline in text)
CITATION_PATTERNS = [
    r'\[\^(\w+)\]',          # Standard markdown: [^1], [^note]
    r'\[(\d+)\]',            # Common academic: [1], [2]
    r'\$\^\{?(\d+)\}?\$',    # LaTeX style: $^1$, $^{1}$
    r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+',       # Unicode superscript numbers
]

# Footnote definition patterns (typically at line start)
FOOTNOTE_DEF_PATTERNS = [
    r'^\[\^(\w+)\]:',        # Standard markdown: [^1]:
    r'^\[(\d+)\]\s',         # Academic style at line start: [1] 
    r'^(\d+)\.\s',           # Numbered list style: 1. 
]

# Combined patterns for efficiency
CITATION_REGEX = re.compile('|'.join(f'({pat})' for pat in CITATION_PATTERNS))
FOOTNOTE_DEF_REGEX = re.compile('|'.join(FOOTNOTE_DEF_PATTERNS), re.MULTILINE)


def extract_first_sentence(text: str, max_length: int = 100) -> str:
    """
    Extract the first sentence or first max_length characters of text.
    
    Args:
        text: The text to extract from
        max_length: Maximum length of the extracted text
        
    Returns:
        The first sentence or truncated text
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    if not text:
        return ""
    
    # Try to find first sentence ending
    sentence_endings = ['. ', '! ', '? ', '。', '！', '？']
    min_pos = len(text)
    
    for ending in sentence_endings:
        pos = text.find(ending)
        if pos != -1 and pos < min_pos:
            min_pos = pos + len(ending) - 1  # Include the punctuation
    
    if min_pos < len(text):
        return text[:min_pos + 1].strip()
    
    # If no sentence ending found, return truncated text
    if len(text) > max_length:
        return text[:max_length].strip() + "..."
    
    return text


def create_content_preview(text: str, preview_length: int = 200) -> str:
    """
    Create a preview of content showing beginning and end.
    
    Args:
        text: The full text content
        preview_length: Length of text to show from start and end
        
    Returns:
        Preview string in format "first 200 chars...last 200 chars" or full text if short
    """
    text = text.strip()
    if not text:
        return ""
    
    # If text is short enough, return it all
    if len(text) <= preview_length * 2:
        return text
    
    # Get first and last parts
    first_part = text[:preview_length].strip()
    last_part = text[-preview_length:].strip()
    
    # Make sure we don't cut in the middle of a word
    # Find last space in first part
    last_space = first_part.rfind(' ')
    if last_space > preview_length * 0.8:  # Only trim if we're not losing too much
        first_part = first_part[:last_space]
    
    # Find first space in last part
    first_space = last_part.find(' ')
    if first_space > 0 and first_space < preview_length * 0.2:
        last_part = last_part[first_space + 1:]
    
    return f"{first_part}...{last_part}"


def find_citations_in_text(text: str) -> Set[str]:
    """
    Find all citation markers in a text.
    
    Args:
        text: The text to search
        
    Returns:
        Set of citation keys found
    """
    citations = set()
    
    # Standard markdown citations [^1], [^note]
    for match in re.finditer(r'\[\^(\w+)\]', text):
        citations.add(match.group(1))
    
    # Numbered citations [1], [2] - but not at line start (those are definitions)
    for match in re.finditer(r'(?<!^)\[(\d+)\]', text, re.MULTILINE):
        citations.add(match.group(1))
    
    # LaTeX style $^1$, $^{1}$, ${ }^{1}$
    for match in re.finditer(r'\$\^\{?(\d+)\}?\$', text):
        citations.add(match.group(1))
    
    # OCR variant with space: { }^{41} or { }^41
    for match in re.finditer(r'\{\s*\}\^\{?(\d+)\}?', text):
        citations.add(match.group(1))
    
    # Unicode superscript numbers
    superscript_map = str.maketrans('¹²³⁴⁵⁶⁷⁸⁹⁰', '1234567890')
    for match in re.finditer(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', text):
        normal_number = match.group(0).translate(superscript_map)
        citations.add(normal_number)
    
    return citations


def find_footnote_definitions(text: str) -> Set[str]:
    """
    Find all footnote definitions in a text.
    
    Args:
        text: The text to search
        
    Returns:
        Set of footnote keys defined
    """
    definitions = set()
    lines = text.split('\n')
    
    # Check if this section looks like a footnotes/notes/references section
    text_lower = text.lower()
    is_footnote_section = any(marker in text_lower[:500] for marker in [
        '## notes', '## footnotes', '## references', '## endnotes',
        '# notes', '# footnotes', '# references', '# endnotes',
        'notes:', 'footnotes:', 'references:', 'endnotes:'
    ])
    
    for line in lines:
        line = line.strip()
        
        # Standard markdown definition [^1]:
        match = re.match(r'^\[\^(\w+)\]:', line)
        if match:
            definitions.add(match.group(1))
            continue
        
        # Academic style [1] at line start (must be followed by actual content)
        # More strict: require colon or substantial text after
        match = re.match(r'^\[(\d+)\][\s:]\s*(.+)', line)
        if match and len(match.group(2)) > 10:  # Require some actual content
            definitions.add(match.group(1))
            continue
        
        # Numbered list style 1. - ONLY in footnote sections
        # This avoids false positives from regular numbered lists
        if is_footnote_section:
            match = re.match(r'^(\d+)\.\s+(.+)', line)
            if match:
                number = match.group(1)
                content = match.group(2)
                # Only consider as footnote if:
                # 1. It's a reasonable footnote number (1-999)
                # 2. Has substantial content (not just a few words)
                # 3. Is in a footnote section
                if len(number) <= 3 and len(content) > 20:
                    definitions.add(number)
    
    return definitions


def identify_sections(content: str) -> List[Tuple[int, int, str, str]]:
    """
    Identify sections in markdown content based on headings.
    
    Args:
        content: The markdown content
        
    Returns:
        List of tuples: (start_pos, end_pos, heading_level, heading_text)
    """
    sections = []
    
    # Find all markdown headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    matches = list(heading_pattern.finditer(content))
    
    for i, match in enumerate(matches):
        start_pos = match.start()
        heading_level = match.group(1)
        heading_text = match.group(2)
        
        # Find the end position (start of next heading or end of content)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        sections.append((start_pos, end_pos, heading_level, heading_text))
    
    # If there's content before the first heading, add it as a section
    if sections and sections[0][0] > 0:
        sections.insert(0, (0, sections[0][0], "", ""))
    elif not sections:
        # No headings found, treat entire content as one section
        sections.append((0, len(content), "", ""))
    
    return sections


def analyze_document_structure(content: str, content_type: str = "general") -> List[Dict]:
    """
    Analyze the structure of a markdown document.
    
    This function creates a "structural map" of the document, identifying sections,
    calculating token counts, and (for academic content) tracking citations and
    footnote relationships.
    
    Args:
        content: The markdown content to analyze
        content_type: Type of content ("general", "academic", "japanese")
        
    Returns:
        List of dictionaries, each representing a section with metadata:
        - section_index: Sequential index starting from 0
        - section: The section heading (e.g., "## Introduction")
        - tokens: Token count of the section
        - cumulative_tokens: Running total of tokens up to and including this section
        - content_preview: Preview showing beginning and end of section content
        - position: Dict with 'start' and 'end' character positions
        - citations_made: (academic only) Citations found in the section
        - footnotes_defined: (academic only) Footnotes defined in the section
    """
    logger.info(f"Analyzing document structure for {content_type} content")
    
    sections = identify_sections(content)
    structural_map = []
    cumulative_tokens = 0
    
    for i, (start_pos, end_pos, heading_level, heading_text) in enumerate(sections):
        # Extract section content
        section_content = content[start_pos:end_pos].strip()
        
        # Skip empty sections
        if not section_content:
            continue
        
        # Calculate token count
        token_count = len(tokenizer.encode(section_content))
        cumulative_tokens += token_count
        
        # Extract the actual content (skipping heading)
        lines = section_content.split('\n')
        content_start_idx = 0
        
        if heading_level and lines and lines[0].startswith(heading_level):
            content_start_idx = 1
        
        # Get actual content (skipping heading)
        actual_content = '\n'.join(lines[content_start_idx:]).strip()
        
        # Create content preview showing beginning and end
        content_preview = create_content_preview(actual_content) if actual_content else ""
        
        # Build section info
        section_info = {
            "section_index": len(structural_map),  # Sequential index
            "section": f"{heading_level} {heading_text}" if heading_level else "(No heading)",
            "tokens": token_count,
            "cumulative_tokens": cumulative_tokens,
            "content_preview": content_preview,
            "position": {
                "start": start_pos,
                "end": end_pos
            }
        }
        
        # For academic content, analyze citations and footnotes
        if content_type == "academic":
            citations = find_citations_in_text(section_content)
            definitions = find_footnote_definitions(section_content)
            
            # Convert sets to sorted lists for consistency
            citations_list = sorted(list(citations))
            definitions_list = sorted(list(definitions))
            
            # Store both raw and formatted versions
            section_info["citations_made"] = citations_list
            section_info["footnotes_defined"] = definitions_list
            
            # Add formatted versions for display
            section_info["citations_formatted"] = format_citation_ranges(citations_list)
            section_info["footnotes_formatted"] = format_citation_ranges(definitions_list)
            
            logger.debug(
                f"Section {section_info['section_index']}: '{section_info['section']}': "
                f"citations: {section_info['citations_formatted'] or 'none'}, "
                f"footnotes: {section_info['footnotes_formatted'] or 'none'}, "
                f"cumulative tokens: {cumulative_tokens:,}"
            )
        
        structural_map.append(section_info)
    
    logger.info(f"Document analysis complete: {len(structural_map)} sections identified")
    
    # Log summary statistics
    total_tokens = cumulative_tokens
    logger.info(f"Total document tokens: {total_tokens:,}")
    
    if content_type == "academic":
        total_citations = sum(len(s.get("citations_made", [])) for s in structural_map)
        total_definitions = sum(len(s.get("footnotes_defined", [])) for s in structural_map)
        logger.info(
            f"Academic analysis: {total_citations} citations, "
            f"{total_definitions} definitions found"
        )
    
    return structural_map


def find_split_positions(
    content: str, split_markers: List[str], structural_map: List[Dict]
) -> List[int]:
    """
    Find the positions in content where splits should occur based on markers.
    This version finds the position of the HEADING before the marker, with fuzzy matching.

    Args:
        content: The original content
        split_markers: List of "starts_with" strings marking new parts
        structural_map: The structural map of the document

    Returns:
        List of character positions where splits should occur
    """
    split_positions = [0]  # First part always starts at beginning

    # Create a lookup table from starts_with to the full section heading
    marker_to_heading = {
        section.get("starts_with", ""): section["section"] for section in structural_map
    }

    for marker in split_markers:
        heading = marker_to_heading.get(marker)
        if not heading:
            logger.warning(f"Could not find heading for split marker: '{marker[:50]}...'")
            # Fallback to original behavior if heading not found
            pos = content.find(marker)
            if pos != -1:
                split_positions.append(pos)
            continue

        # Find the position of the heading in the content
        # We need to be careful with headings that might appear multiple times.
        # Let's search from the last found position to be safer.
        last_pos = split_positions[-1] if len(split_positions) > 1 else 0
        pos = content.find(heading, last_pos)

        if pos != -1:
            # The position should be the start of the heading line
            split_positions.append(pos)
            logger.debug(f"Found split point at heading position {pos}: '{heading}'")
        else:
            logger.warning(f"Could not find heading for split marker: '{heading}'")
            # Fallback to marker search
            pos = content.find(marker, last_pos)
            if pos != -1:
                split_positions.append(pos)

    # Add end position
    split_positions.append(len(content))

    # Sort and remove duplicates
    split_positions = sorted(set(split_positions))

    return split_positions


def find_split_positions_by_indices(
    content: str, split_indices: List[int], structural_map: List[Dict]
) -> List[int]:
    """
    Find the positions in content where splits should occur based on section indices.
    
    Args:
        content: The original content
        split_indices: List of section indices where new parts should begin
        structural_map: The structural map of the document
    
    Returns:
        List of character positions where splits should occur
    """
    split_positions = [0]  # First part always starts at beginning
    
    for index in split_indices:
        if index < 0 or index >= len(structural_map):
            logger.warning(f"Invalid section index: {index} (map has {len(structural_map)} sections)")
            continue
        
        # Get the character position from the structural map
        position = structural_map[index]["position"]["start"]
        split_positions.append(position)
        logger.debug(
            f"Split at section {index} ('{structural_map[index]['section']}'): "
            f"position {position}"
        )
    
    # Add end position
    split_positions.append(len(content))
    
    # Sort and remove duplicates
    split_positions = sorted(set(split_positions))
    
    return split_positions
