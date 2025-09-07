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
    
    for line in lines:
        line = line.strip()
        
        # Standard markdown definition [^1]:
        match = re.match(r'^\[\^(\w+)\]:', line)
        if match:
            definitions.add(match.group(1))
            continue
        
        # Academic style [1] at line start (followed by space/tab)
        match = re.match(r'^\[(\d+)\]\s', line)
        if match:
            definitions.add(match.group(1))
            continue
        
        # Numbered list style 1. (be more careful with this)
        # Only treat as footnote if it's in a Notes/References section or follows a pattern
        match = re.match(r'^(\d+)\.\s', line)
        if match:
            # Check if we're likely in a footnotes section
            # This is a heuristic - we can refine it based on context
            number = match.group(1)
            # Only consider single or double digit numbers as likely footnotes
            if len(number) <= 2:
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
        - section: The section heading (e.g., "## Introduction")
        - tokens: Token count of the section
        - starts_with: First sentence/words of the section
        - citations_made: (academic only) Citations found in the section
        - footnotes_defined: (academic only) Footnotes defined in the section
    """
    logger.info(f"Analyzing document structure for {content_type} content")
    
    sections = identify_sections(content)
    structural_map = []
    
    for i, (start_pos, end_pos, heading_level, heading_text) in enumerate(sections):
        # Extract section content
        section_content = content[start_pos:end_pos].strip()
        
        # Skip empty sections
        if not section_content:
            continue
        
        # Calculate token count
        token_count = len(tokenizer.encode(section_content))
        
        # Extract the first sentence/words for identification
        # Skip the heading line itself if present
        lines = section_content.split('\n')
        content_start_idx = 0
        
        if heading_level and lines and lines[0].startswith(heading_level):
            content_start_idx = 1
        
        # Get actual content (skipping heading)
        actual_content = '\n'.join(lines[content_start_idx:]).strip()
        first_sentence = extract_first_sentence(actual_content) if actual_content else ""
        
        # Build section info
        section_info = {
            "section": f"{heading_level} {heading_text}" if heading_level else "(No heading)",
            "tokens": token_count,
            "starts_with": first_sentence
        }
        
        # For academic content, analyze citations and footnotes
        if content_type == "academic":
            citations = find_citations_in_text(section_content)
            definitions = find_footnote_definitions(section_content)
            
            # Convert sets to sorted lists for consistency
            section_info["citations_made"] = sorted(list(citations))
            section_info["footnotes_defined"] = sorted(list(definitions))
            
            logger.debug(
                f"Section '{section_info['section']}': "
                f"{len(citations)} citations, {len(definitions)} definitions"
            )
        
        structural_map.append(section_info)
    
    logger.info(f"Document analysis complete: {len(structural_map)} sections identified")
    
    # Log summary statistics
    total_tokens = sum(s["tokens"] for s in structural_map)
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
    This version finds the position of the HEADING before the marker.

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
        section["starts_with"]: section["section"] for section in structural_map
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