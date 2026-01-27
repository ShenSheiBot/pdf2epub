"""
Pure functions for generating polish prompts.

These functions are stateless and easy to unit test.
"""

import re
from typing import Dict, Optional


def detect_content_type(content: str) -> str:
    """
    Auto-detect content type based on content characteristics.

    Args:
        content: The content to analyze

    Returns:
        Content type: "japanese", "academic", or "general"
    """
    # Check for Japanese characters
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', content)
    if len(japanese_chars) > len(content) * 0.1:  # More than 10% Japanese characters
        return "japanese"

    # Check for academic indicators
    footnote_indicators = [
        r'\[\^\d+\]',  # Markdown footnotes [^1]
        r'\$\^\{?\d+\}?\$',  # LaTeX style superscripts
        r'\[\d+\]',  # Bracketed references
        r'References\s*\n',  # References section
        r'Bibliography\s*\n',  # Bibliography section
        r'\\cite\{',  # LaTeX citations
    ]

    academic_score = sum(
        1 for pattern in footnote_indicators
        if re.search(pattern, content[:5000])  # Check first 5000 chars
    )

    if academic_score >= 2:
        return "academic"

    return "general"


def create_academic_polish_prompt(
    chapter_name: str,
    book_title: str = "",
    part_idx: int = 1,
    total_parts: int = 1,
    previous_part_context: Optional[Dict[str, str]] = None
) -> str:
    """
    Create prompt specifically for academic content with references.

    Args:
        chapter_name: Name of the chapter
        book_title: Title of the book
        part_idx: Current part index (1-based)
        total_parts: Total number of parts
        previous_part_context: Context from previous part (for sequential mode)

    Returns:
        The prompt string
    """
    book_info = f' from the book titled "{book_title}"' if book_title else ""

    prompt = f"""You are an expert academic document editor specializing in scholarly texts. Polish this OCR-extracted academic content from "{chapter_name}"{book_info}.

Your tasks for ACADEMIC content:

1. **Remove page artifacts**:
   - Delete page numbers, headers, and footers
   - Remove horizontal separators between pages
   - Join sentences broken by page boundaries

2. **Preserve academic structure**:
   - Main chapter title: # (H1)
   - Sections: ## (H2), subsections: ### (H3)
   - Keep abstract, introduction, conclusion sections intact
   - Preserve figure/table captions and numbering

3. **Handle citations and footnotes CAREFULLY**:
   - INLINE CITATIONS: Convert citation markers ($^1$, ¹, {{ }}^{{1}}, etc.) to [^1] format
   - IMPORTANT: Text immediately after a citation marker is NOT the footnote definition
   - Example of WRONG interpretation:
     * Input: "This is discussed by Smith$^5$ The next sentence continues..."
     * WRONG: "This is discussed by Smith[^5]" then "[^5]: The next sentence continues..."
     * RIGHT: "This is discussed by Smith[^5] The next sentence continues..."
   - FOOTNOTE DEFINITIONS: Only create [^1]: format for ACTUAL footnotes found at:
     * Bottom of pages (separated from main text)
     * End of chapters in dedicated Notes/References sections
     * Clearly marked footnote sections
   - Convert OCR footnote formats to markdown:
     * "[footnote] $^N$ text" or "[footnote] $ ^{{N}} $ text" → "[^N]: text"
     * "^N text" at page bottom → "[^N]: text"
   - Preserve exact footnote numbering from the source
   - If footnote is at the bottom of page, group and move them to the end of section
   - Never invent or add missing footnotes
   - Keep footnote index as-is, do not try to mitigate duplicate [1] in the same section

4. **Preserve academic elements**:
   - Keep equations, formulas, and mathematical notation
   - Preserve code blocks and technical examples
   - Maintain definition lists and theorems
   - Keep cross-references ("see Section 2.3")

5. **Organize bibliography**:
   - Move all footnotes to "### Notes" section if they exist (use ### to keep as subsection, not chapter)
   - Organize references under "### References" if present
   - Format citations consistently:
     * Books: Author(s). (Year). *Title*. Publisher.
     * Articles: Author(s). (Year). "Title." *Journal*, Volume(Issue), pages.
   - Final structure: Main Content → ### Notes → ### References

6. **Quality checks**:
   - Ensure all citations have corresponding footnotes
   - Verify footnote numbering is sequential
   - Check that academic terminology is preserved"""

    # Add context for multi-part chapters
    if total_parts > 1:
        prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""

        if part_idx > 1:
            prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2).
Convert any # (H1) headings to ## (H2). You don't necessarily need to start with ##."""

            # Add context from previous part if in sequential mode
            if previous_part_context:
                prompt += f"""

CONTEXT FROM PREVIOUS PART: The previous part has been polished and ends with:
...{previous_part_context['polished'][-500:]}

Please ensure continuity with the previous part."""

    prompt += """

IMPORTANT: Return ONLY the polished markdown. Do not add explanations.
Preserve all tables, figures, and images unless duplicated.

Polish the following academic content:"""

    return prompt


def create_academic_global_prompt(
    chapter_name: str,
    book_title: str = "",
    part_idx: int = 1,
    total_parts: int = 1,
    previous_part_context: Optional[Dict[str, str]] = None
) -> str:
    """
    Create prompt for academic content when footnotes are in separate Notes chapter.

    Args:
        chapter_name: Name of the chapter
        book_title: Title of the book
        part_idx: Current part index (1-based)
        total_parts: Total number of parts
        previous_part_context: Context from previous part (for sequential mode)

    Returns:
        The prompt string
    """
    book_info = f' from the book titled "{book_title}"' if book_title else ""

    prompt = f"""You are an expert academic document editor specializing in scholarly texts. Polish this OCR-extracted academic content from "{chapter_name}"{book_info}.

Your tasks for ACADEMIC content with centralized footnotes:

1. **Remove page artifacts**:
   - Delete page numbers, headers, and footers
   - Remove horizontal separators between pages
   - Join sentences broken by page boundaries

2. **Preserve academic structure**:
   - Main chapter title: # (H1)
   - Sections: ## (H2), subsections: ### (H3)
   - Keep abstract, introduction, conclusion sections intact
   - Preserve figure/table captions and numbering

3. **Handle citations ONLY**:
   - Convert inline citation markers ($^1$, ¹, {{ }}^{{1}}, etc.) to [^1] format
   - IMPORTANT: Just convert the markers - footnotes are managed in a separate Notes chapter
   - Do NOT look for or create footnote definitions ([^1]: text)
   - Text immediately after a citation is part of the main content, not a footnote

4. **Preserve academic elements**:
   - Keep equations, formulas, and mathematical notation
   - Preserve code blocks and technical examples
   - Maintain definition lists and theorems
   - Keep cross-references ("see Section 2.3")

5. **Quality checks**:
   - Ensure academic terminology is preserved
   - Verify proper markdown formatting"""

    # Add context for multi-part chapters
    if total_parts > 1:
        prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""

        if part_idx > 1:
            prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2).
Convert any # (H1) headings to ## (H2)."""

            # Add context from previous part if in sequential mode
            if previous_part_context:
                prompt += f"""

CONTEXT FROM PREVIOUS PART: The previous part has been polished and ends with:
...{previous_part_context['polished'][-500:]}

Please ensure continuity with the previous part."""

    prompt += """

IMPORTANT: Return ONLY the polished markdown. Do not add explanations.
Preserve all tables, figures, and images unless duplicated.

Polish the following academic content:"""

    return prompt


def create_notes_chapter_prompt(
    chapter_name: str,
    book_title: str = "",
    part_idx: int = 1,
    total_parts: int = 1,
    previous_part_context: Optional[Dict[str, str]] = None
) -> str:
    """
    Create prompt specifically for Notes/References chapters.

    Args:
        chapter_name: Name of the chapter
        book_title: Title of the book
        part_idx: Current part index (1-based)
        total_parts: Total number of parts
        previous_part_context: Context from previous part

    Returns:
        The prompt string
    """
    book_info = f' from "{book_title}"' if book_title else ""

    prompt = f"""You are an expert editor processing a Notes/References section{book_info}. This chapter contains all footnotes and references for the entire book.

Your tasks:

1. **Clean up OCR artifacts**:
   - Remove page numbers, headers, and footers
   - Join sentences broken across pages
   - Fix line breaks within footnote entries

2. **Format footnote entries properly**:
   - Convert numbered notes to markdown format:
     * Input: "1. Author, Title..." → Output: "[^1]: Author, Title..."
     * Input: "[1] Author, Title..." → Output: "[^1]: Author, Title..."
     * Input: "1 Author, Title..." → Output: "[^1]: Author, Title..."
   - Each [^n]: definition should start on its own line
   - Keep multi-line footnotes properly indented

3. **Preserve structure**:
   - Keep section headings like "## Introduction", "## Chapter 1", etc.
   - Maintain the organization by chapter/section
   - Keep the original footnote numbering exactly as it appears

4. **Handle citations properly**:
   - Format book citations: Author. *Title*. Publisher, Year.
   - Format article citations: Author. "Article Title." *Journal*, vol(issue), Year, pages.
   - Preserve all bibliographic details

5. **DO NOT**:
   - Change footnote numbers
   - Reorder or reorganize content
   - Add or remove footnotes
   - Convert section headings to footnote format"""

    # Add context for multi-part chapters
    if total_parts > 1:
        prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of the Notes section."""

        if part_idx > 1 and previous_part_context:
            prompt += f"""

CONTEXT FROM PREVIOUS PART: The previous part ends with:
...{previous_part_context['polished'][-500:]}

Please ensure continuity with the previous part."""

    prompt += """

IMPORTANT: Return ONLY the formatted markdown.

Format the following Notes/References content:"""

    return prompt


def create_japanese_polish_prompt(
    chapter_name: str,
    book_title: str = "",
    part_idx: int = 1,
    total_parts: int = 1,
    previous_part_context: Optional[Dict[str, str]] = None
) -> str:
    """
    Create prompt specifically for Japanese content with furigana.

    Args:
        chapter_name: Name of the chapter
        book_title: Title of the book
        part_idx: Current part index (1-based)
        total_parts: Total number of parts
        previous_part_context: Context from previous part

    Returns:
        The prompt string
    """
    book_info = f' from "{book_title}"' if book_title else ""

    prompt = f"""You are an expert editor specializing in Japanese literature and light novels. Polish this OCR-extracted Japanese content from "{chapter_name}"{book_info}.

Your tasks for JAPANESE content:

1. **Remove page artifacts**:
   - Delete page numbers and headers/footers
   - Remove separators (---) between pages and join sentences
   - Join continuous sentences broken by OCR
   - Handle vertical text OCR artifacts

2. **Preserve Japanese text features**:
   - KEEP all furigana/ruby text: 一人(ひとり), 今更(いまさら), 幼馴染(おさななじみ)
   - DO NOT add new furigana not in the original
   - DO NOT change () of furigana to （）
   - DO NOT remove furigana from the original text
   - DO NOT remove ルビ芸 like 妄想 in「何もないからこ(妄)ういう(想)話に逃(に)げてんじゃん!」

3. **Tables**:
   - Convert HTML tables (<table>, <tr>, <td>, <th>) to markdown tables
   - Use proper markdown table syntax with | for columns and --- for headers
   - Preserve table structure, merging cells where appropriate
   - For complex tables with merged cells, use simplified markdown representation

4. **Images and illustrations**:
   - PRESERVE ALL IMAGE LINKS EXACTLY AS THEY ARE
   - Keep markdown image syntax: ![Image](../images/filename.png)
   - DO NOT replace image links with [illustration] or any other placeholder
   - DO NOT modify image paths or filenames
   - Verify furigana is attached to correct kanji

5. **Fix incorrect headings from vertical text OCR**:
   - Vertical text OCR often incorrectly marks normal text as headings (## or #)
   - REMOVE the ## or # prefix if the line is clearly NOT a heading, such as:
     * Dialogue: 「...」 or lines starting with quotes
     * Exclamations or onomatopoeia: やめろっ!, あんっ♡, etc.
     * Sentence fragments: lines that are clearly mid-sentence or incomplete
     * Sound effects or emotional expressions with ♡, ♪, etc.
   - KEEP ## only for actual section/scene headings (e.g., chapter titles, scene breaks)
   - When in doubt, remove the heading marker - normal paragraphs are safer than false headings
"""

    # Add context for multi-part chapters
    if total_parts > 1:
        prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""

        if part_idx > 1:
            prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2).
Convert any # (H1) headings to ## (H2)."""

            # Add context from previous part if in sequential mode
            if previous_part_context:
                prompt += f"""

CONTEXT FROM PREVIOUS PART: The previous part has been polished and ends with:
...{previous_part_context['polished'][-500:]}

Please ensure continuity with the previous part."""

    prompt += """

IMPORTANT: Return ONLY the polished markdown. Do not add explanations.
Preserve all images and illustrations.

Polish the following Japanese content:"""

    return prompt


def create_general_polish_prompt(
    chapter_name: str,
    book_title: str = "",
    part_idx: int = 1,
    total_parts: int = 1,
    previous_part_context: Optional[Dict[str, str]] = None
) -> str:
    """
    Create prompt for general content (fallback).

    Args:
        chapter_name: Name of the chapter
        book_title: Title of the book
        part_idx: Current part index (1-based)
        total_parts: Total number of parts
        previous_part_context: Context from previous part

    Returns:
        The prompt string
    """
    book_info = f' from "{book_title}"' if book_title else ""

    prompt = f"""You are an expert document editor. Polish this OCR-extracted content from "{chapter_name}"{book_info}.

Your tasks:

1. **Remove page artifacts**:
   - Delete page numbers, headers, and footers
   - Remove page separators (---)
   - Join sentences broken across pages

2. **Fix structure**:
   - Main title: # (H1)
   - Sections: ## (H2), subsections: ### (H3)
   - Remove excessive blank lines

3. **Clean up text**:
   - Fix obvious OCR errors
   - Join hyphenated words at line breaks
   - Preserve emphasis (*italic*, **bold**)

4. **Preserve content**:
   - Keep all images and tables
   - Maintain lists and quotes
   - Preserve code blocks if present"""

    # Add context for multi-part chapters
    if total_parts > 1:
        prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""

        if part_idx > 1:
            prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2)."""

            # Add context from previous part if in sequential mode
            if previous_part_context:
                prompt += f"""

CONTEXT FROM PREVIOUS PART: The previous part has been polished and ends with:
...{previous_part_context['polished'][-500:]}

Please ensure continuity with the previous part."""

    prompt += """

Return ONLY the polished markdown.

Polish the following content:"""

    return prompt


def create_polish_prompt(
    chapter_name: str,
    book_title: str = "",
    part_idx: int = 1,
    total_parts: int = 1,
    content: str = "",
    content_type: str = "auto",
    use_global_footnotes: bool = False,
    is_notes_chapter: bool = False,
    previous_part_context: Optional[Dict[str, str]] = None
) -> str:
    """
    Create the appropriate polish prompt based on content type and context.

    This is the main entry point for prompt generation.

    Args:
        chapter_name: Name of the chapter
        book_title: Title of the book
        part_idx: Current part index (1-based)
        total_parts: Total number of parts
        content: The content to polish (used for auto-detection)
        content_type: Type of content ("academic", "japanese", "general", "auto")
        use_global_footnotes: Whether footnotes are in a separate Notes chapter
        is_notes_chapter: Whether this is the Notes chapter itself
        previous_part_context: Context from previous part

    Returns:
        The prompt string
    """
    # Special handling for notes chapters
    if is_notes_chapter:
        return create_notes_chapter_prompt(
            chapter_name, book_title, part_idx, total_parts, previous_part_context
        )

    # Determine content type
    if content_type == "auto" and content:
        content_type = detect_content_type(content)
    elif content_type == "auto":
        content_type = "general"

    # Route to appropriate prompt creator
    if content_type == "academic":
        if use_global_footnotes:
            return create_academic_global_prompt(
                chapter_name, book_title, part_idx, total_parts, previous_part_context
            )
        else:
            return create_academic_polish_prompt(
                chapter_name, book_title, part_idx, total_parts, previous_part_context
            )
    elif content_type == "japanese":
        return create_japanese_polish_prompt(
            chapter_name, book_title, part_idx, total_parts, previous_part_context
        )
    else:
        return create_general_polish_prompt(
            chapter_name, book_title, part_idx, total_parts, previous_part_context
        )
