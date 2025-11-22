"""
Structure analysis for PDF books.

Provides:
- Initial PDF structure analysis (extract TOC tree)
- Re-breakdown when verification fails
- Discovery of hidden subsections
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google.genai.types import Part
from loguru import logger

from ..utils.network_utils import GeminiClient
from .toc_tree import TOCNode, dict_list_to_toc_tree


class StructureAnalyzer:
    """
    Analyzes PDF structure and discovers subsections.
    """

    def __init__(
        self,
        client: GeminiClient,
        structure_model: str = "gemini-2.5-pro",
        analysis_model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the structure analyzer.

        Args:
            client: GeminiClient for API calls
            structure_model: Model for full PDF analysis (needs large context)
            analysis_model: Model for re-breakdown and subsection discovery
        """
        self.client = client
        self.structure_model = structure_model
        self.analysis_model = analysis_model

    def analyze_pdf_structure(
        self,
        pdf_path: Path,
        book_title: str
    ) -> Tuple[List[TOCNode], Dict]:
        """
        Analyze PDF structure and extract recursive TOC tree.

        Args:
            pdf_path: Path to PDF file (should be preprocessed with page patches)
            book_title: Book title for prompts

        Returns:
            Tuple of (list of top-level TOCNodes, book metadata dict)
        """
        prompt = f"""
Analyze this book PDF with title "{book_title}" and provide a detailed breakdown of its structure.

**CRITICAL**: Extract the COMPLETE hierarchical structure from the table of contents.
- Extract ALL levels: Part, Chapter, Section, Subsection, etc.
- DO NOT create artificial subdivisions beyond what the TOC shows
- Use PDF page numbers (not printed page numbers)

Include:
1. Author name(s)
2. Cover page (page number)
3. Table of contents (page numbers)
4. All chapters with their COMPLETE substructure as a recursive tree
5. Back cover page (page number)

Additionally identify special chapter types:
- If a chapter consists ONLY of footnotes/endnotes for other chapters, add "type": "notes"
- If any chapter's notes are at the end of itself, then there should be NO notes chapter
- A book contains at most one notes chapter
- Abbreviations, Bibliography, Index, or Summary Table are NOT considered as notes
- Only literal "Notes" or "Endnotes" chapters with [1], [2], [3]... definitions are considered as notes
- Regular content chapters should not have a "type" field

Also analyze content characteristics:
- **language**: Primary language (e.g., "english", "japanese", "chinese")
- **is_vertical_text**: true if vertical text layout (縦書き)
- **has_footnotes**: true if content has footnotes/citations

Return JSON:
{{
    "author": string,
    "language": string,
    "is_vertical_text": boolean,
    "has_footnotes": boolean,
    "cover_page": {{"page_number": int}},
    "table_of_contents": {{
        "start_page": int,
        "end_page": int
    }},
    "chapters": [
        {{
            "title": string,
            "start_page": int,
            "end_page": int,
            "level": int,
            "type": string,  // Optional: "notes" for footnote chapters
            "children": [    // Recursive - can have unlimited depth
                {{
                    "title": string,
                    "start_page": int,
                    "end_page": int,
                    "level": int,
                    "children": [...]
                }}
            ]
        }}
    ],
    "back_cover": {{"page_number": int}}
}}

**IMPORTANT**:
- The "children" field is recursive and can contain unlimited levels
- If a section has no subsections, its "children" should be an empty array []
- Preserve the original language for all titles and author names
- Note that nearby chapters may overlap if there are no page breaks

Example of a notes chapter:
{{
    "title": "Notes",
    "start_page": 250,
    "end_page": 275,
    "level": 1,
    "type": "notes"
}}
"""

        # Read PDF
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        parts = [
            prompt,
            Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
        ]

        # Generate with JSON response
        generation_config = self.client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        response_text = self.client.generate_content_stream(
            model=self.structure_model,
            contents=parts,
            config=generation_config,
            operation_name="PDF structure analysis"
        )

        result = json.loads(response_text)

        # Validate and fix notes type - remove from non-notes chapters
        self._fix_invalid_notes_type(result.get('chapters', []))

        # Convert to TOCNode tree
        toc_tree = dict_list_to_toc_tree(result.get('chapters', []))

        # Extract metadata
        book_metadata = {
            'author': result.get('author', 'Unknown'),
            'language': result.get('language', 'english'),
            'is_vertical_text': result.get('is_vertical_text', False),
            'has_footnotes': result.get('has_footnotes', False),
            'cover_page': result.get('cover_page'),
            'table_of_contents': result.get('table_of_contents'),
            'back_cover': result.get('back_cover'),
            'book_title': book_title
        }

        logger.info(f"Extracted {len(toc_tree)} top-level chapters")
        return toc_tree, book_metadata

    def _fix_invalid_notes_type(self, chapters: List[Dict]):
        """
        Remove type='notes' from chapters that are clearly not notes.

        Bibliography, Index, Abbreviations, Summary Table should not be marked as notes.
        Only literal "Notes" or "Endnotes" chapters should have this type.
        """
        invalid_keywords = ['bibliography', 'index', 'abbreviation', 'summary', 'glossary', 'appendix']
        valid_keywords = ['notes', 'endnotes']

        for chapter in chapters:
            if chapter.get('type') == 'notes':
                title_lower = chapter.get('title', '').lower()
                # Check if title contains invalid keywords
                has_invalid = any(kw in title_lower for kw in invalid_keywords)
                # Check if title contains valid keywords
                has_valid = any(kw in title_lower for kw in valid_keywords)

                if has_invalid and not has_valid:
                    logger.warning(
                        f"Removing invalid type='notes' from '{chapter['title']}' "
                        f"(Bibliography/Index/etc are not notes chapters)"
                    )
                    del chapter['type']

            # Recursively check children
            if chapter.get('children'):
                self._fix_invalid_notes_type(chapter['children'])

    def rebreakdown_chapter(
        self,
        start_page: int,
        end_page: int,
        pages_dir: Path,
        chapter_title: str
    ) -> List[TOCNode]:
        """
        Re-analyze a chapter's structure when verification fails.

        Sends the chapter's page content to LLM for re-analysis.

        Args:
            start_page: First page of the chapter
            end_page: Last page of the chapter
            pages_dir: Directory containing page files
            chapter_title: Title of the chapter being re-analyzed

        Returns:
            List of TOCNodes representing the chapter's structure
        """
        # Collect page content
        content_parts = []
        for page_num in range(start_page, end_page + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if page_file.exists():
                page_content = page_file.read_text(encoding='utf-8')
                content_parts.append(f"--- Page {page_num} ---\n{page_content}")

        full_content = "\n\n".join(content_parts)

        prompt = f"""
以下是"{chapter_title}"（第 {start_page}-{end_page} 页）的内容：

{full_content}

**任务**：重新分析这个章节的结构。

找出所有的小节标题，提取它们的：
- 标题
- 起始页码
- 结束页码
- 层级

返回 JSON：
{{
    "sections": [
        {{
            "title": string,
            "start_page": int,
            "end_page": int,
            "level": int,
            "children": []
        }}
    ]
}}

**重要**：
- 只提取实际存在的小节标题，不要创造
- 如果没有小节，返回空数组
- 页码使用 PDF 页码（已在内容中标注）
"""

        generation_config = self.client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.client.generate_content_stream(
                model=self.analysis_model,
                contents=prompt,
                config=generation_config,
                operation_name=f"Re-breakdown: {chapter_title}"
            )

            result = json.loads(response_text)
            sections = result.get('sections', [])

            if sections:
                logger.info(f"Re-breakdown found {len(sections)} sections in '{chapter_title}'")
                return dict_list_to_toc_tree(sections)
            else:
                logger.info(f"Re-breakdown found no sections in '{chapter_title}'")
                return []

        except Exception as e:
            logger.error(f"Re-breakdown failed for '{chapter_title}': {e}")
            return []

