"""
Structure analysis for PDF books.

Provides:
- Initial PDF structure analysis (extract TOC tree)
- Re-breakdown when verification fails
- Discovery of hidden subsections
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google.genai.types import Part
from loguru import logger

from ..utils.common import parse_llm_json
from ..utils.llm_client import BoundLLMClient
from .toc_tree import TOCNode, dict_list_to_toc_tree
from .refiner_state import RefinerState


class StructureAnalyzer:
    """
    Analyzes PDF structure and discovers subsections.
    """

    def __init__(
        self,
        structure_client: BoundLLMClient,
        structure_model: str,
        toc_model: str,
        analysis_client: BoundLLMClient,
        analysis_model: str
    ):
        """
        Initialize the structure analyzer.

        Args:
            structure_client: BoundLLMClient for PDF operations (needs PDF support)
            structure_model: Model for full PDF analysis (needs large context)
            toc_model: Model for TOC detection/extraction (cheaper, still needs PDF)
            analysis_client: BoundLLMClient for re-breakdown (text only, no PDF needed)
            analysis_model: Model for re-breakdown and subsection discovery
        """
        self.structure_client = structure_client
        self.structure_model = structure_model
        self.toc_model = toc_model
        self.analysis_client = analysis_client
        self.analysis_model = analysis_model

    def detect_toc_location(self, pdf_path: Path) -> Optional[Dict]:
        """
        Detect the location of table of contents in the PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with {has_toc, toc_start, toc_end} or None if detection fails
        """
        prompt = """
Analyze this PDF and find the Table of Contents (TOC) pages.

Look for pages that contain:
- A list of chapter/section titles with page numbers
- Typically titled "Table of Contents", "Contents", "目次", "Table des matières", "Sommaire", "Inhalt", etc.
- Usually appears near the beginning or end of the book

Return JSON:
{
    "has_toc": boolean,  // true if a TOC exists
    "toc_start": int,    // PDF page number where TOC starts (1-indexed)
    "toc_end": int       // PDF page number where TOC ends (1-indexed)
}

If no TOC exists, return: {"has_toc": false, "toc_start": null, "toc_end": null}

**IMPORTANT**: Use PDF page numbers from the "PDF Page: X" labels, not printed page numbers.
"""

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        parts = [
            prompt,
            Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
        ]

        # Use structure client + model since this needs PDF support
        generation_config = self.structure_client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.structure_client.generate_content_stream(
                model=self.toc_model,
                contents=parts,
                config=generation_config,
                operation_name="TOC location detection"
            )
            result = parse_llm_json(response_text, operation_name="TOC location detection")

            if result.get('has_toc'):
                logger.info(f"TOC detected: pages {result['toc_start']}-{result['toc_end']}")
            else:
                logger.info("No TOC detected in PDF")

            return result
        except Exception as e:
            logger.error(f"TOC detection failed: {e}")
            return None

    def extract_toc_structure(self, pdf_path: Path, toc_start: int, toc_end: int) -> Optional[Dict]:
        """
        Extract TOC structure (titles and hierarchy only, NO page numbers).

        Args:
            pdf_path: Path to PDF file (will try to use original uncompressed version)
            toc_start: First page of TOC
            toc_end: Last page of TOC

        Returns:
            Dict with chapters list (no page numbers) or None if extraction fails
        """
        # Step 2a: Use original PDF for better quality, extract only TOC pages
        original_pdf = pdf_path.parent / "input_original.pdf"
        pdf_to_use = original_pdf if original_pdf.exists() else pdf_path

        toc_pages = list(range(toc_start, toc_end + 1))
        pdf_data = self._prepare_pdf(pdf_to_use, include_pages=toc_pages)

        if pdf_data is None:
            logger.warning("Failed to extract TOC pages, using full compressed PDF")
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
        else:
            logger.info(f"Using {'original' if pdf_to_use == original_pdf else 'compressed'} PDF, TOC pages only")

        prompt = f"""
Extract the structure from these Table of Contents pages.

**CRITICAL**: Extract ONLY the hierarchical structure - titles and their nesting levels.
**DO NOT include any page numbers** - we will determine those separately.

Return JSON:
{{
    "author": string,  // Author name if visible
    "chapters": [
        {{
            "title": string,  // Chapter/section title
            "level": int,     // 1 for top-level, 2 for subsections, etc.
            "children": [...]  // Nested structure, same format
        }}
    ]
}}

**IMPORTANT**:
- Preserve the exact titles as written in the TOC
- Maintain the correct hierarchy (Parts > Chapters > Sections > Subsections)
- Do NOT include page numbers in the output
- Keep original language for all titles
"""

        parts = [
            prompt,
            Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
        ]

        # Use structure client + model since this needs PDF support
        generation_config = self.structure_client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.structure_client.generate_content_stream(
                model=self.toc_model,
                contents=parts,
                config=generation_config,
                operation_name="TOC structure extraction"
            )
            result = parse_llm_json(response_text, operation_name="TOC structure extraction")

            chapter_count = len(result.get('chapters', []))
            logger.info(f"Extracted {chapter_count} top-level items from TOC")

            return result
        except Exception as e:
            logger.error(f"TOC structure extraction failed: {e}")
            return None

    def match_toc_with_content(
        self,
        pdf_path: Path,
        toc_structure: Dict,
        book_title: str,
        toc_start: int,
        toc_end: int
    ) -> Dict:
        """
        Match TOC structure with actual PDF content to determine page numbers.

        Args:
            pdf_path: Path to PDF file
            toc_structure: TOC structure without page numbers
            book_title: Book title
            toc_start: TOC start page (to exclude from search)
            toc_end: TOC end page (to exclude from search)

        Returns:
            Complete structure with page numbers
        """
        import json
        toc_json = json.dumps(toc_structure.get('chapters', []), ensure_ascii=False, indent=2)

        prompt = f"""
Match this Table of Contents structure with the actual content in the PDF to determine correct PDF page numbers.

**Book Title**: {book_title}

**TOC Structure** (extracted from table of contents, NO page numbers):
{toc_json}

**Your Task**:
1. Find where each chapter/section actually starts in the PDF
2. Determine the correct PDF page numbers (from "PDF Page: X" labels)
3. Calculate end pages based on where the next section starts

Return JSON:
{{
    "author": string,
    "language": string,  // e.g., "english", "french", "japanese"
    "is_vertical_text": boolean,
    "has_footnotes": boolean,
    "cover_page": {{"page_number": int}},
    "table_of_contents": {{
        "start_page": {toc_start},
        "end_page": {toc_end}
    }},
    "chapters": [
        {{
            "title": string,
            "start_page": int,  // REQUIRED: PDF page number where this section starts
            "end_page": int,    // REQUIRED: PDF page number where this section ends
            "level": int,
            "type": string,     // Optional: "notes" for endnotes chapters
            "children": [...]   // Same structure - EVERY child MUST also have start_page and end_page
        }}
    ],
    "back_cover": {{"page_number": int}}
}}

**CRITICAL**:
- Use PDF page numbers from "PDF Page: X" labels, NOT printed page numbers
- The printed page numbers in the original TOC are WRONG - ignore them
- Find each chapter title in the actual content and note its PDF page
"""

        # Step 2b: Use compressed PDF, exclude TOC pages to avoid confusion
        toc_pages = list(range(toc_start, toc_end + 1))
        pdf_data = self._prepare_pdf(pdf_path, exclude_pages=toc_pages)

        if pdf_data is None:
            logger.warning("Failed to exclude TOC pages, using full PDF")
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
        else:
            logger.info(f"Using compressed PDF, excluded TOC pages {toc_start}-{toc_end}")

        parts = [
            prompt,
            Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
        ]

        generation_config = self.structure_client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        response_text = self.structure_client.generate_content_stream(
            model=self.structure_model,
            contents=parts,
            config=generation_config,
            operation_name="TOC page matching"
        )

        result = parse_llm_json(response_text, operation_name="TOC page matching")

        # Step 2c: Validate and retry if issues found
        issues = self._validate_toc_structure(result.get('chapters', []))
        if issues:
            logger.warning(f"Found {len(issues)} issues in TOC structure, requesting fix...")
            for issue in issues[:5]:  # Log first 5 issues
                logger.warning(f"  {issue}")

            # Build error message for LLM
            error_msg = "The following issues were found in your response:\n\n"
            error_msg += "\n".join(f"- {issue}" for issue in issues)
            error_msg += "\n\nPlease fix these issues and return the corrected JSON."

            # Append to conversation and retry
            parts.append(response_text)  # Add previous response
            parts.append(error_msg)  # Add error feedback

            response_text = self.structure_client.generate_content_stream(
                model=self.structure_model,
                contents=parts,
                config=generation_config,
                operation_name="TOC page matching (retry)"
            )

            result = parse_llm_json(response_text, operation_name="TOC page matching (retry)")

            # Check if issues are fixed
            remaining_issues = self._validate_toc_structure(result.get('chapters', []))
            if remaining_issues:
                logger.warning(f"After retry, {len(remaining_issues)} issues remain")

        return result

    def _prepare_pdf(
        self,
        pdf_path: Path,
        include_pages: Optional[List[int]] = None,
        exclude_pages: Optional[List[int]] = None
    ) -> Optional[bytes]:
        """
        Prepare PDF for LLM by selecting/excluding specific pages.

        Args:
            pdf_path: Path to the source PDF
            include_pages: List of pages to include (1-indexed). If None, include all.
            exclude_pages: List of pages to exclude (1-indexed). Applied after include.

        Returns:
            PDF bytes, or None if preparation fails
        """
        try:
            import fitz  # pymupdf
            import tempfile
            import os

            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            # Determine which pages to keep
            if include_pages is not None:
                pages_to_keep = set(include_pages)
            else:
                pages_to_keep = set(range(1, total_pages + 1))

            # Apply exclusions
            if exclude_pages:
                pages_to_keep -= set(exclude_pages)

            if not pages_to_keep:
                logger.error("No pages left after filtering")
                doc.close()
                return None

            # Determine which pages to delete (1-indexed)
            pages_to_delete = set(range(1, total_pages + 1)) - pages_to_keep

            # Delete pages in reverse order to avoid index shifting
            for page_num in sorted(pages_to_delete, reverse=True):
                doc.delete_page(page_num - 1)  # delete_page uses 0-indexed

            # Save to temp file (preserves original compression)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                doc.save(tmp_path, garbage=3, deflate=True)
                doc.close()

                with open(tmp_path, 'rb') as f:
                    pdf_bytes = f.read()

                logger.debug(
                    f"Prepared PDF from {pdf_path.name}: {len(pages_to_keep)} pages "
                    f"({len(pdf_bytes) / 1024 / 1024:.2f} MB)"
                )
                return pdf_bytes
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to prepare PDF: {e}")
            return None

    def _validate_toc_structure(self, chapters: List[Dict], path: str = "") -> List[str]:
        """Validate TOC structure for common issues.

        Returns list of issue descriptions.
        """
        issues = []

        for i, chapter in enumerate(chapters):
            chapter_path = f"{path}/{chapter.get('title', 'unknown')[:30]}" if path else chapter.get('title', 'unknown')[:30]

            # Check required fields
            if 'start_page' not in chapter:
                issues.append(f"Missing start_page: {chapter_path}")
            if 'end_page' not in chapter:
                issues.append(f"Missing end_page: {chapter_path}")

            # Check page range validity
            start = chapter.get('start_page')
            end = chapter.get('end_page')
            if start is not None and end is not None:
                if end < start:
                    issues.append(f"Invalid range (end < start): {chapter_path} (p{start}-p{end})")

                # Check for overlap with next sibling
                if i + 1 < len(chapters):
                    next_chapter = chapters[i + 1]
                    next_start = next_chapter.get('start_page')
                    if next_start is not None and end >= next_start:
                        issues.append(f"Overlap: {chapter_path} ends at p{end} but next chapter starts at p{next_start}")

            # Recursively check children
            children = chapter.get('children', [])
            if children:
                issues.extend(self._validate_toc_structure(children, chapter_path))

        return issues

    def analyze_pdf_structure(
        self,
        pdf_path: Path,
        book_title: str,
        state: Optional[RefinerState] = None,
        state_path: Optional[Path] = None
    ) -> Tuple[List[TOCNode], Dict]:
        """
        Analyze PDF structure and extract recursive TOC tree.

        Uses a two-phase approach to avoid being misled by printed page numbers:
        1. Detect and extract TOC structure (titles only, no page numbers)
        2. Match TOC titles with actual content to get correct PDF page numbers

        Supports resume from any step via state parameter.

        Args:
            pdf_path: Path to PDF file (should be preprocessed with page patches)
            book_title: Book title for prompts
            state: RefinerState for resume capability
            state_path: Path to save state after each step

        Returns:
            Tuple of (list of top-level TOCNodes, book metadata dict)
        """
        def save_state():
            if state and state_path:
                state.save(state_path)

        # Step 1: Detect TOC location
        if state and state.toc_location:
            logger.info("Step 1: Using cached TOC location...")
            toc_location = state.toc_location
        else:
            logger.info("Step 1: Detecting TOC location...")
            toc_location = self.detect_toc_location(pdf_path)
            if state and toc_location:
                state.toc_location = toc_location
                save_state()

        if toc_location and toc_location.get('has_toc'):
            toc_start = toc_location['toc_start']
            toc_end = toc_location['toc_end']

            # Step 2a: Extract TOC structure (no page numbers)
            if state and state.toc_structure and state.toc_structure.get('chapters'):
                logger.info("Step 2a: Using cached TOC structure...")
                toc_structure = state.toc_structure
            else:
                logger.info("Step 2a: Extracting TOC structure (without page numbers)...")
                toc_structure = self.extract_toc_structure(pdf_path, toc_start, toc_end)
                if state and toc_structure:
                    state.toc_structure = toc_structure
                    save_state()

            if toc_structure and toc_structure.get('chapters'):
                # Step 2b: Match TOC with content to get page numbers
                logger.info("Step 2b: Matching TOC with content for page numbers...")
                result = self.match_toc_with_content(
                    pdf_path, toc_structure, book_title, toc_start, toc_end
                )
            else:
                # TOC extraction failed, fall back to direct analysis
                logger.warning("TOC structure extraction failed, using direct analysis")
                result = self._analyze_pdf_directly(pdf_path, book_title)
        else:
            # No TOC detected, use direct analysis
            logger.info("No TOC detected, using direct analysis...")
            result = self._analyze_pdf_directly(pdf_path, book_title)

        # Mark structure analysis complete
        if state:
            state.structure_analysis_complete = True
            save_state()

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

    def _analyze_pdf_directly(self, pdf_path: Path, book_title: str) -> Dict:
        """
        Analyze PDF directly without separate TOC extraction.
        Used when no TOC is detected or TOC extraction fails.
        """
        prompt = f"""
Analyze this book PDF with title "{book_title}" and provide a detailed breakdown of its structure.

**CRITICAL**: Extract the COMPLETE hierarchical structure.
- Extract ALL levels: Part, Chapter, Section, Subsection, etc.
- DO NOT create artificial subdivisions beyond what actually exists
- Use PDF page numbers from "PDF Page: X" labels (not printed page numbers)

Include:
1. Author name(s)
2. Cover page (page number)
3. Table of contents (page numbers) if exists
4. All chapters with their COMPLETE substructure as a recursive tree
5. Back cover page (page number)

Additionally identify special chapter types:
- If a chapter consists ONLY of footnotes/endnotes for other chapters, add "type": "notes"
- If any chapter's notes are at the end of itself, then there should be NO notes chapter
- A book contains at most one notes chapter
- Abbreviations, Bibliography, Index, or Summary Table are NOT considered as notes
- Only literal "Notes" or "Endnotes" chapters with [1], [2], [3]... definitions are considered as notes

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
            "children": [...]  // Recursive
        }}
    ],
    "back_cover": {{"page_number": int}}
}}

**IMPORTANT**:
- Use PDF page numbers from "PDF Page: X" labels, NOT printed page numbers
- Preserve the original language for all titles and author names
"""

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        parts = [
            prompt,
            Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
        ]

        generation_config = self.structure_client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        response_text = self.structure_client.generate_content_stream(
            model=self.structure_model,
            contents=parts,
            config=generation_config,
            operation_name="PDF direct structure analysis"
        )

        return parse_llm_json(response_text, operation_name="PDF direct structure analysis")

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

        generation_config = self.analysis_client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.analysis_client.generate_content_stream(
                model=self.analysis_model,
                contents=prompt,
                config=generation_config,
                operation_name=f"Re-breakdown: {chapter_title}"
            )

            result = parse_llm_json(response_text, operation_name=f"Re-breakdown: {chapter_title}")
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

