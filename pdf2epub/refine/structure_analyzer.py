"""
Structure analysis for PDF books.

Provides:
- Initial PDF structure analysis (extract TOC tree)
- Re-breakdown when verification fails
- Discovery of hidden subsections
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz
from google.genai.types import Part
from loguru import logger

from ..utils.common import parse_llm_json
from ..utils.llm_client import BoundLLMClient
from .toc_tree import TOCNode, dict_list_to_toc_tree
from .refiner_state import RefinerState
from .pdf_batching import (
    PdfBatchContext,
    get_toc_detection_pages,
    convert_toc_page_to_original,
    create_content_batches,
    deduplicate_chapters,
    merge_batch_chapters,
)
from .pdf_rasterizer import rasterize_to_limit


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
        analysis_model: str,
        config: Dict = None
    ):
        """
        Initialize the structure analyzer.

        Args:
            structure_client: BoundLLMClient for PDF operations (needs PDF support)
            structure_model: Model for full PDF analysis (needs large context)
            toc_model: Model for TOC detection/extraction (cheaper, still needs PDF)
            analysis_client: BoundLLMClient for re-breakdown (text only, no PDF needed)
            analysis_model: Model for re-breakdown and subsection discovery
            config: Configuration dict for compression settings
        """
        self.structure_client = structure_client
        self.structure_model = structure_model
        self.toc_model = toc_model
        self.analysis_client = analysis_client
        self.analysis_model = analysis_model
        self.config = config or {}

    def _compress_pdf_to_limit(self, input_path: Path, output_path: Path, target_mb: float) -> bool:
        """
        Iteratively compress PDF until it's below target size.

        Tries progressively more aggressive compression settings until the PDF
        is below the target size in MB.

        Args:
            input_path: Path to input PDF
            output_path: Path to save compressed PDF
            target_mb: Target size in MB

        Returns:
            True if compression succeeded, False otherwise
        """
        from ..pdf_compressor import compress_pdf

        # Progressive compression strategies (from moderate to extreme)
        strategies = [
            {"dpi": 150, "quality": 60, "grayscale": False, "desc": "moderate (150dpi, q60)"},
            {"dpi": 120, "quality": 50, "grayscale": False, "desc": "aggressive (120dpi, q50)"},
            {"dpi": 100, "quality": 40, "grayscale": False, "desc": "very aggressive (100dpi, q40)"},
            {"dpi": 80, "quality": 30, "grayscale": False, "desc": "extreme (80dpi, q30)"},
            {"dpi": 72, "quality": 20, "grayscale": True, "desc": "maximum (72dpi, q20, grayscale)"},
        ]

        input_size_mb = os.path.getsize(input_path) / 1024 / 1024
        logger.info(f"Input PDF size: {input_size_mb:.2f} MB, target: {target_mb:.2f} MB")

        for i, strategy in enumerate(strategies, 1):
            logger.info(f"Compression attempt {i}/{len(strategies)}: {strategy['desc']}")

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                success, stats = compress_pdf(
                    str(input_path),
                    str(tmp_path),
                    dpi=strategy['dpi'],
                    quality=strategy['quality'],
                    grayscale=strategy['grayscale']
                )

                if not success:
                    logger.warning(f"Compression failed with {strategy['desc']}")
                    tmp_path.unlink(missing_ok=True)
                    continue

                output_size_mb = stats['output_size_mb']
                logger.info(f"Compressed to {output_size_mb:.2f} MB")

                if output_size_mb <= target_mb:
                    # Success! Move to final output path
                    tmp_path.rename(output_path)
                    logger.success(f"Successfully compressed to {output_size_mb:.2f} MB (under {target_mb:.2f} MB limit)")
                    return True
                else:
                    logger.warning(f"Still too large ({output_size_mb:.2f} MB > {target_mb:.2f} MB), trying more aggressive compression...")
                    tmp_path.unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"Error during compression: {e}")
                tmp_path.unlink(missing_ok=True)
                continue

        logger.error(f"Failed to compress PDF below {target_mb:.2f} MB after {len(strategies)} attempts")
        return False

    def _is_503_error(self, error: Exception) -> bool:
        """Check if an exception is a 503 UNAVAILABLE error."""
        error_str = str(error).lower()
        return '503' in error_str or 'unavailable' in error_str

    def _prepare_pdf_rasterized(
        self,
        pdf_path: Path,
        include_pages: Optional[List[int]] = None,
        target_mb: float = 30.0
    ) -> Optional[bytes]:
        """
        Prepare PDF using JBIG2 rasterization (for 503 fallback).

        Args:
            pdf_path: Path to the source PDF
            include_pages: Pages to include (1-indexed), None for all
            target_mb: Target file size limit in MB

        Returns:
            PDF bytes, or None if rasterization fails
        """
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            success, stats = rasterize_to_limit(
                pdf_path, tmp_path, include_pages, target_mb
            )
            if not success:
                return None

            logger.info(
                f"Rasterized PDF: {stats['page_count']} pages, "
                f"{stats['output_size_mb']:.1f} MB ({stats['method']} @ {stats['dpi']} DPI)"
            )

            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            tmp_path.unlink(missing_ok=True)

    def detect_toc_location(
        self, pdf_path: Path, batch_ctx: Optional[PdfBatchContext] = None
    ) -> Optional[Dict]:
        """
        Detect the location of table of contents in the PDF.

        Args:
            pdf_path: Path to PDF file
            batch_ctx: Optional batch context for large PDFs

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

        # For large PDFs, only send first/last pages for TOC detection
        if batch_ctx and batch_ctx.needs_batching:
            pages = get_toc_detection_pages(batch_ctx)
            pdf_data = self._prepare_pdf(pdf_path, include_pages=pages)
            if pdf_data is None:
                logger.warning("Failed to prepare PDF subset for TOC detection, using full PDF")
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
            else:
                logger.info(f"TOC detection: using {len(pages)} pages (first/last {batch_ctx.toc_sample_pages})")
        else:
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

            # Validate result is a dict
            if not isinstance(result, dict):
                logger.warning(f"TOC detection returned unexpected type: {type(result)}, expected dict")
                return None

            if result.get('has_toc'):
                # Convert page numbers from subset PDF to original PDF
                if batch_ctx and batch_ctx.needs_batching:
                    orig_start = convert_toc_page_to_original(result['toc_start'], batch_ctx)
                    orig_end = convert_toc_page_to_original(result['toc_end'], batch_ctx)
                    logger.info(f"TOC detected in subset: pages {result['toc_start']}-{result['toc_end']} -> original: {orig_start}-{orig_end}")
                    result['toc_start'] = orig_start
                    result['toc_end'] = orig_end
                else:
                    logger.info(f"TOC detected: pages {result['toc_start']}-{result['toc_end']}")
            else:
                logger.info("No TOC detected in PDF")

            return result
        except Exception as e:
            if self._is_503_error(e):
                logger.warning("503 error during TOC detection, retrying with JBIG2 rasterization...")
                return self._detect_toc_location_rasterized(
                    pdf_path, batch_ctx, prompt, generation_config
                )
            logger.error(f"TOC detection failed: {e}")
            return None

    def _detect_toc_location_rasterized(
        self,
        pdf_path: Path,
        batch_ctx: Optional[PdfBatchContext],
        prompt: str,
        generation_config
    ) -> Optional[Dict]:
        """Retry TOC detection with rasterized PDF."""
        if batch_ctx and batch_ctx.needs_batching:
            pages = get_toc_detection_pages(batch_ctx)
        else:
            pages = None

        pdf_data = self._prepare_pdf_rasterized(pdf_path, include_pages=pages)
        if pdf_data is None:
            logger.error("Rasterization failed for TOC detection")
            return None

        parts = [
            prompt,
            Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
        ]

        try:
            response_text = self.structure_client.generate_content_stream(
                model=self.toc_model,
                contents=parts,
                config=generation_config,
                operation_name="TOC location detection (rasterized)"
            )
            result = parse_llm_json(response_text, operation_name="TOC location detection (rasterized)")

            if not isinstance(result, dict):
                logger.warning(f"TOC detection returned unexpected type: {type(result)}, expected dict")
                return None

            if result.get('has_toc'):
                # Convert page numbers from rasterized PDF to original PDF
                if batch_ctx and batch_ctx.needs_batching:
                    orig_start = convert_toc_page_to_original(result['toc_start'], batch_ctx)
                    orig_end = convert_toc_page_to_original(result['toc_end'], batch_ctx)
                    logger.info(f"TOC detected (rasterized): pages {result['toc_start']}-{result['toc_end']} -> original: {orig_start}-{orig_end}")
                    result['toc_start'] = orig_start
                    result['toc_end'] = orig_end
                else:
                    logger.info(f"TOC detected (rasterized): pages {result['toc_start']}-{result['toc_end']}")
            else:
                logger.info("No TOC detected in rasterized PDF")

            return result
        except Exception as e:
            logger.error(f"TOC detection failed even with rasterization: {e}")
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
            if self._is_503_error(e):
                logger.warning("503 error during TOC structure extraction, retrying with rasterization...")
                pdf_data = self._prepare_pdf_rasterized(pdf_to_use, include_pages=toc_pages)
                if pdf_data is None:
                    logger.error("Rasterization failed for TOC structure extraction")
                    return None
                parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
                response_text = self.structure_client.generate_content_stream(
                    model=self.toc_model,
                    contents=parts,
                    config=generation_config,
                    operation_name="TOC structure extraction (rasterized)"
                )
                result = parse_llm_json(response_text, operation_name="TOC structure extraction (rasterized)")
                chapter_count = len(result.get('chapters', []))
                logger.info(f"Extracted {chapter_count} top-level items from TOC (rasterized)")
                return result
            logger.error(f"TOC structure extraction failed: {e}")
            return None

    def match_toc_with_content(
        self,
        pdf_path: Path,
        toc_structure: Dict,
        book_title: str,
        toc_start: int,
        toc_end: int,
        batch_ctx: Optional[PdfBatchContext] = None
    ) -> Dict:
        """
        Match TOC structure with actual PDF content to determine page numbers.

        Args:
            pdf_path: Path to PDF file
            toc_structure: TOC structure without page numbers
            book_title: Book title
            toc_start: TOC start page (to exclude from search)
            toc_end: TOC end page (to exclude from search)
            batch_ctx: Optional batch context for large PDFs

        Returns:
            Complete structure with page numbers
        """
        # For large PDFs, use batch processing
        if batch_ctx and batch_ctx.needs_batching:
            return self._match_toc_batched(
                pdf_path, toc_structure, book_title, toc_start, toc_end, batch_ctx
            )

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

        try:
            result = self._do_toc_matching(parts, generation_config)
        except Exception as e:
            if self._is_503_error(e):
                logger.warning("503 error during TOC matching, retrying with rasterization...")
                # Prepare rasterized PDF (exclude TOC pages)
                all_pages = list(range(1, len(list(fitz.open(pdf_path))) + 1))
                content_pages = [p for p in all_pages if p not in toc_pages]
                pdf_data = self._prepare_pdf_rasterized(pdf_path, include_pages=content_pages)
                if pdf_data is None:
                    raise RuntimeError("Rasterization failed for TOC matching") from e
                parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
                result = self._do_toc_matching(parts, generation_config, rasterized=True)
            else:
                raise

        return result

    def _do_toc_matching(self, parts: List, generation_config, rasterized: bool = False) -> Dict:
        """Execute TOC matching with validation retry logic."""
        suffix = " (rasterized)" if rasterized else ""

        response_text = self.structure_client.generate_content_stream(
            model=self.structure_model,
            contents=parts,
            config=generation_config,
            operation_name=f"TOC page matching{suffix}"
        )

        result = parse_llm_json(response_text, operation_name=f"TOC page matching{suffix}")

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
                operation_name=f"TOC page matching (retry){suffix}"
            )

            result = parse_llm_json(response_text, operation_name=f"TOC page matching (retry){suffix}")

            # Check if issues are fixed
            remaining_issues = self._validate_toc_structure(result.get('chapters', []))
            if remaining_issues:
                logger.warning(f"After retry, {len(remaining_issues)} issues remain")

        return result

    def _match_toc_batched(
        self,
        pdf_path: Path,
        toc_structure: Dict,
        book_title: str,
        toc_start: int,
        toc_end: int,
        batch_ctx: PdfBatchContext
    ) -> Dict:
        """Match TOC with content in batches for large PDFs.

        Strategy:
        1. Send each batch to LLM to find chapter start pages
        2. Collect all found chapters from all batches
        3. Send collected results + original TOC structure to LLM for final merge
        """
        import json

        toc_pages = set(range(toc_start, toc_end + 1))
        batches = create_content_batches(batch_ctx, exclude_pages=toc_pages)

        logger.info(f"Processing {len(batches)} batches for TOC matching...")

        all_found_chapters = []
        toc_json = json.dumps(toc_structure.get('chapters', []), ensure_ascii=False, indent=2)

        # Phase 1: Collect chapter locations from each batch
        for i, batch_pages in enumerate(batches):
            batch_start, batch_end = min(batch_pages), max(batch_pages)
            logger.info(f"Batch {i+1}/{len(batches)}: pages {batch_start}-{batch_end}")

            pdf_data = self._prepare_pdf(pdf_path, include_pages=batch_pages)
            if pdf_data is None:
                logger.warning(f"Failed to prepare batch {i+1}, skipping")
                continue

            prompt = f"""
Match this Table of Contents structure with the actual content in the PDF.

**Book Title**: {book_title}

**BATCH INFO**: This is batch {i+1}/{len(batches)}, pages {batch_start}-{batch_end}.
Only report chapters/sections that START within this page range.

**TOC Structure** (no page numbers):
{toc_json}

**Task**: Find where each chapter/section starts (PDF page number from "PDF Page: X" labels).
Look for ALL levels of headings (Parts, Chapters, Sections, etc.) - not just top-level ones.

Return JSON:
{{
    "chapters_found": [
        {{"title": string, "start_page": int}}
    ]
}}

Only include chapters whose title heading appears in pages {batch_start}-{batch_end}.
"""

            parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
            generation_config = self.structure_client.get_default_config(temperature=0.1)
            generation_config.response_mime_type = "application/json"

            try:
                response_text = self.structure_client.generate_content_stream(
                    model=self.structure_model,
                    contents=parts,
                    config=generation_config,
                    operation_name=f"TOC matching batch {i+1}/{len(batches)}"
                )
            except Exception as e:
                if self._is_503_error(e):
                    logger.warning(f"503 error in batch {i+1}, retrying with rasterization...")
                    pdf_data = self._prepare_pdf_rasterized(pdf_path, include_pages=batch_pages)
                    if pdf_data is None:
                        logger.warning(f"Rasterization failed for batch {i+1}, skipping")
                        continue
                    parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
                    response_text = self.structure_client.generate_content_stream(
                        model=self.structure_model,
                        contents=parts,
                        config=generation_config,
                        operation_name=f"TOC matching batch {i+1}/{len(batches)} (rasterized)"
                    )
                else:
                    raise

            batch_result = parse_llm_json(response_text, operation_name="TOC matching")
            all_found_chapters.extend(batch_result.get('chapters_found', []))

        logger.info(f"Found {len(all_found_chapters)} chapter locations across all batches")

        # Phase 2: Let LLM merge the results with original TOC structure
        found_json = json.dumps(all_found_chapters, ensure_ascii=False, indent=2)

        merge_prompt = f"""
Merge these chapter location results with the original TOC structure.

**Book Title**: {book_title}
**Total Pages**: {batch_ctx.total_pages}

**Original TOC Structure** (hierarchical, no page numbers):
{toc_json}

**Found Chapter Locations** (flat list from scanning the PDF):
{found_json}

**Task**:
1. Fill in start_page for each chapter/section by matching titles
2. Calculate end_page based on where the next section starts
3. For parent sections (like "VOLUME 1" or "PART I") that don't appear as headings in the PDF,
   infer their start_page from their first child's start_page
4. Preserve the complete hierarchical structure

Return the complete structure as JSON:
{{
    "author": string or null,
    "language": string,  // e.g., "english"
    "is_vertical_text": boolean,
    "has_footnotes": boolean,
    "cover_page": {{"page_number": int}} or null,
    "table_of_contents": {{
        "start_page": {toc_start},
        "end_page": {toc_end}
    }},
    "chapters": [
        {{
            "title": string,
            "start_page": int,
            "end_page": int,
            "level": int,
            "children": [...]  // Recursive, same structure
        }}
    ],
    "back_cover": {{"page_number": {batch_ctx.total_pages}}} or null
}}

**CRITICAL**:
- Every chapter/section MUST have start_page and end_page
- Preserve the original hierarchy exactly
- Use PDF page numbers, not printed page numbers
"""

        generation_config = self.structure_client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        response_text = self.structure_client.generate_content_stream(
            model=self.structure_model,
            contents=merge_prompt,
            config=generation_config,
            operation_name="TOC batch merge"
        )

        result = parse_llm_json(response_text, operation_name="TOC batch merge")

        # Validate and retry if needed
        issues = self._validate_toc_structure(result.get('chapters', []))
        if issues:
            logger.warning(f"Found {len(issues)} issues in merged TOC, requesting fix...")
            for issue in issues[:5]:
                logger.warning(f"  {issue}")

            error_msg = "Issues found:\n" + "\n".join(f"- {issue}" for issue in issues)
            error_msg += "\n\nPlease fix and return corrected JSON."

            response_text = self.structure_client.generate_content_stream(
                model=self.structure_model,
                contents=[merge_prompt, response_text, error_msg],
                config=generation_config,
                operation_name="TOC batch merge (retry)"
            )
            result = parse_llm_json(response_text, operation_name="TOC batch merge (retry)")

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

                file_size_mb = os.path.getsize(tmp_path) / 1024 / 1024

                # Check if compression is needed
                compression_config = self.config.get('refine', {}).get('pdf_compression', {})
                payload_limit_mb = compression_config.get('payload_limit_mb', 30)
                should_compress = compression_config.get('compress_if_exceeds', True)

                if should_compress and file_size_mb > payload_limit_mb:
                    logger.warning(
                        f"PDF size ({file_size_mb:.2f} MB) exceeds payload limit ({payload_limit_mb} MB). "
                        f"Applying JPEG compression..."
                    )

                    # Import compress_pdf
                    from ..pdf_compressor import compress_pdf

                    # Create compressed output path
                    with tempfile.NamedTemporaryFile(suffix='_compressed.pdf', delete=False) as compressed_tmp:
                        compressed_path = compressed_tmp.name

                    # Compress with config settings
                    dpi = compression_config.get('dpi', 150)
                    quality = compression_config.get('quality', 60)
                    grayscale = compression_config.get('grayscale', False)

                    success, stats = compress_pdf(
                        tmp_path,
                        compressed_path,
                        dpi=dpi,
                        quality=quality,
                        grayscale=grayscale
                    )

                    if success:
                        compressed_size_mb = os.path.getsize(compressed_path) / 1024 / 1024
                        logger.info(
                            f"Compressed PDF: {file_size_mb:.2f} MB → {compressed_size_mb:.2f} MB "
                            f"({stats['compression_ratio']:.1f}x compression)"
                        )

                        # Replace with compressed version
                        os.unlink(tmp_path)
                        tmp_path = compressed_path
                        file_size_mb = compressed_size_mb
                    else:
                        logger.warning("Compression failed, using uncompressed version")
                        if os.path.exists(compressed_path):
                            os.unlink(compressed_path)

                with open(tmp_path, 'rb') as f:
                    pdf_bytes = f.read()

                logger.debug(
                    f"Prepared PDF from {pdf_path.name}: {len(pages_to_keep)} pages "
                    f"({file_size_mb:.2f} MB)"
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

        # Step 0: Compress PDF if needed to fit within payload limit
        compression_config = self.config.get('refine', {}).get('pdf_compression', {})
        payload_limit_mb = compression_config.get('payload_limit_mb', 30)
        should_compress = compression_config.get('compress_if_exceeds', True)

        pdf_size_mb = os.path.getsize(pdf_path) / 1024 / 1024
        working_pdf_path = pdf_path  # By default, use original PDF

        if should_compress and pdf_size_mb > payload_limit_mb:
            logger.warning(f"PDF size ({pdf_size_mb:.2f} MB) exceeds payload limit ({payload_limit_mb} MB)")
            logger.info("Compressing PDF to fit within API payload limit...")

            # Create compressed PDF in same directory as input
            compressed_pdf_path = pdf_path.parent / f"{pdf_path.stem}_compressed.pdf"

            if self._compress_pdf_to_limit(pdf_path, compressed_pdf_path, payload_limit_mb):
                logger.success(f"Using compressed PDF: {compressed_pdf_path}")
                working_pdf_path = compressed_pdf_path
            else:
                logger.error("PDF compression failed, attempting to use original PDF (may fail with 413 error)")
                working_pdf_path = pdf_path
        else:
            logger.info(f"PDF size ({pdf_size_mb:.2f} MB) is within payload limit ({payload_limit_mb} MB), no compression needed")

        # Create batch context for large PDF handling
        batch_ctx = PdfBatchContext.from_pdf(working_pdf_path)
        if batch_ctx.needs_batching:
            logger.info(f"Large PDF detected ({batch_ctx.total_pages} pages), will use batch processing")

        # Step 1: Detect TOC location
        if state and state.toc_location:
            logger.info("Step 1: Using cached TOC location...")
            toc_location = state.toc_location
        else:
            logger.info("Step 1: Detecting TOC location...")
            toc_location = self.detect_toc_location(working_pdf_path, batch_ctx)
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
                toc_structure = self.extract_toc_structure(working_pdf_path, toc_start, toc_end)
                if state and toc_structure:
                    state.toc_structure = toc_structure
                    save_state()

            if toc_structure and toc_structure.get('chapters'):
                # Step 2b: Match TOC with content to get page numbers
                logger.info("Step 2b: Matching TOC with content for page numbers...")
                result = self.match_toc_with_content(
                    working_pdf_path, toc_structure, book_title, toc_start, toc_end, batch_ctx
                )
            else:
                # TOC extraction failed, fall back to direct analysis
                logger.warning("TOC structure extraction failed, using direct analysis")
                result = self._analyze_pdf_directly(working_pdf_path, book_title, batch_ctx)
        else:
            # No TOC detected, use direct analysis
            logger.info("No TOC detected, using direct analysis...")
            result = self._analyze_pdf_directly(working_pdf_path, book_title, batch_ctx)

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

    def _analyze_pdf_directly(
        self, pdf_path: Path, book_title: str,
        batch_ctx: Optional[PdfBatchContext] = None
    ) -> Dict:
        """
        Analyze PDF directly without separate TOC extraction.
        Used when no TOC is detected or TOC extraction fails.
        """
        # For large PDFs, use batch processing
        if batch_ctx and batch_ctx.needs_batching:
            return self._analyze_pdf_batched(pdf_path, book_title, batch_ctx)
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

        try:
            response_text = self.structure_client.generate_content_stream(
                model=self.structure_model,
                contents=parts,
                config=generation_config,
                operation_name="PDF direct structure analysis"
            )
        except Exception as e:
            if self._is_503_error(e):
                logger.warning("503 error during direct analysis, retrying with rasterization...")
                pdf_data = self._prepare_pdf_rasterized(pdf_path)
                if pdf_data is None:
                    raise RuntimeError("Rasterization failed for direct analysis") from e
                parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
                response_text = self.structure_client.generate_content_stream(
                    model=self.structure_model,
                    contents=parts,
                    config=generation_config,
                    operation_name="PDF direct structure analysis (rasterized)"
                )
            else:
                raise

        return parse_llm_json(response_text, operation_name="PDF direct structure analysis")

    def _analyze_pdf_batched(
        self, pdf_path: Path, book_title: str, batch_ctx: PdfBatchContext
    ) -> Dict:
        """Analyze large PDF in batches."""
        batches = create_content_batches(batch_ctx)
        logger.info(f"Analyzing {batch_ctx.total_pages} pages in {len(batches)} batches...")

        all_chapters = []
        metadata = {}

        for i, batch_pages in enumerate(batches):
            batch_start, batch_end = min(batch_pages), max(batch_pages)
            is_first, is_last = (i == 0), (i == len(batches) - 1)

            logger.info(f"Batch {i+1}/{len(batches)}: pages {batch_start}-{batch_end}")

            pdf_data = self._prepare_pdf(pdf_path, include_pages=batch_pages)
            if pdf_data is None:
                logger.warning(f"Failed to prepare batch {i+1}, skipping")
                continue

            # Build batch-specific prompt
            prompt = self._build_direct_analysis_prompt(
                book_title, batch_start, batch_end,
                i + 1, len(batches), is_first, is_last
            )

            parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
            generation_config = self.structure_client.get_default_config(temperature=0.1)
            generation_config.response_mime_type = "application/json"

            try:
                response_text = self.structure_client.generate_content_stream(
                    model=self.structure_model,
                    contents=parts,
                    config=generation_config,
                    operation_name=f"Direct analysis batch {i+1}/{len(batches)}"
                )
            except Exception as e:
                if self._is_503_error(e):
                    logger.warning(f"503 error in batch {i+1}, retrying with rasterization...")
                    pdf_data = self._prepare_pdf_rasterized(pdf_path, include_pages=batch_pages)
                    if pdf_data is None:
                        logger.warning(f"Rasterization failed for batch {i+1}, skipping")
                        continue
                    parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
                    response_text = self.structure_client.generate_content_stream(
                        model=self.structure_model,
                        contents=parts,
                        config=generation_config,
                        operation_name=f"Direct analysis batch {i+1}/{len(batches)} (rasterized)"
                    )
                else:
                    raise

            result = parse_llm_json(response_text, operation_name="Direct analysis")

            # Extract metadata from first batch only
            if is_first:
                metadata = {
                    'author': result.get('author'),
                    'language': result.get('language'),
                    'is_vertical_text': result.get('is_vertical_text'),
                    'has_footnotes': result.get('has_footnotes'),
                    'cover_page': result.get('cover_page'),
                    'table_of_contents': result.get('table_of_contents'),
                }

            if is_last:
                metadata['back_cover'] = result.get('back_cover')

            all_chapters.extend(result.get('chapters', []))

        return {
            **metadata,
            'chapters': deduplicate_chapters(all_chapters)
        }

    def _build_direct_analysis_prompt(
        self, book_title: str, batch_start: int, batch_end: int,
        batch_num: int, total_batches: int, is_first: bool, is_last: bool
    ) -> str:
        """Build prompt for direct PDF analysis (batch-aware)."""

        metadata_section = ""
        if is_first:
            metadata_section = """
Include in response:
- "author": string
- "language": string (e.g., "english", "japanese")
- "is_vertical_text": boolean
- "has_footnotes": boolean
- "cover_page": {"page_number": int}
- "table_of_contents": {"start_page": int, "end_page": int} if exists
"""
        if is_last:
            metadata_section += '\n- "back_cover": {"page_number": int}'

        return f"""
Analyze this book PDF section and extract chapter structure.

**Book Title**: {book_title}
**BATCH INFO**: Batch {batch_num}/{total_batches}, pages {batch_start}-{batch_end}

**Task**: Find all chapter/section headings in this page range.
{metadata_section}

Return JSON:
{{
    "chapters": [
        {{
            "title": string,
            "start_page": int,  // PDF page number
            "end_page": int,    // Use {batch_end} if continues beyond
            "level": int,
            "children": [...]
        }}
    ]
}}

**CRITICAL**: Use PDF page numbers from "PDF Page: X" labels.
"""

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

