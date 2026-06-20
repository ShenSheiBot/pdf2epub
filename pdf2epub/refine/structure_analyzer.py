"""
Structure analysis for PDF books.

Provides:
- Initial PDF structure analysis (extract TOC tree)
- Re-breakdown when verification fails
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz
from loguru import logger

from ..utils.common import parse_llm_json
from ..utils.llm_client import BoundLLMClient
from .toc_tree import TOCNode, dict_list_to_toc_tree
from .refiner_state import RefinerState
from .pdf_batching import (
    PdfBatchContext,
    get_toc_detection_pages,
)
from .pdf_rasterizer import rasterize_to_limit
from .adaptive_pdf_call import (
    PdfPageLimitLearner,
    TocDetectionCall,
    DirectAnalysisCall,
    validate_chapter_structure,
)


def _update_levels_recursive(chapters: List[Dict], target_level: int) -> None:
    """Update level field for chapters and all descendants."""
    for ch in chapters:
        ch['level'] = target_level
        children = ch.get('children', [])
        if children:
            _update_levels_recursive(children, target_level + 1)


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
        self._corrupted_xref_pdfs: set = set()  # PDFs with known corrupted xref
        self._rasterized_pdf_path: Optional[Path] = None  # Cached rasterized PDF path
        self._prefer_rasterized: bool = False  # Set True after first successful rasterization

        # Initialize adaptive page limit learner
        adaptive_config = self.config.get('refine', {}).get('adaptive_page_limit', {})
        self._learner = PdfPageLimitLearner(
            initial_limit=adaptive_config.get('initial_pages', 900),
            min_limit=adaptive_config.get('min_pages', 100),
        )

    def _compress_pdf_to_limit(self, input_path: Path, output_path: Path, target_mb: float) -> bool:
        """
        Compress PDF until it's below target size.

        Priority: JBIG2 (primary) -> binarized PNG (fallback)

        Args:
            input_path: Path to input PDF
            output_path: Path to save compressed PDF
            target_mb: Target size in MB

        Returns:
            True if compression succeeded, False otherwise
        """
        input_size_mb = os.path.getsize(input_path) / 1024 / 1024
        logger.info(f"Input PDF size: {input_size_mb:.2f} MB, target: {target_mb:.2f} MB")

        # Step 1: Try JBIG2 rasterization (preserves text sharpness)
        logger.info("Attempting JBIG2 compression...")
        success, stats = rasterize_to_limit(input_path, output_path, pages=None, target_mb=target_mb)

        if success:
            logger.success(
                f"Successfully compressed to {stats['output_size_mb']:.2f} MB "
                f"using {stats['method']} @ {stats['dpi']} DPI"
            )
            return True

        # Step 2: JBIG2 unavailable/failed, fall back to binarized PNG at 120 DPI
        logger.warning("JBIG2 compression failed, falling back to binarized PNG at 120 DPI...")
        from ..pdf_compressor import compress_pdf

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            success, stats = compress_pdf(
                str(input_path),
                str(tmp_path),
                dpi=120,
            )

            if not success:
                logger.error("Binarized PNG compression failed")
                tmp_path.unlink(missing_ok=True)
                return False

            output_size_mb = stats['output_size_mb']
            if output_size_mb <= target_mb:
                tmp_path.rename(output_path)
                logger.success(f"Successfully compressed to {output_size_mb:.2f} MB (binarized PNG)")
                return True
            else:
                logger.error(f"Binarized PNG still too large ({output_size_mb:.2f} MB > {target_mb:.2f} MB)")
                tmp_path.unlink(missing_ok=True)
                return False

        except Exception as e:
            logger.error(f"Error during PNG compression: {e}")
            tmp_path.unlink(missing_ok=True)
            return False

    def detect_toc_location(
        self, pdf_path: Path, batch_ctx: Optional[PdfBatchContext] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> Optional[Dict]:
        """
        Detect the location of table of contents in the PDF.

        Uses TocDetectionCall with adaptive batching: if the API returns 503,
        automatically splits the page set and retries with fewer pages.
        """
        # Determine pages for TOC detection
        if batch_ctx and batch_ctx.total_pages > batch_ctx.page_limit:
            pages = get_toc_detection_pages(batch_ctx)
            logger.info(f"TOC detection: using {len(pages)} pages (first/last {batch_ctx.toc_sample_pages})")
        else:
            total = batch_ctx.total_pages if batch_ctx else len(fitz.open(pdf_path))
            pages = list(range(1, total + 1))

        call = TocDetectionCall(
            self.structure_client, self.toc_model,
            self._prepare_pdf, self._learner,
            self._prepare_pdf_rasterized,
        )
        toc_artifacts = artifacts_dir / "toc_detection" if artifacts_dir else None
        try:
            return call.run(pdf_path, pages, artifacts_dir=toc_artifacts)
        except Exception as e:
            logger.error(f"TOC detection failed: {e}")
            return None

    def _clean_toc_text(
        self, toc_start: int, toc_end: int, pages_dir: Path
    ) -> Optional[str]:
        """
        Extract TOC text from OCR pages and strip page numbers via LLM.

        Returns cleaned TOC text (structure + titles, no page numbers),
        or None if extraction fails.
        """
        # Read OCR markdown for TOC pages
        toc_parts = []
        for page_num in range(toc_start, toc_end + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if page_file.exists():
                toc_parts.append(page_file.read_text(encoding='utf-8'))

        if not toc_parts:
            logger.warning("No TOC page files found, skipping TOC reference")
            return None

        raw_toc = "\n\n".join(toc_parts)
        logger.info(f"Extracting TOC reference from pages {toc_start}-{toc_end}...")

        prompt = f"""Below is the raw text of a book's Table of Contents page(s).

Remove ALL page numbers (digits at the end of lines, or standalone numbers used as page references).
Keep the hierarchical structure, indentation, and all titles/headings exactly as they appear.
Return ONLY the cleaned text, nothing else.

---
{raw_toc}
---"""

        try:
            config = self.analysis_client.get_default_config(temperature=0.1)
            # TOC text typically ~500-1000 tokens. 4096 is conservative upper bound
            # that reduces API costs. If truncated, returns None (handled by caller).
            config.max_tokens = 4096
            response = self.analysis_client.generate_content_stream(
                model=self.analysis_model,
                contents=prompt,
                config=config,
                operation_name="TOC page number stripping",
            )
            cleaned = response.strip()
            if cleaned:
                logger.info(f"TOC reference extracted ({len(cleaned)} chars)")
                return cleaned
            else:
                logger.warning("TOC stripping returned empty result")
                return None
        except Exception as e:
            logger.warning(f"TOC page number stripping failed: {e}")
            return None

    def _find_toc_reference_pages(
        self, toc_location: Optional[dict], pages_dir: Path, total_pages: int
    ) -> Optional[tuple]:
        """
        Find the most detailed TOC pages for use as reference context.

        Searches OCR pages for actual TOC headings (e.g. "Table des matières"),
        preferring the most detailed (usually back-of-book) table of contents
        over brief front-matter listings.

        Falls back to TocDetectionCall result if no heading match is found.
        """
        import re

        TOC_HEADING_PATTERNS = [
            r'TABLE\s+DES\s+MATI[ÈE]RES',
            r'Table\s+des\s+mati[èe]res',
            r'TABLE\s+OF\s+CONTENTS',
            r'Table\s+of\s+Contents',
            r'(?<!\w)CONTENTS(?!\w)',
            r'SOMMAIRE',
            r'Sommaire',
            r'目\s*次',
            r'INHALTSVERZEICHNIS',
            r'Inhaltsverzeichnis',
        ]
        combined = re.compile(
            '|'.join(f'(?:{p})' for p in TOC_HEADING_PATTERNS)
        )

        # Search front (first 30 pages) and back (last 30 pages) of book
        search_pages = set()
        search_pages.update(range(1, min(31, total_pages + 1)))
        search_pages.update(range(max(1, total_pages - 30), total_pages + 1))

        candidates = []  # [(page_num, numbered_line_count)]

        for page_num in sorted(search_pages):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if not page_file.exists():
                continue

            content = page_file.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Check first 5 lines for TOC heading
            first_lines = '\n'.join(lines[:5])
            if combined.search(first_lines):
                # Count lines ending with page numbers (TOC entries)
                num_count = sum(
                    1 for line in lines
                    if re.search(r'\d{2,4}\s*$', line.strip())
                )
                candidates.append((page_num, num_count))

        if not candidates:
            # Fall back to TocDetectionCall result
            if toc_location and toc_location.get('has_toc'):
                toc_start = toc_location['toc_start']
                toc_end = toc_location['toc_end']
                logger.info(
                    f"No TOC heading found in OCR pages, "
                    f"falling back to detected pages {toc_start}-{toc_end}"
                )
                return (toc_start, toc_end)
            return None

        # Pick candidate with most numbered lines (most detailed TOC)
        best_page, best_count = max(candidates, key=lambda x: x[1])

        # Extend range: include consecutive pages with TOC-like content
        toc_start = best_page
        toc_end = best_page

        for next_page in range(best_page + 1, min(best_page + 10, total_pages + 1)):
            page_file = pages_dir / f"page_{next_page:03d}.md"
            if not page_file.exists():
                break
            content = page_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            first_lines = '\n'.join(lines[:5])
            has_heading = bool(combined.search(first_lines))
            num_count = sum(
                1 for line in lines
                if re.search(r'\d{2,4}\s*$', line.strip())
            )
            if has_heading or num_count >= 3:
                toc_end = next_page
            else:
                break

        logger.info(
            f"Found detailed TOC at pages {toc_start}-{toc_end} "
            f"({best_count} entries on best page)"
        )
        return (toc_start, toc_end)

    def _check_xref_corrupted(self, pdf_path: Path) -> bool:
        """
        Probe whether a PDF has corrupted xref by selecting a small subset
        and checking if the output shrinks. Result is cached per PDF path.
        """
        import tempfile
        import os

        pdf_key = str(pdf_path.resolve())
        if pdf_key in self._corrupted_xref_pdfs:
            return True

        try:
            doc = fitz.open(pdf_path)
            total = len(doc)
            if total <= 2:
                doc.close()
                return False

            # Select ~10% of pages (min 2, max 20) as probe
            probe_count = max(2, min(20, total // 10))
            doc.select(list(range(probe_count)))

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = tmp.name
            doc.save(tmp_path, garbage=4, deflate=True)
            doc.close()

            probe_size = os.path.getsize(tmp_path) / 1024 / 1024
            original_size = os.path.getsize(pdf_path) / 1024 / 1024
            os.unlink(tmp_path)

            size_fraction = probe_size / original_size if original_size > 0 else 0
            page_fraction = probe_count / total

            # Probe selected a small fraction but file barely shrank → corrupted
            if page_fraction < 0.5 and size_fraction > 0.8:
                logger.warning(
                    f"Corrupted xref detected: {probe_count}/{total} pages probe "
                    f"= {probe_size:.1f}/{original_size:.1f} MB ({size_fraction:.0%}). "
                    f"All subsetting will use image rendering."
                )
                self._corrupted_xref_pdfs.add(pdf_key)
                return True

            return False
        except Exception as e:
            logger.warning(f"xref probe failed: {e}")
            return False

    def _prepare_pdf(
        self,
        pdf_path: Path,
        include_pages: Optional[List[int]] = None,
        exclude_pages: Optional[List[int]] = None
    ) -> Optional[bytes]:
        """
        Prepare PDF for LLM by selecting/excluding specific pages.

        If rasterization was previously triggered (503 error), automatically
        uses the cached rasterized version for better API compatibility.
        """
        # If we've determined rasterization is needed, use cached rasterized PDF
        if self._prefer_rasterized and self._rasterized_pdf_path:
            logger.debug("Using rasterized PDF (503 fallback active)")
            return self._prepare_pdf_internal(
                self._rasterized_pdf_path,
                include_pages=include_pages,
                exclude_pages=exclude_pages
            )

        return self._prepare_pdf_internal(pdf_path, include_pages, exclude_pages)

    def _prepare_pdf_internal(
        self,
        pdf_path: Path,
        include_pages: Optional[List[int]] = None,
        exclude_pages: Optional[List[int]] = None
    ) -> Optional[bytes]:
        """
        Internal: Prepare PDF for LLM by selecting/excluding specific pages.

        If the PDF has corrupted xref (detected once, cached), always renders
        pages as images instead of using select().
        """
        try:
            import tempfile
            import os

            # Suppress MuPDF warnings for corrupted PDFs
            fitz.TOOLS.mupdf_warnings(reset=True)

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            # Determine which pages to keep
            if include_pages is not None:
                pages_to_keep = set(include_pages)
            else:
                pages_to_keep = set(range(1, total_pages + 1))

            if exclude_pages:
                pages_to_keep -= set(exclude_pages)

            if not pages_to_keep:
                logger.error("No pages left after filtering")
                return None

            pages_0indexed = sorted(p - 1 for p in pages_to_keep)

            # If keeping all pages, just read the file directly
            if len(pages_to_keep) == total_pages:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                file_size_mb = len(pdf_bytes) / 1024 / 1024
            elif self._check_xref_corrupted(pdf_path):
                # Corrupted xref: select() can't free resources, render instead
                pdf_bytes = self._render_pages_to_pdf(pdf_path, pages_0indexed)
                if pdf_bytes is None:
                    return None
                file_size_mb = len(pdf_bytes) / 1024 / 1024
            else:
                # Normal PDF: select() works
                doc = fitz.open(pdf_path)
                doc.select(pages_0indexed)
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    doc.save(tmp_path, garbage=4, deflate=True)
                    doc.close()
                    with open(tmp_path, 'rb') as f:
                        pdf_bytes = f.read()
                    file_size_mb = len(pdf_bytes) / 1024 / 1024
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # Suppress any MuPDF warnings that accumulated
            fitz.TOOLS.mupdf_warnings(reset=True)

            compression_config = self.config.get('refine', {}).get('pdf_compression', {})
            payload_limit_mb = compression_config.get('payload_limit_mb', 30)

            if file_size_mb > payload_limit_mb:
                logger.warning(
                    f"Prepared PDF ({len(pages_to_keep)} pages, {file_size_mb:.1f} MB) "
                    f"exceeds limit ({payload_limit_mb} MB). "
                    f"Consider reducing batch size via adaptive splitting."
                )

            logger.debug(
                f"Prepared PDF from {pdf_path.name}: {len(pages_to_keep)} pages "
                f"({file_size_mb:.2f} MB)"
            )
            return pdf_bytes

        except Exception as e:
            logger.error(f"Failed to prepare PDF: {e}")
            return None

    def _render_pages_to_pdf(
        self,
        pdf_path: Path,
        pages_0indexed: List[int],
        dpi: int = 200,
    ) -> Optional[bytes]:
        """
        Render selected pages as images and build a new clean PDF.

        Used when the source PDF has corrupted xref / shared resources that
        prevent proper page subsetting via select().

        Args:
            pdf_path: Source PDF path
            pages_0indexed: Page indices (0-indexed) to render
            dpi: Render resolution

        Returns:
            PDF bytes, or None on failure
        """
        try:
            src = fitz.open(pdf_path)
            new_doc = fitz.open()
            scale = dpi / 72
            mat = fitz.Matrix(scale, scale)

            for page_idx in pages_0indexed:
                page = src[page_idx]
                pix = page.get_pixmap(matrix=mat)
                # Create page at original dimensions
                rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                new_page = new_doc.new_page(width=rect.width, height=rect.height)
                new_page.insert_image(rect, pixmap=pix)

            src.close()

            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                new_doc.save(tmp_path, garbage=4, deflate=True)
                new_doc.close()

                file_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
                logger.info(
                    f"Rendered {len(pages_0indexed)} pages to new PDF: "
                    f"{file_size_mb:.2f} MB ({dpi} DPI)"
                )

                with open(tmp_path, 'rb') as f:
                    return f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to render pages to PDF: {e}")
            return None

    def _ensure_rasterized_pdf(self, pdf_path: Path, target_mb: float = 30.0) -> Optional[Path]:
        """
        Ensure a rasterized version of the PDF exists (session-level cache).

        Creates rasterized PDF on first call, returns cached path on subsequent calls.
        The rasterized PDF is stored at pdf_path.parent / "input_rasterized.pdf".

        Returns:
            Path to rasterized PDF, or None if rasterization fails
        """
        # Check in-memory cache
        if self._rasterized_pdf_path and self._rasterized_pdf_path.exists():
            return self._rasterized_pdf_path

        # Check disk cache
        cache_path = pdf_path.parent / "input_rasterized.pdf"
        if cache_path.exists():
            logger.info(f"Using cached rasterized PDF: {cache_path}")
            self._rasterized_pdf_path = cache_path
            self._prefer_rasterized = True
            return cache_path

        # Create new rasterized PDF (full book, no page subset)
        logger.info("Creating rasterized PDF (JBIG2 binarization, full book)...")
        success, stats = rasterize_to_limit(
            pdf_path, cache_path, pages=None, target_mb=target_mb
        )
        if not success:
            logger.warning("JBIG2 rasterization failed")
            return None

        logger.info(
            f"Rasterized PDF saved: {stats.get('page_count', '?')} pages, "
            f"{stats['output_size_mb']:.1f} MB ({stats['method']} @ {stats['dpi']} DPI)"
        )
        self._rasterized_pdf_path = cache_path
        self._prefer_rasterized = True
        return cache_path

    def _prepare_pdf_rasterized(
        self,
        pdf_path: Path,
        include_pages: Optional[List[int]] = None,
        target_mb: float = 30.0
    ) -> Optional[bytes]:
        """
        Prepare PDF using cached JBIG2 rasterization (for 503 fallback).

        Uses session-level cache: rasterizes full PDF once, then subsets from cache.
        Sets _prefer_rasterized=True so subsequent calls use rasterized version.

        Args:
            pdf_path: Path to the source PDF
            include_pages: Pages to include (1-indexed), None for all
            target_mb: Target file size limit in MB

        Returns:
            PDF bytes, or None if rasterization fails
        """
        rasterized_path = self._ensure_rasterized_pdf(pdf_path, target_mb)
        if rasterized_path is None:
            return None

        # Subset from rasterized PDF using normal _prepare_pdf logic
        # (but bypass _prefer_rasterized check to avoid recursion)
        return self._prepare_pdf_internal(rasterized_path, include_pages=include_pages)

    def analyze_pdf_structure(
        self,
        pdf_path: Path,
        book_title: str,
        state: Optional[RefinerState] = None,
        state_path: Optional[Path] = None,
        pages_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> Tuple[List[TOCNode], Dict]:
        """
        Analyze PDF structure and extract recursive TOC tree.

        Uses a two-phase approach to avoid being misled by printed page numbers:
        1. Detect TOC location → exclude TOC pages
        2. Analyze remaining pages directly for chapter structure

        All PDF→LLM calls use adaptive page splitting: on 503 errors,
        the page count is automatically halved until the API succeeds.

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

        # Create batch context for page counting and TOC detection page selection
        batch_ctx = PdfBatchContext.from_pdf(working_pdf_path)
        logger.info(
            f"PDF has {batch_ctx.total_pages} pages "
            f"(adaptive limit: {self._learner.limit})"
        )

        # Step 1: Detect TOC location
        if state and state.toc_location:
            logger.info("Step 1: Using cached TOC location...")
            toc_location = state.toc_location
        else:
            logger.info("Step 1: Detecting TOC location...")
            toc_location = self.detect_toc_location(working_pdf_path, batch_ctx, artifacts_dir=artifacts_dir)
            if state and toc_location:
                state.toc_location = toc_location
                save_state()

        # Determine pages to exclude from analysis
        if toc_location and toc_location.get('has_toc'):
            toc_start = toc_location['toc_start']
            toc_end = toc_location['toc_end']
            exclude_toc = set(range(toc_start, toc_end + 1))
            logger.info(
                f"TOC detected at pages {toc_start}-{toc_end}. "
                f"Excluding TOC pages, analyzing remaining {batch_ctx.total_pages - len(exclude_toc)} pages."
            )
        else:
            exclude_toc = set()
            logger.info("No TOC detected, will analyze all pages.")

        # Find best TOC pages for reference (may differ from TocDetectionCall result)
        toc_reference = None
        if pages_dir:
            toc_ref_pages = self._find_toc_reference_pages(
                toc_location, pages_dir, batch_ctx.total_pages
            )
            if toc_ref_pages:
                toc_reference = self._clean_toc_text(
                    toc_ref_pages[0], toc_ref_pages[1], pages_dir
                )
                # Also exclude reference TOC pages from analysis
                exclude_toc.update(range(toc_ref_pages[0], toc_ref_pages[1] + 1))

        # === GATE: verify xref detection and setup before spending tokens ===
        is_corrupted = self._check_xref_corrupted(working_pdf_path)
        logger.info(
            f"Pre-flight check: corrupted_xref={is_corrupted}, "
            f"exclude_pages={sorted(exclude_toc) if exclude_toc else 'none'}, "
            f"remaining_pages={batch_ctx.total_pages - len(exclude_toc)}, "
            f"adaptive_limit={self._learner.limit}"
        )
        result = self._analyze_pdf_directly(
            working_pdf_path, book_title, batch_ctx,
            exclude_pages=exclude_toc if exclude_toc else None,
            toc_reference=toc_reference,
            artifacts_dir=artifacts_dir,
        )

        # Mark structure analysis complete
        if state:
            state.structure_analysis_complete = True
            save_state()

        # Validate and fix notes type - remove from non-notes chapters
        self._fix_invalid_notes_type(result.get('chapters', []))

        # Fix containment overlaps: reparent siblings where one fully contains another
        self._fix_containment_overlaps(result.get('chapters', []))

        # Final structural validation
        chapters = result.get('chapters', [])
        issues = validate_chapter_structure(chapters)
        if issues:
            logger.warning(f"Final TOC has {len(issues)} structural issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        # Warn when the cleaned TOC reference suggests many more chapters than
        # the final tree contains. This is advisory only; do not block output.
        if toc_reference:
            self._validate_toc_completeness(
                chapters, toc_reference, batch_ctx.total_pages
            )

        # Convert to TOCNode tree
        toc_tree = dict_list_to_toc_tree(chapters)

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
        batch_ctx: Optional[PdfBatchContext] = None,
        exclude_pages: Optional[set] = None,
        toc_reference: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Analyze PDF directly using DirectAnalysisCall.

        Determines pages, creates a DirectAnalysisCall, and runs it.
        The call object handles batching, 503 recovery, and LLM merge.
        """
        total_pages = batch_ctx.total_pages if batch_ctx else len(fitz.open(pdf_path))
        all_pages = [p for p in range(1, total_pages + 1)
                     if not exclude_pages or p not in exclude_pages]

        call = DirectAnalysisCall(
            self.structure_client, self.structure_model,
            self._prepare_pdf, self._learner, book_title,
            toc_reference=toc_reference,
            prepare_pdf_rasterized=self._prepare_pdf_rasterized,
        )
        analysis_artifacts = artifacts_dir / "direct_analysis" if artifacts_dir else None
        return call.run(pdf_path, all_pages, artifacts_dir=analysis_artifacts)

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

    @staticmethod
    def _fix_containment_overlaps(chapters: List[Dict]) -> None:
        """
        Fix siblings where one's page range fully contains another.

        When sibling A (p100-300) fully contains sibling B (p100-200),
        B should be a child of A, not a sibling. This is a deterministic
        tree restructuring that doesn't require LLM judgment.

        Mutates the list in place. Recurses into all children.
        """
        if not chapters or len(chapters) < 2:
            # Still recurse into existing children
            for ch in (chapters or []):
                StructureAnalyzer._fix_containment_overlaps(ch.get('children', []))
            return

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(chapters):
                container = chapters[i]
                c_start = container.get('start_page', 0)
                c_end = container.get('end_page', 0)

                j = i + 1
                while j < len(chapters):
                    sibling = chapters[j]
                    s_start = sibling.get('start_page', 0)
                    s_end = sibling.get('end_page', 0)

                    if s_start >= c_start and s_end <= c_end:
                        # sibling fully contained in container → reparent
                        moved = chapters.pop(j)
                        container.setdefault('children', []).append(moved)
                        logger.info(
                            f"Reparented '{moved.get('title', '')[:50]}' (p{s_start}-{s_end}) "
                            f"as child of '{container.get('title', '')[:50]}' (p{c_start}-{c_end})"
                        )
                        changed = True
                    elif c_start >= s_start and c_end <= s_end:
                        # container fully contained in sibling → swap roles
                        moved = chapters.pop(i)
                        sibling.setdefault('children', []).append(moved)
                        logger.info(
                            f"Reparented '{moved.get('title', '')[:50]}' (p{c_start}-{c_end}) "
                            f"as child of '{sibling.get('title', '')[:50]}' (p{s_start}-{s_end})"
                        )
                        changed = True
                        break  # restart outer loop since i was removed
                    else:
                        j += 1

                if changed:
                    break  # restart from the top
                i += 1

        # Sort children by start_page and fix levels, then recurse
        for ch in chapters:
            children = ch.get('children', [])
            if children:
                parent_level = ch.get('level', 1)
                children.sort(key=lambda c: c.get('start_page', 0))
                _update_levels_recursive(children, parent_level + 1)
                StructureAnalyzer._fix_containment_overlaps(children)

    @staticmethod
    def _validate_toc_completeness(
        chapters: List[Dict],
        toc_reference: str,
        total_pages: int,
    ) -> None:
        """Log a warning when extracted top-level chapters under-cover the TOC."""
        if not toc_reference:
            return

        numbered_entries = re.findall(r'(?:^|\n)\s*(\d+)\s+\S', toc_reference)
        expected_top_level = len(numbered_entries)
        if expected_top_level == 0:
            numbered_lines = [
                line.strip()
                for line in toc_reference.strip().split('\n')
                if re.match(r'^\d+\s', line.strip())
            ]
            expected_top_level = len(numbered_lines)

        if expected_top_level == 0:
            return

        front_matter_titles = {
            'table of contents', 'inhalt', 'contents', 'sommaire',
            'vorwort', 'preface', 'préface', 'avant-propos',
            'zitierweise', 'siglen', 'abbreviations', 'abkürzungen',
            'introduction', 'introducción', 'einleitung',
        }
        actual_chapters = [
            ch for ch in chapters
            if ch.get('title', '').strip().lower() not in front_matter_titles
        ]
        actual_top_level = len(actual_chapters)

        if actual_top_level >= expected_top_level:
            logger.info(
                f"TOC completeness OK: {actual_top_level} top-level chapters "
                f"(expected ~{expected_top_level} from TOC reference)"
            )
            return

        ratio = actual_top_level / expected_top_level
        logger.warning(
            f"TOC COMPLETENESS ISSUE: Only {actual_top_level} top-level chapters "
            f"found, but TOC reference suggests ~{expected_top_level} numbered "
            f"entries for a {total_pages}-page book (coverage: {ratio:.0%}). "
            f"The structure analysis may be incomplete; consider re-running refine."
        )

        if ratio < 0.5:
            logger.warning(
                f"CRITICAL: Less than half of expected chapters were found "
                f"({actual_top_level}/{expected_top_level}). The resulting TOC tree "
                f"is likely unusable; remaining content may be lumped into a "
                f"single section."
            )

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
