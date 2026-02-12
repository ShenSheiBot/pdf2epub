"""Tests for _prepare_pdf corrupted xref detection and _render_pages_to_pdf."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz


def _make_test_pdf(num_pages: int, tmp_dir: str, filename: str = "test.pdf") -> Path:
    """Create a simple multi-page PDF for testing."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        tw = fitz.TextWriter(page.rect)
        tw.append((72, 72), f"Page {i + 1}", fontsize=24)
        tw.write_text(page)
    path = Path(tmp_dir) / filename
    doc.save(str(path))
    doc.close()
    return path


class TestRenderPagesToPdf:
    """Test _render_pages_to_pdf produces valid, compact PDFs."""

    def test_renders_subset_of_pages(self):
        from pdf2epub.refine.structure_analyzer import StructureAnalyzer

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(10, tmp_dir)
            analyzer = StructureAnalyzer.__new__(StructureAnalyzer)

            result = analyzer._render_pages_to_pdf(pdf_path, [0, 1, 2], dpi=72)

            assert result is not None
            # Verify it's a valid PDF with 3 pages
            doc = fitz.open(stream=result, filetype="pdf")
            assert len(doc) == 3
            doc.close()

    def test_renders_single_page(self):
        from pdf2epub.refine.structure_analyzer import StructureAnalyzer

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(5, tmp_dir)
            analyzer = StructureAnalyzer.__new__(StructureAnalyzer)

            result = analyzer._render_pages_to_pdf(pdf_path, [3], dpi=72)

            assert result is not None
            doc = fitz.open(stream=result, filetype="pdf")
            assert len(doc) == 1
            doc.close()

    def test_output_is_compact(self):
        """Rendered PDF should be much smaller than a corrupted PDF would be."""
        from pdf2epub.refine.structure_analyzer import StructureAnalyzer

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(20, tmp_dir)
            analyzer = StructureAnalyzer.__new__(StructureAnalyzer)

            result = analyzer._render_pages_to_pdf(pdf_path, [0, 1], dpi=72)
            full = analyzer._render_pages_to_pdf(pdf_path, list(range(20)), dpi=72)

            # 2-page render should be significantly smaller than 20-page render
            assert len(result) < len(full) * 0.5


class TestCorruptedXrefDetection:
    """Test _check_xref_corrupted and _prepare_pdf with cached detection."""

    def _make_analyzer(self):
        from pdf2epub.refine.structure_analyzer import StructureAnalyzer
        analyzer = StructureAnalyzer.__new__(StructureAnalyzer)
        analyzer.config = {
            'refine': {
                'pdf_compression': {
                    'payload_limit_mb': 30.0,
                    'compress_if_exceeds': True,
                }
            }
        }
        analyzer._corrupted_xref_pdfs = set()
        return analyzer

    def test_normal_pdf_not_corrupted(self):
        """Normal PDF: _check_xref_corrupted returns False, select() used."""
        analyzer = self._make_analyzer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(30, tmp_dir)
            assert analyzer._check_xref_corrupted(pdf_path) is False
            result = analyzer._prepare_pdf(pdf_path, include_pages=[1, 2])
            assert result is not None
            doc = fitz.open(stream=result, filetype="pdf")
            assert len(doc) == 2
            doc.close()

    def test_corrupted_pdf_detected_and_cached(self):
        """Corrupted xref detected once, then cached for all future calls."""
        analyzer = self._make_analyzer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(30, tmp_dir)

            # Simulate corrupted xref: patch save to produce bloated output
            original_save = fitz.Document.save

            def bloated_save(self_doc, path, **kwargs):
                original_save(self_doc, path, **kwargs)
                with open(path, 'ab') as f:
                    f.write(b'\x00' * 2 * 1024 * 1024)

            with patch.object(fitz.Document, 'save', bloated_save):
                assert analyzer._check_xref_corrupted(pdf_path) is True

            # Cached — no need to probe again
            assert str(pdf_path.resolve()) in analyzer._corrupted_xref_pdfs
            assert analyzer._check_xref_corrupted(pdf_path) is True

    def test_corrupted_pdf_uses_render(self):
        """When xref is known corrupted, _prepare_pdf uses _render_pages_to_pdf."""
        analyzer = self._make_analyzer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(30, tmp_dir)
            # Pre-mark as corrupted
            analyzer._corrupted_xref_pdfs.add(str(pdf_path.resolve()))

            result = analyzer._prepare_pdf(pdf_path, include_pages=[1, 2, 3])
            assert result is not None
            doc = fitz.open(stream=result, filetype="pdf")
            assert len(doc) == 3
            doc.close()

    def test_corrupted_cache_works_for_large_batches(self):
        """Even 90% of pages still renders when xref is cached as corrupted."""
        analyzer = self._make_analyzer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _make_test_pdf(30, tmp_dir)
            analyzer._corrupted_xref_pdfs.add(str(pdf_path.resolve()))

            # Request 27/30 pages — old threshold (page_fraction < 0.8) would miss this
            pages = list(range(1, 28))
            result = analyzer._prepare_pdf(pdf_path, include_pages=pages)
            assert result is not None
            doc = fitz.open(stream=result, filetype="pdf")
            assert len(doc) == 27
            doc.close()


class TestExcludePages:
    """Test that _analyze_pdf_directly respects exclude_pages."""

    def test_exclude_pages_filters_page_list(self):
        """exclude_pages should remove specified pages from all_pages."""
        from pdf2epub.refine import adaptive_pdf_call as apc_module
        from pdf2epub.refine.structure_analyzer import StructureAnalyzer
        from pdf2epub.refine.adaptive_pdf_call import PdfPageLimitLearner

        analyzer = StructureAnalyzer.__new__(StructureAnalyzer)
        analyzer._learner = PdfPageLimitLearner(initial_limit=100, min_limit=5)

        batch_ctx = MagicMock()
        batch_ctx.total_pages = 20

        captured_pages = []

        def capture_run(pages, process_batch, learner, is_503_fn, operation_name, overlap=0):
            captured_pages.extend(pages)
            return [{'chapters': [], 'author': 'Test', 'language': 'en',
                     'is_vertical_text': False, 'has_footnotes': False,
                     'cover_page': 1}]

        # Patch in adaptive_pdf_call module where AdaptivePdfCall.run() calls it
        with patch.object(apc_module, 'run_adaptive_batches', capture_run):
            analyzer._prepare_pdf = MagicMock(return_value=b'%PDF-fake')
            analyzer.structure_client = MagicMock()
            analyzer.structure_model = 'test'

            analyzer._analyze_pdf_directly(
                Path("/fake.pdf"), "Test Book", batch_ctx,
                exclude_pages={3, 4, 5, 10}
            )

        expected = [p for p in range(1, 21) if p not in {3, 4, 5, 10}]
        assert captured_pages == expected

    def test_no_exclude_pages_includes_all(self):
        """Without exclude_pages, all pages are included."""
        from pdf2epub.refine import adaptive_pdf_call as apc_module
        from pdf2epub.refine.structure_analyzer import StructureAnalyzer
        from pdf2epub.refine.adaptive_pdf_call import PdfPageLimitLearner

        analyzer = StructureAnalyzer.__new__(StructureAnalyzer)
        analyzer._learner = PdfPageLimitLearner(initial_limit=100, min_limit=5)

        batch_ctx = MagicMock()
        batch_ctx.total_pages = 10

        captured_pages = []

        def capture_run(pages, process_batch, learner, is_503_fn, operation_name, overlap=0):
            captured_pages.extend(pages)
            return [{'chapters': [], 'author': 'Test', 'language': 'en',
                     'is_vertical_text': False, 'has_footnotes': False,
                     'cover_page': 1}]

        with patch.object(apc_module, 'run_adaptive_batches', capture_run):
            analyzer._prepare_pdf = MagicMock(return_value=b'%PDF-fake')
            analyzer.structure_client = MagicMock()
            analyzer.structure_model = 'test'

            analyzer._analyze_pdf_directly(
                Path("/fake.pdf"), "Test Book", batch_ctx
            )

        assert captured_pages == list(range(1, 11))
