# Refine Pipeline Refactor - Handoff Document

## What Was Done (commit 727c2f8)

### PDF Compression Cleanup
- **JBIG2**: Rewrote to per-page mode (no `-s` symbol table flag), each page individually binarized with Otsu then compressed with `jbig2 -p`
- **Binarized PNG**: `compress_pdf()` now always does Otsu binarization + 1-bit PNG + PyMuPDF deflate. All JPEG logic deleted.
- **CCITT G4**: Deleted entirely (8x larger than JBIG2, no value)
- **DPI levels**: Reduced from `[150, 120, 100, 80, 72]` to `[150, 120]`
- **tqdm**: Added progress bars to both rasterization and compression

### Bug Fixes
- **TOC page conversion**: `convert_toc_page_to_original()` was double-converting page numbers. LLM sees "PDF Page: X" patches which are already original page numbers. Removed the conversion entirely.
- **Redundant rasterization guard**: Added `_pdf_already_rasterized` flag. When Step 0 already rasterized the PDF, 503 fallback no longer re-rasterizes (pointless on already-rasterized PDFs).

### Files Modified
- `pdf2epub/pdf_compressor.py` - Simplified to binarized PNG only
- `pdf2epub/refine/pdf_rasterizer.py` - JBIG2 per-page, deleted CCITT
- `pdf2epub/refine/structure_analyzer.py` - Simplified fallbacks, fixed TOC bug, added rasterization guard
- `pdf2epub/utils/pdf_utils.py` - Simplified compression cascade

---

## Key Findings From Testing

### Format Compatibility with Google API
Tested on "Les Chevaliers de la Table ronde" (1084 pages, 64MB original):

| Format | 10 pages | 300 pages | 500 pages | 900 pages |
|--------|----------|-----------|-----------|-----------|
| JBIG2 (our `_jbig2_to_pdf`) | OK | OK | 503 | 503 |
| Binarized PNG (PyMuPDF) | OK | OK | not tested | 503 |
| Original PDF (unmodified) | 400 INVALID_ARGUMENT | - | - | - |

**Conclusion**: Google's 503 is a **page count** issue, not a format issue. Both JBIG2 and binarized PNG work fine for small page counts. The threshold varies (300 OK, 500 fails on this book) and is likely not fixed across different PDFs or API load conditions.

### Compression Sizes (1084 pages, 150 DPI)
| Method | Size | Notes |
|--------|------|-------|
| JBIG2 per-page | 16 MB | Smallest, but same 503 issue as PNG |
| Binarized PNG (PyMuPDF deflate) | 26 MB | Google-compatible |
| Binarized JPEG q60 | 111 MB | JPEG is terrible for binary images, deleted |
| Original scanned PDF | 64 MB | Google returns 400 on large originals |

### Single Page Comparison (page 555, 120 DPI)
| Format | Size |
|--------|------|
| 1-bit PNG | 17 KB |
| TIFF G4 | 17 KB |
| Binarized → L → JPEG q60 | 121 KB |
| Grayscale JPEG q60 (no binarize) | 62 KB |

JPEG fundamentally cannot compress sharp binary edges efficiently (DCT ringing). The "19 MB binarized JPEG" claim from the original plan was actually binarized PNG all along (file was saved as `.png`, not `.jpg`).

---

## Remaining Work: Adaptive Batch Splitting

### Problem
The current pipeline uses fixed batch sizes (900 pages via `PdfBatchContext.batch_size`). When Google 503s on a batch, the current code either retries the same request or tries rasterization fallback. Neither helps because the issue is page count.

### Design
All PDF→Google API calls should implement adaptive page-count reduction:

1. Start with current batch size (900 pages)
2. On 503, **halve the page count** and retry (450 → 225 → ...)
3. Sub-batch results must be **merged** by the caller
4. "Learned" page limit should persist within a session (once we discover 300 works, don't try 900 again)

### Architecture Problem
Currently, PDF→API calls with 503 handling are scattered across **6+ methods** in `structure_analyzer.py`:
- `detect_toc_location` (+ `_detect_toc_location_rasterized`)
- `extract_toc_structure`
- `match_toc_with_content`
- `_analyze_pdf_directly`
- `_analyze_pdf_batched`
- `_match_toc_batched`

Each has its own `try/except` for 503 with copy-pasted rasterization fallback logic. This should be **centralized** into a single method that handles:
- PDF preparation (page subsetting, compression)
- API call with retry
- 503 → page count halving
- Result merging

### Where Batch Context Lives
- `pdf2epub/refine/pdf_batching.py` - `PdfBatchContext` dataclass, `create_content_batches()`, `get_toc_detection_pages()`
- `batch_size=900`, `overlap=50`, `page_limit=1000`

---

## Other Known Issues

### `compress_pdf_bytes` (not related to this refactor)
`pdf2epub/ocr_backends.py:278` imports `compress_pdf_bytes` from `pdf_utils`, but this function was never implemented. It's used in the Vertex AI Mistral OCR backend for compressing PDF chunks > 20MB before base64 encoding. Likely never triggered in practice.

### JBIG2 Still Used in Pipeline
JBIG2 rasterization (`rasterize_to_limit`) is still the primary compression method in:
- `_compress_pdf_to_limit()` (refine Step 0)
- `_prepare_pdf()` (subset compression)
- `preprocess_pdf()` (preprocessing)

This works fine for local compression and for Google API when page count is low. The binarized PNG fallback (`compress_pdf()`) is used when JBIG2 is unavailable (no `jbig2` binary installed).
