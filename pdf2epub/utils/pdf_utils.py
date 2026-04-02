"""
Common PDF utility functions.

Provides:
- add_page_number_patches: Add PDF page number patches to help LLM avoid confusion
- preprocess_pdf: Full preprocessing pipeline (patches + compression)
"""

import shutil
import tempfile
import os
from pathlib import Path
from loguru import logger
import fitz  # PyMuPDF

from ..pdf_compressor import compress_pdf


def extract_cover_image(pdf_path, output_path, page_number=1, dpi=150):
    """
    Extract a page from PDF as a JPEG image for use as cover.

    Args:
        pdf_path: Path to the PDF file
        output_path: Path for the output image (e.g., cover.jpg)
        page_number: Page number to extract (1-indexed)
        dpi: Resolution for the image

    Returns:
        Path to the extracted image, or None if failed
    """
    try:
        doc = fitz.open(pdf_path)

        # Convert to 0-indexed
        page_idx = page_number - 1
        if page_idx < 0 or page_idx >= len(doc):
            logger.error(f"Page {page_number} out of range (1-{len(doc)})")
            doc.close()
            return None

        page = doc[page_idx]

        # Create a matrix for the desired DPI
        # Default PDF resolution is 72 DPI
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pixmap = page.get_pixmap(matrix=matrix)

        # Save as JPEG
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(output_path))

        doc.close()
        logger.info(f"Extracted cover image from page {page_number} to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to extract cover image: {e}")
        return None


def add_page_number_patches(pdf_path, output_path=None):
    """
    Add white patches with actual PDF page numbers in the corners of each page.
    This helps the LLM avoid being misled by printed page numbers in the book.

    Args:
        pdf_path: Path to the input PDF
        output_path: Path for the output PDF (if None, overwrites input)

    Returns:
        Path to the patched PDF
    """
    if output_path is None:
        output_path = pdf_path

    # Check if we're overwriting the same file
    same_file = (Path(pdf_path).resolve() == Path(output_path).resolve())

    try:
        doc = fitz.open(pdf_path)

        total_pages = len(doc)
        for page_num, page in enumerate(doc, 1):
            # Show progress every 10 pages or at the end
            if page_num % 10 == 0 or page_num == total_pages:
                logger.info(f"Processing page {page_num}/{total_pages}...")

            # Get page dimensions
            rect = page.rect
            width = rect.width
            height = rect.height

            # Define patch size relative to page dimensions
            # Use ~10% of width and ~5% of height as base, with minimum sizes
            patch_width = max(120, int(width * 0.15))  # 15% of page width, min 120px
            patch_height = max(50, int(height * 0.05))  # 5% of page height, min 50px
            # Scale font size based on patch height
            font_size = max(16, int(patch_height * 0.32))  # ~32% of patch height, min 16pt

            # First: Add full-width white strips at top and bottom to cover printed page numbers
            # These are drawn first (behind the corner patches)
            # Use 150% of patch height to ensure printed page numbers are fully covered
            strip_height = int(patch_height * 1.5)

            # Top strip (covers header/printed page numbers at top)
            top_strip = fitz.Rect(0, 0, width, strip_height)
            page.draw_rect(top_strip, color=(1, 1, 1), fill=(1, 1, 1))

            # Bottom strip (covers footer/printed page numbers at bottom)
            bottom_strip = fitz.Rect(0, height - strip_height, width, height)
            page.draw_rect(bottom_strip, color=(1, 1, 1), fill=(1, 1, 1))

            # Corner positions: top-left, top-right, bottom-left, bottom-right
            # Adjusted margins to ensure better coverage
            corner_positions = [
                (5, 5),  # top-left
                (width - patch_width - 5, 5),  # top-right
                (5, height - patch_height - 5),  # bottom-left
                (width - patch_width - 5, height - patch_height - 5)  # bottom-right
            ]

            for x, y in corner_positions:
                white_rect = fitz.Rect(x, y, x + patch_width, y + patch_height)

                # Draw white filled rectangle
                page.draw_rect(white_rect, color=(1, 1, 1), fill=(1, 1, 1))

                # Add the black border
                page.draw_rect(white_rect, color=(0, 0, 0), width=1, fill=None)

                # Add the text on top
                text_point = fitz.Point(x + 15, y + patch_height / 2 + 6)
                page.insert_text(
                    text_point,
                    f"PDF Page: {page_num}",  # More descriptive label
                    fontsize=font_size,
                    color=(0, 0, 0),
                    fontname="helv"  # Explicitly specify font
                )

        # Get page count before closing
        page_count = len(doc)

        # Save the modified PDF
        # Try to save with garbage collection for PDFs with malformed dictionaries
        save_successful = False
        save_options = [
            {"garbage": 4, "clean": True},  # Most aggressive cleanup
            {"garbage": 3},  # Medium cleanup
            {"garbage": 0},  # No cleanup (for clean PDFs)
        ]

        for opts in save_options:
            try:
                if same_file:
                    # When overwriting the same file, we need to use a temp file
                    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf", dir=Path(output_path).parent)
                    os.close(temp_fd)  # Close the file descriptor

                    doc.save(temp_path, **opts)
                    doc.close()

                    # Replace the original with the temp file
                    os.replace(temp_path, output_path)
                else:
                    # Different files, can save directly
                    doc.save(output_path, **opts)
                    doc.close()

                save_successful = True
                break
            except Exception as save_error:
                if "invalid key in dict" in str(save_error):
                    logger.debug(f"Retrying save with different options due to: {save_error}")
                    continue
                else:
                    raise save_error

        if not save_successful:
            raise Exception("Could not save PDF with any garbage collection option")

        logger.info(f"Added page number patches to {page_count} pages")
        return Path(output_path)

    except Exception as e:
        logger.error(f"Failed to add page number patches: {e}")
        # Return original path if patching fails
        return Path(pdf_path)


def _has_page_stamps(pdf_path) -> bool:
    """Check if a processed PDF has 'PDF Page: X' stamps on the first page."""
    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text()
        doc.close()
        return 'PDF Page:' in text
    except Exception:
        return False


def preprocess_pdf(input_pdf, output_dir):
    """
    Preprocess PDF: add page number patches and compress if necessary.
    Keeps original as input_original.pdf and creates processed version as input.pdf.
    Returns the path to the PDF that should be used.

    Args:
        input_pdf: Path to the original PDF file
        output_dir: Output directory for processed files

    Returns:
        Path to the processed PDF (input.pdf)
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    processed_pdf = output_dir / "input.pdf"
    original_pdf = output_dir / "input_original.pdf"

    # If already processed, verify stamps are present before reusing
    if processed_pdf.exists() and original_pdf.exists():
        if _has_page_stamps(processed_pdf):
            logger.info("Using existing preprocessed PDF (stamps verified)")
            return processed_pdf
        else:
            logger.warning("Existing input.pdf missing page stamps, re-patching...")

    # First time processing - save the original
    if not original_pdf.exists():
        shutil.copy2(input_pdf, original_pdf)
        logger.info(f"Saved original PDF as: {original_pdf}")

    # Get file size in MB (check original before adding stamps)
    file_size_mb = original_pdf.stat().st_size / (1024 * 1024)

    # Compress BEFORE adding stamps — stamp patches (white rectangles)
    # distort Otsu binarization thresholds, causing JBIG2 to produce
    # all-black pages on scanned PDFs.
    source_for_stamps = original_pdf
    if file_size_mb > 45:
        logger.warning(f"PDF file size ({file_size_mb:.2f}MB) exceeds 45MB. Compressing...")

        temp_output = output_dir / "compressed_temp.pdf"

        # Step 1: Try JBIG2 rasterization (best compression for scanned books)
        from ..refine.pdf_rasterizer import rasterize_to_limit

        logger.info("Attempting JBIG2 compression...")
        success, stats = rasterize_to_limit(
            original_pdf, temp_output, pages=None, target_mb=45.0
        )

        if success:
            source_for_stamps = temp_output
            logger.info(
                f"JBIG2 compressed: {stats['output_size_mb']:.2f}MB @ {stats['dpi']} DPI"
            )
        else:
            # Step 2: Fallback to binarized PNG at 120 DPI
            logger.warning("JBIG2 failed, falling back to binarized PNG at 120 DPI...")
            try:
                success, stats = compress_pdf(
                    str(original_pdf),
                    str(temp_output),
                    dpi=120,
                )

                if success and stats["output_size_mb"] < file_size_mb:
                    source_for_stamps = temp_output
                    logger.info(f"Binarized PNG compressed: {stats['output_size_mb']:.2f}MB")
                else:
                    logger.warning("Binarized PNG compression did not help")
                    if temp_output.exists():
                        temp_output.unlink()
            except Exception as e:
                logger.error(f"Compression attempt failed: {e}")
                if temp_output.exists():
                    temp_output.unlink()

    # Add page number patches AFTER compression
    logger.info("Adding page number patches to PDF...")
    patched_pdf = add_page_number_patches(source_for_stamps, processed_pdf)

    # Clean up temp file if used
    temp_output = output_dir / "compressed_temp.pdf"
    if temp_output.exists() and temp_output != processed_pdf:
        temp_output.unlink()

    # If patching failed, use the source directly
    if patched_pdf != processed_pdf:
        logger.warning("PDF patching failed, using PDF without patches")
        shutil.copy2(source_for_stamps, processed_pdf)

    # Check final file size
    final_size_mb = processed_pdf.stat().st_size / (1024 * 1024)
    if final_size_mb > 45:
        logger.warning(f"PDF is still {final_size_mb:.2f}MB (larger than 45MB) after compression")

    return processed_pdf
