#!/usr/bin/env python3
"""
pdf_compressor.py - Standalone PDF compression utility

Compresses PDF files by:
1. Rendering each page at specified DPI
2. Binarizing with Otsu's method (1-bit black/white)
3. Saving as PNG and assembling into a new PDF with deflate compression

Usage:
  python pdf_compressor.py input.pdf output.pdf --dpi 150
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from loguru import logger
from tqdm import tqdm
from .utils.logging_config import configure_logging

# Configure logger
logger = configure_logging()


def compress_pdf(input_path, output_path, dpi=150):
    """
    Compress a PDF by rasterizing each page with Otsu binarization and PNG encoding.

    Each page is rendered at the specified DPI, binarized to 1-bit black/white,
    saved as PNG, and assembled into a new PDF with deflate compression.

    Args:
        input_path (str): Path to input PDF file
        output_path (str): Path to save compressed PDF
        dpi (int): Resolution for rendering PDF pages (default: 150)

    Returns:
        tuple: (bool success, dict stats)
    """
    try:
        input_size = os.path.getsize(input_path)

        # Open the PDF
        pdf_document = fitz.open(input_path)
        page_count = len(pdf_document)

        # Create a new PDF for output
        output_pdf = fitz.open()

        # Calculate zoom factor based on DPI (72 is the base DPI)
        zoom = dpi / 72

        # Set up temp directory for page images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            from .refine.pdf_rasterizer import _binarize_image

            for page_num, page in enumerate(tqdm(pdf_document, desc=f"Binarize {dpi}dpi", unit="page")):
                # Get page dimensions
                rect = page.rect

                # Render grayscale -> Otsu binarize -> save as 1-bit PNG
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img = img.convert("L")
                img = _binarize_image(img)

                img_path = temp_dir_path / f"page_{page_num + 1}.png"
                img.save(img_path, "PNG")

                # Add image back to new PDF
                new_page = output_pdf.new_page(width=rect.width, height=rect.height)
                new_page.insert_image(rect, filename=str(img_path))

            logger.info("Saving compressed PDF...")
            output_pdf.save(output_path, garbage=4, deflate=True, clean=True)
            output_pdf.close()

        # Get stats
        output_size = os.path.getsize(output_path)
        compression_ratio = input_size / output_size if output_size > 0 else 0
        saved_percentage = (1 - output_size / input_size) * 100

        stats = {
            "input_size_mb": input_size / 1024 / 1024,
            "output_size_mb": output_size / 1024 / 1024,
            "compression_ratio": compression_ratio,
            "saved_percentage": saved_percentage,
            "page_count": page_count,
        }

        logger.success(f"PDF compression complete:")
        logger.info(f"Original size: {stats['input_size_mb']:.2f} MB")
        logger.info(f"Compressed size: {stats['output_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        logger.info(f"Space saved: {stats['saved_percentage']:.2f}%")

        return True, stats

    except Exception as e:
        logger.error(f"Error compressing PDF: {e}")
        return False, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Compress PDF by binarizing pages to 1-bit PNG"
    )
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("output", help="Output PDF file path")
    parser.add_argument(
        "--dpi", type=int, default=150, help="DPI for rendering (default: 150)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        logger.error(f"Error: Input file '{args.input}' not found")
        return 1

    logger.info(f"Compressing {args.input} to {args.output}")
    logger.info(f"Settings: DPI={args.dpi}")

    success, _ = compress_pdf(args.input, args.output, args.dpi)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
