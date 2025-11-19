#!/usr/bin/env python3
"""Test script for OCR backends in unified ocr_pages.py"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf2epub.ocr_pages import ocr_pdf_chunk, extract_pdf_pages, pdf_to_image
from pdf2epub.utils.common import load_config
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Test OCR backend")
    parser.add_argument("--backend", default="azure", help="Backend to test: azure, vision, vertex, vllm, mistral")
    args = parser.parse_args()

    backend = args.backend
    logger.info(f"Testing backend: {backend}")

    # Load config
    config = load_config()
    book_title = config.get("title", "book")

    # Find PDF
    pdf_path = Path("output") / book_title / "input_original.pdf"
    if not pdf_path.exists():
        pdf_path = Path("output") / book_title / "input.pdf"

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return 1

    logger.info(f"Testing Azure backend with: {pdf_path}")

    # Extract page 1
    pdf_bytes = extract_pdf_pages(pdf_path, 1, 1)
    logger.info(f"Extracted page 1: {len(pdf_bytes)} bytes")

    # Test pdf_to_image
    zoom_factor = config.get('vision_ocr_settings', {}).get('zoom_factor', 1.0)
    img_bytes = pdf_to_image(pdf_bytes, zoom_factor)
    logger.info(f"Converted to image: {len(img_bytes)} bytes")

    # Create temp output dir
    output_dir = Path("output") / book_title
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Test OCR with selected backend
    logger.info(f"Testing ocr_pdf_chunk with {backend} backend...")

    # Prepare backend-specific parameters
    kwargs = {
        "pdf_bytes": pdf_bytes,
        "chunk_info": "Test Page 1",
        "images_dir": images_dir,
        "chapter_index": 1,
        "image_counter": 0,
        "backend": backend,
        "config": config
    }

    # Add backend-specific parameters
    if backend == "mistral":
        kwargs["api_key"] = config.get("mistral_api_key")
        kwargs["base_url"] = config.get("mistral_base_url")
    elif backend == "vertex":
        # Vertex needs session, project_id, location
        from google.oauth2 import service_account
        from google.auth.transport.requests import AuthorizedSession
        import json

        sa_key_path = config.get("service_account_key_path", "sa-keys.json")
        with open(sa_key_path, "r") as f:
            sa_key_data = json.load(f)

        project_id = sa_key_data.get("project_id")
        location = config.get("gcp_location", "us-central1")

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
            sa_key_path, scopes=scopes
        )
        session = AuthorizedSession(credentials)

        kwargs["session"] = session
        kwargs["project_id"] = project_id
        kwargs["location"] = location

    try:
        markdown, illustrations, counter = ocr_pdf_chunk(**kwargs)

        logger.success(f"{backend} backend test PASSED!")
        logger.info(f"Markdown length: {len(markdown)} chars")
        logger.info(f"Illustrations: {len(illustrations)}")
        logger.info(f"Image counter: {counter}")

        # Show first 500 chars of markdown
        if markdown:
            logger.info(f"Preview:\n{markdown[:500]}...")

        return 0

    except Exception as e:
        logger.error(f"{backend} backend test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
