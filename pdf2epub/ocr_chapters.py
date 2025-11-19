"""
Chapter aggregation from OCR'd pages.

This module aggregates page-level OCR results into chapter markdown files.
It assumes pages have already been OCR'd by ocr_pages.py.
"""

import json
import argparse
from pathlib import Path
from typing import Dict
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from loguru import logger
from .utils.logging_config import configure_logging
from .utils.common import load_config, load_book_structure

# Configure logger
logger = configure_logging()


def aggregate_pages(
    pages_dir: Path,
    start_page: int,
    end_page: int,
    title: str = None
) -> str:
    """Aggregate page markdown files into one document.

    Args:
        pages_dir: Directory containing page_*.md files
        start_page: First page number
        end_page: Last page number
        title: Optional title to prepend

    Returns:
        Combined markdown content
    """
    parts = []

    if title:
        parts.append(f"# {title}\n")

    for page_num in range(start_page, end_page + 1):
        page_file = pages_dir / f"page_{page_num:03d}.md"
        if page_file.exists():
            with open(page_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parts.append(content)
        else:
            logger.warning(f"Page {page_num} file not found: {page_file}")

    return "\n\n".join(parts)


def aggregate_chapter(
    chapter: Dict,
    chapter_index: int,
    output_dir: Path,
    pages_dir: Path
) -> None:
    """Aggregate pages into chapter markdown.

    Args:
        chapter: Chapter dictionary with title, start_page, end_page, and optional subchapters
        chapter_index: Chapter number
        output_dir: Base output directory
        pages_dir: Directory containing OCR'd pages
    """
    chapter_title = chapter["title"]
    start_page = chapter["start_page"]
    end_page = chapter["end_page"]
    subchapters = chapter.get("subchapters", [])

    # If there are subchapters, extend the end page to include all subchapter pages
    if subchapters:
        # Find the last page of the last subchapter
        last_subchapter_end = max(sub["end_page"] for sub in subchapters)
        actual_end_page = max(end_page, last_subchapter_end)
        logger.info(f"Aggregating Chapter {chapter_index}: {chapter_title}")
        logger.info(f"  Chapter header pages: {start_page}-{end_page}")
        logger.info(f"  Contains {len(subchapters)} subchapters")
        for i, sub in enumerate(subchapters, 1):
            logger.info(f"    Subchapter {i}: pages {sub['start_page']}-{sub['end_page']}")
        logger.info(f"  Total pages to aggregate: {start_page}-{actual_end_page}")
        end_page = actual_end_page
    else:
        logger.info(f"Aggregating Chapter {chapter_index}: {chapter_title} (pages {start_page}-{end_page})")

    # Aggregate pages
    markdown = aggregate_pages(pages_dir, start_page, end_page, chapter_title)

    # Save
    ocr_markdown_dir = output_dir / "ocr_markdown"
    ocr_markdown_dir.mkdir(parents=True, exist_ok=True)
    output_file = ocr_markdown_dir / f"chapter_{chapter_index}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    logger.success(f"Saved Chapter {chapter_index} to {output_file}")


def aggregate_matter(
    matter_type: str,
    start_page: int,
    end_page: int,
    output_dir: Path,
    pages_dir: Path
) -> None:
    """Aggregate matter pages (front or back).

    Args:
        matter_type: "front" or "back"
        start_page: First page number
        end_page: Last page number
        output_dir: Base output directory
        pages_dir: Directory containing OCR'd pages
    """
    logger.info(f"Aggregating {matter_type} matter (pages {start_page}-{end_page})")

    # Aggregate pages
    title = f"{matter_type.capitalize()} Matter"
    markdown = aggregate_pages(pages_dir, start_page, end_page, title)

    # Save
    ocr_markdown_dir = output_dir / "ocr_markdown"
    ocr_markdown_dir.mkdir(parents=True, exist_ok=True)
    output_file = ocr_markdown_dir / f"{matter_type}_matter.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    logger.success(f"Saved {matter_type} matter to {output_file}")


def all_pages_complete_for_structure(pages_dir: Path, structure: Dict) -> bool:
    """Check if all pages required by structure are OCR'd.

    Args:
        pages_dir: Directory containing pages and progress file
        structure: Book structure dict from breakdown

    Returns:
        True if all required pages are complete
    """
    progress_file = pages_dir / "ocr_progress.json"
    if not progress_file.exists():
        logger.info("No progress file found")
        return False

    with open(progress_file) as f:
        progress = json.load(f)

    pages_processed = set(progress.get('pages_processed', []))

    # Calculate required pages
    required_pages = set()

    # Front matter
    if "front_matter" in structure:
        fm = structure["front_matter"]
        required_pages.update(range(fm["start_page"], fm["end_page"] + 1))

    # Chapters (including subchapters)
    for chapter in structure["chapters"]:
        start = chapter["start_page"]
        end = chapter["end_page"]

        if chapter.get("subchapters"):
            last_sub_end = max(sub["end_page"] for sub in chapter["subchapters"])
            end = max(end, last_sub_end)

        required_pages.update(range(start, end + 1))

    # Back matter
    if "back_matter" in structure:
        bm = structure["back_matter"]
        required_pages.update(range(bm["start_page"], bm["end_page"] + 1))

    # Check if all complete
    missing = required_pages - pages_processed
    if missing:
        missing_list = sorted(list(missing))
        if len(missing_list) <= 10:
            logger.info(f"Missing pages: {missing_list}")
        else:
            logger.info(f"Missing pages: {missing_list[:10]}... ({len(missing)} total)")
        return False

    logger.info(f"All {len(required_pages)} required pages are complete")
    return True


def main():
    """Main function: check OCR status, trigger if needed, then aggregate chapters."""
    parser = argparse.ArgumentParser(description="Aggregate OCR'd pages into chapters")
    parser.add_argument("-i", "--input", help="Path to input PDF file (optional)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume OCR if incomplete")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate, assume OCR is done")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return

    # Load book structure
    structure = load_book_structure(book_title)

    # Setup directories
    output_dir = Path("output") / book_title
    pages_dir = output_dir / "pages"
    images_dir = output_dir / "images"

    # Check if OCR is complete
    if not args.aggregate_only:
        if not all_pages_complete_for_structure(pages_dir, structure):
            logger.info("Pages not complete, running OCR first...")

            # Import and run page-wise OCR
            from pdf2epub.ocr_pages import ocr_full_book_pagewise

            # Determine OCR backend
            ocr_backend = config.get("ocr_backend", "vertex").lower()
            logger.info(f"Using OCR backend: {ocr_backend}")

            # Setup backend-specific configuration
            session = None
            project_id = None
            location = None
            api_key = None
            base_url = None

            if ocr_backend == "vertex":
                # Setup Google Cloud authentication
                sa_key_path = config.get("service_account_key_path", "sa-keys.json")

                if not Path(sa_key_path).exists():
                    raise FileNotFoundError(f"Service account key file not found: {sa_key_path}")

                # Load project ID from service account JSON
                with open(sa_key_path, "r") as f:
                    sa_key_data = json.load(f)

                project_id = sa_key_data.get("project_id")
                if not project_id:
                    raise ValueError(f"No project_id found in service account key file: {sa_key_path}")

                location = config.get("gcp_location", "us-central1")

                logger.info(f"Using GCP project: {project_id}, location: {location}")

                # Create authenticated session
                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                credentials = service_account.Credentials.from_service_account_file(
                    sa_key_path, scopes=scopes
                )
                session = AuthorizedSession(credentials)

            elif ocr_backend == "mistral":
                # Get Mistral API key and base URL
                api_key = config.get("mistral_api_key")
                if not api_key:
                    raise ValueError("mistral_api_key not found in config.yaml")

                base_url = config.get("mistral_base_url", "https://api.mistral.ai/v1")
                logger.info(f"Using Mistral API with key: {api_key[:8]}... at {base_url}")

            elif ocr_backend == "vllm":
                # VLLM backend uses init_client from vllm.py
                logger.info("Using VLLM backend")

            elif ocr_backend == "azure":
                # Azure Document Intelligence backend
                logger.info("Using Azure Document Intelligence backend")

            elif ocr_backend == "vision":
                # Google Cloud Vision API backend
                logger.info("Using Google Cloud Vision backend")

            else:
                raise ValueError(f"Unknown OCR backend: {ocr_backend}. Supported: vertex, mistral, vllm, azure, vision")

            # Determine PDF path
            if args.input:
                pdf_path = Path(args.input)
            else:
                # Check for input_original.pdf first, then fall back to input.pdf
                pdf_original_path = Path("output") / book_title / "input_original.pdf"
                pdf_path = Path("output") / book_title / "input.pdf"

                if pdf_original_path.exists():
                    pdf_path = pdf_original_path
                    logger.info(f"Using original PDF: {pdf_path}")
                elif pdf_path.exists():
                    logger.info(f"Using PDF: {pdf_path}")
                else:
                    logger.error(f"No PDF file found. Looked for: {pdf_original_path} and {pdf_path}")
                    return

            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return

            # Run OCR
            ocr_full_book_pagewise(
                pdf_path=pdf_path,
                output_dir=output_dir,
                session=session,
                project_id=project_id,
                location=location,
                backend=ocr_backend,
                api_key=api_key,
                base_url=base_url,
                resume=args.resume,
                config=config
            )
        else:
            logger.info("All required pages already OCR'd")

    # Aggregate chapters
    logger.info("Aggregating chapters from pages...")

    # Front matter
    if "front_matter" in structure:
        aggregate_matter(
            "front",
            structure["front_matter"]["start_page"],
            structure["front_matter"]["end_page"],
            output_dir,
            pages_dir
        )

    # Chapters
    for chapter_idx, chapter in enumerate(structure["chapters"], 1):
        aggregate_chapter(chapter, chapter_idx, output_dir, pages_dir)

    # Back matter
    if "back_matter" in structure:
        aggregate_matter(
            "back",
            structure["back_matter"]["start_page"],
            structure["back_matter"]["end_page"],
            output_dir,
            pages_dir
        )

    logger.success(f"All chapters aggregated! Files saved to {output_dir / 'ocr_markdown'}")

    # Summary
    logger.info(f"\n=== Aggregation Summary ===")
    logger.info(f"Chapters: {len(structure['chapters'])} aggregated")

    if "front_matter" in structure:
        logger.info(f"Front matter: ✓ Aggregated")

    if "back_matter" in structure:
        logger.info(f"Back matter: ✓ Aggregated")


if __name__ == "__main__":
    main()
