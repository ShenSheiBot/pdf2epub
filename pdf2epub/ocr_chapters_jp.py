#!/usr/bin/env python3
"""OCR chapters using either Azure or Vision backend for Japanese vertical text."""

import json
import yaml
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from loguru import logger

from pdf2epub.ocr import inject_illustrations_into_text
from pdf2epub.utils.logging_config import configure_logging
from pdf2epub.utils.common import load_config, load_book_structure

# Configure logger
logger = configure_logging()


def load_or_create_progress(progress_file: Path, structure: Dict) -> Dict:
    """Load existing progress or create new progress tracking."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
            return progress
    
    # Create new progress tracking
    return {
        "total_chapters": len(structure.get("chapters", [])),
        "chapters_processed": [],
        "pages_processed": {},  # Track individual pages
        "processed_sections": [],  # For front/back matter
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def save_progress(progress_file: Path, progress: Dict):
    """Save progress to file."""
    progress["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def get_backend_module(backend: str):
    """Import the appropriate backend module based on configuration."""
    from pdf2epub.ocr.backends import get_backend
    return get_backend(backend)


def ocr_page(client, process_page_func, img_bytes: bytes, page_num: int, config: Dict, base_output_dir: Path) -> str:
    """Process a single page using the selected backend."""
    try:
        # Pass the base output directory (book dir) to the backend
        # The backend will create the 'images' subdirectory
        result = process_page_func(client, img_bytes, page_num, config, base_output_dir)
        
        text = result['text']
        illustrations = result.get('illustrations', [])
        
        # Inject illustrations into the text at appropriate positions
        if illustrations:
            text = inject_illustrations_into_text(text, illustrations)
        
        return text
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        raise


def process_chapter(client, process_page_func, chapter, book_title, pdf_path, config, progress, progress_file, backend):
    """Process a single chapter with all its pages."""
    chapter_name = chapter["title"]
    start_page = chapter["start_page"]
    end_page = chapter["end_page"]
    chapter_idx = chapter.get("index", 0)
    
    # Check if chapter already processed
    if chapter_idx in progress.get("chapters_processed", []):
        logger.info(f"Skipping chapter {chapter_name} (already processed)")
        return
    
    logger.info(f"Processing chapter: {chapter_name} (pages {start_page}-{end_page})")
    
    # Create output directories based on backend
    output_dir = Path("output") / book_title / "ocr_markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    # Track which pages we've processed for this chapter
    chapter_key = f"chapter_{chapter_idx}"
    if chapter_key not in progress["pages_processed"]:
        progress["pages_processed"][chapter_key] = []
    
    # Process each page
    markdown_content = []
    zoom_factor = config.get('vision_ocr_settings', {}).get('zoom_factor', 1.0) if backend == 'vision' else 1.0
    
    with fitz.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page + 1):
            # Check if page already processed
            if page_num in progress["pages_processed"][chapter_key]:
                # Load existing page content
                page_file = pages_dir / f"page_{page_num}.txt"
                if page_file.exists():
                    with open(page_file, "r", encoding="utf-8") as f:
                        page_text = f.read()
                    if page_text.strip():
                        if markdown_content:
                            markdown_content.append("\n\n---\n\n")
                        markdown_content.append(page_text)
                    logger.debug(f"Loaded existing page {page_num}")
                    continue
            
            try:
                logger.info(f"Processing page {page_num}/{end_page}")
                
                # Extract page as image
                page = pdf[page_num - 1]  # 0-indexed
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                # OCR the page
                # Pass the book output directory for images
                page_text = ocr_page(client, process_page_func, img_bytes, page_num, config, Path("output") / book_title)
                
                # Save individual page
                page_file = pages_dir / f"page_{page_num}.txt"
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_text)
                
                # Add to markdown (without page headers)
                if page_text.strip():  # Only add non-empty pages
                    if markdown_content:  # Add page separator
                        markdown_content.append("\n\n---\n\n")
                    markdown_content.append(page_text)
                
                # Update progress
                progress["pages_processed"][chapter_key].append(page_num)
                save_progress(progress_file, progress)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                # Don't add error text to the output, just log it
    
    # Save chapter markdown with chapter heading
    output_file = output_dir / f"chapter_{chapter_idx}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        # Add chapter heading at the top
        f.write(f"# {chapter_name}\n\n")
        f.write("\n".join(markdown_content))
    
    # Mark chapter as completed
    progress["chapters_processed"].append(chapter_idx)
    save_progress(progress_file, progress)
    
    logger.success(f"Saved chapter to {output_file}")


def process_matter_pages(client, process_page_func, start_page, end_page, matter_type, book_title, pdf_path, config, progress, progress_file, backend):
    """Process front or back matter pages."""
    section_key = f"{matter_type}_matter"
    
    # Check if already processed
    if section_key in progress.get("processed_sections", []):
        logger.info(f"Skipping {matter_type} matter (already processed)")
        return
    
    logger.info(f"Processing {matter_type} matter (pages {start_page}-{end_page})")
    
    # Create output directories based on backend
    output_dir = Path("output") / book_title / "ocr_markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    # Track which pages we've processed
    if section_key not in progress["pages_processed"]:
        progress["pages_processed"][section_key] = []
    
    # Process each page
    markdown_content = []
    zoom_factor = config.get('vision_ocr_settings', {}).get('zoom_factor', 1.0) if backend == 'vision' else 1.0
    
    with fitz.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page + 1):
            # Check if page already processed
            if page_num in progress["pages_processed"][section_key]:
                # Load existing page content
                page_file = pages_dir / f"page_{page_num}.txt"
                if page_file.exists():
                    with open(page_file, "r", encoding="utf-8") as f:
                        page_text = f.read()
                    if page_text.strip():
                        if markdown_content:
                            markdown_content.append("\n\n---\n\n")
                        markdown_content.append(page_text)
                    logger.debug(f"Loaded existing page {page_num}")
                    continue
            
            try:
                logger.info(f"Processing {matter_type} matter page {page_num}/{end_page}")
                
                # Extract page as image
                page = pdf[page_num - 1]  # 0-indexed
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                # OCR the page
                # Pass the book output directory for images
                page_text = ocr_page(client, process_page_func, img_bytes, page_num, config, Path("output") / book_title)
                
                # Save individual page
                page_file = pages_dir / f"page_{page_num}.txt"
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_text)
                
                # Add to markdown (without page headers)
                if page_text.strip():  # Only add non-empty pages
                    if markdown_content:  # Add page separator
                        markdown_content.append("\n\n---\n\n")
                    markdown_content.append(page_text)
                
                # Update progress
                progress["pages_processed"][section_key].append(page_num)
                save_progress(progress_file, progress)
                
            except Exception as e:
                logger.error(f"Failed to process {matter_type} matter page {page_num}: {e}")
    
    # Save matter markdown
    output_file = output_dir / f"{matter_type}_matter.md"
    with open(output_file, "w", encoding="utf-8") as f:
        # Add a header for the matter type
        f.write(f"# {matter_type.capitalize()} Matter\n\n")
        f.write("\n".join(markdown_content))
    
    # Mark section as completed
    progress["processed_sections"].append(section_key)
    save_progress(progress_file, progress)
    
    logger.success(f"Saved {matter_type} matter to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="OCR book chapters using Azure or Vision API")
    parser.add_argument("--backend", choices=['azure', 'vision'], help="Override backend from config")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers (default: from config or 4)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    book_title = config.get("title", "book")
    
    # Determine backend
    backend = args.backend or config.get('jp_ocr_backend', 'azure')
    logger.info(f"Using OCR backend: {backend}")
    
    # Import backend functions
    init_client, process_page_func = get_backend_module(backend)
    
    # Load book structure
    structure = load_book_structure(book_title)
    chapters = structure["chapters"]
    
    # Add chapter indices if not present (1-based)
    for idx, chapter in enumerate(chapters):
        chapter["index"] = idx + 1
    
    # Get PDF path
    pdf_path = Path("output") / book_title / "input_original.pdf"
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Setup progress tracking
    output_dir = Path("output") / book_title / "ocr_markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "ocr_progress.json"
    progress = load_or_create_progress(progress_file, structure)
    
    # Initialize OCR client
    try:
        client = init_client(config)
    except Exception as e:
        logger.error(f"Failed to initialize {backend} client: {e}")
        return
    
    # Process sections with threading
    # Use max_workers from args if provided, otherwise from config
    if args.max_workers is not None:
        max_workers = min(args.max_workers, config.get('vision_ocr_settings', {}).get('max_workers', 4))
    else:
        max_workers = config.get('max_concurrent_workers', config.get('vision_ocr_settings', {}).get('max_workers', 4))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Add front matter if it exists
        if "front_matter" in structure:
            if not args.resume or "front_matter" not in progress.get("processed_sections", []):
                front_matter = structure["front_matter"]
                future = executor.submit(
                    process_matter_pages,
                    client,
                    process_page_func,
                    front_matter["start_page"],
                    front_matter["end_page"],
                    "front",
                    book_title,
                    pdf_path,
                    config,
                    progress,
                    progress_file,
                    backend
                )
                futures.append(("Front Matter", future))
        
        # Add chapters
        for chapter in chapters:
            if not args.resume or chapter["index"] not in progress.get("chapters_processed", []):
                future = executor.submit(
                    process_chapter,
                    client,
                    process_page_func,
                    chapter,
                    book_title,
                    pdf_path,
                    config,
                    progress,
                    progress_file,
                    backend
                )
                futures.append((chapter["title"], future))
        
        # Add back matter if it exists
        if "back_matter" in structure:
            if not args.resume or "back_matter" not in progress.get("processed_sections", []):
                back_matter = structure["back_matter"]
                future = executor.submit(
                    process_matter_pages,
                    client,
                    process_page_func,
                    back_matter["start_page"],
                    back_matter["end_page"],
                    "back",
                    book_title,
                    pdf_path,
                    config,
                    progress,
                    progress_file,
                    backend
                )
                futures.append(("Back Matter", future))
        
        # Wait for completion
        for section_name, future in futures:
            try:
                future.result()
                logger.success(f"Completed: {section_name}")
            except Exception as e:
                logger.error(f"Failed to process {section_name}: {e}")
    
    # Log final progress
    logger.info("\n" + "=" * 60)
    logger.info(f"OCR Processing Summary ({backend} backend):")
    logger.info(f"Chapters: {len(progress.get('chapters_processed', []))}/{progress['total_chapters']} processed")
    
    if "front_matter" in structure:
        status = "✓ Processed" if "front_matter" in progress.get("processed_sections", []) else "✗ Not processed"
        logger.info(f"Front matter: {status}")
    
    if "back_matter" in structure:
        status = "✓ Processed" if "back_matter" in progress.get("processed_sections", []) else "✗ Not processed"
        logger.info(f"Back matter: {status}")
    
    total_pages = sum(len(pages) for pages in progress.get("pages_processed", {}).values())
    logger.info(f"Total pages processed: {total_pages}")
    
    logger.success("All sections processed!")


if __name__ == "__main__":
    main()
