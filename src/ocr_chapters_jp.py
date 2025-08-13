#!/usr/bin/env python3
import json
import fitz  # PyMuPDF
import yaml
import numpy as np
import argparse
import re
import base64
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from google.cloud import vision_v1p4beta1 as vision
from google.oauth2 import service_account
from PIL import Image, ImageDraw
import io
from loguru import logger
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception,
    before_sleep_log
)
import logging
from utils.logging_config import configure_logging
from utils.ocr_client import OCRClient

# Configure logger
logger = configure_logging()

# Global lock for thread-safe operations
progress_lock = Lock()
file_write_lock = Lock()

# Configuration defaults (can be overridden from config.yaml)
DEFAULT_OCR_SETTINGS = {
    'zoom_factor': 1.0,
    'max_workers': 4,
    'page_retry_backoff': 2,
    'illustration_padding': 25,
    'trim_step_percent': 0.02,
    'min_black_pixels': 200
}


def is_transient_vision_error(exception: Exception) -> bool:
    """Check if a Cloud Vision API error is transient and should be retried."""
    error_str = str(exception).lower()
    
    # Check for specific error codes that indicate transient issues
    transient_keywords = [
        '500', '503', '429',
        'internal', 'unavailable', 
        'resource_exhausted', 'deadline_exceeded',
        'timeout', 'temporarily'
    ]
    
    return any(keyword in error_str for keyword in transient_keywords)


def is_transient_page_error(exception: Exception) -> bool:
    """Check if a page processing error is transient and should be retried.
    
    Args:
        exception: The exception to check
    
    Returns:
        True if the error is transient and should be retried
    """
    error_str = str(exception).lower()
    
    # Don't retry on permanent errors
    permanent_keywords = [
        'page not found',
        'invalid pdf',
        'corrupted',
        'value error',
        'index out of range'
    ]
    
    if any(keyword in error_str for keyword in permanent_keywords):
        return False
    
    # Retry on transient errors
    transient_keywords = [
        '500', '502', '503', '504', '429',
        'timeout', 'timed out',
        'connection', 'network',
        'rate limit', 'quota',
        'resource_exhausted', 'deadline',  # Covers both 'deadline_exceeded' and 'deadline exceeded'
        'internal', 'unavailable',
        'temporarily', 'transient'
    ]
    
    return any(keyword in error_str for keyword in transient_keywords)


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # Merge OCR settings with defaults
    ocr_settings = config.get('ocr_settings', {})
    config['ocr_settings'] = {**DEFAULT_OCR_SETTINGS, **ocr_settings}
    
    return config


def load_book_structure(book_title):
    """Load the book structure JSON file."""
    structure_path = Path("output") / Path(book_title) / "book_structure.json"
    with open(structure_path, "r", encoding="utf-8") as file:
        structure = json.load(file)
    return structure


def extract_page_as_image(pdf_path: Path, page_num: int, zoom: float = None, config: Dict = None) -> tuple:
    """Extract a single page from PDF as an image.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        zoom: Zoom factor for image quality (overrides config)
        config: Configuration dictionary
    
    Returns: (img_bytes, width, height)
    """
    
    # Use zoom from parameters, config, or default
    if zoom is None:
        zoom = config.get('ocr_settings', {}).get('zoom_factor', 1.0) if config else 1.0
    with fitz.open(pdf_path) as pdf:
        if page_num > len(pdf):
            raise ValueError(f"Page {page_num} not found in PDF (total pages: {len(pdf)})")
        
        # Get the page (0-indexed internally)
        page = pdf[page_num - 1]
        
        # Render page as image
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # Get dimensions
        width = pix.width
        height = pix.height
        
    return img_bytes, width, height


def ocr_page_with_llm(img_bytes: bytes, ocr_client: OCRClient, page_num: int = None, config: Dict = None) -> str:
    """OCR a page using LLM with multi-model fallback.
    
    Args:
        img_bytes: Image bytes to OCR
        ocr_client: OCR client instance
        page_num: Page number for logging
        config: Configuration dictionary
    
    Returns:
        OCR text result
    """
    
    # Get model configs from config
    model_configs = config.get('ocr_models') if config else None
    
    # Use OCRClient to handle multi-model fallback
    return ocr_client.ocr_page(
        img_bytes=img_bytes,
        page_num=page_num,
        model_configs=model_configs
    )


@retry(
    retry=retry_if_exception(is_transient_vision_error),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True
)
def get_text_blocks_from_vision(img_bytes: bytes, client, page_num: Optional[int] = None) -> List[Dict]:
    """Get all text blocks from Cloud Vision API with automatic retry.
    
    Args:
        img_bytes: Image bytes to analyze
        client: Cloud Vision client
        page_num: Page number for logging
    
    Returns:
        List of text blocks
    """
    page_info = f"page {page_num}" if page_num else "image"
    logger.debug(f"Calling Cloud Vision API for {page_info}")
    
    image = vision.Image(content=img_bytes)
    
    response = client.annotate_image({
        "image": image,
        "features": [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
        "image_context": vision.ImageContext(language_hints=["ja"])
    })
    
    text_blocks = []
    
    if response.full_text_annotation:
        ann = response.full_text_annotation
        
        for page_idx, page in enumerate(ann.pages):
            for block_idx, block in enumerate(page.blocks):
                vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                text_blocks.append({
                    'block_idx': block_idx,
                    'vertices': vertices,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'type': block.block_type
                })
    
    logger.debug(f"Found {len(text_blocks)} text blocks in {page_info}")
    return text_blocks


def white_out_text_regions(img_bytes: bytes, text_blocks: list, padding: int = None, config: Dict = None) -> Image.Image:
    """Create image with text regions whited out."""
    
    # Use padding from parameters, config, or default
    if padding is None:
        padding = config.get('ocr_settings', {}).get('illustration_padding', 25) if config else 25
    
    img = Image.open(io.BytesIO(img_bytes))
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    for block in text_blocks:
        if block['type'] == 1:  # TEXT
            x_min, y_min, x_max, y_max = block['bbox']
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width - 1, x_max + padding)
            y_max = min(height - 1, y_max + padding)
            
            draw.rectangle([(x_min, y_min), (x_max, y_max)], fill='white')
    
    return img


def trim_white_borders(img: Image.Image, step_percent: float = None, min_black_pixels: int = None, config: Dict = None) -> tuple:
    """Progressively trim white borders until we find the content area."""
    
    # Use parameters from config or defaults
    if step_percent is None:
        step_percent = config.get('ocr_settings', {}).get('trim_step_percent', 0.02) if config else 0.02
    if min_black_pixels is None:
        min_black_pixels = config.get('ocr_settings', {}).get('min_black_pixels', 200) if config else 200
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    if len(img_array.shape) == 3:
        img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        img_gray = img_array
    
    # Initialize borders
    top = 0
    bottom = height - 1
    left = 0
    right = width - 1
    
    # Calculate step sizes
    h_step = max(1, int(height * step_percent))
    w_step = max(1, int(width * step_percent))
    
    # Trim from top
    while top < bottom:
        strip = img_gray[top:top+h_step, left:right+1]
        non_white_count = np.sum(strip < 240)
        if non_white_count > min_black_pixels:
            break
        top += h_step
    
    # Trim from bottom
    while bottom > top:
        strip = img_gray[max(bottom-h_step, 0):bottom+1, left:right+1]
        non_white_count = np.sum(strip < 240)
        if non_white_count > min_black_pixels:
            break
        bottom -= h_step
    
    # Trim from left
    while left < right:
        strip = img_gray[top:bottom+1, left:left+w_step]
        non_white_count = np.sum(strip < 240)
        if non_white_count > min_black_pixels:
            break
        left += w_step
    
    # Trim from right
    while right > left:
        strip = img_gray[top:bottom+1, max(right-w_step, 0):right+1]
        non_white_count = np.sum(strip < 240)
        if non_white_count > min_black_pixels:
            break
        right -= w_step
    
    # Add small margin
    margin = 5
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width - 1, right + margin)
    bottom = min(height - 1, bottom + margin)
    
    return (left, top, right, bottom)


def extract_illustration(img_bytes: bytes, bbox: tuple) -> Image.Image:
    """Extract illustration based on bounding box."""
    
    img = Image.open(io.BytesIO(img_bytes))
    x_min, y_min, x_max, y_max = bbox
    cropped = img.crop((x_min, y_min, x_max, y_max))
    
    return cropped


def _process_single_page_impl(
    page_num: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    images_dir: Path,
    chapter_index: int,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    section_key: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[int, str, bool, str]:
    """Process a single page with integrated OCR and illustration extraction.
    
    Args:
        page_num: Page number to process
        pdf_path: Path to PDF file
        ocr_client: OCR client for multi-model fallback
        vision_client: Cloud Vision API client
        images_dir: Directory for saving images
        chapter_index: Chapter number for image naming
        progress: Progress dictionary for tracking
        progress_file: Path to progress file
        section_key: Section key for progress tracking
        config: Configuration dictionary
    
    Returns: (page_num, markdown, has_illustration, illustration_path)
    """
    
    try:
        # Extract page as image
        img_bytes, width, height = extract_page_as_image(pdf_path, page_num, config=config)
        
        # OCR with multi-model fallback
        markdown_text = ocr_page_with_llm(img_bytes, ocr_client, page_num=page_num, config=config)
        
        # Check for [illustration] markers
        illustration_pattern = r'\[illustration\]'
        illustration_matches = list(re.finditer(illustration_pattern, markdown_text, re.IGNORECASE))
        
        if illustration_matches:
            # Extract illustration using Cloud Vision pipeline
            text_blocks = get_text_blocks_from_vision(img_bytes, vision_client, page_num=page_num)
            
            # White out text regions
            img_whited = white_out_text_regions(img_bytes, text_blocks, config=config)
            
            # Find illustration boundaries
            illustration_bbox = trim_white_borders(img_whited, config=config)
            
            # Check if valid bounding box was found
            left, top, right, bottom = illustration_bbox
            if right <= left or bottom <= top:
                logger.warning(f"No valid illustration found on page {page_num} despite [illustration] marker")
                # Remove the [illustration] marker from the text
                markdown_text = markdown_text.replace('[illustration]', '')
                return (page_num, markdown_text, False, None)
            
            # Extract illustration from ORIGINAL image
            illustration = extract_illustration(img_bytes, illustration_bbox)
            
            # Use page number for unique naming across threads
            img_filename = f"chapter_{chapter_index}_page_{page_num}.png"
            img_path = images_dir / img_filename
            
            with file_write_lock:
                illustration.save(img_path)
            
            # Update markdown - replace first [illustration], remove others
            if len(illustration_matches) > 1:
                # Remove all [illustration] markers except the first
                parts = markdown_text.split('[illustration]')
                updated_markdown = parts[0] + f'![Image](../images/{img_filename})'
                for part in parts[1:]:
                    updated_markdown += part
            else:
                # Replace single [illustration]
                updated_markdown = markdown_text.replace(
                    '[illustration]', 
                    f'![Image](../images/{img_filename})',
                    1
                )
            
            # Update progress after successful processing
            if progress and progress_file and section_key:
                update_page_progress(progress, progress_file, section_key, page_num)
            
            return (page_num, updated_markdown, True, str(img_path))
        else:
            # Update progress after successful processing
            if progress and progress_file and section_key:
                update_page_progress(progress, progress_file, section_key, page_num)
            
            return (page_num, markdown_text, False, None)
            
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        raise


def process_single_page(
    page_num: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    images_dir: Path,
    chapter_index: int,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    section_key: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[int, str, bool, str]:
    """Process a single page with retry logic.
    
    This is a wrapper around _process_single_page_impl that adds retry logic
    for transient errors.
    """
    
    # Get retry configuration from ocr_models or use defaults
    # We use the max retry attempts from all configured models
    retry_attempts = 3  # Default
    if config and 'ocr_models' in config:
        retry_attempts = max(m.get('max_retries', 1) for m in config['ocr_models'])
    
    retry_backoff = config.get('ocr_settings', {}).get('page_retry_backoff', 2) if config else 2
    
    # Create retry decorator dynamically based on config
    retry_decorator = retry(
        retry=retry_if_exception(is_transient_page_error),
        wait=wait_random_exponential(multiplier=retry_backoff, max=60),
        stop=stop_after_attempt(retry_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    
    # Apply retry decorator and call the implementation
    retryable_process = retry_decorator(_process_single_page_impl)
    
    return retryable_process(
        page_num=page_num,
        pdf_path=pdf_path,
        ocr_client=ocr_client,
        vision_client=vision_client,
        images_dir=images_dir,
        chapter_index=chapter_index,
        progress=progress,
        progress_file=progress_file,
        section_key=section_key,
        config=config
    )


def process_page_range(
    start_page: int,
    end_page: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    images_dir: Path,
    chapter_index: int,
    max_workers: int = None,
    fail_on_error: bool = False,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    section_key: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Process a range of pages using multithreading.
    
    Args:
        start_page: First page number
        end_page: Last page number
        pdf_path: Path to PDF file
        ocr_client: OCR client for multi-model fallback
        vision_client: Cloud Vision API client
        images_dir: Directory for saving images
        chapter_index: Chapter number
        max_workers: Number of parallel workers (overrides config)
        fail_on_error: If True, raise on first error; if False, collect errors
        progress: Progress dictionary
        progress_file: Path to progress file
        section_key: Section key for progress tracking
        config: Configuration dictionary
    
    Returns: 
        Tuple of (page_markdown_dict, error_dict)
        - page_markdown_dict: Successfully processed pages
        - error_dict: Failed pages with error messages
    """
    
    # Get max_workers from parameters, config, or default
    if max_workers is None:
        max_workers = config.get('ocr_settings', {}).get('max_workers', 4) if config else 4
    
    page_markdowns = {}
    page_errors = {}
    
    # Create tasks for all pages
    pages_to_process = list(range(start_page, end_page + 1))
    
    # Skip already processed pages if resume is enabled
    if progress and section_key and "pages_processed" in progress:
        processed_pages = progress.get("pages_processed", {}).get(section_key, [])
        pages_to_process = [p for p in pages_to_process if p not in processed_pages]
        if len(processed_pages) > 0:
            logger.info(f"Resuming: skipping {len(processed_pages)} already processed pages")
    
    total_pages = len(pages_to_process)
    
    if total_pages == 0:
        logger.info(f"All pages already processed for {section_key}")
        return {}, {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {}
        for page_num in pages_to_process:
            future = executor.submit(
                process_single_page,
                page_num,
                pdf_path,
                ocr_client,
                vision_client,
                images_dir,
                chapter_index,
                progress,
                progress_file,
                section_key,
                config
            )
            future_to_page[future] = page_num
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            completed += 1
            
            try:
                result_page, markdown, has_illustration, img_path = future.result()
                
                # Store markdown
                page_markdowns[result_page] = markdown
                
                logger.info(f"[{completed}/{total_pages}] Processed page {result_page}" + 
                          (f" (with illustration)" if has_illustration else ""))
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{completed}/{total_pages}] Failed to process page {page_num}: {error_msg}")
                page_errors[page_num] = error_msg
                
                if fail_on_error:
                    # Cancel remaining tasks and raise
                    for f in future_to_page:
                        f.cancel()
                    raise
    
    # Log summary
    if page_errors:
        logger.warning(f"Completed with {len(page_errors)} errors out of {total_pages} pages")
    else:
        logger.success(f"Successfully processed all {total_pages} pages")
    
    return page_markdowns, page_errors


def process_content_section(
    title: str,
    start_page: int,
    end_page: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    output_dir: Path,
    images_dir: Path,
    output_file: Path,
    chapter_index: int = 0,
    max_workers: int = None,
    fail_on_error: bool = False,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    section_key: Optional[str] = None,
    add_header: bool = True,
    config: Optional[Dict] = None
) -> bool:
    """Process any content section (chapter, matter, etc.) with common logic.
    
    Args:
        title: Section title
        start_page: First page number
        end_page: Last page number
        pdf_path: Path to PDF file
        ocr_client: OCR client for multi-model fallback
        vision_client: Cloud Vision API client
        output_dir: Output directory for markdown
        images_dir: Output directory for images
        output_file: Path to output markdown file
        chapter_index: Chapter number for image naming
        max_workers: Number of worker threads
        fail_on_error: If True, raise on first error
        progress: Progress dictionary
        progress_file: Path to progress file
        section_key: Key for progress tracking
        add_header: Whether to add a header to the markdown
    
    Returns:
        True if fully successful, False if there were errors
    """
    
    logger.info(f"Processing {title} (pages {start_page}-{end_page})")
    
    # Check if file already exists and all pages are processed (for resume)
    if output_file.exists() and progress and section_key:
        pages_processed = progress.get("pages_processed", {}).get(section_key, [])
        total_pages = end_page - start_page + 1
        if len(pages_processed) == total_pages:
            logger.info(f"Skipping {title} - already fully processed with {total_pages} pages")
            return True
    
    # Process all pages
    page_markdowns, page_errors = process_page_range(
        start_page,
        end_page,
        pdf_path,
        ocr_client,
        vision_client,
        images_dir,
        chapter_index,
        max_workers,
        fail_on_error,
        progress,
        progress_file,
        section_key,
        config
    )
    
    # Combine successful pages in order
    markdown_parts = []
    for page_num in sorted(page_markdowns.keys()):
        markdown_parts.append(page_markdowns[page_num])
    
    # Add placeholders for failed pages
    if page_errors and not fail_on_error:
        all_pages = set(range(start_page, end_page + 1))
        for page_num in sorted(all_pages):
            if page_num in page_errors:
                markdown_parts.insert(
                    page_num - start_page,
                    f"\n\n[ERROR: Failed to process page {page_num}]\n\n"
                )
    
    full_markdown = "\n\n---\n\n".join(markdown_parts) if markdown_parts else ""
    
    # Add header if requested
    if add_header:
        final_markdown = f"# {title}\n\n{full_markdown}"
    else:
        final_markdown = full_markdown
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_markdown)
    
    if page_errors:
        logger.warning(f"Saved {title} to {output_file} with {len(page_errors)} errors")
        return False
    else:
        logger.success(f"Saved {title} to {output_file}")
        return True


def process_chapter(
    chapter: Dict,
    chapter_index: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    output_dir: Path,
    images_dir: Path,
    max_workers: int = None,
    fail_on_error: bool = False,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    config: Optional[Dict] = None
) -> bool:
    """Process a single chapter.
    
    Returns:
        True if fully successful, False if there were errors
    """
    
    chapter_title = chapter["title"]
    start_page = chapter["start_page"]
    end_page = chapter["end_page"]
    output_file = output_dir / f"chapter_{chapter_index}.md"
    section_key = f"chapter_{chapter_index}"
    
    return process_content_section(
        title=f"Chapter {chapter_index}: {chapter_title}",
        start_page=start_page,
        end_page=end_page,
        pdf_path=pdf_path,
        ocr_client=ocr_client,
        vision_client=vision_client,
        output_dir=output_dir,
        images_dir=images_dir,
        output_file=output_file,
        chapter_index=chapter_index,
        max_workers=max_workers,
        fail_on_error=fail_on_error,
        progress=progress,
        progress_file=progress_file,
        section_key=section_key,
        add_header=True,
        config=config
    )


def process_subchapters(
    chapter: Dict,
    chapter_index: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    output_dir: Path,
    images_dir: Path,
    max_workers: int = None,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    config: Optional[Dict] = None
) -> bool:
    """Process subchapters within a chapter.
    
    Returns:
        True if fully successful, False if there were errors
    """
    
    subchapters = chapter.get("subchapters", [])
    
    if not subchapters:
        # No subchapters, process as a single chapter
        return process_chapter(
            chapter, chapter_index, pdf_path, ocr_client, vision_client, 
            output_dir, images_dir, max_workers, False, progress, progress_file, config
        )
    
    # Process main chapter header if it has its own content
    chapter_title = chapter["title"]
    chapter_start = chapter["start_page"]
    
    # Check if there's content before the first subchapter
    first_subchapter_start = subchapters[0]["start_page"] if subchapters else chapter["end_page"] + 1
    
    markdown_parts = [f"# {chapter_title}\n"]
    has_errors = False
    
    if first_subchapter_start > chapter_start:
        # There's content before the first subchapter
        logger.info(f"Processing chapter {chapter_index} intro (pages {chapter_start}-{first_subchapter_start - 1})")
        
        page_markdowns, page_errors = process_page_range(
            chapter_start,
            first_subchapter_start - 1,
            pdf_path,
            ocr_client,
            vision_client,
            images_dir,
            chapter_index,
            max_workers,
            False,  # fail_on_error
            progress,
            progress_file,
            f"chapter_{chapter_index}_intro",
            config
        )
        
        if page_errors:
            has_errors = True
        
        # Add intro pages in order
        for page_num in sorted(page_markdowns.keys()):
            markdown_parts.append(page_markdowns[page_num])
    
    # Process each subchapter
    for sub_idx, subchapter in enumerate(subchapters, 1):
        sub_title = subchapter["title"]
        sub_start = subchapter["start_page"]
        sub_end = subchapter["end_page"]
        
        logger.info(f"Processing Subchapter {chapter_index}.{sub_idx}: {sub_title} (pages {sub_start}-{sub_end})")
        
        # Add subchapter header
        markdown_parts.append(f"\n## {sub_title}\n")
        
        # Process subchapter pages
        page_markdowns, page_errors = process_page_range(
            sub_start,
            sub_end,
            pdf_path,
            ocr_client,
            vision_client,
            images_dir,
            chapter_index,
            max_workers,
            False,  # fail_on_error
            progress,
            progress_file,
            f"chapter_{chapter_index}_sub_{sub_idx}",
            config
        )
        
        if page_errors:
            has_errors = True
        
        # Add subchapter pages in order
        for page_num in sorted(page_markdowns.keys()):
            markdown_parts.append(page_markdowns[page_num])
    
    # Combine all parts and save
    full_markdown = "\n\n".join(markdown_parts)
    output_file = output_dir / f"chapter_{chapter_index}.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    
    if has_errors:
        logger.warning(f"Saved Chapter {chapter_index} with {len(subchapters)} subchapters to {output_file} (with errors)")
        return False
    else:
        logger.success(f"Saved Chapter {chapter_index} with {len(subchapters)} subchapters to {output_file}")
        return True


def load_or_create_progress(progress_file: Path, structure: Dict) -> Dict:
    """Load existing progress or create new progress tracking."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    
    # Create new progress
    progress = {
        "chapters_processed": [],
        "total_chapters": len(structure["chapters"]),
        "pages_processed": {}  # Track individual pages: {"chapter_1": [1,2,3], ...}
    }
    return progress


def save_progress(progress_file: Path, progress: Dict):
    """Save progress to file.
    
    Note: This function should be called with progress_lock already held
    or without any lock if called from single-threaded context.
    """
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def update_page_progress(progress: Dict, progress_file: Path, section_key: str, page_num: int):
    """Update progress for a single processed page.
    
    Args:
        progress: Progress dictionary
        progress_file: Path to progress file
        section_key: Key for the section (e.g., "chapter_1", "front_matter")
        page_num: Page number that was processed
    """
    with progress_lock:
        if "pages_processed" not in progress:
            progress["pages_processed"] = {}
        
        if section_key not in progress["pages_processed"]:
            progress["pages_processed"][section_key] = []
        
        if page_num not in progress["pages_processed"][section_key]:
            progress["pages_processed"][section_key].append(page_num)
            save_progress(progress_file, progress)


def process_matter_pages(
    start_page: int,
    end_page: int,
    pdf_path: Path,
    ocr_client: OCRClient,
    vision_client,
    output_dir: Path,
    images_dir: Path,
    matter_type: str,
    max_workers: int = None,
    progress: Optional[Dict] = None,
    progress_file: Optional[Path] = None,
    config: Optional[Dict] = None
) -> bool:
    """Process front matter or back matter pages.
    
    Args:
        start_page: First page number
        end_page: Last page number
        pdf_path: Path to PDF file
        ocr_client: OCR client for multi-model fallback
        vision_client: Cloud Vision API client
        output_dir: Output directory for markdown
        images_dir: Output directory for images
        matter_type: "front" or "back"
        max_workers: Number of worker threads
        progress: Progress dictionary
        progress_file: Path to progress file
    
    Returns:
        True if fully successful, False if there were errors
    """
    
    title = f"{matter_type.capitalize()} Matter"
    output_file = output_dir / f"{matter_type}_matter.md"
    section_key = f"{matter_type}_matter"
    
    return process_content_section(
        title=title,
        start_page=start_page,
        end_page=end_page,
        pdf_path=pdf_path,
        ocr_client=ocr_client,
        vision_client=vision_client,
        output_dir=output_dir,
        images_dir=images_dir,
        output_file=output_file,
        chapter_index=0,  # Use 0 for matter pages
        max_workers=max_workers,
        fail_on_error=False,
        progress=progress,
        progress_file=progress_file,
        section_key=section_key,
        add_header=True,
        config=config
    )


def main():
    parser = argparse.ArgumentParser(description="OCR Japanese PDF chapters using Gemini + Cloud Vision")
    parser.add_argument("-i", "--input", help="Path to input PDF file (default: output/{book_title}/input_original.pdf)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of worker threads (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    
    if not book_title:
        logger.error("No title found in config.yaml")
        return
    
    api_key = config.get("google_api_key")
    if not api_key:
        logger.error("No Google API key found in config.yaml")
        return
    
    # Load book structure
    structure = load_book_structure(book_title)
    
    # Setup output directories (same as regular ocr_chapters)
    output_dir = Path("output") / book_title / "ocr_markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path("output") / book_title / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup progress tracking
    progress_file = output_dir / "ocr_progress.json"
    progress = load_or_create_progress(progress_file, structure)
    
    # Setup OCR client with multi-model fallback
    logger.info("Setting up OCR client with multi-model fallback...")
    ocr_client = OCRClient(config)
    
    # Setup Cloud Vision client
    sa_key_path = config.get("service_account_key_path", "sa-keys.json")
    if not Path(sa_key_path).exists():
        logger.error(f"Service account key file not found: {sa_key_path}")
        return
    
    credentials = service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    logger.info("Setting up Cloud Vision client...")
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    
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
    
    logger.info(f"Using PDF: {pdf_path}")
    
    # Get max_workers from args or config
    max_workers = args.max_workers if args.max_workers else config.get('ocr_settings', {}).get('max_workers', 4)
    logger.info(f"Using {max_workers} worker threads")
    
    # Process front matter if it exists
    if "front_matter" in structure:
        front_matter = structure["front_matter"]
        # Skip if already processed when resume is enabled
        if args.resume and "front_matter" in progress.get("processed_sections", []):
            logger.info("Skipping front matter (already processed)")
        else:
            try:
                success = process_matter_pages(
                    front_matter["start_page"],
                    front_matter["end_page"],
                    pdf_path,
                    ocr_client,
                    vision_client,
                    output_dir,
                    images_dir,
                    "front",
                    max_workers,
                    progress,
                    progress_file,
                    config
                )
                
                # Update progress
                with progress_lock:
                    if "processed_sections" not in progress:
                        progress["processed_sections"] = []
                    if "front_matter" not in progress["processed_sections"]:
                        progress["processed_sections"].append("front_matter")
                    save_progress(progress_file, progress)
                    
            except Exception as e:
                logger.error(f"Failed to process front matter: {e}")
                raise
    
    # Process each chapter
    for chapter_idx, chapter in enumerate(structure["chapters"], 1):
        # Check if already processed
        if args.resume and chapter_idx in progress["chapters_processed"]:
            # Check if all pages were processed for page-level resume
            section_key = f"chapter_{chapter_idx}"
            if "subchapters" in chapter:
                # Check all subchapter keys too
                all_processed = True
                for sub_idx in range(1, len(chapter["subchapters"]) + 1):
                    sub_key = f"chapter_{chapter_idx}_sub_{sub_idx}"
                    if sub_key not in progress.get("pages_processed", {}):
                        all_processed = False
                        break
                if all_processed:
                    logger.info(f"Skipping Chapter {chapter_idx} (already processed)")
                    continue
                else:
                    logger.info(f"Resuming Chapter {chapter_idx} (partially processed)")
            elif section_key in progress.get("pages_processed", {}):
                logger.info(f"Skipping Chapter {chapter_idx} (already processed)")
                continue
            else:
                logger.info(f"Resuming Chapter {chapter_idx} (partially processed)")
        
        try:
            # Process chapter with subchapters if they exist
            success = process_subchapters(
                chapter, 
                chapter_idx, 
                pdf_path, 
                ocr_client,
                vision_client,
                output_dir,
                images_dir,
                max_workers,
                progress,
                progress_file,
                config
            )
            
            # Update progress only if successful or we allow partial success
            if chapter_idx not in progress["chapters_processed"]:
                with progress_lock:
                    progress["chapters_processed"].append(chapter_idx)
                    save_progress(progress_file, progress)
            
        except Exception as e:
            logger.error(f"Failed to process Chapter {chapter_idx}: {e}")
            # Fail immediately on error
            raise
    
    # Process back matter if it exists
    if "back_matter" in structure:
        back_matter = structure["back_matter"]
        # Skip if already processed when resume is enabled
        if args.resume and "back_matter" in progress.get("processed_sections", []):
            logger.info("Skipping back matter (already processed)")
        else:
            try:
                success = process_matter_pages(
                    back_matter["start_page"],
                    back_matter["end_page"],
                    pdf_path,
                    ocr_client,
                    vision_client,
                    output_dir,
                    images_dir,
                    "back",
                    max_workers,
                    progress,
                    progress_file,
                    config
                )
                
                # Update progress
                with progress_lock:
                    if "processed_sections" not in progress:
                        progress["processed_sections"] = []
                    if "back_matter" not in progress["processed_sections"]:
                        progress["processed_sections"].append("back_matter")
                    save_progress(progress_file, progress)
                    
            except Exception as e:
                logger.error(f"Failed to process back matter: {e}")
                raise
    
    logger.success(f"OCR processing complete! Markdown files saved to {output_dir}")
    
    # Summary
    logger.info(f"\n=== Processing Summary ===")
    logger.info(f"Chapters: {len(progress['chapters_processed'])}/{progress['total_chapters']} processed")
    
    if "front_matter" in structure:
        status = "✓ Processed" if "front_matter" in progress.get("processed_sections", []) else "✗ Not processed"
        logger.info(f"Front matter: {status}")
    
    if "back_matter" in structure:
        status = "✓ Processed" if "back_matter" in progress.get("processed_sections", []) else "✗ Not processed"
        logger.info(f"Back matter: {status}")
    
    if len(progress["chapters_processed"]) < progress["total_chapters"]:
        missing = set(range(1, progress["total_chapters"] + 1)) - set(progress["chapters_processed"])
        logger.warning(f"Missing chapters: {sorted(missing)}")
    
    # Log safety block statistics
    safety_stats = ocr_client.get_safety_stats()
    if safety_stats:
        logger.info("\n=== Safety Block Statistics ===")
        for provider, blocked_count in safety_stats.items():
            logger.info(f"{provider}: {blocked_count} pages blocked for safety")


if __name__ == "__main__":
    main()
