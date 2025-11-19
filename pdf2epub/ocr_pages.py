"""
Page-wise OCR processing for PDF files.

This module handles OCR at the page level, producing individual markdown files
for each page. The pages can later be aggregated into chapters.

Supports multiple backends:
- mistral: Mistral OCR API
- vertex: Vertex AI Mistral OCR
- vllm: VLLM-based OCR
- azure: Azure Document Intelligence (for Japanese vertical text)
- vision: Google Cloud Vision API (for Japanese vertical text)
"""

import json
import argparse
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from loguru import logger
from .utils.logging_config import configure_logging
from .utils.common import load_config
from .ocr_backends import ocr_pdf_chunk_mistral, ocr_pdf_chunk_vertex, ocr_pdf_chunk_vllm

# Configure logger
logger = configure_logging()

# Cache for Azure/Vision clients (to avoid re-initialization)
_backend_clients = {}


def pdf_to_image(pdf_bytes: bytes, zoom_factor: float = 1.0) -> bytes:
    """Convert single-page PDF to PNG image bytes for vision backends.

    Args:
        pdf_bytes: PDF content as bytes (should be single page)
        zoom_factor: Image quality factor (1.0-3.0, higher = better quality)

    Returns:
        PNG image bytes
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        page = pdf[0]
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")


def ocr_pdf_chunk(
    pdf_bytes: bytes,
    session=None,
    project_id: str = None,
    location: str = None,
    chunk_info: str = "",
    images_dir: Path = None,
    chapter_index: int = 1,
    image_counter: int = 0,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
    backend: str = "vertex",
    api_key: str = None,
    base_url: str = None,
    config: Dict = None
) -> Tuple[str, List, int]:
    """OCR a PDF chunk using selected backend.

    Routes to the appropriate backend based on configuration.
    Supports: vertex, mistral, vllm, azure, vision

    Args:
        pdf_bytes: PDF content as bytes
        session: Authorized session for Vertex AI (required for vertex backend)
        project_id: GCP project ID (required for vertex backend)
        location: GCP location (required for vertex backend)
        chunk_info: Description of the chunk being processed
        images_dir: Directory to save extracted images
        chapter_index: Page number for image naming (use page_num)
        image_counter: Starting counter for image numbering
        max_retries: Maximum number of retry attempts for 429 errors
        initial_backoff: Initial backoff time in seconds
        backend: OCR backend to use ('vertex', 'mistral', 'vllm', 'azure', 'vision')
        api_key: API key (for mistral backend)
        base_url: API base URL (for mistral backend)
        config: Configuration dict (required for azure/vision backends)

    Returns:
        Tuple of (markdown_content, images_info, updated_image_counter)
    """
    global _backend_clients

    # Route to appropriate backend
    if backend == "mistral":
        if not api_key:
            raise ValueError("Mistral API key is required for mistral backend")

        kwargs = {
            "pdf_bytes": pdf_bytes,
            "api_key": api_key,
            "chunk_info": chunk_info,
            "images_dir": images_dir,
            "chapter_index": chapter_index,
            "image_counter": image_counter,
            "max_retries": max_retries,
            "initial_backoff": initial_backoff
        }

        if base_url:
            kwargs["base_url"] = base_url

        return ocr_pdf_chunk_mistral(**kwargs)

    elif backend == "vertex":
        if not session or not project_id or not location:
            raise ValueError("session, project_id, and location are required for vertex backend")

        return ocr_pdf_chunk_vertex(
            pdf_bytes=pdf_bytes,
            session=session,
            project_id=project_id,
            location=location,
            chunk_info=chunk_info,
            images_dir=images_dir,
            chapter_index=chapter_index,
            image_counter=image_counter,
            max_retries=max_retries,
            initial_backoff=initial_backoff
        )

    elif backend == "vllm":
        # Load config for vllm backend if not provided
        if config is None:
            from .utils.common import load_config
            config = load_config()

        return ocr_pdf_chunk_vllm(
            pdf_bytes=pdf_bytes,
            config=config,
            chunk_info=chunk_info,
            images_dir=images_dir,
            chapter_index=chapter_index,
            image_counter=image_counter,
            max_retries=max_retries,
            initial_backoff=initial_backoff
        )

    elif backend == "azure":
        # Azure Document Intelligence backend (for Japanese vertical text)
        if config is None:
            raise ValueError("config is required for azure backend")

        # Convert PDF to image
        zoom_factor = config.get('vision_ocr_settings', {}).get('zoom_factor', 1.0)
        img_bytes = pdf_to_image(pdf_bytes, zoom_factor)

        # Initialize client (cached)
        if 'azure' not in _backend_clients:
            from .ocr.backends import get_backend
            init_client_func, _ = get_backend('azure')
            _backend_clients['azure'] = init_client_func(config)
            logger.info("Initialized Azure Document Intelligence client")

        # Get process_page function
        from .ocr.backends import get_backend
        _, process_page_func = get_backend('azure')

        # Determine base output directory for images
        base_output_dir = images_dir.parent if images_dir else None

        # Process page
        result = process_page_func(
            client=_backend_clients['azure'],
            img_bytes=img_bytes,
            page_num=chapter_index,  # chapter_index is actually page_num
            config=config,
            base_output_dir=base_output_dir
        )

        # Extract results
        markdown = result.get('text', '')
        illustrations = result.get('illustrations', [])

        # Inject illustrations into markdown if present
        if illustrations:
            from .ocr import inject_illustrations_into_text
            markdown = inject_illustrations_into_text(markdown, illustrations)

        # Count illustrations as images
        updated_counter = image_counter + len(illustrations)

        return markdown, illustrations, updated_counter

    elif backend == "vision":
        # Google Cloud Vision API backend (for Japanese vertical text)
        if config is None:
            raise ValueError("config is required for vision backend")

        # Convert PDF to image
        zoom_factor = config.get('vision_ocr_settings', {}).get('zoom_factor', 1.0)
        img_bytes = pdf_to_image(pdf_bytes, zoom_factor)

        # Initialize client (cached)
        if 'vision' not in _backend_clients:
            from .ocr.backends import get_backend
            init_client_func, _ = get_backend('vision')
            _backend_clients['vision'] = init_client_func(config)
            logger.info("Initialized Google Cloud Vision client")

        # Get process_page function
        from .ocr.backends import get_backend
        _, process_page_func = get_backend('vision')

        # Determine base output directory for images
        base_output_dir = images_dir.parent if images_dir else None

        # Process page
        result = process_page_func(
            client=_backend_clients['vision'],
            img_bytes=img_bytes,
            page_num=chapter_index,  # chapter_index is actually page_num
            config=config,
            base_output_dir=base_output_dir
        )

        # Extract results
        markdown = result.get('text', '')
        illustrations = result.get('illustrations', [])

        # Inject illustrations into markdown if present
        if illustrations:
            from .ocr import inject_illustrations_into_text
            markdown = inject_illustrations_into_text(markdown, illustrations)

        # Count illustrations as images
        updated_counter = image_counter + len(illustrations)

        return markdown, illustrations, updated_counter

    else:
        raise ValueError(f"Unknown OCR backend: {backend}. Supported: vertex, mistral, vllm, azure, vision")


def count_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple approximation: ~4 characters per token for English,
    ~2 characters per token for CJK languages.
    """
    # Simple heuristic: check if text contains CJK characters
    import re
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]')
    has_cjk = bool(cjk_pattern.search(text))

    if has_cjk:
        # CJK languages: ~2 chars per token
        return len(text) // 2
    else:
        # English: ~4 chars per token
        return len(text) // 4


def extract_pdf_pages(pdf_path: Path, start_page: int, end_page: int) -> bytes:
    """Extract specific pages from PDF and return as bytes."""
    with fitz.open(pdf_path) as full_pdf:
        # Create a new PDF with just the specified pages
        extracted_pdf = fitz.open()
        for page_num in range(start_page - 1, end_page):  # Convert to 0-based indexing
            if page_num < len(full_pdf):
                extracted_pdf.insert_pdf(full_pdf, from_page=page_num, to_page=page_num)

        # Save to bytes
        pdf_bytes = extracted_pdf.tobytes()
        extracted_pdf.close()

    return pdf_bytes


def ocr_full_book_pagewise(
    pdf_path: Path,
    output_dir: Path,
    session: AuthorizedSession = None,
    project_id: str = None,
    location: str = None,
    start_page: int = 1,
    end_page: int = None,
    backend: str = "vertex",
    api_key: str = None,
    base_url: str = None,
    resume: bool = False,
    config: Dict = None,
    max_workers: int = 5
) -> None:
    """OCR全书，并行处理多页。

    Args:
        pdf_path: Path to PDF file
        output_dir: Base output directory (will create pages/ subdirectory)
        session: Authorized session for API calls
        project_id: GCP project ID
        location: GCP location
        start_page: First page to process (default: 1)
        end_page: Last page to process (default: all pages)
        backend: OCR backend to use
        api_key: API key for Mistral backend
        base_url: Base URL for Mistral backend
        resume: Resume from previous progress
        config: Configuration dict (for retry settings)
        max_workers: Number of parallel OCR requests (default: 5)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Get retry settings from config
    if config is None:
        config = {}

    ocr_config = config.get('ocr', {})
    max_retries = ocr_config.get('max_retries', 5)
    initial_backoff = ocr_config.get('initial_backoff', 4.0)

    # Optional: backend-specific override
    backend_config = ocr_config.get('backends', {}).get(backend, {})
    if backend_config:
        max_retries = backend_config.get('max_retries', max_retries)
        initial_backoff = backend_config.get('initial_backoff', initial_backoff)
    # Create pages directory
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress_file = pages_dir / "ocr_progress.json"
    page_stats_file = pages_dir / "page_stats.json"

    if resume and progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        logger.info(f"Resuming from progress: {len(progress['pages_processed'])} pages already processed")
    else:
        progress = {
            'pages_processed': [],
            'failed_pages': [],
            'global_image_counter': 0
        }

    if page_stats_file.exists():
        with open(page_stats_file, 'r') as f:
            page_stats = json.load(f)
    else:
        page_stats = {}

    # Determine page range
    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)

    if end_page is None:
        end_page = total_pages

    end_page = min(end_page, total_pages)

    logger.info(f"Processing pages {start_page}-{end_page} (total: {end_page - start_page + 1} pages, {max_workers} workers)")

    # Determine pages to process
    pages_to_process = []
    for page_num in range(start_page, end_page + 1):
        page_file = pages_dir / f"page_{page_num:03d}.md"

        # Check if already processed
        if resume and page_num in progress['pages_processed']:
            if page_file.exists():
                # Update stats if not already recorded
                if str(page_num) not in page_stats:
                    with open(page_file, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    token_count = count_tokens(existing_content)
                    page_stats[str(page_num)] = {
                        'tokens': token_count,
                        'file': str(page_file.relative_to(output_dir)),
                        'char_count': len(existing_content)
                    }
                logger.debug(f"Skipping page {page_num} (already processed)")
                continue
            else:
                logger.warning(f"Page {page_num} marked as processed but file missing, reprocessing...")
                progress['pages_processed'].remove(page_num)

        pages_to_process.append(page_num)

    if not pages_to_process:
        logger.info("All pages already processed")
    else:
        logger.info(f"Processing {len(pages_to_process)} pages...")

        # Define worker function
        def process_single_page(page_num):
            """Process a single page and return result."""
            try:
                pdf_bytes = extract_pdf_pages(pdf_path, page_num, page_num)
                chunk_info = f"Page {page_num}"

                # Use page_num as image counter base to avoid conflicts
                markdown, images, img_count = ocr_pdf_chunk(
                    pdf_bytes,
                    session,
                    project_id,
                    location,
                    chunk_info,
                    images_dir,
                    page_num,
                    page_num * 100,  # Use page-based counter to avoid conflicts
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                    backend=backend,
                    api_key=api_key,
                    base_url=base_url,
                    config=config
                )

                return {
                    'page_num': page_num,
                    'markdown': markdown,
                    'images': images,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                import traceback
                return {
                    'page_num': page_num,
                    'markdown': None,
                    'images': None,
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }

        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_page, page_num): page_num
                      for page_num in pages_to_process}

            for future in as_completed(futures):
                result = future.result()
                page_num = result['page_num']
                page_file = pages_dir / f"page_{page_num:03d}.md"

                if result['success']:
                    # Save page markdown
                    with open(page_file, 'w', encoding='utf-8') as f:
                        f.write(result['markdown'])

                    # Count tokens
                    token_count = count_tokens(result['markdown'])

                    # Update stats
                    page_stats[str(page_num)] = {
                        'tokens': token_count,
                        'file': str(page_file.relative_to(output_dir)),
                        'char_count': len(result['markdown'])
                    }

                    # Update progress
                    if page_num not in progress['pages_processed']:
                        progress['pages_processed'].append(page_num)

                    # Remove from failed list if it was there
                    if page_num in progress.get('failed_pages', []):
                        progress['failed_pages'].remove(page_num)

                    logger.success(f"Saved page {page_num} ({token_count} tokens)")
                else:
                    logger.error(f"Failed to process page {page_num}: {result['error']}")
                    logger.debug(result.get('traceback', ''))

                    # Record failed page
                    if 'failed_pages' not in progress:
                        progress['failed_pages'] = []
                    if page_num not in progress['failed_pages']:
                        progress['failed_pages'].append(page_num)

                # Save progress after each page
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)

                with open(page_stats_file, 'w') as f:
                    json.dump(page_stats, f, indent=2)

    # Summary
    total_tokens = sum(stats['tokens'] for stats in page_stats.values())
    avg_tokens = total_tokens / len(page_stats) if page_stats else 0

    logger.success(f"\n=== Page-wise OCR Complete ===")
    logger.info(f"Total pages processed: {len(progress['pages_processed'])}/{end_page - start_page + 1}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Average tokens per page: {avg_tokens:.0f}")

    # Report failed pages
    failed_pages = progress.get('failed_pages', [])
    if failed_pages:
        logger.warning(f"Failed pages ({len(failed_pages)}): {sorted(failed_pages)}")
        logger.warning(f"To retry failed pages, run again with --resume flag")
    else:
        logger.success(f"All pages processed successfully!")

    logger.info(f"Output directory: {pages_dir}")


def main():
    """CLI entry point for page-wise OCR."""
    parser = argparse.ArgumentParser(description="OCR PDF pages individually")
    parser.add_argument("-i", "--input", help="Path to input PDF file")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--start-page", type=int, default=1, help="First page to process")
    parser.add_argument("--end-page", type=int, help="Last page to process")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--max-workers", type=int, default=5, help="Number of parallel OCR requests (default: 5)")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return

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
        # Azure Document Intelligence backend (for Japanese vertical text)
        # Client will be initialized lazily in ocr_pdf_chunk
        logger.info("Using Azure Document Intelligence backend")

    elif ocr_backend == "vision":
        # Google Cloud Vision API backend (for Japanese vertical text)
        # Client will be initialized lazily in ocr_pdf_chunk
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

    logger.info(f"Using PDF: {pdf_path}")

    # Output directory
    output_dir = Path("output") / book_title

    # Run page-wise OCR
    ocr_full_book_pagewise(
        pdf_path=pdf_path,
        output_dir=output_dir,
        session=session,
        project_id=project_id,
        location=location,
        start_page=args.start_page,
        end_page=args.end_page,
        backend=ocr_backend,
        api_key=api_key,
        base_url=base_url,
        resume=args.resume,
        config=config,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
