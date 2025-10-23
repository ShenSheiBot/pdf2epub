import json
import base64
import fitz  # PyMuPDF
import yaml
import argparse
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from loguru import logger
from .utils.logging_config import configure_logging
from .ocr_backends import ocr_pdf_chunk_mistral, ocr_pdf_chunk_vertex, ocr_pdf_chunk_vllm
from .utils.common import load_config, load_book_structure
from .pdf_compressor import compress_pdf

# Configure logger
logger = configure_logging()


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


def compress_pdf_bytes(pdf_bytes: bytes, max_size_mb: float = 20.0) -> bytes:
    """Compress PDF bytes if they exceed the size limit."""
    size_mb = len(pdf_bytes) / (1024 * 1024)
    
    if size_mb <= max_size_mb:
        logger.debug(f"PDF size {size_mb:.2f}MB is under limit, no compression needed")
        return pdf_bytes
    
    logger.info(f"PDF size {size_mb:.2f}MB exceeds {max_size_mb}MB limit, compressing...")
    
    # Write bytes to temporary file for compression
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_input:
        temp_input.write(pdf_bytes)
        temp_input_path = temp_input.name
    
    try:
        # Try different compression settings
        compression_settings = [
            (150, 60, False),  # Medium compression
            (120, 40, False),  # Higher compression
            (100, 30, True),   # Aggressive compression with grayscale
            (80, 20, True),    # Very aggressive
        ]
        
        for dpi, quality, grayscale in compression_settings:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                logger.debug(f"Trying compression with DPI={dpi}, quality={quality}, grayscale={grayscale}")
                success, stats = compress_pdf(
                    temp_input_path, 
                    temp_output_path, 
                    dpi=dpi, 
                    quality=quality, 
                    grayscale=grayscale
                )
                
                if success:
                    compressed_size_mb = stats["output_size_mb"]
                    logger.info(f"Compressed to {compressed_size_mb:.2f}MB ({stats['saved_percentage']:.1f}% reduction)")
                    
                    if compressed_size_mb <= max_size_mb:
                        # Read compressed PDF
                        with open(temp_output_path, 'rb') as f:
                            compressed_bytes = f.read()
                        
                        # Clean up temp output file
                        Path(temp_output_path).unlink(missing_ok=True)
                        
                        return compressed_bytes
                
                # Clean up temp output file if compression didn't help enough
                Path(temp_output_path).unlink(missing_ok=True)
                
            except Exception as e:
                logger.warning(f"Compression attempt failed: {e}")
                Path(temp_output_path).unlink(missing_ok=True)
        
        # If all compression attempts failed, return original
        logger.warning("Could not compress PDF below size limit, returning original")
        return pdf_bytes
        
    finally:
        # Clean up temp input file
        Path(temp_input_path).unlink(missing_ok=True)


def split_chapter_ranges(start_page: int, end_page: int, max_pages: int = 25) -> List[Tuple[int, int]]:
    """Split a chapter into chunks that don't exceed max_pages."""
    total_pages = end_page - start_page + 1
    
    if total_pages <= max_pages:
        return [(start_page, end_page)]
    
    chunks = []
    current_start = start_page
    
    while current_start <= end_page:
        current_end = min(current_start + max_pages - 1, end_page)
        chunks.append((current_start, current_end))
        current_start = current_end + 1
    
    return chunks


def ocr_pdf_chunk(
    pdf_bytes: bytes,
    session: AuthorizedSession = None,
    project_id: str = None,
    location: str = None,
    chunk_info: str = "",
    images_dir: Path = None,
    chapter_index: int = 1,
    image_counter: int = 0,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
    backend: str = "vertex",
    api_key: str = None
) -> Tuple[str, List[Dict], int]:
    """OCR a PDF chunk using selected backend (Vertex AI or Mistral API).

    Routes to the appropriate backend based on configuration.

    Args:
        pdf_bytes: PDF content as bytes
        session: Authorized session for Vertex AI (required for vertex backend)
        project_id: GCP project ID (required for vertex backend)
        location: GCP location (required for vertex backend)
        chunk_info: Description of the chunk being processed
        images_dir: Directory to save extracted images
        chapter_index: Chapter number for image naming
        image_counter: Starting counter for image numbering
        max_retries: Maximum number of retry attempts for 429 errors
        initial_backoff: Initial backoff time in seconds
        backend: OCR backend to use ('vertex' or 'mistral')
        api_key: Mistral API key (required for mistral backend)

    Returns:
        Tuple of (markdown_content, images_info, updated_image_counter)
    """

    # Route to appropriate backend
    if backend == "mistral":
        if not api_key:
            raise ValueError("Mistral API key is required for mistral backend")

        return ocr_pdf_chunk_mistral(
            pdf_bytes=pdf_bytes,
            api_key=api_key,
            chunk_info=chunk_info,
            images_dir=images_dir,
            chapter_index=chapter_index,
            image_counter=image_counter,
            max_retries=max_retries,
            initial_backoff=initial_backoff
        )

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
        # Load config for vllm backend
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

    else:
        raise ValueError(f"Unknown OCR backend: {backend}")

    # The original Vertex AI implementation follows below
    # This code is now unreachable but kept for reference

    # Check size and compress if needed
    size_mb = len(pdf_bytes) / (1024 * 1024)
    logger.debug(f"PDF size for {chunk_info}: {size_mb:.2f}MB")
    
    # Compress if over 20MB to leave buffer for base64 encoding (which adds ~33%)
    # 20MB * 1.33 = ~26.6MB, leaving room under 30MB limit
    if size_mb > 20:
        pdf_bytes = compress_pdf_bytes(pdf_bytes, max_size_mb=20.0)
        new_size_mb = len(pdf_bytes) / (1024 * 1024)
        logger.info(f"Compressed PDF from {size_mb:.2f}MB to {new_size_mb:.2f}MB")
    
    # Convert PDF to base64 data URL
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    data_url = f"data:application/pdf;base64,{pdf_b64}"
    
    # Vertex AI Mistral OCR endpoint
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/publishers/mistralai/models/mistral-ocr-2505:rawPredict"
    )
    
    # Request payload
    payload = {
        "model": "mistral-ocr-2505",
        "document": {
            "type": "document_url",
            "document_url": data_url,
        },
        "include_image_base64": True,  # Include images in base64 format
    }
    
    logger.info(f"Sending OCR request for {chunk_info}...")
    
    # Retry logic with exponential backoff for rate limits
    backoff = initial_backoff
    result = None
    
    for attempt in range(max_retries):
        try:
            resp = session.post(url, json=payload, timeout=300)
            
            # Check for rate limit error
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    wait_time = backoff * (2 ** attempt) + (backoff * 0.1 * (0.5 - time.time() % 1))
                    logger.warning(
                        f"Rate limit hit for {chunk_info} (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time:.2f} seconds before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP 429 for {chunk_info} after {max_retries} attempts")
                    logger.error(f"Response: {resp.text[:500]}")
            
            # Log other error responses
            if resp.status_code != 200:
                logger.error(f"HTTP {resp.status_code} for {chunk_info}")
                logger.error(f"Response: {resp.text[:500]}")  # First 500 chars of error
                
            resp.raise_for_status()
            result = resp.json()
            break  # Success, exit retry loop
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1 and "429" in str(e):
                wait_time = backoff * (2 ** attempt) + (backoff * 0.1 * (0.5 - time.time() % 1))
                logger.warning(
                    f"Request exception with rate limit for {chunk_info} (attempt {attempt + 1}/{max_retries}). "
                    f"Error: {e}. Waiting {wait_time:.2f} seconds before retry..."
                )
                time.sleep(wait_time)
                continue
            else:
                # Re-raise if not a rate limit error or if we've exhausted retries
                raise
    
    # Process the result if we got one
    if result is None:
        raise Exception(f"Failed to get OCR response for {chunk_info} after {max_retries} attempts")
    
    # Extract markdown and images from all pages
    pages = result.get("pages", [])
    markdown_content = []
    all_images = []
    
    for page_idx, page in enumerate(pages):
        md = page.get("markdown", "")
        if md:
            markdown_content.append(md)
        
        # Extract images from this page
        images = page.get("images", [])
        for img_idx, img_data in enumerate(images):
            # Check for image_base64 field (API returns data URL format)
            if img_data.get("image_base64"):
                # Extract base64 data from data URL
                base64_data = img_data["image_base64"]
                if base64_data.startswith("data:"):
                    # Remove data URL prefix
                    base64_data = base64_data.split(",")[1] if "," in base64_data else base64_data
                
                # Get image ID and format
                img_id = img_data.get("id", f"img-{img_idx}")
                # Extract format from ID (e.g., "img-0.jpeg" -> "jpeg")
                img_format = "jpeg"
                if "." in img_id:
                    img_format = img_id.split(".")[-1]
                
                all_images.append({
                    "page": page_idx,
                    "index": img_idx,
                    "id": img_id,
                    "base64": base64_data,
                    "format": img_format
                })
    
    combined_markdown = "\n\n---\n\n".join(markdown_content)
    
    # Save images if directory provided
    if images_dir and all_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        import re
        
        # Save images and build mapping
        for img_info in all_images:
            img_filename = f"chapter_{chapter_index}_img_{image_counter}.{img_info['format']}"
            img_path = images_dir / img_filename
            
            # Decode and save image
            import base64 as b64
            img_bytes = b64.b64decode(img_info['base64'])
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            
            # Replace image reference in markdown
            # The API returns references like ![img-0.jpeg](img-0.jpeg)
            img_id = img_info['id']
            # Escape special characters in the ID for regex
            escaped_id = re.escape(img_id)
            # Match patterns like ![img-0.jpeg](img-0.jpeg) or similar
            old_pattern = rf'!\[{escaped_id}\]\({escaped_id}\)'
            new_ref = f"![Image](../images/{img_filename})"
            combined_markdown = re.sub(old_pattern, new_ref, combined_markdown)
            
            logger.debug(f"Saved image: {img_filename}, replaced {img_id}")
            image_counter += 1
            
        logger.info(f"Saved {len(all_images)} images for {chunk_info}")
    
    logger.success(f"OCR completed for {chunk_info} ({len(pages)} pages, {len(all_images)} images)")
    return combined_markdown, all_images, image_counter


def process_chapter(
    chapter: Dict,
    chapter_index: int,
    pdf_path: Path,
    session: AuthorizedSession = None,
    project_id: str = None,
    location: str = None,
    output_dir: Path = None,
    images_dir: Path = None,
    max_pages: int = 30,
    max_size_mb: float = 20.0,
    backend: str = "vertex",
    api_key: str = None
) -> None:
    """Process a chapter at the chapter level.
    Treats the entire chapter as one unit, regardless of subchapter boundaries.
    Only splits into chunks when exceeding max_pages or max_size_mb.

    Args:
        chapter: Chapter dictionary with title, start_page, end_page, and optional subchapters
        chapter_index: Chapter number
        pdf_path: Path to PDF file
        session: Authorized session for API calls
        project_id: GCP project ID
        location: GCP location
        output_dir: Output directory for markdown files
        images_dir: Directory to save extracted images
        max_pages: Maximum pages per OCR request (default: 30)
        max_size_mb: Maximum size in MB per request before compression (default: 20.0)
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
        logger.info(f"Processing Chapter {chapter_index}: {chapter_title}")
        logger.info(f"  Chapter header pages: {start_page}-{end_page}")
        logger.info(f"  Contains {len(subchapters)} subchapters")
        for i, sub in enumerate(subchapters, 1):
            logger.info(f"    Subchapter {i}: pages {sub['start_page']}-{sub['end_page']}")
        logger.info(f"  Total pages to process: {start_page}-{actual_end_page}")
        end_page = actual_end_page
    else:
        logger.info(f"Processing Chapter {chapter_index}: {chapter_title} (pages {start_page}-{end_page})")

    total_pages = end_page - start_page + 1
    logger.info(f"  Total pages: {total_pages}")

    # Process the entire chapter including all subchapters as one unit
    # Build the chapter header
    markdown_parts = [f"# {chapter_title}\n"]
    image_counter = 0

    # First try to process the entire chapter as a single chunk
    pdf_bytes = extract_pdf_pages(pdf_path, start_page, end_page)
    size_mb = len(pdf_bytes) / (1024 * 1024)

    if total_pages <= max_pages and size_mb <= max_size_mb:
        # Can process in one go
        logger.info(f"  Processing entire chapter in one request ({total_pages} pages, {size_mb:.2f}MB)")

        chunk_info = f"Chapter {chapter_index} (pages {start_page}-{end_page})"
        markdown, images, image_counter = ocr_pdf_chunk(
            pdf_bytes, session, project_id, location, chunk_info,
            images_dir, chapter_index, image_counter,
            max_retries=5, initial_backoff=4.0,
            backend=backend, api_key=api_key
        )

        markdown_parts.append(markdown)
    else:
        # Need to split
        logger.info(f"  Chapter exceeds limits ({total_pages} pages, {size_mb:.2f}MB), splitting...")

        # Split based on pages first
        chunks = split_chapter_ranges(start_page, end_page, max_pages)
        logger.info(f"  Split into {len(chunks)} chunks")

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
            chunk_info = f"Chapter {chapter_index}"
            if len(chunks) > 1:
                chunk_info += f" (chunk {chunk_idx}/{len(chunks)}, pages {chunk_start}-{chunk_end})"
            else:
                chunk_info += f" (pages {chunk_start}-{chunk_end})"

            pdf_bytes = extract_pdf_pages(pdf_path, chunk_start, chunk_end)

            # Check size and potentially split further if still too large
            chunk_size_mb = len(pdf_bytes) / (1024 * 1024)
            if chunk_size_mb > max_size_mb:
                # Need to split this chunk further
                logger.warning(f"    Chunk {chunk_idx} is {chunk_size_mb:.2f}MB, needs further splitting")

                # Calculate how many sub-chunks we need
                chunk_pages = chunk_end - chunk_start + 1
                estimated_pages_per_sub = max(1, int(chunk_pages * (max_size_mb / chunk_size_mb) * 0.9))  # 90% to be safe

                sub_chunks = split_chapter_ranges(chunk_start, chunk_end, estimated_pages_per_sub)
                logger.info(f"    Further split into {len(sub_chunks)} sub-chunks")

                for sub_idx, (sub_start, sub_end) in enumerate(sub_chunks, 1):
                    sub_info = f"Chapter {chapter_index} (chunk {chunk_idx}.{sub_idx}/{len(chunks)}.{len(sub_chunks)}, pages {sub_start}-{sub_end})"

                    sub_pdf_bytes = extract_pdf_pages(pdf_path, sub_start, sub_end)
                    sub_markdown, sub_images, image_counter = ocr_pdf_chunk(
                        sub_pdf_bytes, session, project_id, location, sub_info,
                        images_dir, chapter_index, image_counter,
                        max_retries=5, initial_backoff=4.0,
                        backend=backend, api_key=api_key
                    )

                    markdown_parts.append(sub_markdown)
            else:
                # Size is OK, process normally
                chunk_markdown, chunk_images, image_counter = ocr_pdf_chunk(
                    pdf_bytes, session, project_id, location, chunk_info,
                    images_dir, chapter_index, image_counter,
                    max_retries=5, initial_backoff=4.0,
                    backend=backend, api_key=api_key
                )

                markdown_parts.append(chunk_markdown)

    # Combine all parts and save
    full_markdown = "\n\n".join(markdown_parts)
    output_file = output_dir / f"chapter_{chapter_index}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    logger.success(f"Saved Chapter {chapter_index} to {output_file}")


# process_subchapters function removed - functionality merged into process_chapter


def load_or_create_progress(progress_file: Path, structure: Dict) -> Dict:
    """Load existing progress or create new progress tracking."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    
    # Create new progress
    progress = {
        "chapters_processed": [],
        "total_chapters": len(structure["chapters"])
    }
    return progress


def save_progress(progress_file: Path, progress: Dict):
    """Save progress to file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def process_matter_pages(
    start_page: int,
    end_page: int,
    pdf_path: Path,
    session: AuthorizedSession = None,
    project_id: str = None,
    location: str = None,
    output_dir: Path = None,
    images_dir: Path = None,
    matter_type: str = "front",
    max_pages: int = 30,
    backend: str = "vertex",
    api_key: str = None
) -> None:
    """Process front matter or back matter pages.
    
    Args:
        start_page: First page number
        end_page: Last page number
        pdf_path: Path to PDF file
        session: Authorized session for API calls
        project_id: GCP project ID
        location: GCP location
        output_dir: Output directory for markdown
        images_dir: Output directory for images
        matter_type: "front" or "back"
        max_pages: Maximum pages per OCR request
    """
    
    logger.info(f"Processing {matter_type} matter (pages {start_page}-{end_page})")
    
    # Split into chunks if needed
    chunks = split_chapter_ranges(start_page, end_page, max_pages)
    
    markdown_parts = []
    image_counter = 0
    
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        chunk_info = f"{matter_type.capitalize()} matter"
        if len(chunks) > 1:
            chunk_info += f" (chunk {chunk_idx}/{len(chunks)}, pages {chunk_start}-{chunk_end})"
        else:
            chunk_info += f" (pages {chunk_start}-{chunk_end})"
        
        # Extract PDF pages for this chunk
        pdf_bytes = extract_pdf_pages(pdf_path, chunk_start, chunk_end)
        
        # OCR the chunk
        markdown, images, image_counter = ocr_pdf_chunk(
            pdf_bytes, session, project_id, location, chunk_info,
            images_dir, 0, image_counter,  # Use 0 as chapter index for matter pages
            max_retries=5, initial_backoff=4.0,
            backend=backend, api_key=api_key
        )
        markdown_parts.append(markdown)
    
    # Combine all parts
    full_markdown = "\n\n".join(markdown_parts)
    
    # Add header
    if matter_type == "front":
        final_markdown = f"# Front Matter\n\n{full_markdown}"
        output_file = output_dir / "front_matter.md"
    else:
        final_markdown = f"# Back Matter\n\n{full_markdown}"
        output_file = output_dir / "back_matter.md"
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_markdown)
    
    logger.success(f"Saved {matter_type} matter to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="OCR PDF chapters using Mistral OCR")
    parser.add_argument("-i", "--input", help="Path to input PDF file (default: output/{book_title}/input.pdf)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--max-pages", type=int, default=25, help="Maximum pages per OCR request (default: 25)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    
    if not book_title:
        logger.error("No title found in config.yaml")
        return
    
    # Load book structure
    structure = load_book_structure(book_title)
    
    # Setup output directories
    output_dir = Path("output") / book_title / "ocr_markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path("output") / book_title / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup progress tracking
    progress_file = output_dir / "ocr_progress.json"
    progress = load_or_create_progress(progress_file, structure)
    
    # Determine OCR backend
    ocr_backend = config.get("ocr_backend", "vertex").lower()
    logger.info(f"Using OCR backend: {ocr_backend}")

    # Setup backend-specific configuration
    session = None
    project_id = None
    location = None
    api_key = None

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
        # Get Mistral API key
        api_key = config.get("mistral_api_key")
        if not api_key:
            raise ValueError("mistral_api_key not found in config.yaml")

        logger.info(f"Using Mistral API with key: {api_key[:8]}...")

    elif ocr_backend == "vllm":
        # VLLM backend uses init_client from vllm.py
        logger.info("Using VLLM backend")

    else:
        raise ValueError(f"Unknown OCR backend: {ocr_backend}. Use 'vertex', 'mistral', or 'vllm'")
    
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
    
    # Process front matter if it exists
    if "front_matter" in structure:
        front_matter = structure["front_matter"]
        # Skip if already processed when resume is enabled
        if args.resume and "front_matter" in progress.get("processed_sections", []):
            logger.info("Skipping front matter (already processed)")
        else:
            try:
                process_matter_pages(
                    front_matter["start_page"],
                    front_matter["end_page"],
                    pdf_path,
                    session,
                    project_id,
                    location,
                    output_dir,
                    images_dir,
                    "front",
                    args.max_pages,
                    backend=ocr_backend,
                    api_key=api_key
                )
                
                # Update progress
                if "processed_sections" not in progress:
                    progress["processed_sections"] = []
                progress["processed_sections"].append("front_matter")
                save_progress(progress_file, progress)
                
            except Exception as e:
                logger.error(f"Failed to process front matter: {e}")
                raise
    
    # Process each chapter
    for chapter_idx, chapter in enumerate(structure["chapters"], 1):
        # Check if already processed
        if chapter_idx in progress["chapters_processed"]:
            logger.info(f"Skipping Chapter {chapter_idx} (already processed)")
            continue
        
        try:
            # Process chapter (handles subchapters internally)
            process_chapter(
                chapter,
                chapter_idx,
                pdf_path,
                session,
                project_id,
                location,
                output_dir,
                images_dir,
                args.max_pages,
                backend=ocr_backend,
                api_key=api_key
            )
            
            # Update progress
            progress["chapters_processed"].append(chapter_idx)
            save_progress(progress_file, progress)
            
        except Exception as e:
            logger.error(f"Failed to process Chapter {chapter_idx}: {e}")
            # Always fail immediately on error
            raise
    
    # Process back matter if it exists
    if "back_matter" in structure:
        back_matter = structure["back_matter"]
        # Skip if already processed when resume is enabled
        if args.resume and "back_matter" in progress.get("processed_sections", []):
            logger.info("Skipping back matter (already processed)")
        else:
            try:
                process_matter_pages(
                    back_matter["start_page"],
                    back_matter["end_page"],
                    pdf_path,
                    session,
                    project_id,
                    location,
                    output_dir,
                    images_dir,
                    "back",
                    args.max_pages,
                    backend=ocr_backend,
                    api_key=api_key
                )
                
                # Update progress
                if "processed_sections" not in progress:
                    progress["processed_sections"] = []
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


if __name__ == "__main__":
    main()
