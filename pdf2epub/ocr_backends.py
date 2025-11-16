"""
OCR backend implementations for Mistral and Vertex AI.
"""

import base64
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from loguru import logger


def ocr_pdf_chunk_mistral(
    pdf_bytes: bytes,
    api_key: str,
    chunk_info: str,
    images_dir: Optional[Path] = None,
    chapter_index: int = 0,
    image_counter: int = 0,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
    base_url: str = "https://api.mistral.ai/v1"
) -> Tuple[str, List[Dict], int]:
    """
    OCR a PDF chunk using Mistral's official API.

    Args:
        pdf_bytes: PDF content as bytes
        api_key: Mistral API key
        chunk_info: Description of the chunk for logging
        images_dir: Directory to save extracted images
        chapter_index: Index of current chapter
        image_counter: Current image counter
        max_retries: Maximum retry attempts for 429 errors
        initial_backoff: Initial backoff time in seconds
        base_url: Base URL for Mistral API (default: https://api.mistral.ai/v1)

    Returns:
        Tuple of (markdown_content, images_info, updated_image_counter)
    """

    # Convert PDF to base64 data URL
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    data_url = f"data:application/pdf;base64,{pdf_b64}"

    # Mistral OCR API endpoint
    url = f"{base_url}/ocr"

    # Headers with Bearer token
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request payload
    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": data_url,
        },
        "include_image_base64": True,  # Include images in base64 format
    }

    logger.info(f"Sending OCR request to Mistral API for {chunk_info}...")

    # Retry logic with exponential backoff for rate limits
    backoff = initial_backoff
    result = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)

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
                logger.error(f"Response: {resp.text[:500]}")

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
                    "format": img_format,
                    "base64": base64_data
                })

    # Combine markdown from all pages
    combined_markdown = "\n\n".join(markdown_content)

    # Save images if directory is provided
    if images_dir and all_images:
        images_dir.mkdir(parents=True, exist_ok=True)

        for img_info in all_images:
            # Create filename using chapter and image counter
            img_filename = f"chapter_{chapter_index}_img_{image_counter:03d}.{img_info['format']}"
            img_path = images_dir / img_filename

            # Decode and save image
            img_bytes = base64.b64decode(img_info['base64'])
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            # Replace image reference in markdown
            # Mistral can return references in different formats:
            # - ![img-0.jpeg](img-0.jpeg)
            # - ![Image](img-0.jpeg)
            # - ![](img-0.jpeg)
            img_id = img_info['id']
            new_ref = f"![Image](../images/{img_filename})"

            # Try different possible formats
            possible_refs = [
                f"![{img_id}]({img_id})",  # ![img-0.jpeg](img-0.jpeg)
                f"![Image]({img_id})",      # ![Image](img-0.jpeg)
                f"![]({img_id})",           # ![](img-0.jpeg)
            ]

            replaced = False
            for old_ref in possible_refs:
                if old_ref in combined_markdown:
                    combined_markdown = combined_markdown.replace(old_ref, new_ref)
                    logger.debug(f"Saved image: {img_filename}, replaced {old_ref} with {new_ref}")
                    replaced = True
                    break

            if not replaced:
                logger.warning(f"Could not find reference for {img_id} in markdown")
            image_counter += 1

        logger.info(f"Saved {len(all_images)} images for {chunk_info}")

    logger.success(f"OCR completed for {chunk_info} ({len(pages)} pages, {len(all_images)} images)")
    return combined_markdown, all_images, image_counter


def ocr_pdf_chunk_vertex(
    pdf_bytes: bytes,
    session: AuthorizedSession,
    project_id: str,
    location: str,
    chunk_info: str,
    images_dir: Optional[Path] = None,
    chapter_index: int = 0,
    image_counter: int = 0,
    max_retries: int = 5,
    initial_backoff: float = 4.0
) -> Tuple[str, List[Dict], int]:
    """
    OCR a PDF chunk using Vertex AI Mistral OCR API.

    This is the existing implementation, just renamed for clarity.

    Args:
        pdf_bytes: PDF content as bytes
        session: Authorized session for API calls
        project_id: GCP project ID
        location: GCP location
        chunk_info: Description of the chunk for logging
        images_dir: Directory to save extracted images
        chapter_index: Index of current chapter
        image_counter: Current image counter
        max_retries: Maximum retry attempts for 429 errors
        initial_backoff: Initial backoff time in seconds

    Returns:
        Tuple of (markdown_content, images_info, updated_image_counter)
    """

    # Check size and compress if needed
    size_mb = len(pdf_bytes) / (1024 * 1024)
    logger.debug(f"PDF size for {chunk_info}: {size_mb:.2f}MB")

    # Compress if over 20MB to leave buffer for base64 encoding (which adds ~33%)
    if size_mb > 20:
        from .utils.pdf_utils import compress_pdf_bytes
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

    logger.info(f"Sending OCR request to Vertex AI for {chunk_info}...")

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
                logger.error(f"Response: {resp.text[:500]}")

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
                    "format": img_format,
                    "base64": base64_data
                })

    # Combine markdown from all pages
    combined_markdown = "\n\n".join(markdown_content)

    # Save images if directory is provided
    if images_dir and all_images:
        images_dir.mkdir(parents=True, exist_ok=True)

        for img_info in all_images:
            # Create filename using chapter and image counter
            img_filename = f"chapter_{chapter_index}_img_{image_counter:03d}.{img_info['format']}"
            img_path = images_dir / img_filename

            # Decode and save image
            img_bytes = base64.b64decode(img_info['base64'])
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            # Replace image reference in markdown
            # Mistral can return references in different formats:
            # - ![img-0.jpeg](img-0.jpeg)
            # - ![Image](img-0.jpeg)
            # - ![](img-0.jpeg)
            img_id = img_info['id']
            new_ref = f"![Image](../images/{img_filename})"

            # Try different possible formats
            possible_refs = [
                f"![{img_id}]({img_id})",  # ![img-0.jpeg](img-0.jpeg)
                f"![Image]({img_id})",      # ![Image](img-0.jpeg)
                f"![]({img_id})",           # ![](img-0.jpeg)
            ]

            replaced = False
            for old_ref in possible_refs:
                if old_ref in combined_markdown:
                    combined_markdown = combined_markdown.replace(old_ref, new_ref)
                    logger.debug(f"Saved image: {img_filename}, replaced {old_ref} with {new_ref}")
                    replaced = True
                    break

            if not replaced:
                logger.warning(f"Could not find reference for {img_id} in markdown")
            image_counter += 1

        logger.info(f"Saved {len(all_images)} images for {chunk_info}")

    logger.success(f"OCR completed for {chunk_info} ({len(pages)} pages, {len(all_images)} images)")
    return combined_markdown, all_images, image_counter

def ocr_pdf_chunk_vllm(
    pdf_bytes: bytes,
    config: Dict,
    chunk_info: str,
    images_dir: Optional[Path] = None,
    chapter_index: int = 0,
    image_counter: int = 0,
    max_retries: int = 5,
    initial_backoff: float = 4.0
) -> Tuple[str, List[Dict], int]:
    """
    OCR a PDF chunk using VLLM backend (processes page by page).

    Args:
        pdf_bytes: PDF content as bytes
        config: Configuration dictionary
        chunk_info: Description of the chunk for logging
        images_dir: Directory to save extracted images
        chapter_index: Index of current chapter
        image_counter: Current image counter
        max_retries: Maximum retry attempts (not used for vllm)
        initial_backoff: Initial backoff time (not used for vllm)

    Returns:
        Tuple of (markdown_content, images_info, updated_image_counter)
    """
    import io
    import re
    import fitz  # PyMuPDF
    from PIL import Image
    from pdf2epub.ocr.backends.vllm import init_client

    logger.info(f"Starting VLLM OCR for {chunk_info}")

    # Initialize VLLM client
    client = init_client(config)

    # Open PDF from bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    markdown_parts = []
    all_images = []

    # Process each page
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        logger.info(f"  Processing page {page_idx + 1}/{len(doc)}")

        # Render page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        # Perform OCR
        try:
            text = client.ocr(img)
            markdown_parts.append(text)
        except Exception as e:
            logger.error(f"OCR failed for page {page_idx + 1}: {e}")
            markdown_parts.append(f"\n\n[OCR Error on page {page_idx + 1}]\n\n")

    doc.close()

    # Combine all pages
    combined_markdown = "\n\n".join(markdown_parts)

    # Extract and save base64 images from markdown
    if images_dir:
        images_dir.mkdir(parents=True, exist_ok=True)

        # Pattern to match ![...](data:image/...;base64,...)
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^\)]+)\)'

        def replace_image(match):
            nonlocal image_counter
            alt_text = match.group(1)
            image_format = match.group(2)
            base64_data = match.group(3)

            # Create filename
            img_filename = f"chapter_{chapter_index}_img_{image_counter:03d}.{image_format}"
            img_path = images_dir / img_filename

            try:
                # Decode and save image
                img_data = base64.b64decode(base64_data)
                with open(img_path, 'wb') as f:
                    f.write(img_data)

                logger.debug(f"Saved image: {img_path} ({len(img_data) / 1024:.2f} KB)")

                # Track image info
                all_images.append({
                    "filename": img_filename,
                    "format": image_format,
                    "size": len(img_data)
                })

                image_counter += 1

                # Return new markdown reference
                return f"![{alt_text}](../images/{img_filename})"
            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                return match.group(0)

        # Replace all base64 images with file references
        combined_markdown = re.sub(pattern, replace_image, combined_markdown)

        if all_images:
            logger.info(f"Extracted and saved {len(all_images)} images for {chunk_info}")

    logger.success(f"VLLM OCR completed for {chunk_info} ({len(markdown_parts)} pages, {len(all_images)} images)")
    return combined_markdown, all_images, image_counter
