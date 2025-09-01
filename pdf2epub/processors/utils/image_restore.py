"""
Image restoration utilities for markdown processors.

This module provides utilities to restore lost images during text processing,
particularly useful when LLMs accidentally remove image references during
content transformation.
"""

import re
from typing import List, Tuple
from loguru import logger


def extract_images(text: str) -> List[Tuple[str, str]]:
    """
    Extract all image references from markdown text.
    
    Args:
        text: Markdown text containing image references
    
    Returns:
        List of tuples (alt_text, image_path)
    """
    # Pattern to match markdown images: ![alt_text](path)
    pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    matches = re.findall(pattern, text)
    return matches


def restore_lost_images(original: str, processed: str) -> str:
    """
    Restore any images that were lost during processing.
    
    This function compares images in the original and processed text,
    and attempts to restore any missing images at appropriate locations.
    
    Args:
        original: Original markdown text with images
        processed: Processed markdown text potentially missing images
    
    Returns:
        Processed text with restored images
    """
    # Extract images from both texts
    original_images = extract_images(original)
    processed_images = extract_images(processed)
    
    # Find missing images
    processed_paths = {img[1] for img in processed_images}
    missing_images = [img for img in original_images if img[1] not in processed_paths]
    
    if not missing_images:
        return processed
    
    logger.info(f"Found {len(missing_images)} missing images to restore")
    
    # For each missing image, try to find where it should be inserted
    result = processed
    for alt_text, img_path in missing_images:
        # Try to find context around the image in original
        img_markdown = f"![{alt_text}]({img_path})"
        img_pos = original.find(img_markdown)
        
        if img_pos == -1:
            continue
        
        # Get context before and after the image (up to 100 chars)
        context_before_start = max(0, img_pos - 100)
        context_before = original[context_before_start:img_pos].strip()
        
        context_after_end = min(len(original), img_pos + len(img_markdown) + 100)
        context_after = original[img_pos + len(img_markdown):context_after_end].strip()
        
        # Try to find a good insertion point in processed text
        insertion_point = find_best_insertion_point(
            result, context_before, context_after, img_markdown
        )
        
        if insertion_point != -1:
            # Insert the image
            result = (
                result[:insertion_point] + 
                "\n\n" + img_markdown + "\n\n" + 
                result[insertion_point:]
            )
            logger.debug(f"Restored image: {img_path}")
    
    return result


def find_best_insertion_point(
    text: str,
    context_before: str,
    context_after: str,
    image_markdown: str
) -> int:
    """
    Find the best position to insert a missing image.
    
    Args:
        text: Text to insert image into
        context_before: Text that appeared before the image
        context_after: Text that appeared after the image
        image_markdown: The image markdown to insert
    
    Returns:
        Position to insert at, or -1 if no good position found
    """
    # Try to find exact context match
    if context_before and context_after:
        # Look for both contexts near each other
        before_pos = text.find(context_before)
        if before_pos != -1:
            # Check if context_after appears nearby
            search_start = before_pos + len(context_before)
            search_end = min(len(text), search_start + 500)
            after_pos = text.find(context_after, search_start, search_end)
            
            if after_pos != -1:
                # Found both contexts, insert between them
                return before_pos + len(context_before)
    
    # Try just context before
    if context_before:
        # Find the last occurrence of context_before
        before_pos = text.rfind(context_before)
        if before_pos != -1:
            return before_pos + len(context_before)
    
    # Try just context after
    if context_after:
        # Find the first occurrence of context_after
        after_pos = text.find(context_after)
        if after_pos != -1:
            return after_pos
    
    # If we have a chapter or section reference in alt text, try to find it
    if "[" not in image_markdown:  # Avoid nested brackets
        alt_match = re.search(r'\[([^\]]+)\]', image_markdown)
        if alt_match:
            alt_text = alt_match.group(1)
            # Look for chapter or section headings
            if any(keyword in alt_text.lower() for keyword in ['chapter', 'section', 'figure', 'illustration']):
                # Try to find a related heading
                heading_patterns = [
                    r'^#{1,3}\s+.*' + re.escape(word) + r'.*$'
                    for word in alt_text.split()
                    if len(word) > 3
                ]
                
                for pattern in heading_patterns:
                    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                    if match:
                        # Insert after the heading
                        return match.end()
    
    return -1