"""
EPUB to Markdown Converter

Converts EPUB content to Markdown format, similar to ocr_chapters.py output.
Reads book_structure.json and converts each chapter's XHTML to Markdown.

Output structure:
  output/{book_name}/
    epub_markdown/
      chapter_1.md
      chapter_2.md
      ...
      epub_progress.json
    images/
      chapter_1_img_0.jpg
      chapter_2_img_0.jpg
      ...
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import base64
import hashlib
from urllib.parse import unquote, urljoin
from bs4 import BeautifulSoup

from .epub_input.epub_parser import EPUBParser
from .epub_input.custom_converters import convert_html_to_markdown


def save_progress(progress_file: Path, progress: Dict):
    """Save progress to JSON file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def load_progress(progress_file: Path) -> Dict:
    """Load progress from JSON file."""
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"chapters_processed": [], "total_chapters": 0}


def extract_and_save_images(
    soup: BeautifulSoup,
    chapter_idx: int,
    epub_parser: EPUBParser,
    images_dir: Path,
    base_href: str
) -> int:
    """
    Extract images from HTML and save to images directory.

    Args:
        soup: BeautifulSoup object of the HTML
        chapter_idx: Chapter number
        epub_parser: EPUBParser instance
        images_dir: Directory to save images
        base_href: Base href of current XHTML file for resolving relative paths

    Returns:
        Number of images extracted
    """
    img_counter = 0
    base_path = Path(base_href).parent

    for img in soup.find_all('img'):
        src = img.get('src', '')
        if not src:
            continue

        try:
            # Resolve relative path
            if not src.startswith(('http://', 'https://', 'data:')):
                # Handle relative paths
                if src.startswith('../'):
                    # Go up from base_path
                    resolved_src = (base_path / src).as_posix()
                elif src.startswith('./'):
                    resolved_src = (base_path / src[2:]).as_posix()
                else:
                    resolved_src = (base_path / src).as_posix()

                # Normalize path
                resolved_src = Path(resolved_src).as_posix()
            else:
                resolved_src = src

            # Get image from EPUB
            img_item = None

            # Try exact match first
            img_item = epub_parser.get_item_by_href(resolved_src)

            # If not found, try without leading path components
            if not img_item:
                img_name = Path(resolved_src).name
                for resource in epub_parser.resources['images']:
                    if resource['href'].endswith(img_name):
                        img_item = epub_parser.get_item_by_href(resource['href'])
                        break

            if not img_item:
                logger.warning(f"Image not found in EPUB: {src} (resolved: {resolved_src})")
                continue

            # Get image content
            img_bytes = img_item.get_content()

            # Determine extension from media type
            media_type = img_item.get_type()
            if 'jpeg' in str(media_type) or 'jpg' in str(media_type):
                ext = 'jpg'
            elif 'png' in str(media_type):
                ext = 'png'
            elif 'gif' in str(media_type):
                ext = 'gif'
            elif 'svg' in str(media_type):
                ext = 'svg'
            else:
                # Fallback: use extension from filename
                ext = Path(resolved_src).suffix.lstrip('.') or 'jpg'

            # Save image
            img_filename = f"chapter_{chapter_idx}_img_{img_counter}.{ext}"
            img_path = images_dir / img_filename
            img_path.write_bytes(img_bytes)

            # Update img tag src to new path
            img['src'] = f"../images/{img_filename}"

            img_counter += 1
            logger.debug(f"Extracted image: {src} → {img_filename}")

        except Exception as e:
            logger.warning(f"Failed to extract image {src}: {e}")

    return img_counter


def process_chapter(
    chapter: Dict,
    chapter_idx: int,
    epub_parser: EPUBParser,
    output_dir: Path,
    images_dir: Path
) -> bool:
    """
    Process a single chapter: convert XHTML to Markdown.

    Args:
        chapter: Chapter dictionary from book_structure.json
        chapter_idx: Chapter index (1-based)
        epub_parser: EPUBParser instance
        output_dir: Output directory for markdown
        images_dir: Directory for images

    Returns:
        True if successful
    """
    chapter_title = chapter['title']
    chapter_href = chapter['href']

    logger.info(f"Processing Chapter {chapter_idx}: {chapter_title}")

    try:
        # Get XHTML content from EPUB
        item = epub_parser.get_item_by_href(chapter_href)
        if not item:
            logger.error(f"Chapter file not found: {chapter_href}")
            return False

        html_content = item.get_content().decode('utf-8')

        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove navigation elements
        for nav in soup.find_all('nav'):
            nav.decompose()

        # Extract and save images (modifies soup in-place)
        img_count = extract_and_save_images(
            soup,
            chapter_idx,
            epub_parser,
            images_dir,
            chapter_href
        )

        logger.info(f"Extracted {img_count} images")

        # Convert to Markdown
        markdown = convert_html_to_markdown(
            str(soup),
            preserve_mathml=True
        )

        # Add chapter title if not already present
        if not markdown.startswith('#'):
            markdown = f"# {chapter_title}\n\n{markdown}"

        # Process subchapters (append to same file)
        for subchapter in chapter.get('subchapters', []):
            sub_href = subchapter['href']
            sub_title = subchapter['title']

            # Check if subchapter is in same file or different file
            if sub_href != chapter_href:
                # Different file, need to read and append
                logger.debug(f"Processing subchapter from different file: {sub_href}")
                sub_item = epub_parser.get_item_by_href(sub_href)
                if sub_item:
                    sub_html = sub_item.get_content().decode('utf-8')
                    sub_soup = BeautifulSoup(sub_html, 'lxml')

                    # Remove navigation
                    for nav in sub_soup.find_all('nav'):
                        nav.decompose()

                    # Extract images
                    sub_img_count = extract_and_save_images(
                        sub_soup,
                        chapter_idx,
                        epub_parser,
                        images_dir,
                        sub_href
                    )

                    sub_markdown = convert_html_to_markdown(str(sub_soup))
                    markdown += f"\n\n{sub_markdown}"

        # Save Markdown file
        md_filename = output_dir / f"chapter_{chapter_idx}.md"
        md_filename.write_text(markdown, encoding='utf-8')

        logger.success(f"Saved: {md_filename.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to process chapter {chapter_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_epub_to_markdown(
    epub_path: str,
    structure_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    resume: bool = False
):
    """
    Convert EPUB to Markdown based on book_structure.json.

    Args:
        epub_path: Path to EPUB file
        structure_file: Path to book_structure.json (auto-detected if None)
        output_dir: Output directory (auto-detected if None)
        resume: If True, resume from previous progress
    """
    epub_path = Path(epub_path)

    # Auto-detect structure file and output dir
    if not structure_file or not output_dir:
        # Try to find from epub filename
        book_name = epub_path.stem
        default_output = Path(f"output/{book_name}")

        if not structure_file:
            structure_file = default_output / "book_structure.json"

        if not output_dir:
            output_dir = default_output

    structure_file = Path(structure_file)
    output_dir = Path(output_dir)

    # Check if structure file exists
    if not structure_file.exists():
        logger.error(f"Structure file not found: {structure_file}")
        logger.info("Please run epub_breakdown first to generate book_structure.json")
        return

    # Load structure
    logger.info(f"Loading structure from: {structure_file}")
    with open(structure_file, 'r', encoding='utf-8') as f:
        structure = json.load(f)

    # Create output directories
    # EPUB input produces clean markdown directly, output to polished_markdown
    markdown_dir = output_dir / "polished_markdown"
    images_dir = output_dir / "images"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Progress file
    progress_file = markdown_dir / "epub_progress.json"

    # Load progress if resuming
    if resume:
        progress = load_progress(progress_file)
        logger.info(f"Resuming: {len(progress['chapters_processed'])}/{progress.get('total_chapters', 0)} chapters completed")
    else:
        progress = {"chapters_processed": [], "total_chapters": len(structure['chapters'])}

    # Parse EPUB
    logger.info(f"Loading EPUB: {epub_path}")
    epub_parser = EPUBParser(str(epub_path))

    # Process each chapter
    chapters = structure['chapters']
    progress['total_chapters'] = len(chapters)

    for chapter_idx, chapter in enumerate(chapters, 1):
        # Skip if already processed
        if resume and chapter_idx in progress['chapters_processed']:
            logger.info(f"Skipping Chapter {chapter_idx} (already processed)")
            continue

        # Process chapter
        success = process_chapter(
            chapter,
            chapter_idx,
            epub_parser,
            markdown_dir,
            images_dir
        )

        if success:
            # Update progress
            progress['chapters_processed'].append(chapter_idx)
            save_progress(progress_file, progress)
        else:
            logger.error(f"Failed to process chapter {chapter_idx}, stopping")
            break

    # Summary
    logger.success(f"\nConversion complete!")
    logger.info(f"Processed {len(progress['chapters_processed'])}/{len(chapters)} chapters")
    logger.info(f"Output: {markdown_dir}")
    logger.info(f"Images: {images_dir}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EPUB to Markdown (requires book_structure.json)"
    )
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument(
        "-s", "--structure",
        help="Path to book_structure.json (auto-detected if not specified)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )

    args = parser.parse_args()

    convert_epub_to_markdown(
        args.epub_path,
        structure_file=args.structure,
        output_dir=args.output,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
