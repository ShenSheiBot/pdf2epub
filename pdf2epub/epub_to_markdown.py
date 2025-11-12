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
import tiktoken
import yaml

from .epub_input.epub_parser import EPUBParser
from .epub_input.custom_converters import convert_html_to_markdown
from .epub_input.semantic_enrichment import SemanticEnricher, semantic_pipeline
from .processors.utils.content_splitter import split_content
from .utils.llm_client import LLMClient

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    images_dir: Path,
    llm_client=None,
    max_tokens_per_part: int = 8000,
    semantic_enricher: Optional[SemanticEnricher] = None,
    use_semantic_pipeline: bool = True
) -> bool:
    """
    Process a single chapter: convert XHTML to Markdown.

    Automatically splits long chapters into parts if they exceed max_tokens_per_part.

    Pipeline:
    1. Playwright: Inject semantic headings based on CSS styles
    2. Trafilatura: Clean and linearize HTML
    3. Markdownify: Convert to Markdown

    Args:
        chapter: Chapter dictionary from book_structure.json
        chapter_idx: Chapter index (1-based)
        epub_parser: EPUBParser instance
        output_dir: Output directory for markdown
        images_dir: Directory for images
        llm_client: LLM client for intelligent splitting (optional)
        max_tokens_per_part: Maximum tokens per part (default: 8000)
        semantic_enricher: Reusable SemanticEnricher instance (for batch processing)
        use_semantic_pipeline: Enable Playwright+Trafilatura pipeline (default: True)

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

        # **STAGE 1 & 2: Semantic enrichment pipeline**
        # Playwright: Inject semantic headings based on visual styles
        # Trafilatura: Clean and linearize HTML
        if use_semantic_pipeline:
            logger.debug(f"Applying semantic pipeline to {chapter_href}")
            html_content = semantic_pipeline(
                html_content,
                use_playwright=True,
                use_trafilatura=False,  # Disabled: Trafilatura destroys semantic headings
                enricher=semantic_enricher
            )

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

        # **STAGE 3: Convert to Markdown**
        # Markdownify will now recognize semantic <h2>, <h3> tags
        markdown = convert_html_to_markdown(
            str(soup),
            preserve_mathml=True
        )

        # Add chapter title if not already present
        if not markdown.startswith('#'):
            markdown = f"# {chapter_title}\n\n{markdown}"

        # Recursive function to process all levels of subchapters
        def process_subchapters_recursive(subchapters_list, parent_href):
            """Recursively process subchapters at all levels.

            With semantic pipeline enabled, headings are auto-detected from HTML styles.
            We don't manually add markdown headings - they come from semantic enrichment.

            Args:
                subchapters_list: List of subchapter dictionaries
                parent_href: Parent file href for path resolution
            """
            result_markdown = ""
            for subchapter in subchapters_list:
                sub_href = subchapter['href']
                sub_title = subchapter['title']

                # Check if subchapter is in same file or different file
                if sub_href != parent_href:
                    # Different file, need to read and append
                    logger.debug(f"Processing subchapter from different file: {sub_href}")
                    sub_item = epub_parser.get_item_by_href(sub_href)
                    if sub_item:
                        sub_html = sub_item.get_content().decode('utf-8')

                        # Apply semantic pipeline to subchapter
                        if use_semantic_pipeline:
                            logger.debug(f"Applying semantic pipeline to subchapter {sub_href}")
                            sub_html = semantic_pipeline(
                                sub_html,
                                use_playwright=True,
                                use_trafilatura=False,  # Disabled: Trafilatura destroys semantic headings
                                enricher=semantic_enricher
                            )

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

                        # Convert to markdown (headings already semantic from pipeline)
                        sub_markdown = convert_html_to_markdown(str(sub_soup))
                        result_markdown += f"\n\n{sub_markdown}"

                # Recursively process nested subchapters
                if 'subchapters' in subchapter and subchapter['subchapters']:
                    nested_markdown = process_subchapters_recursive(subchapter['subchapters'], sub_href)
                    result_markdown += nested_markdown

            return result_markdown

        # Process subchapters (append to same file) - now recursively with semantic pipeline
        subchapters_markdown = process_subchapters_recursive(chapter.get('subchapters', []), chapter_href)
        markdown += subchapters_markdown

        # Check if chapter needs to be split
        actual_tokens = len(tokenizer.encode(markdown))

        if actual_tokens > max_tokens_per_part:
            logger.info(f"Chapter {chapter_idx} has {actual_tokens:,} tokens (max: {max_tokens_per_part:,}), splitting into parts")

            # Split content using intelligent splitter if llm_client available
            if llm_client:
                # Auto-detect content type for splitting strategy
                import re
                japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', markdown[:5000])
                if len(japanese_chars) > 500:
                    strategy = "japanese"
                else:
                    # Check for academic indicators
                    academic_indicators = [r'\[\^\d+\]', r'References\s*\n', r'Bibliography\s*\n']
                    if any(re.search(pattern, markdown[:5000]) for pattern in academic_indicators):
                        strategy = "academic"
                    else:
                        strategy = "general"

                logger.info(f"Using '{strategy}' splitting strategy")
                parts = split_content(markdown, max_tokens_per_part, llm_client, strategy=strategy)
            else:
                # Fallback: simple splitting by section
                from .processors.utils.content_splitter import SimpleSplitter
                parts = SimpleSplitter().split(markdown, max_tokens_per_part)

            # Save each part
            for part_idx, part_content in enumerate(parts, 1):
                part_tokens = len(tokenizer.encode(part_content))
                part_filename = output_dir / f"chapter_{chapter_idx}.part{part_idx}.md"
                part_filename.write_text(part_content, encoding='utf-8')
                logger.success(f"Saved: {part_filename.name} ({part_tokens:,} tokens)")

            logger.info(f"Split chapter {chapter_idx} into {len(parts)} parts")
        else:
            # Save as single file
            md_filename = output_dir / f"chapter_{chapter_idx}.md"
            md_filename.write_text(markdown, encoding='utf-8')
            logger.success(f"Saved: {md_filename.name} ({actual_tokens:,} tokens)")

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
    resume: bool = False,
    config_path: str = "config.yaml",
    max_tokens_per_part: int = 8000
):
    """
    Convert EPUB to Markdown based on book_structure.json.

    Automatically splits long chapters into parts if they exceed max_tokens_per_part.

    Args:
        epub_path: Path to EPUB file
        structure_file: Path to book_structure.json (auto-detected if None)
        output_dir: Output directory (auto-detected if None)
        resume: If True, resume from previous progress
        config_path: Path to config file (default: config.yaml)
        max_tokens_per_part: Maximum tokens per part for splitting (default: 8000)
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

    # Load config and create LLM client for intelligent splitting
    llm_client = None
    try:
        config = load_config(config_path)
        llm_client = LLMClient(config)
        logger.info("LLM client initialized for intelligent chapter splitting")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM client: {e}. Will use simple splitting if needed.")

    # Parse EPUB
    logger.info(f"Loading EPUB: {epub_path}")
    epub_parser = EPUBParser(str(epub_path))

    # Initialize semantic enricher (reuse browser for all chapters)
    logger.info("Initializing semantic enrichment pipeline (Playwright + Trafilatura)")
    with SemanticEnricher(headless=True) as semantic_enricher:
        # Process each chapter
        chapters = structure['chapters']
        progress['total_chapters'] = len(chapters)

        for chapter_idx, chapter in enumerate(chapters, 1):
            # Skip if already processed
            if resume and chapter_idx in progress['chapters_processed']:
                logger.info(f"Skipping Chapter {chapter_idx} (already processed)")
                continue

            # Process chapter with semantic enrichment
            success = process_chapter(
                chapter,
                chapter_idx,
                epub_parser,
                markdown_dir,
                images_dir,
                llm_client=llm_client,
                max_tokens_per_part=max_tokens_per_part,
                semantic_enricher=semantic_enricher,  # Reuse enricher
                use_semantic_pipeline=True
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
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens per part for splitting long chapters (default: 8000)"
    )

    args = parser.parse_args()

    convert_epub_to_markdown(
        args.epub_path,
        structure_file=args.structure,
        output_dir=args.output,
        resume=args.resume,
        config_path=args.config,
        max_tokens_per_part=args.max_tokens
    )


if __name__ == "__main__":
    main()
