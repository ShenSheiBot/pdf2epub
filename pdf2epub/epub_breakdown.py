"""
EPUB Breakdown Module

Analyzes EPUB structure and generates book_structure.json
compatible with the downstream processing pipeline.

Similar to breakdown.py for PDFs, but adapted for EPUB format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from .epub_input.epub_parser import EPUBParser
from .epub_input.toc_extractor import TOCExtractor, TOCEntry


def toc_entry_to_chapter(
    entry: TOCEntry,
    skip_patterns: Optional[List[str]] = None
) -> Optional[Dict]:
    """
    Convert a TOCEntry to chapter dictionary format.

    Args:
        entry: TOCEntry object
        skip_patterns: Optional list of regex patterns to skip (e.g., [r'^Cover$', r'Copyright'])

    Returns:
        Chapter dictionary or None if should be skipped
    """
    # Skip based on user-provided patterns
    if skip_patterns:
        import re
        for pattern in skip_patterns:
            if re.search(pattern, entry.title, re.IGNORECASE):
                logger.debug(f"Skipping '{entry.title}' (matches pattern: {pattern})")
                return None

    chapter = {
        "title": entry.title,
        "href": entry.href,
        "level": entry.level
    }

    # Add anchor if present
    if entry.anchor:
        chapter["anchor"] = entry.anchor

    # Recursively convert children to subchapters
    subchapters = []
    for child in entry.children:
        subchapter = toc_entry_to_chapter(child, skip_patterns=None)  # Only apply to top level
        if subchapter:
            subchapters.append(subchapter)

    chapter["subchapters"] = subchapters

    return chapter


def identify_front_matter_by_position(
    toc_entries: List[TOCEntry],
    spine_order: List[str],
    max_position: int = 5
) -> List[Dict]:
    """
    Identify front matter by position in spine (more robust than keywords).

    Front matter is typically the first N items in the book.

    Args:
        toc_entries: List of TOCEntry objects
        spine_order: List of hrefs in spine order
        max_position: Consider first N spine items as potential front matter

    Returns:
        List of front matter items
    """
    front_matter = []

    # Build href to position mapping
    href_positions = {href: idx for idx, href in enumerate(spine_order)}

    for entry in toc_entries:
        # Get position in spine
        position = href_positions.get(entry.href, 999)

        # If in first N items, likely front matter
        if position < max_position:
            front_matter.append({
                "title": entry.title,
                "href": entry.href,
                "anchor": entry.anchor,
                "spine_position": position
            })

    return front_matter


def identify_back_matter_by_position(
    toc_entries: List[TOCEntry],
    spine_order: List[str],
    min_position_from_end: int = 3
) -> List[Dict]:
    """
    Identify back matter by position (last N items).

    Args:
        toc_entries: List of TOCEntry objects
        spine_order: List of hrefs in spine order
        min_position_from_end: Consider last N items as potential back matter

    Returns:
        List of back matter items
    """
    back_matter = []

    # Build href to position mapping
    href_positions = {href: idx for idx, href in enumerate(spine_order)}
    total_items = len(spine_order)
    threshold = total_items - min_position_from_end

    for entry in toc_entries:
        position = href_positions.get(entry.href, -1)

        # If in last N items, likely back matter
        if position >= threshold and position != -1:
            back_matter.append({
                "title": entry.title,
                "href": entry.href,
                "anchor": entry.anchor,
                "spine_position": position
            })

    return back_matter


def breakdown_epub(
    epub_path: str,
    output_dir: str,
    skip_patterns: Optional[List[str]] = None,
    front_matter_position_threshold: int = 5,
    back_matter_position_threshold: int = 3
) -> Dict:
    """
    Analyze EPUB structure and generate book_structure.json.

    Args:
        epub_path: Path to EPUB file
        output_dir: Output directory for book_structure.json
        skip_patterns: Optional list of regex patterns to skip chapters
                      (e.g., [r'^Cover', r'Copyright', r'Table.+Contents'])
        front_matter_position_threshold: First N spine items considered as front matter
        back_matter_position_threshold: Last N spine items considered as back matter

    Returns:
        Structure dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing EPUB structure: {epub_path}")

    # Parse EPUB
    parser = EPUBParser(epub_path)

    # Extract metadata
    metadata = parser.metadata
    logger.info(f"Book: {metadata['title']} by {metadata['author']}")

    # Extract TOC
    toc_extractor = TOCExtractor(parser.book)
    toc_entries = toc_extractor.toc
    toc_summary = toc_extractor.get_summary()

    logger.info(
        f"TOC: {toc_summary['total_entries']} entries, "
        f"max depth: {toc_summary['max_depth']}"
    )

    # Build structure
    structure = {
        "author": metadata['author'],
        "book_title": metadata['title'],
        "language": metadata['language'],
        "source_type": "epub"
    }

    # Add publisher if available
    if metadata.get('publisher'):
        structure["publisher"] = metadata['publisher']

    # Find cover
    cover = parser.extract_cover_image()

    # Look for cover page in spine (by href name or position)
    for item in parser.spine:
        if 'cover' in item['href'].lower():
            structure["cover_page"] = {"href": item['href']}
            logger.debug(f"Found cover page by href: {item['href']}")
            break

    # If no cover found by name, check first item in TOC
    if "cover_page" not in structure and toc_entries:
        first_entry = toc_entries[0]
        if any(keyword in first_entry.title.lower() for keyword in ['cover', 'couverture', '封面', '表紙']):
            structure["cover_page"] = {"href": first_entry.href}
            logger.debug(f"Found cover page by TOC title: {first_entry.title}")

    # Fallback: use first spine item if it's not obviously content
    if "cover_page" not in structure and parser.spine:
        first_item = parser.spine[0]
        # Only use as cover if it looks like a cover page
        if len(parser.spine) > 1:  # Don't use if it's the only page
            structure["cover_page"] = {"href": first_item['href']}
            logger.debug(f"Using first spine item as cover: {first_item['href']}")

    # Get spine order for position-based identification
    spine_hrefs = [item['href'] for item in parser.spine]

    # Identify front matter and back matter by position (robust)
    flat_toc = toc_extractor.get_flat_toc()
    all_entries = [TOCEntry(
        title=item['title'],
        href=item['href'],
        level=item['level'],
        anchor=item.get('anchor')
    ) for item in flat_toc]

    front_matter_items = identify_front_matter_by_position(
        all_entries,
        spine_hrefs,
        max_position=front_matter_position_threshold
    )
    back_matter_items = identify_back_matter_by_position(
        all_entries,
        spine_hrefs,
        min_position_from_end=back_matter_position_threshold
    )

    if front_matter_items:
        structure["front_matter"] = {"items": front_matter_items}
        logger.info(f"Identified {len(front_matter_items)} front matter items")

    if back_matter_items:
        structure["back_matter"] = {"items": back_matter_items}
        logger.info(f"Identified {len(back_matter_items)} back matter items")

    # Build table of contents
    structure["table_of_contents"] = {
        "entries": flat_toc
    }

    # Convert TOC entries to chapters
    # Only top-level (level 1) entries become chapters
    # Apply skip_patterns only to top-level entries
    chapters = []
    for entry in toc_entries:
        if entry.level == 1:
            chapter = toc_entry_to_chapter(entry, skip_patterns=skip_patterns)
            if chapter:
                chapters.append(chapter)

    structure["chapters"] = chapters

    logger.info(f"Generated {len(chapters)} chapters")

    # Count subchapters
    total_subchapters = sum(
        len(ch.get('subchapters', []))
        for ch in chapters
    )
    logger.info(f"Total subchapters: {total_subchapters}")

    # Find back cover (last spine item or last image)
    if parser.spine:
        last_item = parser.spine[-1]
        if 'back' in last_item['href'].lower() or 'cover' in last_item['href'].lower():
            structure["back_cover"] = {"href": last_item['href']}

    # Save structure
    output_file = output_path / "book_structure.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    logger.success(f"Book structure saved to: {output_file}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Structure Summary")
    logger.info("="*60)
    logger.info(f"Author: {structure['author']}")
    logger.info(f"Title: {structure['book_title']}")
    logger.info(f"Language: {structure['language']}")
    if structure.get('publisher'):
        logger.info(f"Publisher: {structure['publisher']}")
    logger.info(f"Chapters: {len(chapters)}")
    logger.info(f"Subchapters: {total_subchapters}")
    logger.info(f"TOC entries: {toc_summary['total_entries']}")
    logger.info(f"Max depth: {toc_summary['max_depth']}")
    logger.info("="*60)

    return structure


def main():
    """CLI entry point for EPUB breakdown."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze EPUB structure and generate book_structure.json"
    )
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: output/<book_title>)"
    )
    parser.add_argument(
        "--skip-pattern",
        action="append",
        dest="skip_patterns",
        help="Regex pattern to skip chapters (can be used multiple times). "
             "Example: --skip-pattern '^Cover' --skip-pattern 'Copyright'"
    )
    parser.add_argument(
        "--front-matter-threshold",
        type=int,
        default=5,
        help="First N spine items to mark as front matter (default: 5)"
    )
    parser.add_argument(
        "--back-matter-threshold",
        type=int,
        default=3,
        help="Last N spine items to mark as back matter (default: 3)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Extract book title from EPUB for default output path
        temp_parser = EPUBParser(args.epub_path)
        book_title = temp_parser.metadata['title']
        # Clean title for directory name
        import re
        clean_title = re.sub(r'[^\w\s-]', '', book_title)
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        output_dir = f"output/{clean_title}"

    # Run breakdown
    breakdown_epub(
        args.epub_path,
        output_dir,
        skip_patterns=args.skip_patterns,
        front_matter_position_threshold=args.front_matter_threshold,
        back_matter_position_threshold=args.back_matter_threshold
    )


if __name__ == "__main__":
    main()
