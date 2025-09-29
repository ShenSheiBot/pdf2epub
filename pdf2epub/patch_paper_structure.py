#!/usr/bin/env python3
"""
Patch book structure for academic papers and continuous documents.

This script modifies the book_structure.json to treat the entire document
as a single chapter, which is more appropriate for academic papers that
don't have page breaks between sections.
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict
from loguru import logger
from .utils.logging_config import configure_logging

# Configure logger
logger = configure_logging()


def patch_paper_structure(
    book_title: str,
    chapter_name: Optional[str] = None,
    preserve_toc: bool = True
) -> bool:
    """
    Patch the book structure to treat the entire document as a single chapter.

    Args:
        book_title: Title of the book/paper
        chapter_name: Name for the single chapter (default: use book title)
        preserve_toc: Whether to preserve TOC entries as metadata

    Returns:
        True if successful, False otherwise
    """
    # Locate the book structure file
    output_dir = Path("output") / book_title
    structure_file = output_dir / "book_structure.json"

    if not structure_file.exists():
        logger.error(f"Book structure file not found: {structure_file}")
        return False

    # Load the current structure
    logger.info(f"Loading structure from {structure_file}")
    with open(structure_file, "r", encoding="utf-8") as f:
        structure = json.load(f)

    # Create backup
    backup_file = structure_file.with_suffix(".json.backup")
    if not backup_file.exists():
        shutil.copy2(structure_file, backup_file)
        logger.info(f"Original structure backed up to {backup_file}")

    # Determine the page range
    # Find the last page from all chapters
    last_page = 1
    if structure.get("chapters"):
        for chapter in structure["chapters"]:
            end_page = chapter.get("end_page", 1)
            if end_page > last_page:
                last_page = end_page
            # Also check subchapters
            for subchapter in chapter.get("subchapters", []):
                sub_end = subchapter.get("end_page", 1)
                if sub_end > last_page:
                    last_page = sub_end

    logger.info(f"Document spans from page 1 to {last_page}")

    # Determine chapter name
    if not chapter_name:
        # Use the book/paper title as the chapter name
        chapter_name = structure.get("title", book_title)

    # Create the new single-chapter structure
    new_structure = {
        "title": structure.get("title", book_title),
        "author": structure.get("author", ""),
        "cover_page": structure.get("cover_page", {"page_number": 1}),
        "table_of_contents": structure.get("table_of_contents", {
            "start_page": None,
            "end_page": None,
            "entries": []
        }) if preserve_toc else {
            "start_page": None,
            "end_page": None,
            "entries": []
        },
        "chapters": [
            {
                "title": chapter_name,
                "start_page": 1,
                "end_page": last_page,
                "level": 1,
                "subchapters": []  # No subchapters in single-chapter mode
            }
        ]
    }

    # Add metadata about original structure if preserving TOC
    if preserve_toc and structure.get("table_of_contents", {}).get("entries"):
        new_structure["_original_chapters_metadata"] = {
            "note": "Original chapter structure preserved for reference",
            "chapters": structure.get("chapters", [])
        }

    # Log the changes
    original_chapter_count = len(structure.get("chapters", []))
    logger.info(f"Merging {original_chapter_count} chapters into 1 chapter: '{chapter_name}'")

    # Save the patched structure
    with open(structure_file, "w", encoding="utf-8") as f:
        json.dump(new_structure, f, ensure_ascii=False, indent=2)

    logger.success(f"Patched structure saved to {structure_file}")
    logger.info(f"The entire document (pages 1-{last_page}) will now be processed as a single chapter")

    return True


def main():
    """CLI interface for patching paper structure."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Patch book structure for academic papers (single chapter mode)"
    )
    parser.add_argument("book_title", help="Title of the book/paper")
    parser.add_argument(
        "--chapter-name",
        help="Name for the single chapter (default: use document title)"
    )
    parser.add_argument(
        "--no-preserve-toc",
        action="store_true",
        help="Don't preserve original TOC entries as metadata"
    )

    args = parser.parse_args()

    success = patch_paper_structure(
        book_title=args.book_title,
        chapter_name=args.chapter_name,
        preserve_toc=not args.no_preserve_toc
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())