"""
Book structure validator and fixer for PDF to EPUB conversion.

This module provides functions to validate and fix book structure JSON files,
handling overlapping page ranges and missing pages.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger


def resolve_overlaps(structure: Dict) -> Dict:
    """
    Fix overlapping page ranges in book structure.
    
    When a chapter has subchapters, ensures that:
    1. Subchapters don't overlap with each other
    2. The parent chapter's range is adjusted to only cover pages not in subchapters
    
    Args:
        structure: Book structure dictionary
        
    Returns:
        Fixed book structure with no overlapping page ranges
    """
    if "chapters" not in structure:
        return structure
    
    fixed_chapters = []
    
    for chapter in structure["chapters"]:
        fixed_chapter = chapter.copy()
        
        if "subchapters" in chapter and chapter["subchapters"]:
            # First, identify which subchapters are actually subsections of others
            # (i.e., their page range is completely within another subchapter)
            subchapters = chapter["subchapters"]
            main_subchapters = []
            subsection_map = {}  # Maps main subchapter index to its subsections
            
            for i, sub in enumerate(subchapters):
                is_subsection = False
                for j, other in enumerate(subchapters):
                    if i != j and sub["start_page"] >= other["start_page"] and sub["end_page"] <= other["end_page"]:
                        # This is a subsection of 'other'
                        is_subsection = True
                        if j not in subsection_map:
                            subsection_map[j] = []
                        subsection_map[j].append(sub)
                        break
                
                if not is_subsection:
                    main_subchapters.append((i, sub))
            
            # Sort main subchapters by start page
            main_subchapters.sort(key=lambda x: x[1]["start_page"])
            fixed_subchapters = []
            
            # Process only main subchapters for overlap fixing
            for idx, (orig_idx, subchapter) in enumerate(main_subchapters):
                fixed_sub = subchapter.copy()
                
                # Check if this subchapter overlaps with the next main subchapter
                if idx < len(main_subchapters) - 1:
                    next_start = main_subchapters[idx + 1][1]["start_page"]
                    if fixed_sub["end_page"] >= next_start:
                        # Adjust end page to avoid overlap
                        fixed_sub["end_page"] = next_start - 1
                        logger.warning(
                            f"Adjusted subchapter '{fixed_sub['title']}' end page from "
                            f"{subchapter['end_page']} to {fixed_sub['end_page']} to avoid overlap"
                        )
                
                # Ensure subchapter doesn't extend beyond parent chapter
                if fixed_sub["end_page"] > chapter["end_page"]:
                    fixed_sub["end_page"] = chapter["end_page"]
                    logger.warning(
                        f"Adjusted subchapter '{fixed_sub['title']}' to not exceed parent chapter"
                    )
                
                fixed_subchapters.append(fixed_sub)
                
                # Add any subsections of this subchapter
                if orig_idx in subsection_map:
                    for subsection in subsection_map[orig_idx]:
                        # Ensure subsection doesn't exceed the adjusted parent range
                        fixed_subsection = subsection.copy()
                        if fixed_subsection["end_page"] > fixed_sub["end_page"]:
                            fixed_subsection["end_page"] = fixed_sub["end_page"]
                        fixed_subchapters.append(fixed_subsection)
            
            fixed_chapter["subchapters"] = fixed_subchapters
            
            # Adjust parent chapter to not include subchapter pages
            # The parent chapter should only contain intro pages before first subchapter
            if fixed_subchapters:
                first_subchapter_start = fixed_subchapters[0]["start_page"]
                if chapter["start_page"] < first_subchapter_start:
                    # There are intro pages before the first subchapter
                    # Keep the chapter but adjust its range
                    fixed_chapter["has_intro"] = True
                    fixed_chapter["intro_end_page"] = first_subchapter_start - 1
                else:
                    # No intro pages, the chapter is purely organizational
                    fixed_chapter["has_intro"] = False
        
        fixed_chapters.append(fixed_chapter)
    
    structure["chapters"] = fixed_chapters
    return structure


def find_missing_pages(structure: Dict, total_pages: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Find page ranges that are not covered by any chapter.
    
    Args:
        structure: Book structure dictionary
        total_pages: Total number of pages in the PDF (if known)
        
    Returns:
        List of (start_page, end_page) tuples for missing page ranges
    """
    # Collect all covered page ranges
    covered_ranges = []
    
    # Add front matter if exists
    if "front_matter" in structure:
        fm = structure["front_matter"]
        covered_ranges.append((fm["start_page"], fm["end_page"]))
    
    # Add all chapter and subchapter ranges
    for chapter in structure.get("chapters", []):
        # For chapters with subchapters, only count the subchapter pages
        if "subchapters" in chapter and chapter["subchapters"]:
            # If chapter has intro pages, add those
            if chapter.get("has_intro", False):
                intro_end = chapter.get("intro_end_page")
                if intro_end:
                    covered_ranges.append((chapter["start_page"], intro_end))
            
            # Add all subchapter ranges
            for subchapter in chapter["subchapters"]:
                covered_ranges.append((subchapter["start_page"], subchapter["end_page"]))
        else:
            # No subchapters, use the full chapter range
            covered_ranges.append((chapter["start_page"], chapter["end_page"]))
    
    # Add back matter if exists
    if "back_matter" in structure:
        bm = structure["back_matter"]
        covered_ranges.append((bm["start_page"], bm["end_page"]))
    
    # Sort ranges by start page
    covered_ranges.sort(key=lambda x: x[0])
    
    # Merge overlapping ranges
    merged_ranges = []
    for start, end in covered_ranges:
        if merged_ranges and start <= merged_ranges[-1][1] + 1:
            # Overlapping or adjacent, merge
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))
    
    # Find gaps
    missing_ranges = []
    
    # Check for gap before first covered range
    if merged_ranges and merged_ranges[0][0] > 1:
        missing_ranges.append((1, merged_ranges[0][0] - 1))
    
    # Check for gaps between covered ranges
    for i in range(len(merged_ranges) - 1):
        gap_start = merged_ranges[i][1] + 1
        gap_end = merged_ranges[i + 1][0] - 1
        if gap_start <= gap_end:
            missing_ranges.append((gap_start, gap_end))
    
    # Check for gap after last covered range (if total_pages is known)
    if total_pages and merged_ranges:
        last_covered = merged_ranges[-1][1]
        if last_covered < total_pages:
            missing_ranges.append((last_covered + 1, total_pages))
    
    return missing_ranges


def add_missing_pages_as_chapters(structure: Dict, total_pages: Optional[int] = None) -> Dict:
    """
    Add chapters for any missing page ranges.
    
    Args:
        structure: Book structure dictionary
        total_pages: Total number of pages in the PDF
        
    Returns:
        Updated structure with new chapters for missing pages
    """
    missing_ranges = find_missing_pages(structure, total_pages)
    
    if not missing_ranges:
        logger.info("No missing pages found in book structure")
        return structure
    
    # Ensure chapters key exists
    if "chapters" not in structure:
        structure["chapters"] = []
    
    # Determine where to insert the missing page chapters
    chapters = structure.get("chapters", [])
    
    for start, end in missing_ranges:
        # Find the appropriate position to insert this chapter
        insert_position = 0
        for i, chapter in enumerate(chapters):
            if chapter["start_page"] > start:
                insert_position = i
                break
            insert_position = i + 1
        
        # Create a new chapter for the missing pages
        new_chapter = {
            "title": f"Additional Content (Pages {start}-{end})",
            "start_page": start,
            "end_page": end,
            "subchapters": []
        }
        
        logger.info(f"Adding chapter for missing pages {start}-{end}")
        chapters.insert(insert_position, new_chapter)
    
    structure["chapters"] = chapters
    return structure


def validate_structure(structure: Dict, fix: bool = True, total_pages: Optional[int] = None) -> Dict:
    """
    Validate and optionally fix a book structure.
    
    Args:
        structure: Book structure dictionary
        fix: Whether to fix issues found (default: True)
        total_pages: Total number of pages in the PDF
        
    Returns:
        Validated (and possibly fixed) structure
    """
    issues = []
    
    # Check for basic structure
    if "chapters" not in structure:
        issues.append("No chapters found in structure")
        if not fix:
            logger.error("Structure validation failed: " + ", ".join(issues))
            return structure
    
    # Check for overlaps
    original_structure = json.dumps(structure, sort_keys=True)
    
    if fix:
        # Fix overlapping ranges
        structure = resolve_overlaps(structure)
        
        # Add missing pages as chapters
        structure = add_missing_pages_as_chapters(structure, total_pages)
        
        # Check if structure was modified
        if json.dumps(structure, sort_keys=True) != original_structure:
            logger.info("Book structure has been fixed and validated")
        else:
            logger.info("Book structure validated, no fixes needed")
    else:
        # Just report issues without fixing
        missing_ranges = find_missing_pages(structure, total_pages)
        if missing_ranges:
            for start, end in missing_ranges:
                issues.append(f"Missing pages {start}-{end}")
        
        if issues:
            logger.warning("Structure validation issues: " + ", ".join(issues))
        else:
            logger.info("Book structure validated successfully")
    
    return structure


def validate_book_structure_file(
    book_title: str,
    output_dir: Optional[Path] = None,
    fix: bool = True,
    total_pages: Optional[int] = None
) -> bool:
    """
    Validate and fix a book structure JSON file.
    
    Args:
        book_title: Title of the book
        output_dir: Output directory (default: output/{book_title})
        fix: Whether to fix issues found
        total_pages: Total number of pages in the PDF
        
    Returns:
        True if validation/fixing succeeded
    """
    if output_dir is None:
        output_dir = Path("output") / book_title
    
    structure_file = output_dir / "book_structure.json"
    
    if not structure_file.exists():
        logger.error(f"Book structure file not found: {structure_file}")
        return False
    
    # Load the structure
    try:
        with open(structure_file, "r", encoding="utf-8") as f:
            structure = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load book structure: {e}")
        return False
    
    # If total_pages not provided, try to get it from the PDF
    if total_pages is None and fix:
        pdf_path = output_dir / "input.pdf"
        if not pdf_path.exists():
            pdf_path = output_dir / "input_original.pdf"
        
        if pdf_path.exists():
            try:
                import fitz
                with fitz.open(pdf_path) as pdf:
                    total_pages = len(pdf)
                    logger.info(f"Detected {total_pages} pages in PDF")
            except Exception as e:
                logger.warning(f"Could not determine total pages from PDF: {e}")
    
    # Validate and potentially fix the structure
    original_structure = json.dumps(structure, sort_keys=True)
    fixed_structure = validate_structure(structure, fix=fix, total_pages=total_pages)
    
    # Save the fixed structure if it was modified
    if fix and json.dumps(fixed_structure, sort_keys=True) != original_structure:
        # Backup the original
        backup_file = structure_file.with_suffix(".json.backup")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        logger.info(f"Original structure backed up to {backup_file}")
        
        # Save the fixed structure
        with open(structure_file, "w", encoding="utf-8") as f:
            json.dump(fixed_structure, f, ensure_ascii=False, indent=2)
        logger.success(f"Fixed structure saved to {structure_file}")
    
    return True


def main():
    """CLI interface for structure validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and fix book structure JSON files")
    parser.add_argument("book_title", help="Title of the book")
    parser.add_argument("--no-fix", action="store_true", help="Only validate, don't fix issues")
    parser.add_argument("--total-pages", type=int, help="Total number of pages in the PDF")
    
    args = parser.parse_args()
    
    success = validate_book_structure_file(
        args.book_title,
        fix=not args.no_fix,
        total_pages=args.total_pages
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())