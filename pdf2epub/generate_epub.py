#!/usr/bin/env python3
"""
================================================================================
⚠️  DEPRECATED MODULE / 已弃用模块
================================================================================
This module is part of the LEGACY workflow and may be removed in future versions.
此模块属于旧版工作流，可能会在未来版本中移除。

RECOMMENDED command / 推荐的新命令:
    pdf2epub build-epub [--translated]

This module (generate_epub.py) generates EPUB from book_structure.json.
The new 'build-epub' command uses toc_tree.json which supports hierarchical
TOC structures with unlimited nesting levels.
================================================================================

Refactored EPUB generation orchestrator.

This module serves as the high-level orchestrator for EPUB generation,
coordinating the ContentConverter and EpubBuilder components.
"""

import json
import re
import argparse
from pathlib import Path
from loguru import logger

from .epub import EpubConfig, ContentConverter, EpubBuilder
from .epub.footnotes import FootnoteManager
from .utils.logging_config import configure_logging
from .utils.common import load_config, load_book_structure, ensure_directory
from .utils.zip_utils import create_password_protected_zip
from .utils.llm_client import LLMClient
from .chapter_identity import ChapterIdentity

# Configure logger
logger = configure_logging()


def build_subchapter_locations(markdown_dir, parts_info):
    """Build mapping of which subchapters are in which parts with fuzzy matching."""
    import difflib
    import re

    subchapter_locations = {}

    for chapter_index, parts in parts_info.items():
        if parts > 1:
            # This chapter has multiple parts
            for part_num in range(1, parts + 1):
                # Use ChapterIdentity for consistent naming
                part_name = ChapterIdentity.make_part_name(f"chapter_{chapter_index}", part_num)
                part_file = markdown_dir / f"{part_name}.md"
                if part_file.exists():
                    with open(part_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # Find all subchapter headings in this part
                    # We need to identify subchapters, which are:
                    # 1. Level 2 headings (##)
                    # 2. Level 1 headings that are NOT the main chapter title
                    #    (e.g., "2 高强度市场中的媒体分析" or "3 理想的商品？")

                    for line in lines:
                        heading = None

                        # Check for level 1 headings that look like numbered subchapters
                        if line.strip().startswith("#") and not line.strip().startswith("##"):
                            # Extract the heading text
                            heading_text = re.sub(r"^#\s*", "", line).strip()

                            # Check if this looks like a numbered subchapter (starts with a number)
                            # or is clearly not a main chapter title
                            if re.match(r"^\d+\s+", heading_text):  # Starts with number
                                heading = heading_text
                            elif "第" not in heading_text and "部分" not in heading_text:
                                # Not a main part/chapter title (like "第一部分")
                                heading = heading_text

                        # Also check for level 2 headings
                        elif line.strip().startswith("##") and not line.strip().startswith("###"):
                            # Extract the heading text
                            heading = re.sub(r"^##\s*", "", line).strip()

                        if heading:
                            # Store which part this subchapter is in
                            location_key = f"{chapter_index}:{heading}"
                            subchapter_locations[location_key] = {
                                "chapter": chapter_index,
                                "part": part_num,
                                "title": heading,
                                "exact_title": heading,  # Store the exact title from markdown
                            }

    # Add fuzzy matching capability by storing all titles for lookup
    subchapter_locations["_all_titles"] = {}
    for key, info in subchapter_locations.items():
        if key != "_all_titles" and isinstance(info, dict):
            chapter = info.get("chapter")
            if chapter not in subchapter_locations["_all_titles"]:
                subchapter_locations["_all_titles"][chapter] = []
            subchapter_locations["_all_titles"][chapter].append(
                {"title": info["title"], "part": info["part"], "key": key}
            )

    return subchapter_locations


def build_structure_from_markdown(markdown_dir: Path, book_title: str) -> dict:
    """
    Build book structure by extracting headings from markdown files.

    Args:
        markdown_dir: Directory containing markdown files
        book_title: Title of the book

    Returns:
        Dictionary with book structure based on markdown headings
    """
    structure = {"book_title": book_title, "chapters": []}

    # Find all chapter files (including part files)
    chapter_files = {}

    # First collect all files by chapter number using ChapterIdentity
    for md_file in sorted(markdown_dir.glob("*.md")):
        identity = ChapterIdentity.parse(md_file.name)
        if identity and identity.number is not None:
            chapter_num = identity.number

            if chapter_num not in chapter_files:
                chapter_files[chapter_num] = {}

            if identity.part is not None:
                chapter_files[chapter_num][identity.part] = md_file
            else:
                chapter_files[chapter_num]["main"] = md_file

    # Now process each chapter (sort using hierarchical index for proper ordering)
    def chapter_sort_key(num):
        """Sort key for hierarchical chapter numbers like '7', '7.1', '7.1.1'."""
        # Filter out empty strings from split (handles trailing dots like "7.")
        return [int(x) for x in num.split('.') if x]

    for chapter_num in sorted(chapter_files.keys(), key=chapter_sort_key):
        files = chapter_files[chapter_num]

        # Determine which files to read
        files_to_read = []
        if "main" in files and len(files) == 1:
            # Single file chapter
            files_to_read = [files["main"]]
        else:
            # Multi-part chapter - read parts in order
            for part_num in sorted([k for k in files.keys() if isinstance(k, int)]):
                files_to_read.append(files[part_num])

        # Extract headings from files
        chapter_title = f"Chapter {chapter_num}"
        subchapters = []
        first_h1_found = False  # Track if we've seen the first level 1 heading

        for file_path in files_to_read:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Extract headings using regex
                    lines = content.split("\n")
                    for line in lines:
                        line = line.strip()

                        # Check for level 1 heading
                        if line.startswith("# ") and not line.startswith("## "):
                            # Extract title without the # marker
                            title = line[2:].strip()
                            if title:  # Only process if not empty
                                if not first_h1_found:
                                    # First level 1 heading becomes the chapter title
                                    chapter_title = title
                                    first_h1_found = True
                                else:
                                    # Subsequent level 1 headings become subchapters
                                    subchapters.append({"title": title})

                        # Check for level 2 heading
                        elif line.startswith("## ") and not line.startswith("### "):
                            # Extract subtitle without the ## marker
                            subtitle = line[3:].strip()

                            # Check if it's a numbered section (like "2.3 Section Title")
                            starts_with_number = subtitle and subtitle[0].isdigit()
                            ends_with_period = subtitle and subtitle.endswith('.')
                            is_numbered_section = starts_with_number and not ends_with_period

                            # Include if not empty AND (not too long OR is a numbered section)
                            if subtitle and (len(subtitle) <= 50 or is_numbered_section):
                                subchapters.append({"title": subtitle})

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        # Add chapter to structure
        chapter_info = {"title": chapter_title, "index": chapter_num}

        if subchapters:
            chapter_info["subchapters"] = subchapters

        structure["chapters"].append(chapter_info)

    # Fallback for chapters with default titles

    # Also check for front_matter and back_matter
    if (markdown_dir / "front_matter.md").exists():
        structure["front_matter"] = {"title": "Front Matter"}

    if (markdown_dir / "back_matter.md").exists():
        structure["back_matter"] = {"title": "Back Matter"}

    logger.info(
        f"Built structure from markdown: {len(structure['chapters'])} chapters found"
    )
    return structure


def load_or_create_translated_book_structure(
    book_title: str, original_structure: dict, config: dict
) -> dict:
    """
    Load or create a translated version of the book structure.

    First checks if a translated structure already exists. If not, translates
    the original structure using the LLM and saves it.

    Args:
        book_title: Title of the book
        original_structure: The original book structure dict
        config: Configuration dict with API keys

    Returns:
        Translated book structure dict
    """
    # Check if translated structure already exists
    translated_structure_path = (
        Path("output") / book_title / "book_structure_translated.json"
    )

    if translated_structure_path.exists():
        logger.info(
            f"Loading existing translated book structure from {translated_structure_path}"
        )
        with open(translated_structure_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Get translation settings from translation_progress.json if available
    translation_progress_path = (
        Path("output") / book_title / "translated" / "translation_progress.json"
    )
    source_language = "Japanese"  # Default
    target_language = "Chinese"  # Default

    if translation_progress_path.exists():
        with open(translation_progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
            source_language = progress.get("source_language", source_language)
            target_language = progress.get("target_language", target_language)
            logger.info(
                f"Using translation settings: {source_language} → {target_language}"
            )

    # Translate the structure using LLM
    logger.info("Translating book structure using LLM...")

    # Initialize LLM client
    llm_client = LLMClient(config)

    # Create translation prompt
    prompt = f"""Translate the following book structure from {source_language} to {target_language}.

IMPORTANT:
1. Translate ALL text fields including: book_title, title, entries[].title, chapters[].title, chapters[].subchapters[].title
2. Keep all other fields (page numbers, structure) exactly the same
3. Maintain the exact JSON structure
4. Return ONLY the translated JSON, no explanations

Book structure to translate:"""

    # Convert structure to JSON string
    structure_json = json.dumps(original_structure, ensure_ascii=False, indent=2)

    # Create multi-part content for the LLM
    multi_part_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": structure_json},
    ]

    # Use same models as translation
    translation_models = config.get(
        "translation_models",
        [
            {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2},
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250929",
                "max_retries": 2,
            },
        ],
    )

    try:
        # Generate translation
        translated_json = llm_client.generate(
            prompt=multi_part_content,
            model_configs=translation_models,
            operation_name="Translate book structure",
        )

        # Clean response (remove markdown code blocks if present)
        if "```json" in translated_json:
            translated_json = translated_json.split("```json")[1].split("```")[0]
        elif "```" in translated_json:
            translated_json = translated_json.split("```")[1].split("```")[0]

        # Parse the translated JSON
        translated_structure = json.loads(translated_json.strip())

        # Save the translated structure
        translated_structure_path.parent.mkdir(parents=True, exist_ok=True)
        with open(translated_structure_path, "w", encoding="utf-8") as f:
            json.dump(translated_structure, f, ensure_ascii=False, indent=2)

        logger.success(
            f"Saved translated book structure to {translated_structure_path}"
        )
        return translated_structure

    except Exception as e:
        logger.error(f"Failed to translate book structure: {e}")
        logger.warning("Using original structure as fallback")
        return original_structure


def relevel_book_structure(
    structure: dict, reference_structure: dict, config: dict
) -> dict:
    """
    Use LLM to analyze and re-level the markdown book structure using original OCR structure as reference.

    Args:
        structure: Book structure from markdown
        reference_structure: Original OCR structure (optional)
        config: Configuration with API keys

    Returns:
        Dict with level and text changes for markdown headings
    """
    logger.info(
        "Analyzing book structure for re-leveling using original OCR structure as reference..."
    )

    # Initialize LLM client
    llm_client = LLMClient(config)

    # Create simplified structures for both
    markdown_headings = []
    for chapter in structure.get("chapters", []):
        markdown_headings.append(
            {
                "title": chapter["title"],
                "level": 1,
                "index": chapter.get("index", 0),
                "type": "chapter",
            }
        )

        for j, subchapter in enumerate(chapter.get("subchapters", []), 1):
            markdown_headings.append(
                {
                    "title": subchapter["title"],
                    "level": 2,
                    "chapter_index": chapter.get("index", 0),
                    "subchapter_index": j,
                    "type": "subchapter",
                }
            )

    # Create simplified original structure for reference
    original_headings = []
    if reference_structure:
        for chapter in reference_structure.get("chapters", []):
            original_headings.append({"title": chapter.get("title", ""), "level": 1})

            if "subchapters" in chapter:
                for subchapter in chapter["subchapters"]:
                    original_headings.append(
                        {"title": subchapter.get("title", ""), "level": 2}
                    )

    # Create prompt for LLM
    prompt = """You are a book structure expert. Your task is to improve the markdown-generated book structure using the original OCR structure as a reference.

You have TWO structures:
1. ORIGINAL OCR STRUCTURE (reference) - This shows the intended hierarchy from the original book
2. MARKDOWN STRUCTURE (to be modified) - This is what was extracted from markdown files

YOUR TASK:
1. Analyze both structures to understand the intended hierarchy
2. For each heading in the MARKDOWN structure, determine:
   - The appropriate level (1=chapter, 2=section, 3=subsection)
   - Whether the title text should be updated (e.g., if OCR had better formatting or complete text)
3. The NUMBER of headings must remain EXACTLY the same as in the markdown structure
4. You can change both "level" and "title" fields in the markdown structure
5. Use the original OCR structure as guidance for hierarchy and proper titles

IMPORTANT:
- Return the modified markdown structure with the SAME number of entries
- Each entry should have: title (possibly updated), level (possibly updated), and all original fields
- Maintain logical hierarchy - level 3 cannot appear without a parent level 2

ORIGINAL OCR STRUCTURE (reference):
"""

    original_json = json.dumps(original_headings, ensure_ascii=False, indent=2)
    markdown_json = json.dumps(markdown_headings, ensure_ascii=False, indent=2)

    multi_part_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": original_json},
        {"type": "text", "text": "\n\nMARKDOWN STRUCTURE (to modify):\n"},
        {"type": "text", "text": markdown_json},
        {
            "type": "text",
            "text": "\n\nReturn the modified markdown structure with updated levels and titles:",
        },
    ]

    # Use the same models as translation
    releveling_models = [
        {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2},
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "max_retries": 2,
        },
    ]

    try:
        response = llm_client.generate(
            prompt=multi_part_content,
            model_configs=releveling_models,
            operation_name="Re-level book structure with OCR reference",
        )

        # Clean response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        releveled_structure = json.loads(response)

        # Build changes mapping
        changes = {}
        for i, new_item in enumerate(releveled_structure):
            if i >= len(markdown_headings):
                logger.warning(
                    f"Response has more items than original markdown structure"
                )
                break

            original_item = markdown_headings[i]

            # Check for changes
            level_changed = original_item["level"] != new_item.get(
                "level", original_item["level"]
            )
            title_changed = original_item["title"] != new_item.get(
                "title", original_item["title"]
            )

            if level_changed or title_changed:
                if original_item.get("type") == "subchapter":
                    # It's a subchapter
                    key = (
                        original_item["chapter_index"],
                        original_item["subchapter_index"],
                    )
                else:
                    # It's a chapter
                    key = (original_item["index"], 0)

                changes[key] = {
                    "old_title": original_item["title"],
                    "new_title": new_item.get("title", original_item["title"]),
                    "old_level": original_item["level"],
                    "new_level": new_item.get("level", original_item["level"]),
                }

                if title_changed:
                    logger.info(
                        f"Updating title: '{original_item['title']}' → '{new_item['title']}'"
                    )
                if level_changed:
                    logger.info(
                        f"Re-leveling: '{new_item.get('title', original_item['title'])}' from level {original_item['level']} to level {new_item['level']}"
                    )

        return changes

    except Exception as e:
        logger.error(f"Failed to re-level structure: {e}")
        return {}


def update_markdown_heading_levels(markdown_dir: Path, changes: dict):
    """
    Update heading levels and titles in markdown files based on changes.

    Args:
        markdown_dir: Directory containing markdown files
        changes: Dict mapping (chapter_idx, subchapter_idx) to change info with old/new title and level
    """
    if not changes:
        logger.info("No heading changes needed")
        return

    logger.info(f"Updating {len(changes)} headings in markdown files...")

    # Group changes by chapter
    changes_by_chapter = {}
    for (chapter_idx, subchapter_idx), change_info in changes.items():
        if chapter_idx not in changes_by_chapter:
            changes_by_chapter[chapter_idx] = []
        changes_by_chapter[chapter_idx].append((subchapter_idx, change_info))

    # Process each chapter
    for chapter_idx, chapter_changes in changes_by_chapter.items():
        # Find all relevant files (main and parts)
        files_to_update = []

        # Check for part files
        part_files = sorted(markdown_dir.glob(f"chapter_{chapter_idx}.part*.md"))
        if part_files:
            files_to_update.extend(part_files)
        else:
            # Single file
            main_file = markdown_dir / f"chapter_{chapter_idx}.md"
            if main_file.exists():
                files_to_update.append(main_file)

        # Update each file
        for file_path in files_to_update:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Apply each change
                for subchapter_idx, change_info in chapter_changes:
                    old_title = change_info["old_title"]
                    new_title = change_info["new_title"]
                    old_level = change_info["old_level"]
                    new_level = change_info["new_level"]

                    # Create the old and new heading markers
                    old_marker = "#" * old_level
                    new_marker = "#" * new_level

                    # Escape special regex characters in old title
                    escaped_old_title = re.escape(old_title)

                    # Pattern to match the heading line
                    pattern = f"^{re.escape(old_marker)}\\s+{escaped_old_title}\\s*$"
                    replacement = f"{new_marker} {new_title}"

                    # Perform replacement
                    new_content = re.sub(
                        pattern, replacement, content, flags=re.MULTILINE
                    )

                    if new_content != content:
                        if old_title != new_title:
                            logger.debug(
                                f"Updated title: '{old_title}' → '{new_title}' in {file_path.name}"
                            )
                        if old_level != new_level:
                            logger.debug(
                                f"Updated level: {old_level} → {new_level} for '{new_title}' in {file_path.name}"
                            )
                        content = new_content

                # Write back if changed
                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.success(f"Updated headings in {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to update {file_path}: {e}")

    logger.success("Heading updates complete")


def is_file_blank(file_path: Path) -> bool:
    """
    Check if a file is empty or contains only a single level 1 heading
    and nothing else.
    """
    if not file_path.exists():
        return True
    if file_path.stat().st_size == 0:
        return True

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        # File is empty or contains only whitespace
        return True

    if len(lines) == 1:
        # Check if the only line is a level 1 heading
        line = lines[0]
        if line.startswith("# ") and not line.startswith("## "):
            return True

    # File has content other than a single H1 heading
    return False


def filter_blank_files_from_structure(structure: dict, markdown_dir: Path) -> dict:
    """
    Filter out blank markdown files from the book structure.
    
    Args:
        structure: Book structure dictionary
        markdown_dir: Directory containing markdown files
        
    Returns:
        Filtered structure with blank files removed
    """
    # Filter front matter if blank
    if structure.get("front_matter"):
        front_matter_file = markdown_dir / "front_matter.md"
        if is_file_blank(front_matter_file):
            logger.info("Removing blank front_matter from structure")
            del structure["front_matter"]
    
    # Filter back matter if blank
    if structure.get("back_matter"):
        back_matter_file = markdown_dir / "back_matter.md"
        if is_file_blank(back_matter_file):
            logger.info("Removing blank back_matter from structure")
            del structure["back_matter"]
    
    # Filter blank chapters
    if structure.get("chapters"):
        filtered_chapters = []
        for chapter in structure["chapters"]:
            chapter_index = chapter.get("index")
            # Check for multi-part files first
            part_files = sorted(markdown_dir.glob(f"chapter_{chapter_index}.part*.md"))
            
            if part_files:
                # Multi-part chapter exists, keep it
                filtered_chapters.append(chapter)
            else:
                # Single file chapter - check if blank
                chapter_file = markdown_dir / f"chapter_{chapter_index}.md"
                if chapter_file.exists() and not is_file_blank(chapter_file):
                    filtered_chapters.append(chapter)
                else:
                    logger.info(f"Removing chapter {chapter_index} (blank or missing)")
        
        original_chapter_count = len(structure["chapters"])
        structure["chapters"] = filtered_chapters
        if len(filtered_chapters) < original_chapter_count:
            logger.info(f"Kept {len(filtered_chapters)} non-blank chapters out of {original_chapter_count} total")
    
    return structure


def cleanup_old_files(epub_dir: Path):
    """Remove old EPUB files before generating new ones."""
    import shutil

    logger.info(f"Cleaning up old files in {epub_dir}")
    if epub_dir.exists():
        shutil.rmtree(epub_dir)
    logger.info("Old files cleaned up")


def main():
    """Main orchestrator for EPUB generation."""
    parser = argparse.ArgumentParser(
        description="Generate EPUB from polished markdown files"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument("-i", "--input", help="Path to PDF file for cover extraction")
    parser.add_argument(
        "--translated",
        action="store_true",
        help="Use translated markdown instead of polished",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create password-protected ZIP file after EPUB generation",
    )
    parser.add_argument(
        "--relevel",
        action="store_true",
        help="Use LLM to analyze and re-level the book structure",
    )
    parser.add_argument(
        "--global-footnotes",
        action="store_true",
        help="Force global footnotes (use last definition, ignore previous definitions)",
    )

    args = parser.parse_args()

    # Load configuration and create EpubConfig
    config_dict = load_config(args.config)

    if not config_dict.get("title"):
        logger.error("Book title not found in config.yaml")
        return

    # Create EpubConfig instance
    epub_config = EpubConfig.from_dict(
        config_dict,
        use_translated=args.translated,
        use_relevel=args.relevel,
        create_zip=args.zip,
    )

    # Set input PDF path if provided
    if args.input:
        epub_config.input_pdf_path = Path(args.input)
    else:
        # Try to find the original PDF
        original_pdf = epub_config.input_dir / "input_original.pdf"
        if original_pdf.exists():
            epub_config.input_pdf_path = original_pdf

    # Check if markdown directory exists
    if not epub_config.markdown_dir.exists():
        logger.error(f"Markdown directory not found: {epub_config.markdown_dir}")
        if args.translated:
            logger.info("Please run translate command first")
        else:
            logger.info("Please run polish command first")
        return

    logger.info(f"Using markdown from: {epub_config.markdown_dir}")
    
    # Detect language from content
    from .utils.common import guess_language
    detected_language = guess_language(epub_config.markdown_dir)
    logger.info(f"Auto-detected language: {detected_language}")
    
    # Update config with detected language (can be overridden by config file)
    if config_dict.get("language") == "en" or not config_dict.get("language"):
        # Only override if language is default or not set
        epub_config.language = detected_language
        logger.info(f"Using detected language: {detected_language}")
    else:
        logger.info(f"Using configured language: {epub_config.language}")

    # Load book structure to check for notes chapters and get author
    book_structure = load_book_structure(epub_config.book_title)
    has_notes_chapter = False

    if book_structure:
        # Check if any chapter is marked as a notes chapter
        has_notes_chapter = any(
            chapter.get('type') == 'notes'
            for chapter in book_structure.get('chapters', [])
        )

        # Update author from structure if not specified in config
        # Priority: config.yaml > book_structure.json > "Unknown Author"
        if epub_config.author == "Unknown Author" and 'author' in book_structure and book_structure['author']:
            epub_config.author = book_structure['author']
            logger.info(f"Using author from book structure: {epub_config.author}")
    
    # Auto-enable global footnotes if notes chapter detected
    force_global_mode = args.global_footnotes or has_notes_chapter
    
    if has_notes_chapter and not args.global_footnotes:
        logger.info("Detected 'Notes' chapter in book structure - auto-enabling global footnote mode")
    
    # Initialize FootnoteManager to analyze footnote structure
    logger.info("Analyzing footnote structure...")
    footnote_manager = FootnoteManager(epub_config.markdown_dir, force_global=force_global_mode)
    
    # Initialize components
    converter = ContentConverter(epub_config, footnote_manager)
    builder = EpubBuilder(
        epub_config, converter
    )  # Pass converter for file naming helpers

    # Clean and prepare markdown content
    logger.info("Cleaning invalid headings from markdown files...")
    converter.clean_invalid_headings()

    logger.info("Removing duplicate titles from markdown files...")
    converter.remove_duplicate_titles()

    # Build book structure from markdown
    logger.info("Building TOC from markdown headings...")
    structure = build_structure_from_markdown(
        epub_config.markdown_dir, epub_config.book_title
    )
    
    # Filter out blank files from the structure
    logger.info("Filtering out blank markdown files from structure...")
    structure = filter_blank_files_from_structure(structure, epub_config.markdown_dir)
    epub_config.book_structure = structure

    # Merge cover_page info from book_structure.json if available
    if book_structure and 'cover_page' in book_structure:
        structure['cover_page'] = book_structure['cover_page']
        logger.debug(f"Merged cover_page from book_structure.json: {book_structure['cover_page']}")

    # For translated version, determine the display title
    display_title = epub_config.book_title  # Default to original title
    if args.translated:
        # Load or create full translated structure
        original_structure = load_book_structure(epub_config.book_title)
        if original_structure:
            # Use the existing function to load or create translated structure
            translated_structure = load_or_create_translated_book_structure(
                epub_config.book_title, 
                original_structure, 
                config_dict
            )
            display_title = translated_structure.get("book_title") or translated_structure.get("title", epub_config.book_title)
            logger.info(f"Translated structure loaded with title: {display_title}")
        else:
            # Fallback: Check for existing translated structure or translation progress
            translated_structure_path = (
                epub_config.input_dir / "book_structure_translated.json"
            )
            if translated_structure_path.exists():
                logger.info("Loading translated title from existing structure...")
                with open(translated_structure_path, "r", encoding="utf-8") as f:
                    translated_structure = json.load(f)
                    display_title = translated_structure.get("book_title") or translated_structure.get("title", epub_config.book_title)
            else:
                # Try to get language settings from translation_progress.json
                translation_progress_path = epub_config.input_dir / "translated" / "translation_progress.json"
                source_language = "English"  # Default for non-Japanese books
                target_language = "Chinese"  # Default target
                
                if translation_progress_path.exists():
                    with open(translation_progress_path, "r", encoding="utf-8") as f:
                        progress = json.load(f)
                        source_language = progress.get("source_language", source_language)
                        target_language = progress.get("target_language", target_language)
                
                # Create minimal translated structure with just the title
                logger.info(f"Translating book title from {source_language} to {target_language}...")
                from .utils.llm_client import LLMClient
                llm_client = LLMClient(config_dict)
                
                prompt = f"""Translate the book title from {source_language} to {target_language}.
Return ONLY the translated title, no explanations.

Original title: {epub_config.book_title}"""
                try:
                    translated_title = llm_client.query(prompt).strip()
                    display_title = translated_title
                    # Save for future use
                    translated_structure = {"book_title": display_title}
                    with open(translated_structure_path, "w", encoding="utf-8") as f:
                        json.dump(translated_structure, f, ensure_ascii=False, indent=2)
                    logger.info(f"Created translated structure with title: {display_title}")
                except Exception as e:
                    logger.warning(f"Failed to translate title: {e}")

        logger.info(f"Using translated title: {display_title}")
        # Update config with translated title
        epub_config.book_title = display_title
        
        # Rebuild structure from translated markdown to pick up any structural changes
        logger.info("Rebuilding TOC from translated markdown files...")
        structure = build_structure_from_markdown(
            epub_config.markdown_dir, epub_config.book_title
        )
        
        # Update output path with translated title
        from .utils.common import sanitize_filename

        safe_title = sanitize_filename(display_title)
        epub_config.output_epub_path = epub_config.output_dir / f"{safe_title}.epub"

    # Re-level structure if requested
    if args.relevel:
        # Load reference structure
        if args.translated:
            # For translated content, try translated structure first
            translated_structure_path = (
                epub_config.input_dir / "book_structure_translated.json"
            )
            if translated_structure_path.exists():
                with open(translated_structure_path, "r", encoding="utf-8") as f:
                    epub_config.reference_structure = json.load(f)
                logger.info("Using translated book structure as reference")
            else:
                epub_config.reference_structure = load_book_structure(
                    epub_config.book_title
                )
        else:
            epub_config.reference_structure = load_book_structure(
                epub_config.book_title
            )

        if epub_config.reference_structure:
            logger.info("Re-leveling book structure using LLM...")
            changes = relevel_book_structure(
                structure, epub_config.reference_structure, config_dict
            )
            if changes:
                update_markdown_heading_levels(epub_config.markdown_dir, changes)
                # Rebuild structure after changes
                structure = build_structure_from_markdown(
                    epub_config.markdown_dir, epub_config.book_title
                )
                # Filter blank files again after re-leveling
                structure = filter_blank_files_from_structure(structure, epub_config.markdown_dir)
                epub_config.book_structure = structure

    # Clean up old EPUB files
    cleanup_old_files(epub_config.epub_dir)
    ensure_directory(epub_config.epub_dir)

    # Create EPUB directories
    text_dir = epub_config.epub_dir / "text"
    images_dir = epub_config.epub_dir / "images"
    meta_inf_dir = epub_config.epub_dir / "META-INF"

    ensure_directory(text_dir)
    ensure_directory(images_dir)
    ensure_directory(meta_inf_dir)

    # Extract cover if PDF is available
    cover_image = None
    if epub_config.input_pdf_path and epub_config.input_pdf_path.exists():
        logger.info(f"Extracting cover from {epub_config.input_pdf_path}")
        cover_image = converter.extract_cover_image(
            epub_config.input_pdf_path, images_dir
        )
        if cover_image:
            logger.info(f"Cover extracted: {cover_image}")

    # Create stylesheet
    builder.create_stylesheet(epub_config.epub_dir / "stylesheet.css")

    # Create mimetype
    builder.create_mimetype(epub_config.epub_dir / "mimetype")

    # Create container.xml
    builder.create_container_xml(meta_inf_dir / "container.xml")

    # Copy chapter images
    logger.info("Copying chapter images...")
    _, image_mapping = converter.copy_chapter_images(images_dir)

    # For EPUB input: detect cover image after images are copied
    logger.debug(f"Cover image check: cover_image={cover_image}, has_cover_page={bool(structure.get('cover_page'))}")

    if not cover_image and structure.get("cover_page"):
        cover_href = structure["cover_page"].get("href", "")
        logger.info(f"EPUB input detected - looking for cover image (cover_page: {cover_href})")

        # Look for common cover image patterns in the EPUB images directory
        import glob
        for pattern in ["page_1_img_0.*", "cover.*", "Page_1_img_0.*"]:
            matches = glob.glob(str(images_dir / pattern))
            logger.debug(f"Searching {images_dir / pattern}: {len(matches)} matches")
            if matches:
                cover_image = Path(matches[0]).name
                logger.info(f"Found cover image for EPUB input: {cover_image}")
                break

        if not cover_image:
            logger.warning("Cover page specified but no cover image found in images directory")

    # Create cover HTML if cover image exists
    if cover_image:
        builder.create_cover_html(cover_image, text_dir / "cover.html")

    # Detect multi-part chapters for proper TOC generation
    chapters_with_parts = set()
    parts_info = {}

    for chapter in structure.get("chapters", []):
        chapter_index = chapter.get("index")
        # Check for multi-part files
        part_files = sorted(
            epub_config.markdown_dir.glob(f"chapter_{chapter_index}.part*.md")
        )
        if part_files:
            chapters_with_parts.add(chapter_index)
            parts_info[str(chapter_index)] = len(part_files)

    # Build detailed subchapter location mapping BEFORE creating TOC
    # This determines which part contains which subchapter
    subchapter_locations = build_subchapter_locations(
        epub_config.markdown_dir, parts_info
    )

    # Create NCX TOC with multi-part info
    builder.create_toc_ncx(
        structure,
        epub_config.epub_dir / "toc.ncx",
        parts_info=parts_info,
        chapters_with_parts=chapters_with_parts,
        subchapter_locations=subchapter_locations,
    )

    # Create HTML TOC with multi-part info
    builder.create_toc_html(
        structure,
        text_dir / "toc.html",
        parts_info=parts_info,
        chapters_with_parts=chapters_with_parts,
        subchapter_locations=subchapter_locations,
    )

    # Load parts info from progress files for better multi-part handling
    progress_file = None
    if args.translated:
        progress_file = epub_config.markdown_dir / "translation_progress.json"
    else:
        progress_file = epub_config.markdown_dir / "polish_progress.json"

    parts_info_for_toc = {}
    if progress_file and progress_file.exists():
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
            # Support both old and new format
            progress_key = None
            if args.translated and "translations" in progress_data:
                progress_key = "translations"
            elif not args.translated and "parts_info" in progress_data:
                progress_key = "parts_info"
            elif "parts_polished" in progress_data:
                progress_key = "parts_polished"
            elif "polished" in progress_data:
                progress_key = "polished"

            if progress_key:
                if progress_key == "parts_polished":
                    # Old format - direct mapping
                    parts_info_for_toc = progress_data[progress_key]
                else:
                    # New format - extract total_parts from each chapter
                    for chapter_key, chapter_data in progress_data[
                        progress_key
                    ].items():
                        if (
                            isinstance(chapter_data, dict)
                            and "total_parts" in chapter_data
                        ):
                            if chapter_key.startswith("chapter_"):
                                chapter_num = chapter_key.replace("chapter_", "")
                                parts_info_for_toc[chapter_num] = chapter_data[
                                    "total_parts"
                                ]

    # Update parts_info with detected parts from progress files
    for chapter_num in parts_info_for_toc:
        if chapter_num not in parts_info:
            parts_info[chapter_num] = parts_info_for_toc[chapter_num]

    # Note: subchapter_locations was already built earlier and TOC was generated
    # No need to regenerate here

    # Convert markdown files to HTML with proper multi-part handling
    # Track all HTML files created for manifest and spine
    all_html_files = []

    # Front matter
    if structure.get("front_matter"):
        logger.info("Converting front matter to HTML")
        converter.convert_markdown_to_chapter_html(
            epub_config.markdown_dir / "front_matter.md",
            text_dir / "front_matter.html",
            "Front Matter",
            image_mapping=image_mapping,
        )
        all_html_files.append("front_matter.html")

    # Chapters with sophisticated multi-part handling
    for chapter in structure.get("chapters", []):
        chapter_index = chapter.get("index")
        chapter_file = f"chapter_{chapter_index}.md"

        logger.info(f"Processing chapter {chapter_index}")

        # Check for multi-part chapters
        part_files = sorted(
            epub_config.markdown_dir.glob(f"chapter_{chapter_index}.part*.md")
        )

        if part_files:
            # Multi-part chapter: create separate HTML for each part
            logger.info(f"Chapter {chapter_index} has {len(part_files)} parts")

            for i, part_file in enumerate(part_files, 1):
                part_output_file = f"chapter_{chapter_index}_part{i}.html"

                logger.info(f"Converting part {i} of chapter {chapter_index}")

                # Get ALL subchapters from the entire chapter for anchor generation
                # We pass all subchapters to ensure anchors are generated correctly
                # even if a subchapter appears in a different part than expected
                all_subchapters = []
                if chapter.get("subchapters"):
                    for j, subchapter in enumerate(chapter.get("subchapters", []), 1):
                        all_subchapters.append((j, subchapter.get('title', '')))

                # Convert this part to HTML
                converter.convert_markdown_to_chapter_html(
                    part_file,
                    text_dir / part_output_file,
                    f"{chapter.get('title', f'Chapter {chapter_index}')} (Part {i})",
                    chapter_index=chapter_index,
                    subchapter_info=all_subchapters,
                    image_mapping=image_mapping,
                )
                all_html_files.append(part_output_file)
        else:
            # Single file chapter
            output_file = f"chapter_{chapter_index}.html"

            # Check if the single file exists
            single_file = epub_config.markdown_dir / chapter_file
            if single_file.exists():
                logger.info(f"Converting single-file chapter {chapter_index}")

                converter.convert_markdown_to_chapter_html(
                    single_file,
                    text_dir / output_file,
                    chapter.get("title", f"Chapter {chapter_index}"),
                    chapter_index=chapter_index,
                    subchapter_info=chapter.get("subchapters", []),
                    image_mapping=image_mapping,
                )
                all_html_files.append(output_file)
            else:
                logger.warning(f"Chapter file not found: {single_file}")

    # Back matter
    if structure.get("back_matter"):
        logger.info("Converting back matter to HTML")
        converter.convert_markdown_to_chapter_html(
            epub_config.markdown_dir / "back_matter.md",
            text_dir / "back_matter.html",
            "Back Matter",
            image_mapping=image_mapping,
        )
        all_html_files.append("back_matter.html")

    # Create content.opf with all HTML files for proper manifest/spine
    builder.create_content_opf(
        structure,
        epub_config.epub_dir,
        epub_config.epub_dir / "content.opf",
        cover_image=cover_image,
        all_html_files=all_html_files,
    )

    # Create the EPUB file
    builder.create_epub(epub_config.epub_dir, epub_config.output_epub_path)

    logger.success("EPUB generation complete!")
    logger.info(f"Output: {epub_config.output_epub_path}")

    # Create ZIP if requested
    if args.zip:
        logger.info("Creating password-protected ZIP file...")
        zip_path = create_password_protected_zip(
            epub_config.output_epub_path, epub_config.book_title
        )
        logger.success(f"Created ZIP: {zip_path}")


if __name__ == "__main__":
    main()
