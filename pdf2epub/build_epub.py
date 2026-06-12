"""
Build EPUB from toc_tree.json structure.

This module generates EPUB files using toc_tree.json as the authoritative
source for chapter structure and titles. Unlike generate_epub.py which
scans markdown files, this approach provides:
- Accurate title hierarchy from PDF TOC
- Consistent chapter numbering
- Support for unlimited nesting levels
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .epub.builder import EpubBuilder
from .epub.converter import ContentConverter
from .utils.unit_id import generate_unit_id
from .utils.pdf_utils import extract_cover_image


@dataclass
class BuildEpubConfig:
    """Configuration for EPUB building."""
    book_title: str
    output_dir: Path
    markdown_dir: Path
    toc_tree_path: Path
    images_dir: Optional[Path] = None
    cover_image: Optional[Path] = None
    translated: bool = False
    target_language: str = "Chinese"
    config: Optional[Dict] = None


def load_toc_tree(path: Path) -> Dict:
    """Load and validate toc_tree.json."""
    if not path.exists():
        raise FileNotFoundError(f"toc_tree.json not found at {path}")

    with open(path, 'r', encoding='utf-8') as f:
        toc_tree = json.load(f)

    # Validate required fields
    if 'chapters' not in toc_tree:
        raise ValueError("toc_tree.json must contain 'chapters' array")

    return toc_tree


def calculate_tree_depth(entries: List[Dict], current_depth: int = 1) -> int:
    """
    Calculate maximum depth of hierarchical structure.

    Args:
        entries: List of chapter/section entries
        current_depth: Current depth level

    Returns:
        Maximum depth found in the tree
    """
    if not entries:
        return current_depth - 1 if current_depth > 1 else 1

    max_depth = current_depth
    for entry in entries:
        children = entry.get('children', [])
        if children:
            child_depth = calculate_tree_depth(children, current_depth + 1)
            max_depth = max(max_depth, child_depth)

    return max_depth


def find_chapter_file(unit_id: str, markdown_dir: Path) -> Optional[Path]:
    """
    Find markdown file for a unit ID.

    Args:
        unit_id: Unit ID like "chapter_12" or "chapter_7.1.1"
        markdown_dir: Directory containing markdown files

    Returns the base file (first part if split).
    Handles nested splits like chapter_25.part1.part1.md
    """
    # Try exact match first
    exact_file = markdown_dir / f"{unit_id}.md"
    if exact_file.exists():
        return exact_file

    # Try part1 file
    part1_file = markdown_dir / f"{unit_id}.part1.md"
    if part1_file.exists():
        return part1_file

    # Try nested part1.part1 file (when part1 was further split)
    nested_part1_file = markdown_dir / f"{unit_id}.part1.part1.md"
    if nested_part1_file.exists():
        return nested_part1_file

    return None


def find_part_files(base_file: Path) -> List[Path]:
    """
    Find all part files for a chapter.

    Given chapter_10.38_intro.md or chapter_10.38_intro.part1.md,
    returns [part1, part2, part3, ...] in order.

    Handles nested splits like:
    - chapter_25.part1.part1.md
    - chapter_25.part1.part2.md
    - chapter_25.part2.md
    - chapter_25.part3.md
    """
    if base_file is None:
        return []

    # Get the base name without ALL .partN suffixes
    name = base_file.name
    stem = name.rsplit('.md', 1)[0]  # Remove .md

    # Strip all .partN suffixes to get true base
    # chapter_25.part1.part1 -> chapter_25
    base_pattern = stem
    while '.part' in base_pattern:
        base_pattern = base_pattern.rsplit('.part', 1)[0]

    # Find all part files (including nested)
    # This matches chapter_25.part1.md, chapter_25.part2.md, chapter_25.part1.part1.md, etc.
    part_files = list(base_file.parent.glob(f"{base_pattern}.part*.md"))

    if part_files:
        # Sort by extracting part numbers
        def sort_key(p: Path):
            # chapter_25.part1.part2.md -> [1, 2]
            # chapter_25.part2.md -> [2]
            stem = p.stem
            parts = []
            remaining = stem
            while '.part' in remaining:
                idx = remaining.index('.part')
                remaining = remaining[idx + 5:]  # Skip ".part"
                # Extract number
                num_str = ""
                for c in remaining:
                    if c.isdigit():
                        num_str += c
                    else:
                        break
                if num_str:
                    parts.append(int(num_str))
                    remaining = remaining[len(num_str):]
            return parts

        part_files = sorted(part_files, key=sort_key)
        return part_files
    else:
        # No part files, return the single file
        return [base_file]


def process_chapter_content(
    toc_title: str,
    toc_level: int,
    markdown_content: str,
    is_first_part: bool
) -> str:
    """
    Process chapter content with correct title and heading levels.

    Args:
        toc_title: Title from toc_tree.json
        toc_level: Level from toc_tree.json (2, 3, 4, ...)
        markdown_content: Raw markdown content
        is_first_part: True for base file or part1, False for part2+

    Returns:
        Processed markdown with correct title and heading levels
    """
    if not is_first_part:
        # part2, part3... don't get chapter title, just relevel
        return relevel_content(markdown_content, toc_level)

    lines = markdown_content.split('\n')

    # Generate heading with correct level
    heading_prefix = '#' * toc_level
    chapter_heading = f"{heading_prefix} {toc_title}"

    # Check first 3 lines for existing heading
    title_line_idx = None
    for i in range(min(3, len(lines))):
        if lines[i].strip().startswith('#'):
            title_line_idx = i
            break

    if title_line_idx is not None:
        # Replace existing heading
        lines[title_line_idx] = chapter_heading
    else:
        # Add heading at start
        lines.insert(0, chapter_heading)
        lines.insert(1, '')  # Blank line after heading
        title_line_idx = 0

    # Remove duplicate heading if present (OCR artifact)
    _remove_duplicate_heading(lines, title_line_idx, toc_title)

    # Relevel remaining headings
    return relevel_content('\n'.join(lines), toc_level, skip_first=True)


def _remove_duplicate_heading(lines: list, title_line_idx: int, toc_title: str):
    """Remove duplicate heading that follows the chapter title.

    OCR often produces a heading like "## Title" which duplicates the TOC title.
    If we find a semantically similar heading within the next few lines, remove it.
    Uses embedding-based similarity for better multilingual support.
    """
    # Look in lines after the title (skip blank lines)
    for i in range(title_line_idx + 1, min(title_line_idx + 5, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith('#'):
            # Extract heading text (remove # prefix)
            heading_text = re.sub(r'^#+\s*', '', line)
            # Check semantic similarity using embeddings
            similarity = _compute_semantic_similarity(toc_title, heading_text)
            if similarity > 0.6:
                # Remove this duplicate heading
                lines[i] = ''
                logger.debug(f"Removed duplicate heading: '{heading_text}' (similarity: {similarity:.2f})")
            break  # Only check the first heading after title


# Lazy-loaded embedding model
_embedding_model = None


def _get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Small multilingual model (~120MB)
            _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.debug("Loaded embedding model for heading deduplication")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to text similarity")
            _embedding_model = False  # Mark as unavailable
    return _embedding_model


def _compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using embeddings.

    Falls back to text-based similarity if embeddings unavailable.
    """
    import difflib

    model = _get_embedding_model()
    if model is False:
        # Fallback to text similarity
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    try:
        # Compute embeddings
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        # Cosine similarity
        from sentence_transformers import util
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity
    except Exception as e:
        logger.warning(f"Embedding computation failed: {e}, falling back to text similarity")
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def relevel_content(content: str, base_level: int, skip_first: bool = False) -> str:
    """
    Adjust heading levels so all content headings are below base_level.

    For base_level=3, markdown headings become:
    - # -> #### (base_level + 1)
    - ## -> ##### (base_level + 2)
    - etc.

    Args:
        content: Markdown content
        base_level: The level of the chapter heading
        skip_first: If True, don't relevel the first heading (it's the chapter title)

    Returns:
        Content with adjusted heading levels
    """
    lines = content.split('\n')
    first_skipped = False

    for i, line in enumerate(lines):
        match = re.match(r'^(#+)\s', line)
        if match:
            if skip_first and not first_skipped:
                first_skipped = True
                continue

            current_hashes = len(match.group(1))
            # Content headings start at base_level + 1
            new_level = base_level + current_hashes
            # Cap at 6 (maximum markdown heading level)
            new_level = min(new_level, 6)
            lines[i] = '#' * new_level + line[current_hashes:]

    return '\n'.join(lines)


def has_notes_chapter(chapters: List[Dict]) -> bool:
    """
    Check if any chapter in the TOC tree has type 'notes'.

    Args:
        chapters: List of chapter dictionaries from toc_tree

    Returns:
        True if a notes chapter is found
    """
    for chapter in chapters:
        if chapter.get('type') == 'notes':
            return True
        if 'children' in chapter and chapter['children']:
            if has_notes_chapter(chapter['children']):
                return True
    return False


def flatten_toc_tree(
    chapters: List[Dict],
    parent_index_path: List[int] = None
) -> List[Dict]:
    """
    Flatten nested toc_tree structure into a list with file assignments.

    Each entry in the result has:
    - title: Chapter title
    - level: Heading level
    - index_path: Hierarchical index like [7, 1, 1]
    - children: Flattened children (if any)
    """
    result = []

    for i, chapter in enumerate(chapters):
        # Build index_path: parent path + current 1-based index
        if parent_index_path is None:
            index_path = [i + 1]
        else:
            index_path = parent_index_path + [i + 1]

        entry = {
            'title': chapter['title'],
            'level': chapter.get('level', 1),
            'index_path': index_path,
            'start_page': chapter.get('start_page'),
            'end_page': chapter.get('end_page'),
        }

        if 'children' in chapter and chapter['children']:
            entry['children'] = flatten_toc_tree(
                chapter['children'],
                index_path
            )

        result.append(entry)

    return result



def build_epub_structure(
    toc_structure: List[Dict],
    markdown_dir: Path
) -> Dict:
    """
    Build the final EPUB structure with file paths.

    Args:
        toc_structure: Processed toc structure (possibly translated)
        markdown_dir: Directory containing markdown files

    Returns:
        Structure dict for EPUB generation
    """
    def process_entry(entry: Dict) -> Dict:
        result = {
            'title': entry['title'],
            'level': entry['level'],
        }

        # Find markdown file using unit_id
        index_path = entry.get('index_path')

        if index_path is not None:
            # Generate unit_id from hierarchical index
            unit_id = generate_unit_id(index_path)
            result['unit_id'] = unit_id

            file_path = find_chapter_file(unit_id, markdown_dir)

            if file_path:
                result['file_path'] = file_path
                result['part_files'] = find_part_files(file_path)
            elif 'children' not in entry:
                # Only warn if this is a leaf node (no children)
                # Container nodes (with children) don't need their own markdown
                logger.warning(f"No markdown file found for {unit_id}: {entry['title']}")

        # Process children recursively
        if 'children' in entry:
            result['children'] = [process_entry(c) for c in entry['children']]

        return result

    return [process_entry(ch) for ch in toc_structure]


def generate_hierarchical_toc_ncx(
    structure: List[Dict],
    book_title: str,
    output_path: Path
) -> bool:
    """
    Generate NCX TOC with unlimited hierarchy levels.

    Args:
        structure: Hierarchical structure from build_epub_structure
        book_title: Book title
        output_path: Path to save NCX file

    Returns:
        True if successful
    """
    import html
    import uuid

    uid = str(uuid.uuid4())

    # Calculate actual depth from structure (+1 for TOC entry)
    toc_depth = calculate_tree_depth(structure) + 1

    ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{uid}"/>
        <meta name="dtb:depth" content="{toc_depth}"/>
        <meta name="dtb:totalPageCount" content="0"/>
        <meta name="dtb:maxPageNumber" content="0"/>
    </head>
    <docTitle>
        <text>{html.escape(book_title)}</text>
    </docTitle>
    <navMap>
        <navPoint id="navpoint-toc" playOrder="1">
            <navLabel>
                <text>Table of Contents</text>
            </navLabel>
            <content src="text/toc.html"/>
        </navPoint>
"""

    play_order = [2]  # Use list for mutable counter

    def render_entry(entry: Dict, indent: int = 2, parent_href: str = "text/toc.html") -> str:
        """Recursively render a nav entry and its children."""
        nonlocal play_order
        result = ""
        spaces = "    " * indent

        nav_id = f"navpoint-{play_order[0]}"
        title = html.escape(entry['title'])

        # Determine the href
        if 'file_path' in entry:
            # Has content file
            unit_id = entry.get('unit_id')
            if unit_id:
                # Check if this chapter has multiple parts
                part_files = entry.get('part_files', [])
                if len(part_files) > 1:
                    href = f"text/{unit_id}_part1.html"
                else:
                    href = f"text/{unit_id}.html"
            else:
                href = parent_href  # Use parent's file
        else:
            # No file, link to first child with file, or parent's file
            href = find_first_child_href(entry, parent_href)

        result += f"""{spaces}<navPoint id="{nav_id}" playOrder="{play_order[0]}">
{spaces}    <navLabel>
{spaces}        <text>{title}</text>
{spaces}    </navLabel>
{spaces}    <content src="{href}"/>
"""
        play_order[0] += 1

        # Render children, passing current href as parent
        if 'children' in entry:
            for child in entry['children']:
                result += render_entry(child, indent + 1, href)

        result += f"{spaces}</navPoint>\n"
        return result

    def find_first_child_href(entry: Dict, parent_href: str = "text/toc.html") -> str:
        """Find href of first descendant with a file, or fall back to parent."""
        if 'file_path' in entry:
            unit_id = entry.get('unit_id')
            if unit_id:
                # Check if this chapter has multiple parts
                part_files = entry.get('part_files', [])
                if len(part_files) > 1:
                    return f"text/{unit_id}_part1.html"
                else:
                    return f"text/{unit_id}.html"
            return parent_href
        if 'children' in entry:
            for child in entry['children']:
                href = find_first_child_href(child, parent_href)
                if href != "text/toc.html" and href != parent_href:
                    return href
        # No file found in this subtree, use parent's file
        return parent_href

    # Render all top-level entries
    for entry in structure:
        ncx += render_entry(entry)

    ncx += """    </navMap>
</ncx>"""

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ncx)
        logger.success(f"Created hierarchical NCX TOC: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create NCX: {e}")
        return False


def generate_hierarchical_toc_html(
    structure: List[Dict],
    book_title: str,
    output_path: Path,
    language: str = "en"
) -> bool:
    """
    Generate HTML TOC with unlimited hierarchy levels.

    Args:
        structure: Hierarchical structure from build_epub_structure
        book_title: Book title
        output_path: Path to save HTML file
        language: Language code for HTML

    Returns:
        True if successful
    """
    import html

    lang_class = f"lang-{language}"

    toc_html = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head>
    <title>Table of Contents</title>
    <link rel="stylesheet" type="text/css" href="../stylesheet.css"/>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
</head>
<body class="{lang_class}">
    <div class="toc">
        <h1>Table of Contents</h1>
"""

    def render_entry(entry: Dict, level: int = 0, parent_href: str = "toc.html") -> str:
        """Recursively render a TOC entry and its children."""
        result = ""
        indent = "            " + "    " * level

        title = html.escape(entry['title'])

        # Determine the href
        if 'file_path' in entry:
            unit_id = entry.get('unit_id')
            if unit_id:
                # Check if this chapter has multiple parts
                part_files = entry.get('part_files', [])
                if len(part_files) > 1:
                    href = f"{unit_id}_part1.html"
                else:
                    href = f"{unit_id}.html"
            else:
                href = parent_href  # Use parent's file
        else:
            # No file - link to first child with file, or parent's file
            href = find_first_child_href_html(entry, parent_href)

        result += f"{indent}<li>\n"
        result += f"{indent}    <a href=\"{href}\">{title}</a>\n"

        # Render children, passing current href as parent
        if 'children' in entry and entry['children']:
            result += f"{indent}    <ul>\n"
            for child in entry['children']:
                result += render_entry(child, level + 1, href)
            result += f"{indent}    </ul>\n"

        result += f"{indent}</li>\n"
        return result

    def find_first_child_href_html(entry: Dict, parent_href: str = "toc.html") -> str:
        """Find href of first descendant with a file, or fall back to parent."""
        if 'file_path' in entry:
            unit_id = entry.get('unit_id')
            if unit_id:
                # Check if this chapter has multiple parts
                part_files = entry.get('part_files', [])
                if len(part_files) > 1:
                    return f"{unit_id}_part1.html"
                else:
                    return f"{unit_id}.html"
            return parent_href
        if 'children' in entry:
            for child in entry['children']:
                href = find_first_child_href_html(child, parent_href)
                if href != "toc.html" and href != parent_href:
                    return href
        # No file found in this subtree, use parent's file
        return parent_href

    toc_html += "        <ul>\n"
    for entry in structure:
        toc_html += render_entry(entry)
    toc_html += "        </ul>\n"

    toc_html += """    </div>
</body>
</html>"""

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(toc_html)
        logger.success(f"Created hierarchical HTML TOC: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create HTML TOC: {e}")
        return False


def build_epub(config: BuildEpubConfig) -> Path:
    """
    Main entry point for building EPUB from toc_tree.json.

    Args:
        config: Build configuration

    Returns:
        Path to generated EPUB file
    """
    import shutil
    import tempfile
    from .epub.builder import EpubBuilder
    from .epub.converter import ContentConverter
    from .epub.footnotes import FootnoteManager
    from .utils.config_manager import load_config

    logger.info(f"Building EPUB for: {config.book_title}")
    logger.info(f"Source: {'translated' if config.translated else 'polished_markdown'}")

    # Load config for LLM-based footnote matching
    try:
        llm_config = load_config()
    except Exception as e:
        logger.debug(f"Could not load config for LLM: {e}")
        llm_config = None

    # Load toc_tree.json
    toc_tree = load_toc_tree(config.toc_tree_path)
    logger.info(f"Loaded toc_tree.json with {len(toc_tree.get('chapters', []))} top-level entries")

    # Extract cover image from PDF if not provided
    if not config.cover_image:
        cover_page_info = toc_tree.get('cover_page') or {}
        cover_page_num = cover_page_info.get('page_number')

        # Fallback: use page 1 as cover if no cover page specified
        if not cover_page_num:
            cover_page_num = 1
            logger.info("No cover page in TOC, using page 1 as cover")

        if cover_page_num:
            # Look for original PDF (without page number patches) for clean cover
            pdf_path = config.output_dir / "input_original.pdf"
            if not pdf_path.exists():
                pdf_path = config.output_dir / "input.pdf"

            if pdf_path.exists():
                # Extract cover to images directory
                images_dir = config.images_dir or (config.output_dir / "images")
                images_dir.mkdir(parents=True, exist_ok=True)
                cover_output = images_dir / "cover.jpg"

                extracted = extract_cover_image(pdf_path, cover_output, cover_page_num)
                if extracted:
                    config.cover_image = extracted
                    logger.info(f"Extracted cover from PDF page {cover_page_num}")
            else:
                logger.warning("Cannot extract cover: PDF not found in output directory")

    # Load translated TOC if building translated EPUB
    if config.translated:
        translated_toc_path = config.output_dir / "toc_tree_translated.json"
        if translated_toc_path.exists():
            with open(translated_toc_path, 'r', encoding='utf-8') as f:
                toc_tree = json.load(f)
            logger.info("Loaded translated TOC from toc_tree_translated.json")

            # Use translated book title
            if 'book_title' in toc_tree:
                config.book_title = toc_tree['book_title']
                logger.info(f"Using translated title: {config.book_title}")
        else:
            logger.warning("toc_tree_translated.json not found, using original titles")

    # Flatten and process structure
    toc_structure = flatten_toc_tree(toc_tree['chapters'])

    # Build structure with file paths
    epub_structure = build_epub_structure(toc_structure, config.markdown_dir)

    # Count chapters with files
    def count_with_files(entries):
        count = 0
        for e in entries:
            if 'file_path' in e:
                count += 1
            if 'children' in e:
                count += count_with_files(e['children'])
        return count

    chapters_with_files = count_with_files(epub_structure)
    logger.info(f"Found markdown files for {chapters_with_files} chapters")

    # Create temporary EPUB directory
    epub_dir = config.output_dir / "epub_build"
    if epub_dir.exists():
        shutil.rmtree(epub_dir)
    epub_dir.mkdir(parents=True)

    # Create EPUB directory structure
    (epub_dir / "META-INF").mkdir()
    (epub_dir / "text").mkdir()
    (epub_dir / "images").mkdir()

    # Copy and compress images, get mapping
    image_mapping = {}
    if config.images_dir and config.images_dir.exists():
        images_dest = epub_dir / "images"
        for img_file in config.images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                dest_path = images_dest / img_file.name
                shutil.copy(img_file, dest_path)
                # Map original name to itself (no renaming for now)
                image_mapping[img_file.name] = img_file.name
        logger.info(f"Copied {len(image_mapping)} images from {config.images_dir}")

    # Create a minimal config object for EpubBuilder and ContentConverter
    class MinimalConfig:
        def __init__(self, book_title, author, language, markdown_dir):
            self.book_title = book_title
            self.author = author
            self.language = language
            self.markdown_dir = markdown_dir

    author = toc_tree.get('author', 'Unknown')
    language = 'zh' if config.translated else toc_tree.get('language', 'en')
    if language.lower() in ['english', 'japanese', 'chinese']:
        language = {'english': 'en', 'japanese': 'ja', 'chinese': 'zh'}.get(language.lower(), 'en')

    minimal_config = MinimalConfig(config.book_title, author, language, config.markdown_dir)
    builder = EpubBuilder(minimal_config)

    # Create ContentConverter for cleanup operations
    converter = ContentConverter(minimal_config)

    # Clean up markdown content before conversion
    removed_headings = converter.clean_invalid_headings()
    if removed_headings > 0:
        logger.info(f"Cleaned {removed_headings} invalid headings")

    removed_duplicates = converter.remove_duplicate_titles()
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate titles")

    # Initialize FootnoteManager after cleanup so scanned refs/defs match the
    # markdown that will actually be converted.
    auto_global = has_notes_chapter(toc_tree.get('chapters', []))
    if auto_global:
        logger.info("Detected notes chapter, enabling global footnote mode")
    footnote_manager = FootnoteManager(
        config.markdown_dir,
        auto_global=auto_global,
        config=llm_config,
        epub_structure=epub_structure,
    )
    converter.footnote_manager = footnote_manager
    logger.debug(f"Initialized FootnoteManager in {footnote_manager.style.value} mode")

    # Create basic EPUB files
    builder.create_mimetype(epub_dir / "mimetype")
    builder.create_container_xml(epub_dir / "META-INF" / "container.xml")
    builder.create_stylesheet(epub_dir / "stylesheet.css")

    # Generate hierarchical TOC
    generate_hierarchical_toc_ncx(epub_structure, config.book_title, epub_dir / "toc.ncx")
    generate_hierarchical_toc_html(epub_structure, config.book_title, epub_dir / "text" / "toc.html", language)

    # Configure footnote manager with epub structure (for HTML filename mapping)
    footnote_manager.configure_from_structure(epub_structure)

    # Process and convert chapters
    all_html_files = []

    def process_chapters(entries: List[Dict]):
        """Recursively process all chapters with files."""
        for entry in entries:
            if 'file_path' in entry:
                unit_id = entry.get('unit_id')
                if unit_id:
                    # Process this chapter
                    part_files = entry.get('part_files', [entry['file_path']])

                    for part_idx, part_file in enumerate(part_files):
                        is_first_part = (part_idx == 0)

                        # Read markdown content
                        with open(part_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Process content with correct title and levels
                        processed = process_chapter_content(
                            entry['title'],
                            entry['level'],
                            content,
                            is_first_part
                        )

                        # Convert to HTML with full footnote and image support
                        # Use actual part file name for correct footnote linking
                        part_file_stem = Path(part_file).stem
                        html_content = markdown_to_html(
                            processed,
                            config.book_title,
                            language,
                            footnote_manager=footnote_manager,
                            image_mapping=image_mapping,
                            source_chapter=part_file_stem
                        )

                        # Add subchapter anchors for TOC navigation
                        if 'children' in entry and entry['children']:
                            # Extract chapter index from unit_id (e.g., "chapter_8" -> 8)
                            try:
                                from .chapter_identity import ChapterIdentity
                                identity = ChapterIdentity.parse(unit_id)
                                if identity and identity.index_path:
                                    chapter_index = identity.index_path[0]
                                    # Build subchapter_info from children
                                    subchapter_info = [
                                        {'title': child['title']}
                                        for child in entry['children']
                                    ]
                                    html_content = converter._add_subchapter_anchors(
                                        html_content, chapter_index, subchapter_info
                                    )
                            except Exception as e:
                                logger.debug(f"Could not add subchapter anchors for {unit_id}: {e}")

                        # Determine output filename using unit_id
                        if len(part_files) > 1:
                            html_filename = f"{unit_id}_part{part_idx + 1}.html"
                        else:
                            html_filename = f"{unit_id}.html"

                        # Write HTML file
                        html_path = epub_dir / "text" / html_filename
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)

                        all_html_files.append(html_filename)
                        logger.debug(f"Created {html_filename}")

            # Process children
            if 'children' in entry:
                process_chapters(entry['children'])

    process_chapters(epub_structure)
    logger.info(f"Converted {len(all_html_files)} HTML files")

    # Handle cover image
    cover_image = None
    if config.cover_image and config.cover_image.exists():
        cover_filename = config.cover_image.name
        shutil.copy(config.cover_image, epub_dir / "images" / cover_filename)
        cover_image = cover_filename
        # Create cover HTML
        builder.create_cover_html(cover_filename, epub_dir / "text" / "cover.html")

    # Create content.opf
    # Convert our structure to the format expected by create_content_opf
    flat_structure = {
        'book_title': config.book_title,
        'chapters': []  # We use all_html_files instead
    }
    builder.create_content_opf(
        flat_structure,
        epub_dir,
        epub_dir / "content.opf",
        cover_image,
        all_html_files
    )

    # Create final EPUB with sanitized filename
    from .utils.common import sanitize_filename
    safe_title = sanitize_filename(config.book_title)
    epub_path = config.output_dir / f"{safe_title}.epub"
    builder.create_epub(epub_dir, epub_path)

    # Keep build directory for debugging (don't clean up)
    # shutil.rmtree(epub_dir)

    logger.success(f"EPUB created: {epub_path}")
    logger.info(f"Build directory kept for debugging: {epub_dir}")
    return epub_path


def markdown_to_html(
    markdown_content: str,
    book_title: str,
    language: str = "en",
    footnote_manager=None,
    image_mapping=None,
    source_chapter=None
) -> str:
    """
    Convert markdown to XHTML for EPUB.

    Uses the full conversion pipeline with LaTeX/math support.
    """
    import html
    from .markdown_to_html import convert_markdown_to_html

    # Use full conversion with math/formula processing
    # standalone=False returns just the body content
    body = convert_markdown_to_html(
        markdown_content,
        title=book_title,
        include_css=False,
        standalone=False,
        image_mapping=image_mapping,
        footnote_manager=footnote_manager,
        source_chapter=source_chapter
    )

    lang_class = f"lang-{language}"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head>
    <title>{html.escape(book_title)}</title>
    <link rel="stylesheet" type="text/css" href="../stylesheet.css"/>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
</head>
<body class="{lang_class}">
{body}
</body>
</html>"""


# CLI entry point will be added in cli.py
