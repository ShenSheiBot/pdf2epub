#!/usr/bin/env python3
"""
Generate EPUB from polished markdown files.
Version 3 - Simplified without progress tracking, with automatic cleanup.
"""

import json
import os
import uuid
import shutil
import yaml
import zipfile
import re
import argparse
import fitz  # PyMuPDF for cover extraction
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
from loguru import logger
from .utils.logging_config import configure_logging
from .utils.zip_utils import create_password_protected_zip
from .markdown_to_html import convert_markdown_to_html

# Configure logger
logger = configure_logging()


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_book_structure(book_title):
    """Load the book structure JSON file."""
    structure_path = Path("output") / Path(book_title) / "book_structure.json"
    if structure_path.exists():
        with open(structure_path, "r", encoding="utf-8") as file:
            structure = json.load(file)
        return structure
    return None


def load_or_create_translated_book_structure(book_title, original_structure, config):
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
    translated_structure_path = Path("output") / book_title / "book_structure_translated.json"
    
    if translated_structure_path.exists():
        logger.info(f"Loading existing translated book structure from {translated_structure_path}")
        with open(translated_structure_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Get translation settings from translation_progress.json if available
    translation_progress_path = Path("output") / book_title / "translated" / "translation_progress.json"
    source_language = "Japanese"  # Default
    target_language = "Chinese"   # Default
    
    if translation_progress_path.exists():
        with open(translation_progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
            source_language = progress.get("source_language", source_language)
            target_language = progress.get("target_language", target_language)
            logger.info(f"Using translation settings: {source_language} → {target_language}")
    
    # Translate the structure using LLM
    logger.info("Translating book structure using LLM...")
    
    # Import and initialize the same LLM client as processors use
    from pdf2epub.utils.llm_client import LLMClient
    
    # Initialize LLM client with the same setup as TranslateProcessor
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
    
    # Create multi-part content for the LLM (same format as TranslateProcessor)
    multi_part_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": structure_json}
    ]
    
    # Use same models as translation (matching TranslateProcessor defaults)
    translation_models = config.get("translation_models", [
        {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2},
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "max_retries": 2}
    ])
    
    try:
        # Generate translation using the same method as processors
        translated_json = llm_client.generate(
            prompt=multi_part_content,
            model_configs=translation_models,
            operation_name="Translate book structure"
        )
        
        # Clean response (remove markdown code blocks if present)
        if "```json" in translated_json:
            translated_json = translated_json.split("```json")[1].split("```")[0]
        elif "```" in translated_json:
            translated_json = translated_json.split("```")[1].split("```")[0]
        
        # Parse the translated JSON
        translated_structure = json.loads(translated_json)
        
        # Save for future use
        with open(translated_structure_path, "w", encoding="utf-8") as f:
            json.dump(translated_structure, f, ensure_ascii=False, indent=2)
        
        logger.success(f"Translated book structure saved to {translated_structure_path}")
        
        # Log some examples of translations
        if "chapters" in original_structure and "chapters" in translated_structure:
            for i in range(min(3, len(original_structure["chapters"]))):
                orig = original_structure["chapters"][i]["title"]
                trans = translated_structure["chapters"][i]["title"]
                logger.info(f"Chapter {i+1}: '{orig}' → '{trans}'")
        
        return translated_structure
        
    except Exception as e:
        logger.error(f"Failed to translate book structure: {e}")
        logger.warning("Falling back to original structure")
        return original_structure


def ensure_directory(directory_path):
    """Ensure a directory exists, create it if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def cleanup_old_files(epub_dir):
    """Clean up old intermediate files from previous generations."""
    if epub_dir.exists():
        logger.info(f"Cleaning up old files in {epub_dir}")
        shutil.rmtree(epub_dir)
        logger.info("Old files cleaned up")


def copy_chapter_images(source_dir, dest_dir):
    """Copy all chapter images from source to destination directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'}
    images_copied = 0
    
    # Look for images in the parent directory of polished_markdown (usually 'images' folder)
    images_source = source_dir.parent / "images"
    if images_source.exists():
        for img_file in images_source.iterdir():
            if img_file.suffix.lower() in image_extensions:
                dest_path = dest_dir / img_file.name
                shutil.copy2(img_file, dest_path)
                images_copied += 1
                logger.debug(f"Copied image: {img_file.name}")
    
    # Also look for images in the polished_markdown directory itself
    for img_file in source_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            dest_path = dest_dir / img_file.name
            shutil.copy2(img_file, dest_path)
            images_copied += 1
            logger.debug(f"Copied image: {img_file.name}")
    
    if images_copied > 0:
        logger.success(f"Copied {images_copied} images to EPUB")
    
    return images_copied


def extract_cover_image(pdf_path, output_dir):
    """Extract the first page of the PDF as the cover image."""
    try:
        pdf_doc = fitz.open(pdf_path)
        first_page = pdf_doc[0]
        
        # Render page at high resolution
        mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better quality
        pix = first_page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        # Save as cover.jpg (EPUB standard prefers JPEG for covers)
        cover_path = output_dir / "cover.jpg"
        img.convert("RGB").save(cover_path, "JPEG", quality=90)
        
        pdf_doc.close()
        
        logger.success(f"Extracted cover image: {cover_path}")
        return "cover.jpg"
        
    except Exception as e:
        logger.error(f"Failed to extract cover: {e}")
        return None


def convert_markdown_to_chapter_html(markdown_path, output_path, chapter_title, chapter_index=None, subchapter_info=None):
    """Convert a markdown file to HTML chapter format with proper anchors for subchapters.
    
    Args:
        subchapter_info: List of tuples (subchapter_index, subchapter_title) or list of titles
    """
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML (just the body content, not a full document)
        html_content = convert_markdown_to_html(markdown_content, standalone=False)
        
        # Add/update anchors to subchapter headings if we have the info
        if chapter_index and subchapter_info:
            # Handle both formats: list of titles or list of (index, title) tuples
            if subchapter_info and isinstance(subchapter_info[0], tuple):
                subchapters = subchapter_info
            else:
                # Convert simple list to tuples with indices
                subchapters = [(j, title) for j, title in enumerate(subchapter_info, 1)]
            
            for j, sub_title in subchapters:
                # Look for headings that might already have IDs from markdown conversion
                # Try different heading levels (h2, h3, h4)
                anchor_added = False
                for h_level in ['h2', 'h3', 'h4']:
                    if anchor_added:
                        break
                    
                    # First, try to find heading with existing ID and replace the ID
                    # The markdown converter creates IDs like id="madness-and-civilization"
                    pattern = f'<{h_level}[^>]*id="[^"]*"[^>]*>([^<]*{re.escape(sub_title)}[^<]*)</{h_level}>'
                    if re.search(pattern, html_content, flags=re.IGNORECASE):
                        # Replace the existing ID
                        replacement = f'<{h_level} id="{chapter_index}-{j}">\\1</{h_level}>'
                        html_content = re.sub(pattern, replacement, html_content, count=1, flags=re.IGNORECASE)
                        anchor_added = True
                        logger.debug(f"Replaced anchor with #{chapter_index}-{j} for '{sub_title}'")
                        break
                    
                    # If exact match fails, try fuzzy matching
                    # Look for headings with similar text
                    import difflib
                    heading_pattern = f'<{h_level}[^>]*id="([^"]*)"[^>]*>([^<]+)</{h_level}>'
                    for match in re.finditer(heading_pattern, html_content):
                        heading_id = match.group(1)
                        heading_text = match.group(2)
                        similarity = difflib.SequenceMatcher(None, sub_title.lower(), heading_text.lower()).ratio()
                        if similarity > 0.8:  # 80% similarity threshold
                            # Replace this heading's ID
                            old_heading = match.group(0)
                            new_heading = f'<{h_level} id="{chapter_index}-{j}">{heading_text}</{h_level}>'
                            html_content = html_content.replace(old_heading, new_heading, 1)
                            anchor_added = True
                            logger.debug(f"Replaced anchor with #{chapter_index}-{j} for '{sub_title}' (fuzzy matched to '{heading_text}')")
                            break
                    
                    if anchor_added:
                        break
                    
                    # If no existing ID, try to find heading without ID
                    escaped_title = re.escape(sub_title)
                    pattern = f'<{h_level}>({escaped_title})</{h_level}>'
                    replacement = f'<{h_level} id="{chapter_index}-{j}">\\1</{h_level}>'
                    new_content = re.sub(pattern, replacement, html_content, count=1, flags=re.IGNORECASE)
                    if new_content != html_content:
                        html_content = new_content
                        anchor_added = True
                        logger.debug(f"Added anchor #{chapter_index}-{j} to '{sub_title}'")
                        break
                
                if not anchor_added:
                    # Try a more flexible match (handle minor variations)
                    # Look for any heading that contains key words from the title
                    for h_level in ['h2', 'h3', 'h4']:
                        # Try to match heading with existing ID first
                        pattern = f'<{h_level}[^>]*id="[^"]*"[^>]*>[^<]*</{h_level}>'
                        matches = re.finditer(pattern, html_content, flags=re.IGNORECASE)
                        for match in matches:
                            heading_text = match.group(0)
                            # Check if this heading contains the key words from sub_title
                            key_words = sub_title.upper().replace('AND', '').replace('THE', '').replace('OF', '').split()
                            if all(word in heading_text.upper() for word in key_words[:2] if len(word) > 3):
                                # Replace this heading's ID
                                new_heading = re.sub(r'id="[^"]*"', f'id="{chapter_index}-{j}"', heading_text)
                                html_content = html_content.replace(heading_text, new_heading)
                                logger.debug(f"Updated anchor to #{chapter_index}-{j} for approximate match of '{sub_title}'")
                                anchor_added = True
                                break
                        if anchor_added:
                            break
        
        # Create full HTML document
        html_doc = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>{chapter_title}</title>
    <link rel="stylesheet" type="text/css" href="../stylesheet.css"/>
</head>
<body>
    {html_content}
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_doc)
        
        logger.success(f"Converted {markdown_path.name} to HTML")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {markdown_path}: {e}")
        return False


def create_cover_html(cover_image_filename, book_title, output_path):
    """Create the cover HTML page."""
    cover_html = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>Cover</title>
    <link rel="stylesheet" type="text/css" href="../stylesheet.css"/>
</head>
<body>
    <div class="cover">
        <img src="../images/{cover_image_filename}" alt="{book_title} Cover" />
    </div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cover_html)
    
    logger.success(f"Created cover HTML: {output_path}")


def create_stylesheet(output_path):
    """Create the CSS stylesheet for the EPUB."""
    css_content = """/* Enhanced EPUB Stylesheet */
@namespace h "http://www.w3.org/1999/xhtml";

body {
    font-family: "Hiragino Mincho ProN", "MS Mincho", serif;
    line-height: 1.8;
    max-width: 800px;
    margin: 2em auto;
    padding: 0 1em;
    background-color: #fdfdfd;
    color: #333;
    text-align: justify;
}

h1 {
    text-align: center;
    margin-top: 1em;
    margin-bottom: 2em;
    font-weight: bold;
    font-size: 2em;
    border-bottom: 2px solid #ccc;
    padding-bottom: 0.5em;
    page-break-before: always;
}

h2 {
    font-size: 1.5em;
    font-weight: bold;
    margin-top: 2.5em;
    margin-bottom: 1em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.3em;
}

h3 {
    font-size: 1.2em;
    font-weight: bold;
    margin-top: 2em;
    margin-bottom: 0.8em;
}

h4 {
    font-size: 1.1em;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.6em;
}

p {
    margin-bottom: 1.2em;
    text-indent: 1em;
    text-align: justify;
}

/* No indent after headings or block elements */
h1 + p,
h2 + p,
h3 + p,
h4 + p,
blockquote + p,
ul + p,
ol + p,
pre + p {
    text-indent: 0;
}

blockquote {
    margin: 1.5em 2em;
    padding-left: 1em;
    border-left: 3px solid #ddd;
    font-style: italic;
    color: #555;
}

code {
    font-family: "Courier New", monospace;
    font-size: 0.9em;
    background-color: #f4f4f4;
    padding: 0.2em 0.4em;
    border-radius: 3px;
}

pre {
    font-family: "Courier New", monospace;
    font-size: 0.9em;
    background-color: #f4f4f4;
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
    white-space: pre-wrap;
    margin: 1em 0;
}

pre code {
    background-color: transparent;
    padding: 0;
}

ul, ol {
    margin: 1em 0;
    padding-left: 2em;
}

li {
    margin: 0.5em 0;
}

/* Links */
a {
    color: #0066cc;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1.5em 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 0.5em;
    text-align: left;
}

th {
    background-color: #f4f4f4;
    font-weight: bold;
}

/* Images */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.5em auto;
}

/* Superscript and subscript */
sup, sub {
    font-size: 0.8em;
    line-height: 0;
}

sup {
    vertical-align: super;
}

sub {
    vertical-align: sub;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2em 0;
}

.cover {
    text-align: center;
    page-break-after: always;
}

.cover img {
    max-width: 100%;
    height: auto;
}

.toc {
    page-break-after: always;
}

.toc ul {
    list-style-type: none;
    padding-left: 0;
}

.toc li {
    margin: 0.5em 0;
}

.toc a {
    text-decoration: none;
    color: #000;
}

/* Footnote references in text */
sup {
    font-size: 0.8em;
    vertical-align: super;
}

sup a, .footnote-ref {
    text-decoration: none;
    color: #0066cc;
}

sup a:hover, .footnote-ref:hover {
    text-decoration: underline;
}

/* Footnote section at end of chapter */
.footnote, .footnotes {
    margin-top: 2em;
    padding-top: 0.5em;
    border-top: none;  /* Remove top border since Notes h2 already has bottom border */
    font-size: 0.9em;
}

/* Keep the Notes h2 border for visual separation */
.footnote h2, .footnotes h2 {
    font-size: 1.2em;
    border-bottom: 1px solid #ddd;  /* Use same style as regular h2 */
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 0.8em;
}

/* Hide the hr element in footnote sections */
.footnote hr, .footnotes hr {
    display: none;
}

.footnote ol, .footnotes ol {
    padding-left: 1.5em;
    list-style-type: decimal;
    margin-top: 0;
}

.footnote li, .footnotes li, .footnote-item {
    margin-bottom: 0.8em;
    line-height: 1.6;
}

.footnote-item p {
    text-indent: 0;
    margin-bottom: 0.5em;
}

.footnote-backref, .footnote-item a[href^="#fnref"] {
    text-decoration: none;
    color: #0066cc;
    margin-left: 0.3em;
}

.footnote-backref:hover, .footnote-item a[href^="#fnref"]:hover {
    text-decoration: underline;
}"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(css_content)
    
    logger.success(f"Created stylesheet: {output_path}")


def create_toc_ncx(structure, book_title, book_uuid, output_path, parts_info=None, subchapter_locations=None):
    """Create the NCX table of contents file.
    
    Args:
        subchapter_locations: Dict mapping (chapter_idx, subchapter_idx) to part number
    """
    ncx_content = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="{book_uuid}"/>
    <meta name="dtb:depth" content="2"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle>
    <text>{book_title}</text>
  </docTitle>
  <navMap>
    <navPoint id="navpoint-toc" playOrder="1">
      <navLabel>
        <text>Table of Contents</text>
      </navLabel>
      <content src="text/toc.html"/>
    </navPoint>"""
    
    play_order = 2
    for i, chapter in enumerate(structure["chapters"], 1):
        chapter_title = chapter["title"]
        # Check if this chapter has parts
        chapter_key = str(i)
        if parts_info and chapter_key in parts_info and parts_info[chapter_key] > 1:
            # Multi-part chapter - link to first part
            chapter_file = f"chapter_{i}.part1.html"
        else:
            # Single file chapter
            chapter_file = f"chapter_{i}.html"
        
        ncx_content += f"""
    <navPoint id="navpoint-{i}" playOrder="{play_order}">
      <navLabel>
        <text>{chapter_title}</text>
      </navLabel>
      <content src="text/{chapter_file}"/>"""
        
        # Add subchapters if they exist
        if "subchapters" in chapter and chapter["subchapters"]:
            for j, subchapter in enumerate(chapter["subchapters"], 1):
                play_order += 1
                sub_title = subchapter["title"]
                
                # Determine which part contains this subchapter
                if subchapter_locations and (i, j) in subchapter_locations:
                    part_num = subchapter_locations[(i, j)]
                    if part_num and part_num > 1:
                        sub_chapter_file = f"chapter_{i}.part{part_num}.html"
                    else:
                        sub_chapter_file = chapter_file
                else:
                    # Default to main chapter file (or part1 for multi-part)
                    sub_chapter_file = chapter_file
                
                ncx_content += f"""
      <navPoint id="navpoint-{i}-{j}" playOrder="{play_order}">
        <navLabel>
          <text>{sub_title}</text>
        </navLabel>
        <content src="text/{sub_chapter_file}#{i}-{j}"/>
      </navPoint>"""
        
        ncx_content += """
    </navPoint>"""
        play_order += 1
    
    ncx_content += """
  </navMap>
</ncx>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ncx_content)
    
    logger.success(f"Created NCX TOC: {output_path}")


def create_toc_html(structure, book_title, output_path, parts_info=None, subchapter_locations=None):
    """Create the HTML table of contents.
    
    Args:
        subchapter_locations: Dict mapping (chapter_idx, subchapter_idx) to part number
    """
    toc_html = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>Table of Contents</title>
    <style type="text/css">
        body {{
            font-family: Georgia, "Times New Roman", serif;
            line-height: 1.8;
            margin: 2em auto;
            max-width: 800px;
            padding: 0 1em;
            background-color: #fdfdfd;
            color: #333;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 1.5em;
            color: #222;
            border-bottom: 3px solid #4a5568;
            padding-bottom: 0.5em;
            font-weight: bold;
            font-size: 2em;
        }}
        .toc {{
            margin: 0;
            padding: 0;
        }}
        .toc > ul {{
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        }}
        .toc > ul > li {{
            margin-bottom: 1.5em;
            padding-left: 1.5em;
            border-left: 4px solid #e2e8f0;
            transition: border-color 0.3s ease;
        }}
        .toc > ul > li:hover {{
            border-left-color: #4a5568;
        }}
        .toc a {{
            text-decoration: none;
            color: #2563eb;
            font-weight: 600;
            font-size: 1.1em;
            display: block;
            transition: color 0.2s ease;
        }}
        .toc a:hover {{
            color: #1d4ed8;
            text-decoration: underline;
        }}
        .chapter-number {{
            display: inline-block;
            min-width: 2em;
            margin-right: 0.5em;
            font-weight: bold;
            color: #555;
        }}
        /* Nested subchapters */
        .toc ul ul {{
            list-style-type: none;
            padding-left: 2em;
            margin-top: 0.5em;
            margin-bottom: 0;
        }}
        .toc ul ul li {{
            margin-bottom: 0.4em;
            padding-left: 1em;
            border-left: 2px solid #cbd5e1;
            transition: border-color 0.3s ease;
        }}
        .toc ul ul li:hover {{
            border-left-color: #64748b;
        }}
        .toc ul ul a {{
            font-size: 0.95em;
            font-weight: normal;
            color: #475569;
        }}
        .toc ul ul a:hover {{
            color: #1e293b;
        }}
        /* Part labels */
        .part-label {{
            font-size: 0.85em;
            color: #64748b;
            font-style: italic;
            margin-left: 0.5em;
        }}
    </style>
</head>
<body>
    <h1>Table of Contents</h1>
    <div class="toc">
        <ul>"""
    
    for i, chapter in enumerate(structure["chapters"], 1):
        chapter_title = chapter["title"]
        # Check if this chapter has parts
        chapter_key = str(i)
        if parts_info and chapter_key in parts_info and parts_info[chapter_key] > 1:
            # Multi-part chapter - link to first part
            chapter_file = f"chapter_{i}.part1.html"
        else:
            # Single file chapter
            chapter_file = f"chapter_{i}.html"
        
        toc_html += f"""
            <li>
                <a href="{chapter_file}">{chapter_title}</a>"""
        
        # Add subchapters if they exist
        if "subchapters" in chapter and chapter["subchapters"]:
            toc_html += """
                <ul>"""
            for j, subchapter in enumerate(chapter["subchapters"], 1):
                sub_title = subchapter["title"]
                # Determine which part contains this subchapter
                if subchapter_locations and (i, j) in subchapter_locations:
                    part_num = subchapter_locations[(i, j)]
                    if part_num and part_num > 1:
                        sub_chapter_file = f"chapter_{i}.part{part_num}.html"
                    else:
                        sub_chapter_file = chapter_file
                else:
                    # Default to main chapter file (or part1 for multi-part)
                    sub_chapter_file = chapter_file
                
                toc_html += f"""
                    <li><a href="{sub_chapter_file}#{i}-{j}">{sub_title}</a></li>"""
            toc_html += """
                </ul>"""
        
        toc_html += """
            </li>"""
    
    toc_html += """
        </ul>
    </div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(toc_html)
    
    logger.success(f"Created HTML TOC: {output_path}")


def create_container_xml(output_path):
    """Create the container.xml file for the EPUB."""
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(container_xml)
    
    logger.success(f"Created container.xml: {output_path}")


def create_mimetype(output_path):
    """Create the mimetype file for the EPUB."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("application/epub+zip")
    
    logger.success(f"Created mimetype: {output_path}")


def create_content_opf(structure, book_title, author, book_uuid, output_path, has_cover=False, all_html_files=None, images_dir=None):
    """Create the content.opf file."""
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    
    # If all_html_files not provided, fallback to old behavior
    if not all_html_files:
        all_html_files = [(i, None, f"chapter_{i}.html") for i in range(1, len(structure["chapters"]) + 1)]
    
    # Check if front_matter and back_matter exist
    has_front_matter = (output_path.parent / "text" / "front_matter.html").exists()
    has_back_matter = (output_path.parent / "text" / "back_matter.html").exists()
    
    opf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<package version="2.0" unique-identifier="BookId" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title>{book_title}</dc:title>
    <dc:creator opf:role="aut">{author}</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="BookId" opf:scheme="UUID">{book_uuid}</dc:identifier>
    <dc:date>{current_date}</dc:date>
    <meta name="generator" content="PDF2EPUB v3"/>"""
    
    if has_cover:
        opf_content += """
    <meta name="cover" content="cover-image"/>"""
    
    opf_content += """
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="stylesheet" href="stylesheet.css" media-type="text/css"/>"""
    
    if has_cover:
        opf_content += """
    <item id="cover" href="text/cover.html" media-type="application/xhtml+xml"/>
    <item id="cover-image" href="images/cover.jpg" media-type="image/jpeg"/>"""
    
    if has_front_matter:
        opf_content += """
    <item id="front_matter" href="text/front_matter.html" media-type="application/xhtml+xml"/>"""
    
    opf_content += """
    <item id="toc" href="text/toc.html" media-type="application/xhtml+xml"/>"""
    
    # Add all HTML files to manifest
    for chapter_idx, part_num, html_filename in sorted(all_html_files):
        if part_num is None:
            item_id = f"chapter_{chapter_idx}"
        else:
            item_id = f"chapter_{chapter_idx}_part{part_num}"
        opf_content += f"""
    <item id="{item_id}" href="text/{html_filename}" media-type="application/xhtml+xml"/>"""
    
    if has_back_matter:
        opf_content += """
    <item id="back_matter" href="text/back_matter.html" media-type="application/xhtml+xml"/>"""
    
    # Add all images to manifest (except cover which is already added)
    if images_dir and images_dir.exists():
        image_extensions = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp'
        }
        
        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() in image_extensions and img_file.name != "cover.jpg":
                media_type = image_extensions[img_file.suffix.lower()]
                item_id = img_file.stem.replace(' ', '_').replace('-', '_')
                opf_content += f"""
    <item id="{item_id}" href="images/{img_file.name}" media-type="{media_type}"/>"""
    
    opf_content += """
  </manifest>
  <spine toc="ncx">"""
    
    if has_cover:
        opf_content += """
    <itemref idref="cover" linear="no"/>"""
    
    if has_front_matter:
        opf_content += """
    <itemref idref="front_matter"/>"""
    
    opf_content += """
    <itemref idref="toc"/>"""
    
    # Add all HTML files to spine (reading order)
    for chapter_idx, part_num, html_filename in sorted(all_html_files):
        if part_num is None:
            item_id = f"chapter_{chapter_idx}"
        else:
            item_id = f"chapter_{chapter_idx}_part{part_num}"
        opf_content += f"""
    <itemref idref="{item_id}"/>"""
    
    if has_back_matter:
        opf_content += """
    <itemref idref="back_matter"/>"""
    
    opf_content += """
  </spine>
</package>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(opf_content)
    
    logger.success(f"Created content.opf: {output_path}")


def create_epub(book_title, epub_dir, output_filename="output.epub"):
    """Create the final EPUB file from the directory structure."""
    epub_path = Path("output") / book_title / output_filename
    
    # Create EPUB file
    with zipfile.ZipFile(epub_path, 'w', zipfile.ZIP_DEFLATED) as epub:
        # Add mimetype first (uncompressed as per EPUB spec)
        epub.write(epub_dir / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        
        # Add all other files
        for root, dirs, files in os.walk(epub_dir):
            for file in files:
                if file == "mimetype":
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(epub_dir)
                epub.write(file_path, arcname)
    
    logger.success(f"Created EPUB: {epub_path}")
    return epub_path




def main():
    parser = argparse.ArgumentParser(description="Generate EPUB from polished markdown files")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("-i", "--input", help="Path to PDF file for cover extraction")
    parser.add_argument("--translated", action="store_true", help="Use translated markdown instead of polished")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    author = config.get("author", "Unknown Author")
    
    if not book_title:
        logger.error("Book title not found in config.yaml")
        return
    
    # Load book structure
    structure = load_book_structure(book_title)
    if not structure:
        logger.error(f"Book structure not found for {book_title}")
        return
    
    # Determine which markdown directory to use and get display title
    display_title = book_title  # Default to original title
    
    if args.translated:
        markdown_dir = Path("output") / book_title / "translated"
        if not markdown_dir.exists():
            logger.error(f"Translated markdown directory not found: {markdown_dir}")
            logger.info("Please run translate command first")
            return
        logger.info(f"Using translated markdown from: {markdown_dir}")
        
        # Load or create translated book structure
        structure = load_or_create_translated_book_structure(book_title, structure, config)
        
        # Use translated title if available
        if 'book_title' in structure:
            display_title = structure['book_title']
            logger.info(f"Using translated title: {display_title}")
            # Use translated title for output filename (sanitize for filesystem)
            safe_title = "".join(c for c in display_title if c not in '<>:"/\\|?*')
            output_filename = f"{safe_title}.epub"
        else:
            output_filename = "output_translated.epub"
        
        # Setup paths for translated EPUB
        epub_dir = Path("output") / book_title / "epub_translated"
    else:
        markdown_dir = Path("output") / book_title / "polished_markdown"
        if not markdown_dir.exists():
            logger.error(f"Polished markdown directory not found: {markdown_dir}")
            logger.info("Please run polish_ocr_markdown.py first")
            return
        logger.info(f"Using polished markdown from: {markdown_dir}")
        
        # Use original title from structure if available
        if 'book_title' in structure:
            display_title = structure['book_title']
            # Use book title for output filename (sanitize for filesystem)
            safe_title = "".join(c for c in display_title if c not in '<>:"/\\|?*')
            output_filename = f"{safe_title}.epub"
        else:
            output_filename = "output.epub"
        
        # Setup paths for original EPUB
        epub_dir = Path("output") / book_title / "epub"
    
    text_dir = epub_dir / "text"
    images_dir = epub_dir / "images"
    meta_inf_dir = epub_dir / "META-INF"
    
    # Clean up old files from previous generations
    cleanup_old_files(epub_dir)
    
    # Create directories
    ensure_directory(epub_dir)
    ensure_directory(text_dir)
    ensure_directory(images_dir)
    ensure_directory(meta_inf_dir)
    
    # Generate UUID for the book
    book_uuid = str(uuid.uuid4())
    
    # Extract cover image
    cover_filename = ""
    has_cover = False
    
    # Determine PDF path
    if args.input:
        pdf_path = Path(args.input)
    else:
        # Check for input_original.pdf first, then fall back to input.pdf
        pdf_original_path = Path("output") / book_title / "input_original.pdf"
        pdf_path = Path("output") / book_title / "input.pdf"
        
        if pdf_original_path.exists():
            pdf_path = pdf_original_path
            logger.info(f"Using original PDF for cover extraction: {pdf_path}")
        elif pdf_path.exists():
            logger.info(f"Using PDF for cover extraction: {pdf_path}")
        else:
            logger.warning(f"No PDF file found for cover extraction. Looked for: {pdf_original_path} and {pdf_path}")
            pdf_path = None
    
    if pdf_path and pdf_path.exists():
        cover_filename = extract_cover_image(pdf_path, images_dir)
        if cover_filename:
            has_cover = True
            logger.info(f"Cover extracted from {pdf_path}")
            # Create cover HTML
            create_cover_html(cover_filename, display_title, text_dir / "cover.html")
        else:
            logger.warning(f"Failed to extract cover from {pdf_path}")
    else:
        logger.warning(f"PDF file not found for cover extraction: {pdf_path}")
    
    # Create stylesheet
    create_stylesheet(epub_dir / "stylesheet.css")
    
    # Create mimetype file
    create_mimetype(epub_dir / "mimetype")
    
    # Create container.xml
    create_container_xml(meta_inf_dir / "container.xml")
    
    # Copy all chapter images to EPUB images directory
    copy_chapter_images(markdown_dir, images_dir)
    
    # Load polish progress to get parts info for TOC generation
    # Load progress file based on mode
    if args.translated:
        progress_file = markdown_dir / "translation_progress.json"
    else:
        progress_file = markdown_dir / "polish_progress.json"
    parts_info_for_toc = {}
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            # Support both old and new format
            # Handle both polish and translation progress formats
            if "parts_info" in progress_data:
                parts_info_for_toc = {k: v["total_parts"] for k, v in progress_data["parts_info"].items() if v.get("is_complete", False)}
            else:
                parts_info_for_toc = progress_data.get("parts_polished", {})
    
    # Build subchapter locations mapping (which part contains which subchapter)
    subchapter_locations = {}
    for i, chapter in enumerate(structure["chapters"], 1):
        if "subchapters" in chapter and chapter["subchapters"]:
            chapter_key = str(i)
            if parts_info_for_toc and chapter_key in parts_info_for_toc and parts_info_for_toc[chapter_key] > 1:
                # Multi-part chapter - need to find which subchapters are in which parts
                for j, subchapter in enumerate(chapter["subchapters"], 1):
                    sub_title = subchapter["title"]
                    # Search through all parts to find where this subchapter is
                    for part_num in range(1, parts_info_for_toc[chapter_key] + 1):
                        part_file = markdown_dir / f"chapter_{i}.part{part_num}.md"
                        if part_file.exists():
                            with open(part_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Check if this subchapter heading is in this part
                                # Look for markdown headings with the subchapter title
                                # First try exact match
                                if re.search(f'^#+\\s*{re.escape(sub_title)}', content, re.MULTILINE | re.IGNORECASE):
                                    subchapter_locations[(i, j)] = part_num
                                    logger.debug(f"Found subchapter '{sub_title}' in part {part_num}")
                                    break
                                
                                # If exact match fails, try fuzzy matching
                                # Look for any heading that contains most of the important words
                                import difflib
                                lines = content.split('\n')
                                for line in lines:
                                    if line.strip().startswith('#'):
                                        # Extract the heading text
                                        heading = re.sub(r'^#+\s*', '', line).strip()
                                        # Use sequence matcher for fuzzy comparison
                                        similarity = difflib.SequenceMatcher(None, sub_title.lower(), heading.lower()).ratio()
                                        if similarity > 0.8:  # 80% similarity threshold
                                            subchapter_locations[(i, j)] = part_num
                                            logger.debug(f"Found subchapter '{sub_title}' (fuzzy matched to '{heading}') in part {part_num}")
                                            break
                                else:
                                    continue  # Continue to next part if not found
                                break  # Break from part loop if found
            else:
                # Single part chapter - all subchapters are in the main file
                for j, subchapter in enumerate(chapter["subchapters"], 1):
                    subchapter_locations[(i, j)] = None  # None means main file
    
    # Create TOC files
    create_toc_ncx(structure, display_title, book_uuid, epub_dir / "toc.ncx", parts_info_for_toc, subchapter_locations)
    create_toc_html(structure, display_title, text_dir / "toc.html", parts_info_for_toc, subchapter_locations)
    
    # Convert front matter to HTML if it exists
    front_matter_md = markdown_dir / "front_matter.md"
    if front_matter_md.exists():
        logger.info("Converting front matter to HTML")
        front_matter_html = text_dir / "front_matter.html"
        if convert_markdown_to_chapter_html(front_matter_md, front_matter_html, "Front Matter"):
            logger.success("Front matter converted to HTML")
        else:
            logger.warning("Failed to convert front matter")
    
    # Convert markdown chapters to HTML
    # Use the same parts_info we loaded for TOC
    parts_info = parts_info_for_toc
    
    # Track all HTML files created (for manifest and spine)
    all_html_files = []  # List of (chapter_index, part_num, html_filename)
    
    # Build chapters_dict based on known parts
    chapters_dict = {}
    for chapter_idx_str, num_parts in parts_info.items():
        chapter_index = int(chapter_idx_str)
        chapters_dict[chapter_index] = []
        
        if num_parts == 1:
            # Single file chapter
            md_file = markdown_dir / f"chapter_{chapter_index}.md"
            if md_file.exists():
                chapters_dict[chapter_index].append((None, md_file))
        else:
            # Multi-part chapter
            for part_num in range(1, num_parts + 1):
                md_file = markdown_dir / f"chapter_{chapter_index}.part{part_num}.md"
                if md_file.exists():
                    chapters_dict[chapter_index].append((part_num, md_file))
    
    # Also check for any chapters not in parts_info (backward compatibility)
    all_md_files = sorted(markdown_dir.glob("chapter_*.md"))
    for md_file in all_md_files:
        match = re.search(r'chapter_(\d+)(\.part(\d+))?\.md', md_file.name)
        if not match:
            continue
        
        chapter_index = int(match.group(1))
        part_num = int(match.group(3)) if match.group(3) else None
        
        # Only add if not already tracked
        if chapter_index not in chapters_dict:
            chapters_dict[chapter_index] = []
            chapters_dict[chapter_index].append((part_num, md_file))
    
    # Process each chapter and its parts
    for chapter_index in sorted(chapters_dict.keys()):
        # Get chapter title and subchapters from structure
        chapter_title = f"Chapter {chapter_index}"
        subchapter_titles = []
        if chapter_index <= len(structure["chapters"]):
            chapter_data = structure["chapters"][chapter_index - 1]
            chapter_title = chapter_data["title"]
            # Get subchapter titles for anchor generation
            if "subchapters" in chapter_data:
                subchapter_titles = [sub["title"] for sub in chapter_data["subchapters"]]
        
        # Sort parts by part number (None sorts before numbers)
        parts = sorted(chapters_dict[chapter_index], key=lambda x: x[0] if x[0] is not None else 0)
        
        all_parts_success = True
        
        for part_num, md_file in parts:
            if part_num is None:
                # Single file chapter or main file
                html_filename = f"chapter_{chapter_index}.html"
                html_path = text_dir / html_filename
                logger.info(f"Converting chapter {chapter_index}: {md_file.name}")
            else:
                # Part file
                html_filename = f"chapter_{chapter_index}.part{part_num}.html"
                html_path = text_dir / html_filename
                logger.info(f"Converting chapter {chapter_index} part {part_num}: {md_file.name}")
            
            # Convert markdown to HTML (use chapter title only for first part)
            part_title = chapter_title if part_num is None or part_num == 1 else f"{chapter_title} (continued)"
            
            # Find which subchapters are in this part (with their original indices)
            part_subchapters = []
            if subchapter_titles:
                for j, sub_title in enumerate(subchapter_titles, 1):
                    # Check if this subchapter is in the current part
                    if part_num is None:
                        # Single file chapter - all subchapters are here
                        part_subchapters.append((j, sub_title))
                    else:
                        # Multi-part chapter - check if subchapter is in this part
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if re.search(f'^#+\\s*{re.escape(sub_title)}', content, re.MULTILINE | re.IGNORECASE):
                                part_subchapters.append((j, sub_title))
            
            # Pass subchapter info for this specific part
            if part_subchapters:
                success = convert_markdown_to_chapter_html(md_file, html_path, part_title, chapter_index, part_subchapters)
            else:
                success = convert_markdown_to_chapter_html(md_file, html_path, part_title)
            
            if success:
                all_html_files.append((chapter_index, part_num, html_filename))
            else:
                all_parts_success = False
                logger.error(f"Failed to convert {md_file.name}")
                break
        
        if not all_parts_success:
            logger.error(f"Failed to convert chapter {chapter_index}, stopping")
            return
    
    # Convert back matter to HTML if it exists
    back_matter_md = markdown_dir / "back_matter.md"
    if back_matter_md.exists():
        logger.info("Converting back matter to HTML")
        back_matter_html = text_dir / "back_matter.html"
        if convert_markdown_to_chapter_html(back_matter_md, back_matter_html, "Back Matter"):
            logger.success("Back matter converted to HTML")
        else:
            logger.warning("Failed to convert back matter")
    
    # Create content.opf
    create_content_opf(structure, display_title, author, book_uuid, epub_dir / "content.opf", has_cover, all_html_files, images_dir)
    
    # Create the EPUB file
    epub_path = create_epub(book_title, epub_dir, output_filename)
    
    logger.success(f"EPUB generation complete!")
    logger.info(f"Output: {epub_path}")
    
    # Create password-protected ZIP file
    logger.info("Creating password-protected ZIP file...")
    zip_path, password = create_password_protected_zip(epub_path)
    
    if zip_path:
        logger.success(f"Password-protected ZIP created: {zip_path}")
        logger.info(f"The password is embedded in the filename: 【密码：{password}】")
    else:
        logger.warning("Failed to create password-protected ZIP file")


if __name__ == "__main__":
    main()