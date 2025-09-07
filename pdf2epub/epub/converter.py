"""
Content converter for EPUB generation.

This module handles all content transformation tasks including:
- Markdown to HTML conversion
- Image processing and optimization
- Content cleaning and normalization
"""

import re
import shutil
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from loguru import logger

from ..markdown_to_html import convert_markdown_to_html


class ContentConverter:
    """Handles all content conversion and preparation for EPUB generation."""

    def __init__(self, config, footnote_manager=None):
        """
        Initialize the ContentConverter.

        Args:
            config: EpubConfig instance with all settings
            footnote_manager: Optional FootnoteManager instance for handling cross-chapter footnotes
        """
        self.config = config
        self.footnote_manager = footnote_manager

    def remove_duplicate_titles(self) -> int:
        """
        Remove duplicate titles in two cases:
        1. Level 2 heading immediately follows level 1 heading with same/similar text
        2. Consecutive headings at the same level with exactly the same title (keep first)

        Returns:
            Number of duplicate titles removed
        """
        total_removed = 0
        markdown_dir = self.config.markdown_dir

        # Process all markdown files
        for md_file in markdown_dir.glob("*.md"):
            if not md_file.name.startswith(("chapter_", "front_matter", "back_matter")):
                continue

            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                removed_count = 0
                i = 0

                while i < len(lines):
                    current_line = lines[i]

                    # First check: level 1 -> level 2 duplicates (existing logic)
                    if current_line.strip().startswith(
                        "# "
                    ) and not current_line.strip().startswith("## "):
                        # Extract title from level 1 heading
                        title1 = current_line.strip()[2:].strip()

                        # Look ahead for next heading (skip author names and other short content)
                        j = i + 1
                        lines_between = []

                        # Skip up to 5 non-heading lines (could be author name, empty lines, etc.)
                        while j < len(lines) and not lines[j].strip().startswith("#"):
                            lines_between.append(lines[j])
                            j += 1
                            # Don't look too far ahead
                            if j - i > 5:
                                break

                        # Check if we found a level 2 heading
                        if (
                            j < len(lines)
                            and lines[j].strip().startswith("## ")
                            and not lines[j].strip().startswith("### ")
                        ):
                            # Extract title from level 2 heading
                            title2 = lines[j].strip()[3:].strip()

                            # Check for similarity
                            similarity = difflib.SequenceMatcher(
                                None, title1.lower(), title2.lower()
                            ).ratio()

                            # If titles are very similar (>80% match) or identical, remove the level 2 heading
                            if similarity > 0.8 or title1.lower() == title2.lower():
                                logger.debug(
                                    f"Removing duplicate title in {md_file.name}: '## {title2}' (follows '# {title1}')"
                                )
                                new_lines.append(current_line)  # Keep level 1 heading
                                new_lines.extend(
                                    lines_between
                                )  # Keep lines between (like author name)
                                # Add a newline to replace the removed heading (maintain spacing)
                                new_lines.append("\n")
                                i = j + 1  # Skip the level 2 heading
                                removed_count += 1
                                continue

                    # Second check: same-level consecutive duplicates
                    if current_line.strip().startswith("#"):
                        # Extract heading level and title
                        stripped = current_line.strip()
                        level = 0
                        for char in stripped:
                            if char == "#":
                                level += 1
                            else:
                                break

                        # Get the title (skip the # characters and following space)
                        if (
                            level > 0
                            and len(stripped) > level
                            and stripped[level] == " "
                        ):
                            current_title = stripped[level + 1 :].strip()

                            # Look ahead for consecutive headings at the same level with same title
                            j = i + 1
                            duplicates_found = []

                            while j < len(lines):
                                next_line = lines[j]

                                # Check if it's a heading at the same level
                                if next_line.strip().startswith(
                                    "#" * level + " "
                                ) and not next_line.strip().startswith(
                                    "#" * (level + 1)
                                ):
                                    # Extract title from next heading
                                    next_title = next_line.strip()[level + 1 :].strip()

                                    # Check if titles are exactly the same
                                    if current_title == next_title:
                                        duplicates_found.append(j)
                                        logger.debug(
                                            f"Found same-level duplicate in {md_file.name}: "
                                            f"{'#' * level} {next_title} at line {j + 1}"
                                        )
                                        j += 1
                                        continue
                                    else:
                                        # Different title at same level, stop looking
                                        break
                                elif next_line.strip().startswith("#"):
                                    # Different level heading, stop looking
                                    break
                                else:
                                    # Non-heading line, continue looking
                                    j += 1
                                    # Don't look too far ahead
                                    if j - i > 10:
                                        break

                            # If we found duplicates, skip them
                            if duplicates_found:
                                new_lines.append(
                                    current_line
                                )  # Keep the first occurrence

                                # Add all lines between current and first duplicate
                                for k in range(i + 1, duplicates_found[0]):
                                    new_lines.append(lines[k])

                                # Skip all duplicate headings, but preserve content between them
                                last_duplicate = duplicates_found[0]
                                for dup_idx in duplicates_found:
                                    # Add content between this duplicate and the next
                                    if dup_idx != duplicates_found[-1]:
                                        next_dup = duplicates_found[
                                            duplicates_found.index(dup_idx) + 1
                                        ]
                                        for k in range(dup_idx + 1, next_dup):
                                            if (
                                                not lines[k]
                                                .strip()
                                                .startswith("#" * level + " ")
                                            ):
                                                new_lines.append(lines[k])
                                    removed_count += 1
                                    last_duplicate = dup_idx

                                # Continue from after the last duplicate
                                i = last_duplicate + 1
                                continue

                    new_lines.append(current_line)
                    i += 1

                # Write back if any changes were made
                if removed_count > 0:
                    with open(md_file, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    logger.info(
                        f"Removed {removed_count} duplicate titles from {md_file.name}"
                    )
                    total_removed += removed_count

            except Exception as e:
                logger.error(f"Failed to remove duplicate titles in {md_file}: {e}")

        if total_removed > 0:
            logger.success(f"Total duplicate titles removed: {total_removed}")

        return total_removed

    def clean_invalid_headings(self) -> int:
        """
        Remove invalid heading lines from markdown files.

        Invalid headings are:
        - Empty headings (just # markers with no text)
        - Too-long headings (>50 chars) for level 2+ headings
        - Exception: Numbered sections (like "2.3 Section Title") are kept regardless of length

        Returns:
            Number of headings removed
        """
        total_removed = 0
        markdown_dir = self.config.markdown_dir

        # Process all markdown files
        for md_file in markdown_dir.glob("*.md"):
            if not md_file.name.startswith(("chapter_", "front_matter", "back_matter")):
                continue

            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                removed_count = 0

                for line in lines:
                    should_keep = True
                    stripped = line.strip()

                    # Check if it's a heading line
                    if stripped.startswith("#"):
                        # Extract the heading level and text
                        heading_match = re.match(r"^(#+)\s*(.*)", stripped)
                        if heading_match:
                            markers = heading_match.group(1)
                            title = heading_match.group(2).strip()

                            # Remove if empty or too long (for level 2+ headings)
                            # BUT keep if it starts with a number and doesn't end with a period (like "2.3 Section Title")
                            if len(markers) >= 2:  # ## or more
                                # Check if title starts with a number and doesn't end with period
                                starts_with_number = title and title[0].isdigit()
                                ends_with_period = title and title.endswith(".")
                                is_numbered_section = (
                                    starts_with_number and not ends_with_period
                                )

                                if not title:
                                    logger.debug(
                                        f"Removing empty heading in {md_file.name}: {stripped}"
                                    )
                                    should_keep = False
                                    removed_count += 1
                                elif len(title) > 50 and not is_numbered_section:
                                    logger.debug(
                                        f"Removing too-long heading ({len(title)} chars) in {md_file.name}: {title[:50]}..."
                                    )
                                    should_keep = False
                                    removed_count += 1

                    if should_keep:
                        new_lines.append(line)

                # Write back if any changes were made
                if removed_count > 0:
                    with open(md_file, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    logger.info(
                        f"Removed {removed_count} invalid headings from {md_file.name}"
                    )
                    total_removed += removed_count

            except Exception as e:
                logger.error(f"Failed to clean headings in {md_file}: {e}")

        if total_removed > 0:
            logger.success(f"Total invalid headings removed: {total_removed}")

        return total_removed

    def copy_chapter_images(
        self,
        dest_dir: Path,
        compress: bool = True,
        max_width: int = 1600,
        jpeg_quality: int = 85,
    ) -> Tuple[int, Dict[str, str]]:
        """
        Copy all chapter images from source to destination directory with optional compression.

        Args:
            dest_dir: Destination directory for images
            compress: Whether to compress images
            max_width: Maximum width for compressed images
            jpeg_quality: JPEG quality for compression

        Returns:
            Tuple of (number of images processed, mapping of original to new filenames)
        """
        source_dir = self.config.markdown_dir
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}
        images_processed = 0
        total_size_before = 0
        total_size_after = 0
        image_mapping = {}  # Maps original filename to new filename

        # Look for images in the parent directory of polished_markdown (usually 'images' folder)
        images_source = source_dir.parent / "images"
        if images_source.exists():
            for img_file in images_source.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    total_size_before += img_file.stat().st_size

                    if compress and img_file.suffix.lower() not in [".svg", ".gif"]:
                        # Compress and convert to JPEG (except SVG and GIF)
                        dest_path = dest_dir / img_file.stem
                        final_path = self._compress_and_copy_image(
                            img_file, dest_path, max_width, jpeg_quality
                        )
                    else:
                        # Copy as-is for SVG, GIF, or if compression disabled
                        dest_path = dest_dir / img_file.name
                        shutil.copy2(img_file, dest_path)
                        final_path = dest_path

                    total_size_after += final_path.stat().st_size
                    images_processed += 1
                    image_mapping[img_file.name] = final_path.name
                    logger.debug(
                        f"Processed image: {img_file.name} -> {final_path.name}"
                    )

        # Also look for images in the polished_markdown directory itself
        for img_file in source_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                total_size_before += img_file.stat().st_size

                if compress and img_file.suffix.lower() not in [".svg", ".gif"]:
                    # Compress and convert to JPEG
                    dest_path = dest_dir / img_file.stem
                    final_path = self._compress_and_copy_image(
                        img_file, dest_path, max_width, jpeg_quality
                    )
                else:
                    # Copy as-is
                    dest_path = dest_dir / img_file.name
                    shutil.copy2(img_file, dest_path)
                    final_path = dest_path

                total_size_after += final_path.stat().st_size
                images_processed += 1
                image_mapping[img_file.name] = final_path.name
                logger.debug(f"Processed image: {img_file.name} -> {final_path.name}")

        if images_processed > 0:
            if compress and total_size_before > 0:
                reduction = (1 - total_size_after / total_size_before) * 100
                logger.success(
                    f"Processed {images_processed} images (size reduced by {reduction:.1f}%)"
                )
            else:
                logger.success(f"Copied {images_processed} images to EPUB")

        return images_processed, image_mapping

    def _compress_and_copy_image(
        self, source_path: Path, dest_path_stem: Path, max_width: int, jpeg_quality: int
    ) -> Path:
        """
        Compress and copy an image, converting to JPEG if beneficial.

        Args:
            source_path: Source image path
            dest_path_stem: Destination path without extension
            max_width: Maximum width for the image
            jpeg_quality: JPEG quality (1-100)

        Returns:
            Path to the final saved image
        """
        try:
            img = Image.open(source_path)

            # Convert RGBA/LA/P to RGB if necessary (for JPEG compatibility)
            if img.mode in ("RGBA", "LA", "P"):
                # Create a white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if "A" in img.mode:
                    background.paste(
                        img, mask=img.split()[-1]
                    )  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert("RGB")
            elif img.mode not in ["RGB", "L"]:
                img = img.convert("RGB")

            # Resize if too large
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            # Save as JPEG
            dest_path = Path(str(dest_path_stem) + ".jpg")
            img.save(dest_path, "JPEG", quality=jpeg_quality, optimize=True)
            return dest_path

        except Exception as e:
            logger.warning(f"Failed to compress {source_path}: {e}. Copying as-is.")
            # Fall back to simple copy
            dest_path = Path(str(dest_path_stem) + source_path.suffix)
            shutil.copy2(source_path, dest_path)
            return dest_path

    def extract_cover_image(self, pdf_path: Path, output_dir: Path) -> Optional[str]:
        """
        Extract the first page of the PDF as the cover image.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save the cover image

        Returns:
            Filename of the cover image or None if failed
        """
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

    def create_xhtml_document(
        self, title: str, body_content: str, css_path: str = "../stylesheet.css"
    ) -> str:
        """
        Create a standard XHTML document wrapper with language support.

        Args:
            title: The document title
            body_content: The HTML content for the body
            css_path: Path to the CSS file (relative to the HTML file)

        Returns:
            Complete XHTML document as string
        """
        # Get the language from config
        language = self.config.language if hasattr(self.config, "language") else "en"

        # Create language class for CSS styling
        lang_class = f"lang-{language}"

        return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>{title}</title>
    <link rel="stylesheet" type="text/css" href="{css_path}"/>
</head>
<body class="{lang_class}">
    {body_content}
</body>
</html>"""

    def convert_markdown_to_chapter_html(
        self,
        markdown_path: Path,
        output_path: Path,
        chapter_title: str,
        chapter_index: Optional[int] = None,
        subchapter_info: Optional[List] = None,
        image_mapping: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Convert a markdown file to HTML chapter format with proper anchors for subchapters.

        Args:
            markdown_path: Path to the markdown file
            output_path: Path to save the HTML file
            chapter_title: Title of the chapter
            chapter_index: Index of the chapter (for anchor generation)
            subchapter_info: List of tuples (subchapter_index, subchapter_title) or list of titles
            image_mapping: Mapping of original image names to new names

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # Ensure image_mapping is a dict or None
            if image_mapping is not None and not isinstance(image_mapping, dict):
                logger.warning(
                    f"Invalid image_mapping type: {type(image_mapping)}. Setting to None."
                )
                image_mapping = None

            # Convert markdown to HTML (just the body content, not a full document)
            # Pass footnote_manager and source chapter for cross-chapter footnote support
            source_chapter = markdown_path.stem  # e.g., "chapter_1" or "chapter_7"
            html_content = convert_markdown_to_html(
                markdown_content,
                standalone=False,
                image_mapping=image_mapping,
                footnote_manager=self.footnote_manager,
                source_chapter=source_chapter,
            )

            # Add/update anchors to subchapter headings if we have the info
            if chapter_index and subchapter_info:
                html_content = self._add_subchapter_anchors(
                    html_content, chapter_index, subchapter_info
                )

            # Wrap in XHTML structure using the helper function
            full_html = self.create_xhtml_document(chapter_title, html_content)

            # Write HTML file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_html)

            logger.success(f"Converted {markdown_path.name} to HTML")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to convert {markdown_path} to HTML: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            logger.debug(
                f"image_mapping type: {type(image_mapping)}, value: {image_mapping}"
            )
            return False

    def get_chapter_html_filename(
        self,
        chapter_index: int,
        parts_info: Optional[Dict] = None,
        chapters_with_parts: Optional[set] = None,
    ) -> str:
        """
        Determine the HTML filename for a chapter based on whether it has parts.

        Args:
            chapter_index: The chapter number (1-based)
            parts_info: Dict with chapter keys and number of parts
            chapters_with_parts: Set of chapter indices that have part files

        Returns:
            The HTML filename for the chapter (links to part1 if multi-part)
        """
        chapter_key = str(chapter_index)
        if chapters_with_parts and chapter_index in chapters_with_parts:
            # This chapter has part files - link to first part
            return f"chapter_{chapter_index}_part1.html"
        elif parts_info and chapter_key in parts_info and parts_info[chapter_key] > 1:
            # Multi-part chapter according to progress info - link to first part
            return f"chapter_{chapter_index}_part1.html"
        else:
            # Single file chapter
            return f"chapter_{chapter_index}.html"

    def get_subchapter_html_file(
        self,
        chapter_index: int,
        subchapter_index: int,
        subchapter_locations: Optional[Dict],
        chapter_file: str,
        subchapter_title: str = None,
    ) -> str:
        """
        Determine which HTML file contains a specific subchapter with fuzzy matching support.

        Args:
            chapter_index: The chapter number (1-based)
            subchapter_index: The subchapter number (1-based)
            subchapter_locations: Dict mapping (chapter_idx, subchapter_idx) to part number
            chapter_file: Default chapter file if location not found
            subchapter_title: Optional title for fuzzy matching

        Returns:
            The HTML filename containing the subchapter
        """
        # First try exact match with title key
        if subchapter_locations and subchapter_title:
            location_key = f"{chapter_index}:{subchapter_title}"
            if location_key in subchapter_locations:
                info = subchapter_locations[location_key]
                if isinstance(info, dict) and "part" in info:
                    part_num = info["part"]
                    return f"chapter_{chapter_index}_part{part_num}.html"

            # Try fuzzy matching
            if "_all_titles" in subchapter_locations:
                chapter_titles = subchapter_locations["_all_titles"].get(
                    str(chapter_index), []
                )
                if chapter_titles:
                    import difflib

                    best_match = None
                    best_ratio = 0
                    for title_info in chapter_titles:
                        similarity = difflib.SequenceMatcher(
                            None, subchapter_title.lower(), title_info["title"].lower()
                        ).ratio()
                        if similarity > best_ratio and similarity > 0.8:
                            best_ratio = similarity
                            best_match = title_info

                    if best_match:
                        logger.debug(
                            f"Fuzzy matched subchapter '{subchapter_title}' to '{best_match['title']}'"
                        )
                        return f"chapter_{chapter_index}_part{best_match['part']}.html"

        # Fall back to old tuple key
        if (
            subchapter_locations
            and (chapter_index, subchapter_index) in subchapter_locations
        ):
            part_num = subchapter_locations[(chapter_index, subchapter_index)]
            if part_num is None:
                # In main file (single file chapter)
                return f"chapter_{chapter_index}.html"
            else:
                # In a specific part
                return f"chapter_{chapter_index}_part{part_num}.html"
        else:
            # Default to chapter file
            return chapter_file

    def _add_subchapter_anchors(
        self, html_content: str, chapter_index: int, subchapter_info: List
    ) -> str:
        """
        Add anchors to subchapter headings for table of contents navigation.

        Args:
            html_content: HTML content to process
            chapter_index: Index of the chapter
            subchapter_info: List of subchapter titles or (index, title) tuples

        Returns:
            Modified HTML content with anchors added
        """
        # The structure from build_structure_from_markdown is a list of dicts: [{'title': '...'}, ...]
        # We extract the title from each dict.
        if subchapter_info and isinstance(subchapter_info[0], dict):
            subchapters = [
                (j, info.get("title", "")) for j, info in enumerate(subchapter_info, 1)
            ]
        elif subchapter_info and isinstance(subchapter_info[0], tuple):
            subchapters = subchapter_info  # Legacy format
        else:
            # Convert simple list of strings to tuples with indices
            subchapters = [(j, title) for j, title in enumerate(subchapter_info, 1)]

        import html as html_module

        for j, sub_title in subchapters:
            # Decode HTML entities in the subtitle for better matching
            sub_title_decoded = html_module.unescape(sub_title)

            # Look for headings that might already have IDs from markdown conversion
            # Try different heading levels (h2, h3, h4)
            anchor_added = False
            for h_level in ["h2", "h3", "h4"]:
                if anchor_added:
                    break

                # First, try to find heading with existing ID and replace the ID
                # Try both the original and decoded versions
                for title_variant in [sub_title, sub_title_decoded]:
                    pattern = f'<{h_level}[^>]*id="[^"]*"[^>]*>([^<]*{re.escape(title_variant)}[^<]*)</{h_level}>'
                    if re.search(pattern, html_content, flags=re.IGNORECASE):
                        # Replace the existing ID
                        replacement = (
                            f'<{h_level} id="{chapter_index}-{j}">\\1</{h_level}>'
                        )
                        html_content = re.sub(
                            pattern,
                            replacement,
                            html_content,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                        anchor_added = True
                        logger.debug(
                            f"Replaced anchor with #{chapter_index}-{j} for '{sub_title}'"
                        )
                        break

                if anchor_added:
                    break

                # If exact match fails, try fuzzy matching
                heading_pattern = (
                    f'<{h_level}[^>]*id="([^"]*)"[^>]*>([^<]+)</{h_level}>'
                )
                for match in re.finditer(heading_pattern, html_content):
                    heading_text = match.group(2)
                    # Decode HTML entities in heading text too
                    heading_text_decoded = html_module.unescape(heading_text)

                    # Compare both decoded versions for better matching
                    similarity = difflib.SequenceMatcher(
                        None, sub_title_decoded.lower(), heading_text_decoded.lower()
                    ).ratio()
                    if similarity > 0.8:  # 80% similarity threshold
                        # Replace this heading's ID
                        old_heading = match.group(0)
                        new_heading = f'<{h_level} id="{chapter_index}-{j}">{heading_text}</{h_level}>'
                        html_content = html_content.replace(old_heading, new_heading, 1)
                        anchor_added = True
                        logger.debug(
                            f"Replaced anchor with #{chapter_index}-{j} for '{sub_title}' (fuzzy matched to '{heading_text}')"
                        )
                        break

                if anchor_added:
                    break

                # If no existing ID, try to find heading without ID
                # Try both original and decoded versions
                for title_variant in [sub_title, sub_title_decoded]:
                    escaped_title = re.escape(title_variant)
                    pattern = f"<{h_level}>({escaped_title})</{h_level}>"
                    replacement = f'<{h_level} id="{chapter_index}-{j}">\\1</{h_level}>'
                    new_content = re.sub(
                        pattern, replacement, html_content, count=1, flags=re.IGNORECASE
                    )
                    if new_content != html_content:
                        html_content = new_content
                        anchor_added = True
                        logger.debug(
                            f"Added anchor #{chapter_index}-{j} to '{sub_title}'"
                        )
                        break

                if anchor_added:
                    break

        return html_content
