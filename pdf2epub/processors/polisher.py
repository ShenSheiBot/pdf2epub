"""
Polish processor for OCR-extracted markdown content.

This processor cleans up OCR-extracted markdown, removing artifacts,
fixing formatting, and organizing content structure.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from loguru import logger

from .base import BaseMarkdownProcessor
from .prompts import create_polish_prompt
from ..chapter_identity import ChapterIdentity

if TYPE_CHECKING:
    from ..core._protocol import ProcessContext


def convert_html_images_to_markdown(content: str) -> str:
    """
    Convert HTML img tags to markdown image syntax before sending to LLM.

    This prevents LLMs from incorrectly attempting to convert HTML to markdown
    and producing corrupted output like split `![` and `Image](...)`.

    Handles various wrapper elements: div, figure, p, span, center, etc.

    Args:
        content: Markdown content that may contain HTML img tags

    Returns:
        Content with HTML img tags converted to markdown syntax
    """

    def extract_img_info(img_tag: str) -> tuple:
        """Extract src and alt from an img tag."""
        # Extract src attribute
        src_match = re.search(r'src=["\']([^"\']+)["\']', img_tag, re.IGNORECASE)
        if not src_match:
            return None, None
        src = src_match.group(1)

        # Extract alt attribute (optional)
        alt_match = re.search(r'alt=["\']([^"\']*)["\']', img_tag, re.IGNORECASE)
        alt = alt_match.group(1) if alt_match else "Image"

        return src, alt

    def replace_img_tag(match: re.Match) -> str:
        """Replace an img tag with markdown syntax."""
        img_tag = match.group(0)
        src, alt = extract_img_info(img_tag)
        if src:
            return f"![{alt}]({src})"
        return img_tag  # Keep original if can't parse

    # Pattern to match wrapper elements containing img tags
    # Matches: <div ...>...<img ...>...</div>, <figure>...<img ...>...</figure>, etc.
    wrapper_tags = r'div|figure|p|span|center|aside|section'
    wrapper_pattern = re.compile(
        rf'<({wrapper_tags})[^>]*>\s*'  # Opening tag
        rf'(<img\s[^>]+/?\s*>)\s*'  # The img tag (captured)
        rf'</\1>',  # Matching closing tag
        re.IGNORECASE
    )

    def replace_wrapper(match: re.Match) -> str:
        """Replace wrapped img with markdown."""
        img_tag = match.group(2)
        src, alt = extract_img_info(img_tag)
        if src:
            return f"![{alt}]({src})"
        return match.group(0)  # Keep original if can't parse

    # Replace wrapped images (may need multiple passes for nested structures)
    prev_content = None
    while prev_content != content:
        prev_content = content
        content = wrapper_pattern.sub(replace_wrapper, content)

    # Then, replace any remaining standalone img tags
    standalone_img_pattern = re.compile(
        r'<img\s[^>]+/?\s*>',
        re.IGNORECASE
    )
    content = standalone_img_pattern.sub(replace_img_tag, content)

    return content


class PolishProcessor(BaseMarkdownProcessor):
    """Processor for polishing OCR-extracted markdown content."""

    def __init__(
        self,
        config: Dict,
        book_title: str,
        max_workers: int = 4,
        resume: bool = False,
        skip_truncation_check: bool = False,
        polish_models: Optional[List[Dict]] = None,
        content_type: str = "auto",
        use_longest_on_failure: bool = False,
        book_structure: Optional[Dict] = None
    ):
        """
        Initialize the polish processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            max_workers: Maximum number of concurrent workers
            resume: Whether to resume from previous progress
            skip_truncation_check: Whether to skip truncation detection
            polish_models: Optional override for model configurations
            content_type: Type of content ("academic", "japanese", "general", "auto")
            use_longest_on_failure: If True, use longest response when all attempts fail validation
            book_structure: Optional book structure from breakdown phase
        """
        super().__init__(
            config=config,
            book_title=book_title,
            input_dir="ocr_markdown",
            output_dir="polished_markdown",
            max_workers=max_workers,
            resume=resume,
            use_longest_on_failure=use_longest_on_failure
        )

        self.skip_truncation_check = skip_truncation_check
        self.polish_models = polish_models or config.get("polish", {}).get("models") or config.get("polish_models")
        self.book_structure = book_structure or {}

        # Determine content_type from book_structure if set to "auto"
        if content_type == "auto" and self.book_structure:
            language = self.book_structure.get('language', '').lower()
            is_vertical = self.book_structure.get('is_vertical_text', False)
            has_footnotes = self.book_structure.get('has_footnotes', False)

            if language == "japanese" and is_vertical:
                self.content_type = "japanese"
                logger.info("Auto-detected content type: japanese (vertical Japanese text)")
            elif has_footnotes:
                self.content_type = "academic"
                logger.info("Auto-detected content type: academic (has footnotes)")
            else:
                self.content_type = "general"
                logger.info("Auto-detected content type: general")
        else:
            self.content_type = content_type

        # Get processing mode from config (default to parallel for backward compatibility)
        self.processing_mode = config.get("polish_processing_mode", "parallel")

        # Check if book has global notes chapter
        self.use_global_footnotes = self._has_notes_chapter()

        # Log global footnote detection
        if self.use_global_footnotes:
            logger.info("Detected Notes chapter - using global footnote mode for academic content")

    def _has_notes_chapter(self) -> bool:
        """Check if book structure contains a notes chapter."""
        for chapter in self.book_structure.get('chapters', []):
            if chapter.get('type') == 'notes':
                return True
        return False

    def _get_chapter_info(self, file_name: str) -> Dict:
        """
        Get chapter information from book structure.

        Args:
            file_name: The markdown file name (e.g., "chapter_12.md", "chapter_7.1.1.md")

        Returns:
            Chapter info dict with 'type', 'title', etc., or empty dict if not found
        """
        # Use ChapterIdentity to parse the filename
        identity = ChapterIdentity.parse(file_name)
        if not identity or not identity.number:
            return {}

        # Get the index path for hierarchical lookup
        index_path = identity.index_path

        # Navigate through the tree structure
        chapters = self.book_structure.get('chapters', [])
        current_level = chapters

        for idx in index_path:
            # Convert 1-based index to 0-based
            array_idx = idx - 1

            if 0 <= array_idx < len(current_level):
                current = current_level[array_idx]
                # Move to children for next iteration
                current_level = current.get('children', [])
            else:
                return {}

        return current

    def get_operation_name(self, file_name: str) -> str:
        """Get the operation name for logging."""
        # Use ChapterIdentity for consistent parsing
        identity = ChapterIdentity.parse(file_name)

        if identity:
            if identity.is_front_matter:
                return "Front Matter"
            elif identity.is_back_matter:
                return "Back Matter"
            elif identity.number:
                return f"Chapter {identity.number}"

        return file_name

    def get_model_configs(self) -> List[Dict]:
        """Get the model configurations for polishing."""
        return self.polish_models

    def build_prompt(self, content: str, context: "ProcessContext") -> Any:
        """
        Build the polishing prompt with optional conversation history.

        Args:
            content: Content to polish
            context: Processing context with file info, part info, previous context

        Returns:
            Multi-part content for LLM (may include conversation history)
        """
        file_name = context.file_key
        part_idx = context.part_index or 1
        total_parts = context.total_parts or 1

        # Build previous_context dict if context injection is available
        previous_context = None
        if context.has_previous_context:
            previous_context = {
                'original': context.previous_original,
                'processed': context.previous_processed
            }

        # Preprocess: convert HTML img tags to markdown to avoid LLM corruption
        content = convert_html_images_to_markdown(content)

        # Get chapter info for title and type
        chapter_info = self._get_chapter_info(file_name)
        is_notes_chapter = chapter_info.get('type') == 'notes' or context.is_notes_chapter

        # Use actual chapter title if available, otherwise use operation name for logging
        chapter_name = context.chapter_title or chapter_info.get('title') or self.get_operation_name(file_name)

        # Also detect Notes chapter by content (fallback for mapping issues)
        if not is_notes_chapter:
            content_start = content[:500].lower()
            if any(marker in content_start for marker in ['# notes\n', '# notes \n', '## notes\n', '# references\n', '# bibliography\n']):
                if '[^' not in content and re.search(r'^\d+\.\s+\w', content, re.MULTILINE):
                    is_notes_chapter = True
                    logger.info(f"Detected {chapter_name} as Notes chapter based on content")

        # Use the pure function from prompts module
        prompt = create_polish_prompt(
            chapter_name=chapter_name,
            book_title=self.book_title,
            part_idx=part_idx,
            total_parts=total_parts,
            content=content,
            content_type=self.content_type,
            use_global_footnotes=self.use_global_footnotes or context.has_global_footnotes,
            is_notes_chapter=is_notes_chapter,
            previous_part_context=previous_context
        )

        # Build multi-part content for the LLM
        if previous_context and part_idx > 1:
            # Use conversation history for context continuity
            prev_user_content = [
                {"type": "text", "text": prompt + f"\n\nContent to polish (part {part_idx-1}/{total_parts}):"},
                {"type": "text", "text": previous_context['original']}
            ]
            current_user_content = [
                {"type": "text", "text": f"Now polish part {part_idx}/{total_parts} (continuation of the same chapter).\n\nIMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2). Convert any # (H1) headings to ## (H2).\n\nContent to polish:"},
                {"type": "text", "text": content}
            ]
            return [
                {"role": "user", "content": prev_user_content},
                {"role": "assistant", "content": previous_context['processed']},
                {"role": "user", "content": current_user_content}
            ]
        else:
            # No previous context, use standard format
            return [
                {"type": "text", "text": prompt},
                {"type": "text", "text": content}
            ]

    def post_process(self, result: str, context: "ProcessContext") -> str:
        """
        Post-process the polished result.

        Applies markdown cleanup. Image restoration is handled by hooks.

        Args:
            result: Cleaned LLM response
            context: Processing context

        Returns:
            Post-processed result
        """
        # Apply markdown post-processing
        return self._post_process_markdown(result)

    def _post_process_markdown(self, markdown: str) -> str:
        """Post-process the polished markdown to clean up any issues."""
        # Remove any leading/trailing whitespace
        markdown = markdown.strip()

        # Remove trailing empty "Notes" header (academic content artifact)
        # Pattern: "### Notes" or similar at end of file with no content
        markdown = re.sub(r'\n#{1,6}\s+Notes\s*$', '', markdown, flags=re.IGNORECASE)
        markdown = markdown.strip()

        # Remove images that point to non-existent files
        # Find all markdown images: ![alt](src)
        md_image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        # Find all HTML images: <img src="..." />
        html_image_pattern = r'(?:<div[^>]*>)?\s*<img\s+[^>]*src="([^"]+)"[^>]*/?\s*>\s*(?:</div>)?'
        images_to_remove = []

        # Check markdown images
        for match in re.finditer(md_image_pattern, markdown):
            image_path = match.group(2)

            # Skip URLs (http/https)
            if image_path.startswith(('http://', 'https://')):
                continue

            # Check if it's a relative path that should exist
            if image_path.startswith('../'):
                resolved_path = self.output_dir.parent / image_path[3:]
                if not resolved_path.exists():
                    images_to_remove.append(match.group(0))
                    logger.debug(f"Removing markdown image with non-existent file: {image_path}")
            elif not image_path.startswith(('/', 'http')):
                images_to_remove.append(match.group(0))
                logger.debug(f"Removing markdown image with invalid path: {image_path}")

        # Check HTML images
        for match in re.finditer(html_image_pattern, markdown, re.IGNORECASE):
            image_path = match.group(1)

            # Skip URLs (http/https)
            if image_path.startswith(('http://', 'https://')):
                continue

            # Check if it's a relative path that should exist
            if image_path.startswith('../'):
                resolved_path = self.output_dir.parent / image_path[3:]
                if not resolved_path.exists():
                    images_to_remove.append(match.group(0))
                    logger.debug(f"Removing HTML image with non-existent file: {image_path}")
            elif not image_path.startswith(('/', 'http')):
                images_to_remove.append(match.group(0))
                logger.debug(f"Removing HTML image with invalid path: {image_path}")

        # Remove the non-existent images
        for image_markdown in images_to_remove:
            markdown = markdown.replace(image_markdown, '')

        if images_to_remove:
            logger.info(f"Removed {len(images_to_remove)} images with non-existent files")

        # Fix common markdown issues
        # 1. Ensure headers have space after #
        markdown = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', markdown, flags=re.MULTILINE)

        # 2. Ensure blank lines around headers
        markdown = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(#{1,6} [^\n]+)\n([^\n#])', r'\1\n\n\2', markdown)

        # 3. Remove excessive blank lines (more than 2)
        markdown = re.sub(r'\n{4,}', '\n\n', markdown)

        # 4. Ensure valid images have blank lines around them
        markdown = re.sub(r'([^\n])\n(!\[)', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(!\[[^\]]*\]\([^\)]*\))\n([^\n])', r'\1\n\n\2', markdown)

        return markdown
