"""
Polish processor for OCR-extracted markdown content.

This processor cleans up OCR-extracted markdown, removing artifacts,
fixing formatting, and organizing content structure.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import tiktoken

from .base import BaseMarkdownProcessor
from .utils.truncation import NGramTruncationDetector
from .utils.agent_verifier import PolishVerificationAgent
from .utils.verification_tools import VerificationTools, VerificationFile
from .utils.image_restore import restore_lost_images
from .utils.split_manager import SplitManager
from .tracker import ProcessingTracker
from .prompts import create_polish_prompt, detect_content_type
from ..chapter_identity import ChapterIdentity

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


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

        # Enable batch validation mode
        self.validation_mode = "batch"
        self.auto_save = False

        self.skip_truncation_check = skip_truncation_check
        self.polish_models = polish_models or config.get("polish_models")
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

        # Initialize n-gram truncation detector for fast screening
        self.ngram_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=0.60,
            allow_deduplication=True
        )

        # Initialize agent verifier for batch validation
        # Will be initialized lazily when needed
        self._agent_verifier = None

        # Initialize ProcessingTracker for audit trail
        tracker_path = self.output_dir / "processing_tracker.json"
        self.processing_tracker = ProcessingTracker(tracker_path, "PolishProcessor")

        # Initialize SplitManager for dynamic splitting
        splitting_config = config.get('splitting', {})
        self.split_manager = SplitManager(
            tracker=self.processing_tracker,
            output_dir=self.output_dir,
            default_max_tokens=self.get_max_tokens_per_part(),
            max_resplits=splitting_config.get('max_resplits', 3),
            consecutive_failures_threshold=splitting_config.get('consecutive_failures_threshold', 2)
        )

        # Log global footnote detection
        if self.use_global_footnotes:
            logger.info("Detected Notes chapter - using global footnote mode for academic content")

    @property
    def agent_verifier(self) -> PolishVerificationAgent:
        """Lazily initialize agent verifier when needed."""
        if self._agent_verifier is None:
            # Create empty tools instance - will be populated per batch
            tools = VerificationTools({})
            self._agent_verifier = PolishVerificationAgent(tools)
        return self._agent_verifier

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

    def build_prompt(self, content: str, unit_key: str, **context) -> List[Dict]:
        """
        Build the polishing prompt with optional conversation history.

        Args:
            content: Content to polish
            unit_key: Unit identifier for tracking
            **context: Context including file_name, part_idx, total_parts, previous_context

        Returns:
            Multi-part content for LLM (may include conversation history)
        """
        file_name = context.get('file_name', unit_key)
        part_idx = context.get('part_idx', 1)
        total_parts = context.get('total_parts', 1)
        previous_context = context.get('previous_context')

        # Preprocess: convert HTML img tags to markdown to avoid LLM corruption
        content = convert_html_images_to_markdown(content)

        # Get chapter info for title and type
        chapter_info = self._get_chapter_info(file_name)
        is_notes_chapter = chapter_info.get('type') == 'notes'

        # Use actual chapter title if available, otherwise use operation name for logging
        chapter_name = chapter_info.get('title') or self.get_operation_name(file_name)

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
            use_global_footnotes=self.use_global_footnotes,
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

    def post_process(self, result: str, **context) -> str:
        """
        Post-process the polished result.

        Applies markdown cleanup and restores lost images for single-part files.

        Args:
            result: Cleaned LLM response
            **context: Context including original_content, total_parts

        Returns:
            Post-processed result
        """
        # Apply markdown post-processing
        result = self._post_process_markdown(result)

        # Restore lost images if single part
        total_parts = context.get('total_parts', 1)
        if total_parts == 1:
            original_content = context.get('original_content', '')
            result = restore_lost_images(original_content, result)

        return result

    def get_context_for_next_part(self, content: str, result: str, **context) -> Optional[Dict]:
        """
        Get context to inject into the next part's build_prompt.

        Only provides context in sequential processing mode.

        Args:
            content: Original content of this part
            result: Processed result of this part
            **context: Processing context

        Returns:
            Context dict for next part, or None if parallel mode
        """
        if self.processing_mode == "sequential":
            return {"original": content, "processed": result}
        return None

    def on_validation_failure(self, file_name: str, reason: str, response: str) -> None:
        """
        Save truncated output for debugging.

        Args:
            file_name: Name of the file being processed
            reason: Validation failure reason
            response: The failed response
        """
        if "truncat" in reason.lower():
            # Extract part info from file_name if present
            # file_name format: "chapter_5.md part 2/3" or "chapter_5.md"
            if " part " in file_name:
                base_file, part_info = file_name.rsplit(" part ", 1)
                part_idx, total_parts = map(int, part_info.split("/"))
            else:
                base_file = file_name
                part_idx, total_parts = 1, 1
            self._save_truncated_output(base_file, part_idx, total_parts, response, reason)

    def _cleanup_part_files(self, file_name: str) -> None:
        """Clean up any existing part files and progress entries for this file."""
        output_dir = Path(self.output_dir)
        base_name = Path(file_name).stem
        
        # Find and delete all part files for this base name
        part_files = list(output_dir.glob(f"{base_name}.part*.md"))
        if part_files:
            logger.debug(f"Cleaning up {len(part_files)} existing part files for {base_name}")
            for part_file in part_files:
                try:
                    part_file.unlink()
                    logger.debug(f"Deleted old part file: {part_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete part file {part_file.name}: {e}")

    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate the polished output with basic sanity checks.

        Full validation (N-gram + Agent) is deferred to batch phase after all files processed.
        This method performs basic checks to catch obvious failures early.

        Args:
            original: Original content
            processed: Polished content
            file_name: Name of the file

        Returns:
            Tuple of (is_valid, reason)
        """
        # Skip truncation check for front/back matter files
        if "front_matter" in file_name.lower() or "back_matter" in file_name.lower():
            return True, "Truncation check skipped for front/back matter"

        # Skip if flag is set
        if self.skip_truncation_check:
            return True, "Truncation check skipped by flag"

        # Basic sanity check: processed should not be empty
        if not processed.strip():
            return False, "Empty output"

        # Basic length check: processed should be at least 1% of original
        # (Too aggressive check causes false positives; batch validation will do thorough check)
        if len(processed) < len(original) * 0.01:
            return False, f"Output suspiciously short: {len(processed)}/{len(original)} chars (< 1%)"

        # Pass inline check - full validation in batch phase
        return True, "Inline validation passed - full check deferred to batch"
    
    def get_split_strategy(self) -> str:
        """
        Get splitting strategy based on content type.

        Returns:
            Strategy string: "auto", "general", "academic", or "japanese"
        """
        return self.content_type

    def _save_truncated_output(self, file_name: str, part_idx: int, total_parts: int, content: str, reason: str) -> None:
        """Save truncated output for debugging purposes."""
        output_dir = Path(self.output_dir)
        truncated_dir = output_dir / "truncated_outputs"
        truncated_dir.mkdir(exist_ok=True)

        base_name = Path(file_name).stem
        if total_parts > 1:
            truncated_file = truncated_dir / f"{base_name}.part{part_idx}_TRUNCATED.md"
        else:
            truncated_file = truncated_dir / f"{base_name}_TRUNCATED.md"

        # Save the truncated content with metadata
        with open(truncated_file, 'w', encoding='utf-8') as f:
            f.write(f"<!-- TRUNCATION DETECTED: {reason} -->\n\n")
            f.write(content)

        logger.warning(f"Saved truncated output for debugging: {truncated_file.name}")

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
        markdown = re.sub(r'\n{4,}', '\n\n', markdown)  # Changed from 3+ to 4+ to handle image removal

        # 4. Ensure valid images have blank lines around them
        markdown = re.sub(r'([^\n])\n(!\[)', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(!\[[^\]]*\]\([^\)]*\))\n([^\n])', r'\1\n\n\2', markdown)

        return markdown

    def get_inject_context(self) -> bool:
        """
        Get whether to inject context between parts.

        For polisher, this depends on processing_mode:
        - "sequential": inject context for consistency
        - "parallel": no context injection for speed

        Returns:
            True if context should be injected between parts
        """
        return self.processing_mode == "sequential"

    def _get_original_content(self, file_key: str) -> str:
        """
        Get original content for a file key.

        Args:
            file_key: File key (e.g., "chapter_1" or "chapter_1.part1")

        Returns:
            Original content from input directory
        """
        # Handle both "chapter_1" and "chapter_1.part1" format
        if ".part" in file_key:
            base_key = file_key.split(".part")[0]
        else:
            base_key = file_key

        input_file = self.input_dir / f"{base_key}.md"
        if not input_file.exists():
            logger.warning(f"Original file not found: {input_file}")
            return ""

        with open(input_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _save_result(self, file_key: str, content: str) -> None:
        """
        Save processed result to output directory.

        Args:
            file_key: File key
            content: Processed content
        """
        output_file = self.output_dir / f"{file_key}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.debug(f"Saved result: {output_file.name}")

    def _batch_validate_and_save(
        self,
        all_units: List,  # List[WorkUnit]
        completed_results: Dict[str, str],
        failed_ids: set
    ) -> tuple:
        """
        Override base class method to implement two-phase validation.

        Phase 1: N-gram screening (fast)
        Phase 2: Agent verification (accurate)

        Args:
            all_units: All work units
            completed_results: Dict of unit_id -> processed content
            failed_ids: Set of failed unit IDs

        Returns:
            Tuple of (validated_results, updated_failed_ids)
        """
        from .utils.verification_tools import VerificationFile

        if not completed_results:
            return completed_results, failed_ids

        # Separate newly processed units from already-saved units
        # Only validate newly processed units to avoid re-validating stable results
        newly_processed = {}
        already_saved = {}

        for unit in all_units:
            unit_id = unit.id
            if unit_id in completed_results:
                # Check if this unit was just processed (output file doesn't exist or is older)
                if not unit.output_path.exists():
                    # Newly processed in this round
                    newly_processed[unit_id] = completed_results[unit_id]
                else:
                    # Already saved from previous round - skip validation
                    already_saved[unit_id] = completed_results[unit_id]

        logger.info(f"Validating {len(newly_processed)} newly processed units, skipping {len(already_saved)} already saved")

        # CRITICAL: Save all attempts for longest fallback strategy
        # Only track newly processed units for attempt history
        self._last_batch_attempts = newly_processed.copy()

        if not newly_processed:
            # No new units to validate, all are from previous rounds
            return completed_results, failed_ids

        # Check if truncation check should be skipped globally
        if self.skip_truncation_check:
            logger.info("Truncation check disabled by skip_truncation_check flag")
            passed = list(newly_processed.keys())
            # Save all and return
            logger.info(f"Saving {len(passed)} units (validation skipped)")
            for unit in all_units:
                if unit.id in passed and unit.id in completed_results:
                    with open(unit.output_path, 'w', encoding='utf-8') as f:
                        f.write(completed_results[unit.id])
            all_passed = passed + list(already_saved.keys())
            self._aggregate_validated_files(all_units, all_passed, completed_results)
            return completed_results, failed_ids

        passed = []
        suspicious = {}

        # Phase 1: N-gram screening (only for newly processed units)
        logger.info(f"Phase 1: N-gram screening for {len(newly_processed)} units")
        for unit_id, processed in newly_processed.items():
            # Skip validation for front/back matter
            if "front_matter" in unit_id.lower() or "back_matter" in unit_id.lower():
                passed.append(unit_id)
                logger.debug(f"{unit_id}: Skipped (front/back matter)")
                continue

            # Get file_key from unit_id (e.g., "chapter_1.part1" -> "chapter_1")
            file_key = unit_id.split('.part')[0] if '.part' in unit_id else unit_id
            original = self._get_original_content(file_key)

            is_truncated, reason, details = self.ngram_detector.detect(
                original=original,
                processed=processed
            )

            if not is_truncated:
                # N-gram passed, will save later
                passed.append(unit_id)
                logger.debug(f"{unit_id}: N-gram passed")
            else:
                # Suspicious, queue for agent verification
                suspicious[unit_id] = VerificationFile(
                    key=unit_id,
                    original=original,
                    processed=processed
                )
                logger.debug(f"{unit_id}: N-gram flagged - {reason}")

        logger.info(f"{len(passed)} units passed n-gram, {len(suspicious)} suspicious")

        # Phase 2: Agent batch verification (if any suspicious)
        if suspicious:
            logger.info(f"Phase 2: Agent verification for {len(suspicious)} suspicious units")

            try:
                from .utils.agent_verifier import verify_batch

                verification_results = verify_batch(
                    files=suspicious,
                    task_type="polish"
                )

                for result in verification_results:
                    if result.status == "complete":
                        # Agent confirmed complete
                        passed.append(result.file_key)
                        logger.info(f"{result.file_key}: Agent verified complete - {result.reason}")
                    else:
                        # Agent confirmed truncation - mark as failed
                        failed_ids.add(result.file_key)
                        # Remove from completed_results so it won't be saved
                        if result.file_key in completed_results:
                            del completed_results[result.file_key]
                        logger.warning(f"{result.file_key}: Agent confirmed truncation - {result.reason}")

            except Exception as e:
                logger.error(f"Agent verification failed: {e}")
                logger.warning("Marking all suspicious units as failed")
                # Fallback: mark all suspicious as failed
                for unit_id in suspicious.keys():
                    failed_ids.add(unit_id)
                    if unit_id in completed_results:
                        del completed_results[unit_id]

        # Save all newly passed units
        logger.info(f"Saving {len(passed)} validated units")
        for unit in all_units:
            if unit.id in passed and unit.id in completed_results:
                with open(unit.output_path, 'w', encoding='utf-8') as f:
                    f.write(completed_results[unit.id])
                logger.debug(f"Saved: {unit.output_path.name}")

        # Include already_saved units as implicitly passed (they were validated in previous rounds)
        all_passed = passed + list(already_saved.keys())

        # Aggregate multi-part files
        self._aggregate_validated_files(all_units, all_passed, completed_results)

        return completed_results, failed_ids

    def _aggregate_validated_files(
        self,
        all_units: List,
        passed_unit_ids: List[str],
        completed_results: Dict[str, str]
    ):
        """
        Aggregate multi-part files after validation.

        Similar to base class _aggregate_file_results but only for validated units.
        """
        from pdf2epub.processors.utils.work_unit import WorkUnitDiscovery

        discovery = WorkUnitDiscovery(
            self.input_dir,
            self.output_dir,
            splits_dir=self.splits_dir
        )
        file_groups = discovery.group_units_by_file(all_units)

        for file_key, units in file_groups.items():
            if len(units) <= 1:
                continue  # Single file, no aggregation needed

            # Check if all parts passed validation
            if not all(u.id in passed_unit_ids for u in units):
                logger.warning(f"Not all parts validated for {file_key}, skipping aggregation")
                continue

            # Aggregate in order
            parts = [completed_results[u.id] for u in units if u.id in completed_results]
            if not parts:
                continue

            aggregated = "\n\n".join(p for p in parts if p)

            # Save combined file
            combined_path = self.output_dir / f"{file_key}.md"
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write(aggregated)

            logger.info(f"Aggregated {len(units)} parts into {combined_path.name}")

    def _save_longest_attempts(
        self,
        failed_keys: List[str],
        attempt_history: Dict[str, List[Dict]]
    ) -> int:
        """
        Save the longest attempt for files that failed after max retries.

        This is a fallback strategy: when a file fails validation after all retries,
        we save the longest version rather than losing all content. The file is marked
        with a warning note generated by the agent explaining the issue.

        Args:
            failed_keys: List of keys that failed all retries
            attempt_history: Dict mapping key to list of attempts
                            Each attempt: {'text': str, 'length': int, 'retry_count': int}

        Returns:
            Number of files saved
        """
        saved_count = 0

        for key in failed_keys:
            attempts = attempt_history.get(key, [])
            if not attempts:
                logger.error(f"{key}: No attempts recorded")
                continue

            # Find longest attempt
            longest = max(attempts, key=lambda x: x['length'])

            # Get original content for diagnostic
            original = self._get_original_content(key)

            # Generate diagnostic note
            diagnostic_note = self._generate_diagnostic_note(
                key=key,
                original=original,
                processed=longest['text'],
                attempts_count=len(attempts)
            )

            # Prepend diagnostic note
            content_with_note = diagnostic_note + "\n\n---\n\n" + longest['text']

            # Save with note
            self._save_result(key, content_with_note)

            logger.warning(
                f"{key}: Saved longest attempt ({longest['length']} chars) "
                f"from {len(attempts)} attempts (retry {longest['retry_count']}) "
                f"with diagnostic note - content may be incomplete"
            )
            saved_count += 1

        return saved_count

    def _generate_diagnostic_note(
        self,
        key: str,
        original: str,
        processed: str,
        attempts_count: int
    ) -> str:
        """
        Use agent to generate a diagnostic note explaining why the file failed.

        The agent analyzes the failure and provides information about:
        - What type of problem was detected
        - Where in the content the problem likely occurs
        - Recommendations for the user

        Args:
            key: File key
            original: Original content
            processed: Processed content (longest attempt)
            attempts_count: Number of attempts made

        Returns:
            Diagnostic note in markdown format
        """
        try:
            from pdf2epub.processors.utils.agent_verifier import get_verification_model
            from pydantic_ai import Agent

            model = get_verification_model()

            # Create a simple agent for diagnostic
            diagnostic_prompt = f"""You are analyzing a file that failed polish validation after {attempts_count} attempts.

Your task: Write a brief diagnostic note (2-3 sentences) explaining:
1. What type of problem was detected (truncation, corruption, etc.)
2. Where in the content the problem likely occurs
3. Brief suggestion for the user

Keep it concise and actionable. Use markdown format.

File: {key}
Original length: {len(original)} chars
Processed length: {len(processed)} chars
Ratio: {len(processed)/len(original)*100:.1f}%

Original ending (last 200 chars):
{original[-200:]}

Processed ending (last 200 chars):
{processed[-200:]}
"""

            agent = Agent(model, output_type=str)
            result = agent.run_sync(diagnostic_prompt)

            # Format the diagnostic note
            note = f"""<!-- DIAGNOSTIC NOTE: Auto-generated by agent verification -->
> ⚠️ **Content may be incomplete** - This file failed validation after {attempts_count} attempts.
> The longest version ({len(processed)} chars) has been saved.

{result.output}

---
"""
            return note

        except Exception as e:
            logger.warning(f"Failed to generate diagnostic note for {key}: {e}")
            # Fallback to simple note
            return f"""<!-- DIAGNOSTIC NOTE -->
> ⚠️ **Content may be incomplete** - This file failed validation after {attempts_count} attempts.
> Original: {len(original)} chars → Processed: {len(processed)} chars ({len(processed)/len(original)*100:.1f}%)
> Please manually review this file for completeness.

---
"""

    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all files with retry and longest fallback.

        This method wraps the base class processing with:
        - Retry loop for failed files
        - Attempt history tracking
        - Longest fallback for persistent failures

        Returns:
            Summary statistics
        """
        # Configuration
        max_retries = self.config.get('validation_strategy', {}).get('max_retries', 3)

        # Track all attempts for longest fallback
        # Maps unit_id -> list of {'text': str, 'length': int, 'retry_count': int}
        attempt_history: Dict[str, List[Dict]] = {}

        # Initialize tracking
        retry_count = 0
        failed_unit_ids = set()

        while retry_count <= max_retries:
            if retry_count == 0:
                logger.info(f"Processing all files (max_retries: {max_retries})")
            else:
                logger.info(f"Retry attempt {retry_count}/{max_retries} for {len(failed_unit_ids)} failed units")

            # Call base class processing
            # This will call process_all_units() which calls _batch_validate_and_save()
            summary = super().process_all_files()

            # Extract failures from summary
            new_failed_ids = summary.get('failed_ids', set())

            # Record attempts from this round
            # _batch_validate_and_save() stores attempts in self._last_batch_attempts
            if hasattr(self, '_last_batch_attempts'):
                for unit_id, result_text in self._last_batch_attempts.items():
                    if unit_id not in attempt_history:
                        attempt_history[unit_id] = []
                    attempt_history[unit_id].append({
                        'text': result_text,
                        'length': len(result_text),
                        'retry_count': retry_count
                    })

            if not new_failed_ids:
                # Success! All files processed
                logger.success(f"All files completed successfully on attempt {retry_count + 1}")
                return summary

            # Check if we should retry
            if retry_count < max_retries:
                logger.warning(
                    f"{len(new_failed_ids)} units failed validation, "
                    f"will retry (attempt {retry_count + 2}/{max_retries + 1})"
                )
                failed_unit_ids = new_failed_ids
                retry_count += 1

                # Mark failed units as pending in tracker so they'll be re-processed
                for unit_id in failed_unit_ids:
                    self.processing_tracker.reset_unit_status(unit_id)

                continue
            else:
                # Max retries exhausted
                logger.warning(
                    f"{len(new_failed_ids)} units still failing after {max_retries + 1} attempts, "
                    f"using longest fallback strategy"
                )
                failed_unit_ids = new_failed_ids
                break

        # Longest fallback for persistent failures
        if failed_unit_ids:
            failed_list = list(failed_unit_ids)
            logger.info(f"Applying longest fallback to {len(failed_list)} failed units")
            saved_count = self._save_longest_attempts(failed_list, attempt_history)
            logger.info(f"Saved {saved_count} files with longest attempt (may be incomplete)")

            # Update summary to reflect saved files
            summary['succeeded'] = summary.get('succeeded', 0) + saved_count
            summary['failed_with_fallback'] = saved_count

        return summary

