"""
Polish processor for OCR-extracted markdown content.

This processor cleans up OCR-extracted markdown, removing artifacts,
fixing formatting, and organizing content structure.
"""

import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import tiktoken

from .base import BaseMarkdownProcessor
from .utils.truncation import NGramTruncationDetector
from .utils.content_splitter import split_content
from .utils.image_restore import restore_lost_images

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


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
        use_longest_on_failure: bool = False
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
        self.polish_models = polish_models or config.get("polish_models")
        self.content_type = content_type
        
        # Initialize truncation detector
        self.truncation_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=0.60,
            allow_deduplication=True
        )
        
        # Thread lock for progress updates
        self.progress_lock = threading.Lock()
    
    def get_progress_filename(self) -> str:
        """Get the name for the progress file."""
        return "polish_progress"
    
    def get_progress_key(self) -> str:
        """Get the key used in progress tracking."""
        return "parts_info"
    
    def get_operation_name(self, file_name: str) -> str:
        """Get the operation name for logging."""
        # Extract chapter info from filename
        if file_name == "front_matter.md":
            return "Front Matter"
        elif file_name == "back_matter.md":
            return "Back Matter"
        else:
            match = re.search(r'chapter_(\d+)', file_name)
            if match:
                return f"Chapter {match.group(1)}"
            return file_name
    
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
        
        # Also clean up progress entries for this file
        progress_key = self.get_progress_key()
        if base_name in self.progress.get(progress_key, {}):
            logger.debug(f"Cleaning up progress entries for {base_name}")
            del self.progress[progress_key][base_name]
            self.save_progress()
    
    def process_content(
        self,
        content: str,
        file_name: str,
        **kwargs
    ) -> str:
        """
        Process markdown content by polishing it.
        
        Args:
            content: The markdown content to polish
            file_name: Name of the file being processed
            **kwargs: Additional arguments
        
        Returns:
            Polished markdown content
        """
        # Check if we should clean up part files
        # Only clean up if we're not resuming, or if the file isn't marked as completed
        file_key = Path(file_name).stem
        progress_key = self.get_progress_key()
        should_cleanup = True
        
        if self.resume and file_key in self.progress.get(progress_key, {}):
            file_progress = self.progress[progress_key][file_key]
            if file_progress.get("completed", False):
                # File is already completed, don't clean up part files
                should_cleanup = False
                logger.debug(f"Preserving existing part files for completed file: {file_name}")
        
        if should_cleanup:
            # Clean up any existing part files from previous failed attempts
            self._cleanup_part_files(file_name)
        
        # Determine if we need to split the content
        max_tokens_per_part = self._determine_max_tokens()
        # Use proper token counting with tiktoken
        actual_tokens = len(tokenizer.encode(content))
        
        if actual_tokens > max_tokens_per_part:
            logger.info(f"Content has {actual_tokens:,} tokens (max: {max_tokens_per_part:,}), splitting into parts")
            parts = self._split_content(content, max_tokens_per_part)
            
            # Store split information in progress
            with self.progress_lock:
                if file_key not in self.progress.get(progress_key, {}):
                    self.progress[progress_key][file_key] = {}
                
                # Calculate and store split points (character positions where each part ends)
                split_points = []
                current_pos = 0
                for part in parts[:-1]:  # Don't need the end position of the last part
                    current_pos += len(part)
                    split_points.append(current_pos)
                
                self.progress[progress_key][file_key]["split_points"] = split_points
                self.progress[progress_key][file_key]["total_tokens"] = actual_tokens
                self.save_progress()
        else:
            parts = [content]
        
        logger.info(f"Processing {len(parts)} part(s) for {file_name}")
        
        # Process parts in parallel if multiple
        chapter_name = self.get_operation_name(file_name)
        file_key = Path(file_name).stem
        progress_key = self.get_progress_key()
        
        if len(parts) > 1:
            # Process multiple parts in parallel
            polished_parts = [None] * len(parts)  # Pre-allocate list to maintain order
            all_parts_valid = True
            
            # Determine max workers for parts (use half of configured workers or at least 2)
            part_workers = max(2, self.max_workers // 2)
            
            with ThreadPoolExecutor(max_workers=part_workers) as executor:
                futures = {}
                
                for part_idx, part_content in enumerate(parts, 1):
                    # Check if this part was already processed (for resume)
                    if self.resume and file_key in self.progress[progress_key]:
                        part_info = self.progress[progress_key][file_key].get("parts", {}).get(str(part_idx), {})
                        if part_info.get("completed", False):
                            # Load the already processed part
                            part_file = self.output_dir / f"{file_key}.part{part_idx}.md"
                            if part_file.exists():
                                with open(part_file, 'r', encoding='utf-8') as f:
                                    polished_parts[part_idx - 1] = f.read()
                                logger.info(f"Skipping {file_name} part {part_idx}/{len(parts)} (already processed)")
                                continue
                    
                    # Submit part for processing
                    future = executor.submit(
                        self._process_and_validate_part,
                        part_content=part_content,
                        chapter_name=chapter_name,
                        part_idx=part_idx,
                        total_parts=len(parts),
                        original_content=content,
                        file_name=file_name
                    )
                    futures[future] = part_idx
                
                # Process completed futures
                for future in as_completed(futures):
                    part_idx = futures[future]
                    try:
                        polished_part, is_valid = future.result()
                        polished_parts[part_idx - 1] = polished_part
                        
                        if not is_valid:
                            all_parts_valid = False
                        
                        # Save part file
                        self._save_part_file(file_name, part_idx, polished_part)
                        # Update progress for this part with original content tokens
                        original_part = parts[part_idx - 1]
                        self._update_part_progress(file_name, part_idx, len(parts), success=is_valid, part_content=original_part)
                        
                    except Exception as e:
                        logger.error(f"Failed to process part {part_idx}/{len(parts)}: {e}")
                        all_parts_valid = False
                        polished_parts[part_idx - 1] = ""  # Empty string for failed parts
            
            # Check if all parts were processed
            if not all_parts_valid and not self.skip_truncation_check:
                logger.warning(f"Some parts of {file_name} failed validation")
        else:
            # Single part - process normally
            all_parts_valid = True
            polished_part, is_valid = self._process_and_validate_part(
                part_content=parts[0],
                chapter_name=chapter_name,
                part_idx=1,
                total_parts=1,
                original_content=content,
                file_name=file_name
            )
            polished_parts = [polished_part]
            all_parts_valid = is_valid
        
        # If not all parts are valid, we might want to handle this differently
        if not all_parts_valid and not self.skip_truncation_check:
            logger.warning(f"Some parts of {file_name} failed validation")
        
        # Combine parts if multiple
        if len(parts) > 1:
            # Filter out None or empty parts
            valid_parts = [p for p in polished_parts if p]
            if not valid_parts:
                raise ValueError("All parts failed to process")
            combined = "\n\n".join(valid_parts)
            # Restore any lost images
            combined = restore_lost_images(content, combined)
            return combined
        else:
            if not polished_parts or not polished_parts[0]:
                raise ValueError("Failed to process content")
            return polished_parts[0]
    
    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate the polished output using truncation detection.
        
        Args:
            original: Original content
            processed: Polished content
            file_name: Name of the file
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if self.skip_truncation_check:
            return True, "Truncation check skipped"
        
        is_truncated, reason, details = self.truncation_detector.detect(
            original=original,
            processed=processed
        )
        
        # Log the summary
        summary = self.truncation_detector.get_summary(is_truncated, reason, details)
        if is_truncated:
            logger.warning(f"{file_name} truncation analysis:\n{summary}")
        else:
            logger.info(f"{file_name} validation passed:\n{summary}")
        
        return not is_truncated, reason
    
    def _determine_max_tokens(self) -> int:
        """Determine maximum tokens per part based on model configuration."""
        max_tokens_per_part = 10000  # Conservative default
        
        if self.polish_models:
            # Check if any model has limited context
            limited_model_patterns = ["flash", "haiku", "-mini"]
            
            has_limited_model = any(
                any(pattern in model_config.get("model", "").lower() 
                    for pattern in limited_model_patterns)
                for model_config in self.polish_models
            )

            is_deepseek = any(
                "seek" in model_config.get("model", "").lower()
                for model_config in self.polish_models
            )

            if is_deepseek:
                max_tokens_per_part = 15000
                logger.info("Using max_tokens_per_part=15000 (Deepseek detected)")
            elif not has_limited_model:
                max_tokens_per_part = 20000
                logger.info("Using max_tokens_per_part=20000 (no limited-context models detected)")
            else:
                logger.info("Using max_tokens_per_part=10000 (limited-context model detected)")
        
        return max_tokens_per_part
    
    def _split_content(self, content: str, max_tokens: int) -> List[str]:
        """Split content into manageable parts."""
        return split_content(
            content,
            max_tokens,
            self.llm_client,
            self.polish_models,
            self.content_type,
        )
    
    def _process_and_validate_part(
        self,
        part_content: str,
        chapter_name: str,
        part_idx: int,
        total_parts: int,
        original_content: str,
        file_name: str
    ) -> Tuple[str, bool]:
        """Process and validate a single part with retry logic."""
        # Get max retries from model config
        max_retries = 3
        if self.polish_models:
            max_retries = self.polish_models[0].get('max_retries', 3)
        
        # last_error = None
        best_attempt = None
        best_attempt_valid = False
        
        for attempt in range(max_retries):
            try:
                # Polish the part
                polished_part = self._polish_part(
                    part_content=part_content,
                    chapter_name=chapter_name,
                    part_idx=part_idx,
                    total_parts=total_parts,
                    original_content=original_content,
                    file_name=file_name
                )
                
                # Validate this part
                is_valid = True
                if not self.skip_truncation_check:
                    is_valid, reason = self.validate_output(
                        original=part_content,
                        processed=polished_part,
                        file_name=f"{file_name} part {part_idx}/{total_parts}" if total_parts > 1 else file_name
                    )
                    
                    if not is_valid:
                        # last_error = reason
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Part {part_idx}/{total_parts} validation failed (attempt {attempt + 1}/{max_retries}): {reason}"
                            )
                            logger.info(f"Retrying part {part_idx}/{total_parts}...")
                            # Store best attempt so far
                            if best_attempt is None or len(polished_part) > len(best_attempt):
                                best_attempt = polished_part
                            continue
                        else:
                            logger.error(
                                f"Part {part_idx}/{total_parts} validation failed after {max_retries} attempts: {reason}"
                            )
                            # Use best attempt if we have one
                            if best_attempt and self.use_longest_on_failure:
                                logger.warning(f"Using longest response for part {part_idx}/{total_parts}")
                                return best_attempt, False
                            return polished_part, False
                
                # Success!
                return polished_part, True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Error processing part {part_idx}/{total_parts} (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    continue
                else:
                    logger.error(
                        f"Failed to process part {part_idx}/{total_parts} after {max_retries} attempts: {e}"
                    )
                    if best_attempt:
                        return best_attempt, False
                    raise
        
        # Should not reach here, but just in case
        if best_attempt:
            return best_attempt, best_attempt_valid
        raise ValueError(f"Failed to process part {part_idx} after all attempts")
    
    def _save_part_file(self, file_name: str, part_idx: int, content: str) -> None:
        """Save a part file separately."""
        output_dir = Path(self.output_dir)
        base_name = Path(file_name).stem
        part_file = output_dir / f"{base_name}.part{part_idx}.md"
        
        with open(part_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.debug(f"Saved part file: {part_file.name}")
    
    def _update_part_progress(self, file_name: str, part_idx: int, total_parts: int, success: bool, part_content: str = None) -> None:
        """Update progress for a specific part."""
        with self.progress_lock:
            file_key = Path(file_name).stem
            progress_key = self.get_progress_key()
            
            # Ensure progress_key exists
            if progress_key not in self.progress:
                self.progress[progress_key] = {}
            
            # Initialize file progress if not exists
            if file_key not in self.progress[progress_key]:
                self.progress[progress_key][file_key] = {
                    "total_parts": total_parts,
                    "parts": {}
                }
            
            # Ensure 'parts' key exists
            if "parts" not in self.progress[progress_key][file_key]:
                self.progress[progress_key][file_key]["parts"] = {}
            
            # Update part status with token count
            part_data = {
                "completed": success,
                "timestamp": time.time()
            }
            
            # Add token count if content is provided
            if part_content:
                part_data["tokens"] = len(tokenizer.encode(part_content))
            
            self.progress[progress_key][file_key]["parts"][str(part_idx)] = part_data
            
            # Check if all parts are complete
            all_complete = all(
                self.progress[progress_key][file_key]["parts"].get(str(i), {}).get("completed", False)
                for i in range(1, total_parts + 1)
            )
            
            self.progress[progress_key][file_key]["completed"] = all_complete
            
            # Save progress
            self.save_progress()
    
    def _polish_part(
        self,
        part_content: str,
        chapter_name: str,
        part_idx: int,
        total_parts: int,
        original_content: str,
        file_name: str
    ) -> str:
        """Polish a single part of content."""
        # Create the polish prompt with content for auto-detection
        prompt = self._create_polish_prompt(
            chapter_name, part_idx, total_parts, part_content
        )
        
        # Create multi-part content for the LLM
        multi_part_content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": part_content}
        ]
        
        # Generate polished content
        operation_name = (
            f"{chapter_name} part {part_idx}/{total_parts}"
            if total_parts > 1
            else chapter_name
        )
        
        polished_content = self.llm_client.generate(
            prompt=multi_part_content,
            model_configs=self.polish_models,
            operation_name=operation_name
        )
        
        # Clean and post-process
        polished_content = self.clean_markdown_response(polished_content)
        polished_content = self._post_process_markdown(polished_content)
        
        # Restore lost images if single part
        if total_parts == 1:
            polished_content = restore_lost_images(original_content, polished_content)
        
        return polished_content
    
    def _detect_content_type(self, content: str) -> str:
        """Auto-detect content type based on content characteristics."""
        # Check for Japanese characters
        japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', content)
        if len(japanese_chars) > len(content) * 0.1:  # More than 10% Japanese characters
            logger.info("Auto-detected content type: Japanese")
            return "japanese"
        
        # Check for academic indicators
        footnote_indicators = [
            r'\[\^\d+\]',  # Markdown footnotes [^1]
            r'\$\^\{?\d+\}?\$',  # LaTeX style superscripts
            r'\[\d+\]',  # Bracketed references
            r'References\s*\n',  # References section
            r'Bibliography\s*\n',  # Bibliography section
            r'\\cite\{',  # LaTeX citations
        ]
        
        academic_score = sum(
            1 for pattern in footnote_indicators 
            if re.search(pattern, content[:5000])  # Check first 5000 chars
        )
        
        if academic_score >= 2:
            logger.info("Auto-detected content type: Academic")
            return "academic"
        
        logger.info("Auto-detected content type: General")
        return "general"
    
    def _create_polish_prompt(
        self,
        chapter_name: str,
        part_idx: int,
        total_parts: int,
        part_content: str = None
    ) -> str:
        """Create the prompt for polishing content based on content type."""
        # Determine content type
        content_type = self.content_type
        if content_type == "auto" and part_content:
            content_type = self._detect_content_type(part_content)
        elif content_type == "auto":
            content_type = "general"
        
        # Route to appropriate prompt creator
        if content_type == "academic":
            return self._create_academic_polish_prompt(chapter_name, part_idx, total_parts)
        elif content_type == "japanese":
            return self._create_japanese_polish_prompt(chapter_name, part_idx, total_parts)
        else:
            return self._create_general_polish_prompt(chapter_name, part_idx, total_parts)
    
    def _create_academic_polish_prompt(
        self,
        chapter_name: str,
        part_idx: int,
        total_parts: int
    ) -> str:
        """Create prompt specifically for academic content with references."""
        book_info = f' from the book titled "{self.book_title}"' if self.book_title else ""
        
        prompt = f"""You are an expert academic document editor specializing in scholarly texts. Polish this OCR-extracted academic content from "{chapter_name}"{book_info}.

Your tasks for ACADEMIC content:

1. **Remove page artifacts**:
   - Delete page numbers, headers, and footers
   - Remove horizontal separators between pages
   - Join sentences broken by page boundaries

2. **Preserve academic structure**:
   - Main chapter title: # (H1)
   - Sections: ## (H2), subsections: ### (H3)
   - Keep abstract, introduction, conclusion sections intact
   - Preserve figure/table captions and numbering

3. **Handle citations and footnotes CAREFULLY**:
   - INLINE CITATIONS: Convert citation markers ($^1$, ¹, {{ }}^{{1}}, etc.) to [^1] format
   - IMPORTANT: Text immediately after a citation marker is NOT the footnote definition
   - Example of WRONG interpretation:
     * Input: "This is discussed by Smith$^5$ The next sentence continues..."
     * WRONG: "This is discussed by Smith[^5]" then "[^5]: The next sentence continues..."
     * RIGHT: "This is discussed by Smith[^5] The next sentence continues..."
   - FOOTNOTE DEFINITIONS: Only create [^1]: format for ACTUAL footnotes found at:
     * Bottom of pages (separated from main text)
     * End of chapters in dedicated Notes/References sections
     * Clearly marked footnote sections
   - DO NOT convert regular paragraph text after citations into footnote definitions
   - Preserve exact footnote numbering from the source
   - Never invent or add missing footnotes

4. **Preserve academic elements**:
   - Keep equations, formulas, and mathematical notation
   - Preserve code blocks and technical examples
   - Maintain definition lists and theorems
   - Keep cross-references ("see Section 2.3")

5. **Organize bibliography**:
   - Move all footnotes to "### Notes" section if they exist (use ### to keep as subsection, not chapter)
   - Organize references under "### References" if present
   - Format citations consistently:
     * Books: Author(s). (Year). *Title*. Publisher.
     * Articles: Author(s). (Year). "Title." *Journal*, Volume(Issue), pages.
   - Final structure: Main Content → ### Notes → ### References

6. **Quality checks**:
   - Ensure all citations have corresponding footnotes
   - Verify footnote numbering is sequential
   - Check that academic terminology is preserved"""
        
        # Add context for multi-part chapters
        if total_parts > 1:
            prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""
            
            if part_idx > 1:
                prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2).
Convert any # (H1) headings to ## (H2)."""
        
        prompt += """

IMPORTANT: Return ONLY the polished markdown. Do not add explanations.
Preserve all tables, figures, and images unless duplicated.

Polish the following academic content:"""
        
        return prompt
    
    def _create_japanese_polish_prompt(
        self,
        chapter_name: str,
        part_idx: int,
        total_parts: int
    ) -> str:
        """Create prompt specifically for Japanese content with furigana."""
        book_info = f' from "{self.book_title}"' if self.book_title else ""
        
        prompt = f"""You are an expert editor specializing in Japanese literature and light novels. Polish this OCR-extracted Japanese content from "{chapter_name}"{book_info}.

Your tasks for JAPANESE content:

1. **Remove page artifacts**:
   - Delete page numbers and headers/footers
   - Remove separators (---) between pages and join sentences
   - Join continuous sentences broken by OCR
   - Handle vertical text OCR artifacts

2. **Preserve Japanese text features**:
   - KEEP all furigana/ruby text: 一人(ひとり), 今更(いまさら), 幼馴染(おさななじみ)
   - DO NOT add new furigana not in the original
   - DO NOT change () of furigana to （）
   - DO NOT remove furigana from the original text
   - DO NOT remove ルビ芸 like 妄想 in「何もないからこ(妄)ういう(想)話に逃(に)げてんじゃん!」

3. **Images and illustrations**:
   - PRESERVE ALL IMAGE LINKS EXACTLY AS THEY ARE
   - Keep markdown image syntax: ![Image](../images/filename.png)
   - DO NOT replace image links with [illustration] or any other placeholder
   - DO NOT modify image paths or filenames
   - Verify furigana is attached to correct kanji
"""
        
        # Add context for multi-part chapters
        if total_parts > 1:
            prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""
            
            if part_idx > 1:
                prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2).
Convert any # (H1) headings to ## (H2)."""
        
        prompt += """

IMPORTANT: Return ONLY the polished markdown. Do not add explanations.
Preserve all images and illustrations.

Polish the following Japanese content:"""
        
        return prompt
    
    def _create_general_polish_prompt(
        self,
        chapter_name: str,
        part_idx: int,
        total_parts: int
    ) -> str:
        """Create prompt for general content (fallback)."""
        book_info = f' from "{self.book_title}"' if self.book_title else ""
        
        prompt = f"""You are an expert document editor. Polish this OCR-extracted content from "{chapter_name}"{book_info}.

Your tasks:

1. **Remove page artifacts**:
   - Delete page numbers, headers, and footers
   - Remove page separators (---)
   - Join sentences broken across pages

2. **Fix structure**:
   - Main title: # (H1)
   - Sections: ## (H2), subsections: ### (H3)
   - Remove excessive blank lines

3. **Clean up text**:
   - Fix obvious OCR errors
   - Join hyphenated words at line breaks
   - Preserve emphasis (*italic*, **bold**)

4. **Preserve content**:
   - Keep all images and tables
   - Maintain lists and quotes
   - Preserve code blocks if present"""
        
        # Add context for multi-part chapters
        if total_parts > 1:
            prompt += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""
            
            if part_idx > 1:
                prompt += """
IMPORTANT: Since this is a continuation, your MAXIMUM heading level is ## (H2)."""
        
        prompt += """

Return ONLY the polished markdown.

Polish the following content:"""
        
        return prompt
    
    def _post_process_markdown(self, markdown: str) -> str:
        """Post-process the polished markdown to clean up any issues."""
        # Remove any leading/trailing whitespace
        markdown = markdown.strip()
        
        # Remove images that point to non-existent files
        # Find all markdown images
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        images_to_remove = []
        
        for match in re.finditer(image_pattern, markdown):
            # alt_text = match.group(1)
            image_path = match.group(2)
            
            # Skip URLs (http/https)
            if image_path.startswith(('http://', 'https://')):
                continue
            
            # Check if it's a relative path that should exist
            if image_path.startswith('../'):
                # Resolve the path relative to the output directory
                # The markdown is in output/book_title/polished_markdown/
                # Images are in output/book_title/images/
                resolved_path = self.output_dir.parent / image_path[3:]  # Remove '../'
                
                if not resolved_path.exists():
                    images_to_remove.append(match.group(0))
                    logger.debug(f"Removing image with non-existent file: {image_path}")
            elif not image_path.startswith(('/', 'http')):
                # Relative path without ../ or absolute path - likely invalid
                images_to_remove.append(match.group(0))
                logger.debug(f"Removing image with invalid path: {image_path}")
        
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
