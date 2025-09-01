"""
Polish processor for OCR-extracted markdown content.

This processor cleans up OCR-extracted markdown, removing artifacts,
fixing formatting, and organizing content structure.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .base import BaseMarkdownProcessor
from .utils.truncation import NGramTruncationDetector
from .utils.content_splitter import split_content_intelligently, split_content_simple
from .utils.image_restore import restore_lost_images
from pdf2epub.utils.network_utils import GeminiClient


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
        
        # Initialize Gemini client for intelligent splitting if available
        self.gemini_client = None
        if config.get("google_api_key"):
            self.gemini_client = GeminiClient(config["google_api_key"])
    
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
        # Determine if we need to split the content
        max_tokens_per_part = self._determine_max_tokens()
        estimated_tokens = len(content) // 4
        
        if estimated_tokens > max_tokens_per_part:
            logger.info(f"Content has ~{estimated_tokens:,} tokens, splitting into parts")
            parts = self._split_content(content, max_tokens_per_part)
        else:
            parts = [content]
        
        logger.info(f"Processing {len(parts)} part(s) for {file_name}")
        
        # Process each part
        polished_parts = []
        chapter_name = self.get_operation_name(file_name)
        
        for part_idx, part_content in enumerate(parts, 1):
            polished_part = self._polish_part(
                part_content=part_content,
                chapter_name=chapter_name,
                part_idx=part_idx,
                total_parts=len(parts),
                original_content=content
            )
            polished_parts.append(polished_part)
        
        # Combine parts if multiple
        if len(parts) > 1:
            combined = "\n\n".join(polished_parts)
            # Restore any lost images
            combined = restore_lost_images(content, combined)
            return combined
        else:
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
            
            if not has_limited_model:
                max_tokens_per_part = 30000
                logger.info("Using max_tokens_per_part=30000 (no limited-context models detected)")
            else:
                logger.info("Using max_tokens_per_part=10000 (limited-context model detected)")
        
        return max_tokens_per_part
    
    def _split_content(self, content: str, max_tokens: int) -> List[str]:
        """Split content into manageable parts."""
        if self.gemini_client:
            try:
                return split_content_intelligently(
                    content, max_tokens, self.gemini_client
                )
            except Exception as e:
                logger.warning(f"Intelligent split failed: {e}, using simple split")
        
        return split_content_simple(content, max_tokens)
    
    def _polish_part(
        self,
        part_content: str,
        chapter_name: str,
        part_idx: int,
        total_parts: int,
        original_content: str
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
   - Convert ALL citation formats ($^1$, ¹, [1], etc.) to [^1]
   - Convert footnote definitions to "[^1]: content" format
   - CRITICAL: Only include footnotes ACTUALLY in the source
   - Preserve exact footnote content and numbering
   - Check for both inline and end-of-section footnotes
   - Never invent or add missing footnotes

4. **Preserve academic elements**:
   - Keep equations, formulas, and mathematical notation
   - Preserve code blocks and technical examples
   - Maintain definition lists and theorems
   - Keep cross-references ("see Section 2.3")

5. **Organize bibliography**:
   - Move all footnotes to "## Notes" section if they exist
   - Organize references under "## References" if present
   - Format citations consistently:
     * Books: Author(s). (Year). *Title*. Publisher.
     * Articles: Author(s). (Year). "Title." *Journal*, Volume(Issue), pages.
   - Final structure: Main Content → ## Notes → ## References

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
        
        # Fix common markdown issues
        # 1. Ensure headers have space after #
        markdown = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', markdown, flags=re.MULTILINE)
        
        # 2. Ensure blank lines around headers
        markdown = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(#{1,6} [^\n]+)\n([^\n#])', r'\1\n\n\2', markdown)
        
        # 3. Remove excessive blank lines (more than 2)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # 4. Ensure images have blank lines around them
        markdown = re.sub(r'([^\n])\n(!\[)', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(!\[[^\]]*\]\([^\)]*\))\n([^\n])', r'\1\n\n\2', markdown)
        
        return markdown
