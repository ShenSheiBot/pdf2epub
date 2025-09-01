"""
Translation processor for markdown content.

This processor translates markdown content from one language to another
while preserving formatting and structure.
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from .base import BaseMarkdownProcessor
from .utils.truncation import LLMTruncationDetector


class TranslateProcessor(BaseMarkdownProcessor):
    """Processor for translating markdown content."""
    
    def __init__(
        self,
        config: Dict,
        book_title: str,
        source_language: str = "English",
        target_language: str = "Chinese",
        max_workers: int = 4,
        resume: bool = False,
        translation_models: Optional[list] = None
    ):
        """
        Initialize the translation processor.
        
        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            source_language: Source language for translation
            target_language: Target language for translation
            max_workers: Maximum number of concurrent workers
            resume: Whether to resume from previous progress
            translation_models: Optional override for model configurations
        """
        super().__init__(
            config=config,
            book_title=book_title,
            input_dir="polished_markdown",
            output_dir="translated",
            max_workers=max_workers,
            resume=resume
        )
        
        self.source_language = source_language
        self.target_language = target_language
        
        # Set default translation models if not provided
        self.translation_models = translation_models or [
            {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2},
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "max_retries": 2}
        ]
        
        # Initialize truncation detector
        self.truncation_detector = LLMTruncationDetector(
            llm_client=self.llm_client,
            num_lines=3
        )
        
        # Store language info in progress
        if self.progress.get("target_language") != target_language:
            if self.progress.get("target_language") is not None:
                logger.warning(
                    f"Target language changed from {self.progress['target_language']} "
                    f"to {target_language}"
                )
                if resume:
                    logger.warning("Resuming with different target language may produce mixed results")
            self.progress["target_language"] = target_language
            self.progress["source_language"] = source_language
            self.save_progress()
    
    def get_progress_filename(self) -> str:
        """Get the name for the progress file."""
        return "translation_progress"
    
    def get_progress_key(self) -> str:
        """Get the key used in progress tracking."""
        return "translations"
    
    def get_operation_name(self, file_name: str) -> str:
        """Get the operation name for logging."""
        return f"Translate {file_name}"
    
    def process_content(
        self,
        content: str,
        file_name: str,
        **kwargs
    ) -> str:
        """
        Process markdown content by translating it.
        
        Args:
            content: The markdown content to translate
            file_name: Name of the file being processed
            **kwargs: Additional arguments
        
        Returns:
            Translated markdown content
        """
        # Create the translation prompt
        prompt = self._create_translation_prompt()
        
        # Create multi-part content for the LLM
        multi_part_content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": content}
        ]
        
        # Generate translation
        translated_content = self.llm_client.generate(
            prompt=multi_part_content,
            model_configs=self.translation_models,
            operation_name=self.get_operation_name(file_name)
        )
        
        # Clean the response
        translated_content = self.clean_markdown_response(translated_content)
        
        return translated_content
    
    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate the translated output using LLM-based truncation detection.
        
        Args:
            original: Original content
            processed: Translated content
            file_name: Name of the file
        
        Returns:
            Tuple of (is_valid, reason)
        """
        is_truncated, reason, details = self.truncation_detector.detect(
            original=original,
            processed=processed,
            source_language=self.source_language,
            target_language=self.target_language
        )
        
        # Log the summary
        summary = self.truncation_detector.get_summary(is_truncated, reason, details)
        if is_truncated:
            logger.warning(f"{file_name} translation truncation detected:\n{summary}")
        else:
            logger.info(f"{file_name} translation validated:\n{summary}")
        
        return not is_truncated, reason
    
    def process_file(
        self,
        input_path,
        output_path,
        **kwargs
    ) -> bool:
        """
        Process a single markdown file for translation.
        
        Override to add language information to progress tracking.
        """
        success = super().process_file(input_path, output_path, **kwargs)
        
        if success:
            # Update progress with language info
            file_key = str(input_path.stem)
            progress_key = self.get_progress_key()
            if file_key in self.progress[progress_key]:
                self.progress[progress_key][file_key].update({
                    "source_language": self.source_language,
                    "target_language": self.target_language
                })
                self.save_progress()
        
        return success
    
    def _create_translation_prompt(self) -> str:
        """Create the prompt for translation."""
        return f"""You are a professional translator specializing in academic and literary texts.

Translate the following markdown content from {self.source_language} to {self.target_language}.

IMPORTANT REQUIREMENTS:
1. **Preserve ALL markdown formatting**: Keep headers (#, ##, ###), emphasis (*italic*, **bold**), lists, quotes, code blocks, etc.
2. **Keep image links unchanged**: Do not translate or modify image paths like ![...](../images/xxx.png)
3. **Translate footnotes properly**: 
   - Keep the footnote format [^1], [^2], etc. unchanged
   - Translate the footnote content but keep the reference numbers
4. **Maintain document structure**: Keep the same paragraph breaks, section divisions, and overall layout
5. **For academic texts**: Use appropriate academic terminology in the target language
6. **For literary texts**: Preserve the style and tone of the original
7. **Do NOT add explanations**: Return ONLY the translated markdown, no explanations or comments

Translate the following content:"""