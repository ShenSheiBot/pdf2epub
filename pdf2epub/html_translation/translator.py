"""
HTML Translation Processor.

Translates compressed HTML content line-by-line while preserving structure.
"""

import re
import json
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from loguru import logger

from pdf2epub.processors.base import BaseMarkdownProcessor
from pdf2epub.processors.utils.split_manager import SplitManager
from pdf2epub.processors.tracker import ProcessingTracker

from .prompts import create_compressed_translation_prompt, create_compressed_retry_prompt


class HTMLTranslateProcessor(BaseMarkdownProcessor):
    """
    Processor for translating compressed HTML content.

    Works with HTMLCompressor output: one translation unit per line.
    Much simpler than raw HTML translation - just translate line by line.

    Input: compressed_units/ (.txt files with compressed content)
    Output: translated_compressed/ (.txt files with translated lines)
    """

    def __init__(
        self,
        config: Dict,
        book_title: str,
        source_language: str = "Japanese",
        target_language: str = "Chinese",
        max_workers: int = 4,
        resume: bool = False,
        translation_models: Optional[List] = None,
        use_entities: Optional[bool] = None,
        use_longest_on_failure: bool = False
    ):
        """
        Initialize HTML translation processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book
            source_language: Source language
            target_language: Target language
            max_workers: Concurrent workers
            resume: Resume from progress
            translation_models: Model configurations
            use_entities: Use entity consistency file
            use_longest_on_failure: Fallback behavior
        """
        super().__init__(
            config=config,
            book_title=book_title,
            input_dir="compressed_units",
            output_dir="translated_compressed",
            max_workers=max_workers,
            resume=resume,
            use_longest_on_failure=use_longest_on_failure
        )

        self.source_language = source_language
        self.target_language = target_language

        # Set default translation models if not provided
        self.translation_models = translation_models or config.get('html_translation_models') or [
            {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "max_retries": 2}
        ]

        # Get validation settings
        validation_config = config.get('validation_strategy', {})
        self.validate_target_language = validation_config.get('validate_chinese_translation', True)

        # Load entities if available
        self.entities = self._load_entities() if use_entities else None
        if use_entities is None:
            entities_file = Path("output") / self.book_title / "translation_entities.json"
            if entities_file.exists():
                logger.info("Auto-detected translation entities file")
                self.entities = self._load_entities()

        # Initialize ProcessingTracker
        tracker_path = self.output_dir / "processing_tracker.json"
        self.processing_tracker = ProcessingTracker(tracker_path, "HTMLTranslateProcessor")

        # Initialize SplitManager
        splitting_config = config.get('splitting', {})
        self.split_manager = SplitManager(
            tracker=self.processing_tracker,
            output_dir=self.output_dir,
            default_max_tokens=self.get_max_tokens_per_part(),
            max_resplits=splitting_config.get('max_resplits', 3),
            consecutive_failures_threshold=splitting_config.get('consecutive_failures_threshold', 2)
        )

        # Track retry context for enhanced prompts
        self._retry_context: Dict[str, str] = {}

    def _wrap_lines_with_div(self, content: str) -> str:
        """将每行内容用 <div> 包裹，形成单行输出"""
        lines = content.split('\n')
        return ''.join(f'<div>{line}</div>' for line in lines)

    def _load_entities(self) -> Optional[Dict]:
        """Load translation entities from file."""
        entities_file = Path("output") / self.book_title / "translation_entities.json"
        if entities_file.exists():
            try:
                with open(entities_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load entities: {e}")
        return None

    def get_operation_name(self, file_name: str) -> str:
        """Get the operation name for logging."""
        return f"Translate HTML {file_name}"

    def get_model_configs(self) -> List[Dict]:
        """Get the model configurations for translation."""
        return self.translation_models

    def get_split_strategy(self) -> str:
        """Use compressed content splitting strategy (line-based)."""
        return 'compressed'

    def build_prompt(self, content: str, unit_key: str, **context) -> str:
        """
        Build the HTML translation prompt.

        Args:
            content: Compressed content (one translation unit per line)
            unit_key: Unit identifier for tracking
            **context: Context including file_name, part_idx, total_parts

        Returns:
            Prompt string with content appended
        """
        # Wrap each line with <div> tags to preserve line structure
        # LLMs preserve standard HTML tags better than custom markers like <nl/>
        marked_content = self._wrap_lines_with_div(content)

        # Create the translation prompt
        prompt = create_compressed_translation_prompt(
            source_language=self.source_language,
            target_language=self.target_language,
            entities=self.entities
        )

        # Add retry context if this is a retry
        retry_error = self._retry_context.get(unit_key)
        if retry_error:
            prompt += create_compressed_retry_prompt(retry_error)

        # Return prompt with content appended
        return f"{prompt}\n\n{marked_content}"

    def post_process(self, result: str, **context) -> str:
        """
        Post-process the translated result.

        No post-processing needed for HTML translation.

        Args:
            result: Cleaned LLM response
            **context: Processing context

        Returns:
            Result unchanged
        """
        return result

    def get_context_for_next_part(self, content: str, result: str, **context) -> Optional[Dict]:
        """
        Get context to inject into the next part's build_prompt.

        HTML translation does not need context injection.

        Args:
            content: Original content of this part
            result: Processed result of this part
            **context: Processing context

        Returns:
            None (no context injection needed)
        """
        return None

    def clean_response(self, response: str) -> str:
        """
        Clean LLM response.

        Removes markdown code blocks if present.
        Extracts content from <div>...</div> wrappers.
        Overrides base class method.
        """
        # Remove markdown code block wrappers
        if response.startswith("```"):
            lines = response.split('\n')
            # Remove first line (```xxx or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = '\n'.join(lines)

        cleaned = response.strip()

        # Remove all real newlines (LLM may add arbitrary line breaks for formatting)
        cleaned = cleaned.replace('\n', '')

        # Extract content from <div>...</div> wrappers
        div_pattern = re.compile(r'<div>(.*?)</div>', re.DOTALL)
        matches = div_pattern.findall(cleaned)

        if matches:
            # Filter out empty <div></div> that LLM may produce
            matches = [m for m in matches if m.strip()]
            return '\n'.join(matches)

        # Fallback: try old <nl/> format for backward compatibility
        if '<nl' in cleaned:
            cleaned = re.sub(r'<nl\s*/?>', '\n', cleaned)
            return cleaned

        # Cannot parse, return as-is
        return cleaned

    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate translated compressed output.

        Checks:
        1. Line count matches (each <div> represents one line)
        2. Target language content present (if configured)

        Args:
            original: Original compressed content (with \\n line breaks)
            processed: Translated compressed content (cleaned, with newlines restored)
            file_name: Name of the file

        Returns:
            Tuple of (is_valid, reason)
        """
        # Count lines - original has \n separators, processed has \n restored from <div> extraction
        # Line count = number of \n + 1
        original_line_count = original.count('\n') + 1
        processed_line_count = processed.count('\n') + 1

        # 1. Line count validation
        if original_line_count != processed_line_count:
            self._retry_context[file_name] = "div_count_mismatch"
            return False, f"Line count mismatch: expected {original_line_count}, got {processed_line_count}"

        # 2. Target language validation
        if self.validate_target_language:
            target_lower = self.target_language.lower()
            if target_lower in ["chinese", "中文", "chinese simplified", "zh", "zh-cn"]:
                if not self._contains_chinese(processed):
                    self._retry_context[file_name] = "language_wrong"
                    return False, "Translation does not contain Chinese characters"

        # Clear retry context on success
        if file_name in self._retry_context:
            del self._retry_context[file_name]

        return True, "OK"

    def _contains_chinese(self, text: str) -> bool:
        """
        Check if text contains Chinese characters.

        Uses sampling for efficiency on large texts.
        """
        # Remove any HTML tags that might be in the content
        text_only = re.sub(r'<[^>]+>', '', text)

        if not text_only.strip():
            # No text content, consider valid
            return True

        # Check for Chinese characters
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

        # Sample check for efficiency
        sample_size = min(1000, len(text_only))
        sample = text_only[:sample_size]

        matches = chinese_pattern.findall(sample)
        return len(matches) >= 5  # At least 5 Chinese chars in sample
