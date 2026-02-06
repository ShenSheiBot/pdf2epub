"""
HTML Translation Processor.

Translates compressed HTML content line-by-line while preserving structure.
This is a standalone processor that does NOT use the V2 executor architecture.
"""

import re
import json
import time
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from pdf2epub.utils.llm_client import LLMClient
from pdf2epub.processors.tracker import ProcessingTracker

from .prompts import create_compressed_translation_prompt, create_compressed_retry_prompt


class HTMLTranslateProcessor:
    """
    Processor for translating compressed HTML content.

    Works with HTMLCompressor output: one translation unit per line.
    Much simpler than raw HTML translation - just translate line by line.

    Input: compressed_units/ (.md files with compressed content)
    Output: translated_compressed/ (.md files with translated lines)

    This processor is standalone and does not use the V2 executor architecture.
    It processes files directly with its own retry and validation logic.
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
        self.config = config
        self.book_title = book_title
        self.max_workers = max_workers if max_workers != 4 else config.get('max_concurrent_workers', 4)
        self.resume = resume
        self.use_longest_on_failure = use_longest_on_failure

        self.source_language = source_language
        self.target_language = target_language

        # Setup directories
        self.input_dir = Path("output") / book_title / "compressed_units"
        self.output_dir = Path("output") / book_title / "translated_compressed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default translation models if not provided
        self.translation_models = translation_models or config.get('html_translation_models') or config.get('translation', {}).get('models') or [
            {"provider": "gemini", "model": "gemini-2.5-pro", "api_retries": 2, "validation_retries": 2},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_retries": 2, "validation_retries": 1}
        ]

        # Get validation settings
        validation_config = config.get('validation_strategy', {})
        self.validate_target_language = validation_config.get('validate_chinese_translation', True)

        # Load entities if available
        self.entities = None
        if use_entities:
            self.entities = self._load_entities()
        elif use_entities is None:
            entities_file = Path("output") / self.book_title / "translation_entities.json"
            if entities_file.exists():
                logger.info("Auto-detected translation entities file")
                self.entities = self._load_entities()

        # Initialize ProcessingTracker
        tracker_path = self.output_dir / "processing_tracker.json"
        self.processing_tracker = ProcessingTracker(tracker_path, "HTMLTranslateProcessor")

        # Initialize LLM client
        self.llm_client = LLMClient(config)

        # Track retry context for enhanced prompts
        self._retry_context: Dict[str, str] = {}

    def _wrap_lines_with_div(self, content: str) -> str:
        """Wrap each line with <div> tags for line preservation."""
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

    def get_model_configs(self) -> List[Dict]:
        """Get the model configurations for translation."""
        return self.translation_models

    def build_prompt(self, content: str, file_name: str) -> str:
        """
        Build the HTML translation prompt.

        Args:
            content: Compressed content (one translation unit per line)
            file_name: File name for tracking

        Returns:
            Prompt string with content appended
        """
        # Wrap each line with <div> tags to preserve line structure
        marked_content = self._wrap_lines_with_div(content)

        # Create the translation prompt
        prompt = create_compressed_translation_prompt(
            source_language=self.source_language,
            target_language=self.target_language,
            entities=self.entities
        )

        # Add retry context if this is a retry
        retry_error = self._retry_context.get(file_name)
        if retry_error:
            prompt += create_compressed_retry_prompt(retry_error)

        # Return prompt with content appended
        return f"{prompt}\n\n{marked_content}"

    def clean_response(self, response: str) -> str:
        """
        Clean LLM response.

        Removes markdown code blocks if present.
        Extracts content from <div>...</div> wrappers.
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

    def _process_single_file(self, file_name: str) -> Dict:
        """
        Process a single file with retry logic.

        Args:
            file_name: Name of the file (without extension)

        Returns:
            Dict with status information
        """
        input_file = self.input_dir / f"{file_name}.md"
        output_file = self.output_dir / f"{file_name}.md"

        # Check if already completed
        if self.resume and output_file.exists():
            if self.processing_tracker.is_file_completed(file_name):
                logger.debug(f"Skipping completed file: {file_name}")
                return {"file": file_name, "status": "skipped", "reason": "already completed"}

        # Read input content
        if not input_file.exists():
            return {"file": file_name, "status": "error", "reason": f"Input file not found: {input_file}"}

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {"file": file_name, "status": "error", "reason": f"Failed to read input: {e}"}

        if not content.strip():
            # Empty file, just copy
            output_file.write_text("")
            self.processing_tracker.mark_file_completed(file_name)
            return {"file": file_name, "status": "success", "reason": "empty file"}

        # Process with retry logic
        model_configs = self.get_model_configs()
        last_error = None
        longest_response = None
        longest_length = 0

        for model_config in model_configs:
            provider = model_config.get('provider', 'gemini')
            model = model_config.get('model', 'gemini-2.5-pro')
            api_retries = model_config.get('api_retries', 2)
            validation_retries = model_config.get('validation_retries', 2)

            for attempt in range(api_retries + validation_retries):
                try:
                    # Build prompt
                    prompt = self.build_prompt(content, file_name)

                    # Call LLM
                    response = self.llm_client.generate(
                        provider=provider,
                        model=model,
                        prompt=prompt
                    )

                    # Clean response
                    cleaned = self.clean_response(response)

                    # Track longest response for fallback
                    if len(cleaned) > longest_length:
                        longest_response = cleaned
                        longest_length = len(cleaned)

                    # Validate
                    is_valid, reason = self.validate_output(content, cleaned, file_name)

                    if is_valid:
                        # Save output
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(cleaned)
                        self.processing_tracker.mark_file_completed(file_name)
                        return {"file": file_name, "status": "success", "model": f"{provider}/{model}"}

                    last_error = reason
                    logger.warning(f"Validation failed for {file_name}: {reason}")

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Error processing {file_name} with {provider}/{model}: {e}")
                    time.sleep(1)  # Brief pause before retry

        # All attempts failed
        if self.use_longest_on_failure and longest_response:
            logger.warning(f"Using longest response for {file_name} after all attempts failed")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(longest_response)
            self.processing_tracker.mark_file_completed(file_name)
            return {"file": file_name, "status": "fallback", "reason": last_error}

        return {"file": file_name, "status": "error", "reason": last_error}

    def process_all_files(self) -> Dict:
        """
        Process all files in the input directory.

        Returns:
            Summary dict with counts of successful, failed, etc.
        """
        # Get list of input files
        input_files = sorted(self.input_dir.glob("*.md"))
        file_names = [f.stem for f in input_files]

        if not file_names:
            logger.warning(f"No input files found in {self.input_dir}")
            return {"total": 0, "successful": 0, "failed": 0, "skipped": 0}

        logger.info(f"Processing {len(file_names)} files with {self.max_workers} workers")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, name): name
                for name in file_names
            }

            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get('status', 'unknown')
                    if status == 'success':
                        logger.info(f"Completed: {file_name}")
                    elif status == 'skipped':
                        logger.debug(f"Skipped: {file_name}")
                    elif status == 'fallback':
                        logger.warning(f"Fallback: {file_name}")
                    else:
                        logger.error(f"Failed: {file_name} - {result.get('reason', 'unknown')}")
                except Exception as e:
                    logger.error(f"Exception processing {file_name}: {e}")
                    results.append({"file": file_name, "status": "error", "reason": str(e)})

        # Summarize
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'error')
        skipped = sum(1 for r in results if r.get('status') == 'skipped')
        fallback = sum(1 for r in results if r.get('status') == 'fallback')

        return {
            "total": len(file_names),
            "successful": successful + fallback,
            "failed": failed,
            "skipped": skipped,
            "fallback": fallback
        }

    def process_specific_files(self, file_names: List[str]) -> Dict:
        """
        Process specific files by name.

        Args:
            file_names: List of file names (without extension)

        Returns:
            Summary dict with counts
        """
        if not file_names:
            return {"total": 0, "successful": 0, "failed": 0, "skipped": 0}

        logger.info(f"Processing {len(file_names)} specific files with {self.max_workers} workers")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, name): name
                for name in file_names
            }

            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get('status', 'unknown')
                    if status == 'success':
                        logger.info(f"Completed: {file_name}")
                    elif status == 'skipped':
                        logger.debug(f"Skipped: {file_name}")
                    elif status == 'fallback':
                        logger.warning(f"Fallback: {file_name}")
                    else:
                        logger.error(f"Failed: {file_name} - {result.get('reason', 'unknown')}")
                except Exception as e:
                    logger.error(f"Exception processing {file_name}: {e}")
                    results.append({"file": file_name, "status": "error", "reason": str(e)})

        # Summarize
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'error')
        skipped = sum(1 for r in results if r.get('status') == 'skipped')
        fallback = sum(1 for r in results if r.get('status') == 'fallback')

        return {
            "total": len(file_names),
            "successful": successful + fallback,
            "failed": failed,
            "skipped": skipped,
            "fallback": fallback
        }
