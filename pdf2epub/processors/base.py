"""
Base processor class for markdown transformations.

This module provides the abstract base class for all markdown processors.
The actual processing pipeline is handled by the V2 executor.
Processors only need to implement the ProcessorProtocol methods.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..core._protocol import ProcessContext


class BaseMarkdownProcessor(ABC):
    """
    Abstract base class for markdown processors.

    Implements the ProcessorProtocol for use with the V2 executor.
    Subclasses must implement:
    - name (property)
    - build_prompt(content, context)
    - clean_response(response)
    - post_process(result, context)
    - get_model_configs()
    """

    def __init__(
        self,
        config: Dict,
        book_title: str,
        input_dir: str,
        output_dir: str,
        max_workers: int = 4,
        resume: bool = False,
        use_longest_on_failure: bool = False
    ):
        """
        Initialize the base processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            input_dir: Input directory name (e.g., "polished_markdown")
            output_dir: Output directory name (e.g., "translated")
            max_workers: Maximum number of concurrent workers
            resume: Whether to resume from previous progress
            use_longest_on_failure: If True, use longest response when all attempts fail validation
        """
        self.config = config
        self.book_title = book_title
        self.max_workers = max_workers if max_workers != 4 else config.get('max_concurrent_workers', 4)
        self.resume = resume
        self.use_longest_on_failure = use_longest_on_failure

        # Setup directories (V2 infrastructure handles directory creation)
        self.input_dir = Path("output") / book_title / input_dir
        self.output_dir = Path("output") / book_title / output_dir

    # ==================== ProcessorProtocol methods ====================

    @property
    def name(self) -> str:
        """
        Processor name for logging and tracking.

        Returns:
            Short identifier like "polish" or "translate"
        """
        return self.__class__.__name__.replace("Processor", "").lower()

    @abstractmethod
    def build_prompt(self, content: str, context: "ProcessContext") -> Any:
        """
        Build the prompt to send to LLM.

        Args:
            content: The content to process
            context: Processing context (file info, language, previous context, etc.)

        Returns:
            Either:
            - str: Simple prompt
            - List[Dict]: Multi-turn conversation (for context injection)
        """
        pass

    def clean_response(self, response: str) -> str:
        """
        Clean the raw LLM response.

        Remove markdown code blocks, fix formatting, etc.
        This is called BEFORE validation.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response
        """
        return self.clean_markdown_response(response)

    @abstractmethod
    def post_process(self, result: str, context: "ProcessContext") -> str:
        """
        Post-process the validated result.

        Apply any final transformations after validation passes.
        This is called AFTER validation.

        Args:
            result: Cleaned and validated result
            context: Processing context

        Returns:
            Final processed result
        """
        pass

    @abstractmethod
    def get_model_configs(self) -> List[Dict]:
        """
        Get model configurations for LLM calls.

        Returns:
            List of model config dicts with provider, model, retries, etc.
        """
        pass

    # ==================== Helper methods ====================

    def clean_markdown_response(self, content: str) -> str:
        """
        Clean up markdown response from LLM.

        Args:
            content: Raw response from LLM

        Returns:
            Cleaned markdown content
        """
        lines = content.strip().split('\n')

        # Look for code block markers in first 3 non-empty lines
        non_empty_count = 0
        code_block_start = -1

        for i, line in enumerate(lines):
            if line.strip():  # Non-empty line
                non_empty_count += 1
                # Check if this line is a code block marker
                if line.strip() in ['```markdown', '```'] or line.strip().startswith('```'):
                    code_block_start = i + 1  # Start from the line after the marker
                    break
                if non_empty_count >= 3:
                    break

        # If we found a code block marker, remove everything before and including it
        if code_block_start > 0:
            lines = lines[code_block_start:]

        # Rejoin the content
        content = '\n'.join(lines)

        # Also handle case where ``` appears at the end
        if content.strip().endswith('```'):
            lines = content.strip().split('\n')
            if lines[-1].strip() == '```':
                lines = lines[:-1]
                content = '\n'.join(lines)

        return content.strip()

    def _is_image_only_content(self, content: str, min_text_chars: int = 100) -> bool:
        """
        Check if content is primarily images with minimal text.

        Used to skip LLM processing for pages that are just images.

        Args:
            content: Content to check
            min_text_chars: Minimum text characters to consider as having meaningful content

        Returns:
            True if content is image-only (should skip processing)
        """
        # Remove markdown images: ![alt](src)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', content)

        # Remove HTML images with optional wrapping divs
        text = re.sub(r'(?:<div[^>]*>)?\s*<img\s+[^>]*/?\s*>\s*(?:</div>)?', '', text, flags=re.IGNORECASE)

        # Remove markdown headings (they're just titles, not content)
        text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)

        # Remove whitespace
        text = text.strip()

        # If remaining text is very short, it's image-only
        return len(text) < min_text_chars
