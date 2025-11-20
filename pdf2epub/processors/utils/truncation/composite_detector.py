"""
Composite truncation detector with fallback strategies.

This module provides a composite detector that combines n-gram detection
with LLM verification for more accurate truncation detection.
"""

from typing import Tuple, Dict, List, Optional
from loguru import logger
from .base import BaseTruncationDetector
from .ngram_detector import NGramTruncationDetector
from .llm_detector import LLMTruncationDetector


class CompositeTruncationDetector(BaseTruncationDetector):
    """
    Composite truncation detector that combines n-gram and LLM detection.

    First tries n-gram detection, then falls back to LLM verdict
    if n-gram detection suggests truncation due to unique content loss.
    """

    def __init__(
        self,
        llm_client,
        min_unique_preserved_ratio: float = 0.60,
        allow_deduplication: bool = True,
        truncation_check_lines: int = 5,
        cheapest_model_configs: Optional[List[Dict]] = None,
        task_type: str = "translate"
    ):
        """
        Initialize the composite detector.

        Args:
            llm_client: LLM client for verification
            min_unique_preserved_ratio: Minimum ratio for n-gram detector
            allow_deduplication: Whether deduplication is acceptable
            truncation_check_lines: Number of lines to check with LLM
            cheapest_model_configs: Optional model configs for LLM fallback
            task_type: Type of task - "translate" or "polish" (default: "translate")
        """
        self.llm_client = llm_client
        self.cheapest_model_configs = cheapest_model_configs
        self.task_type = task_type

        # Initialize sub-detectors
        self.ngram_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=min_unique_preserved_ratio,
            allow_deduplication=allow_deduplication
        )
        self.llm_detector = LLMTruncationDetector(
            llm_client=llm_client,
            num_lines=truncation_check_lines,
            task_type=task_type
        )

    def detect(
        self,
        original: str,
        processed: str,
        **kwargs
    ) -> Tuple[bool, str, Dict]:
        """
        Composite truncation detection with LLM fallback.

        First tries n-gram detection, then falls back to LLM verdict
        if n-gram detection suggests truncation due to unique content loss.

        Args:
            original: Original content
            processed: Processed content
            **kwargs: Additional arguments (passed to sub-detectors)

        Returns:
            Tuple of (is_truncated, reason, details)
        """
        # First try n-gram detection
        is_truncated, reason, details = self.ngram_detector.detect(
            original=original,
            processed=processed
        )

        # If n-gram detector says it's truncated due to unique content loss, try LLM as fallback
        if is_truncated and "unique content lost" in reason.lower():
            logger.info("N-gram detector flagged truncation, checking with LLM verdict...")

            try:
                # Get cheapest models for LLM check
                model_configs = self.cheapest_model_configs
                if not model_configs:
                    from ....utils.model_utils import get_cheapest_model_configs
                    model_configs = get_cheapest_model_configs(
                        self.llm_client.config,
                        max_models=1
                    )

                if not model_configs:
                    model_configs = [
                        {"provider": "gemini", "model": "gemini-2.0-flash", "max_retries": 1}
                    ]

                # Try the LLM truncation detector
                llm_is_truncated, llm_reason, llm_details = self.llm_detector.detect(
                    original=original,
                    processed=processed
                )

                if not llm_is_truncated:
                    # LLM says it's complete, override n-gram detector
                    logger.info("LLM verdict: Content is complete despite n-gram concerns")
                    details['llm_verdict'] = 'complete'
                    details['llm_override'] = True
                    return False, "LLM verified content is complete (n-gram override)", details
                else:
                    # LLM agrees it's truncated
                    details['llm_verdict'] = 'truncated'
                    return True, f"{reason} (LLM confirmed)", details

            except Exception as e:
                logger.warning(f"LLM verdict check failed: {e}")
                details['llm_error'] = str(e)
                # Fall back to original n-gram result
                return is_truncated, reason, details

        # If n-gram says it's complete, trust it
        return is_truncated, reason, details

    def get_summary(self, is_truncated: bool, reason: str, details: Dict) -> str:
        """
        Get summary of truncation detection.

        Args:
            is_truncated: Whether truncation was detected
            reason: Reason for the decision
            details: Analysis details

        Returns:
            Summary string
        """
        # Use n-gram detector's summary method
        summary = self.ngram_detector.get_summary(is_truncated, reason, details)

        # Add LLM verdict if available
        if details.get('llm_verdict'):
            summary += f"\n  LLM verdict: {details['llm_verdict']}"

        return summary
