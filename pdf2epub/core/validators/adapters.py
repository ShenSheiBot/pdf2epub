"""
Validator adapters - Bridge between detector implementations and IndividualValidator.

This module provides adapters that wrap existing detectors (truncation, etc.)
to implement IndividualValidator protocol for use as Individual validators.

Note: Batch validators (like AgentVerifier) have their own interface and
should NOT be adapted - they are fundamentally different.
"""

from typing import Optional, Dict, Any
from loguru import logger

from .._protocol import IndividualValidator, ValidationResult
from .truncation import (
    BaseTruncationDetector,
    NGramTruncationDetector,
)


class TruncationValidatorAdapter:
    """
    Adapts BaseTruncationDetector to IndividualValidator protocol.

    Individual validator - fast screening for truncation.
    """

    def __init__(
        self,
        detector: BaseTruncationDetector,
        name: Optional[str] = None
    ):
        """
        Initialize adapter.

        Args:
            detector: The truncation detector to wrap
            name: Validator name (defaults to detector class name)
        """
        self._detector = detector
        self._name = name or detector.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def validate(
        self,
        original: str,
        processed: str,
        file_key: str
    ) -> ValidationResult:
        """
        Validate using truncation detector.

        Args:
            original: Original content
            processed: Processed content
            file_key: File identifier

        Returns:
            ValidationResult
        """
        try:
            is_truncated, reason, details = self._detector.detect(original, processed)

            # Truncated = invalid
            is_valid = not is_truncated

            # Determine confidence based on details
            confidence = "high"
            if details.get("token_ratio", 1.0) > 0.8:
                confidence = "medium"  # Close to threshold

            return ValidationResult(
                key=file_key,
                is_valid=is_valid,
                reason=reason,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"{file_key}: Truncation detection failed: {e}")
            # On error, pass validation (don't block on detector failure)
            return ValidationResult(
                key=file_key,
                is_valid=True,
                reason=f"Detector error: {e}",
                confidence="low"
            )


class LengthValidator:
    """
    Simple length-based validator.

    Implements IndividualValidator protocol.
    Checks output isn't drastically shorter than input.
    """

    def __init__(
        self,
        min_ratio: float = 0.3,
        max_ratio: float = 3.0
    ):
        """
        Initialize length validator.

        Args:
            min_ratio: Minimum output/input length ratio
            max_ratio: Maximum output/input length ratio
        """
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio

    @property
    def name(self) -> str:
        return "LengthValidator"

    def validate(
        self,
        original: str,
        processed: str,
        file_key: str
    ) -> ValidationResult:
        """Validate based on length ratio."""
        if not original:
            return ValidationResult(
                key=file_key,
                is_valid=True,
                reason="Empty original",
                confidence="low"
            )

        ratio = len(processed) / len(original)

        if ratio < self._min_ratio:
            return ValidationResult(
                key=file_key,
                is_valid=False,
                reason=f"Output too short: {ratio:.1%} of original",
                confidence="high"
            )

        if ratio > self._max_ratio:
            return ValidationResult(
                key=file_key,
                is_valid=False,
                reason=f"Output too long: {ratio:.1%} of original (possible hallucination)",
                confidence="medium"
            )

        return ValidationResult(
            key=file_key,
            is_valid=True,
            reason=f"Length ratio OK: {ratio:.1%}",
            confidence="high"
        )


def create_individual_validators(task_type: str = "translate") -> list:
    """
    Create default Individual validators.

    Individual validators run immediately after each file is processed.
    They use single-file interface: validate(original, processed, key).

    Args:
        task_type: "translate" or "polish"

    Returns:
        List of Individual validators
    """
    validators = []

    # Length check (works for all tasks)
    validators.append(LengthValidator(min_ratio=0.3, max_ratio=3.0))

    # N-gram detection (only for same-language tasks like polish)
    # For translate tasks, n-grams don't work cross-language
    if task_type != "translate":
        ngram_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=0.6,
            allow_deduplication=True
        )
        validators.append(TruncationValidatorAdapter(ngram_detector))

    return validators


# Batch validators (like AgentVerifier) are NOT adapted here.
# They have their own interface: validate_batch(files: Dict[str, VerificationFile])
# and should be used directly, not through adapters.
