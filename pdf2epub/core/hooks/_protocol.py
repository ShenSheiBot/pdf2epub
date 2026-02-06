"""
Hook protocols for the new architecture.

Hooks handle all edge cases outside the main processing flow:
- PreProcessors: Decide whether to process (image-only, empty content)
- Transformers: Modify results (restore images, remove artifacts)
- Validators: Judge results (length check, n-gram detection)
- SkipValidators: Skip validation for certain content types
- ErrorClassifier: Classify errors and determine effects on unit state
"""

from typing import Protocol, Dict, Any, Optional, Tuple, List, Set, Literal, TYPE_CHECKING, runtime_checkable
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .._protocol import ProcessContext


# ============================================================
# Error Types - Re-export from core.types (Single Source of Truth)
# ============================================================

from ..types import ErrorType


@dataclass
class ErrorEffect:
    """
    How an error affects unit state.

    Attributes:
        remove_current_model: Remove the current model from chain
        remove_provider: Remove all entries from the same provider (safety block)
        remove_all_batch: Remove all batch entries (when few failures)
        quota_type: Which quota to decrement
    """
    remove_current_model: bool = False
    remove_provider: bool = False
    remove_all_batch: bool = False
    quota_type: ErrorType = ErrorType.UNKNOWN


# ============================================================
# Pre-processing
# ============================================================

@dataclass
class PreProcessResult:
    """Result of pre-processing check."""
    should_process: bool
    skip_reason: str = ""
    fallback_result: Optional[str] = None


@runtime_checkable
class PreProcessor(Protocol):
    """
    Pre-processor - decides whether to process content.

    Examples:
    - ImageOnlyFilter: Skip image-only pages
    - EmptyContentFilter: Skip empty content
    """

    @property
    def name(self) -> str:
        """Pre-processor name for logging."""
        ...

    def check(
        self,
        key: str,
        content: str,
        context: "ProcessContext"
    ) -> PreProcessResult:
        """
        Check if content should be processed.

        Args:
            key: Unit identifier
            content: Content to check
            context: Processing context

        Returns:
            PreProcessResult with should_process flag
        """
        ...


# ============================================================
# Post-processing: Transform + Validate
# ============================================================

@dataclass
class HookResult:
    """Result of validation hook."""
    accepted: bool
    context_ready: bool = False
    rejection_reason: Optional[str] = None  # Detailed reason when rejected


@runtime_checkable
class Transformer(Protocol):
    """
    Transformer - modifies processing results.

    Examples:
    - RestoreImagesTransformer: Restore images removed by LLM
    - RemoveArtifactsTransformer: Remove LLM artifacts
    """

    @property
    def name(self) -> str:
        """Transformer name for logging."""
        ...

    def transform(
        self,
        key: str,
        original: str,
        result: str
    ) -> str:
        """
        Transform the result.

        Args:
            key: Unit identifier
            original: Original content
            result: Processing result to transform

        Returns:
            Transformed result
        """
        ...


@runtime_checkable
class Validator(Protocol):
    """
    Validator - judges processing results.

    Examples:
    - LengthValidator: Check length ratio
    - NGramValidator: Check content integrity
    """

    @property
    def name(self) -> str:
        """Validator name for logging."""
        ...

    def validate(
        self,
        key: str,
        original: str,
        result: str
    ) -> HookResult:
        """
        Validate the result.

        Args:
            key: Unit identifier
            original: Original content
            result: Processing result to validate

        Returns:
            HookResult with accepted flag
        """
        ...


# ============================================================
# Skip Validation
# ============================================================

@runtime_checkable
class SkipValidator(Protocol):
    """
    Skip validator - decides whether to skip validation.

    Examples:
    - ChapterTypeSkipper: Skip front_matter, back_matter, etc.
    """

    @property
    def name(self) -> str:
        """Skip validator name for logging."""
        ...

    def should_skip(
        self,
        key: str,
        chapter_type: str,
        context: "ProcessContext"
    ) -> bool:
        """
        Check if validation should be skipped.

        Args:
            key: Unit identifier
            chapter_type: Type of chapter
            context: Processing context

        Returns:
            True if validation should be skipped
        """
        ...


# ============================================================
# Error Classification
# ============================================================

@runtime_checkable
class ErrorClassifier(Protocol):
    """
    Error classifier - categorizes errors and determines effects.
    """

    def classify(self, error: Exception) -> ErrorType:
        """
        Classify the error type.

        Args:
            error: The exception to classify

        Returns:
            ErrorType enum value
        """
        ...

    def get_effect(self, error_type: ErrorType) -> ErrorEffect:
        """
        Get the effect of an error type on unit state.

        Args:
            error_type: The error type

        Returns:
            ErrorEffect describing how to modify unit state
        """
        ...


# ============================================================
# Batch Validation
# ============================================================

@dataclass
class ValidationRecord:
    """Record of a validation judgment."""
    timestamp: float
    validator_name: str
    file_key: str
    is_valid: bool
    reason: str
    confidence: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "validator_name": self.validator_name,
            "file_key": self.file_key,
            "is_valid": self.is_valid,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class BatchValidationResult:
    """Result of batch validation."""
    passed: Set[str] = field(default_factory=set)
    failed: Set[str] = field(default_factory=set)
    records: List[ValidationRecord] = field(default_factory=list)


class BatchValidatorHook(Protocol):
    """
    Batch validator hook - validates multiple files at once.

    This is between a hook and a pipeline component.
    """

    @property
    def name(self) -> str:
        """Validator name for logging."""
        ...

    def validate_batch(
        self,
        files: Dict[str, Any],  # Dict[str, VerificationFile]
        skip_keys: Optional[Set[str]] = None
    ) -> BatchValidationResult:
        """
        Validate multiple files.

        Args:
            files: Dict mapping key to VerificationFile
            skip_keys: Keys to skip validation

        Returns:
            BatchValidationResult with passed/failed sets
        """
        ...
