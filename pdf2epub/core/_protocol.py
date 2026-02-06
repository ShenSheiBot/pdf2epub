"""
Protocols defining what Processors and Validators must implement.

Processors can ONLY implement these methods. Any attempt to add
validation, saving, or state management methods will be caught
by architecture tests.

Design:
- ProcessorProtocol: build_prompt, clean_response, post_process
- ValidatorProtocol: validate, phase
- ProcessContext: immutable context passed to processors
"""

from typing import Protocol, Dict, Any, Optional, Tuple, Union, Literal, runtime_checkable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .validators import VerificationFile


@dataclass(frozen=True)
class ProcessContext:
    """
    Immutable context passed to processors.

    Contains ALL information needed to build prompts and process results,
    including context injection from previous parts.

    This is a COMPLETE context - no information should be missing.
    """
    # === Basic identification ===
    file_key: str
    book_title: str

    # === Part information ===
    part_index: Optional[int] = None  # None for single file, 1-based for parts
    total_parts: int = 1
    split_version: int = 0  # Current split version

    # === Language information ===
    source_language: str = "Japanese"
    target_language: str = "Chinese"

    # === Content type (auto-detected or specified) ===
    content_type: str = "general"  # "general", "japanese", "academic"

    # === Chapter information (from BookStructure) ===
    chapter_type: Optional[str] = None  # "chapter", "notes", "appendix", "front_matter", "back_matter"
    chapter_title: Optional[str] = None
    chapter_number: Optional[str] = None  # "5" or "7.1.1"
    is_notes_chapter: bool = False

    # === Context injection (from previous part) ===
    previous_original: Optional[str] = None
    previous_processed: Optional[str] = None

    # === Book structure info ===
    is_vertical_text: bool = False
    has_global_footnotes: bool = False
    book_language: Optional[str] = None

    # === TOC and page info ===
    toc_path: Optional[str] = None
    page_range: Optional[Tuple[int, int]] = None

    # === Unit type (for special units) ===
    unit_type: str = "content"  # "content", "toc", "metadata"

    # === Extension fields ===
    extra: Optional[Dict[str, Any]] = None

    @property
    def is_part(self) -> bool:
        """Whether this is a part of a multi-part file."""
        return self.part_index is not None

    @property
    def is_first_part(self) -> bool:
        """Whether this is the first part."""
        return self.part_index == 1

    @property
    def is_front_back_matter(self) -> bool:
        """Whether this is front/back matter (skip validation)."""
        return self.chapter_type in ("front_matter", "back_matter", "notes", "appendix")

    @property
    def has_previous_context(self) -> bool:
        """Whether context injection is available."""
        return self.previous_original is not None and self.previous_processed is not None

    def with_part(self, part_index: int, total_parts: int) -> "ProcessContext":
        """Create a new context with updated part info."""
        return ProcessContext(
            file_key=self.file_key,
            book_title=self.book_title,
            part_index=part_index,
            total_parts=total_parts,
            split_version=self.split_version,
            source_language=self.source_language,
            target_language=self.target_language,
            content_type=self.content_type,
            chapter_type=self.chapter_type,
            chapter_title=self.chapter_title,
            chapter_number=self.chapter_number,
            is_notes_chapter=self.is_notes_chapter,
            previous_original=self.previous_original,
            previous_processed=self.previous_processed,
            is_vertical_text=self.is_vertical_text,
            has_global_footnotes=self.has_global_footnotes,
            book_language=self.book_language,
            toc_path=self.toc_path,
            page_range=self.page_range,
            extra=self.extra
        )

    def with_previous_context(
        self,
        previous_original: str,
        previous_processed: str
    ) -> "ProcessContext":
        """Create a new context with injected previous part context."""
        return ProcessContext(
            file_key=self.file_key,
            book_title=self.book_title,
            part_index=self.part_index,
            total_parts=self.total_parts,
            split_version=self.split_version,
            source_language=self.source_language,
            target_language=self.target_language,
            content_type=self.content_type,
            chapter_type=self.chapter_type,
            chapter_title=self.chapter_title,
            chapter_number=self.chapter_number,
            is_notes_chapter=self.is_notes_chapter,
            previous_original=previous_original,
            previous_processed=previous_processed,
            is_vertical_text=self.is_vertical_text,
            has_global_footnotes=self.has_global_footnotes,
            book_language=self.book_language,
            toc_path=self.toc_path,
            page_range=self.page_range,
            extra=self.extra
        )

    @classmethod
    def from_work_unit(
        cls,
        unit: "WorkUnit",
        book_title: str = "",
        source_language: str = "Japanese",
        target_language: str = "Chinese",
        is_vertical_text: bool = False,
        has_global_footnotes: bool = False,
        book_language: Optional[str] = None,
    ) -> "ProcessContext":
        """
        Create a ProcessContext from a WorkUnit.

        This ensures all WorkUnit metadata is properly propagated to the context,
        making skip validators and processor prompts reliable.

        Args:
            unit: The WorkUnit to build context from
            book_title: Book title (from config or BookStructure)
            source_language: Source language for translation
            target_language: Target language for translation
            is_vertical_text: Whether the book uses vertical text
            has_global_footnotes: Whether the book has global footnotes
            book_language: Detected book language

        Returns:
            ProcessContext with all fields populated from WorkUnit
        """
        # Import here to avoid circular import
        from .work_unit import WorkUnit as WU

        return cls(
            file_key=unit.file_key,
            book_title=book_title,
            part_index=unit.part_index,
            total_parts=unit.total_parts,
            split_version=unit.split_version,
            source_language=source_language,
            target_language=target_language,
            chapter_type=unit.chapter_type,
            chapter_title=unit.chapter_title,
            chapter_number=unit.chapter_number,
            is_notes_chapter=(unit.chapter_type == "notes"),
            is_vertical_text=is_vertical_text,
            has_global_footnotes=has_global_footnotes,
            book_language=book_language,
            toc_path=unit.toc_path,
            page_range=unit.page_range,
            unit_type=unit.unit_type,
        )


@runtime_checkable
class ProcessorProtocol(Protocol):
    """
    Protocol that all processors must implement.

    IMPORTANT: Processors can ONLY implement these methods.
    They are NOT allowed to implement:
    - validate, _validate, validate_output
    - save, _save, _save_result, save_raw
    - _batch_validate_and_save, _batch_validate
    - Any state management methods

    Architecture tests will fail if forbidden methods are found.
    """

    @property
    def name(self) -> str:
        """
        Processor name for logging and tracking.

        Returns:
            Short identifier like "polish" or "translate"
        """
        ...

    def build_prompt(self, content: str, context: ProcessContext) -> Any:
        """
        Build the prompt to send to LLM.

        This is the ONLY place where processor-specific logic should live.
        All prompt construction, entity references, language-specific rules
        go here.

        IMPORTANT: If context.has_previous_context is True, you MUST use
        previous_original and previous_processed for terminology consistency.

        Args:
            content: The content to process
            context: Processing context (file info, language, previous context, etc.)

        Returns:
            Either:
            - str: Simple prompt
            - List[Dict]: Multi-turn conversation (for context injection)
        """
        ...

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
        ...

    def post_process(self, result: str, context: ProcessContext) -> str:
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
        ...

    def get_model_configs(self) -> list:
        """
        Get model configurations for LLM calls.

        IMPORTANT: Return MULTIPLE configs for fallback chain.
        If first model fails, next model will be tried.

        Returns:
            List of model config dicts with provider, model, retries, etc.
            Example:
            [
                {"provider": "gemini", "model": "gemini-2.5-flash", "max_retries": 3},
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250514", "max_retries": 2}
            ]
        """
        ...


# === Validation Architecture ===
# Two orthogonal dimensions:
# 1. Validator type: Individual vs Batch
# 2. Validator role: Screener vs Final (configured via ValidatorConfig)


@runtime_checkable
class IndividualValidator(Protocol):
    """
    Individual validator - runs immediately after each file is processed.

    Interface: validate(original, processed, key) -> ValidationResult

    Use cases:
    - Fast screening (length check, n-gram detection)
    - Cross-language validation (LLM-based truncation detection)

    Individual validators can be configured as screener or final via ValidatorConfig.
    """

    @property
    def name(self) -> str:
        """Validator name for logging and tracking."""
        ...

    def validate(
        self,
        original: str,
        processed: str,
        file_key: str
    ) -> "ValidationResult":
        """
        Validate a single file.

        Args:
            original: Original content before processing
            processed: Processed content to validate
            file_key: File identifier

        Returns:
            ValidationResult with is_valid, reason, confidence
        """
        ...


@runtime_checkable
class BatchValidator(Protocol):
    """
    Batch validator - runs after all files are processed.

    Interface: validate_batch(files) -> Dict[str, ValidationResult]

    Use cases:
    - Agent-based verification (can see multiple files for context)
    - Cross-file consistency checks

    Batch validators can be configured as screener or final via ValidatorConfig.
    """

    @property
    def name(self) -> str:
        """Validator name for logging and tracking."""
        ...

    def validate_batch(
        self,
        files: Dict[str, "VerificationFile"]
    ) -> Dict[str, "ValidationResult"]:
        """
        Validate multiple files in batch.

        Args:
            files: Dict mapping file_key to VerificationFile(original, processed)

        Returns:
            Dict mapping file_key to ValidationResult
        """
        ...


@dataclass
class ValidationResult:
    """Result of a validation check."""
    key: str
    is_valid: bool
    reason: str
    confidence: str = "high"  # high, medium, low

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class ValidatorConfig:
    """
    Configuration for a validator.

    Role is a configuration-level concept, not a validator property.
    The same validator can be configured as screener or final.

    Attributes:
        validator: The validator instance (IndividualValidator or BatchValidator)
        role: "screener" (OR logic) or "final" (AND logic)
            - Screener: pass = pass (short-circuit), fail = uncertain (continue)
            - Final: all must pass, any fail = fail (short-circuit)
        context_ready: If True, passing this validator means raw result can be
            used for context injection immediately (before all validation completes)
    """
    validator: Union["IndividualValidator", "BatchValidator"]
    role: Literal["screener", "final"]
    context_ready: bool = False

    @property
    def name(self) -> str:
        """Get validator name."""
        return self.validator.name

    @property
    def is_screener(self) -> bool:
        return self.role == "screener"

    @property
    def is_final(self) -> bool:
        return self.role == "final"


@dataclass
class ValidationRecord:
    """
    Record of a single validation judgment.

    All validator judgments must be recorded for observability.
    """
    timestamp: float
    validator_name: str
    role: Literal["screener", "final"]
    file_key: str
    result: ValidationResult
    context_ready_triggered: bool = False  # Whether context_ready was triggered

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "validator_name": self.validator_name,
            "role": self.role,
            "file_key": self.file_key,
            "is_valid": self.result.is_valid,
            "reason": self.result.reason,
            "confidence": self.result.confidence,
            "context_ready_triggered": self.context_ready_triggered,
        }
