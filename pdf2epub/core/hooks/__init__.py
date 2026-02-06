"""
Hooks module - handles all edge cases outside the main processing flow.

Core design principle:
- Main flow (Pipeline + Executor) stays pure: content -> LLM -> result
- All edge cases (skip, transform, validate, error handling) go through hooks

Components:
- PreProcessors: Decide if content should be processed
- Transformers: Modify LLM output (restore images, remove artifacts)
- Validators: Judge if result is acceptable
- SkipValidators: Skip validation for certain content types
- ErrorClassifier: Classify errors and determine effects
- CompositeHooks: Combines all hooks into a single interface
"""

from ._protocol import (
    # Error types
    ErrorType,
    ErrorEffect,
    # Results
    PreProcessResult,
    HookResult,
    ValidationRecord,
    BatchValidationResult,
    # Protocols
    PreProcessor,
    Transformer,
    Validator,
    SkipValidator,
    ErrorClassifier,
    BatchValidatorHook,
)

from .pre_processors import (
    ImageOnlyFilter,
    EmptyContentFilter,
    MinLengthFilter,
)

from .transformers import (
    RestoreImagesTransformer,
    RemoveArtifactsTransformer,
    NormalizeWhitespaceTransformer,
    StripTransformer,
)

from .validators import (
    IndividualValidatorAdapter,
    LengthRatioValidator,
    NonEmptyValidator,
    TruncationValidator,
    CompositeTruncationValidator,
)

from .skip_validators import (
    ChapterTypeSkipper,
    ShortContentSkipper,
    KeyPatternSkipper,
)

from .error_classifier import (
    DefaultErrorClassifier,
    StrictErrorClassifier,
    # Batch error handling
    Attribution,
    BatchFailureAction,
    BatchCircuitBreaker,
    attribute_job_failure,
    extract_unit_key_from_error,
    get_batch_failure_action,
)

from .composite import CompositeHooks

__all__ = [
    # Error types
    'ErrorType',
    'ErrorEffect',
    # Results
    'PreProcessResult',
    'HookResult',
    'ValidationRecord',
    'BatchValidationResult',
    # Protocols
    'PreProcessor',
    'Transformer',
    'Validator',
    'SkipValidator',
    'ErrorClassifier',
    'BatchValidatorHook',
    # Pre-processors
    'ImageOnlyFilter',
    'EmptyContentFilter',
    'MinLengthFilter',
    # Transformers
    'RestoreImagesTransformer',
    'RemoveArtifactsTransformer',
    'NormalizeWhitespaceTransformer',
    'StripTransformer',
    # Validators
    'IndividualValidatorAdapter',
    'LengthRatioValidator',
    'NonEmptyValidator',
    'TruncationValidator',
    'CompositeTruncationValidator',
    # Skip validators
    'ChapterTypeSkipper',
    'ShortContentSkipper',
    'KeyPatternSkipper',
    # Error classifiers
    'DefaultErrorClassifier',
    'StrictErrorClassifier',
    # Batch error handling
    'Attribution',
    'BatchFailureAction',
    'BatchCircuitBreaker',
    'attribute_job_failure',
    'extract_unit_key_from_error',
    'get_batch_failure_action',
    # Composite
    'CompositeHooks',
]
