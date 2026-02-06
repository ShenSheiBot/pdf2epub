"""
Core infrastructure for pdf2epub processing pipeline.

This module contains the V2 architecture:
- WorkUnit: Complete work unit with all fields
- ProcessContext: Immutable context for processors
- ProcessingPipelineV2: Processing orchestrator
- Executor: LLM execution with state management
- Hooks: Validation and transformation
- Phase: Composable processing stages

Design Principles:
1. Composition over inheritance - components are injected, not overridden
2. Single responsibility - each class does one thing
3. Fail fast - violations crash at import time
4. Single source of truth - file system is the state
"""

# === Core Types (Single Source of Truth) ===
from .types import (
    SplitType,
    ErrorType,
    WorkUnit,
)

# === Work Unit and Discovery ===
from .work_unit import (
    WorkUnitDiscovery,
    ProcessingResult,
    # Unit type constants
    UNIT_TYPE_CONTENT,
    UNIT_TYPE_TOC,
    UNIT_TYPE_METADATA,
    # Reserved IDs
    TOC_UNIT_ID,
    METADATA_UNIT_ID,
    # Factory functions
    create_toc_unit,
    create_metadata_unit,
)

# === Protocol ===
from ._protocol import (
    ProcessContext,
    ProcessorProtocol,
    # Validator protocols
    IndividualValidator,
    BatchValidator,
    # Validation data classes
    ValidationResult,
    ValidatorConfig,
    ValidationRecord,
)

# === Persistence ===
from .persistence import ResultPersistence

# === Book Structure ===
from .book_structure import BookStructure, ChapterInfo

# === Context Injection ===
from .context import ContextInjector, sort_by_dependencies, group_by_file_key

# === Registry ===
from .registry import ComponentRegistry, register_processor, register_validator

# === Tracking Subsystem ===
from .tracking import (
    ProcessingTracker,
    AttemptRecord,
    SplitRecord,
    ValidationStrategy,
    AttemptResult,
)

# === Validators Subsystem ===
from .validators import (
    # Truncation detectors
    BaseTruncationDetector,
    NGramTruncationDetector,
    LLMTruncationDetector,
    CompositeTruncationDetector,
    # Batch validators (implement BatchValidator protocol)
    PolishBatchValidator,
    TranslationBatchValidator,
    # Internal agent classes
    AgentVerifier,
    PolishVerificationAgent,
    TranslationVerificationAgent,
    verify_batch,
    VerificationFile,
    VerificationTools,
    # Individual validators (implement IndividualValidator protocol)
    TruncationValidatorAdapter,
    LengthValidator,
    create_individual_validators,
)

# === Utils (from processors.utils) ===
from ..processors.utils import (
    restore_lost_images_fast,
    extract_images_from_markdown,
    extract_images,
    find_best_insertion_point,
    split_content,
    fuzzy_find_sentence,
)
from ..processors.utils.response_cleaner import clean_markdown_response

# === Hooks ===
from .hooks import (
    # Error types and effects
    ErrorType as HooksErrorType,
    ErrorEffect,
    # Results
    PreProcessResult,
    HookResult,
    # Pre-processors
    ImageOnlyFilter,
    EmptyContentFilter,
    MinLengthFilter,
    # Transformers
    RestoreImagesTransformer,
    RemoveArtifactsTransformer,
    NormalizeWhitespaceTransformer,
    StripTransformer,
    # Validators
    IndividualValidatorAdapter,
    LengthRatioValidator,
    NonEmptyValidator,
    TruncationValidator,
    CompositeTruncationValidator,
    # Skip validators
    ChapterTypeSkipper,
    ShortContentSkipper,
    KeyPatternSkipper,
    # Error classifiers
    DefaultErrorClassifier,
    StrictErrorClassifier,
    # Composite
    CompositeHooks,
)

# === Executor ===
from .executor import (
    # Types
    ChainEntry,
    ExecutionResult,
    ProcessResult,
    ExecutorProtocol,
    # State
    UnitState,
    QuotaConfig,
    create_unit_state,
    remove_batch_entries,
    remove_provider,
    chain_from_model_configs,
    # Executor
    Executor,
    handle_split,
)

# === Phase ===
from .phase import (
    Phase,
    PhaseResult,
    PartBasedLoader,
    ChapterBasedLoader,
    HTMLLoader,
)

# === Pipeline V2 ===
from .pipeline_v2 import ProcessingPipelineV2, ProcessingResultV2
from .factory_v2 import (
    # Config extractors
    get_validation_config,
    get_retry_config,
    get_truncation_config,
    get_hooks_config,
    get_quota_config,
    # Config-based factories (recommended)
    create_hooks_from_config,
    create_quota_config_from_config,
    create_model_chain_from_config,
    # Legacy factories (for backwards compatibility)
    create_default_hooks,
    create_default_model_chain,
    create_default_quota_config,
    create_processing_pipeline_v2,
)

__all__ = [
    # Core Types
    'SplitType',
    'ErrorType',
    'WorkUnit',

    # Work Unit
    'WorkUnitDiscovery',
    'ProcessingResult',
    'UNIT_TYPE_CONTENT',
    'UNIT_TYPE_TOC',
    'UNIT_TYPE_METADATA',
    'TOC_UNIT_ID',
    'METADATA_UNIT_ID',
    'create_toc_unit',
    'create_metadata_unit',

    # Context
    'ProcessContext',

    # Protocols
    'ProcessorProtocol',
    'IndividualValidator',
    'BatchValidator',
    'ValidationResult',
    'ValidatorConfig',
    'ValidationRecord',

    # Persistence
    'ResultPersistence',

    # Book Structure
    'BookStructure',
    'ChapterInfo',

    # Context Injection
    'ContextInjector',
    'sort_by_dependencies',
    'group_by_file_key',

    # Registry
    'ComponentRegistry',
    'register_processor',
    'register_validator',

    # Tracking
    'ProcessingTracker',
    'AttemptRecord',
    'SplitRecord',
    'ValidationStrategy',
    'AttemptResult',

    # Validators
    'BaseTruncationDetector',
    'NGramTruncationDetector',
    'LLMTruncationDetector',
    'CompositeTruncationDetector',
    'PolishBatchValidator',
    'TranslationBatchValidator',
    'AgentVerifier',
    'PolishVerificationAgent',
    'TranslationVerificationAgent',
    'verify_batch',
    'VerificationFile',
    'VerificationTools',
    'TruncationValidatorAdapter',
    'LengthValidator',
    'create_individual_validators',

    # Utils
    'restore_lost_images_fast',
    'extract_images_from_markdown',
    'extract_images',
    'find_best_insertion_point',
    'clean_markdown_response',
    'split_content',
    'fuzzy_find_sentence',

    # Hooks
    'HooksErrorType',
    'ErrorEffect',
    'PreProcessResult',
    'HookResult',
    'ImageOnlyFilter',
    'EmptyContentFilter',
    'MinLengthFilter',
    'RestoreImagesTransformer',
    'RemoveArtifactsTransformer',
    'NormalizeWhitespaceTransformer',
    'StripTransformer',
    'IndividualValidatorAdapter',
    'LengthRatioValidator',
    'NonEmptyValidator',
    'TruncationValidator',
    'CompositeTruncationValidator',
    'ChapterTypeSkipper',
    'ShortContentSkipper',
    'KeyPatternSkipper',
    'DefaultErrorClassifier',
    'StrictErrorClassifier',
    'CompositeHooks',

    # Executor
    'ChainEntry',
    'ExecutionResult',
    'ProcessResult',
    'ExecutorProtocol',
    'UnitState',
    'QuotaConfig',
    'create_unit_state',
    'remove_batch_entries',
    'remove_provider',
    'chain_from_model_configs',
    'Executor',
    'handle_split',

    # Phase
    'Phase',
    'PhaseResult',
    'PartBasedLoader',
    'ChapterBasedLoader',
    'HTMLLoader',

    # Pipeline V2
    'ProcessingPipelineV2',
    'ProcessingResultV2',
    'get_validation_config',
    'get_retry_config',
    'get_truncation_config',
    'get_hooks_config',
    'get_quota_config',
    'create_hooks_from_config',
    'create_quota_config_from_config',
    'create_model_chain_from_config',
    'create_default_hooks',
    'create_default_model_chain',
    'create_default_quota_config',
    'create_processing_pipeline_v2',
]
