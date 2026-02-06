"""
Factory functions for creating components with new architecture.

These functions create properly configured pipelines with the new
Executor + Hooks architecture, reading configuration from config dict.
"""

from typing import Optional, Any, List, Dict
from pathlib import Path

from ._protocol import ProcessorProtocol
from .executor import (
    ChainEntry, QuotaConfig,
    chain_from_model_configs,
)
from .hooks import (
    CompositeHooks,
    # Pre-processors
    EmptyContentFilter, ImageOnlyFilter,
    # Transformers
    RestoreImagesTransformer, RemoveArtifactsTransformer, StripTransformer,
    # Validators
    LengthRatioValidator, TruncationValidator, CompositeTruncationValidator,
    # Skip validators
    ChapterTypeSkipper, ShortContentSkipper,
    # Error classifiers
    DefaultErrorClassifier, StrictErrorClassifier,
    ErrorType,
)
from .pipeline_v2 import ProcessingPipelineV2
from .persistence import ResultPersistence
from .tracking import ProcessingTracker
from .context import ContextInjector
from .book_structure import BookStructure
from ..processors.utils.split_manager import SplitManager
from ..processors.utils.splitter_strategies import SimpleSplitter
from .validators import (
    TranslationBatchValidator,
    PolishBatchValidator,
)


# ============================================================
# Configuration Extraction
# ============================================================

def get_validation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract validation configuration from config dict.

    Reads from:
    - validation_strategy.*
    - translation.truncation_check_lines (for translate)
    - polish.truncation_check_lines (for polish)
    """
    validation = config.get('validation_strategy', {})
    return {
        'max_attempts': validation.get('max_attempts', 2),
        'use_longest_on_failure': validation.get('use_longest_on_failure', False),
        'fallback_between_models': validation.get('fallback_between_models', True),
    }


def get_retry_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract retry configuration from config dict.

    Reads from:
    - retry.*
    - splitting.*
    """
    retry = config.get('retry', {})
    splitting = config.get('splitting', {})
    return {
        'num_retries': retry.get('num_retries', 3),
        'max_backoff_seconds': retry.get('max_backoff_seconds', 30),
        'max_resplits': splitting.get('max_resplits', 3),
        'consecutive_failures_threshold': splitting.get('consecutive_failures_threshold', 2),
        'max_retries_before_resplit': splitting.get('max_retries_before_resplit', 2),
    }


def get_truncation_config(config: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """
    Extract truncation detection configuration.

    Reads from:
    - validation_v2.truncation.* (new V2 config)
    - translation.truncation_check_lines (fallback)
    - polish.truncation_check_lines (fallback)
    """
    v2_config = config.get('validation_v2', {}).get('truncation', {})
    task_config = config.get(task_type, {}) if task_type in ('translation', 'polish') else {}

    return {
        'min_unique_preserved_ratio': v2_config.get('min_unique_preserved_ratio', 0.60),
        'allow_deduplication': v2_config.get('allow_deduplication', True),
        'truncation_check_lines': (
            v2_config.get('truncation_check_lines') or
            task_config.get('truncation_check_lines', 5)
        ),
        'enable_llm_fallback': v2_config.get('enable_llm_fallback', True),
    }


def get_hooks_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hooks configuration.

    Reads from validation_v2.hooks.*
    """
    hooks = config.get('validation_v2', {}).get('hooks', {})
    return {
        'length_ratio': {
            'min_ratio': hooks.get('length_ratio', {}).get('min_ratio', 0.3),
            'max_ratio': hooks.get('length_ratio', {}).get('max_ratio', 3.0),
        },
        'skip_chapter_types': hooks.get('skip_chapter_types', [
            'front_matter', 'back_matter', 'toc', 'notes', 'appendix'
        ]),
        'skip_short_content_chars': hooks.get('skip_short_content_chars', 50),
    }


def get_quota_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract quota configuration.

    Reads from validation_v2.quotas.*
    """
    quotas = config.get('validation_v2', {}).get('quotas', {})
    return {
        'total': quotas.get('total', 5),
        'network': quotas.get('network', 3),
        'validation': quotas.get('validation', 1),
        'truncation': quotas.get('truncation', 2),
        'rate_limit': quotas.get('rate_limit', 3),
        'timeout': quotas.get('timeout', 3),
        'content_filter': quotas.get('content_filter', 1),
        'parse_error': quotas.get('parse_error', 2),
        'unknown': quotas.get('unknown', 2),
    }


def get_executor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract executor configuration.

    Reads from executor.* and batch.*
    """
    executor = config.get('executor', {})
    batch = config.get('batch', {})
    return {
        'batch_poll_interval': batch.get('poll_interval', executor.get('batch_poll_interval', 60)),
        # Backward compatibility: batch.online_polish_fallback_threshold -> executor.online_fallback_threshold
        'online_fallback_threshold': (
            executor.get('online_fallback_threshold') or
            batch.get('online_polish_fallback_threshold', 5)
        ),
    }


def get_split_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract splitting configuration.

    Reads from splitting.* and model_output_limits.*
    """
    splitting = config.get('splitting', {})
    model_output_limits = config.get('model_output_limits', {})
    return {
        'default_max_tokens': splitting.get('default_max_tokens', 4000),
        'max_resplits': splitting.get('max_resplits', 3),
        'consecutive_failures_threshold': splitting.get('consecutive_failures_threshold', 2),
        # Per-model token limits for dynamic split threshold
        'model_output_limits': model_output_limits,
    }


# ============================================================
# Factory Functions
# ============================================================

def create_hooks_from_config(
    config: Dict[str, Any],
    task_type: str = "translate",
    book_structure: Optional[BookStructure] = None,
    tracker: Optional[ProcessingTracker] = None,
    llm_client: Optional[Any] = None,
    restore_images: bool = True,
) -> CompositeHooks:
    """
    Create hooks from configuration.

    Args:
        config: Full config dict (from load_config)
        task_type: "translate" or "polish"
        book_structure: Optional book structure for filtering
        tracker: Optional tracker for recording
        llm_client: Optional LLM client for composite truncation detection
        restore_images: Whether to restore images

    Returns:
        Configured CompositeHooks
    """
    hooks_config = get_hooks_config(config)
    truncation_config = get_truncation_config(config, task_type)

    # Pre-processors
    pre_processors = [
        EmptyContentFilter(),
    ]
    if book_structure:
        pre_processors.append(ImageOnlyFilter(book_structure))

    # Transformers
    transformers = [
        RemoveArtifactsTransformer(),
        StripTransformer(),
    ]
    if restore_images:
        transformers.insert(0, RestoreImagesTransformer())

    # Validators
    validators = []

    # Length ratio validator (configurable thresholds)
    validators.append(LengthRatioValidator(
        min_ratio=hooks_config['length_ratio']['min_ratio'],
        max_ratio=hooks_config['length_ratio']['max_ratio'],
        role="screener",
        context_ready=True,
    ))

    # Truncation validator - only for same-language tasks (polish)
    # For translation, length ratios vary too much
    if task_type != "translate":
        if llm_client and truncation_config['enable_llm_fallback']:
            # Use composite (N-gram + LLM fallback)
            validators.append(CompositeTruncationValidator(
                llm_client=llm_client,
                min_unique_preserved_ratio=truncation_config['min_unique_preserved_ratio'],
                allow_deduplication=truncation_config['allow_deduplication'],
                truncation_check_lines=truncation_config['truncation_check_lines'],
                task_type=task_type,
                role="screener",
                context_ready=True,
            ))
        else:
            # Use N-gram only
            validators.append(TruncationValidator(
                min_unique_preserved_ratio=truncation_config['min_unique_preserved_ratio'],
                allow_deduplication=truncation_config['allow_deduplication'],
                role="screener",
                context_ready=True,
            ))

    # Skip validators (configurable chapter types)
    skip_validators = [
        ChapterTypeSkipper(skip_types=set(hooks_config['skip_chapter_types'])),
    ]

    # Add short content skipper if configured
    if hooks_config['skip_short_content_chars'] > 0:
        skip_validators.append(
            ShortContentSkipper(min_chars=hooks_config['skip_short_content_chars'])
        )

    # Error classifier - use strict for translate (removes model on validation fail)
    error_classifier = (
        StrictErrorClassifier() if task_type == "translate"
        else DefaultErrorClassifier()
    )

    return CompositeHooks(
        pre_processors=pre_processors,
        transformers=transformers,
        validators=validators,
        skip_validators=skip_validators,
        error_classifier=error_classifier,
        tracker=tracker,
    )


def create_quota_config_from_config(config: Dict[str, Any]) -> QuotaConfig:
    """
    Create QuotaConfig from configuration.

    Args:
        config: Full config dict

    Returns:
        QuotaConfig with configured quotas
    """
    quota_cfg = get_quota_config(config)

    return QuotaConfig(
        total=quota_cfg['total'],
        per_type={
            ErrorType.SAFETY: 999,  # Always unlimited (semantically correct)
            ErrorType.NETWORK: quota_cfg['network'],
            ErrorType.VALIDATION: quota_cfg['validation'],
            ErrorType.TRUNCATION: quota_cfg['truncation'],
            ErrorType.RATE_LIMIT: quota_cfg['rate_limit'],
            ErrorType.TIMEOUT: quota_cfg['timeout'],
            ErrorType.CONTENT_FILTER: quota_cfg['content_filter'],
            ErrorType.PARSE_ERROR: quota_cfg['parse_error'],
            ErrorType.UNKNOWN: quota_cfg['unknown'],
        },
    )


def create_model_chain_from_config(
    config: Dict[str, Any],
    task_type: str = "translate",
    include_batch: bool = False,
) -> List[ChainEntry]:
    """
    Create model chain from configuration.

    Reads models from translation.models or polish.models.

    Args:
        config: Full config dict
        task_type: "translate" or "polish"
        include_batch: Whether to include batch entries

    Returns:
        List of ChainEntry
    """
    # Get task-specific models
    if task_type == "translate":
        models = config.get('translation', {}).get('models', [])
    else:
        models = config.get('polish', {}).get('models', [])

    # Fallback to default
    if not models:
        models = [{'provider': 'gemini', 'model': 'gemini-2.0-flash'}]

    chain = []
    for model_config in models:
        provider = model_config.get('provider', 'gemini')
        model = model_config.get('model', 'gemini-2.0-flash')

        if include_batch:
            chain.append(ChainEntry(
                provider=provider,
                model=model,
                mode="batch",
            ))

        chain.append(ChainEntry(
            provider=provider,
            model=model,
            mode="online",
        ))

    return chain


# ============================================================
# Legacy Factory Functions (for backwards compatibility)
# ============================================================

def create_default_hooks(
    task_type: str = "translate",
    book_structure: Optional[BookStructure] = None,
    tracker: Optional[ProcessingTracker] = None,
    restore_images: bool = True,
    # Configurable parameters (legacy interface)
    length_min_ratio: float = 0.3,
    length_max_ratio: float = 3.0,
    truncation_min_ratio: float = 0.60,
    truncation_allow_dedup: bool = True,
) -> CompositeHooks:
    """
    Create default hooks configuration (legacy interface).

    For new code, prefer create_hooks_from_config().
    """
    # Pre-processors
    pre_processors = [
        EmptyContentFilter(),
    ]
    if book_structure:
        pre_processors.append(ImageOnlyFilter(book_structure))

    # Transformers
    transformers = [
        RemoveArtifactsTransformer(),
        StripTransformer(),
    ]
    if restore_images:
        transformers.insert(0, RestoreImagesTransformer())

    # Validators
    validators = [
        LengthRatioValidator(
            min_ratio=length_min_ratio,
            max_ratio=length_max_ratio,
            role="screener",
            context_ready=True,
        ),
    ]

    # Add truncation for polish
    if task_type != "translate":
        validators.append(TruncationValidator(
            min_unique_preserved_ratio=truncation_min_ratio,
            allow_deduplication=truncation_allow_dedup,
            role="screener",
            context_ready=True,
        ))

    # Skip validators
    skip_validators = [
        ChapterTypeSkipper(),
    ]

    return CompositeHooks(
        pre_processors=pre_processors,
        transformers=transformers,
        validators=validators,
        skip_validators=skip_validators,
        error_classifier=DefaultErrorClassifier(),
        tracker=tracker,
    )


def create_default_model_chain(
    processor: ProcessorProtocol,
    include_batch: bool = False,
) -> List[ChainEntry]:
    """
    Create default model chain from processor config (legacy interface).

    For new code, prefer create_model_chain_from_config().
    """
    model_configs = processor.get_model_configs()
    chain = []

    for config in model_configs:
        provider = config.get("provider", "gemini")
        model = config.get("model", "gemini-2.0-flash")

        if include_batch:
            chain.append(ChainEntry(
                provider=provider,
                model=model,
                mode="batch",
            ))

        chain.append(ChainEntry(
            provider=provider,
            model=model,
            mode="online",
        ))

    return chain


def create_default_quota_config(
    total: int = 5,
    network: int = 3,
    validation: int = 1,
    truncation: int = 2,
) -> QuotaConfig:
    """
    Create default quota configuration (legacy interface).

    For new code, prefer create_quota_config_from_config().
    """
    return QuotaConfig(
        total=total,
        per_type={
            ErrorType.SAFETY: 999,
            ErrorType.NETWORK: network,
            ErrorType.VALIDATION: validation,
            ErrorType.TRUNCATION: truncation,
            ErrorType.RATE_LIMIT: network,
            ErrorType.TIMEOUT: network,
            ErrorType.CONTENT_FILTER: 1,
            ErrorType.PARSE_ERROR: 2,
            ErrorType.UNKNOWN: 2,
        },
    )


# ============================================================
# Main Factory Function
# ============================================================

def create_processing_pipeline_v2(
    processor: ProcessorProtocol,
    output_dir: Path,
    llm_client: Any,
    config: Optional[Dict[str, Any]] = None,
    # Optional components
    book_structure: Optional[BookStructure] = None,
    batch_client: Optional[Any] = None,
    # Configuration overrides (used if config not provided)
    task_type: str = "translate",
    use_batch_validation: bool = True,
    sequential_mode: bool = True,
    max_workers: Optional[int] = None,
    restore_images: bool = True,
    include_batch_mode: bool = False,
    batch_retry_threshold: int = 5,
) -> ProcessingPipelineV2:
    """
    Create a fully configured ProcessingPipelineV2.

    If config dict is provided, reads configuration from it.
    Otherwise uses default values and explicit parameters.

    Args:
        processor: Processor implementing ProcessorProtocol
        output_dir: Directory for output files
        llm_client: LLM client for API calls
        config: Full config dict (from load_config) - recommended
        book_structure: Optional book structure
        batch_client: Optional batch client (enables batch mode)
        task_type: "translate" or "polish"
        use_batch_validation: Enable batch validation
        sequential_mode: Enable context injection
        max_workers: Maximum concurrent workers (default: from config)
        restore_images: Restore images removed by LLM
        include_batch_mode: Include batch entries in chain
        batch_retry_threshold: Use online for <= this many failures

    Returns:
        Configured ProcessingPipelineV2
    """
    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get max_workers from config if not specified
    if max_workers is None:
        max_workers = config.get('general', {}).get('max_concurrent_workers', 4)

    # Create persistence (uses default raw/validated subdirs)
    persistence = ResultPersistence(output_dir=output_dir)

    # Create tracker with file checker for atomicity guarantee
    # This ensures is_unit_complete() returns False if file is missing
    tracker_path = output_dir / "processing_tracker.json"
    processor_name = getattr(processor, "name", processor.__class__.__name__)
    if not isinstance(processor_name, str) or not processor_name:
        processor_name = processor.__class__.__name__

    def file_checker(key: str) -> bool:
        """Check if output file exists (raw or validated)."""
        return persistence.has_raw(key) or persistence.has_validated(key)

    tracker = ProcessingTracker(tracker_path, processor_name, file_checker=file_checker)

    # Create hooks (from config if available)
    if config:
        hooks = create_hooks_from_config(
            config=config,
            task_type=task_type,
            book_structure=book_structure,
            tracker=tracker,
            llm_client=llm_client,
            restore_images=restore_images,
        )
    else:
        hooks = create_default_hooks(
            task_type=task_type,
            book_structure=book_structure,
            tracker=tracker,
            restore_images=restore_images,
        )

    # Create model chain (from config if available)
    if config:
        model_chain = create_model_chain_from_config(
            config=config,
            task_type=task_type,
            include_batch=include_batch_mode or batch_client is not None,
        )
    else:
        model_chain = create_default_model_chain(
            processor=processor,
            include_batch=include_batch_mode or batch_client is not None,
        )

    # Create quota config (from config if available)
    if config:
        quota_config = create_quota_config_from_config(config)
    else:
        quota_config = create_default_quota_config()

    # Create context injector
    context_injector = ContextInjector(
        mode="sequential" if sequential_mode else "parallel",
        persistence=persistence,
    )

    # Create split manager with config
    split_cfg = get_split_config(config)

    # For proactive split, use the max of model_output_limits if available
    # This prevents re-splitting files that were already split at a higher threshold
    model_limits = split_cfg.get('model_output_limits', {})
    proactive_split_threshold = split_cfg['default_max_tokens']
    if model_limits:
        max_model_limit = max(model_limits.values())
        proactive_split_threshold = max(proactive_split_threshold, max_model_limit)

    split_manager = SplitManager(
        tracker=tracker,
        output_dir=output_dir / "splits",
        default_max_tokens=proactive_split_threshold,  # Use max of model limits for proactive split
        max_resplits=split_cfg['max_resplits'],
        consecutive_failures_threshold=split_cfg['consecutive_failures_threshold'],
    )

    # Create content splitter for dynamic splitting in Executor
    content_splitter = SimpleSplitter()

    # Create batch validators
    batch_validators = []
    if use_batch_validation:
        if task_type == "translate":
            batch_validators.append(TranslationBatchValidator())
        else:
            batch_validators.append(PolishBatchValidator())

    # Get executor config
    executor_cfg = get_executor_config(config)

    # Create pipeline
    # Note: longest_fallback is handled by Executor, not Pipeline (per design v2)
    return ProcessingPipelineV2(
        processor=processor,
        llm_client=llm_client,
        persistence=persistence,
        tracker=tracker,
        hooks=hooks,
        batch_client=batch_client,
        batch_validators=batch_validators,
        split_manager=split_manager,
        content_splitter=content_splitter,
        context_injector=context_injector,
        book_structure=book_structure,
        model_chain=model_chain,
        quota_config=quota_config,
        max_workers=max_workers,
        batch_retry_threshold=batch_retry_threshold,
        batch_poll_interval=executor_cfg['batch_poll_interval'],
        split_max_tokens=split_cfg['default_max_tokens'],
        model_output_limits=split_cfg['model_output_limits'],  # P0-1: per-model token limits
    )
