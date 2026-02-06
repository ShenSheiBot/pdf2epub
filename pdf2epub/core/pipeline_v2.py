"""
ProcessingPipeline V2 - uses new Executor + Hooks architecture.

Key design:
- Pipeline orchestrates high-level flow
- Executor handles LLM calls + retries
- Hooks handle all edge cases
- No retry loop in Pipeline (delegated to Executor)
"""

from typing import List, Dict, Set, Optional, Any, TYPE_CHECKING
from pathlib import Path
from loguru import logger
import time

from .types import SplitType, is_sub_key, filter_sub_keys
from .executor import (
    WorkUnit, ExecutionResult, ChainEntry,
    Executor,
    QuotaConfig, chain_from_model_configs,
)
from .hooks import (
    CompositeHooks,
    BatchValidationResult,
)
from .persistence import ResultPersistence
from .tracking import ProcessingTracker

if TYPE_CHECKING:
    from ._protocol import ProcessContext, ProcessorProtocol, BatchValidator
    from .context import ContextInjector
    from .book_structure import BookStructure
    from ..processors.utils.split_manager import SplitManager
    from ..processors.utils.splitter_strategies import ContentSplitter


class ProcessingPipelineV2:
    """
    Processing pipeline using new architecture.

    Responsibilities:
    - Proactive split (SplitManager)
    - Resume tracking (ProcessingTracker)
    - Call Executor with units + hooks
    - Batch validation (after all units complete)
    - Longest fallback (ValidationStrategy)
    - Save results (Persistence)

    NOT responsible for:
    - Retry loops (Executor handles)
    - Per-unit state (Executor handles)
    - Pre/post processing (Hooks handle)
    - Error classification (Hooks handle)
    """

    def __init__(
        self,
        processor: "ProcessorProtocol",
        llm_client: Any,
        persistence: ResultPersistence,
        tracker: ProcessingTracker,
        hooks: CompositeHooks,
        # Executor (optional - will create default)
        executor: Optional[Executor] = None,
        batch_client: Optional[Any] = None,
        # Batch validation
        batch_validators: Optional[List["BatchValidator"]] = None,
        # Components
        split_manager: Optional["SplitManager"] = None,
        content_splitter: Optional["ContentSplitter"] = None,
        context_injector: Optional["ContextInjector"] = None,
        book_structure: Optional["BookStructure"] = None,
        # Configuration
        model_chain: Optional[List[ChainEntry]] = None,
        quota_config: Optional[QuotaConfig] = None,
        max_workers: int = 4,
        batch_retry_threshold: int = 5,
        batch_poll_interval: int = 60,
        split_max_tokens: int = 4000,
        model_output_limits: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            processor: Processor for building prompts
            llm_client: LLM client for API calls
            persistence: For saving results
            tracker: For resume tracking
            hooks: CompositeHooks for pre/post processing
            executor: Optional executor (will create if not provided)
            batch_client: Optional batch client (enables batch mode)
            batch_validators: Batch validators to run after all units complete
            split_manager: For proactive splitting
            content_splitter: For dynamic splitting in Executor
            context_injector: For sequential mode
            book_structure: For chapter info
            model_chain: Model chain (required if executor not provided)
            quota_config: Quota config (default: QuotaConfig())
            max_workers: Max concurrent workers
            batch_retry_threshold: Use online for <= this many failures
            batch_poll_interval: Seconds between batch status polls
            split_max_tokens: Default max tokens per split part for dynamic splitting
            model_output_limits: Per-model token limits (e.g., {"gemini-3-pro-preview": 8000})
        """
        self._processor = processor
        self._llm_client = llm_client
        self._persistence = persistence
        self._tracker = tracker
        self._hooks = hooks
        self._batch_validators = batch_validators or []
        self._split_manager = split_manager
        self._content_splitter = content_splitter
        self._context_injector = context_injector
        self._book_structure = book_structure
        self._batch_retry_threshold = batch_retry_threshold

        # Create executor if not provided
        if executor:
            self._executor = executor
        else:
            if not model_chain:
                # Build from processor config
                model_configs = processor.get_model_configs()
                model_chain = chain_from_model_configs(model_configs)

            quota_config = quota_config or QuotaConfig()

            # Unified executor handles both batch and online
            self._executor = Executor(
                llm_client=llm_client,
                model_chain=model_chain,
                processor=processor,
                hooks=hooks,
                batch_client=batch_client,  # None = online only
                quota_config=quota_config,
                max_workers=max_workers,
                context_injector=context_injector,
                tracker=tracker,
                splitter=content_splitter,  # For dynamic splitting on failure
                split_max_tokens=split_max_tokens,
                batch_poll_interval=batch_poll_interval,
                online_fallback_threshold=batch_retry_threshold,
                model_output_limits=model_output_limits,  # P0-1: per-model token limits
                persistence=persistence,  # Realtime save on completion
            )

    def process_all(
        self,
        units: List[WorkUnit],
        context_base: Optional["ProcessContext"] = None
    ) -> "ProcessingResultV2":
        """
        Process all units.

        Args:
            units: Units to process
            context_base: Base context for all units

        Returns:
            ProcessingResultV2 with statistics
        """
        if not units:
            return ProcessingResultV2(total=0, completed=0, failed=0)

        start_time = time.time()

        # Step 1: Proactive split
        if self._split_manager:
            units = self._proactive_split(units)

        all_keys = {u.id for u in units}
        originals = {u.id: u.content for u in units}

        # Step 2: Filter completed (resume)
        pending_keys = self._get_pending_keys(all_keys)
        pending_units = [u for u in units if u.id in pending_keys]

        if not pending_units:
            logger.info(f"All {len(units)} units already completed")
            return ProcessingResultV2(
                total=len(units),
                completed=len(units),
                failed=0,
            )

        logger.info(f"Processing {len(pending_units)}/{len(units)} units")

        # Step 3: Execute via Executor
        exec_result = self._executor.execute(pending_units, context_base)

        # Step 4: Save raw results
        for key, content in exec_result.results.items():
            self._persistence.save_raw(key, content)

        # Step 5: Filter out .sub virtual units from processing
        # .sub units are Executor's runtime splits - they should:
        # - Remain in raw/ (already saved above, good for debugging)
        # - NOT be promoted to validated/
        # - NOT be tracked as completed/failed units
        # - NOT inflate statistics
        # - NOT be batch-validated (only aggregate parent matters)
        all_result_keys = set(exec_result.results.keys())
        real_result_keys = filter_sub_keys(all_result_keys)
        sub_keys = all_result_keys - real_result_keys

        if sub_keys:
            logger.debug(f"Filtering {len(sub_keys)} .sub virtual units from processing")

        # Step 6: Batch validation (if configured)
        # Only validate real units, not .sub children
        # Pass screener_passed as skip_keys - these already passed individual screeners
        batch_failed: Set[str] = set()
        if self._batch_validators and real_result_keys:
            # Build filtered results dict for batch validation
            real_results = {k: exec_result.results[k] for k in real_result_keys}
            batch_failed = self._run_batch_validation(
                real_results, originals,
                screener_passed=exec_result.screener_passed
            )

        # Step 7: Determine final failures (only real units)
        # Note: longest-fallback is handled by Executor (per design v2)
        real_failed = filter_sub_keys(exec_result.failed)  # Filter .sub from executor failures
        all_failed = real_failed | batch_failed

        # Step 8: Promote successful to validated (only real units)
        # Separate fallback results from normal results for proper warning handling
        successful = real_result_keys - all_failed
        fallback_keys = exec_result.fallback_used & successful  # Only successful fallbacks
        normal_keys = successful - fallback_keys

        # Save fallback results with warning header (for audit/traceability)
        for key in fallback_keys:
            content = exec_result.results[key]
            self._persistence.save_with_warning(
                key, content,
                warning="LONGEST_FALLBACK: validation failed after max retries"
            )
            self._mark_complete(key, fallback=True)

        # Promote normal successful results
        if normal_keys:
            self._persistence.promote_batch(list(normal_keys))
            for key in normal_keys:
                self._mark_complete(key)

        # Step 9: Mark failures via attempt record (only real units)
        from .tracking import AttemptRecord
        for key in all_failed:
            attempt = AttemptRecord(
                timestamp=time.time(),
                status="failed",
                model="executor",
                error_type="exhausted",
                error_message="All retries exhausted",
            )
            self._tracker.record_attempt(key, attempt)

        # NOTE: Aggregation removed from process_all per architecture v2.
        # Aggregation breaks phase composability (polish -> translate -> polish).
        # Aggregation should ONLY happen at build-epub time.

        # Build result (statistics use real units only)
        duration = time.time() - start_time

        return ProcessingResultV2(
            total=len(units),
            completed=len(successful),
            failed=len(all_failed),
            skipped=len(exec_result.skipped),
            failed_keys=list(all_failed),
            results=exec_result.results,  # Full results (including .sub) for debugging
            duration=duration,
        )

    def _proactive_split(self, units: List[WorkUnit]) -> List[WorkUnit]:
        """
        Apply proactive splitting to large units.

        Uses .part{N} naming for persistent splits (per design v2).
        .sub{N} is reserved for Executor's runtime virtual splits.
        """
        if not self._split_manager:
            return units

        result = []
        for unit in units:
            # Get or create split via SplitManager
            split_result = self._split_manager.get_or_create_split(
                base_key=unit.id,
                content=unit.content,
            )

            # Check if actually split (more than 1 part)
            if len(split_result.parts) > 1:
                # Create part-units with .part{N} naming (1-indexed per convention)
                for i, part_content in enumerate(split_result.parts, start=1):
                    part_id = f"{unit.id}.part{i}"
                    result.append(WorkUnit(
                        id=part_id,
                        file_key=unit.file_key,
                        content=part_content,
                        input_path=unit.input_path,
                        part_index=i,
                        total_parts=len(split_result.parts),
                        parent_id=unit.id,
                        split_type=SplitType.PROACTIVE,
                        chapter_type=unit.chapter_type,
                    ))
                logger.info(f"Split {unit.id} into {len(split_result.parts)} parts (.part1-.part{len(split_result.parts)})")
            else:
                result.append(unit)

        return result

    def _get_pending_keys(self, all_keys: Set[str]) -> Set[str]:
        """Get keys that haven't been completed yet OR have missing files.

        Safety check: Even if tracker shows "completed", if the output file
        doesn't exist, we must reprocess. This handles edge cases like:
        - Interrupted between tracker update and file save
        - Files manually deleted
        - Disk corruption
        """
        pending = set()
        for key in all_keys:
            if not self._tracker.is_unit_complete(key):
                pending.add(key)
            elif not (self._persistence.has_raw(key) or self._persistence.has_validated(key)):
                # Safety: tracker says complete but file missing - must reprocess
                logger.warning(
                    f"{key}: Tracker shows complete but file missing, will reprocess"
                )
                pending.add(key)
        return pending

    def _mark_complete(self, key: str, fallback: bool = False):
        """
        Mark a key as complete in tracker via successful attempt record.

        Args:
            key: The file key
            fallback: If True, marks as completed via longest fallback (for audit)
        """
        from .tracking import AttemptRecord
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed_fallback" if fallback else "completed",
            model="executor",  # Generic marker
        )
        self._tracker.record_attempt(key, attempt)

    def _run_batch_validation(
        self,
        results: Dict[str, str],
        originals: Dict[str, str],
        screener_passed: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Run batch validation on results.

        Args:
            results: Dict of key -> processed content
            originals: Dict of key -> original content
            screener_passed: Keys that passed individual screeners (skip batch validation)

        Returns set of failed keys.
        """
        from .validators import VerificationFile

        failed: Set[str] = set()

        # Get skip keys based on chapter type
        skip_keys: Set[str] = set(screener_passed or set())
        if self._book_structure:
            for key in results:
                info = self._book_structure.get_chapter_info(key)
                if info and self._hooks.should_skip_validation(
                    key, info.chapter_type or "", None
                ):
                    skip_keys.add(key)

        # Log skip info
        if skip_keys:
            logger.info(f"Batch validation: skipping {len(skip_keys)} keys (screener passed or chapter_type skip)")

        # Create verification files (only for keys NOT in skip_keys)
        files = {}
        for key, content in results.items():
            if key in skip_keys:
                continue  # Don't even create file for skipped keys (saves agent cost)
            if key in originals:
                files[key] = VerificationFile(
                    key=key,
                    original=originals[key],
                    processed=content,
                )

        if not files:
            logger.info("Batch validation: all keys skipped, no validation needed")
            return failed

        # Run each batch validator
        for validator in self._batch_validators:
            try:
                batch_result = validator.validate_batch(files)

                # Process results
                for key, validation_result in batch_result.items():
                    if not validation_result.is_valid:
                        failed.add(key)

                    # Record validation
                    self._tracker.record_validation(key, {
                        "validator": validator.name,
                        "is_valid": validation_result.is_valid,
                        "reason": validation_result.reason,
                    })

            except Exception as e:
                logger.error(f"Batch validator {validator.name} error: {e}")

        return failed

    def _aggregate_all_parts(self, successful_keys: Set[str]):
        """
        Aggregate multi-part files (.part{N} only).

        Per design v2:
        - .part{N} = proactive split by Pipeline, persisted, aggregated here
        - .sub{N} = runtime virtual split by Executor, aggregated internally, NOT here

        Args:
            successful_keys: Set of successfully completed keys
        """
        from collections import defaultdict

        # Group by base key - ONLY .part files (not .sub)
        parts_by_base: Dict[str, List[str]] = defaultdict(list)

        for key in successful_keys:
            # Only aggregate .part files, NOT .sub (which Executor handles)
            if '.part' in key and '.sub' not in key:
                base_key = key.rsplit('.part', 1)[0]
                parts_by_base[base_key].append(key)

        # Aggregate each group
        for base_key, part_keys in parts_by_base.items():
            if len(part_keys) > 1:
                try:
                    result = self._persistence.aggregate_parts(base_key, part_keys)
                    if result:
                        logger.info(f"Aggregated {len(part_keys)} parts into {base_key}")
                except Exception as e:
                    logger.warning(f"Failed to aggregate {base_key}: {e}")


# Result class
from dataclasses import dataclass, field


@dataclass
class ProcessingResultV2:
    """Result of processing."""
    total: int
    completed: int
    failed: int
    skipped: int = 0
    failed_keys: List[str] = field(default_factory=list)
    results: Dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
