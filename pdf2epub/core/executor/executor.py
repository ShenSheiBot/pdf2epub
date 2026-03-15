"""
Unified Executor - Mega Unit Architecture.

Key design:
- Batch job = "Mega Unit" (just a bigger unit)
- No retry loop: failed units update state and re-enter pending pool
- Batch + Online run at the same time in unified main loop
- Chain[0].mode decides execution mode for each unit
- Unified dependency tree (context injection + aggregation)

Termination condition:
    pending empty && futures empty && batch_futures empty → terminate
"""

from typing import Dict, List, Set, Optional, Any, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from loguru import logger
import time
import signal
import tiktoken

from pathlib import Path

# Lazy-loaded tokenizer for token counting
_tokenizer = None


def _get_tokenizer():
    """Get or create tiktoken tokenizer (lazy loaded)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if not text or not isinstance(text, str):
        return 0
    try:
        return len(_get_tokenizer().encode(text))
    except Exception:
        return 0


from ._protocol import WorkUnit, ChainEntry, ExecutionResult, ProcessResult
from .state import UnitState, QuotaConfig, create_unit_state
from .batch_state import MegaUnitState, get_mega_unit_id, is_safety_error
from .saver import DiskFirstSaver
from ..hooks import CompositeHooks, ErrorType
from ..types import is_sub_key

if TYPE_CHECKING:
    from .._protocol import ProcessContext, ProcessorProtocol
    from ..context import ContextInjector
    from ...processors.utils.splitter_strategies import ContentSplitter


def get_split_depth(unit_id: str) -> int:
    """Get the nesting depth of a unit based on .sub suffixes."""
    return unit_id.count('.sub')


def handle_split(
    unit_id: str,
    unit_states: Dict[str, UnitState],
    pending: Set[str],
    splitter: "ContentSplitter",
    max_tokens: int = 4000,
) -> Tuple[bool, int]:
    """
    Split a failed unit into virtual children.

    Key design:
    - Parent becomes aggregation unit, waits for children
    - Children inherit half quota (integer division)
    - No splitting if content has no newlines

    Args:
        unit_id: ID of the unit to split
        unit_states: All unit states (modified in place)
        pending: Pending units set (modified in place)
        splitter: Content splitter to use
        max_tokens: Max tokens per part for splitter

    Returns:
        Tuple of (success, new_depth) - new_depth is the depth of created children
    """
    state = unit_states[unit_id]

    # Cannot split if no newlines
    if '\n' not in state.content:
        return (False, 0)

    # Try to split
    try:
        child_contents = splitter.split(state.content, max_tokens)
    except Exception as e:
        logger.warning(f"{unit_id}: Splitter error: {e}")
        return (False, 0)

    if len(child_contents) <= 1:
        return (False, 0)

    # Calculate children's quota (half, integer division)
    child_total_quota = state.total_quota // 2
    child_quotas_template = {k: v // 2 for k, v in state.quotas.items()}

    # Create virtual children
    child_ids = []
    for i, content in enumerate(child_contents):
        child_id = f"{unit_id}.sub{i}"
        child_ids.append(child_id)

        unit_states[child_id] = UnitState(
            chain=list(state.chain),  # Copy chain
            total_quota=child_total_quota,
            quotas=dict(child_quotas_template),  # Each child gets own dict!
            is_virtual=True,
            aggregates_to=unit_id,
            content=content,
        )
        pending.add(child_id)

    # Parent becomes aggregation unit
    state.children = child_ids
    state.is_aggregation = True
    pending.add(unit_id)  # Parent also enters pool (is_ready will wait for children)

    # Calculate new depth (children are one level deeper)
    new_depth = get_split_depth(child_ids[0])

    logger.info(f"{unit_id}: Split into {len(child_ids)} children (quota={child_total_quota}, depth={new_depth})")
    return (True, new_depth)


# Default batch threshold: minimum units to use batch mode
DEFAULT_BATCH_THRESHOLD = 5


class Executor:
    """
    Unified Executor with Mega Unit Architecture.

    Batch jobs are treated as "Mega Units" - just bigger units that process
    multiple items at once. They share the same lifecycle as regular units:
    - Have an ID (hash of contained unit IDs)
    - Have state files (batch_states/batch_{id}.json)
    - Return results through unified _handle_result()
    - Failed units requeue naturally

    Termination:
    - When pending is empty AND no futures AND no batch_futures → terminate
    """

    def __init__(
        self,
        llm_client: Any,
        model_chain: List[ChainEntry],
        processor: "ProcessorProtocol",
        hooks: CompositeHooks,
        batch_client: Optional[Any] = None,
        quota_config: Optional[QuotaConfig] = None,
        max_workers: int = 4,
        context_injector: Optional["ContextInjector"] = None,
        saver: Optional[DiskFirstSaver] = None,
        splitter: Optional["ContentSplitter"] = None,
        split_max_tokens: int = 4000,
        batch_poll_interval: int = 60,
        online_fallback_threshold: int = 5,
        network_circuit_breaker_threshold: int = 5,
        model_output_limits: Optional[Dict[str, int]] = None,
        batch_state_dir: Optional[Path] = None,
    ):
        """
        Initialize Executor.

        Args:
            llm_client: LLM client for online API calls
            model_chain: List of ChainEntry for model fallback
            processor: Processor for building prompts and cleaning responses
            hooks: CompositeHooks for pre/post processing
            batch_client: Optional batch client (None = online only)
            quota_config: Quota configuration (default: QuotaConfig())
            max_workers: Maximum concurrent online workers
            context_injector: Optional context injector for sequential mode
            saver: DiskFirstSaver for persistence (disk-first architecture)
            splitter: Optional content splitter for dynamic splitting
            split_max_tokens: Default max tokens per split part
            batch_poll_interval: Seconds between batch status polls
            online_fallback_threshold: Min batch failures before considering online
            network_circuit_breaker_threshold: Consecutive network failures to trigger abort
            model_output_limits: Per-model token limits
            batch_state_dir: Directory for batch state persistence (enables resume)
        """
        self._llm_client = llm_client
        self._model_chain = model_chain
        self._processor = processor
        self._hooks = hooks
        self._batch_client = batch_client
        self._quota_config = quota_config or QuotaConfig()
        self._max_workers = max_workers
        self._context_injector = context_injector
        self._saver = saver
        self._splitter = splitter
        self._split_max_tokens = split_max_tokens
        self._batch_poll_interval = batch_poll_interval
        self._online_fallback_threshold = online_fallback_threshold
        self._network_circuit_breaker_threshold = network_circuit_breaker_threshold
        self._model_output_limits = model_output_limits or {}

        # Batch state directory (batch_states/)
        self._batch_states_dir = batch_state_dir / "batch_states" if batch_state_dir else None
        if self._batch_states_dir:
            self._batch_states_dir.mkdir(parents=True, exist_ok=True)

        # Register signal handler for batch cancellation on interrupt
        self._original_sigint = None
        self._original_sigterm = None
        if self._batch_client and self._batch_states_dir:
            self._original_sigint = signal.getsignal(signal.SIGINT)
            self._original_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)

        # Circuit breaker state
        self._consecutive_network_failures = 0
        self._network_circuit_broken = False

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM - keep batch jobs running for resume."""
        logger.warning("Interrupt received. Batch jobs will continue running in background.")
        logger.info("Use --resume to continue and retrieve results.")
        logger.info("Use 'pdf2epub cancel-batch' to cancel batch jobs.")

        # Restore original handlers and re-raise
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)

        raise KeyboardInterrupt("Interrupted. Use --resume to continue or 'pdf2epub cancel-batch' to cancel.")

    def _cancel_all_batch_jobs(self) -> None:
        """Cancel all active batch jobs (iterates batch_states directory)."""
        if not self._batch_states_dir or not self._batch_states_dir.exists():
            return

        for state_file in self._batch_states_dir.glob("batch_*.json"):
            try:
                state = MegaUnitState.load(state_file)
                if state and state.job_name:
                    self._batch_client.cancel(state.job_name)
                    logger.info(f"Cancelled batch job: {state.job_name}")
            except Exception as e:
                logger.warning(f"Failed to cancel batch job: {e}")
            finally:
                state_file.unlink(missing_ok=True)

    def _get_split_threshold(self, model_name: Optional[str] = None) -> int:
        """Get the split threshold for a model."""
        if model_name and model_name in self._model_output_limits:
            return self._model_output_limits[model_name]
        return self._split_max_tokens

    def execute(
        self,
        units: List[WorkUnit],
        context_base: Optional["ProcessContext"] = None,
        resume_batch: bool = False,
    ) -> ExecutionResult:
        """
        Execute all units with unified batch/online processing.

        Algorithm:
        1. Initialize per-unit state
        2. Main loop:
           a. Get ready units
           b. Dispatch by chain[0].mode: batch queue or online submit
           c. If batch queue >= threshold: submit as mega unit
           d. Wait for completions
           e. Handle results through unified _handle_result()
        3. Failed units requeue, next round decides batch/online again

        Args:
            units: Units to process
            context_base: Base context for all units
            resume_batch: Whether to resume existing batch jobs

        Returns:
            ExecutionResult with all outcomes
        """
        if not units:
            return ExecutionResult()

        # Cancel old batch jobs if not resuming
        if not resume_batch:
            self._cancel_all_batch_jobs()

        start_time = time.time()

        # Initialize per-unit state
        unit_states: Dict[str, UnitState] = {}
        for u in units:
            unit_states[u.id] = create_unit_state(
                chain=self._model_chain,
                quota_config=self._quota_config,
                content=u.content,
            )

        # Track originals and unit map
        originals: Dict[str, str] = {u.id: u.content for u in units}
        unit_map: Dict[str, WorkUnit] = {u.id: u for u in units}

        # Result collections
        results: Dict[str, str] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()
        skipped: Set[str] = set()
        safety_blocked: Set[str] = set()
        validation_failed: Set[str] = set()
        screener_passed: Set[str] = set()
        fallback_used: Set[str] = set()

        # Concurrent state
        pending: Set[str] = {u.id for u in units}
        in_progress: Set[str] = set()
        futures: Dict[Future, str] = {}  # Online futures: future -> unit_id
        batch_futures: Dict[Future, List[str]] = {}  # Mega unit futures: future -> [unit_ids]

        # Stats
        total_attempts = 0
        successful_attempts = 0
        splits_performed = 0
        max_depth_reached = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending or futures or batch_futures:
                # 1. Get ready units
                ready_ids = self._get_ready_ids(pending, completed, in_progress, unit_states)

                # 2. Dispatch by chain[0].mode
                batch_queue: List[str] = []

                for uid in ready_ids:
                    pending.discard(uid)
                    state = unit_states[uid]

                    # Aggregation: handle directly
                    if state.is_aggregation:
                        agg_result = self._handle_aggregation(
                            uid, state, unit_states, results, completed, failed
                        )
                        if agg_result:
                            results[uid] = agg_result
                            completed.add(uid)
                        else:
                            # Child data lost (disk corruption during resume), mark as failed
                            failed.add(uid)
                            logger.error(f"{uid}: Aggregation failed - child data missing")
                        continue

                    # Check chain[0].mode
                    current_mode = state.get_current_mode()

                    if current_mode == "batch" and self._batch_client:
                        batch_queue.append(uid)
                    else:
                        # Online: submit single task
                        in_progress.add(uid)
                        unit = self._get_or_create_unit(uid, unit_map, unit_states, originals)
                        context = self._build_context(unit, context_base, completed, results, originals)
                        future = pool.submit(
                            self._process_single,
                            unit, state, context, originals.get(uid, state.content)
                        )
                        futures[future] = uid

                # 3. Handle batch queue
                # Use online_fallback_threshold as minimum batch size
                # When threshold=0, always use batch if any units (for testing)
                batch_threshold = max(1, self._online_fallback_threshold)
                if batch_queue:
                    if len(batch_queue) >= batch_threshold:
                        # Submit as mega unit
                        in_progress.update(batch_queue)
                        future = pool.submit(
                            self._process_batch_as_unit,
                            batch_queue, unit_states, unit_map, context_base, originals, resume_batch
                        )
                        batch_futures[future] = batch_queue
                    else:
                        # Too few: remove batch entries, force online
                        for uid in batch_queue:
                            unit_states[uid].remove_batch_entries()
                            in_progress.add(uid)
                            unit = self._get_or_create_unit(uid, unit_map, unit_states, originals)
                            context = self._build_context(unit, context_base, completed, results, originals)
                            future = pool.submit(
                                self._process_single,
                                unit, unit_states[uid], context, originals.get(uid, unit_states[uid].content)
                            )
                            futures[future] = uid

                # 4. Check circuit breaker
                if self._network_circuit_broken:
                    logger.error(f"Circuit breaker tripped. Marking {len(pending)} pending as failed.")
                    failed.update(pending)
                    pending.clear()
                    break

                # 5. Check termination
                all_futures = set(futures.keys()) | set(batch_futures.keys())
                if not all_futures:
                    if pending:
                        logger.warning(f"{len(pending)} units have unsatisfied dependencies")
                        failed.update(pending)
                        pending.clear()
                    break

                # 6. Wait for any completion
                done, _ = wait(all_futures, return_when=FIRST_COMPLETED, timeout=1.0)

                # 7. Handle completed futures
                for future in done:
                    if future in batch_futures:
                        # Mega unit completed
                        batch_unit_ids = batch_futures.pop(future)
                        in_progress.difference_update(batch_unit_ids)

                        try:
                            batch_results = future.result()
                            for uid, result in batch_results:
                                state = unit_states[uid]
                                # Only count non-skipped as attempts
                                if not result.skipped:
                                    total_attempts += 1
                                current_entry = state.get_current_entry()
                                model_str = f"{current_entry.provider}/{current_entry.model}" if current_entry else "batch"

                                if result.skipped:
                                    skipped.add(uid)
                                    if result.content is not None:
                                        results[uid] = result.content
                                    completed.add(uid)
                                    if self._saver:
                                        with self._saver.attempt(uid, "skipped") as attempt:
                                            attempt.skip(result.skip_reason or "pre-process", result.content)

                                elif result.success:
                                    successful_attempts += 1
                                    results[uid] = result.content
                                    completed.add(uid)
                                    if result.context_ready:
                                        screener_passed.add(uid)
                                    if self._saver:
                                        with self._saver.attempt(uid, model_str) as attempt:
                                            attempt.success(
                                                result.content,
                                                output_tokens=result.output_tokens,
                                                context_ready=result.context_ready,
                                            )

                                else:
                                    # Failed: unified handling
                                    requeued, split_depth = self._handle_failure(
                                        uid, result, state, unit_states,
                                        pending, completed, failed, safety_blocked, validation_failed,
                                        results, fallback_used, model_str
                                    )
                                    if split_depth > 0:
                                        splits_performed += 1
                                        max_depth_reached = max(max_depth_reached, split_depth)

                        except Exception as e:
                            logger.error(f"Batch future error: {e}")
                            # Route through _handle_failure so units can
                            # fallback to online models via the normal chain
                            for uid in batch_unit_ids:
                                total_attempts += 1
                                state = unit_states[uid]
                                result = ProcessResult(success=False, error=e)
                                current_entry = state.get_current_entry()
                                model_str = f"{current_entry.provider}/{current_entry.model}" if current_entry else "batch"
                                requeued, split_depth = self._handle_failure(
                                    uid, result, state, unit_states,
                                    pending, completed, failed, safety_blocked, validation_failed,
                                    results, fallback_used, model_str
                                )

                    else:
                        # Online unit completed
                        uid = futures.pop(future)
                        in_progress.discard(uid)
                        state = unit_states[uid]

                        try:
                            result = future.result()
                            # Only count non-skipped as attempts
                            if not result.skipped:
                                total_attempts += 1

                            current_entry = state.get_current_entry()
                            model_str = f"{current_entry.provider}/{current_entry.model}" if current_entry else "unknown"

                            if result.skipped:
                                skipped.add(uid)
                                if result.content is not None:
                                    results[uid] = result.content
                                completed.add(uid)
                                if self._saver:
                                    with self._saver.attempt(uid, "skipped") as attempt:
                                        attempt.skip(result.skip_reason or "pre-process", result.content)

                            elif result.success:
                                successful_attempts += 1
                                results[uid] = result.content
                                completed.add(uid)
                                if result.context_ready:
                                    screener_passed.add(uid)
                                    if self._context_injector:
                                        self._context_injector.cache_completed(
                                            uid, originals.get(uid, ""), result.content
                                        )
                                if self._saver:
                                    with self._saver.attempt(uid, model_str) as attempt:
                                        attempt.success(
                                            result.content,
                                            output_tokens=result.output_tokens,
                                            duration_seconds=result.duration_seconds,
                                            context_ready=result.context_ready,
                                        )

                            else:
                                requeued, split_depth = self._handle_failure(
                                    uid, result, state, unit_states,
                                    pending, completed, failed, safety_blocked, validation_failed,
                                    results, fallback_used, model_str
                                )
                                if split_depth > 0:
                                    splits_performed += 1
                                    max_depth_reached = max(max_depth_reached, split_depth)

                        except Exception as e:
                            import traceback
                            logger.error(f"{uid}: Unexpected error: {e}")
                            logger.debug(f"{uid}: Traceback:\n{traceback.format_exc()}")
                            if self._saver:
                                with self._saver.attempt(uid, "unknown") as attempt:
                                    attempt.failure("unknown", str(e)[:500])
                            failed.add(uid)

        # Build stats
        duration = time.time() - start_time
        stats = {
            "duration_seconds": duration,
            "total_units": len(units),
            "completed": len(completed),
            "failed": len(failed),
            "skipped": len(skipped),
        }

        return ExecutionResult(
            results=results,
            completed=completed,
            failed=failed,
            skipped=skipped,
            safety_blocked=safety_blocked,
            validation_failed=validation_failed,
            screener_passed=screener_passed,
            fallback_used=fallback_used,
            stats=stats,
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
            splits_performed=splits_performed,
            max_depth_reached=max_depth_reached,
        )

    def _process_batch_as_unit(
        self,
        unit_ids: List[str],
        unit_states: Dict[str, UnitState],
        unit_map: Dict[str, WorkUnit],
        context_base: Optional["ProcessContext"],
        originals: Dict[str, str],
        resume_batch: bool,
    ) -> List[Tuple[str, ProcessResult]]:
        """
        Process a batch job as a mega unit.

        This method is self-contained:
        - Has its own ID (hash of unit_ids)
        - Has its own state file
        - Resume through ID matching

        Args:
            unit_ids: Unit IDs to process in this batch
            unit_states: All unit states
            unit_map: Unit ID to WorkUnit map
            context_base: Base context
            originals: Original content map
            resume_batch: Whether to resume existing jobs

        Returns:
            List of (unit_id, ProcessResult) tuples
        """
        from .._protocol import ProcessContext
        from ...utils.batch_utils import BatchRequest, BatchJobState

        results: List[Tuple[str, ProcessResult]] = []

        # Compute mega unit ID
        mega_id = get_mega_unit_id(unit_ids)
        state_file = self._batch_states_dir / f"{mega_id}.json" if self._batch_states_dir else None

        # Get batch entry from first unit's chain
        first_state = unit_states[unit_ids[0]]
        batch_entry = first_state.get_batch_entry()
        if not batch_entry:
            # No batch entry available, return failures
            for uid in unit_ids:
                results.append((uid, ProcessResult(
                    success=False,
                    error=Exception("No batch entry in chain"),
                )))
            return results

        # Check for resume
        job_name = None
        if state_file and state_file.exists():
            existing_state = MegaUnitState.load(state_file)
            if existing_state:
                if existing_state.job_state == "SUCCEEDED":
                    # Already completed, get cached results
                    logger.info(f"Mega unit {mega_id} already completed, fetching results")
                    job_name = existing_state.job_name
                elif existing_state.job_state in ("PENDING", "RUNNING"):
                    # Still running, continue waiting
                    logger.info(f"Resuming mega unit {mega_id}: {existing_state.job_name}")
                    job_name = existing_state.job_name
                # Restore key order for Vertex line-order correlation
                if job_name and existing_state.processing_keys:
                    if hasattr(self._batch_client, '_job_keys'):
                        self._batch_client._job_keys[job_name] = existing_state.processing_keys
                        logger.debug(f"Restored {len(existing_state.processing_keys)} processing keys for {job_name}")

        # Build requests if not resuming
        batch_requests: List[BatchRequest] = []
        units_to_process: Dict[str, Tuple[WorkUnit, "ProcessContext"]] = {}

        if job_name is None:
            for uid in unit_ids:
                unit = self._get_or_create_unit(uid, unit_map, unit_states, originals)

                # Build context
                if context_base is None:
                    context = ProcessContext.from_work_unit(unit)
                else:
                    context = ProcessContext(
                        file_key=unit.file_key,
                        book_title=context_base.book_title,
                        part_index=unit.part_index,
                        total_parts=unit.total_parts,
                        split_version=unit.split_version,
                        source_language=context_base.source_language,
                        target_language=context_base.target_language,
                        content_type=context_base.content_type,
                        chapter_type=unit.chapter_type or context_base.chapter_type,
                        chapter_title=unit.chapter_title or context_base.chapter_title,
                        chapter_number=unit.chapter_number or context_base.chapter_number,
                        is_notes_chapter=(unit.chapter_type == "notes"),
                        is_vertical_text=context_base.is_vertical_text,
                        has_global_footnotes=context_base.has_global_footnotes,
                        book_language=context_base.book_language,
                        toc_path=unit.toc_path or context_base.toc_path,
                        page_range=unit.page_range or context_base.page_range,
                        extra=context_base.extra,
                    )

                # Pre-process check
                pre_result = self._hooks.pre_process(uid, unit.content, context)
                if not pre_result.should_process:
                    results.append((uid, ProcessResult(
                        success=True,
                        content=pre_result.fallback_result,
                        skipped=True,
                        skip_reason=pre_result.skip_reason,
                    )))
                    continue

                # Build prompt and request
                prompt = self._processor.build_prompt(unit.content, context)
                contents = self._convert_prompt_to_batch_contents(prompt)
                batch_requests.append(BatchRequest(key=uid, contents=contents))
                units_to_process[uid] = (unit, context)

            # Submit job if we have requests
            skipped_count = len(unit_ids) - len(batch_requests)
            if batch_requests:
                job_name = self._batch_client.submit(batch_requests)
                if skipped_count > 0:
                    logger.info(
                        f"Submitted mega unit {mega_id}: {job_name} "
                        f"({len(batch_requests)} units, {skipped_count} skipped by pre-process)"
                    )
                else:
                    logger.info(f"Submitted mega unit {mega_id}: {job_name} ({len(batch_requests)} units)")

                # Save state (include key order for Vertex line-order correlation on resume)
                if state_file:
                    processing_keys = [req.key for req in batch_requests]
                    state = MegaUnitState(
                        job_name=job_name,
                        job_state="RUNNING",
                        processing_keys=processing_keys,
                    )
                    state.save(state_file)
            else:
                # All units were skipped
                return results
        else:
            # Resuming: rebuild units_to_process from unit_ids
            for uid in unit_ids:
                unit = self._get_or_create_unit(uid, unit_map, unit_states, originals)
                if context_base is None:
                    context = ProcessContext.from_work_unit(unit)
                else:
                    context = ProcessContext(
                        file_key=unit.file_key,
                        book_title=context_base.book_title,
                        part_index=unit.part_index,
                        total_parts=unit.total_parts,
                        split_version=unit.split_version,
                        source_language=context_base.source_language,
                        target_language=context_base.target_language,
                        content_type=context_base.content_type,
                        chapter_type=unit.chapter_type or context_base.chapter_type,
                        chapter_title=unit.chapter_title or context_base.chapter_title,
                        chapter_number=unit.chapter_number or context_base.chapter_number,
                        is_notes_chapter=(unit.chapter_type == "notes"),
                        is_vertical_text=context_base.is_vertical_text,
                        has_global_footnotes=context_base.has_global_footnotes,
                        book_language=context_base.book_language,
                        toc_path=unit.toc_path or context_base.toc_path,
                        page_range=unit.page_range or context_base.page_range,
                        extra=context_base.extra,
                    )
                units_to_process[uid] = (unit, context)

        # Poll for completion
        if job_name:
            poll_count = 0
            poll_errors = 0
            max_poll_errors = 5  # Give up after 5 consecutive poll failures
            while True:
                try:
                    job_info = self._batch_client.get_status(job_name)
                    poll_errors = 0  # Reset on success
                except Exception as poll_exc:
                    poll_errors += 1
                    logger.warning(
                        f"Batch poll error ({poll_errors}/{max_poll_errors}): {poll_exc}"
                    )
                    if poll_errors >= max_poll_errors:
                        raise  # Propagate to batch future error handler
                    time.sleep(self._batch_poll_interval)
                    continue
                poll_count += 1

                # Log polling status periodically
                if poll_count % 5 == 1:  # Every 5 polls (~5 min with 60s interval)
                    logger.info(f"Batch job {job_name}: {job_info.state.name} (poll #{poll_count})")

                # Use client's COMPLETED_STATES to handle both Gemini and Vertex terminal states
                completed_states = getattr(self._batch_client, 'COMPLETED_STATES', {
                    BatchJobState.SUCCEEDED, BatchJobState.FAILED,
                    BatchJobState.CANCELLED, BatchJobState.EXPIRED,
                })

                if job_info.state in completed_states and job_info.state in (
                    BatchJobState.SUCCEEDED, BatchJobState.PARTIALLY_SUCCEEDED,
                ):
                    if state_file:
                        state = MegaUnitState(job_name=job_name, job_state="SUCCEEDED")
                        state.save(state_file)
                    break

                elif job_info.state in completed_states:
                    logger.error(f"Batch job {job_info.state.name}: {job_info.error}")

                    # Classify job-level error
                    error_msg = str(job_info.error) if job_info.error else f"batch job {job_info.state.name}"
                    error = Exception(error_msg)

                    for uid in units_to_process.keys():
                        results.append((uid, ProcessResult(
                            success=False,
                            error=error,
                        )))

                    # Clear state file
                    if state_file:
                        state_file.unlink(missing_ok=True)

                    return results

                time.sleep(self._batch_poll_interval)

            # Get results
            batch_responses = self._batch_client.get_results(job_name)
            response_map = {r.key: r for r in batch_responses}

            for uid, (unit, context) in units_to_process.items():
                resp = response_map.get(uid)

                if resp is None or resp.error or resp.text is None:
                    error_msg = str(resp.error) if resp and resp.error else "No response or empty text"
                    # Log raw response for debugging when text is None but no error
                    if resp and not resp.error and resp.text is None:
                        raw = getattr(resp, 'raw_response', None)
                        logger.debug(f"{uid}: Batch response has no text. Raw: {str(raw)[:500]}")
                    results.append((uid, ProcessResult(
                        success=False,
                        error=Exception(error_msg),
                    )))
                    continue

                # Clean response
                cleaned = self._processor.clean_response(resp.text)
                original = originals.get(uid, unit.content)
                chapter_type = unit.chapter_type or ''

                # Post-process: transform + validate
                transformed, hook_result = self._hooks.post_process(
                    uid, original, cleaned, chapter_type, context
                )

                if hook_result.accepted:
                    final = self._processor.post_process(transformed, context)
                    output_tokens = _count_tokens(final)

                    # Record attempt
                    unit_states[uid].record_attempt(final)

                    results.append((uid, ProcessResult(
                        success=True,
                        content=final,
                        context_ready=hook_result.context_ready,
                        output_tokens=output_tokens,
                    )))
                else:
                    # Validation failed
                    unit_states[uid].record_attempt(transformed)
                    results.append((uid, ProcessResult(
                        success=False,
                        error=Exception(hook_result.rejection_reason or "Validation failed"),
                    )))

            # Clear state file after processing
            if state_file:
                state_file.unlink(missing_ok=True)

        return results

    def _get_or_create_unit(
        self,
        uid: str,
        unit_map: Dict[str, WorkUnit],
        unit_states: Dict[str, UnitState],
        originals: Dict[str, str],
    ) -> WorkUnit:
        """Get existing unit or create from state (for virtual units)."""
        if uid in unit_map:
            return unit_map[uid]

        state = unit_states[uid]
        file_key = uid
        if is_sub_key(file_key):
            file_key = file_key.rsplit('.sub', 1)[0]
        if '.part' in file_key:
            file_key = file_key.rsplit('.part', 1)[0]

        unit = WorkUnit(id=uid, file_key=file_key, content=state.content)
        unit_map[uid] = unit
        originals[uid] = state.content
        return unit

    def _handle_aggregation(
        self,
        uid: str,
        state: UnitState,
        unit_states: Dict[str, UnitState],
        results: Dict[str, str],
        completed: Set[str],
        failed: Set[str],
    ) -> Optional[str]:
        """Handle aggregation unit: combine children results."""
        child_results = []
        for child_id in sorted(state.children):
            content = None
            if self._saver:
                content = self._saver.load(child_id)
            if content is None and child_id in results:
                content = results[child_id]

            if content is not None:
                child_results.append(content)
            else:
                logger.warning(f"{uid}: Child {child_id} not found")
                return None

        aggregated = "\n\n".join(child_results)

        # Save via saver
        if self._saver:
            current_entry = state.get_current_entry() or (self._model_chain[0] if self._model_chain else None)
            model_str = f"{current_entry.provider}/{current_entry.model}" if current_entry else "aggregation"
            with self._saver.attempt(uid, model_str) as attempt:
                attempt.success(aggregated, output_tokens=_count_tokens(aggregated))

        logger.info(f"{uid}: Aggregated {len(state.children)} children")

        # Remove children from completed
        for child_id in state.children:
            completed.discard(child_id)

        return aggregated

    def _convert_prompt_to_batch_contents(self, prompt: Any) -> List[Dict]:
        """Convert processor prompt to batch API contents format."""
        if isinstance(prompt, str):
            return [{"parts": [{"text": prompt}], "role": "user"}]

        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                text_parts = []
                for msg in prompt:
                    role = msg.get("role", "user")
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        text_parts.append(f"[{role.upper()}]: {content}")
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                text_parts.append(f"[{role.upper()}]: {part['text']}")
                            elif isinstance(part, str):
                                text_parts.append(f"[{role.upper()}]: {part}")
                combined = "\n\n".join(text_parts)
                return [{"parts": [{"text": combined}], "role": "user"}]

            elif prompt and isinstance(prompt[0], dict) and "type" in prompt[0]:
                text_parts = []
                for part in prompt:
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                combined = "\n\n".join(text_parts)
                return [{"parts": [{"text": combined}], "role": "user"}]

        return [{"parts": [{"text": str(prompt)}], "role": "user"}]

    def _process_single(
        self,
        unit: WorkUnit,
        state: UnitState,
        context: "ProcessContext",
        original: str,
    ) -> ProcessResult:
        """Process a single unit (runs in thread pool)."""
        # Pre-processing
        pre_result = self._hooks.pre_process(unit.id, unit.content, context)
        if not pre_result.should_process:
            return ProcessResult(
                success=True,
                content=pre_result.fallback_result,
                skipped=True,
                skip_reason=pre_result.skip_reason,
            )

        # Get current model
        entry = state.get_current_entry()
        if not entry:
            return ProcessResult(
                success=False,
                error=Exception("No models available in chain"),
            )

        try:
            # Build prompt
            prompt = self._processor.build_prompt(unit.content, context)

            # Record start time
            llm_start_time = time.time()

            # Call LLM
            response = self._llm_client.generate(
                prompt=prompt,
                model_configs=[entry.to_dict()],
                operation_name=f"{self._processor.name}:{unit.id}",
            )

            # Calculate duration
            llm_duration = time.time() - llm_start_time

            # Clean response
            cleaned = self._processor.clean_response(response)
            output_tokens = _count_tokens(cleaned)

            # Post-process: transform + validate
            chapter_type = getattr(context, 'chapter_type', "") or ""
            transformed, hook_result = self._hooks.post_process(
                unit.id, original, cleaned, chapter_type, context
            )

            if hook_result.accepted:
                final = self._processor.post_process(transformed, context)
            else:
                final = transformed

            # Record attempt
            state.record_attempt(final)

            if hook_result.accepted:
                return ProcessResult(
                    success=True,
                    content=final,
                    context_ready=hook_result.context_ready,
                    output_tokens=output_tokens,
                    duration_seconds=llm_duration,
                )
            else:
                return ProcessResult(
                    success=False,
                    error=Exception(hook_result.rejection_reason or "Validation failed"),
                    output_tokens=output_tokens,
                    duration_seconds=llm_duration,
                )

        except Exception as e:
            import traceback
            logger.debug(f"{unit.id}: Exception traceback:\n{traceback.format_exc()}")
            return ProcessResult(success=False, error=e)

    def _handle_failure(
        self,
        unit_id: str,
        result: ProcessResult,
        state: UnitState,
        unit_states: Dict[str, UnitState],
        pending: Set[str],
        completed: Set[str],
        failed: Set[str],
        safety_blocked: Set[str],
        validation_failed: Set[str],
        results: Dict[str, str],
        fallback_used: Set[str],
        model_str: str = "unknown",
    ) -> Tuple[bool, int]:
        """
        Handle a failed processing result.

        Unified for both batch and online paths.

        Returns:
            Tuple of (requeued, split_depth)
        """
        # Classify error
        error_type, effect = self._hooks.classify_error(result.error)
        error_msg = str(result.error)[:500] if result.error else "Unknown error"

        # Network circuit breaker
        network_errors = (ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT)
        if error_type in network_errors:
            self._consecutive_network_failures += 1
            if self._consecutive_network_failures >= self._network_circuit_breaker_threshold:
                if not self._network_circuit_broken:
                    self._network_circuit_broken = True
                    logger.error(f"CIRCUIT BREAKER: {self._consecutive_network_failures} consecutive network failures")
        else:
            self._consecutive_network_failures = 0

        # Get current entry before applying effect
        current_entry = state.get_current_entry()
        if model_str == "unknown" and current_entry:
            model_str = f"{current_entry.provider}/{current_entry.model}"

        # Apply effect to state
        state.apply_effect(effect, current_entry)

        # Track by error type
        if error_type == ErrorType.SAFETY:
            safety_blocked.add(unit_id)
        elif error_type == ErrorType.VALIDATION:
            validation_failed.add(unit_id)

        # Record failure via saver
        if self._saver:
            with self._saver.attempt(unit_id, model_str) as attempt:
                attempt.failure(error_type.value, error_msg)

        # Check if can retry
        if state.can_retry(effect.quota_type):
            pending.add(unit_id)
            logger.info(
                f"{unit_id}: {error_type.value}, re-queued "
                f"(quota: {state.total_quota}, chain: {len(state.chain)}) - {error_msg[:100]}"
            )
            return (True, 0)

        # Network errors: skipping split (splitting won't fix network issues)
        if error_type in network_errors:
            logger.warning(f"{unit_id}: {error_type.value} exhausted chain, skipping split - {error_msg[:200]}")
        else:
            # Content errors: try splitting
            if self._splitter:
                model_name = current_entry.model if current_entry else None
                split_threshold = self._get_split_threshold(model_name)
                split_success, split_depth = handle_split(
                    unit_id, unit_states, pending, self._splitter, split_threshold
                )
                if split_success:
                    return (True, split_depth)

        # Try longest fallback
        longest = state.get_longest()
        if longest:
            results[unit_id] = longest
            completed.add(unit_id)
            fallback_used.add(unit_id)
            # Record fallback via saver
            if self._saver:
                with self._saver.attempt(unit_id, f"{model_str}/fallback") as attempt:
                    attempt.success(longest, output_tokens=_count_tokens(longest))
            logger.warning(f"{unit_id}: Using longest fallback ({len(longest)} chars)")
            return (False, 0)

        # No fallback available
        failed.add(unit_id)
        logger.error(f"{unit_id}: Failed, no fallback available - {error_msg[:200]}")
        return (False, 0)

    def _get_ready_ids(
        self,
        pending: Set[str],
        completed: Set[str],
        in_progress: Set[str],
        unit_states: Dict[str, UnitState],
    ) -> List[str]:
        """Get unit IDs that are ready (dependencies satisfied)."""
        ready = []
        is_sequential = self._context_injector and self._context_injector.is_sequential

        for unit_id in pending:
            if unit_id in in_progress:
                continue

            state = unit_states.get(unit_id)
            if state:
                # Check children (always)
                if state.children and not all(c in completed for c in state.children):
                    continue

                # Sequential: check depends_on
                if is_sequential:
                    if state.depends_on and not state.depends_on.issubset(completed):
                        continue

            # Sequential: check part ordering
            if is_sequential and '.part' in unit_id and '.sub' not in unit_id:
                parts = unit_id.rsplit('.part', 1)
                if len(parts) == 2:
                    base, part_num_str = parts
                    try:
                        part_num = int(part_num_str)
                        if part_num > 0:
                            prev_id = f"{base}.part{part_num - 1}"
                            if prev_id not in completed and prev_id in pending | in_progress:
                                continue
                    except ValueError:
                        pass

            ready.append(unit_id)

        return ready

    def _build_context(
        self,
        unit: WorkUnit,
        context_base: Optional["ProcessContext"],
        completed: Set[str],
        results: Dict[str, str],
        originals: Dict[str, str],
    ) -> "ProcessContext":
        """Build context with previous part injection (sequential mode)."""
        from .._protocol import ProcessContext

        if context_base is None:
            context_base = ProcessContext.from_work_unit(unit)
        else:
            context_base = ProcessContext(
                file_key=unit.file_key,
                book_title=context_base.book_title,
                part_index=unit.part_index,
                total_parts=unit.total_parts,
                split_version=unit.split_version,
                source_language=context_base.source_language,
                target_language=context_base.target_language,
                content_type=context_base.content_type,
                chapter_type=unit.chapter_type or context_base.chapter_type,
                chapter_title=unit.chapter_title or context_base.chapter_title,
                chapter_number=unit.chapter_number or context_base.chapter_number,
                is_notes_chapter=(unit.chapter_type == "notes"),
                is_vertical_text=context_base.is_vertical_text,
                has_global_footnotes=context_base.has_global_footnotes,
                book_language=context_base.book_language,
                toc_path=unit.toc_path or context_base.toc_path,
                page_range=unit.page_range or context_base.page_range,
                extra=context_base.extra,
            )

        if not self._context_injector or not self._context_injector.is_sequential:
            return context_base

        prev_context = self._context_injector.get_context_for_unit(unit, results, originals)
        if prev_context:
            prev_original, prev_processed = prev_context
            return context_base.with_previous_context(prev_original, prev_processed)

        return context_base
