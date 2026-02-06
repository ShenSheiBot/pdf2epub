"""
Unified Executor - handles batch + online simultaneously.

Key design:
- No retry loop: failed units update state and re-enter pending pool
- Batch + Online run at the same time (not if-else)
- Unified dependency tree (context injection + aggregation)
- Dynamic splitting creates virtual children

Termination condition:
    pending empty && futures empty && batch_job empty → terminate
"""

from typing import Dict, List, Set, Optional, Any, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from loguru import logger
import time
import threading

from ._protocol import WorkUnit, ChainEntry, ExecutionResult, ProcessResult
from .state import UnitState, QuotaConfig, create_unit_state
from ..hooks import CompositeHooks, ErrorType
from ..types import is_sub_key

if TYPE_CHECKING:
    from .._protocol import ProcessContext, ProcessorProtocol
    from ..context import ContextInjector
    from ..tracking import ProcessingTracker
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


class Executor:
    """
    Unified Executor - processes units with batch + online simultaneously.

    Key features:
    - No retry loop: failures update state and re-enter pending pool
    - Batch job runs asynchronously while online processes concurrently
    - Unified dependency tree for context injection and aggregation
    - Dynamic splitting for failed units

    Termination:
    - When pending is empty AND no futures AND no batch job → terminate
    - Remaining pending units (from deadlock) are marked as failed
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
        tracker: Optional["ProcessingTracker"] = None,
        splitter: Optional["ContentSplitter"] = None,
        split_max_tokens: int = 4000,
        batch_poll_interval: int = 60,
        online_fallback_threshold: int = 5,
        network_circuit_breaker_threshold: int = 5,
        model_output_limits: Optional[Dict[str, int]] = None,
        persistence: Optional[Any] = None,  # For realtime saving
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
            tracker: Optional tracker for recording progress
            splitter: Optional content splitter for dynamic splitting
            split_max_tokens: Default max tokens per split part
            batch_poll_interval: Seconds between batch status polls
            online_fallback_threshold: Max batch failures before online fallback
            network_circuit_breaker_threshold: Consecutive network failures to trigger abort
            model_output_limits: Per-model token limits (e.g., {"gemini-3-pro-preview": 8000})
        """
        self._llm_client = llm_client
        self._model_chain = model_chain
        self._processor = processor
        self._hooks = hooks
        self._batch_client = batch_client
        self._quota_config = quota_config or QuotaConfig()
        self._max_workers = max_workers
        self._context_injector = context_injector
        self._tracker = tracker
        self._splitter = splitter
        self._split_max_tokens = split_max_tokens
        self._batch_poll_interval = batch_poll_interval
        self._online_fallback_threshold = online_fallback_threshold
        self._network_circuit_breaker_threshold = network_circuit_breaker_threshold
        self._model_output_limits = model_output_limits or {}
        self._persistence = persistence  # For realtime saving on completion

        # Circuit breaker state
        self._consecutive_network_failures = 0
        self._network_circuit_broken = False

    def _get_split_threshold(self, model_name: Optional[str] = None) -> int:
        """
        Get the split threshold for a model.

        Uses model_output_limits if configured for the model,
        otherwise falls back to default split_max_tokens.

        Args:
            model_name: Model name (e.g., "gemini-3-pro-preview")

        Returns:
            Token limit for splitting
        """
        if model_name and model_name in self._model_output_limits:
            return self._model_output_limits[model_name]
        return self._split_max_tokens

    def execute(
        self,
        units: List[WorkUnit],
        context_base: Optional["ProcessContext"] = None,
    ) -> ExecutionResult:
        """
        Execute all units - batch and online simultaneously.

        Algorithm:
        1. Initialize per-unit state
        2. Separate units by batch/online availability
        3. Submit batch job asynchronously (if any)
        4. Process online units concurrently
        5. Wait for batch results, handle failures
        6. Merge all results

        Args:
            units: Units to process
            context_base: Base context for all units

        Returns:
            ExecutionResult with all outcomes
        """
        if not units:
            return ExecutionResult()

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
        screener_passed: Set[str] = set()  # Keys that passed individual screener
        fallback_used: Set[str] = set()    # Keys that used longest fallback

        # Check if batch is available
        batch_entry = self._get_batch_entry()
        has_batch = batch_entry is not None and self._batch_client is not None

        # Separate units by batch/online
        batch_units: List[WorkUnit] = []
        online_units: List[WorkUnit] = []

        if has_batch:
            for unit in units:
                state = unit_states[unit.id]
                if state.has_batch_available():
                    batch_units.append(unit)
                else:
                    online_units.append(unit)
        else:
            online_units = list(units)

        logger.info(
            f"Processing {len(batch_units)} via batch, {len(online_units)} via online"
        )

        # Batch processing (asynchronous)
        batch_future: Optional[Future] = None
        batch_executor = ThreadPoolExecutor(max_workers=1)

        try:
            if batch_units and has_batch:
                batch_future = batch_executor.submit(
                    self._process_batch,
                    batch_units, batch_entry, context_base, originals, unit_states
                )

            # Aggregate stats
            total_attempts = 0
            successful_attempts = 0
            splits_performed = 0
            max_depth_reached = 0

            # Online processing (concurrent)
            if online_units:
                online_result = self._process_online(
                    online_units, context_base, unit_states, originals, unit_map
                )
                results.update(online_result.results)
                completed.update(online_result.completed)
                failed.update(online_result.failed)
                skipped.update(online_result.skipped)
                safety_blocked.update(online_result.safety_blocked)
                validation_failed.update(online_result.validation_failed)
                screener_passed.update(online_result.screener_passed)
                fallback_used.update(online_result.fallback_used)
                # Aggregate stats
                total_attempts += online_result.total_attempts
                successful_attempts += online_result.successful_attempts
                splits_performed += online_result.splits_performed
                max_depth_reached = max(max_depth_reached, online_result.max_depth_reached)

            # Wait for batch results
            if batch_future:
                batch_result = batch_future.result()
                results.update(batch_result.results)
                completed.update(batch_result.completed)
                skipped.update(batch_result.skipped)
                # Use actual attempts from batch result (excludes pre_process skipped units)
                total_attempts += batch_result.total_attempts
                successful_attempts += batch_result.successful_attempts

                # Handle batch failures with online fallback
                batch_failed = batch_result.failed | batch_result.validation_failed

                if batch_failed:
                    logger.info(f"Batch had {len(batch_failed)} failures")

                    failed_units = [u for u in batch_units if u.id in batch_failed]

                    if len(failed_units) <= self._online_fallback_threshold:
                        # Use online fallback for small number of failures
                        logger.info(f"Using online fallback for {len(failed_units)} units")

                        # Update states: remove batch entries
                        for unit in failed_units:
                            state = unit_states[unit.id]
                            state.remove_batch_entries()

                        # Process via online
                        fallback_result = self._process_online(
                            failed_units, context_base, unit_states, originals, unit_map
                        )
                        results.update(fallback_result.results)
                        completed.update(fallback_result.completed)
                        failed.update(fallback_result.failed)
                        screener_passed.update(fallback_result.screener_passed)
                        fallback_used.update(fallback_result.fallback_used)
                        # Aggregate stats
                        total_attempts += fallback_result.total_attempts
                        successful_attempts += fallback_result.successful_attempts
                        splits_performed += fallback_result.splits_performed
                        max_depth_reached = max(max_depth_reached, fallback_result.max_depth_reached)
                    else:
                        # Too many failures - retry with new batch job
                        # Check if a different batch entry is available after failures
                        # Use first unit's state as representative
                        first_failed = failed_units[0]
                        first_state = unit_states[first_failed.id]
                        new_batch_entry = self._get_batch_entry_for_state(first_state)

                        if new_batch_entry and new_batch_entry != batch_entry:
                            logger.info(f"Retrying {len(failed_units)} units with different batch model: {new_batch_entry.model}")
                            retry_result = self._process_batch(
                                failed_units, new_batch_entry, context_base, originals, unit_states
                            )
                        else:
                            # No new batch entry available, still retry with same entry
                            # (this preserves original behavior for single-batch-entry chains)
                            logger.info(f"Retrying {len(failed_units)} units with batch job")
                            retry_result = self._process_batch(
                                failed_units, batch_entry, context_base, originals, unit_states
                            )
                        results.update(retry_result.results)
                        completed.update(retry_result.completed)
                        # Use actual attempts from retry result
                        total_attempts += retry_result.total_attempts
                        successful_attempts += retry_result.successful_attempts
                        # Remaining failures after retry go to online fallback
                        retry_failed = retry_result.failed | retry_result.validation_failed
                        if retry_failed:
                            retry_failed_units = [u for u in failed_units if u.id in retry_failed]
                            logger.info(f"Batch retry had {len(retry_failed_units)} failures, using online fallback")
                            for unit in retry_failed_units:
                                state = unit_states[unit.id]
                                state.remove_batch_entries()
                            final_fallback = self._process_online(
                                retry_failed_units, context_base, unit_states, originals, unit_map
                            )
                            results.update(final_fallback.results)
                            completed.update(final_fallback.completed)
                            failed.update(final_fallback.failed)
                            screener_passed.update(final_fallback.screener_passed)
                            fallback_used.update(final_fallback.fallback_used)
                            # Aggregate stats
                            total_attempts += final_fallback.total_attempts
                            successful_attempts += final_fallback.successful_attempts
                            splits_performed += final_fallback.splits_performed
                            max_depth_reached = max(max_depth_reached, final_fallback.max_depth_reached)

                validation_failed.update(batch_result.validation_failed)
                safety_blocked.update(batch_result.safety_blocked)

        finally:
            batch_executor.shutdown(wait=False)

        # Build stats
        duration = time.time() - start_time
        stats = {
            "duration_seconds": duration,
            "total_units": len(units),
            "batch_units": len(batch_units),
            "online_units": len(online_units),
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

    def _get_batch_entry(self) -> Optional[ChainEntry]:
        """Get the first batch entry from chain."""
        for entry in self._model_chain:
            if entry.mode == "batch":
                return entry
        return None

    def _get_batch_entry_for_state(self, state: UnitState) -> Optional[ChainEntry]:
        """
        Get the first available batch entry for a unit's current state.

        This is used for batch retry to get the next available batch model
        after the first batch model failed. The state's chain may have been
        modified by apply_effect to remove exhausted entries.

        Args:
            state: The UnitState to check

        Returns:
            The first batch ChainEntry from the unit's current chain, or None
        """
        # Use the state's get_batch_entry method which respects chain modifications
        return state.get_batch_entry()

    def _process_online(
        self,
        units: List[WorkUnit],
        context_base: Optional["ProcessContext"],
        unit_states: Dict[str, UnitState],
        originals: Dict[str, str],
        unit_map: Dict[str, WorkUnit],
    ) -> ExecutionResult:
        """
        Process units via online API with re-queuing.

        Key algorithm:
        1. Add all units to pending pool
        2. While pending or in-progress:
           a. Get ready units (dependencies satisfied)
           b. Handle aggregation units directly
           c. Submit others to thread pool
           d. Wait for any completion
           e. Handle result: success -> completed, failure -> maybe re-queue

        Args:
            units: Units to process
            context_base: Base context
            unit_states: Per-unit states
            originals: Original content map
            unit_map: Unit ID to WorkUnit map

        Returns:
            ExecutionResult
        """
        # Result collections
        results: Dict[str, str] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()
        skipped: Set[str] = set()
        safety_blocked: Set[str] = set()
        validation_failed: Set[str] = set()

        # Concurrent state
        pending: Set[str] = {u.id for u in units}
        in_progress: Set[str] = set()
        futures: Dict[Future, str] = {}

        # Stats
        total_requeued = 0
        total_attempts = 0
        successful_attempts = 0
        splits_performed = 0
        max_depth_reached = 0
        screener_passed: Set[str] = set()  # Keys that passed individual screener (skip batch)
        fallback_used: Set[str] = set()   # Keys that used longest fallback (need warning)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending or futures:
                # Get ready units (dependencies satisfied)
                ready_ids = self._get_ready_ids(
                    pending, completed, in_progress, unit_states
                )

                # Process ready units
                for unit_id in ready_ids:
                    pending.discard(unit_id)
                    state = unit_states[unit_id]

                    # Aggregation unit: aggregate children directly, no LLM call
                    if state.is_aggregation:
                        # All children must be in completed (checked by _get_ready_ids)
                        child_results = []
                        for child_id in sorted(state.children):
                            if child_id in results:
                                child_results.append(results[child_id])
                            else:
                                # Child failed - aggregation fails
                                failed.add(unit_id)
                                break
                        else:
                            # All children succeeded
                            aggregated = "\n\n".join(child_results)
                            results[unit_id] = aggregated
                            completed.add(unit_id)
                            logger.info(f"{unit_id}: Aggregated {len(state.children)} children")
                            # Remove .sub children from results/completed (internal detail)
                            for child_id in state.children:
                                results.pop(child_id, None)
                                completed.discard(child_id)
                        continue

                    # Normal unit: submit to thread pool
                    in_progress.add(unit_id)

                    # Get or create WorkUnit
                    unit = unit_map.get(unit_id)
                    if not unit:
                        # Virtual unit - create from state
                        # Derive file_key from unit_id (strip .sub/.part suffix)
                        file_key = unit_id
                        if '.sub' in file_key:
                            file_key = file_key.rsplit('.sub', 1)[0]
                        if '.part' in file_key:
                            file_key = file_key.rsplit('.part', 1)[0]
                        unit = WorkUnit(id=unit_id, file_key=file_key, content=state.content)
                        unit_map[unit_id] = unit
                        originals[unit_id] = state.content

                    context = self._build_context(
                        unit, context_base, completed, results, originals
                    )

                    future = pool.submit(
                        self._process_single,
                        unit, state, context, originals.get(unit_id, state.content)
                    )
                    futures[future] = unit_id

                # Check circuit breaker - abort on systemic network failure
                if self._network_circuit_broken:
                    logger.error(
                        f"Aborting: circuit breaker tripped. "
                        f"Marking {len(pending)} pending units as failed."
                    )
                    failed.update(pending)
                    pending.clear()
                    # Wait for in-flight futures to complete
                    for future in futures:
                        try:
                            future.result(timeout=5.0)
                        except Exception:
                            pass
                    futures.clear()
                    break

                # Check termination
                if not futures:
                    if pending:
                        # Deadlock - dependencies cannot be satisfied
                        logger.warning(f"{len(pending)} units have unsatisfied dependencies")
                        failed.update(pending)
                        pending.clear()
                    break

                # Wait for any completion
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                for future in done:
                    unit_id = futures.pop(future)
                    in_progress.discard(unit_id)
                    state = unit_states[unit_id]

                    try:
                        result = future.result()
                        total_attempts += 1  # Count every LLM attempt

                        if result.skipped:
                            skipped.add(unit_id)
                            if result.content is not None:
                                results[unit_id] = result.content
                            completed.add(unit_id)
                            logger.debug(f"{unit_id}: Skipped - {result.skip_reason}")

                        elif result.success:
                            successful_attempts += 1
                            results[unit_id] = result.content
                            completed.add(unit_id)

                            # REALTIME SAVE: Save file BEFORE updating tracker
                            # This ensures atomicity: if interrupted after save,
                            # tracker will be updated on next run
                            if self._persistence and not is_sub_key(unit_id):
                                try:
                                    self._persistence.save_raw(unit_id, result.content)
                                except Exception as e:
                                    logger.warning(f"{unit_id}: Failed to save raw: {e}")

                            # Track screener-passed (context_ready = passed individual screener)
                            if result.context_ready:
                                screener_passed.add(unit_id)
                                # Cache for context injection
                                if self._context_injector:
                                    self._context_injector.cache_completed(
                                        unit_id, originals.get(unit_id, ""), result.content
                                    )

                            logger.info(f"{unit_id}: Completed successfully")

                        else:
                            # Failed - handle with state update and maybe re-queue
                            requeued, split_depth = self._handle_failure(
                                unit_id, result, state, unit_states,
                                pending, completed, failed, safety_blocked, validation_failed,
                                results, fallback_used
                            )
                            if requeued:
                                total_requeued += 1
                            if split_depth > 0:
                                splits_performed += 1
                                max_depth_reached = max(max_depth_reached, split_depth)

                    except Exception as e:
                        # Add traceback logging for debugging (P1-5 fix)
                        import traceback
                        logger.error(f"{unit_id}: Unexpected error: {e}")
                        logger.debug(f"{unit_id}: Traceback:\n{traceback.format_exc()}")
                        failed.add(unit_id)

        return ExecutionResult(
            results=results,
            completed=completed,
            failed=failed,
            skipped=skipped,
            safety_blocked=safety_blocked,
            validation_failed=validation_failed,
            screener_passed=screener_passed,
            fallback_used=fallback_used,
            stats={"total_requeued": total_requeued},
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
            splits_performed=splits_performed,
            max_depth_reached=max_depth_reached,
        )

    def _process_single(
        self,
        unit: WorkUnit,
        state: UnitState,
        context: "ProcessContext",
        original: str,
    ) -> ProcessResult:
        """
        Process a single unit (runs in thread pool).

        Args:
            unit: Unit to process
            state: Unit's state
            context: Processing context
            original: Original content

        Returns:
            ProcessResult
        """
        from .._protocol import ProcessContext as PC

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

            # Call LLM
            response = self._llm_client.generate(
                prompt=prompt,
                model_configs=[entry.to_dict()],
                operation_name=f"{self._processor.name}:{unit.id}",
            )

            # Clean response
            cleaned = self._processor.clean_response(response)

            # Hooks: transform + validate (before processor.post_process)
            chapter_type = getattr(context, 'chapter_type', "") or ""
            transformed, hook_result = self._hooks.post_process(
                unit.id, original, cleaned, chapter_type, context
            )

            # processor.post_process is called AFTER validation (per protocol)
            # Only apply if validation passed, otherwise record raw transformed for fallback
            if hook_result.accepted:
                final = self._processor.post_process(transformed, context)
            else:
                final = transformed

            # Record attempt for longest fallback
            state.record_attempt(final)

            # ATOMICITY FIX: Save file BEFORE recording to tracker
            # This ensures that if we crash after save, tracker can be fixed on next run
            # If we crash after tracker but before save, the file is lost permanently
            save_succeeded = True
            if hook_result.accepted and self._persistence and not is_sub_key(unit.id):
                try:
                    self._persistence.save_raw(unit.id, final)
                except Exception as e:
                    logger.warning(f"{unit.id}: Failed to save raw in _process_single: {e}")
                    save_succeeded = False

            # Only record to tracker if save succeeded (atomicity guarantee)
            # Skip .sub virtual units - they should not be tracked
            if self._tracker and not is_sub_key(unit.id):
                from ..tracking import AttemptRecord
                # Use rejection_reason if available (P1-6 fix)
                # ATOMICITY: Only record "completed" if save succeeded
                is_completed = hook_result.accepted and save_succeeded
                error_msg = hook_result.rejection_reason if not hook_result.accepted else None
                if not save_succeeded:
                    error_msg = "File save failed"
                attempt = AttemptRecord(
                    timestamp=time.time(),
                    status="completed" if is_completed else "failed",
                    model=f"{entry.provider}/{entry.model}",
                    error_type="validation" if not hook_result.accepted else ("io_error" if not save_succeeded else None),
                    error_message=error_msg,
                )
                self._tracker.record_attempt(unit.id, attempt)

            if hook_result.accepted and save_succeeded:
                return ProcessResult(
                    success=True,
                    content=final,
                    context_ready=hook_result.context_ready,
                )
            else:
                # Include rejection reason or save failure in error for better debugging
                if not save_succeeded:
                    error_detail = "File save failed"
                else:
                    error_detail = hook_result.rejection_reason or "Validation failed"
                return ProcessResult(
                    success=False,
                    error=Exception(error_detail),
                )

        except Exception as e:
            # Add traceback logging for debugging (P1-5 fix)
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
    ) -> Tuple[bool, int]:
        """
        Handle a failed processing result.

        Returns:
            Tuple of (requeued, split_depth):
            - requeued: True if unit was re-queued
            - split_depth: Depth of split if split occurred, 0 otherwise
        """
        # Classify error
        error_type, effect = self._hooks.classify_error(result.error)

        # Get error message for logging and tracking
        error_msg = str(result.error)[:500] if result.error else "Unknown error"

        # Network circuit breaker: track consecutive network failures
        network_errors = (ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT)
        if error_type in network_errors:
            self._consecutive_network_failures += 1
            if self._consecutive_network_failures >= self._network_circuit_breaker_threshold:
                if not self._network_circuit_broken:
                    self._network_circuit_broken = True
                    logger.error(
                        f"CIRCUIT BREAKER: {self._consecutive_network_failures} consecutive "
                        f"network failures, systemic network issue detected"
                    )
        else:
            # Reset counter on non-network error (network is working)
            self._consecutive_network_failures = 0

        # Get current entry before applying effect
        current_entry = state.get_current_entry()
        model_str = f"{current_entry.provider}/{current_entry.model}" if current_entry else "unknown"

        # Apply effect to state
        state.apply_effect(effect, current_entry)

        # Track by error type
        if error_type == ErrorType.SAFETY:
            safety_blocked.add(unit_id)
        elif error_type == ErrorType.VALIDATION:
            validation_failed.add(unit_id)

        # Record to tracker with full error details (P0-2 fix)
        if self._tracker and not is_sub_key(unit_id):
            from ..tracking import AttemptRecord
            attempt = AttemptRecord(
                timestamp=time.time(),
                status="failed",
                model=model_str,
                error_type=error_type.value,
                error_message=error_msg,
            )
            self._tracker.record_attempt(unit_id, attempt)

        # Check if can retry (use effect.quota_type, not error_type!)
        if state.can_retry(effect.quota_type):
            pending.add(unit_id)  # Re-queue!
            # Log with actual error message for debuggability
            logger.info(
                f"{unit_id}: {error_type.value}, re-queued "
                f"(quota: {state.total_quota}, chain: {len(state.chain)}) - {error_msg[:100]}"
            )
            return (True, 0)

        # Network-class errors NEVER trigger split (splitting won't fix network issues)
        # These errors should fail fast after chain exhaustion
        network_errors = (ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT)
        if error_type in network_errors:
            logger.warning(
                f"{unit_id}: {error_type.value} exhausted chain, skipping split - {error_msg[:200]}"
            )
            # Go directly to longest fallback or fail
        else:
            # Content-related errors can try splitting (TRUNCATION, VALIDATION, etc.)
            if self._splitter:
                # Use model-specific split threshold (P0-1 fix)
                model_name = current_entry.model if current_entry else None
                split_threshold = self._get_split_threshold(model_name)
                split_success, split_depth = handle_split(
                    unit_id, unit_states, pending, self._splitter, split_threshold
                )
                if split_success:
                    # Split succeeded - parent will aggregate when children complete
                    return (True, split_depth)

        # Cannot split or network error - try longest fallback
        longest = state.get_longest()
        if longest:
            results[unit_id] = longest
            completed.add(unit_id)  # Mark as completed when using fallback
            fallback_used.add(unit_id)  # Track for warning header in output
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
        """
        Get unit IDs that are ready (dependencies satisfied).

        Dependency rules:
        - children: UNCONDITIONAL - aggregation units wait for all children
        - depends_on: SEQUENTIAL ONLY - context injection dependencies
        - part ordering: SEQUENTIAL ONLY - chapter_5.part1 waits for part0
        """
        ready = []
        is_sequential = self._context_injector and self._context_injector.is_sequential

        for unit_id in pending:
            if unit_id in in_progress:
                continue

            state = unit_states.get(unit_id)
            if state:
                # Check children (UNCONDITIONAL - parallel and sequential)
                if state.children and not all(c in completed for c in state.children):
                    continue

                # Sequential mode: check depends_on (context injection)
                if is_sequential:
                    if state.depends_on and not state.depends_on.issubset(completed):
                        continue

            # Sequential mode: check part ordering (chapter_5.part1 depends on part0)
            if is_sequential and '.part' in unit_id and '.sub' not in unit_id:
                parts = unit_id.rsplit('.part', 1)
                if len(parts) == 2:
                    base, part_num_str = parts
                    try:
                        part_num = int(part_num_str)
                        if part_num > 0:  # part1 depends on part0
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
            # Create context from WorkUnit to ensure all metadata is propagated
            context_base = ProcessContext.from_work_unit(unit)
        else:
            # Merge WorkUnit fields into existing context_base
            # This ensures part_index, chapter_type etc. are correct for this specific unit
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

        # Get previous part context
        prev_context = self._context_injector.get_context_for_unit(
            unit, results, originals
        )

        if prev_context:
            prev_original, prev_processed = prev_context
            return context_base.with_previous_context(prev_original, prev_processed)

        return context_base

    def _process_batch(
        self,
        units: List[WorkUnit],
        batch_entry: ChainEntry,
        context_base: Optional["ProcessContext"],
        originals: Dict[str, str],
        unit_states: Dict[str, UnitState],
    ) -> ExecutionResult:
        """
        Process units via batch API.

        Uses the actual GeminiBatchClient interface:
        - submit(requests: List[BatchRequest]) -> job_name
        - get_status(job_name) -> BatchJobInfo
        - get_results(job_name) -> List[BatchResponse]

        Args:
            units: Units to process
            batch_entry: Batch chain entry to use
            context_base: Base context
            originals: Original content map

        Returns:
            ExecutionResult
        """
        from .._protocol import ProcessContext
        from ...utils.batch_utils import BatchRequest, BatchJobState

        results: Dict[str, str] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()
        skipped: Set[str] = set()
        validation_failed: Set[str] = set()
        safety_blocked: Set[str] = set()

        try:
            # Build requests (with pre-processing check)
            batch_requests: List[BatchRequest] = []
            unit_contexts: Dict[str, "ProcessContext"] = {}
            units_to_process: Dict[str, WorkUnit] = {}  # Map key -> unit for result processing

            for unit in units:
                # Build context with WorkUnit metadata (same as online path)
                if context_base is None:
                    context = ProcessContext.from_work_unit(unit)
                else:
                    # Merge WorkUnit fields into context_base
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
                unit_contexts[unit.id] = context

                # Pre-processing check (same as online path)
                pre_result = self._hooks.pre_process(unit.id, unit.content, context)
                if not pre_result.should_process:
                    skipped.add(unit.id)
                    if pre_result.fallback_result is not None:
                        results[unit.id] = pre_result.fallback_result
                        completed.add(unit.id)
                    continue

                prompt = self._processor.build_prompt(unit.content, context)

                # Create BatchRequest with proper format
                batch_requests.append(BatchRequest(
                    key=unit.id,
                    contents=[{"parts": [{"text": prompt}], "role": "user"}],
                ))
                units_to_process[unit.id] = unit

            # Early return if all units were skipped
            if not batch_requests:
                return ExecutionResult(
                    results=results,
                    completed=completed,
                    failed=failed,
                    skipped=skipped,
                )

            # Submit batch job
            job_name = self._batch_client.submit(batch_requests)
            logger.info(f"Submitted batch job: {job_name} with {len(batch_requests)} units")

            # Poll for completion
            while True:
                job_info = self._batch_client.get_status(job_name)

                if job_info.state == BatchJobState.SUCCEEDED:
                    break
                elif job_info.state == BatchJobState.FAILED:
                    logger.error(f"Batch job failed: {job_info.error}")

                    # Classify the job-level error (wrap in Exception for classify_error)
                    error_msg = str(job_info.error) if job_info.error else "batch job failed"
                    error = Exception(error_msg)
                    error_type, effect = self._hooks.classify_error(error)

                    # For job-level batch failures, we should advance the chain
                    # to try a different batch model on retry. Force remove_current_model=True.
                    # This ensures _get_batch_entry_for_state returns the next batch model.
                    from ..hooks import ErrorEffect
                    batch_fail_effect = ErrorEffect(
                        remove_current_model=True,  # Force advance to next model
                        remove_provider=effect.remove_provider,
                        remove_all_batch=effect.remove_all_batch,
                        quota_type=effect.quota_type,
                    )

                    # Apply effect to all units in this batch
                    for unit_id in units_to_process.keys():
                        state = unit_states.get(unit_id)
                        if state:
                            state.apply_effect(batch_fail_effect, batch_entry)

                        # Track by error type
                        if error_type == ErrorType.SAFETY:
                            safety_blocked.add(unit_id)

                        # Record to tracker for persistent fallback
                        if self._tracker and not is_sub_key(unit_id):
                            from ..tracking import AttemptRecord
                            attempt = AttemptRecord(
                                timestamp=time.time(),
                                status="failed",
                                model=f"{batch_entry.provider}/{batch_entry.model}",
                                error_type=error_type.value,
                                error_message=error_msg[:200] if error_msg else None,
                            )
                            self._tracker.record_attempt(unit_id, attempt)

                    failed.update(units_to_process.keys())
                    return ExecutionResult(
                        results=results,
                        completed=completed,
                        failed=failed,
                        skipped=skipped,
                        safety_blocked=safety_blocked,
                    )
                elif job_info.state in (BatchJobState.CANCELLED, BatchJobState.EXPIRED):
                    logger.error(f"Batch job {job_info.state.name}: {job_info.error}")

                    # Classify the job-level error (wrap in Exception for classify_error)
                    error_msg = str(job_info.error) if job_info.error else f"batch job {job_info.state.name}"
                    error = Exception(error_msg)
                    error_type, effect = self._hooks.classify_error(error)

                    # For job-level batch failures, we should advance the chain
                    # to try a different batch model on retry. Force remove_current_model=True.
                    batch_fail_effect = ErrorEffect(
                        remove_current_model=True,  # Force advance to next model
                        remove_provider=effect.remove_provider,
                        remove_all_batch=effect.remove_all_batch,
                        quota_type=effect.quota_type,
                    )

                    # Apply effect to all units
                    for unit_id in units_to_process.keys():
                        state = unit_states.get(unit_id)
                        if state:
                            state.apply_effect(batch_fail_effect, batch_entry)

                        if error_type == ErrorType.SAFETY:
                            safety_blocked.add(unit_id)

                        if self._tracker and not is_sub_key(unit_id):
                            from ..tracking import AttemptRecord
                            attempt = AttemptRecord(
                                timestamp=time.time(),
                                status="failed",
                                model=f"{batch_entry.provider}/{batch_entry.model}",
                                error_type=error_type.value,
                                error_message=error_msg[:200] if error_msg else None,
                            )
                            self._tracker.record_attempt(unit_id, attempt)

                    failed.update(units_to_process.keys())
                    return ExecutionResult(
                        results=results,
                        completed=completed,
                        failed=failed,
                        skipped=skipped,
                        safety_blocked=safety_blocked,
                    )

                logger.debug(f"Batch job {job_name}: {job_info.state.name}")
                time.sleep(self._batch_poll_interval)

            # Get results (returns List[BatchResponse])
            batch_responses = self._batch_client.get_results(job_name)

            # Build response map: key -> text, and error map for failed units
            response_map: Dict[str, str] = {}
            batch_response_errors: Dict[str, str] = {}
            for resp in batch_responses:
                if resp.text is not None:
                    response_map[resp.key] = resp.text
                elif resp.error:
                    batch_response_errors[resp.key] = str(resp.error)
                    logger.warning(f"Batch response error for {resp.key}: {resp.error}")

            # Process each result
            for unit_id, unit in units_to_process.items():
                state = unit_states.get(unit_id)

                if unit_id not in response_map:
                    # Per-unit failure - classify and apply effect
                    error_msg = batch_response_errors.get(unit_id, "Unknown batch error")
                    error = Exception(error_msg)
                    error_type, effect = self._hooks.classify_error(error)

                    if state:
                        state.apply_effect(effect, batch_entry)

                    # Track by error type
                    if error_type == ErrorType.SAFETY:
                        safety_blocked.add(unit_id)

                    # Record to tracker
                    if self._tracker and not is_sub_key(unit_id):
                        from ..tracking import AttemptRecord
                        attempt = AttemptRecord(
                            timestamp=time.time(),
                            status="failed",
                            model=f"{batch_entry.provider}/{batch_entry.model}",
                            error_type=error_type.value,
                            error_message=error_msg[:200] if error_msg else None,
                        )
                        self._tracker.record_attempt(unit_id, attempt)

                    failed.add(unit_id)
                    continue

                response_text = response_map[unit_id]
                original = originals.get(unit_id, unit.content)
                context = unit_contexts.get(unit_id, context_base)

                # Clean response
                cleaned = self._processor.clean_response(response_text)

                # Apply hooks: transform + validate (BEFORE processor.post_process)
                chapter_type = unit.chapter_type or ''
                transformed, hook_result = self._hooks.post_process(
                    unit_id, original, cleaned, chapter_type, context
                )

                if hook_result.accepted:
                    # processor.post_process is called AFTER validation (per protocol)
                    final = self._processor.post_process(transformed, context)
                    results[unit_id] = final
                    completed.add(unit_id)

                    # Record successful attempt for longest fallback
                    if state:
                        state.record_attempt(final)

                    # Record to tracker
                    if self._tracker and not is_sub_key(unit_id):
                        from ..tracking import AttemptRecord
                        attempt = AttemptRecord(
                            timestamp=time.time(),
                            status="completed",
                            model=f"{batch_entry.provider}/{batch_entry.model}",
                        )
                        self._tracker.record_attempt(unit_id, attempt)
                else:
                    # Validation failed
                    # 1. Classify as validation error and get effect
                    error_type = ErrorType.VALIDATION
                    effect = self._hooks._error_classifier.get_effect(error_type)

                    # 2. Apply effect to decrement quota (CRITICAL: was missing before)
                    if state:
                        state.apply_effect(effect, batch_entry)

                    # 3. Record attempt for longest fallback
                    if state:
                        state.record_attempt(transformed)

                    # 4. Record to tracker with rejection reason (P1-6 fix)
                    if self._tracker and not is_sub_key(unit_id):
                        from ..tracking import AttemptRecord
                        error_msg = hook_result.rejection_reason or "Validation rejected"
                        attempt = AttemptRecord(
                            timestamp=time.time(),
                            status="failed",
                            model=f"{batch_entry.provider}/{batch_entry.model}",
                            error_type="validation",
                            error_message=error_msg,
                        )
                        self._tracker.record_attempt(unit_id, attempt)

                    validation_failed.add(unit_id)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            failed.update(u.id for u in units if u.id not in skipped and u.id not in completed)

        # Calculate actual attempts (only units that were submitted to batch)
        actual_attempts = len(batch_requests) if batch_requests else 0
        # Calculate successful attempts: completed units minus skipped units
        # (skipped units are marked as completed but don't count as batch attempts)
        actual_successes = len(completed - skipped)

        return ExecutionResult(
            results=results,
            completed=completed,
            failed=failed,
            skipped=skipped,
            safety_blocked=safety_blocked,
            validation_failed=validation_failed,
            total_attempts=actual_attempts,
            successful_attempts=actual_successes,
        )
