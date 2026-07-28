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
from contextlib import contextmanager
from loguru import logger
import hashlib
import json
import re
import time
import signal
import threading
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


def _batch_request_sha256(
    provider: str,
    model: str,
    unit_ids: List[str],
    requests: List[Any],
    skipped_identity: List[Dict[str, Any]],
) -> str:
    """Fingerprint every value that determines a submitted batch request."""
    payload = {
        "provider": provider,
        "model": model,
        "unit_ids": sorted(unit_ids),
        "requests": [request.to_dict() for request in requests],
        "skipped": skipped_identity,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


from ._protocol import WorkUnit, ChainEntry, ExecutionResult, ProcessResult
from .state import UnitState, QuotaConfig, create_unit_state
from .batch_state import (
    BatchRunLock,
    BatchStateConflictError,
    MegaUnitState,
    get_mega_unit_id,
    is_safety_error,
)
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
    fallback_chain: Optional[List[ChainEntry]] = None,
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
        fallback_chain: Chain to use when the failed unit exhausted its
            current chain before splitting. This lets a model retry smaller
            chunks after a truncation failure.

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

    # A truncation failure can remove the only model before splitting. Preserve
    # that model for smaller children; otherwise they inherit an empty chain.
    child_chain = list(state.chain)
    if not child_chain and fallback_chain:
        # A recovered split may contain fewer units than the batch threshold.
        # Run a preserved batch-only model through the online dispatch path so
        # small splits do not immediately lose their only executable entry.
        child_chain = [
            ChainEntry(
                provider=entry.provider,
                model=entry.model,
                mode="online" if entry.mode == "batch" else entry.mode,
                retries=entry.retries,
            )
            for entry in fallback_chain
        ]

    # Calculate children's quota (half, integer division)
    child_total_quota = state.total_quota // 2
    child_quotas_template = {k: v // 2 for k, v in state.quotas.items()}

    # Create virtual children
    child_ids = []
    for i, content in enumerate(child_contents):
        child_id = f"{unit_id}.sub{i}"
        child_ids.append(child_id)

        unit_states[child_id] = UnitState(
            chain=child_chain.copy(),
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

        # Signal handlers are installed only while execute() owns the batch
        # run lock, then restored even after a normal completion.
        self._original_sigint = None
        self._original_sigterm = None
        self._batch_lock_depth = 0
        self._batch_lifecycle_lock = threading.RLock()

        # Circuit breaker state
        self._consecutive_network_failures = 0
        self._network_circuit_broken = False

    def _install_signal_handlers(self) -> None:
        """Install resumable-interrupt handlers for an active batch run."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _restore_signal_handlers(self) -> None:
        """Restore handlers that were active before batch execution."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        self._original_sigint = None
        self._original_sigterm = None

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM - keep batch jobs running for resume."""
        logger.warning("Interrupt received. Batch jobs will continue running in background.")
        logger.info("Use --resume to continue and retrieve results.")
        logger.info("Use 'pdf2epub cancel-batch' to cancel batch jobs.")

        raise KeyboardInterrupt("Interrupted. Use --resume to continue or 'pdf2epub cancel-batch' to cancel.")

    @contextmanager
    def batch_run_lock(self):
        """Own one stage from resume inspection through final promotion."""
        with self._batch_lifecycle_lock:
            if not self._batch_client or not self._batch_states_dir:
                yield
                return
            if self._batch_lock_depth > 0:
                self._batch_lock_depth += 1
                try:
                    yield
                finally:
                    self._batch_lock_depth -= 1
                return
            with BatchRunLock(self._batch_states_dir):
                self._batch_lock_depth = 1
                try:
                    yield
                finally:
                    self._batch_lock_depth = 0

    def _finalize_batch_job(self, unit_ids: List[str]) -> None:
        """Remove provider artifacts only after results have been handled.

        ``_process_batch_as_unit`` intentionally leaves successful state and
        remote output in place. The pipeline calls this method only after
        every returned result has passed validation, retry routing, disk
        persistence, and promotion. A crash before that point therefore
        remains resumable.
        """
        if not self._batch_states_dir:
            return

        state_path = (
            self._batch_states_dir
            / f"{get_mega_unit_id(unit_ids)}.json"
        )
        self._finalize_batch_state_path(state_path)

    def _finalize_batch_state_path(self, state_path: Path) -> None:
        """Finalize one exact persisted state path."""
        if not state_path.exists():
            return
        state = MegaUnitState.load(state_path)
        if state is None:
            raise BatchStateConflictError(
                f"Batch state {state_path} became unreadable before cleanup"
            )
        self._validate_batch_state_for_cleanup(state_path, state)

        # Persist the tombstone before touching remote artifacts. Cleanup is
        # intentionally idempotent, so a crash after remote deletion but
        # before local unlink can safely finish on the next run.
        state.job_state = "FINALIZING"
        state.save(state_path)

        if (
            state.job_name
            and hasattr(self._batch_client, "cleanup_job_artifacts")
        ):
            self._batch_client.cleanup_job_artifacts(state.job_name)
        state_path.unlink()

    def _validate_batch_state_for_cleanup(
        self,
        state_path: Path,
        state: MegaUnitState,
    ) -> None:
        """Require complete identity before deleting provider or local state."""
        unit_ids_valid = (
            isinstance(state.unit_ids, list)
            and bool(state.unit_ids)
            and all(
                isinstance(unit_id, str) and bool(unit_id)
                for unit_id in state.unit_ids
            )
            and len(set(state.unit_ids)) == len(state.unit_ids)
        )
        processing_keys_valid = (
            isinstance(state.processing_keys, list)
            and bool(state.processing_keys)
            and all(
                isinstance(key, str) and bool(key)
                for key in state.processing_keys
            )
            and unit_ids_valid
            and set(state.processing_keys).issubset(set(state.unit_ids))
        )
        if (
            not isinstance(state.job_name, str)
            or not state.job_name.strip()
            or not isinstance(state.provider, str)
            or not state.provider
            or not isinstance(state.model, str)
            or not state.model
            or not unit_ids_valid
            or not processing_keys_valid
            or not isinstance(state.request_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", state.request_sha256)
        ):
            raise BatchStateConflictError(
                f"Batch cleanup state {state_path} has incomplete or invalid "
                "job identity; state was retained"
            )
        expected_path = (
            self._batch_states_dir
            / f"{get_mega_unit_id(state.unit_ids)}.json"
        )
        if state_path != expected_path:
            raise BatchStateConflictError(
                f"Batch cleanup state filename {state_path.name} does not "
                "match its unit membership"
            )
        if not any(
            entry.mode == "batch"
            and entry.provider == state.provider
            and entry.model == state.model
            for entry in self._model_chain
        ):
            raise BatchStateConflictError(
                f"Batch cleanup state {state_path} uses "
                f"{state.provider}/{state.model}, which is absent from the "
                "current model chain"
            )
        client_provider = getattr(
            self._batch_client,
            "batch_provider",
            None,
        )
        if client_provider is not None and client_provider != state.provider:
            raise BatchStateConflictError(
                f"Batch cleanup state provider {state.provider!r} does not "
                f"match client provider {client_provider!r}"
            )
        client_model = getattr(self._batch_client, "model", None)
        if client_model is not None and client_model != state.model:
            raise BatchStateConflictError(
                f"Batch cleanup state model {state.model!r} does not match "
                f"client model {client_model!r}"
            )

    def _recover_finalizing_batches_locked(self) -> None:
        """Finish idempotent cleanup left between remote and local deletion."""
        if not self._batch_states_dir:
            return
        for state_path in sorted(
            self._batch_states_dir.glob("batch_*.json")
        ):
            state = MegaUnitState.load(state_path)
            if state is None:
                raise BatchStateConflictError(
                    f"Batch state {state_path} is unreadable or corrupted"
                )
            if state.job_state != "FINALIZING":
                continue
            self._validate_batch_state_for_cleanup(state_path, state)
            if hasattr(self._batch_client, "cleanup_job_artifacts"):
                self._batch_client.cleanup_job_artifacts(state.job_name)
            state_path.unlink()

    def recover_finalizing_batches(self) -> None:
        """Finish cleanup tombstones before pipeline resume filtering."""
        if not self._batch_client or not self._batch_states_dir:
            return
        with self.batch_run_lock():
            self._recover_finalizing_batches_locked()

    def get_resumable_unit_ids(self) -> Set[str]:
        """Return exact mega-unit membership needed for safe resume."""
        if not self._batch_states_dir:
            return set()
        unit_ids: Set[str] = set()
        with self.batch_run_lock():
            for state_path in sorted(
                self._batch_states_dir.glob("batch_*.json")
            ):
                state = MegaUnitState.load(state_path)
                if state is None:
                    raise BatchStateConflictError(
                        f"Batch state {state_path} is unreadable or corrupted"
                    )
                if state.job_state == "FINALIZING":
                    continue
                if not state.unit_ids:
                    raise BatchStateConflictError(
                        f"Batch state {state_path} has no persisted unit "
                        "membership and cannot be resumed safely"
                    )
                unit_ids.update(state.unit_ids)
        return unit_ids

    def finalize_batch_jobs(
        self,
        batch_jobs: List[List[str]],
    ) -> None:
        """Accept completed jobs after pipeline validation and promotion."""
        if not batch_jobs or not self._batch_states_dir:
            return
        with self.batch_run_lock():
            for unit_ids in batch_jobs:
                self._finalize_batch_job(unit_ids)

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
        initial_failures: Optional[Dict[str, Exception]] = None,
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
            initial_failures: Failures reported by an upstream validator. They
                enter the same retry/split state machine as model failures.

        Returns:
            ExecutionResult with all outcomes
        """
        if not units:
            return ExecutionResult()

        if not self._batch_client or not self._batch_states_dir:
            return self._execute(
                units,
                context_base,
                resume_batch,
                initial_failures,
            )

        with self.batch_run_lock():
            self._recover_finalizing_batches_locked()
            existing_states = sorted(self._batch_states_dir.glob("batch_*.json"))
            if existing_states and not resume_batch:
                state_names = ", ".join(path.name for path in existing_states[:3])
                if len(existing_states) > 3:
                    state_names += f", and {len(existing_states) - 3} more"
                raise RuntimeError(
                    "Existing batch state found "
                    f"({state_names}). Use --resume to inspect/resume it, or "
                    "'pdf2epub cancel-batch' to cancel it explicitly."
                )

            self._install_signal_handlers()
            try:
                return self._execute(
                    units,
                    context_base,
                    resume_batch,
                    initial_failures,
                )
            finally:
                self._restore_signal_handlers()

    def _execute(
        self,
        units: List[WorkUnit],
        context_base: Optional["ProcessContext"] = None,
        resume_batch: bool = False,
        initial_failures: Optional[Dict[str, Exception]] = None,
    ) -> ExecutionResult:
        """Run the unified executor after batch ownership checks."""

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
        batch_jobs_to_finalize: List[List[str]] = []

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

        resume_groups: List[List[str]] = []
        if resume_batch and self._batch_states_dir:
            claimed_ids: Set[str] = set()
            for state_path in sorted(
                self._batch_states_dir.glob("batch_*.json")
            ):
                state = MegaUnitState.load(state_path)
                if state is None:
                    raise BatchStateConflictError(
                        f"Batch state {state_path} is unreadable or corrupted"
                    )
                if state.job_state == "FINALIZING":
                    continue
                if not state.unit_ids:
                    raise BatchStateConflictError(
                        f"Batch state {state_path} has no persisted unit "
                        "membership and cannot be resumed safely"
                    )
                group = sorted(state.unit_ids)
                expected_path = (
                    self._batch_states_dir
                    / f"{get_mega_unit_id(group)}.json"
                )
                if state_path != expected_path:
                    raise BatchStateConflictError(
                        f"Batch state filename {state_path.name} does not "
                        "match its persisted unit membership"
                    )
                missing = set(group) - set(unit_states)
                if missing:
                    raise BatchStateConflictError(
                        f"Resume input is missing persisted units: "
                        f"{sorted(missing)}"
                    )
                overlap = claimed_ids & set(group)
                if overlap:
                    raise BatchStateConflictError(
                        "Persisted batch states overlap on units "
                        f"{sorted(overlap)}. Refusing to guess retry order."
                    )
                claimed_ids.update(group)

                for uid in group:
                    chain = unit_states[uid].chain
                    matching_index = next(
                        (
                            index
                            for index, entry in enumerate(chain)
                            if entry.mode == "batch"
                            and entry.provider == state.provider
                            and entry.model == state.model
                        ),
                        None,
                    )
                    if matching_index is None:
                        raise BatchStateConflictError(
                            f"Persisted batch model {state.provider}/"
                            f"{state.model} is absent from the current chain "
                            f"for {uid!r}"
                        )
                    unit_states[uid].chain = list(
                        chain[matching_index:]
                    )
                resume_groups.append(group)

        # Feed failures from downstream validators back into the same state
        # machine used for model failures. This preserves failure typing and
        # keeps retry/split behavior centralized in the Executor.
        if initial_failures:
            for uid, error in initial_failures.items():
                if uid not in unit_states:
                    logger.warning(f"Ignoring initial failure for unknown unit {uid}")
                    continue

                pending.discard(uid)
                state = unit_states[uid]
                current_entry = state.get_current_entry()
                model_str = (
                    f"{current_entry.provider}/{current_entry.model}"
                    if current_entry else "validator"
                )
                _, split_depth = self._handle_failure(
                    uid,
                    ProcessResult(success=False, error=error),
                    state,
                    unit_states,
                    pending,
                    completed,
                    failed,
                    safety_blocked,
                    validation_failed,
                    results,
                    fallback_used,
                    model_str,
                )
                if split_depth > 0:
                    splits_performed += 1
                    max_depth_reached = max(max_depth_reached, split_depth)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for group in resume_groups:
                pending.difference_update(group)
                in_progress.update(group)
                future = pool.submit(
                    self._process_batch_as_unit,
                    group,
                    unit_states,
                    unit_map,
                    context_base,
                    originals,
                    True,
                )
                batch_futures[future] = group

            while pending or futures or batch_futures:
                # 1. Get ready units
                ready_ids = self._get_ready_ids(pending, completed, in_progress, unit_states)
                logger.debug(f"[EXECUTOR] pending={len(pending)} in_progress={len(in_progress)} completed={len(completed)} ready={len(ready_ids)}")

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
                    has_batch_only_unit = any(
                        not unit_states[uid].has_online_available()
                        for uid in batch_queue
                    )
                    if len(batch_queue) >= batch_threshold or has_batch_only_unit:
                        # Submit as mega unit
                        in_progress.update(batch_queue)
                        future = pool.submit(
                            self._process_batch_as_unit,
                            batch_queue,
                            unit_states,
                            unit_map,
                            context_base,
                            originals,
                            False,
                        )
                        batch_futures[future] = batch_queue
                    else:
                        # Too few and every unit has an online fallback.
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
                                    if self._saver:
                                        try:
                                            with self._saver.attempt(uid, "skipped") as attempt:
                                                saved = attempt.skip(
                                                    result.skip_reason or "pre-process",
                                                    result.content,
                                                )
                                        except Exception as exc:
                                            raise BatchStateConflictError(
                                                f"{uid}: exception while persisting "
                                                "skipped batch result; batch state "
                                                "was retained"
                                            ) from exc
                                        if not saved:
                                            raise BatchStateConflictError(
                                                f"{uid}: failed to persist skipped batch "
                                                "result; batch state was retained"
                                            )
                                    skipped.add(uid)
                                    if result.content is not None:
                                        results[uid] = result.content
                                    completed.add(uid)

                                elif result.success:
                                    if self._saver:
                                        try:
                                            with self._saver.attempt(uid, model_str) as attempt:
                                                saved = attempt.success(
                                                    result.content,
                                                    output_tokens=result.output_tokens,
                                                    context_ready=result.context_ready,
                                                )
                                        except Exception as exc:
                                            raise BatchStateConflictError(
                                                f"{uid}: exception while persisting "
                                                "batch result; batch state was "
                                                "retained"
                                            ) from exc
                                        if not saved:
                                            raise BatchStateConflictError(
                                                f"{uid}: failed to persist batch result; "
                                                "batch state was retained"
                                            )
                                    successful_attempts += 1
                                    results[uid] = result.content
                                    completed.add(uid)
                                    if result.context_ready:
                                        screener_passed.add(uid)

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

                            batch_jobs_to_finalize.append(list(batch_unit_ids))

                        except BatchStateConflictError:
                            raise
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
                                if self._saver:
                                    with self._saver.attempt(uid, "skipped") as attempt:
                                        saved = attempt.skip(
                                            result.skip_reason or "pre-process",
                                            result.content,
                                        )
                                    if not saved:
                                        failed.add(uid)
                                        continue
                                skipped.add(uid)
                                if result.content is not None:
                                    results[uid] = result.content
                                completed.add(uid)

                            elif result.success:
                                if self._saver:
                                    with self._saver.attempt(uid, model_str) as attempt:
                                        saved = attempt.success(
                                            result.content,
                                            output_tokens=result.output_tokens,
                                            duration_seconds=result.duration_seconds,
                                            context_ready=result.context_ready,
                                        )
                                    if not saved:
                                        failed.add(uid)
                                        continue
                                successful_attempts += 1
                                results[uid] = result.content
                                completed.add(uid)
                                if result.context_ready:
                                    screener_passed.add(uid)
                                    if self._context_injector:
                                        self._context_injector.cache_completed(
                                            uid, originals.get(uid, ""), result.content
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
            batch_jobs=batch_jobs_to_finalize,
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

        unit_ids = sorted(unit_ids)
        results: List[Tuple[str, ProcessResult]] = []

        # Compute mega unit ID
        mega_id = get_mega_unit_id(unit_ids)
        state_file = self._batch_states_dir / f"{mega_id}.json" if self._batch_states_dir else None

        if state_file and resume_batch and not state_file.exists():
            other_states = sorted(
                path
                for path in self._batch_states_dir.glob("batch_*.json")
                if path != state_file
            )
            if other_states:
                names = ", ".join(path.name for path in other_states)
                raise BatchStateConflictError(
                    f"Existing batch state ({names}) does not match the "
                    f"current pending unit set ({mega_id}). Refusing to submit "
                    "a replacement batch."
                )

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

        # Rebuild the exact provider requests even on resume. Unit IDs alone
        # are not a safe identity: source text, prompt templates, context, and
        # generation config can all change while filenames stay constant.
        batch_requests: List[BatchRequest] = []
        units_to_process: Dict[str, Tuple[WorkUnit, "ProcessContext"]] = {}
        skipped_identity: List[Dict[str, Any]] = []
        for uid in unit_ids:
            unit = self._get_or_create_unit(
                uid, unit_map, unit_states, originals
            )
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

            pre_result = self._hooks.pre_process(
                uid, unit.content, context
            )
            if not pre_result.should_process:
                fallback = getattr(pre_result, "fallback_result", None)
                reason = getattr(pre_result, "skip_reason", "")
                results.append((uid, ProcessResult(
                    success=True,
                    content=fallback,
                    skipped=True,
                    skip_reason=reason,
                )))
                skipped_identity.append({
                    "key": uid,
                    "fallback": fallback,
                    "reason": reason,
                })
                continue

            prompt = self._processor.build_prompt(unit.content, context)
            contents = self._convert_prompt_to_batch_contents(prompt)
            batch_requests.append(
                BatchRequest(key=uid, contents=contents)
            )
            units_to_process[uid] = (unit, context)

        if not batch_requests:
            return results

        processing_keys = [request.key for request in batch_requests]
        request_sha256 = _batch_request_sha256(
            batch_entry.provider,
            batch_entry.model,
            unit_ids,
            batch_requests,
            skipped_identity,
        )

        content_fps = None
        fingerprint_fn = getattr(
            self._batch_client,
            "_content_fingerprint",
            None,
        )
        if fingerprint_fn:
            content_fps = {}
            for request in batch_requests:
                fingerprint = fingerprint_fn(request.contents)
                if fingerprint in content_fps:
                    raise BatchStateConflictError(
                        "Batch contains duplicate provider requests for "
                        f"{content_fps[fingerprint]!r} and {request.key!r}; "
                        "Vertex cannot correlate unordered duplicate outputs "
                        "safely."
                    )
                content_fps[fingerprint] = request.key

        job_name = None
        existing_state = None
        if state_file and state_file.exists():
            if not resume_batch:
                raise BatchStateConflictError(
                    f"Batch state {state_file} exists but resume was not "
                    "explicitly requested."
                )
            existing_state = MegaUnitState.load(state_file)
            if existing_state is None:
                raise BatchStateConflictError(
                    f"Batch state {state_file} is unreadable or corrupted. "
                    "Refusing to overwrite it."
                )
            if not existing_state.job_name:
                raise BatchStateConflictError(
                    f"Mega unit {mega_id} has no confirmed job name. Inspect "
                    "the provider job list before cancelling state or "
                    "submitting a replacement."
                )
            actual_identity = (
                existing_state.provider,
                existing_state.model,
                existing_state.unit_ids,
                existing_state.processing_keys,
                existing_state.request_sha256,
            )
            expected_identity = (
                batch_entry.provider,
                batch_entry.model,
                unit_ids,
                processing_keys,
                request_sha256,
            )
            if actual_identity != expected_identity:
                raise BatchStateConflictError(
                    f"Mega unit {mega_id} does not match the current provider, "
                    "model, unit membership, or exact request content. "
                    "Refusing to reuse stale batch output."
                )
            if existing_state.job_state == "SUCCEEDED":
                logger.info(
                    f"Mega unit {mega_id} already completed, fetching results"
                )
            elif existing_state.job_state in (
                "PENDING",
                "RUNNING",
                "FAILED",
                "CANCELLED",
                "EXPIRED",
            ):
                logger.info(
                    f"Resuming mega unit {mega_id}: "
                    f"{existing_state.job_name}"
                )
            else:
                raise BatchStateConflictError(
                    f"Mega unit {mega_id} has unsupported persisted state "
                    f"{existing_state.job_state!r}. Inspect or explicitly "
                    "cancel the state before continuing."
                )
            job_name = existing_state.job_name
            if hasattr(self._batch_client, "restore_job_mapping"):
                self._batch_client.restore_job_mapping(
                    job_name,
                    processing_keys,
                    content_fps,
                )
        else:
            new_state = MegaUnitState(
                job_name="",
                job_state="SUBMITTING",
                provider=batch_entry.provider,
                model=batch_entry.model,
                unit_ids=unit_ids,
                processing_keys=processing_keys,
                content_fingerprints=content_fps,
                request_sha256=request_sha256,
            )
            if state_file:
                new_state.save(state_file)
            try:
                job_name = self._batch_client.submit(batch_requests)
            except Exception as exc:
                if state_file:
                    new_state.job_state = "SUBMISSION_UNKNOWN"
                    new_state.save(state_file)
                raise BatchStateConflictError(
                    f"Mega unit {mega_id} may have been accepted by the "
                    "provider, but no job name was confirmed. Refusing to "
                    "retry automatically."
                ) from exc

            skipped_count = len(unit_ids) - len(batch_requests)
            logger.info(
                f"Submitted mega unit {mega_id}: {job_name} "
                f"({len(batch_requests)} units, "
                f"{skipped_count} skipped by pre-process)"
            )
            if state_file:
                new_state.job_name = job_name
                new_state.job_state = "RUNNING"
                new_state.save(state_file)

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
                        state = MegaUnitState.load(state_file)
                        if state is None:
                            raise BatchStateConflictError(
                                f"Batch state {state_file} disappeared before "
                                "completion could be recorded"
                            )
                        state.job_name = job_name
                        state.job_state = "SUCCEEDED"
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

                    if state_file:
                        state = MegaUnitState.load(state_file)
                        if state is None:
                            raise BatchStateConflictError(
                                f"Batch state {state_file} disappeared before "
                                "failure could be recorded"
                            )
                        state.job_name = job_name
                        state.job_state = job_info.state.name
                        state.save(state_file)

                    return results

                time.sleep(self._batch_poll_interval)

            # Get results
            batch_responses = self._batch_client.get_results(
                job_name,
                cleanup=False,
            )
            response_map = {}
            expected_keys = set(units_to_process)
            for response in batch_responses:
                if response.key not in expected_keys:
                    raise BatchStateConflictError(
                        f"Batch job {job_name} returned unexpected key "
                        f"{response.key!r}"
                    )
                if response.key in response_map:
                    raise BatchStateConflictError(
                        f"Batch job {job_name} returned duplicate key "
                        f"{response.key!r}"
                    )
                response_map[response.key] = response

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
                    unit_id,
                    unit_states,
                    pending,
                    self._splitter,
                    split_threshold,
                    fallback_chain=(
                        [current_entry]
                        if error_type == ErrorType.TRUNCATION
                        and current_entry
                        and not state.chain
                        else None
                    ),
                )
                if split_success:
                    return (True, split_depth)

        # Try longest fallback
        longest = state.get_longest()
        if longest:
            if self._saver:
                try:
                    with self._saver.attempt(
                        unit_id,
                        f"{model_str}/fallback",
                    ) as attempt:
                        saved = attempt.success(
                            longest,
                            output_tokens=_count_tokens(longest),
                        )
                except Exception as exc:
                    raise BatchStateConflictError(
                        f"{unit_id}: exception while persisting fallback "
                        "result; batch state was retained"
                    ) from exc
                if not saved:
                    raise BatchStateConflictError(
                        f"{unit_id}: failed to persist fallback result; "
                        "batch state was retained"
                    )
            results[unit_id] = longest
            completed.add(unit_id)
            fallback_used.add(unit_id)
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
