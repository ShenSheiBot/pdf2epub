"""
DiskFirstSaver - Executor's ONLY persistence interface.

Design principles:
1. Disk is the source of truth (not memory)
2. Every attempt MUST be recorded (via AttemptContext)
3. Executor cannot access Persistence or Tracker directly

This module provides structural guarantees:
- AttemptContext ensures every attempt is recorded
- Forgetting to call success()/failure()/skip() raises RuntimeError
- Exceptions are automatically captured and recorded
"""

import time
from typing import Optional, TYPE_CHECKING
from loguru import logger

from ..types import is_sub_key

if TYPE_CHECKING:
    from ..persistence import ResultPersistence
    from ..tracking import ProcessingTracker, AttemptRecord


class AttemptContext:
    """
    Context manager ensuring every attempt is recorded.

    Structural guarantee:
    - success() → writes file + records completed
    - failure() → records failed
    - skip() → records skipped
    - Exception → auto-records failure
    - Forgot to call anything → RuntimeError

    Usage:
        with saver.attempt(unit_id, model) as attempt:
            result = process(...)
            if result.success:
                attempt.success(content, output_tokens=100)
            else:
                attempt.failure("validation", "reason")
    """

    def __init__(self, saver: "DiskFirstSaver", unit_id: str, model: str):
        self._saver = saver
        self._unit_id = unit_id
        self._model = model
        self._recorded = False
        self._result_content: Optional[str] = None  # For aggregation access

    def __enter__(self) -> "AttemptContext":
        return self

    def success(
        self,
        content: str,
        output_tokens: int = 0,
        duration_seconds: float = 0.0,
        context_ready: bool = False,
    ) -> bool:
        """
        Mark attempt as successful.

        Writes file to disk, then records to tracker.
        Returns False if disk write fails (attempt recorded as io_error).

        Args:
            content: Processed content to save
            output_tokens: Token count for statistics
            duration_seconds: LLM call duration
            context_ready: Whether result passed screener (for context injection)

        Returns:
            True if save succeeded, False if disk write failed
        """
        if self._recorded:
            raise RuntimeError(
                f"Attempt for {self._unit_id} already recorded, cannot call success() again"
            )

        self._recorded = True
        self._result_content = content

        return self._saver._save_success_internal(
            self._unit_id,
            content,
            self._model,
            output_tokens,
            duration_seconds,
            context_ready,
        )

    def failure(self, error_type: str, error_message: str) -> None:
        """
        Mark attempt as failed.

        Only records to tracker, does not write file.

        Args:
            error_type: Error classification (validation, network, etc.)
            error_message: Human-readable error description
        """
        if self._recorded:
            raise RuntimeError(
                f"Attempt for {self._unit_id} already recorded, cannot call failure() again"
            )

        self._recorded = True
        self._saver._record_failure_internal(
            self._unit_id, self._model, error_type, error_message
        )

    def skip(self, reason: str, fallback_content: Optional[str] = None) -> None:
        """
        Mark attempt as skipped (pre-process filter).

        Optionally saves fallback content (e.g., for image-only pages).

        Args:
            reason: Why the unit was skipped
            fallback_content: Optional content to save (e.g., original for image-only)
        """
        if self._recorded:
            raise RuntimeError(
                f"Attempt for {self._unit_id} already recorded, cannot call skip() again"
            )

        self._recorded = True
        self._result_content = fallback_content

        self._saver._record_skip_internal(
            self._unit_id, self._model, reason, fallback_content
        )

    @property
    def content(self) -> Optional[str]:
        """Get the result content (for aggregation access)."""
        return self._result_content

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and not self._recorded:
            # Exception occurred and not yet recorded → auto-record failure
            error_type = self._saver._classify_exception(exc_type, exc_val)
            error_message = str(exc_val)[:500] if exc_val else "Unknown error"

            self._saver._record_failure_internal(
                self._unit_id, self._model, error_type, error_message
            )
            self._recorded = True

            logger.debug(f"{self._unit_id}: Auto-recorded exception as {error_type}")

        if not self._recorded:
            # Neither success/failure/skip called, nor exception occurred
            # This is a programming error - raise immediately
            raise RuntimeError(
                f"INVARIANT VIOLATION: Attempt for {self._unit_id} "
                f"completed without calling success()/failure()/skip(). "
                f"This is a bug in the calling code."
            )

        return False  # Don't suppress exceptions


class DiskFirstSaver:
    """
    Executor's ONLY interface for persistence.

    Encapsulates:
    - ResultPersistence (file I/O)
    - ProcessingTracker (state tracking)

    Executor MUST use attempt() context manager for all processing.
    Direct access to persistence/tracker is forbidden.

    Design:
    - Disk is truth: files are written before tracker is updated
    - .sub units: saved to disk (for debug) but not tracked
    - All attempts recorded: success, failure, skip, or exception
    """

    def __init__(
        self,
        persistence: "ResultPersistence",
        tracker: "ProcessingTracker",
    ):
        """
        Initialize saver with persistence and tracker.

        These are held privately - Executor cannot access them directly.

        Args:
            persistence: For file I/O (raw/ directory)
            tracker: For state tracking (processing_tracker.json)
        """
        self._persistence = persistence
        self._tracker = tracker

    def attempt(self, unit_id: str, model: str) -> AttemptContext:
        """
        Create an attempt context for processing a unit.

        This is the ONLY way Executor should record attempts.

        Args:
            unit_id: Unit identifier (e.g., "chapter_1" or "chapter_1.sub0")
            model: Model string (e.g., "gemini/gemini-2.0-flash")

        Returns:
            AttemptContext that must be used as context manager
        """
        return AttemptContext(self, unit_id, model)

    # ===== Read operations (for aggregation) =====

    def load(self, unit_id: str) -> Optional[str]:
        """
        Load content from disk.

        Used for aggregation: parent reads children from disk (not memory).

        Args:
            unit_id: Unit identifier

        Returns:
            Content if file exists, None otherwise
        """
        return self._persistence.get_raw(unit_id)

    def exists(self, unit_id: str) -> bool:
        """
        Check if unit has been saved to disk.

        Args:
            unit_id: Unit identifier

        Returns:
            True if raw file exists
        """
        return self._persistence.has_raw(unit_id)

    # ===== Internal methods (called by AttemptContext) =====

    def _save_success_internal(
        self,
        unit_id: str,
        content: str,
        model: str,
        output_tokens: int,
        duration_seconds: float,
        context_ready: bool,
    ) -> bool:
        """
        Internal: Save successful result.

        1. Write file to disk (truth source)
        2. Record to tracker (index)

        For .sub units: write file but don't track (they're virtual).
        """
        # Step 1: Write to disk FIRST (disk is truth)
        try:
            self._persistence.save_raw(unit_id, content)
        except Exception as e:
            logger.error(f"{unit_id}: Disk write failed: {e}")
            # Record as IO error instead of success
            if not is_sub_key(unit_id):
                self._record_failure_internal(unit_id, model, "io_error", str(e))
            return False

        # Step 2: Record to tracker (skip .sub units)
        if not is_sub_key(unit_id):
            from ..tracking import AttemptRecord
            attempt = AttemptRecord(
                timestamp=time.time(),
                status="completed",
                model=model,
                output_tokens=output_tokens,
                duration_seconds=duration_seconds,
            )
            self._tracker.record_attempt(unit_id, attempt)

        logger.debug(f"{unit_id}: Saved successfully ({output_tokens} tokens)")
        return True

    def _record_failure_internal(
        self,
        unit_id: str,
        model: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """
        Internal: Record failed attempt.

        Only records to tracker, does not write file.
        Skip .sub units (they're virtual).
        """
        if is_sub_key(unit_id):
            logger.debug(f"{unit_id}: Skipping tracker for .sub unit failure")
            return

        from ..tracking import AttemptRecord
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="failed",
            model=model,
            error_type=error_type,
            error_message=error_message[:500] if error_message else None,
        )
        self._tracker.record_attempt(unit_id, attempt)

        logger.debug(f"{unit_id}: Recorded failure ({error_type})")

    def _record_skip_internal(
        self,
        unit_id: str,
        model: str,
        reason: str,
        fallback_content: Optional[str],
    ) -> None:
        """
        Internal: Record skipped attempt.

        If fallback_content provided, saves to disk.
        Records as "completed" with skip metadata.
        """
        # Save fallback content if provided
        if fallback_content is not None:
            try:
                self._persistence.save_raw(unit_id, fallback_content)
            except Exception as e:
                logger.warning(f"{unit_id}: Failed to save fallback: {e}")

        # Record to tracker (skip .sub units)
        if not is_sub_key(unit_id):
            from ..tracking import AttemptRecord
            attempt = AttemptRecord(
                timestamp=time.time(),
                status="completed",  # Skipped counts as completed
                model=model,
                error_message=f"Skipped: {reason}",
            )
            self._tracker.record_attempt(unit_id, attempt)

        logger.debug(f"{unit_id}: Skipped ({reason})")

    def _classify_exception(self, exc_type, exc_val) -> str:
        """
        Classify an exception into error type.

        Used by AttemptContext.__exit__ for auto-recording.
        """
        exc_name = exc_type.__name__ if exc_type else "Unknown"
        exc_str = str(exc_val).lower() if exc_val else ""

        # Network errors
        if "timeout" in exc_str or "timed out" in exc_str:
            return "timeout"
        if "connection" in exc_str or "network" in exc_str:
            return "network"
        if "rate" in exc_str and "limit" in exc_str:
            return "rate_limit"

        # Safety errors
        if "safety" in exc_str or "blocked" in exc_str or "harmful" in exc_str:
            return "safety"
        if "content" in exc_str and "filter" in exc_str:
            return "content_filter"

        # Validation errors
        if "validation" in exc_str or "invalid" in exc_str:
            return "validation"
        if "truncat" in exc_str:
            return "truncation"

        # Parse errors
        if "parse" in exc_str or "json" in exc_str or "decode" in exc_str:
            return "parse_error"

        return "unknown"


# For type hints in other modules
__all__ = ["DiskFirstSaver", "AttemptContext"]
