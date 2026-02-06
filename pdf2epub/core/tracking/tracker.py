#!/usr/bin/env python3
"""
ProcessingTracker - Audit trail and progress tracking system.

Provides comprehensive tracking of processing attempts, split history,
and error categorization for robust resume functionality.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

from ...chapter_identity import ChapterIdentity

# Re-export from core.types (Single Source of Truth)
from ..types import ErrorType


@dataclass
class AttemptRecord:
    """Record of a single processing attempt."""
    timestamp: float
    status: str  # "completed" or "failed"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_output_path: Optional[str] = None  # Path to saved error output for debugging
    used_fallback: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "timestamp": self.timestamp,
            "status": self.status,
            "model": self.model
        }
        if self.input_tokens:
            d["input_tokens"] = self.input_tokens
        if self.output_tokens:
            d["output_tokens"] = self.output_tokens
        if self.duration_seconds:
            d["duration_seconds"] = round(self.duration_seconds, 2)
        if self.error_type:
            d["error_type"] = self.error_type
            d["error_message"] = self.error_message
        if self.error_output_path:
            d["error_output_path"] = self.error_output_path
        if self.used_fallback:
            d["used_fallback"] = True
            d["fallback_reason"] = self.fallback_reason
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'AttemptRecord':
        """Create from dictionary."""
        return cls(
            timestamp=d["timestamp"],
            status=d["status"],
            model=d["model"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            duration_seconds=d.get("duration_seconds", 0.0),
            error_type=d.get("error_type"),
            error_message=d.get("error_message"),
            error_output_path=d.get("error_output_path"),
            used_fallback=d.get("used_fallback", False),
            fallback_reason=d.get("fallback_reason")
        )


@dataclass
class SplitRecord:
    """Record of a content split operation."""
    timestamp: float
    split_points: List[int]
    total_tokens: int
    part_count: int
    method: str  # "content_splitter", "llm_resplit", "manual"
    reason: str
    triggered_by: Optional[str] = None  # unit that caused resplit
    version: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "version": self.version,
            "timestamp": self.timestamp,
            "split_points": self.split_points,
            "total_tokens": self.total_tokens,
            "part_count": self.part_count,
            "method": self.method,
            "reason": self.reason
        }
        if self.triggered_by:
            d["triggered_by"] = self.triggered_by
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'SplitRecord':
        """Create from dictionary."""
        return cls(
            timestamp=d["timestamp"],
            split_points=d["split_points"],
            total_tokens=d["total_tokens"],
            part_count=d["part_count"],
            method=d["method"],
            reason=d["reason"],
            triggered_by=d.get("triggered_by"),
            version=d.get("version", 0)
        )


@dataclass
class ProcessingPlan:
    """Plan for processing a file."""
    action: str  # "skip", "process_all", "process_parts"
    parts_to_process: List[str] = field(default_factory=list)
    reason: str = ""


class ProcessingTracker:
    """
    Manages processing progress and audit trail.

    Provides:
    - Per-unit attempt tracking with full history
    - Split version control (never overwrites)
    - Error categorization and statistics
    - Human-readable status reports
    - Robust resume functionality
    """

    def __init__(
        self,
        progress_path: Path,
        processor_name: str,
        file_checker: Optional[Any] = None,
    ):
        """
        Initialize the tracker.

        Args:
            progress_path: Path to the progress JSON file
            processor_name: Name of the processor (e.g., "PolishProcessor")
            file_checker: Optional callable (key: str) -> bool that checks if output file exists.
                          If provided, is_unit_complete() will also verify file existence.
                          This prevents data loss when tracker shows 'completed' but file is missing.
        """
        self.progress_path = Path(progress_path)
        self.processor_name = processor_name
        self._file_checker = file_checker
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.progress = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing progress or create new structure."""
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Migrate old format if necessary
                    data = self._migrate_if_needed(data)
                    return data
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}, creating new")

        return self._create_empty_progress()

    def _create_empty_progress(self) -> dict:
        """Create empty progress structure."""
        return {
            "units": {},
            "split_history": {},
            "summary": {
                "total_units": 0,
                "completed": 0,
                "failed": 0,
                "pending": 0,
                "total_attempts": 0,
                "total_retries": 0,
                "errors_by_type": {},
                "models_used": {},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_duration_seconds": 0.0
            },
            "processor": self.processor_name,
            "started_at": time.time(),
            "last_updated": time.time()
        }

    def _migrate_if_needed(self, data: dict) -> dict:
        """Migrate from old progress format if necessary."""
        # Check if this is old format (has parts_info instead of units)
        if "parts_info" in data and "units" not in data:
            logger.info("Migrating progress from old format to new format")
            return self._migrate_from_old_format(data)
        return data

    def _migrate_from_old_format(self, old_data: dict) -> dict:
        """Convert old progress format to new format."""
        new_data = self._create_empty_progress()

        parts_info = old_data.get("parts_info", {})

        for file_key, file_info in parts_info.items():
            if isinstance(file_info, dict):
                if "parts" in file_info:
                    # Multi-part file
                    split_points = file_info.get("split_points", [])
                    total_tokens = file_info.get("total_tokens", 0)
                    parts = file_info.get("parts", {})

                    # Record split history
                    new_data["split_history"][file_key] = {
                        "versions": [{
                            "version": 0,
                            "timestamp": time.time(),
                            "split_points": split_points,
                            "total_tokens": total_tokens,
                            "part_count": len(parts),
                            "method": "migrated",
                            "reason": "migrated_from_old_format"
                        }],
                        "current_version": 0
                    }

                    # Record each part
                    for part_num, part_info in parts.items():
                        part_key = ChapterIdentity.make_part_name(file_key, int(part_num))
                        new_data["units"][part_key] = {
                            "status": "completed" if part_info.get("completed") else "failed",
                            "attempts": [{
                                "timestamp": part_info.get("timestamp", time.time()),
                                "status": "completed" if part_info.get("completed") else "failed",
                                "model": "unknown",
                                "output_tokens": part_info.get("tokens", 0)
                            }],
                            "retry_count": 0,
                            "final_output_tokens": part_info.get("tokens", 0)
                        }
                else:
                    # Single file
                    new_data["units"][file_key] = {
                        "status": "completed" if file_info.get("completed") else "failed",
                        "attempts": [{
                            "timestamp": file_info.get("timestamp", time.time()),
                            "status": "completed" if file_info.get("completed") else "failed",
                            "model": "unknown"
                        }],
                        "retry_count": 0
                    }

        # Update summary
        self._update_summary_from_units(new_data)

        return new_data

    def save(self):
        """Save progress to file atomically."""
        self.progress["last_updated"] = time.time()
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file then rename
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.progress_path.parent,
            suffix='.tmp'
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, self.progress_path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    # === Unit Status Management ===

    def record_attempt(self, unit_key: str, attempt: AttemptRecord):
        """
        Record a processing attempt (success or failure).

        Args:
            unit_key: Unit identifier (e.g., "chapter_5.part1")
            attempt: AttemptRecord with attempt details
        """
        with self._lock:
            if unit_key not in self.progress["units"]:
                self.progress["units"][unit_key] = {
                    "status": "pending",
                    "attempts": [],
                    "retry_count": 0
                }

            unit = self.progress["units"][unit_key]
            unit["attempts"].append(attempt.to_dict())

            if attempt.status == "completed":
                unit["status"] = "completed"
                if attempt.output_tokens:
                    unit["final_output_tokens"] = attempt.output_tokens
            else:
                unit["status"] = "failed"
                # Only increment retry_count if this is not the first attempt
                if len(unit["attempts"]) > 1:
                    unit["retry_count"] = len(unit["attempts"]) - 1

            self._update_summary()
            self.save()

        # Log the attempt (outside lock)
        if attempt.status == "completed":
            logger.info(f"[COMPLETED] {unit_key} - {attempt.model} "
                       f"({attempt.output_tokens} tokens, {attempt.duration_seconds:.1f}s)")
        else:
            logger.warning(f"[FAILED] {unit_key} - {attempt.error_type}: {attempt.error_message}")

    def record_validation(self, unit_key: str, validation_record: Dict[str, Any]):
        """
        Record a validation judgment.

        Args:
            unit_key: Unit identifier
            validation_record: Dict with validation details (from ValidationRecord.to_dict())
        """
        with self._lock:
            if unit_key not in self.progress["units"]:
                self.progress["units"][unit_key] = {
                    "status": "pending",
                    "attempts": [],
                    "retry_count": 0
                }

            unit = self.progress["units"][unit_key]
            if "validations" not in unit:
                unit["validations"] = []
            unit["validations"].append(validation_record)
            self.save()

    def get_unit_status(self, unit_key: str) -> str:
        """
        Get unit status.

        Args:
            unit_key: Unit identifier

        Returns:
            Status string: "completed", "failed", or "pending"
        """
        return self.progress["units"].get(unit_key, {}).get("status", "pending")

    def is_unit_complete(self, unit_key: str) -> bool:
        """Check if a unit is completed.

        If a file_checker was provided at init, this also verifies the output file exists.
        This prevents the edge case where tracker shows 'completed' but file is missing
        (e.g., due to interruption between tracker update and file save).
        """
        if self.get_unit_status(unit_key) != "completed":
            return False

        # Safety check: verify file exists if checker provided
        if self._file_checker is not None:
            try:
                if not self._file_checker(unit_key):
                    logger.warning(
                        f"{unit_key}: Tracker shows completed but file missing, "
                        "treating as incomplete"
                    )
                    return False
            except Exception as e:
                logger.debug(f"{unit_key}: File check failed: {e}")
                # On error, trust tracker state
                pass

        return True

    def get_failed_units(self) -> List[str]:
        """Get all failed unit keys."""
        with self._lock:
            return [k for k, v in self.progress["units"].items()
                    if v.get("status") == "failed"]

    def get_units_by_error_type(self, error_type: str) -> List[str]:
        """Get unit keys that have a specific error type."""
        with self._lock:
            result = []
            for unit_key, unit in self.progress["units"].items():
                for attempt in unit.get("attempts", []):
                    if attempt.get("error_type") == error_type:
                        result.append(unit_key)
                        break
            return result

    def get_unit_attempts(self, unit_key: str) -> List[AttemptRecord]:
        """Get all attempts for a unit."""
        unit = self.progress["units"].get(unit_key, {})
        attempts = unit.get("attempts", [])
        return [AttemptRecord.from_dict(a) for a in attempts]

    def get_last_error(self, unit_key: str) -> Optional[str]:
        """Get the last error message for a unit."""
        unit = self.progress["units"].get(unit_key, {})
        attempts = unit.get("attempts", [])
        for attempt in reversed(attempts):
            if attempt.get("error_message"):
                return attempt["error_message"]
        return None

    # === Split History Management ===

    def record_split(self, base_key: str, split_record: SplitRecord):
        """
        Record a new split version (never overwrites).

        Args:
            base_key: Base file key (e.g., "chapter_5")
            split_record: SplitRecord with split details
        """
        with self._lock:
            if base_key not in self.progress["split_history"]:
                self.progress["split_history"][base_key] = {
                    "versions": [],
                    "current_version": -1
                }

            history = self.progress["split_history"][base_key]
            version = len(history["versions"])
            split_record.version = version

            history["versions"].append(split_record.to_dict())
            history["current_version"] = version

            self.save()

        logger.info(f"[SPLIT] {base_key} v{version}: {split_record.part_count} parts "
                   f"({split_record.method}, {split_record.reason})")

    def get_current_split(self, base_key: str) -> Optional[dict]:
        """
        Get current split version for a base key.

        Args:
            base_key: Base file key

        Returns:
            Split info dict or None if no split exists
        """
        history = self.progress["split_history"].get(base_key, {})
        versions = history.get("versions", [])
        current = history.get("current_version", -1)

        if current >= 0 and current < len(versions):
            return versions[current]
        return None

    def get_split_history(self, base_key: str) -> List[dict]:
        """Get all split versions for a base key."""
        history = self.progress["split_history"].get(base_key, {})
        return history.get("versions", [])

    def get_parts_for_base(self, base_key: str) -> List[str]:
        """
        Get all part keys for a base key.

        Args:
            base_key: Base file key (e.g., "chapter_5")

        Returns:
            List of part keys (e.g., ["chapter_5.part1", "chapter_5.part2"])
            or [base_key] if no split exists
        """
        current_split = self.get_current_split(base_key)

        if not current_split:
            return [base_key]

        part_count = current_split["part_count"]
        return [
            ChapterIdentity.make_part_name(base_key, i)
            for i in range(1, part_count + 1)
        ]

    def get_consecutive_failures(self, unit_key: str) -> int:
        """
        Get the number of consecutive failures for a unit.

        Args:
            unit_key: Unit identifier

        Returns:
            Number of consecutive failures from the end
        """
        unit = self.progress["units"].get(unit_key, {})
        attempts = unit.get("attempts", [])

        if not attempts:
            return 0

        count = 0
        for attempt in reversed(attempts):
            if attempt.get("status") == "failed":
                count += 1
            else:
                break

        return count

    def reset_parts_status(self, base_key: str, part_indices: List[int]):
        """
        Reset status of specific parts to pending (used after resplit).

        Args:
            base_key: Base file key
            part_indices: Indices of parts to reset
        """
        with self._lock:
            for idx in part_indices:
                part_key = ChapterIdentity.make_part_name(base_key, idx)
                if part_key in self.progress["units"]:
                    self.progress["units"][part_key]["status"] = "pending"
                else:
                    self.progress["units"][part_key] = {
                        "status": "pending",
                        "attempts": [],
                        "retry_count": 0
                    }

            self.save()

    def reset_unit_status(self, unit_key: str):
        """
        Reset a unit's status to pending (used for retry after validation failure).

        Args:
            unit_key: Unit key to reset
        """
        with self._lock:
            if unit_key in self.progress["units"]:
                self.progress["units"][unit_key]["status"] = "pending"
            else:
                # Create new unit entry if doesn't exist
                self.progress["units"][unit_key] = {
                    "status": "pending",
                    "attempts": [],
                    "retry_count": 0
                }
            self.save()

    def migrate_unit_keys(self, old_to_new: Dict[str, str]):
        """
        Migrate unit data from old keys to new keys (used during resplit renumbering).

        Args:
            old_to_new: Mapping from old unit keys to new unit keys
        """
        with self._lock:
            units = self.progress["units"]

            for old_key, new_key in old_to_new.items():
                if old_key in units and old_key != new_key:
                    units[new_key] = units.pop(old_key)
                    logger.debug(f"Migrated unit {old_key} → {new_key}")

            self.save()

    # === Processing Plan ===

    def get_processing_plan(self, base_key: str, output_dir: Path = None) -> ProcessingPlan:
        """
        Determine what needs to be processed for a file.

        Args:
            base_key: Base file key (e.g., "chapter_5")
            output_dir: Output directory for discovering part files

        Returns:
            ProcessingPlan indicating what action to take
        """
        # Check if there's a split
        current_split = self.get_current_split(base_key)

        if current_split:
            # Has split - check each part
            parts_to_process = []
            part_count = current_split["part_count"]

            for i in range(1, part_count + 1):
                part_key = ChapterIdentity.make_part_name(base_key, i)
                status = self.get_unit_status(part_key)

                if status != "completed":
                    parts_to_process.append(part_key)

            if not parts_to_process:
                return ProcessingPlan(
                    action="skip",
                    reason=f"all {part_count} parts completed"
                )
            else:
                return ProcessingPlan(
                    action="process_parts",
                    parts_to_process=parts_to_process,
                    reason=f"{len(parts_to_process)}/{part_count} parts need processing"
                )
        else:
            # No split - check single file status
            status = self.get_unit_status(base_key)

            if status == "completed":
                return ProcessingPlan(
                    action="skip",
                    reason="already completed"
                )
            else:
                return ProcessingPlan(
                    action="process_all",
                    reason="not started" if status == "pending" else "failed, needs retry"
                )

    # === Summary and Reporting ===

    def _update_summary(self):
        """Update summary statistics."""
        units = self.progress["units"]

        # Count statuses
        completed = sum(1 for u in units.values() if u.get("status") == "completed")
        failed = sum(1 for u in units.values() if u.get("status") == "failed")
        pending = sum(1 for u in units.values() if u.get("status") == "pending")

        # Count attempts and retries
        total_attempts = sum(len(u.get("attempts", [])) for u in units.values())
        total_retries = sum(u.get("retry_count", 0) for u in units.values())

        # Count errors by type
        errors_by_type = {}
        for unit in units.values():
            for attempt in unit.get("attempts", []):
                error_type = attempt.get("error_type")
                if error_type:
                    errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        # Count models used
        models_used = {}
        for unit in units.values():
            for attempt in unit.get("attempts", []):
                model = attempt.get("model", "unknown")
                models_used[model] = models_used.get(model, 0) + 1

        # Sum tokens and duration
        total_input = 0
        total_output = 0
        total_duration = 0.0

        for unit in units.values():
            for attempt in unit.get("attempts", []):
                total_input += attempt.get("input_tokens", 0)
                total_output += attempt.get("output_tokens", 0)
                total_duration += attempt.get("duration_seconds", 0.0)

        self.progress["summary"] = {
            "total_units": len(units),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total_attempts": total_attempts,
            "total_retries": total_retries,
            "errors_by_type": errors_by_type,
            "models_used": models_used,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_duration_seconds": round(total_duration, 1)
        }

    def _update_summary_from_units(self, data: dict):
        """Update summary in the given data dict (for migration)."""
        units = data["units"]

        completed = sum(1 for u in units.values() if u.get("status") == "completed")
        failed = sum(1 for u in units.values() if u.get("status") == "failed")

        data["summary"]["total_units"] = len(units)
        data["summary"]["completed"] = completed
        data["summary"]["failed"] = failed
        data["summary"]["pending"] = len(units) - completed - failed

    def generate_report(self) -> str:
        """
        Generate human-readable status report.

        Returns:
            Formatted report string
        """
        summary = self.progress["summary"]

        # Calculate completion rate
        total = summary["total_units"]
        rate = (summary["completed"] / total * 100) if total > 0 else 0

        report = f"""
=== {self.processor_name} Status Report ===

Progress:
  Total Units: {total}
  Completed:   {summary['completed']} ({rate:.1f}%)
  Failed:      {summary['failed']}
  Pending:     {summary['pending']}

Retry Statistics:
  Total Attempts: {summary['total_attempts']}
  Total Retries:  {summary['total_retries']}

"""

        # Errors by type
        if summary["errors_by_type"]:
            report += "Errors by Type:\n"
            for error_type, count in sorted(summary["errors_by_type"].items(),
                                           key=lambda x: -x[1]):
                report += f"  {error_type}: {count}\n"
            report += "\n"

        # Failed units
        failed_units = self.get_failed_units()
        if failed_units:
            report += f"Failed Units ({len(failed_units)}):\n"
            for unit_key in sorted(failed_units):
                last_error = self.get_last_error(unit_key)
                attempts = len(self.progress["units"][unit_key].get("attempts", []))
                error_preview = (last_error[:50] + "...") if last_error and len(last_error) > 50 else last_error
                report += f"  - {unit_key} ({attempts} attempts): {error_preview}\n"
            report += "\n"

        # Models used
        if summary["models_used"]:
            report += "Models Used:\n"
            for model, count in sorted(summary["models_used"].items(),
                                       key=lambda x: -x[1]):
                report += f"  {model}: {count} calls\n"
            report += "\n"

        # Token usage
        report += f"""Token Usage:
  Input:  {summary['total_input_tokens']:,}
  Output: {summary['total_output_tokens']:,}

Total Duration: {summary['total_duration_seconds']:.1f}s
"""

        return report

    def get_unit_report(self, unit_key: str) -> str:
        """
        Generate detailed report for a specific unit.

        Args:
            unit_key: Unit identifier

        Returns:
            Formatted report string
        """
        unit = self.progress["units"].get(unit_key)
        if not unit:
            return f"Unit '{unit_key}' not found"

        report = f"""
=== Unit Report: {unit_key} ===

Status: {unit.get('status', 'unknown')}
Retry Count: {unit.get('retry_count', 0)}
Final Output Tokens: {unit.get('final_output_tokens', 'N/A')}

Attempts ({len(unit.get('attempts', []))}):\n"""

        for i, attempt in enumerate(unit.get("attempts", []), 1):
            report += f"\n  Attempt {i}:\n"
            report += f"    Status: {attempt['status']}\n"
            report += f"    Model: {attempt['model']}\n"
            if attempt.get("input_tokens"):
                report += f"    Input Tokens: {attempt['input_tokens']}\n"
            if attempt.get("output_tokens"):
                report += f"    Output Tokens: {attempt['output_tokens']}\n"
            if attempt.get("duration_seconds"):
                report += f"    Duration: {attempt['duration_seconds']:.1f}s\n"
            if attempt.get("error_type"):
                report += f"    Error Type: {attempt['error_type']}\n"
                report += f"    Error Message: {attempt.get('error_message', 'N/A')}\n"
            if attempt.get("used_fallback"):
                report += f"    Used Fallback: Yes ({attempt.get('fallback_reason', 'unknown')})\n"

        return report


