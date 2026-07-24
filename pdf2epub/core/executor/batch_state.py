"""
Mega Unit state persistence for V2 executor.

Each mega unit (batch job) has its own state file:
- batch_states/batch_{id}.json

ID = hash(sorted(unit_ids)), so same pending units = same ID = natural resume.
"""

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, IO, List, Optional
from loguru import logger

try:
    import fcntl
except ImportError:  # pragma: no cover - batch execution currently targets POSIX
    fcntl = None


class BatchRunLockedError(RuntimeError):
    """Raised when another process owns an output stage's batch run."""


class BatchStateConflictError(RuntimeError):
    """Batch state cannot be resumed safely without operator inspection."""


class BatchRunLock:
    """Process-level exclusive lock for one output stage.

    Batch state files make interrupted jobs resumable, but they cannot prevent
    two processes from both observing "no state" and submitting at the same
    time. An advisory OS lock closes that race and is released automatically
    if the owning process exits.
    """

    def __init__(self, batch_states_dir: Path):
        self._path = batch_states_dir / ".executor.lock"
        self._handle: Optional[IO[str]] = None

    def __enter__(self) -> "BatchRunLock":
        if fcntl is None:
            raise RuntimeError(
                "Safe batch execution requires POSIX advisory file locking"
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            owner_hint = f" (owner PID {owner})" if owner else ""
            raise BatchRunLockedError(
                f"Another pdf2epub process is already using "
                f"{self._path.parent}{owner_hint}"
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.flush()
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def get_mega_unit_id(unit_ids: List[str]) -> str:
    """
    Compute mega unit ID from unit IDs.

    Same pending units = same ID = natural resume.

    Args:
        unit_ids: List of unit IDs in this batch

    Returns:
        ID string like "batch_a3f2b1c4"
    """
    sorted_ids = sorted(unit_ids)
    hash_input = ",".join(sorted_ids).encode()
    hash_value = hashlib.sha256(hash_input).hexdigest()[:16]
    return f"batch_{hash_value}"


@dataclass
class MegaUnitState:
    """
    Minimal state for a mega unit (batch job).

    Only stores what's needed for resume:
    - job_name: The batch API job name
    - job_state: Current job state (PENDING, RUNNING, SUCCEEDED, etc.)
    - provider/model: Identity of the submitted model
    - unit_ids: Complete mega-unit membership, including pre-process skips
    - processing_keys: Ordered list of unit keys (for Vertex line-order correlation)
    - content_fingerprints: {md5_hash: key} map for content-based matching
    - request_sha256: Canonical fingerprint of the complete provider request
    """
    job_name: str
    job_state: str = "RUNNING"
    provider: Optional[str] = None
    model: Optional[str] = None
    unit_ids: Optional[List[str]] = None
    processing_keys: Optional[List[str]] = None
    content_fingerprints: Optional[Dict[str, str]] = None
    request_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {
            "job_name": self.job_name,
            "job_state": self.job_state,
        }
        if self.provider is not None:
            d["provider"] = self.provider
        if self.model is not None:
            d["model"] = self.model
        if self.unit_ids is not None:
            d["unit_ids"] = self.unit_ids
        if self.processing_keys is not None:
            d["processing_keys"] = self.processing_keys
        if self.content_fingerprints is not None:
            d["content_fingerprints"] = self.content_fingerprints
        if self.request_sha256 is not None:
            d["request_sha256"] = self.request_sha256
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'MegaUnitState':
        """Create from dict (loaded from JSON)."""
        return cls(
            job_name=data["job_name"],
            job_state=data.get("job_state", "RUNNING"),
            provider=data.get("provider"),
            model=data.get("model"),
            unit_ids=data.get("unit_ids"),
            processing_keys=data.get("processing_keys"),
            content_fingerprints=data.get("content_fingerprints"),
            request_sha256=data.get("request_sha256"),
        )

    def save(self, path: Path) -> None:
        """
        Save state to JSON file atomically.

        Args:
            path: Path to state file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically to avoid corruption
        tmp_path = path.with_suffix('.json.tmp')
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            tmp_path.rename(path)
            logger.debug(f"Saved mega unit state to {path}")
        except Exception as e:
            logger.error(f"Failed to save mega unit state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @classmethod
    def load(cls, path: Path) -> Optional['MegaUnitState']:
        """
        Load state from JSON file.

        Args:
            path: Path to state file

        Returns:
            MegaUnitState if file exists and is valid, None otherwise
        """
        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            state = cls.from_dict(data)
            logger.debug(f"Loaded mega unit state from {path}")
            return state
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted mega unit state file {path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load mega unit state from {path}: {e}")
            return None


# Safety block detection keywords
SAFETY_KEYWORDS = [
    'PROHIBITED_CONTENT',
    'SAFETY',
    'BLOCK',
    'blocked',
    'safety',
    'harmful',
    'policy violation',
]


def is_safety_error(error_message: str) -> bool:
    """
    Check if an error message indicates a safety block.

    Args:
        error_message: Error string from batch response

    Returns:
        True if this is a safety-related error
    """
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(kw.lower() in error_lower for kw in SAFETY_KEYWORDS)
