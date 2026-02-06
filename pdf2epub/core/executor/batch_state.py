"""
Batch job state persistence for V2 executor.

Enables:
1. Resume interrupted batch jobs (no resubmission = no wasted money)
2. Prevent duplicate submissions (no double billing)
3. Clean cancellation on interrupt
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger


@dataclass
class PersistentBatchState:
    """
    Persistent state for batch jobs.

    Saved to batch_state.json after job submission.
    Loaded on executor init to enable resume.
    Cleared after job completion or cancellation.
    """
    # Active job tracking
    active_job_name: Optional[str] = None
    processing_keys: List[str] = field(default_factory=list)

    # Request metadata for resume
    batch_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Retry tracking
    retry_count: int = 0

    # Safety-blocked keys (need different fallback)
    safety_blocked_keys: List[str] = field(default_factory=list)

    # Batch entry info for resume
    batch_provider: Optional[str] = None
    batch_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "active_job_name": self.active_job_name,
            "processing_keys": self.processing_keys,
            "batch_metadata": self.batch_metadata,
            "retry_count": self.retry_count,
            "safety_blocked_keys": self.safety_blocked_keys,
            "batch_provider": self.batch_provider,
            "batch_model": self.batch_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentBatchState':
        """Create from dict (loaded from JSON)."""
        return cls(
            active_job_name=data.get("active_job_name"),
            processing_keys=data.get("processing_keys", []),
            batch_metadata=data.get("batch_metadata", {}),
            retry_count=data.get("retry_count", 0),
            safety_blocked_keys=data.get("safety_blocked_keys", []),
            batch_provider=data.get("batch_provider"),
            batch_model=data.get("batch_model"),
        )

    def save(self, path: Path) -> None:
        """
        Save state to JSON file.

        Called immediately after batch job submission to enable resume.

        Args:
            path: Path to state file (e.g., output/title/batch_state.json)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically to avoid corruption
        tmp_path = path.with_suffix('.json.tmp')
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            tmp_path.rename(path)
            logger.debug(f"Saved batch state to {path}")
        except Exception as e:
            logger.error(f"Failed to save batch state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @classmethod
    def load(cls, path: Path) -> Optional['PersistentBatchState']:
        """
        Load state from JSON file.

        Called on executor init to check for active job.

        Args:
            path: Path to state file

        Returns:
            PersistentBatchState if file exists and is valid, None otherwise
        """
        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            state = cls.from_dict(data)
            logger.debug(f"Loaded batch state from {path}")
            return state
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted batch state file {path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load batch state from {path}: {e}")
            return None

    @classmethod
    def clear(cls, path: Path) -> None:
        """
        Clear state file.

        Called after job completion or cancellation.

        Args:
            path: Path to state file
        """
        if path.exists():
            try:
                path.unlink()
                logger.debug(f"Cleared batch state file {path}")
            except Exception as e:
                logger.warning(f"Failed to clear batch state file {path}: {e}")

    def has_active_job(self) -> bool:
        """Check if there's an active batch job."""
        return self.active_job_name is not None

    def add_safety_blocked(self, key: str) -> None:
        """Mark a key as safety-blocked."""
        if key not in self.safety_blocked_keys:
            self.safety_blocked_keys.append(key)

    def is_safety_blocked(self, key: str) -> bool:
        """Check if a key is safety-blocked."""
        return key in self.safety_blocked_keys


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
