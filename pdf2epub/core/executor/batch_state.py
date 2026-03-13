"""
Mega Unit state persistence for V2 executor.

Each mega unit (batch job) has its own state file:
- batch_states/batch_{id}.json

ID = hash(sorted(unit_ids)), so same pending units = same ID = natural resume.
"""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from loguru import logger


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
    - processing_keys: Ordered list of unit keys (for Vertex line-order correlation)
    """
    job_name: str
    job_state: str = "RUNNING"
    processing_keys: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {
            "job_name": self.job_name,
            "job_state": self.job_state,
        }
        if self.processing_keys is not None:
            d["processing_keys"] = self.processing_keys
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'MegaUnitState':
        """Create from dict (loaded from JSON)."""
        return cls(
            job_name=data["job_name"],
            job_state=data.get("job_state", "RUNNING"),
            processing_keys=data.get("processing_keys"),
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
