"""
Core types - SINGLE SOURCE OF TRUTH.

All other modules MUST import from here, not re-define.
This module is the canonical location for:
- SplitType: Type of split (proactive vs dynamic)
- ErrorType: Error classification
- WorkUnit: Unit of work for processing
- is_sub_key / filter_sub_keys: .sub virtual unit detection

IMPORTANT: Do not define these classes elsewhere. Tests enforce this.

Note: SplitType and WorkUnit are defined in work_unit.py to avoid circular imports,
but should be imported from this module (types.py).
"""

import re
from enum import Enum
from typing import Set

# Re-export from work_unit.py (canonical definitions there to avoid circular import)
from .work_unit import SplitType, WorkUnit


# ============================================================
# Error Type (unified error classification)
# ============================================================

class ErrorType(Enum):
    """
    Error type - single source of truth.

    Used for error classification, retry strategy, and quota management.
    """
    # Safety category - usually requires provider switch
    SAFETY = "safety"
    CONTENT_FILTER = "content_filter"

    # Network category - retryable
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"

    # Validation category - may need model switch
    VALIDATION = "validation"
    TRUNCATION = "truncation"

    # Parse category - retryable
    PARSE_ERROR = "parse_error"

    # Fallback
    UNKNOWN = "unknown"


# ============================================================
# .sub key detection (virtual units from dynamic splitting)
# ============================================================

def is_sub_key(key: str) -> bool:
    """
    Check if a key is a .sub virtual unit.

    .sub units are created by Executor's dynamic splitting and should:
    - Be saved to raw/ (for debugging)
    - NOT be promoted to validated/
    - NOT be loaded by next phases
    - NOT be tracked as completed/failed units
    - NOT be counted in statistics

    Pattern: .sub followed by digits (e.g., chapter_1.sub0, chapter_1.sub0.sub1)
    """
    return bool(re.search(r'\.sub\d+', key))


def filter_sub_keys(keys: Set[str]) -> Set[str]:
    """Filter out .sub virtual unit keys from a set."""
    return {k for k in keys if not is_sub_key(k)}


# ============================================================
# Re-exports for convenience
# ============================================================

__all__ = [
    "SplitType",
    "ErrorType",
    "WorkUnit",
    "is_sub_key",
    "filter_sub_keys",
]
