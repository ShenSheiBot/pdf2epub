"""
Executor protocols and data structures.

The Executor is responsible for:
- Managing per-unit state (chain, quotas)
- Concurrent execution with dependency ordering
- Re-queuing failed units (no retry loop)
- Batch + Online simultaneous execution
"""

from typing import Protocol, Dict, Any, Optional, List, Set, Literal, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

if TYPE_CHECKING:
    from .._protocol import ProcessContext, ProcessorProtocol
    from ..hooks import CompositeHooks
    from .state import QuotaConfig


# ============================================================
# Work Unit - Re-export from core.types (Single Source of Truth)
# ============================================================

from ..types import WorkUnit, SplitType


# ============================================================
# Chain Entry
# ============================================================

@dataclass
class ChainEntry:
    """
    A model entry in the chain.

    The chain contains both batch and online entries.
    Mode determines how the entry is executed.
    """
    provider: str                      # "gemini", "deepseek", "anthropic"
    model: str                         # "gemini-2.0-flash", "deepseek-chat"
    mode: Literal["batch", "online"]   # Execution mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM client."""
        return {
            "provider": self.provider,
            "model": self.model,
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, ChainEntry):
            return False
        return (
            self.provider == other.provider and
            self.model == other.model and
            self.mode == other.mode
        )

    def __hash__(self) -> int:
        return hash((self.provider, self.model, self.mode))


# ============================================================
# Execution Result
# ============================================================

@dataclass
class ExecutionResult:
    """
    Result of executing units.

    Contains all results, categorized by outcome.
    """
    results: Dict[str, str] = field(default_factory=dict)  # key -> content
    completed: Set[str] = field(default_factory=set)       # Successfully completed
    failed: Set[str] = field(default_factory=set)          # Failed (exhausted retries)
    skipped: Set[str] = field(default_factory=set)         # Skipped (pre-processing)
    safety_blocked: Set[str] = field(default_factory=set)  # Safety blocked
    validation_failed: Set[str] = field(default_factory=set)  # Validation failed
    screener_passed: Set[str] = field(default_factory=set)  # Passed individual screener (skip batch)
    fallback_used: Set[str] = field(default_factory=set)   # Used longest fallback (needs warning)
    stats: Dict[str, Any] = field(default_factory=dict)    # Execution statistics

    # Detailed statistics (V1 parity)
    total_attempts: int = 0           # Total LLM calls made
    successful_attempts: int = 0      # Successful LLM calls
    splits_performed: int = 0         # Number of dynamic splits
    max_depth_reached: int = 0        # Maximum nesting depth from splits


@dataclass
class ProcessResult:
    """Result of processing a single unit."""
    success: bool
    content: Optional[str] = None
    error: Optional[Exception] = None
    skipped: bool = False
    skip_reason: str = ""
    context_ready: bool = False  # Whether result is ready for context injection


# ============================================================
# Executor Protocol
# ============================================================

class Executor(Protocol):
    """
    Executor protocol - responsible for processing units.

    Key design decisions:
    - No retry loop: failed units update state and re-enter pool
    - Batch + Online run simultaneously
    - Per-unit state management (chain, quotas)
    """

    def execute(
        self,
        units: List[WorkUnit],
        context_base: Optional["ProcessContext"] = None
    ) -> ExecutionResult:
        """
        Execute all units.

        Args:
            units: Units to process
            context_base: Base context for all units

        Returns:
            ExecutionResult with all outcomes
        """
        ...
