"""
Per-unit state management.

Each unit has its own mutable state:
- chain: Available models (modified on failures)
- quotas: Per-error-type retry counts
- dependencies: Context injection + aggregation
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from threading import Lock

from ._protocol import ChainEntry
from ..hooks import ErrorType, ErrorEffect


@dataclass
class QuotaConfig:
    """
    Configuration for retry quotas.

    Attributes:
        total: Maximum total retries across all error types
        per_type: Per-error-type quotas
        per_entry: Default retry count per chain entry (applies to both batch and online)
    """
    total: int = 5
    per_type: Dict[ErrorType, int] = field(default_factory=lambda: {
        ErrorType.SAFETY: 999,        # Unlimited (just removes models)
        ErrorType.NETWORK: 3,
        ErrorType.VALIDATION: 2,      # 2 retries per entry before fallback
        ErrorType.TRUNCATION: 2,      # Truncation gets 2 retries (model switch)
        ErrorType.RATE_LIMIT: 3,
        ErrorType.TIMEOUT: 3,
        ErrorType.CONTENT_FILTER: 1,  # Like safety, 1 retry then switch provider
        ErrorType.PARSE_ERROR: 2,
        ErrorType.UNKNOWN: 2,
    })
    per_entry: int = 2  # Default retry count per chain entry

    def create_quotas(self) -> Dict[ErrorType, int]:
        """Create a copy of per-type quotas for a unit."""
        return dict(self.per_type)


@dataclass
class UnitState:
    """
    Mutable state for a single unit.

    This state is modified during execution:
    - chain shrinks as models fail (each entry has its own retry count)
    - quotas decrement on failures
    - attempts accumulate for longest fallback

    Per-entry retry design:
    - Each chain entry has retry_count (default from QuotaConfig.per_entry)
    - On failure: current entry's count -1
    - When count reaches 0: entry is removed, fallback to next
    - This applies uniformly to batch and online entries
    """
    chain: List[ChainEntry]           # Available models (modified on failures)
    total_quota: int                  # Total retries remaining
    quotas: Dict[ErrorType, int]      # Per-type quotas
    entry_retries: Dict[int, int] = field(default_factory=dict)  # Per-entry retry counts (index -> count)
    default_entry_retries: int = 2  # Default retry count for new entries

    # Dependencies (unified dependency tree)
    depends_on: Set[str] = field(default_factory=set)  # Context injection deps
    children: Optional[List[str]] = None  # Virtual children from splitting
    is_virtual: bool = False          # Created from splitting (not persisted)
    is_aggregation: bool = False      # True = waiting for children to aggregate
    aggregates_to: Optional[str] = None  # Parent to aggregate into
    content: str = ""                 # Content (for virtual units)

    # Attempts for longest fallback
    attempts: List[Tuple[str, int]] = field(default_factory=list)

    # Thread safety
    _lock: Lock = field(default_factory=Lock, repr=False)

    def can_retry(self, error_type: ErrorType) -> bool:
        """Check if retry is possible for this error type."""
        with self._lock:
            return (
                self.total_quota > 0 and
                self.quotas.get(error_type, 0) > 0 and
                len(self.chain) > 0
            )

    def apply_effect(self, effect: ErrorEffect, current_entry: Optional[ChainEntry] = None):
        """
        Apply error effect to state.

        This modifies:
        - chain: removes models based on effect OR when per-entry retry exhausted
        - quotas: decrements appropriate quota
        - total_quota: decrements

        Per-entry retry logic:
        - Each failure decrements current entry's retry count
        - When count reaches 0, entry is removed (regardless of remove_current_model flag)
        - This provides uniform batch/online fallback behavior
        """
        with self._lock:
            if effect.remove_provider and current_entry:
                # Safety block: remove all entries from same provider
                self.chain = [e for e in self.chain if e.provider != current_entry.provider]
                # Clear entry_retries since indices changed
                self.entry_retries.clear()
            elif effect.remove_current_model and current_entry:
                # Remove only current model immediately
                if self.chain and self.chain[0] == current_entry:
                    self.chain.pop(0)
                    # Shift indices in entry_retries
                    self.entry_retries = {k-1: v for k, v in self.entry_retries.items() if k > 0}
            else:
                # Per-entry retry: decrement current entry's count
                if self.chain:
                    # Use entry's configured retries, or default (1 for batch, 2 for online)
                    current_entry = self.chain[0]
                    if current_entry.retries is not None:
                        default_for_entry = current_entry.retries
                    else:
                        default_for_entry = 1 if current_entry.mode == "batch" else self.default_entry_retries
                    current_retries = self.entry_retries.get(0, default_for_entry)
                    current_retries -= 1
                    if current_retries <= 0:
                        # Entry exhausted, remove it
                        self.chain.pop(0)
                        # Shift indices
                        self.entry_retries = {k-1: v for k, v in self.entry_retries.items() if k > 0}
                    else:
                        self.entry_retries[0] = current_retries

            if effect.remove_all_batch:
                # Remove all batch entries
                self.chain = [e for e in self.chain if e.mode != "batch"]
                self.entry_retries.clear()

            # Decrement quotas
            if effect.quota_type in self.quotas:
                self.quotas[effect.quota_type] = max(0, self.quotas[effect.quota_type] - 1)
            self.total_quota = max(0, self.total_quota - 1)

    def record_attempt(self, result: str):
        """Record an attempt for longest fallback."""
        with self._lock:
            self.attempts.append((result, len(result)))

    def get_longest(self) -> Optional[str]:
        """Get the longest result from attempts."""
        with self._lock:
            if not self.attempts:
                return None
            return max(self.attempts, key=lambda x: x[1])[0]

    def has_batch_available(self) -> bool:
        """Check if any batch mode is available."""
        with self._lock:
            return any(e.mode == "batch" for e in self.chain)

    def has_online_available(self) -> bool:
        """Check if any online mode is available."""
        with self._lock:
            return any(e.mode == "online" for e in self.chain)

    def get_current_entry(self) -> Optional[ChainEntry]:
        """Get the current (first) chain entry."""
        with self._lock:
            return self.chain[0] if self.chain else None

    def get_current_mode(self) -> Optional[str]:
        """Get the mode of current entry."""
        entry = self.get_current_entry()
        return entry.mode if entry else None

    def get_batch_entry(self) -> Optional[ChainEntry]:
        """Get the first batch entry."""
        with self._lock:
            for e in self.chain:
                if e.mode == "batch":
                    return e
            return None

    def get_online_entry(self) -> Optional[ChainEntry]:
        """Get the first online entry."""
        with self._lock:
            for e in self.chain:
                if e.mode == "online":
                    return e
            return None

    def remove_batch_entries(self):
        """Remove all batch entries (when < threshold failures)."""
        with self._lock:
            self.chain = [e for e in self.chain if e.mode != "batch"]


def create_unit_state(
    chain: List[ChainEntry],
    quota_config: QuotaConfig,
    content: str = "",
    depends_on: Optional[Set[str]] = None,
    is_virtual: bool = False,
    aggregates_to: Optional[str] = None,
) -> UnitState:
    """
    Factory function to create UnitState.

    Args:
        chain: Model chain (will be copied)
        quota_config: Quota configuration
        content: Unit content
        depends_on: Dependencies for context injection
        is_virtual: Whether unit is virtual (from splitting)
        aggregates_to: Parent unit for aggregation

    Returns:
        New UnitState instance
    """
    return UnitState(
        chain=list(chain),  # Copy chain
        total_quota=quota_config.total,
        quotas=quota_config.create_quotas(),
        default_entry_retries=quota_config.per_entry,
        depends_on=depends_on or set(),
        content=content,
        is_virtual=is_virtual,
        aggregates_to=aggregates_to,
    )


# Chain operations

def remove_batch_entries(chain: List[ChainEntry]) -> List[ChainEntry]:
    """Remove all batch entries from chain."""
    return [e for e in chain if e.mode != "batch"]


def remove_provider(chain: List[ChainEntry], provider: str) -> List[ChainEntry]:
    """Remove all entries from specified provider."""
    return [e for e in chain if e.provider != provider]


def chain_from_model_configs(configs: List[Dict[str, Any]]) -> List[ChainEntry]:
    """
    Create chain from model config dicts.

    Each config can have a 'mode' key (defaults to 'online').
    """
    chain = []
    for config in configs:
        mode = config.get("mode", "online")
        chain.append(ChainEntry(
            provider=config["provider"],
            model=config["model"],
            mode=mode,
        ))
    return chain
