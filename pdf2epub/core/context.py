"""
ContextInjector: Context injection for part-to-part consistency.

This module handles:
1. Extracting context from completed parts
2. Injecting previous part's context into current part

CRITICAL: Without context injection, translation terms will be inconsistent
across parts. Names, places, and terminology will vary randomly.
This makes the entire translation unusable.

Note: Dependency-based scheduling is handled by Executor._get_ready_ids()
using the .part naming convention directly, not via this class.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from loguru import logger

from ._frozen import Frozen, final, check_final_methods
from ._protocol import ProcessContext
from .types import WorkUnit


def _get_prev_part_id(unit_id: str) -> Optional[str]:
    """
    Get the previous sibling part ID from a unit ID.

    Handles nested splits correctly by deriving from unit.id, not file_key.

    Examples:
        "chapter_1.part2" -> "chapter_1.part1"
        "chapter_1.part1.part2" -> "chapter_1.part1.part1"
        "chapter_1.part1" -> None (no previous)
        "chapter_1" -> None (not a part)

    Args:
        unit_id: The unit ID to find previous sibling for

    Returns:
        Previous part ID, or None if no previous part exists
    """
    if '.part' not in unit_id:
        return None

    base, last_part = unit_id.rsplit('.part', 1)
    if not last_part.isdigit():
        return None

    prev_part_num = int(last_part) - 1
    if prev_part_num < 1:
        return None

    return f"{base}.part{prev_part_num}"


@check_final_methods
class ContextInjector(Frozen, frozen=True):
    """
    Context injector for part-to-part consistency.

    FROZEN: Cannot be inherited or modified.

    Features:
    1. Context extraction from completed parts
    2. Context injection into ProcessContext
    3. Caching of completed results for later injection

    Modes:
    - "parallel": No context injection, parts processed independently
    - "sequential": Previous part's output injected for terminology consistency

    Note: Dependency-based scheduling (which units are "ready") is handled
    by Executor._get_ready_ids() using .part naming, not by this class.
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save', 'build_prompt'}

    def __init__(
        self,
        mode: str = "parallel",
        persistence: Optional[Any] = None  # ResultPersistence, avoid circular import
    ):
        """
        Initialize context injector.

        Args:
            mode: "parallel" (no injection) or "sequential" (with injection)
            persistence: ResultPersistence for loading completed results
        """
        if mode not in ("parallel", "sequential"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'parallel' or 'sequential'")

        self._mode = mode
        self._persistence = persistence

        # Cache for completed results
        self._completed_cache: Dict[str, Tuple[str, str]] = {}  # id -> (original, processed)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_sequential(self) -> bool:
        return self._mode == "sequential"

    @final
    def get_context_for_unit(
        self,
        unit: WorkUnit,
        completed_results: Dict[str, str],
        originals: Dict[str, str]
    ) -> Optional[Tuple[str, str]]:
        """
        Get context (previous original, previous processed) for a unit.

        Args:
            unit: The work unit
            completed_results: Dict of completed unit_id -> processed content
            originals: Dict of unit_id -> original content

        Returns:
            Tuple of (previous_original, previous_processed) or None
        """
        if self._mode == "parallel":
            return None

        # Get previous part's ID (handles nested splits correctly)
        prev_id = _get_prev_part_id(unit.id)
        if prev_id is None:
            return None

        if prev_id in completed_results and prev_id in originals:
            return (originals[prev_id], completed_results[prev_id])

        # Try to load from persistence
        if self._persistence is not None:
            try:
                processed = self._persistence.get_validated(prev_id)
                # We need original too - check cache or load from input
                original = originals.get(prev_id)
                if processed and original:
                    return (original, processed)
            except Exception as e:
                logger.debug(f"Could not load context for {prev_id}: {e}")

        return None

    @final
    def inject_context(
        self,
        context: ProcessContext,
        previous_original: str,
        previous_processed: str
    ) -> ProcessContext:
        """
        Inject previous part's context into ProcessContext.

        Args:
            context: Current ProcessContext
            previous_original: Previous part's original content
            previous_processed: Previous part's processed content

        Returns:
            New ProcessContext with injected context
        """
        return context.with_previous_context(previous_original, previous_processed)

    @final
    def cache_completed(
        self,
        unit_id: str,
        original: str,
        processed: str
    ) -> None:
        """
        Cache a completed result for later context injection.

        Args:
            unit_id: Unit identifier
            original: Original content
            processed: Processed content
        """
        self._completed_cache[unit_id] = (original, processed)

    @final
    def get_cached(self, unit_id: str) -> Optional[Tuple[str, str]]:
        """Get cached result."""
        return self._completed_cache.get(unit_id)

    @final
    def clear_cache(self, unit_id: Optional[str] = None) -> None:
        """Clear cache for a unit or all units."""
        if unit_id:
            self._completed_cache.pop(unit_id, None)
        else:
            self._completed_cache.clear()


def sort_by_dependencies(units: List[WorkUnit]) -> List[WorkUnit]:
    """
    Sort units so that dependencies come before dependents.

    Uses topological sort to ensure proper ordering.

    Args:
        units: List of work units

    Returns:
        Sorted list of work units
    """
    # Build dependency graph
    unit_map = {u.id: u for u in units}
    deps: Dict[str, Set[str]] = defaultdict(set)

    for unit in units:
        # Use unit.id to derive previous sibling (handles nested splits)
        prev_id = _get_prev_part_id(unit.id)
        if prev_id and prev_id in unit_map:
            deps[unit.id].add(prev_id)

    # Topological sort
    sorted_units = []
    visited: Set[str] = set()
    temp_visited: Set[str] = set()

    def visit(unit_id: str):
        if unit_id in temp_visited:
            raise ValueError(f"Circular dependency detected for {unit_id}")
        if unit_id in visited:
            return

        temp_visited.add(unit_id)

        for dep_id in deps[unit_id]:
            if dep_id in unit_map:
                visit(dep_id)

        temp_visited.remove(unit_id)
        visited.add(unit_id)
        sorted_units.append(unit_map[unit_id])

    # Visit all units
    for unit in units:
        if unit.id not in visited:
            visit(unit.id)

    return sorted_units


def group_by_file_key(units: List[WorkUnit]) -> Dict[str, List[WorkUnit]]:
    """
    Group units by their base file key.

    Args:
        units: List of work units

    Returns:
        Dict mapping file_key to list of units (sorted by part_index)
    """
    groups: Dict[str, List[WorkUnit]] = defaultdict(list)

    for unit in units:
        groups[unit.file_key].append(unit)

    # Sort each group by part_index
    for file_key in groups:
        groups[file_key].sort(key=lambda u: u.part_index or 0)

    return groups
