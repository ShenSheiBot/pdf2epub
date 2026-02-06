"""
Phase protocols and data structures.

Phases are composable processing stages:
- Any phase can follow any other (polish -> translate -> polish)
- No aggregation between phases (only at build-epub)
- Each phase reads parts and outputs parts
"""

from typing import Protocol, List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

if TYPE_CHECKING:
    from ..executor import WorkUnit


@dataclass
class PhaseResult:
    """Result of running a phase."""
    phase: str
    total: int
    completed: int
    failed: int
    failed_keys: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class WorkUnitLoader(Protocol):
    """
    Loads work units from a directory.

    Each phase can use a different loader based on input format.
    """

    def load_units(
        self,
        input_dir: Path,
        pattern: str = "*.md"
    ) -> List["WorkUnit"]:
        """
        Load work units from directory.

        Args:
            input_dir: Directory to load from
            pattern: Glob pattern for files

        Returns:
            List of WorkUnit
        """
        ...
