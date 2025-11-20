"""
WorkUnit - Represents a single unit of work for parallel processing.

This module provides:
- WorkUnit dataclass for representing processing tasks
- WorkUnitDiscovery for discovering all work units from input directory
- Support for both single files and multi-part files
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger
import tiktoken

from ...chapter_identity import ChapterIdentity

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class WorkUnit:
    """
    Represents a single unit of work (file or part).

    A work unit is the atomic unit of processing that can be submitted
    to the thread pool. It can be either a whole file or a part of a file.
    """
    id: str                           # e.g., "chapter_5" or "chapter_5.part2"
    file_key: str                     # e.g., "chapter_5"
    part_index: Optional[int]         # None for single-file, 1+ for parts
    total_parts: int                  # 1 for single-file
    content: str                      # Content to process
    input_path: Path                  # Source file path
    output_path: Path                 # Destination file path
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite units
    priority: int = 0                 # Lower = higher priority (for scheduling)
    token_count: int = 0              # Token count for content

    def __post_init__(self):
        """Calculate token count after initialization."""
        if self.token_count == 0 and self.content:
            self.token_count = len(tokenizer.encode(self.content))

    @property
    def is_part(self) -> bool:
        """Check if this is a part of a multi-part file."""
        return self.part_index is not None

    @property
    def is_first_part(self) -> bool:
        """Check if this is the first part of a multi-part file."""
        return self.part_index == 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "file_key": self.file_key,
            "part_index": self.part_index,
            "total_parts": self.total_parts,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "dependencies": self.dependencies,
            "priority": self.priority,
            "token_count": self.token_count
        }


class WorkUnitDiscovery:
    """
    Discovers all work units from input directory.

    Handles:
    - Single files without parts
    - Files with existing part files (e.g., chapter_5.part1.md, chapter_5.part2.md)
    - Building dependency graph for context injection
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        inject_context: bool = False,
        splits_dir: Optional[Path] = None
    ):
        """
        Initialize the discovery.

        Args:
            input_dir: Directory containing input markdown files
            output_dir: Directory for output files
            inject_context: If True, parts depend on previous parts
            splits_dir: Directory containing split input files (defaults to output_dir/splits)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.inject_context = inject_context
        self.splits_dir = splits_dir or (output_dir / "splits")

    def discover_all_units(self) -> List[WorkUnit]:
        """
        Discover all work units from input directory and splits directory.

        Priority:
        1. Part files in splits_dir (from proactive splitting in current stage)
        2. Part files in input_dir (from previous stage)
        3. Single files in input_dir

        Returns:
            List of WorkUnit objects with dependencies resolved
        """
        units = []

        # Find all markdown files from input directory
        all_input_files = sorted(self.input_dir.glob("*.md"))

        # Find split files from splits directory (if exists)
        split_files = []
        if self.splits_dir.exists():
            split_files = sorted(self.splits_dir.glob("*.md"))

        if not all_input_files and not split_files:
            logger.warning(f"No markdown files found in {self.input_dir} or {self.splits_dir}")
            return units

        # Group input files by base name
        input_groups = self._group_files_by_base(all_input_files)

        # Group split files by base name
        split_groups = self._group_files_by_base(split_files) if split_files else {}

        # Get all unique base names
        all_base_names = set(input_groups.keys()) | set(split_groups.keys())

        for base_name in all_base_names:
            input_files = input_groups.get(base_name, [])
            splits = split_groups.get(base_name, [])

            # Check for part files - prefer splits_dir over input_dir
            split_part_files = [f for f in splits if '.part' in f.stem]
            input_part_files = [f for f in input_files if '.part' in f.stem]
            combined_file = [f for f in input_files if '.part' not in f.stem]

            if split_part_files:
                # Use part files from splits directory (current stage's proactive split)
                part_units = self._create_part_units(base_name, split_part_files)
                units.extend(part_units)
                logger.debug(f"Created {len(part_units)} part units for {base_name} from splits_dir")
            elif input_part_files:
                # Use part files from input directory (previous stage's output)
                part_units = self._create_part_units(base_name, input_part_files)
                units.extend(part_units)
                logger.debug(f"Created {len(part_units)} part units for {base_name} from input_dir")
            elif combined_file:
                # Single file - create one unit
                unit = self._create_single_file_unit(combined_file[0])
                units.append(unit)
                logger.debug(f"Created single unit for {base_name}")

        logger.info(f"Discovered {len(units)} work units from {len(all_base_names)} files")
        return units

    def _group_files_by_base(self, files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group files by their base name.

        Example:
            chapter_5.md, chapter_5.part1.md, chapter_5.part2.md
            -> {"chapter_5": [all three files]}
        """
        groups: Dict[str, List[Path]] = {}

        for file_path in files:
            # Parse the filename to get base name
            identity = ChapterIdentity.parse(file_path.stem)
            if identity:
                base_name = identity.base_name
            else:
                # Fallback: remove .partN suffix
                stem = file_path.stem
                if '.part' in stem:
                    base_name = stem.split('.part')[0]
                else:
                    base_name = stem

            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(file_path)

        return groups

    def _create_single_file_unit(self, file_path: Path) -> WorkUnit:
        """Create a work unit for a single file (no parts)."""
        file_key = file_path.stem
        content = file_path.read_text(encoding='utf-8')

        return WorkUnit(
            id=file_key,
            file_key=file_key,
            part_index=None,
            total_parts=1,
            content=content,
            input_path=file_path,
            output_path=self.output_dir / file_path.name,
            dependencies=[],
            priority=0
        )

    def _create_part_units(self, base_name: str, part_files: List[Path]) -> List[WorkUnit]:
        """
        Create work units for part files with dependencies.

        Args:
            base_name: Base file name (e.g., "chapter_5")
            part_files: List of part file paths

        Returns:
            List of WorkUnit objects with proper dependencies
        """
        # Sort part files by part number
        sorted_parts = sorted(part_files, key=lambda p: self._extract_part_number(p.stem))
        total_parts = len(sorted_parts)

        part_units = []
        for i, file_path in enumerate(sorted_parts):
            part_num = i + 1
            content = file_path.read_text(encoding='utf-8')

            # Build unit ID
            unit_id = f"{base_name}.part{part_num}"

            # Determine dependencies
            dependencies = []
            if self.inject_context and i > 0:
                # Depends on previous part for context injection
                dependencies = [f"{base_name}.part{part_num - 1}"]

            unit = WorkUnit(
                id=unit_id,
                file_key=base_name,
                part_index=part_num,
                total_parts=total_parts,
                content=content,
                input_path=file_path,
                output_path=self.output_dir / file_path.name,
                dependencies=dependencies,
                priority=part_num  # Later parts have lower priority
            )
            part_units.append(unit)

        return part_units

    def _extract_part_number(self, stem: str) -> int:
        """Extract part number from filename stem."""
        identity = ChapterIdentity.parse(stem)
        if identity and identity.part:
            return identity.part

        # Fallback: parse .partN suffix
        if '.part' in stem:
            try:
                part_str = stem.split('.part')[1]
                return int(part_str)
            except (ValueError, IndexError):
                pass

        return 0

    def get_ready_units(self, units: List[WorkUnit], completed_ids: set) -> List[WorkUnit]:
        """
        Get units that are ready to process (all dependencies satisfied).

        Args:
            units: All work units
            completed_ids: Set of completed unit IDs

        Returns:
            List of units ready to process
        """
        ready = []
        for unit in units:
            # Check if all dependencies are completed
            if all(dep in completed_ids for dep in unit.dependencies):
                if unit.id not in completed_ids:
                    ready.append(unit)

        return ready

    def group_units_by_file(self, units: List[WorkUnit]) -> Dict[str, List[WorkUnit]]:
        """
        Group work units by their file key.

        Useful for aggregation after processing.
        """
        groups: Dict[str, List[WorkUnit]] = {}
        for unit in units:
            if unit.file_key not in groups:
                groups[unit.file_key] = []
            groups[unit.file_key].append(unit)

        # Sort each group by part index
        for file_key in groups:
            groups[file_key].sort(key=lambda u: u.part_index or 0)

        return groups
