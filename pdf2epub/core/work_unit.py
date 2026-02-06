"""
WorkUnit: Complete work unit with ALL fields from old architecture.

This module provides:
- SplitType enum for structural split constraints
- WorkUnit dataclass for representing processing tasks
- WorkUnitDiscovery for discovering all work units from input directory
- Support for both single files and multi-part files
- Special units (TOC, metadata) that don't need file I/O

IMPORTANT: WorkUnit and SplitType are re-exported from core/types.py for convenience.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Set
from dataclasses import dataclass, field
from loguru import logger
import tiktoken

from ..chapter_identity import ChapterIdentity


# ============================================================
# Split Type (structural constraint)
# ============================================================

class SplitType(Enum):
    """
    Split type - structural constraint.

    NONE: Original unsplit unit
    PROACTIVE: Pipeline's .part split (persisted to disk)
    DYNAMIC: Executor's .sub split (virtual, not persisted)
    """
    NONE = "none"
    PROACTIVE = "proactive"
    DYNAMIC = "dynamic"

# Initialize tokenizer
_tokenizer = tiktoken.get_encoding("cl100k_base")


# Unit type constants
UNIT_TYPE_CONTENT = "content"  # Regular content (chapter, section, etc.)
UNIT_TYPE_TOC = "toc"  # Table of contents
UNIT_TYPE_METADATA = "metadata"  # Book metadata

# Reserved IDs for special units
TOC_UNIT_ID = "__toc__"
METADATA_UNIT_ID = "__metadata__"


@dataclass
class WorkUnit:
    """
    A complete work unit for processing.

    Contains ALL fields from the old architecture - nothing omitted.

    Unit Types:
    - "content": Regular content (chapters, sections)
    - "toc": Table of contents (no file I/O, no validation)
    - "metadata": Book metadata translation
    """
    # === Basic identification ===
    id: str  # e.g., "chapter_1" or "chapter_1.part2" or "__toc__"
    file_key: str  # Base file key without part suffix, e.g., "chapter_1"
    content: str  # Content to process

    # === Unit type ===
    unit_type: str = UNIT_TYPE_CONTENT  # "content", "toc", "metadata"

    # === File paths ===
    input_path: Optional[Path] = None  # Source file path (None for special units)
    output_path: Optional[Path] = None  # Destination file path (None for special units)

    # === Part information ===
    part_index: Optional[int] = None  # 1-based part number, None for single file
    total_parts: int = 1  # Total parts (1 for single file)
    split_version: int = 0  # Current split version number

    # === Dependency and scheduling ===
    dependencies: Tuple[str, ...] = ()  # List of prerequisite unit IDs (for context injection)
    priority: int = 0  # For scheduling (lower = higher priority)

    # === Token information ===
    token_count: int = 0  # Cached token count

    # === Chapter metadata (from BookStructure) ===
    chapter_type: Optional[str] = None  # "chapter", "notes", "appendix", "front_matter", "back_matter"
    chapter_title: Optional[str] = None  # Chapter title from structure
    chapter_number: Optional[str] = None  # "5" or "7.1.1"

    # === TOC and page info ===
    toc_path: Optional[str] = None  # Path in table of contents
    page_range: Optional[Tuple[int, int]] = None  # (start_page, end_page)

    # === Footnotes ===
    footnote_refs: Tuple[int, ...] = ()  # Referenced footnote numbers
    footnotes: Optional[Dict[int, str]] = None  # Footnote definitions {number: content}

    # === Split type (structural constraint from Phase 2) ===
    split_type: SplitType = SplitType.NONE  # NONE, PROACTIVE (.part), DYNAMIC (.sub)
    parent_id: Optional[str] = None  # Parent unit ID (for split units)

    def __post_init__(self):
        """Validate naming conventions - fail fast."""
        self._validate_naming()

    def _validate_naming(self):
        """Verify naming matches split_type - structural constraint."""
        if self.split_type == SplitType.PROACTIVE:
            if ".part" not in self.id or ".sub" in self.id:
                raise ValueError(
                    f"Proactive split must use .part naming (not .sub): {self.id}"
                )
        elif self.split_type == SplitType.DYNAMIC:
            if ".sub" not in self.id:
                raise ValueError(
                    f"Dynamic split must use .sub naming: {self.id}"
                )

    @property
    def is_part(self) -> bool:
        """Whether this is a part of a multi-part file."""
        return self.part_index is not None

    @property
    def is_first_part(self) -> bool:
        """Whether this is the first part."""
        return self.part_index == 1

    @property
    def is_last_part(self) -> bool:
        """Whether this is the last part."""
        return self.part_index == self.total_parts

    @property
    def is_front_back_matter(self) -> bool:
        """Whether this is front/back matter (may skip validation)."""
        return self.chapter_type in ("front_matter", "back_matter", "notes", "appendix")

    @property
    def is_toc(self) -> bool:
        """Whether this is a TOC unit."""
        return self.unit_type == UNIT_TYPE_TOC

    @property
    def is_metadata(self) -> bool:
        """Whether this is a metadata unit."""
        return self.unit_type == UNIT_TYPE_METADATA

    @property
    def is_special(self) -> bool:
        """Whether this is a special unit (TOC, metadata) - no file I/O."""
        return self.unit_type != UNIT_TYPE_CONTENT

    @property
    def skip_validation(self) -> bool:
        """Whether to skip validation for this unit."""
        return self.is_special or self.is_front_back_matter

    @property
    def previous_part_id(self) -> Optional[str]:
        """Get the ID of the previous part, if this is a multi-part file."""
        if self.part_index is not None and self.part_index > 1:
            return f"{self.file_key}.part{self.part_index - 1}"
        return None

    @property
    def next_part_id(self) -> Optional[str]:
        """Get the ID of the next part, if this is a multi-part file."""
        if self.part_index is not None and self.part_index < self.total_parts:
            return f"{self.file_key}.part{self.part_index + 1}"
        return None

    def with_split(
        self,
        part_index: int,
        total_parts: int,
        content: str,
        split_version: int
    ) -> "WorkUnit":
        """Create a new WorkUnit for a split part."""
        return WorkUnit(
            id=f"{self.file_key}.part{part_index}",
            file_key=self.file_key,
            content=content,
            unit_type=self.unit_type,  # Preserve unit type
            input_path=self.input_path,
            output_path=self.output_path,
            part_index=part_index,
            total_parts=total_parts,
            split_version=split_version,
            dependencies=(f"{self.file_key}.part{part_index - 1}",) if part_index > 1 else (),
            priority=self.priority + part_index,  # Later parts have lower priority
            token_count=0,  # Will be recalculated
            chapter_type=self.chapter_type,
            chapter_title=self.chapter_title,
            chapter_number=self.chapter_number,
            toc_path=self.toc_path,
            page_range=self.page_range,
            footnote_refs=self.footnote_refs,
            footnotes=self.footnotes
        )


def create_toc_unit(content: str, priority: int = -100) -> WorkUnit:
    """
    Create a TOC WorkUnit.

    Args:
        content: JSON string of TOC entries to translate
        priority: Scheduling priority (default -100 = high priority)

    Returns:
        WorkUnit for TOC translation
    """
    return WorkUnit(
        id=TOC_UNIT_ID,
        file_key=TOC_UNIT_ID,
        content=content,
        unit_type=UNIT_TYPE_TOC,
        input_path=None,  # No file I/O for TOC
        output_path=None,
        part_index=None,
        total_parts=1,
        dependencies=(),  # TOC has no dependencies
        priority=priority,  # High priority to process early
    )


def create_metadata_unit(content: str, priority: int = -100) -> WorkUnit:
    """
    Create a metadata WorkUnit.

    Args:
        content: JSON string of metadata to translate
        priority: Scheduling priority (default -100 = high priority)

    Returns:
        WorkUnit for metadata translation
    """
    return WorkUnit(
        id=METADATA_UNIT_ID,
        file_key=METADATA_UNIT_ID,
        content=content,
        unit_type=UNIT_TYPE_METADATA,
        input_path=None,  # No file I/O for metadata
        output_path=None,
        part_index=None,
        total_parts=1,
        dependencies=(),  # Metadata has no dependencies
        priority=priority,
    )


@dataclass
class ProcessingResult:
    """Result of processing operation."""
    total: int
    completed: int
    failed: int
    failed_keys: List[str] = field(default_factory=list)
    revalidated: int = 0  # Keys recovered from raw/
    safety_blocked: int = 0  # Keys blocked by content filter
    recovered_by_fallback: int = 0  # Keys recovered by online/longest fallback

    # Token statistics
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def success_rate(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0

    @property
    def all_succeeded(self) -> bool:
        return self.failed == 0


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
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.inject_context = inject_context
        self.splits_dir = Path(splits_dir) if splits_dir else (self.output_dir / "splits")

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
        token_count = len(_tokenizer.encode(content))

        return WorkUnit(
            id=file_key,
            file_key=file_key,
            content=content,
            unit_type=UNIT_TYPE_CONTENT,
            input_path=file_path,
            output_path=self.output_dir / file_path.name,
            part_index=None,
            total_parts=1,
            dependencies=(),
            priority=0,
            token_count=token_count,
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
            token_count = len(_tokenizer.encode(content))

            # Build unit ID
            unit_id = f"{base_name}.part{part_num}"

            # Determine dependencies
            dependencies: Tuple[str, ...] = ()
            if self.inject_context and i > 0:
                # Depends on previous part for context injection
                dependencies = (f"{base_name}.part{part_num - 1}",)

            unit = WorkUnit(
                id=unit_id,
                file_key=base_name,
                content=content,
                unit_type=UNIT_TYPE_CONTENT,
                input_path=file_path,
                output_path=self.output_dir / file_path.name,
                part_index=part_num,
                total_parts=total_parts,
                dependencies=dependencies,
                priority=part_num,  # Later parts have lower priority
                token_count=token_count,
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

    def get_ready_units(
        self,
        units: List[WorkUnit],
        completed_ids: Set[str]
    ) -> List[WorkUnit]:
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
