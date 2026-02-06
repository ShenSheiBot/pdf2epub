"""
Work unit loaders for different input formats.
"""

from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

from ..executor import WorkUnit
from ..types import is_sub_key


class PartBasedLoader:
    """
    Loads work units from part-based files.

    Key design (per v2):
    - If parts exist (.part1, .part2), skip the parent file
    - Load parts instead of parent when parts exist
    - Prevents duplicate processing

    Expected naming: chapter_1.part1.md, chapter_1.part2.md, etc. (1-indexed)
    """

    def load_units(
        self,
        input_dir: Path,
        pattern: str = "*.md"
    ) -> List[WorkUnit]:
        """
        Load work units from directory.

        Key logic: if parts exist, skip parent file.

        Args:
            input_dir: Directory to load from
            pattern: Glob pattern for files

        Returns:
            List of WorkUnit sorted by ID
        """
        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return []

        # First pass: collect all files and identify parts
        all_files = sorted(input_dir.glob(pattern))
        part_files = set()
        base_to_parts: Dict[str, List[Path]] = {}

        for path in all_files:
            if not path.is_file():
                continue

            stem = path.stem
            # Skip .sub files (Executor's virtual splits)
            # Use is_sub_key() for reliable detection (matches .sub + digits pattern)
            if is_sub_key(stem):
                continue

            if '.part' in stem:
                # This is a part file
                base_name = stem.rsplit('.part', 1)[0]
                part_files.add(path)
                if base_name not in base_to_parts:
                    base_to_parts[base_name] = []
                base_to_parts[base_name].append(path)

        # Second pass: load files, skipping parents that have parts
        units = []
        for path in all_files:
            if not path.is_file():
                continue

            stem = path.stem

            # Skip .sub files (Executor's virtual splits)
            if is_sub_key(stem):
                continue

            # If this is NOT a part file, check if parts exist
            if '.part' not in stem:
                if stem in base_to_parts:
                    # Parts exist - skip this parent file
                    logger.debug(f"Skipping {stem} (has {len(base_to_parts[stem])} parts)")
                    continue

            # Load this file
            try:
                content = path.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
                continue

            # Determine file_key (base name without .part suffix)
            if '.part' in stem:
                file_key = stem.rsplit('.part', 1)[0]
                # Extract part_index from stem (e.g., "ch1.part2" -> 2)
                part_str = stem.rsplit('.part', 1)[1]
                part_index = int(part_str) if part_str.isdigit() else None
            else:
                file_key = stem
                part_index = None

            units.append(WorkUnit(
                id=stem,
                file_key=file_key,
                content=content,
                input_path=path,
                part_index=part_index,
            ))

        logger.info(f"Loaded {len(units)} units from {input_dir} "
                   f"(skipped {len(base_to_parts)} files with parts)")
        return units


class ChapterBasedLoader:
    """
    Loads work units from chapter files (no parts).

    Expected naming: chapter_1.md, chapter_2.md, etc.
    """

    def load_units(
        self,
        input_dir: Path,
        pattern: str = "*.md"
    ) -> List[WorkUnit]:
        """Load chapters as work units."""
        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return []

        units = []
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file():
                # Filter to only chapter files (not parts)
                if '.part' in path.stem:
                    continue

                unit_id = path.stem
                try:
                    content = path.read_text(encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")
                    continue

                units.append(WorkUnit(
                    id=unit_id,
                    file_key=unit_id,
                    content=content,
                    input_path=path,
                ))

        logger.info(f"Loaded {len(units)} chapters from {input_dir}")
        return units


class HTMLLoader:
    """
    Loads work units from HTML files (for EPUB translation).
    """

    def load_units(
        self,
        input_dir: Path,
        pattern: str = "*.xhtml"
    ) -> List[WorkUnit]:
        """Load HTML files as work units."""
        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return []

        units = []
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file():
                unit_id = path.stem
                try:
                    content = path.read_text(encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")
                    continue

                units.append(WorkUnit(
                    id=unit_id,
                    file_key=unit_id,
                    content=content,
                    input_path=path,
                ))

        logger.info(f"Loaded {len(units)} HTML files from {input_dir}")
        return units
