"""
Result persistence: the ONLY place where files are saved.

Implements two-stage saving:
1. raw/ - LLM responses saved immediately (before validation)
2. validated/ - Promoted after validation passes

This ensures NO data loss even if validation crashes.

FROZEN: This class cannot be inherited or modified.
"""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from ._frozen import Frozen, final, check_final_methods


@check_final_methods
class ResultPersistence(Frozen, frozen=True):
    """
    Result persistence manager.

    FROZEN: Cannot be inherited. All saving goes through here.

    Directory structure:
        output/{book_title}/{processor}/
        ├── raw/           # Immediate save, never deleted
        │   ├── chapter_1.md
        │   └── chapter_2.part1.md
        ├── validated/     # After validation passes
        │   ├── chapter_1.md
        │   └── chapter_2.part1.md
        └── chapter_1.md   # Aggregated final files
    """

    def __init__(self, output_dir: Path):
        """
        Initialize persistence manager.

        Args:
            output_dir: Base output directory (e.g., output/BookTitle/translated)
        """
        self._output_dir = Path(output_dir)
        self._raw_dir = self._output_dir / "raw"
        self._validated_dir = self._output_dir / "validated"

        # Create directories
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._validated_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Persistence initialized: {self._output_dir}")

    @property
    def output_dir(self) -> Path:
        """Base output directory."""
        return self._output_dir

    @property
    def raw_dir(self) -> Path:
        """Directory for raw LLM responses."""
        return self._raw_dir

    @property
    def validated_dir(self) -> Path:
        """Directory for validated results."""
        return self._validated_dir

    # ========== RAW OPERATIONS (before validation) ==========

    @final
    def save_raw(self, key: str, content: str) -> Path:
        """
        Save raw LLM response immediately.

        MUST be called before validation to prevent data loss.
        Raw files are NEVER deleted automatically.

        Args:
            key: File key (e.g., "chapter_1" or "chapter_1.part2")
            content: LLM response content

        Returns:
            Path to saved file
        """
        path = self._raw_dir / f"{key}.md"
        path.write_text(content, encoding='utf-8')
        logger.debug(f"Saved raw: {path.name}")
        return path

    @final
    def save_raw_batch(self, results: Dict[str, str]) -> int:
        """
        Save multiple raw results at once.

        Args:
            results: {key: content} mapping

        Returns:
            Number of files saved
        """
        count = 0
        for key, content in results.items():
            self.save_raw(key, content)
            count += 1
        logger.info(f"Saved {count} raw results to {self._raw_dir}")
        return count

    @final
    def get_raw(self, key: str) -> Optional[str]:
        """
        Read raw content for a key.

        Args:
            key: File key

        Returns:
            Content if exists, None otherwise
        """
        path = self._raw_dir / f"{key}.md"
        if path.exists():
            return path.read_text(encoding='utf-8')
        return None

    @final
    def has_raw(self, key: str) -> bool:
        """Check if raw file exists."""
        return (self._raw_dir / f"{key}.md").exists()

    @final
    def list_raw_keys(self) -> List[str]:
        """List all keys in raw directory."""
        return [f.stem for f in self._raw_dir.glob("*.md")]

    # ========== VALIDATED OPERATIONS (after validation) ==========

    @final
    def promote_to_validated(self, key: str) -> Path:
        """
        Promote a raw file to validated status.

        Called after validation passes. Copies content from raw/ to validated/.

        Args:
            key: File key

        Returns:
            Path to validated file

        Raises:
            FileNotFoundError: If raw file doesn't exist
        """
        raw_path = self._raw_dir / f"{key}.md"
        validated_path = self._validated_dir / f"{key}.md"

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        content = raw_path.read_text(encoding='utf-8')
        validated_path.write_text(content, encoding='utf-8')
        logger.debug(f"Promoted to validated: {key}")
        return validated_path

    @final
    def promote_batch(self, keys: List[str]) -> int:
        """
        Promote multiple files to validated.

        Args:
            keys: List of file keys

        Returns:
            Number of files promoted
        """
        count = 0
        for key in keys:
            try:
                self.promote_to_validated(key)
                count += 1
            except FileNotFoundError:
                logger.warning(f"Cannot promote {key}: raw file not found")
        logger.info(f"Promoted {count} files to validated")
        return count

    @final
    def get_validated(self, key: str) -> Optional[str]:
        """Read validated content for a key."""
        path = self._validated_dir / f"{key}.md"
        if path.exists():
            return path.read_text(encoding='utf-8')
        return None

    @final
    def has_validated(self, key: str) -> bool:
        """Check if validated file exists."""
        return (self._validated_dir / f"{key}.md").exists()

    @final
    def list_validated_keys(self) -> List[str]:
        """List all keys in validated directory."""
        return [f.stem for f in self._validated_dir.glob("*.md")]

    # ========== AGGREGATION ==========

    @final
    def aggregate_parts(self, base_key: str, part_keys: List[str]) -> Optional[Path]:
        """
        Aggregate multiple part files into one.

        Only aggregates if ALL parts are validated.

        Args:
            base_key: Base file key (e.g., "chapter_1")
            part_keys: List of part keys (e.g., ["chapter_1.part1", "chapter_1.part2"])

        Returns:
            Path to aggregated file, or None if not all parts validated
        """
        # Sort part keys by part number
        def get_part_num(key: str) -> int:
            if '.part' in key:
                try:
                    return int(key.split('.part')[1])
                except (ValueError, IndexError):
                    return 0
            return 0

        sorted_keys = sorted(part_keys, key=get_part_num)

        # Check all parts are validated
        missing = [k for k in sorted_keys if not self.has_validated(k)]
        if missing:
            logger.debug(f"Cannot aggregate {base_key}: missing validated parts {missing}")
            return None

        # Read and combine
        parts = []
        for key in sorted_keys:
            content = self.get_validated(key)
            if content:
                parts.append(content)

        if not parts:
            return None

        # Write aggregated file
        output_path = self._output_dir / f"{base_key}.md"
        output_path.write_text('\n\n'.join(parts), encoding='utf-8')
        logger.info(f"Aggregated {len(parts)} parts into {base_key}.md")
        return output_path

    # ========== REVALIDATION SUPPORT ==========

    @final
    def get_keys_needing_revalidation(self) -> List[str]:
        """
        Get keys that have raw files but no validated files.

        Used for --resume to revalidate incomplete processing.

        Returns:
            List of keys needing revalidation
        """
        raw_keys = set(self.list_raw_keys())
        validated_keys = set(self.list_validated_keys())
        return list(raw_keys - validated_keys)

    # ========== SAVE WITH WARNING (for fallback) ==========

    @final
    def save_with_warning(self, key: str, content: str, warning: str) -> Path:
        """
        Save content with a warning header.

        Used for longest-fallback when validation fails after max retries.

        Args:
            key: File key
            content: Content to save
            warning: Warning message to prepend

        Returns:
            Path to saved file
        """
        header = f"<!-- WARNING: {warning} -->\n\n"
        full_content = header + content

        # Save to validated (so it's counted as complete)
        path = self._validated_dir / f"{key}.md"
        path.write_text(full_content, encoding='utf-8')
        logger.warning(f"Saved {key} with warning: {warning}")
        return path
