"""
SplitManager - Central orchestration for content splitting operations.

Provides:
- Get or create splits with version tracking
- Re-splitting with automatic index renumbering
- Content aggregation
- Deterministic splitting (same input = same output)
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger
import tiktoken

from .content_splitter import split_content
from ...core.tracking import ProcessingTracker, SplitRecord, ErrorType
from ...chapter_identity import ChapterIdentity

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class PartInfo:
    """Information about a single part."""
    index: int
    token_count: int
    content_hash: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class SplitResult:
    """Result of a split or resplit operation."""
    parts: List[str]
    part_infos: List[PartInfo]
    version: int
    method: str
    reason: str
    old_to_new_mapping: Optional[Dict[int, List[int]]] = None  # For resplits


class SplitManager:
    """
    Central manager for all content splitting operations.

    Handles:
    - Initial splitting of content
    - Re-splitting when parts fail
    - Version tracking via ProcessingTracker
    - Part index management
    - Content aggregation
    """

    def __init__(
        self,
        tracker: ProcessingTracker,
        output_dir: Path = None,
        default_max_tokens: int = 4000,
        max_resplits: int = 3,
        consecutive_failures_threshold: int = 2,
        # Deprecated parameters for backward compatibility
        llm_client=None,
        model_configs: Optional[List[Dict]] = None
    ):
        """
        Initialize the SplitManager.

        Args:
            tracker: ProcessingTracker for persistence
            output_dir: Directory for part files
            default_max_tokens: Default max tokens per part
            max_resplits: Maximum number of resplit attempts
            consecutive_failures_threshold: Consecutive failures before resplit
            llm_client: Deprecated, ignored
            model_configs: Deprecated, ignored
        """
        self.tracker = tracker
        self.output_dir = Path(output_dir) if output_dir else None
        self.default_max_tokens = default_max_tokens
        self.max_resplits = max_resplits
        self.consecutive_failures_threshold = consecutive_failures_threshold

        # Log deprecation if old params are passed
        if llm_client is not None:
            logger.debug("llm_client parameter is deprecated and ignored")
        if model_configs is not None:
            logger.debug("model_configs parameter is deprecated and ignored")

    def get_or_create_split(
        self,
        base_key: str,
        content: str,
        max_tokens: int = None,
        force_resplit: bool = False,
        strategy: str = "auto"
    ) -> SplitResult:
        """
        Get existing split or create a new one.

        If content hasn't changed (based on hash) and split exists, reuse it.
        Otherwise, create a new split version.

        Args:
            base_key: Base file key (e.g., "chapter_5")
            content: Content to split
            max_tokens: Max tokens per part (uses default if not specified)
            force_resplit: Force a new split even if one exists
            strategy: Splitting strategy ("auto", "general", "academic", "japanese")

        Returns:
            SplitResult with parts and metadata
        """
        max_tokens = max_tokens or self.default_max_tokens
        content_hash = self._hash_content(content)
        actual_tokens = len(tokenizer.encode(content))

        # Check if content fits in a single part
        if actual_tokens <= max_tokens:
            logger.info(f"{base_key}: {actual_tokens:,} tokens fits in single part")
            return SplitResult(
                parts=[content],
                part_infos=[PartInfo(
                    index=1,
                    token_count=actual_tokens,
                    content_hash=content_hash,
                    char_start=0,
                    char_end=len(content)
                )],
                version=0,
                method="no_split",
                reason="content_fits_in_single_part"
            )

        # Check if we can reuse existing split
        if not force_resplit:
            existing = self.tracker.get_current_split(base_key)
            if existing:
                # Check if content hash matches (same content)
                if existing.get("content_hash") == content_hash:
                    logger.info(f"{base_key}: Reusing existing split v{existing['version']}")
                    return self._reconstruct_split_result(base_key, content, existing)

        # Need to create a new split
        logger.info(f"{base_key}: Creating new split ({actual_tokens:,} tokens, max {max_tokens:,})")

        parts = split_content(
            content=content,
            max_tokens=max_tokens,
            strategy=strategy
        )

        # Build part infos
        part_infos = []
        char_pos = 0
        for i, part in enumerate(parts, 1):
            part_start = content.find(part.strip()[:100], char_pos)
            if part_start == -1:
                part_start = char_pos
            part_end = part_start + len(part)

            part_infos.append(PartInfo(
                index=i,
                token_count=len(tokenizer.encode(part)),
                content_hash=self._hash_content(part),
                char_start=part_start,
                char_end=part_end
            ))
            char_pos = part_end

        # Calculate split points (character positions where parts end)
        split_points = [info.char_end for info in part_infos[:-1]]

        # Determine version
        history = self.tracker.get_split_history(base_key)
        version = len(history)

        # Record in tracker
        split_record = SplitRecord(
            timestamp=time.time(),
            split_points=split_points,
            total_tokens=actual_tokens,
            part_count=len(parts),
            method="content_splitter",
            reason="initial_split" if version == 0 else "forced_resplit"
        )

        # Add content hash to the record for future comparison
        self.tracker.record_split(base_key, split_record)

        # Store content hash in split history
        current_split = self.tracker.get_current_split(base_key)
        if current_split:
            current_split["content_hash"] = content_hash
            self.tracker.save()

        return SplitResult(
            parts=parts,
            part_infos=part_infos,
            version=version,
            method="content_splitter",
            reason="initial_split" if version == 0 else "forced_resplit"
        )

    def needs_resplit(
        self,
        base_key: str,
        part_index: int,
        error_type: ErrorType,
        consecutive_failures: int = 3
    ) -> bool:
        """
        Determine if a part needs to be re-split based on error type.

        Args:
            base_key: Base file key
            part_index: Index of the failing part
            error_type: Type of error encountered
            consecutive_failures: Number of consecutive failures

        Returns:
            True if re-splitting should be attempted
        """
        # Errors that might benefit from re-splitting
        resplit_worthy_errors = {
            ErrorType.TRUNCATION,
            ErrorType.TIMEOUT,
            ErrorType.CONTENT_FILTER,  # Maybe smaller chunks won't trigger filter
        }

        if error_type not in resplit_worthy_errors:
            return False

        # Check if we've already hit max resplits
        history = self.tracker.get_split_history(base_key)

        # Limit resplits (prevent infinite loops)
        if len(history) >= self.max_resplits:
            logger.warning(f"{base_key}: Already at max resplits ({self.max_resplits})")
            return False

        # Only resplit after multiple consecutive failures
        if consecutive_failures < self.consecutive_failures_threshold:
            return False

        return True

    def resplit_part(
        self,
        base_key: str,
        part_index: int,
        part_content: str,
        max_tokens: int = None,
        reason: str = "truncation_failure"
    ) -> SplitResult:
        """
        Re-split a specific part into smaller pieces.

        This will:
        1. Split the part content
        2. Renumber all subsequent parts
        3. Create a new split version
        4. Return mapping of old to new indices

        Args:
            base_key: Base file key
            part_index: Index of the part to resplit
            part_content: Content of the part to resplit
            max_tokens: New max tokens (usually smaller)
            reason: Reason for resplit

        Returns:
            SplitResult with new parts and index mapping
        """
        max_tokens = max_tokens or (self.default_max_tokens // 2)  # Halve by default
        current_split = self.tracker.get_current_split(base_key)

        if not current_split:
            raise ValueError(f"No existing split found for {base_key}")

        old_part_count = current_split["part_count"]

        logger.info(f"{base_key}: Re-splitting part {part_index} "
                   f"(was {len(tokenizer.encode(part_content)):,} tokens, "
                   f"new max {max_tokens:,})")

        # Split the part content
        sub_parts = split_content(
            content=part_content,
            max_tokens=max_tokens,
            strategy="markdown"
        )

        num_new_parts = len(sub_parts)

        if num_new_parts <= 1:
            logger.warning(f"{base_key}: Resplit produced only {num_new_parts} parts, no change")
            return SplitResult(
                parts=sub_parts if sub_parts else [part_content],
                part_infos=[PartInfo(
                    index=part_index,
                    token_count=len(tokenizer.encode(part_content)),
                    content_hash=self._hash_content(part_content)
                )],
                version=current_split.get("version", 0),
                method="no_change",
                reason="resplit_produced_single_part"
            )

        # Build part infos for new sub-parts
        part_infos = []
        for i, part in enumerate(sub_parts):
            new_index = part_index + i
            part_infos.append(PartInfo(
                index=new_index,
                token_count=len(tokenizer.encode(part)),
                content_hash=self._hash_content(part)
            ))

        # Calculate old to new mapping
        # - Parts before: unchanged
        # - Part being split: becomes multiple new parts
        # - Parts after: shifted by (num_new_parts - 1)
        old_to_new: Dict[int, List[int]] = {}
        shift = num_new_parts - 1

        for old_idx in range(1, old_part_count + 1):
            if old_idx < part_index:
                old_to_new[old_idx] = [old_idx]
            elif old_idx == part_index:
                old_to_new[old_idx] = list(range(part_index, part_index + num_new_parts))
            else:
                old_to_new[old_idx] = [old_idx + shift]

        new_total_parts = old_part_count + shift

        # Record the resplit in tracker
        # Note: We don't have access to full content here, so split_points are partial
        split_record = SplitRecord(
            timestamp=time.time(),
            split_points=[],  # Would need full content to calculate
            total_tokens=current_split.get("total_tokens", 0),
            part_count=new_total_parts,
            method="resplit",
            reason=reason,
            triggered_by=ChapterIdentity.make_part_name(base_key, part_index)
        )

        self.tracker.record_split(base_key, split_record)

        # Get new version
        history = self.tracker.get_split_history(base_key)
        version = len(history) - 1

        logger.info(f"{base_key}: Resplit complete - part {part_index} → "
                   f"parts {part_index}-{part_index + num_new_parts - 1} "
                   f"(total now {new_total_parts} parts)")

        return SplitResult(
            parts=sub_parts,
            part_infos=part_infos,
            version=version,
            method="resplit",
            reason=reason,
            old_to_new_mapping=old_to_new
        )

    def aggregate(
        self,
        parts: List[str],
        separator: str = "\n\n"
    ) -> str:
        """
        Aggregate parts back into single content.

        Args:
            parts: List of part contents
            separator: Separator between parts

        Returns:
            Combined content
        """
        return separator.join(parts)

    def get_current_parts(self, base_key: str) -> List[str]:
        """
        Get current part keys for a base key.

        Args:
            base_key: Base file key

        Returns:
            List of part keys (e.g., ["chapter_5.part1", "chapter_5.part2"])
        """
        current_split = self.tracker.get_current_split(base_key)

        if not current_split:
            return [base_key]  # Single file, no parts

        part_count = current_split["part_count"]
        return [
            ChapterIdentity.make_part_name(base_key, i)
            for i in range(1, part_count + 1)
        ]

    def get_part_file_paths(
        self,
        base_key: str,
        directory: Path = None
    ) -> List[Path]:
        """
        Get file paths for all parts of a base key.

        Args:
            base_key: Base file key
            directory: Directory to look in (uses output_dir if not specified)

        Returns:
            List of Path objects for part files
        """
        directory = Path(directory) if directory else self.output_dir
        if not directory:
            raise ValueError("No directory specified")

        current_split = self.tracker.get_current_split(base_key)

        if not current_split:
            # Single file
            return [directory / f"{base_key}.md"]

        part_count = current_split["part_count"]
        return [
            directory / f"{base_key}.part{i}.md"
            for i in range(1, part_count + 1)
        ]

    def update_unit_indices_after_resplit(
        self,
        base_key: str,
        old_to_new: Dict[int, List[int]]
    ):
        """
        Update unit tracking after a resplit operation.

        This migrates status from old part indices to new ones.

        Args:
            base_key: Base file key
            old_to_new: Mapping from old indices to new indices
        """
        units = self.tracker.progress.get("units", {})
        updated_units = {}

        for old_idx, new_indices in old_to_new.items():
            old_key = ChapterIdentity.make_part_name(base_key, old_idx)

            if old_key in units:
                old_unit = units[old_key]

                if len(new_indices) == 1 and new_indices[0] == old_idx:
                    # No change for this unit
                    updated_units[old_key] = old_unit
                elif len(new_indices) == 1:
                    # Shifted, just rename
                    new_key = ChapterIdentity.make_part_name(base_key, new_indices[0])
                    updated_units[new_key] = old_unit
                    logger.debug(f"Shifted {old_key} → {new_key}")
                else:
                    # Split into multiple - old status becomes invalid
                    # New parts start fresh
                    for new_idx in new_indices:
                        new_key = ChapterIdentity.make_part_name(base_key, new_idx)
                        updated_units[new_key] = {
                            "status": "pending",
                            "attempts": [],
                            "retry_count": 0,
                            "resplit_from": old_key
                        }
                    logger.debug(f"Split {old_key} → {[ChapterIdentity.make_part_name(base_key, i) for i in new_indices]}")

        # Update tracker
        for key in list(units.keys()):
            if key.startswith(f"{base_key}.part"):
                del units[key]

        units.update(updated_units)
        self.tracker.save()

    def _hash_content(self, content: str) -> str:
        """Generate hash of content for comparison."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _reconstruct_split_result(
        self,
        base_key: str,
        content: str,
        split_info: dict
    ) -> SplitResult:
        """
        Reconstruct SplitResult from saved split info.

        Args:
            base_key: Base file key
            content: Full content
            split_info: Saved split info from tracker

        Returns:
            SplitResult reconstructed from saved data
        """
        split_points = split_info.get("split_points", [])

        # Reconstruct parts from split points
        parts = []
        positions = [0] + split_points + [len(content)]

        part_infos = []
        for i in range(len(positions) - 1):
            part = content[positions[i]:positions[i + 1]].strip()
            parts.append(part)

            part_infos.append(PartInfo(
                index=i + 1,
                token_count=len(tokenizer.encode(part)),
                content_hash=self._hash_content(part),
                char_start=positions[i],
                char_end=positions[i + 1]
            ))

        return SplitResult(
            parts=parts,
            part_infos=part_infos,
            version=split_info.get("version", 0),
            method=split_info.get("method", "reconstructed"),
            reason=split_info.get("reason", "reconstructed_from_history")
        )

    def save_parts(
        self,
        base_key: str,
        parts: List[str],
        directory: Path = None
    ) -> List[Path]:
        """
        Save parts to individual files.

        Args:
            base_key: Base file key
            parts: List of part contents
            directory: Directory to save to (uses output_dir if not specified)

        Returns:
            List of saved file paths
        """
        directory = Path(directory) if directory else self.output_dir
        if not directory:
            raise ValueError("No directory specified")

        directory.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, part in enumerate(parts, 1):
            if len(parts) == 1:
                # Single file, no part suffix
                file_path = directory / f"{base_key}.md"
            else:
                file_path = directory / f"{base_key}.part{i}.md"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(part)

            saved_paths.append(file_path)
            logger.debug(f"Saved {file_path.name}")

        return saved_paths

    def load_parts(
        self,
        base_key: str,
        directory: Path = None
    ) -> List[Tuple[str, str]]:
        """
        Load all parts for a base key.

        Args:
            base_key: Base file key
            directory: Directory to load from (uses output_dir if not specified)

        Returns:
            List of (part_key, content) tuples
        """
        directory = Path(directory) if directory else self.output_dir
        if not directory:
            raise ValueError("No directory specified")

        part_paths = self.get_part_file_paths(base_key, directory)

        parts = []
        for path in part_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Determine part key from filename
                stem = path.stem
                if '.part' in stem:
                    identity = ChapterIdentity.parse(stem)
                    part_key = identity.full_name if identity else stem
                else:
                    part_key = stem

                parts.append((part_key, content))

        return parts

    def delete_old_parts(
        self,
        base_key: str,
        directory: Path = None,
        keep_version: int = None
    ):
        """
        Delete old part files that don't match current split.

        Args:
            base_key: Base file key
            directory: Directory to clean
            keep_version: Only delete if older than this version
        """
        directory = Path(directory) if directory else self.output_dir
        if not directory:
            return

        # Find all existing part files
        existing = list(directory.glob(f"{base_key}.part*.md"))

        # Get current expected parts
        current_parts = self.get_current_parts(base_key)
        expected_files = {
            directory / f"{key.replace('.part', '.part')}.md"
            for key in current_parts
        }

        # Delete files that shouldn't exist
        for path in existing:
            if path not in expected_files:
                logger.info(f"Deleting old part file: {path.name}")
                path.unlink()


if __name__ == "__main__":
    # Quick demonstration
    import tempfile
    from unittest.mock import MagicMock

    print("SplitManager demonstration\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        progress_path = tmpdir / "progress.json"
        output_dir = tmpdir / "output"

        # Create mock LLM client
        mock_llm = MagicMock()

        # Create tracker and manager
        tracker = ProcessingTracker(progress_path, "TestProcessor")
        manager = SplitManager(
            tracker=tracker,
            llm_client=mock_llm,
            output_dir=output_dir,
            default_max_tokens=1000
        )

        # Create test content
        test_content = "This is test content. " * 100  # Small for demo

        # Get or create split
        result = manager.get_or_create_split(
            base_key="chapter_1",
            content=test_content,
            max_tokens=500
        )

        print(f"Split result:")
        print(f"  Parts: {len(result.parts)}")
        print(f"  Version: {result.version}")
        print(f"  Method: {result.method}")
        print(f"  Reason: {result.reason}")

        for info in result.part_infos:
            print(f"  Part {info.index}: {info.token_count} tokens")

        # Test current parts
        current = manager.get_current_parts("chapter_1")
        print(f"\nCurrent parts: {current}")

        # Test save and load
        saved = manager.save_parts("chapter_1", result.parts)
        print(f"\nSaved files: {[p.name for p in saved]}")

        loaded = manager.load_parts("chapter_1")
        print(f"Loaded parts: {[key for key, _ in loaded]}")

        print("\nSplitManager demonstration complete!")
