"""
Promoter - Pipeline's interface for promoting results to validated/.

Design principles:
1. Pipeline can only PROMOTE, not save_raw
2. Only Executor (via Saver) can write to raw/
3. .sub units are NEVER promoted

This separation ensures:
- Executor owns raw/ (writes immediately on completion)
- Pipeline owns validated/ (promotes after batch validation)
- Clear ownership prevents race conditions and data corruption
"""

from typing import List, Set, TYPE_CHECKING
from loguru import logger

from .types import is_sub_key

if TYPE_CHECKING:
    from .persistence import ResultPersistence


class Promoter:
    """
    Pipeline's interface for promoting results to validated/.

    Can only promote existing raw files to validated.
    Cannot write new files (that's Executor/Saver's job).

    Design:
    - promote(): Move single file from raw/ to validated/
    - promote_batch(): Move multiple files
    - .sub units: Silently skipped (never promoted)
    """

    def __init__(self, persistence: "ResultPersistence"):
        """
        Initialize promoter with persistence.

        Args:
            persistence: For file operations (promote_to_validated)
        """
        self._persistence = persistence

    def promote(self, unit_id: str) -> bool:
        """
        Promote a single unit from raw/ to validated/.

        Skips .sub units (they should never be promoted).
        Returns False if raw file doesn't exist.

        Args:
            unit_id: Unit identifier

        Returns:
            True if promoted successfully, False otherwise
        """
        # .sub units are never promoted
        if is_sub_key(unit_id):
            logger.debug(f"{unit_id}: Skipping .sub unit promotion")
            return False

        try:
            self._persistence.promote_to_validated(unit_id)
            return True
        except FileNotFoundError:
            logger.warning(f"{unit_id}: Cannot promote, raw file not found")
            return False
        except Exception as e:
            logger.error(f"{unit_id}: Promotion failed: {e}")
            return False

    def promote_batch(self, unit_ids: List[str]) -> int:
        """
        Promote multiple units to validated/.

        Filters out .sub units automatically.

        Args:
            unit_ids: List of unit identifiers

        Returns:
            Number of successfully promoted units
        """
        # Filter out .sub units first
        real_units = [uid for uid in unit_ids if not is_sub_key(uid)]
        filtered_count = len(unit_ids) - len(real_units)

        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} .sub units from promotion")

        count = 0
        for unit_id in real_units:
            if self.promote(unit_id):
                count += 1

        if count > 0:
            logger.info(f"Promoted {count}/{len(real_units)} units to validated/")

        return count

    def has_validated(self, unit_id: str) -> bool:
        """
        Check if a unit has been validated.

        Args:
            unit_id: Unit identifier

        Returns:
            True if validated file exists
        """
        return self._persistence.has_validated(unit_id)

    def get_validated_keys(self) -> Set[str]:
        """
        Get all validated unit keys.

        Returns:
            Set of unit IDs that have validated files
        """
        return set(self._persistence.list_validated_keys())

    def save_with_warning(self, unit_id: str, content: str, warning: str) -> bool:
        """
        Save content with warning header to validated/.

        Used for longest-fallback results that failed validation.
        The warning header allows audit/traceability.

        Args:
            unit_id: Unit identifier
            content: Content to save
            warning: Warning message to prepend

        Returns:
            True if saved successfully
        """
        if is_sub_key(unit_id):
            logger.debug(f"{unit_id}: Skipping .sub unit warning save")
            return False

        try:
            self._persistence.save_with_warning(unit_id, content, warning)
            return True
        except Exception as e:
            logger.error(f"{unit_id}: Failed to save with warning: {e}")
            return False


__all__ = ["Promoter"]
