"""
Skip validators - decide whether to skip validation for certain content.
"""

from typing import Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .._protocol import ProcessContext


class ChapterTypeSkipper:
    """Skip validation for certain chapter types."""

    # Default chapter types to skip validation for
    DEFAULT_SKIP_TYPES: Set[str] = {
        "front_matter",
        "back_matter",
        "notes",
        "appendix",
        "toc",
        "index",
        "bibliography",
        "colophon",
    }

    def __init__(self, skip_types: Optional[Set[str]] = None):
        """
        Initialize with optional custom skip types.

        Args:
            skip_types: Set of chapter types to skip, or None for defaults
        """
        self._skip_types = skip_types or self.DEFAULT_SKIP_TYPES

    @property
    def name(self) -> str:
        return "ChapterTypeSkipper"

    def should_skip(
        self,
        key: str,
        chapter_type: str,
        context: Optional["ProcessContext"]
    ) -> bool:
        """
        Check if validation should be skipped.

        Args:
            key: Unit identifier
            chapter_type: Type of chapter
            context: Processing context (may be None)

        Returns:
            True if validation should be skipped
        """
        if not chapter_type:
            return False
        return chapter_type.lower() in self._skip_types


class ShortContentSkipper:
    """Skip validation for very short content."""

    def __init__(self, min_chars: int = 50):
        """
        Initialize with minimum character threshold.

        Args:
            min_chars: Minimum characters to require validation
        """
        self._min_chars = min_chars

    @property
    def name(self) -> str:
        return "ShortContentSkipper"

    def should_skip(
        self,
        key: str,
        chapter_type: str,
        context: Optional["ProcessContext"]
    ) -> bool:
        """
        Check if content is too short to validate.

        Note: This requires the original content which isn't in the signature.
        Should be used with context.extra['original_length'] if needed.
        """
        if context and context.extra:
            original_length = context.extra.get('original_length', 0)
            return original_length < self._min_chars
        return False


class KeyPatternSkipper:
    """Skip validation for keys matching certain patterns."""

    def __init__(self, skip_patterns: Optional[Set[str]] = None):
        """
        Initialize with patterns to skip.

        Args:
            skip_patterns: Set of substrings that trigger skip if found in key
        """
        self._skip_patterns = skip_patterns or set()

    @property
    def name(self) -> str:
        return "KeyPatternSkipper"

    def should_skip(
        self,
        key: str,
        chapter_type: str,
        context: Optional["ProcessContext"]
    ) -> bool:
        """Check if key matches any skip pattern."""
        if not key:
            return False
        key_lower = key.lower()
        return any(pattern.lower() in key_lower for pattern in self._skip_patterns)
