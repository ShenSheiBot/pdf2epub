"""
Pre-processors - decide whether content should be processed.

Pre-processors run before LLM calls and can skip processing entirely.
"""

from typing import Optional, TYPE_CHECKING
from ._protocol import PreProcessResult

if TYPE_CHECKING:
    from .._protocol import ProcessContext
    from ..book_structure import BookStructure


class ImageOnlyFilter:
    """Skip image-only pages - return original content."""

    def __init__(self, book_structure: Optional["BookStructure"] = None):
        self._book_structure = book_structure

    @property
    def name(self) -> str:
        return "ImageOnlyFilter"

    def check(
        self,
        key: str,
        content: str,
        context: "ProcessContext"
    ) -> PreProcessResult:
        """Check if content is image-only."""
        if self._book_structure and self._book_structure.is_image_only_content(content):
            return PreProcessResult(
                should_process=False,
                skip_reason="Image-only content",
                fallback_result=content  # Return original
            )
        return PreProcessResult(should_process=True)


class EmptyContentFilter:
    """Skip empty content."""

    @property
    def name(self) -> str:
        return "EmptyContentFilter"

    def check(
        self,
        key: str,
        content: str,
        context: "ProcessContext"
    ) -> PreProcessResult:
        """Check if content is empty."""
        if not content or not content.strip():
            return PreProcessResult(
                should_process=False,
                skip_reason="Empty content",
                fallback_result=""
            )
        return PreProcessResult(should_process=True)


class MinLengthFilter:
    """Skip content that's too short to be worth processing."""

    def __init__(self, min_chars: int = 10):
        self._min_chars = min_chars

    @property
    def name(self) -> str:
        return "MinLengthFilter"

    def check(
        self,
        key: str,
        content: str,
        context: "ProcessContext"
    ) -> PreProcessResult:
        """Check if content meets minimum length."""
        stripped = content.strip() if content else ""
        if len(stripped) < self._min_chars:
            return PreProcessResult(
                should_process=False,
                skip_reason=f"Content too short ({len(stripped)} < {self._min_chars} chars)",
                fallback_result=content  # Return original
            )
        return PreProcessResult(should_process=True)
