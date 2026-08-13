"""Rich page-level OCR artifacts.

Markdown remains the compatibility view consumed by the existing refine
pipeline.  Backends which expose richer output can additionally persist HTML,
the unmodified model response, ordered layout blocks, and extracted assets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OCRPageResult:
    """All representations produced while OCRing one page."""

    markdown: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    image_counter: int = 0
    html: Optional[str] = None
    raw_html: Optional[str] = None
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    page_box: Optional[List[int]] = None
    model_input_size: Optional[List[int]] = None
    token_count: Optional[int] = None
    backend: Optional[str] = None
    model: Optional[str] = None
    model_revision: Optional[str] = None
    assets: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_legacy_tuple(
        cls,
        value: tuple[str, List[Dict[str, Any]], int],
        *,
        backend: str,
    ) -> "OCRPageResult":
        markdown, images, image_counter = value
        return cls(
            markdown=markdown,
            images=images,
            image_counter=image_counter,
            backend=backend,
        )
