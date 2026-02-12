"""
PDF batch processing utilities for large documents.

Handles page splitting and batch context for PDFs exceeding API page limits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz


@dataclass
class PdfBatchContext:
    """Context for PDF batch processing."""
    total_pages: int
    page_limit: int = 1000      # API page limit
    batch_size: int = 900       # Leave margin for safety
    overlap: int = 50           # Pages of overlap between batches
    toc_sample_pages: int = 200  # Pages to sample from start/end for TOC detection

    @property
    def needs_batching(self) -> bool:
        return self.total_pages > self.page_limit

    @classmethod
    def from_pdf(cls, pdf_path: Path, **kwargs) -> "PdfBatchContext":
        """Create context from PDF file."""
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
        return cls(total_pages=total, **kwargs)


def get_toc_detection_pages(ctx: PdfBatchContext) -> List[int]:
    """
    Get pages for TOC detection (first N + last N).
    TOC is typically at beginning or end, never in middle.

    Returns original PDF page numbers (1-indexed).
    """
    if not ctx.needs_batching:
        return list(range(1, ctx.total_pages + 1))

    n = ctx.toc_sample_pages
    pages = set(range(1, min(n + 1, ctx.total_pages + 1)))

    # Add last N pages if not overlapping
    if ctx.total_pages > 2 * n:
        pages.update(range(ctx.total_pages - n + 1, ctx.total_pages + 1))

    return sorted(pages)
