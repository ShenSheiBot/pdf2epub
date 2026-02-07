"""
PDF batch processing utilities for large documents.

Handles page splitting, batch generation, and result merging
for PDFs exceeding API page limits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Optional

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


def convert_toc_page_to_original(
    page_in_subset: int,
    ctx: PdfBatchContext
) -> int:
    """
    Convert page number from TOC detection subset PDF to original PDF page number.

    The subset PDF contains first N + last N pages, so:
    - Pages 1 to N in subset → pages 1 to N in original
    - Pages N+1 to 2N in subset → pages (total - N + 1) to total in original

    Args:
        page_in_subset: Page number in the subset PDF (1-indexed)
        ctx: Batch context with total_pages and toc_sample_pages

    Returns:
        Original PDF page number (1-indexed)
    """
    if not ctx.needs_batching:
        return page_in_subset

    n = ctx.toc_sample_pages

    if page_in_subset <= n:
        # First N pages: direct mapping
        return page_in_subset
    else:
        # Last N pages: offset by the gap
        # subset page N+1 → original page (total - N + 1)
        # subset page N+k → original page (total - N + k)
        offset_in_last_section = page_in_subset - n  # 1-indexed within last section
        return ctx.total_pages - n + offset_in_last_section


def create_content_batches(
    ctx: PdfBatchContext,
    exclude_pages: Optional[Set[int]] = None
) -> List[List[int]]:
    """
    Split content pages into batches with overlap.

    Args:
        ctx: Batch context
        exclude_pages: Pages to exclude (e.g., TOC pages)

    Returns:
        List of page number lists, each representing a batch
    """
    exclude = exclude_pages or set()
    content_pages = [p for p in range(1, ctx.total_pages + 1) if p not in exclude]

    if len(content_pages) <= ctx.batch_size:
        return [content_pages]

    batches = []
    start = 0
    while start < len(content_pages):
        end = min(start + ctx.batch_size, len(content_pages))
        batches.append(content_pages[start:end])
        # Move forward, keeping overlap
        start = end - ctx.overlap if end < len(content_pages) else end

    return batches


def deduplicate_chapters(chapters: List[Dict]) -> List[Dict]:
    """
    Remove duplicate chapters from overlapping batch regions.
    Keeps the one with earlier start_page (more authoritative).
    """
    seen: Dict[str, Dict] = {}
    result = []

    for chapter in chapters:
        title = chapter.get('title', '')
        key = title.strip().lower()

        if key in seen:
            existing = seen[key]
            if chapter.get('start_page', float('inf')) < existing.get('start_page', float('inf')):
                result.remove(existing)
                result.append(chapter)
                seen[key] = chapter
        else:
            seen[key] = chapter
            result.append(chapter)

    result.sort(key=lambda x: x.get('start_page', 0))
    return result


def merge_batch_chapters(
    toc_structure: Dict,
    batch_results: List[Dict],
    total_pages: int
) -> List[Dict]:
    """
    Merge chapter results from multiple batches.

    Strategy:
    - Use TOC structure as template
    - Fill in start_page from batch results
    - Calculate end_page based on next chapter's start
    """
    # Collect all found chapters from batches
    found: Dict[str, Dict] = {}  # title -> chapter info

    for batch_result in batch_results:
        for ch in batch_result.get('chapters_found', []):
            title = ch.get('title', '').strip()
            if title and title not in found:
                found[title] = ch
            elif title and ch.get('start_page', float('inf')) < found[title].get('start_page', float('inf')):
                found[title] = ch  # Keep earlier occurrence

    def fill_pages(chapters: List[Dict], parent_end: int) -> List[Dict]:
        """Recursively fill page numbers from found results."""
        result = []
        for i, ch in enumerate(chapters):
            title = ch.get('title', '').strip()
            match = found.get(title)

            new_ch = dict(ch)
            if match:
                new_ch['start_page'] = match.get('start_page')

            # Calculate end_page from next sibling or parent_end
            if i + 1 < len(chapters):
                next_title = chapters[i + 1].get('title', '').strip()
                next_match = found.get(next_title)
                if next_match:
                    new_ch['end_page'] = next_match['start_page'] - 1
                else:
                    new_ch['end_page'] = parent_end
            else:
                new_ch['end_page'] = parent_end

            # Recurse into children first (we need their start_page info)
            if ch.get('children'):
                new_ch['children'] = fill_pages(
                    ch['children'],
                    new_ch.get('end_page', parent_end)
                )

                # If parent has no start_page, infer from first child
                if 'start_page' not in new_ch and new_ch['children']:
                    for child in new_ch['children']:
                        if child.get('start_page'):
                            new_ch['start_page'] = child['start_page']
                            break

            result.append(new_ch)
        return result

    return fill_pages(toc_structure.get('chapters', []), total_pages)
