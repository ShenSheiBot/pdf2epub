"""
Adaptive PDF→LLM call orchestration.

Provides:
- PdfPageLimitLearner: Session-level learned page limit tracking
- run_adaptive_batches: Unified batch processing with 503 recovery
- is_503_error: Centralized 503 error detection
- AdaptivePdfCall: Base class for PDF→LLM calls with auto-batching and merge
- TocDetectionCall: Detect TOC location in PDF
- DirectAnalysisCall: Analyze PDF structure directly
"""

import json
from pathlib import Path
from typing import Any, Callable, List, TypeVar

from google.genai.types import Part
from loguru import logger

from ..utils.common import parse_llm_json
from ..utils.llm_client import BoundLLMClient

T = TypeVar('T')


def is_503_error(error: Exception) -> bool:
    """Check if an exception is a 503 UNAVAILABLE error."""
    error_str = str(error).lower()
    return '503' in error_str or 'unavailable' in error_str


class PdfPageLimitLearner:
    """
    Tracks learned page limits for PDF→LLM API calls.

    When a 503 error occurs, the limit is halved. This learned limit
    carries across all PDF→LLM calls in the same session, so subsequent
    calls start with the reduced limit instead of retrying at the original.
    """

    def __init__(self, initial_limit: int = 900, min_limit: int = 50):
        self._limit = initial_limit
        self._min_limit = min_limit
        self._had_503 = False

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def min_limit(self) -> int:
        return self._min_limit

    @property
    def had_503(self) -> bool:
        return self._had_503

    def report_503(self, attempted_pages: int) -> int:
        """
        Report a 503 error, reducing the limit.

        Args:
            attempted_pages: Number of pages in the failed request

        Returns:
            New limit

        Raises:
            RuntimeError: If limit would go below minimum
        """
        self._had_503 = True
        new_limit = attempted_pages // 2
        if new_limit < self._min_limit:
            raise RuntimeError(
                f"Adaptive page limit ({new_limit}) below minimum ({self._min_limit}). "
                f"API repeatedly rejects this PDF — likely a structural issue "
                f"(complex fonts, embedded objects, etc.), not a size issue. "
                f"Try pre-processing the PDF or using a different OCR backend."
            )
        self._limit = min(self._limit, new_limit)
        logger.warning(
            f"503 error at {attempted_pages} pages → learned limit: {self._limit}"
        )
        return self._limit

    def report_success(self, pages: int):
        """Report successful call. Currently no-op (conservative strategy)."""
        pass


def split_pages_into_batches(
    pages: List[int],
    batch_size: int,
    overlap: int = 0,
) -> List[List[int]]:
    """
    Split a page list into batches with optional overlap.

    Args:
        pages: List of page numbers (order preserved)
        batch_size: Maximum pages per batch
        overlap: Pages of overlap between consecutive batches

    Returns:
        List of page number lists
    """
    if not pages:
        return []
    if len(pages) <= batch_size:
        return [pages]

    batches = []
    start = 0
    while start < len(pages):
        end = min(start + batch_size, len(pages))
        batches.append(pages[start:end])
        start = end - overlap if end < len(pages) else end

    return batches


def run_adaptive_batches(
    pages: List[int],
    process_batch: Callable[[List[int], int, int], T],
    learner: PdfPageLimitLearner,
    is_503_fn: Callable[[Exception], bool],
    operation_name: str,
    overlap: int = 0,
) -> List[T]:
    """
    Process pages in batches with adaptive 503 recovery.

    On 503, the learner reduces the page limit and the failed batch
    (plus all remaining batches) are re-split with the new limit.

    Args:
        pages: All pages to process (1-indexed)
        process_batch: Callable(batch_pages, batch_idx, total_batches) -> result
        learner: Page limit learner (shared across calls in session)
        is_503_fn: Predicate to identify 503 errors
        operation_name: For logging
        overlap: Pages of overlap between consecutive batches

    Returns:
        List of results from each successful batch
    """
    batches = split_pages_into_batches(pages, learner.limit, overlap)
    results = []

    logger.info(
        f"[{operation_name}] Processing {len(pages)} pages in {len(batches)} batch(es) "
        f"(limit: {learner.limit}, overlap: {overlap})"
    )

    batch_idx = 0
    while batch_idx < len(batches):
        batch = batches[batch_idx]
        batch_start, batch_end = min(batch), max(batch)

        logger.info(
            f"[{operation_name}] Batch {batch_idx + 1}/{len(batches)}: "
            f"pages {batch_start}-{batch_end} ({len(batch)} pages)"
        )

        try:
            result = process_batch(batch, batch_idx, len(batches))
            learner.report_success(len(batch))
            results.append(result)
            batch_idx += 1
        except Exception as e:
            if is_503_fn(e):
                learner.report_503(len(batch))

                # Collect all remaining pages (current failed + future batches)
                remaining_pages = []
                seen = set()
                for b in batches[batch_idx:]:
                    for p in b:
                        if p not in seen:
                            seen.add(p)
                            remaining_pages.append(p)

                # Re-split with new limit
                new_batches = split_pages_into_batches(
                    remaining_pages, learner.limit, overlap
                )
                batches = batches[:batch_idx] + new_batches

                logger.info(
                    f"[{operation_name}] Re-split into {len(new_batches)} batch(es) "
                    f"(new limit: {learner.limit})"
                )
                # Don't increment batch_idx — retry with smaller batch
            elif _is_cloudflare_proxy_error(e):
                raise RuntimeError(
                    "Cloudflare proxy returned HTTP 524 (origin timeout). "
                    "Cloudflare proxies cannot handle large PDF requests. "
                    "Please use Vertex AI or a direct Gemini API endpoint instead."
                ) from e
            else:
                raise

    return results


def _is_cloudflare_proxy_error(e: Exception) -> bool:
    """Check if an exception is a Cloudflare 524 proxy timeout."""
    err = str(e).lower()
    return '524' in err and 'timeout' in err


# ---------------------------------------------------------------------------
# Structural validation for chapter lists
# ---------------------------------------------------------------------------

def validate_chapter_structure(chapters: List[dict], path: str = "") -> List[str]:
    """
    Validate a chapter list for structural issues.

    Checks:
    - Missing start_page or end_page
    - end_page < start_page
    - Overlapping siblings (chapter N end_page >= chapter N+1 start_page)

    Recurses into children.

    Returns:
        List of issue descriptions (empty = valid)
    """
    issues = []

    for i, chapter in enumerate(chapters):
        title = chapter.get('title', 'unknown')[:40]
        chapter_path = f"{path}/{title}" if path else title

        start = chapter.get('start_page')
        end = chapter.get('end_page')

        if start is None:
            issues.append(f"Missing start_page: {chapter_path}")
        if end is None:
            issues.append(f"Missing end_page: {chapter_path}")

        if start is not None and end is not None:
            if end < start:
                issues.append(
                    f"Invalid range (end < start): {chapter_path} "
                    f"(p{start}-p{end})"
                )

            if i + 1 < len(chapters):
                next_start = chapters[i + 1].get('start_page')
                if next_start is not None and end > next_start:
                    next_title = chapters[i + 1].get('title', 'unknown')[:40]
                    issues.append(
                        f"Overlap: '{title}' ends at p{end} "
                        f"but '{next_title}' starts at p{next_start}"
                    )

        children = chapter.get('children', [])
        if children:
            issues.extend(validate_chapter_structure(children, chapter_path))

    return issues


# ---------------------------------------------------------------------------
# Base class for adaptive PDF→LLM calls
# ---------------------------------------------------------------------------

class AdaptivePdfCall:
    """
    Base class for adaptive PDF→LLM calls.

    Centralizes the entire flow:
    1. Split pages into batches (using learned page limit)
    2. For each batch: prepare PDF → build prompt → call LLM
    3. On 503: halve batch size and retry (via PdfPageLimitLearner)
    4. Merge multi-batch results (LLM merge by default, with retry)

    Subclasses override:
    - build_prompt(): prompt for each batch
    - build_merge_prompt(): prompt for LLM-based merging of multi-batch results
    - validate_batch_result(): structural check after each batch (triggers retry with PDF)
    - build_repair_prompt(): prompt for fix retry (includes errors + previous response)
    - validate_merge(): validation after merge (triggers retry if False)
    - merge_results(): override entirely for non-LLM merge (e.g. rule-based)
    """

    operation_name: str = "PDF call"
    overlap: int = 0
    merge_max_retries: int = 2
    batch_validation_retries: int = 2

    def __init__(
        self,
        client: BoundLLMClient,
        model: str,
        prepare_pdf: Callable,
        learner: PdfPageLimitLearner,
    ):
        self.client = client
        self.model = model
        self._prepare_pdf = prepare_pdf
        self._learner = learner

    def build_prompt(self, batch_pages: List[int], batch_idx: int, total_batches: int) -> str:
        """Build the prompt for a single batch. Must be overridden."""
        raise NotImplementedError

    def build_merge_prompt(self, results: List) -> str:
        """Build prompt for LLM-based merge of multi-batch results."""
        raise NotImplementedError(
            f"{self.__class__.__name__} got {len(results)} batches "
            f"but doesn't implement build_merge_prompt"
        )

    def validate_batch_result(self, result: Any, batch_idx: int, total_batches: int) -> List[str]:
        """
        Hook: validate a single batch result after parsing.

        Returns list of issues (empty = OK). When non-empty, the batch will
        be retried with the PDF + error feedback so the LLM can fix issues
        while it still has visual context. Edge issues (first/last chapter
        at batch boundaries) are automatically tolerated.

        Override in subclasses for structural checks.
        """
        return []

    def build_repair_prompt(self, original_prompt: str, result: Any, issues: List[str]) -> str:
        """
        Build a prompt asking the LLM to fix validation issues in its previous response.

        The LLM still has access to the PDF, so it can reference the actual content
        to verify and correct page numbers and structure.
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        if len(result_json) > 8000:
            result_json = result_json[:8000] + "\n... (truncated)"

        return f"""{original_prompt}

--- YOUR PREVIOUS RESPONSE HAD STRUCTURAL ERRORS ---

Previous response:
{result_json}

Structural issues found:
{issues_text}

Please fix these issues while keeping the rest of the response intact.
Return the COMPLETE corrected JSON response (not just the fixed parts).
Look at the PDF pages carefully to verify page numbers are correct."""

    def _filter_edge_issues(
        self, issues: List[str], chapters: List[dict],
        batch_idx: int, total_batches: int,
    ) -> List[str]:
        """
        Remove validation issues that involve edge chapters at batch boundaries.

        At batch boundaries, the first/last chapter is expected to be incomplete
        because the LLM only sees a portion of the book:
        - Non-first batch: first chapter may have wrong start_page
        - Non-final batch: last chapter may have wrong end_page or overlap
        """
        if total_batches <= 1 or not chapters:
            return issues

        edge_titles = set()
        if batch_idx > 0:
            first_title = chapters[0].get('title', '')[:40]
            if first_title:
                edge_titles.add(first_title)
        if batch_idx < total_batches - 1:
            last_title = chapters[-1].get('title', '')[:40]
            if last_title:
                edge_titles.add(last_title)

        if not edge_titles:
            return issues

        filtered = []
        for issue in issues:
            if any(title in issue for title in edge_titles):
                continue
            filtered.append(issue)
        return filtered

    def validate_merge(self, merged: Any, original_results: List) -> bool:
        """Validate merged result. Return True if acceptable."""
        return True

    def parse_result(self, response_text: str) -> Any:
        """Parse LLM response. Default: parse as JSON."""
        return parse_llm_json(response_text, operation_name=self.operation_name)

    def merge_results(self, results: List) -> Any:
        """
        Merge results from multiple batches.

        Default: LLM-based merge using build_merge_prompt(), with retry.
        Override entirely for rule-based merge (e.g. TocDetectionCall).
        """
        if len(results) == 1:
            return results[0]

        merged = None
        for attempt in range(1 + self.merge_max_retries):
            prompt = self.build_merge_prompt(results)
            config = self.client.get_default_config(temperature=0.1)
            config.response_mime_type = "application/json"

            op_name = f"{self.operation_name} merge"
            if attempt > 0:
                op_name += f" (retry {attempt})"

            response_text = self.client.generate_content_stream(
                model=self.model,
                contents=[prompt],
                config=config,
                operation_name=op_name,
            )
            merged = self.parse_result(response_text)

            if self.validate_merge(merged, results):
                logger.info(
                    f"[{self.operation_name}] Merged {len(results)} batches successfully"
                )
                return merged

            logger.warning(
                f"[{self.operation_name}] Merge validation failed "
                f"(attempt {attempt + 1}/{1 + self.merge_max_retries})"
            )

        logger.warning(
            f"[{self.operation_name}] Merge validation still failing after retries, "
            f"using best result"
        )
        return merged

    def run(self, pdf_path: Path, pages: List[int]) -> Any:
        """
        Execute the adaptive PDF→LLM call.

        Args:
            pdf_path: Path to PDF file
            pages: List of 1-indexed page numbers to process

        Returns:
            Parsed result (single batch) or merged result (multi-batch)
        """
        def process_batch(batch_pages, batch_idx, total_batches):
            pdf_data = self._prepare_pdf(pdf_path, include_pages=batch_pages)
            if pdf_data is None:
                batch_start, batch_end = min(batch_pages), max(batch_pages)
                raise RuntimeError(
                    f"Failed to prepare PDF batch (pages {batch_start}-{batch_end})"
                )

            original_prompt = self.build_prompt(batch_pages, batch_idx, total_batches)
            prompt = original_prompt
            result = None

            for attempt in range(1 + self.batch_validation_retries):
                parts = [prompt, Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
                config = self.client.get_default_config(temperature=0.1)
                config.response_mime_type = "application/json"

                op_name = f"{self.operation_name} batch {batch_idx+1}/{total_batches}"
                if attempt > 0:
                    op_name += f" (fix {attempt})"

                response_text = self.client.generate_content_stream(
                    model=self.model,
                    contents=parts,
                    config=config,
                    operation_name=op_name,
                )
                result = self.parse_result(response_text)

                # Run batch validation hook
                all_issues = self.validate_batch_result(result, batch_idx, total_batches)
                if not all_issues:
                    return result

                # Filter out edge issues (expected at batch boundaries)
                chapters = result.get('chapters', []) if isinstance(result, dict) else []
                actionable_issues = self._filter_edge_issues(
                    all_issues, chapters, batch_idx, total_batches
                )

                if not actionable_issues:
                    logger.info(
                        f"[{self.operation_name}] Batch {batch_idx+1}/{total_batches}: "
                        f"{len(all_issues)} edge issue(s) tolerated"
                    )
                    return result

                if attempt < self.batch_validation_retries:
                    logger.warning(
                        f"[{self.operation_name}] Batch {batch_idx+1}/{total_batches} "
                        f"has {len(actionable_issues)} structural issue(s), "
                        f"retrying with PDF (attempt {attempt+1}/{self.batch_validation_retries}):"
                    )
                    for issue in actionable_issues[:5]:
                        logger.warning(f"  - {issue}")
                    if len(actionable_issues) > 5:
                        logger.warning(f"  ... and {len(actionable_issues) - 5} more")
                    prompt = self.build_repair_prompt(original_prompt, result, actionable_issues)
                else:
                    logger.warning(
                        f"[{self.operation_name}] Batch {batch_idx+1}/{total_batches} "
                        f"still has {len(actionable_issues)} issue(s) after "
                        f"{self.batch_validation_retries} fix attempt(s)"
                    )
                    for issue in actionable_issues[:5]:
                        logger.warning(f"  - {issue}")

            return result

        results = run_adaptive_batches(
            pages, process_batch, self._learner, is_503_error,
            self.operation_name, overlap=self.overlap
        )

        return self.merge_results(results)


# ---------------------------------------------------------------------------
# Concrete call types
# ---------------------------------------------------------------------------

class TocDetectionCall(AdaptivePdfCall):
    """Detect TOC location in PDF."""

    operation_name = "TOC location detection"
    overlap = 0

    def build_prompt(self, batch_pages, batch_idx, total_batches):
        return """Analyze this PDF and find the Table of Contents (TOC) pages.

Look for pages that contain:
- A list of chapter/section titles with page numbers
- Typically titled "Table of Contents", "Contents", "目次", "Table des matières", "Sommaire", "Inhalt", etc.
- Usually appears near the beginning or end of the book

Return JSON:
{
    "has_toc": boolean,  // true if a TOC exists
    "toc_start": int,    // PDF page number where TOC starts (1-indexed)
    "toc_end": int       // PDF page number where TOC ends (1-indexed)
}

If no TOC exists, return: {"has_toc": false, "toc_start": null, "toc_end": null}

**IMPORTANT**: Use PDF page numbers from the "PDF Page: X" labels, not printed page numbers.
"""

    def parse_result(self, response_text):
        result = parse_llm_json(response_text, operation_name=self.operation_name)
        if not isinstance(result, dict):
            logger.warning(f"TOC detection returned {type(result)}, expected dict")
            return {'has_toc': False, 'toc_start': None, 'toc_end': None}
        return result

    def merge_results(self, results):
        """Rule-based: pick first result with has_toc=True."""
        if len(results) == 1:
            return results[0]

        for r in results:
            if isinstance(r, dict) and r.get('has_toc'):
                logger.info(f"TOC detected: pages {r['toc_start']}-{r['toc_end']}")
                return r

        if results and isinstance(results[0], dict):
            logger.info("No TOC detected in PDF")
            return results[0]

        return None


class DirectAnalysisCall(AdaptivePdfCall):
    """Analyze PDF structure directly — extract chapter hierarchy."""

    operation_name = "Direct analysis"
    overlap = 50
    merge_max_retries = 2

    def __init__(self, client, model, prepare_pdf, learner, book_title: str, toc_reference: str = None):
        super().__init__(client, model, prepare_pdf, learner)
        self.book_title = book_title
        self.toc_reference = toc_reference

    def build_prompt(self, batch_pages, batch_idx, total_batches):
        batch_start, batch_end = min(batch_pages), max(batch_pages)
        batch_num = batch_idx + 1
        is_first = (batch_idx == 0)
        is_last = (batch_idx == total_batches - 1)

        # Build metadata fields for JSON schema (first/last batch only)
        # NOTE: plain string, not f-string — use single braces for JSON
        metadata_fields = ""
        if is_first:
            metadata_fields = """    "author": string,
    "language": string,  // e.g., "english", "japanese", "chinese"
    "is_vertical_text": boolean,
    "has_footnotes": boolean,  // true if content has footnotes/citations
    "cover_page": {"page_number": int},
    "table_of_contents": {   // omit if no TOC exists
        "start_page": int,
        "end_page": int
    },
"""
        if is_last:
            metadata_fields += '    "back_cover": {"page_number": int},\n'

        # Build optional TOC reference block
        toc_block = ""
        if self.toc_reference:
            toc_block = f"""
**REFERENCE — Book's Own Table of Contents** (page numbers removed):
{self.toc_reference}

Use this as a guide to identify ALL sections. Every section listed above MUST appear in your output.
Do NOT use the reference to determine page numbers — determine page numbers ONLY from "PDF Page: X" labels in the PDF.
"""

        return f"""
Analyze this book PDF section and extract chapter structure.

**Book Title**: {self.book_title}
**BATCH INFO**: Batch {batch_num}/{total_batches}, pages {batch_start}-{batch_end}
{toc_block}
**CRITICAL**: Extract the COMPLETE hierarchical structure.
- Extract ALL levels: Part, Chapter, Section, Subsection, etc.
- DO NOT create artificial subdivisions beyond what actually exists
- Use PDF page numbers from "PDF Page: X" labels (not printed page numbers)

Additionally identify special chapter types:
- If a chapter consists ONLY of footnotes/endnotes for other chapters, add "type": "notes"
- If any chapter's notes are at the end of itself, then there should be NO notes chapter
- A book contains at most one notes chapter
- Abbreviations, Bibliography, Index, or Summary Table are NOT considered as notes
- Only literal "Notes" or "Endnotes" chapters with [1], [2], [3]... definitions are considered as notes

Return JSON:
{{
{metadata_fields}    "chapters": [
        {{
            "title": string,
            "start_page": int,  // PDF page number
            "end_page": int,    // Use {batch_end} if continues beyond
            "level": int,
            "type": string,     // Optional: "notes" for footnote/endnote chapters
            "children": [...]   // Recursive - can have unlimited depth
        }}
    ]
}}

**IMPORTANT**:
- Use PDF page numbers from "PDF Page: X" labels, NOT printed page numbers
- Preserve the original language for all titles and author names
"""

    def build_merge_prompt(self, results):
        batch_chapters = [
            r if isinstance(r, list) else r.get('chapters', [])
            for r in results
        ]

        batch_summaries = []
        for i, chapters in enumerate(batch_chapters):
            batch_summaries.append(
                f"=== Batch {i+1}/{len(batch_chapters)} ===\n"
                f"{json.dumps(chapters, ensure_ascii=False, indent=2)}"
            )

        return f"""You are merging chapter structure results from {len(batch_chapters)} overlapping batches of the same book.

**Book Title**: {self.book_title}

Each batch analyzed a different page range with some overlap. The overlap region may have chapters recognized differently by each batch. Your job is to produce ONE unified, correct chapter list.

Rules:
1. Each chapter should appear exactly ONCE
2. If two batches have the same chapter, use the one with the more accurate page range
3. A chapter that appears as a top-level entry in one batch but as a child in another — trust the batch that saw MORE context around it
4. Preserve the hierarchical structure (children nested under parents)
5. Chapters must be ordered by start_page
6. Do NOT invent chapters that don't appear in any batch

Batch results:
{chr(10).join(batch_summaries)}

Return a single JSON object:
{{
    "chapters": [
        {{
            "title": string,
            "start_page": int,
            "end_page": int,
            "level": int,
            "type": string,     // Preserve "notes" if present in source batches
            "children": [...]
        }}
    ]
}}"""

    def validate_batch_result(self, result, batch_idx, total_batches):
        if isinstance(result, list):
            chapters = result
        else:
            chapters = result.get('chapters', [])
        return validate_chapter_structure(chapters)

    def validate_merge(self, merged, original_results):
        # LLM may return a bare list instead of {"chapters": [...]}
        if isinstance(merged, list):
            merged_chapters = merged
        else:
            merged_chapters = merged.get('chapters', [])

        # Check 1: chapter count sanity
        all_titles = set()
        for r in original_results:
            chapters = r if isinstance(r, list) else r.get('chapters', [])
            for ch in chapters:
                all_titles.add(ch.get('title', '').strip().lower())
        expected_min = max(1, len(all_titles) // 2)

        if len(merged_chapters) < expected_min:
            logger.warning(
                f"Merge returned {len(merged_chapters)} chapters "
                f"(expected >={expected_min})"
            )
            return False

        # Check 2: structural validity
        issues = validate_chapter_structure(merged_chapters)
        if issues:
            logger.warning(
                f"Merge has {len(issues)} structural issues:"
            )
            for issue in issues[:5]:
                logger.warning(f"  - {issue}")
            return False

        return True

    def merge_results(self, results):
        if len(results) == 1:
            return results[0]

        # Extract metadata from first/last batch (rule-based, no LLM needed)
        metadata = {}
        for i, result in enumerate(results):
            if i == 0:
                metadata = {
                    'author': result.get('author'),
                    'language': result.get('language'),
                    'is_vertical_text': result.get('is_vertical_text'),
                    'has_footnotes': result.get('has_footnotes'),
                    'cover_page': result.get('cover_page'),
                    'table_of_contents': result.get('table_of_contents'),
                }
            if i == len(results) - 1:
                metadata['back_cover'] = result.get('back_cover')

        # LLM merge for chapters (via base class)
        merged = super().merge_results(results)

        # LLM may return a bare list instead of {"chapters": [...]}
        if isinstance(merged, list):
            chapters = merged
        else:
            chapters = merged.get('chapters', [])

        return {**metadata, 'chapters': chapters}
