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

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

from loguru import logger

from ..utils.common import parse_llm_json
from ..utils.network_utils import _retry_context
from ..utils.llm_client import BoundLLMClient
from ..core.whole import run_agent_loop, AgentLoopExhausted
from ..core.whole.runner import run_agent_loop_sync
from ..core.whole.prompts.json_refine import JSON_REFINE_PROMPT
from .pdf_transport import (
    GeminiPdfTransport,
    PdfPayloadTooLargeError,
    PdfTransport,
)

T = TypeVar('T')


class MergeValidationError(RuntimeError):
    """Raised when all merge attempts violate a structural invariant.

    A merge is the only point where independently valid batch results become a
    single TOC.  Returning an invalid merge lets downstream boundary repair
    operate on an untrusted tree, so callers must stop and preserve the
    artifacts for an explicit retry or model decision.
    """


class BatchValidationError(RuntimeError):
    """A PDF batch remained structurally invalid after repair attempts."""


def is_503_error(error: Exception) -> bool:
    """Check if adaptive PDF batching can address this request failure."""
    if isinstance(error, PdfPayloadTooLargeError):
        return True
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

    def report_503(
        self,
        attempted_pages: int,
        reason: str = "503 error",
    ) -> int:
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
        new_limit = -(-attempted_pages // 2)  # ceiling division
        if new_limit < self._min_limit:
            raise RuntimeError(
                f"Adaptive page limit ({new_limit}) below minimum ({self._min_limit}). "
                f"API repeatedly rejects this PDF — likely a structural issue "
                f"(complex fonts, embedded objects, etc.), not a size issue. "
                f"Try pre-processing the PDF or using a different OCR backend."
            )
        self._limit = min(self._limit, new_limit)
        logger.warning(
            f"{reason} at {attempted_pages} pages → learned limit: {self._limit}"
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
    batch_size = max(1, batch_size)
    if len(pages) <= batch_size:
        return [pages]

    # Clamp overlap to ensure forward progress (overlap must be < batch_size)
    effective_overlap = min(overlap, batch_size - 1)
    batches = []
    start = 0
    while start < len(pages):
        end = min(start + batch_size, len(pages))
        batches.append(pages[start:end])
        start = end - effective_overlap if end < len(pages) else end

    return batches


def run_adaptive_batches(
    pages: List[int],
    process_batch: Callable[[List[int], int, int, bool], T],
    learner: PdfPageLimitLearner,
    is_503_fn: Callable[[Exception], bool],
    operation_name: str,
    overlap: int = 0,
    can_rasterize: bool = False,
    return_batches: bool = False,
) -> Union[List[T], Tuple[List[T], List[List[int]]]]:
    """
    Process pages in batches with adaptive 503 recovery.

    On 503:
    1. If rasterization available and not yet tried: rasterize and retry same batch
    2. If already rasterized or no rasterization: reduce page limit and re-split

    Args:
        pages: All pages to process (1-indexed)
        process_batch: Callable(batch_pages, batch_idx, total_batches, use_rasterized) -> result
        learner: Page limit learner (shared across calls in session)
        is_503_fn: Predicate to identify 503 errors
        operation_name: For logging
        overlap: Pages of overlap between consecutive batches
        can_rasterize: Whether rasterization fallback is available
        return_batches: Also return the final successful page sets. This is
            useful to downstream merging code: adaptive 503 recovery can
            change batch boundaries, so those boundaries must not be inferred
            from result order alone.

    Returns:
        List of results from each successful batch
    """
    batches = split_pages_into_batches(pages, learner.limit, overlap)
    results = []

    logger.info(
        f"[{operation_name}] Processing {len(pages)} pages in {len(batches)} batch(es) "
        f"(limit: {learner.limit}, overlap: {overlap}, rasterize={can_rasterize})"
    )

    batch_idx = 0
    while batch_idx < len(batches):
        batch = batches[batch_idx]
        batch_start, batch_end = min(batch), max(batch)
        tried_rasterize_this_batch = False  # Reset for each batch

        logger.info(
            f"[{operation_name}] Batch {batch_idx + 1}/{len(batches)}: "
            f"pages {batch_start}-{batch_end} ({len(batch)} pages)"
        )

        try:
            # Tell tenacity not to retry 503 — we handle it here
            _retry_context.skip_503 = True
            try:
                result = process_batch(batch, batch_idx, len(batches), False)
            finally:
                _retry_context.skip_503 = False
            learner.report_success(len(batch))
            results.append(result)
            batch_idx += 1
        except Exception as e:
            if is_503_fn(e):
                rejection_reason = (
                    "payload-too-large rejection"
                    if isinstance(e, PdfPayloadTooLargeError)
                    else "503 error"
                )
                # Strategy: try rasterization first (once per batch), then split
                if can_rasterize and not tried_rasterize_this_batch:
                    logger.warning(
                        f"[{operation_name}] {rejection_reason} on "
                        f"{len(batch)} pages, "
                        f"retrying batch {batch_idx+1} with JBIG2 rasterization..."
                    )
                    tried_rasterize_this_batch = True

                    # Retry same batch with rasterized PDF
                    try:
                        _retry_context.skip_503 = True
                        try:
                            result = process_batch(batch, batch_idx, len(batches), True)
                        finally:
                            _retry_context.skip_503 = False
                        learner.report_success(len(batch))
                        results.append(result)
                        batch_idx += 1
                        continue
                    except Exception as e2:
                        if not is_503_fn(e2):
                            raise
                        rejection_reason = (
                            "payload-too-large rejection"
                            if isinstance(e2, PdfPayloadTooLargeError)
                            else "503 error"
                        )
                        logger.warning(
                            f"[{operation_name}] {rejection_reason} even after "
                            "rasterization, "
                            f"falling back to batch splitting..."
                        )
                        # Fall through to split logic

                # Split: reduce limit and re-batch
                learner.report_503(
                    len(batch),
                    reason=rejection_reason,
                )

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

    if return_batches:
        return results, batches
    return results


def _is_cloudflare_proxy_error(e: Exception) -> bool:
    """Check if an exception is a Cloudflare 524 proxy timeout."""
    err = str(e).lower()
    return '524' in err and 'timeout' in err


# ---------------------------------------------------------------------------
# Structural validation for chapter lists
# ---------------------------------------------------------------------------

def validate_chapter_structure(
    chapters: List[dict],
    path: str = "",
    parent_range: Optional[Tuple[int, int]] = None,
) -> List[str]:
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
        if not isinstance(chapter, dict):
            issues.append(
                f"Chapter entry at {path or '<root>'}[{i}] is not an object"
            )
            continue
        raw_title = chapter.get('title')
        title = (
            raw_title[:40]
            if isinstance(raw_title, str)
            else repr(raw_title)[:40]
        )
        chapter_path = f"{path}/{title}" if path else title
        if not isinstance(raw_title, str) or not raw_title.strip():
            issues.append(f"Missing or invalid title: {chapter_path}")

        start = chapter.get('start_page')
        end = chapter.get('end_page')
        valid_start = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and start >= 1
        )
        valid_end = (
            isinstance(end, int)
            and not isinstance(end, bool)
            and end >= 1
        )

        if start is None:
            issues.append(f"Missing start_page: {chapter_path}")
        elif not valid_start:
            issues.append(
                f"Invalid start_page: {chapter_path} ({start!r})"
            )
        if end is None:
            issues.append(f"Missing end_page: {chapter_path}")
        elif not valid_end:
            issues.append(
                f"Invalid end_page: {chapter_path} ({end!r})"
            )

        if valid_start and valid_end:
            if end < start:
                issues.append(
                    f"Invalid range (end < start): {chapter_path} "
                    f"(p{start}-p{end})"
                )
            if parent_range is not None:
                parent_start, parent_end = parent_range
                if start < parent_start or end > parent_end:
                    issues.append(
                        f"Child range escapes parent: {chapter_path} "
                        f"(p{start}-p{end}, parent p{parent_start}-p{parent_end})"
                    )

            if i + 1 < len(chapters):
                next_chapter = chapters[i + 1]
                next_start = (
                    next_chapter.get('start_page')
                    if isinstance(next_chapter, dict)
                    else None
                )
                if (
                    isinstance(next_start, int)
                    and not isinstance(next_start, bool)
                    and end > next_start
                ):
                    next_title = (
                        next_chapter.get('title', 'unknown')[:40]
                        if isinstance(next_chapter, dict)
                        else 'invalid'
                    )
                    issues.append(
                        f"Overlap: '{title}' ends at p{end} "
                        f"but '{next_title}' starts at p{next_start}"
                    )

        children = chapter.get('children', [])
        if not isinstance(children, list):
            issues.append(f"Invalid children list: {chapter_path}")
        elif children:
            child_parent_range = (
                (start, end)
                if valid_start and valid_end and end >= start
                else None
            )
            issues.extend(
                validate_chapter_structure(
                    children,
                    chapter_path,
                    child_parent_range,
                )
            )

    return issues


def _iter_chapter_nodes(chapters: List[dict]):
    """Yield every chapter node recursively, ignoring malformed children."""
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        yield chapter
        children = chapter.get('children', [])
        if isinstance(children, list):
            yield from _iter_chapter_nodes(children)


def _iter_chapter_paths(
    chapters: List[dict],
    parent_path: Tuple[str, ...] = (),
):
    """Yield normalized structural paths and nodes recursively."""
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title_key = _chapter_title_key(chapter.get('title'))
        node_path = (*parent_path, title_key)
        yield node_path, chapter
        children = chapter.get('children', [])
        if isinstance(children, list):
            yield from _iter_chapter_paths(children, node_path)


def _chapter_title_key(title: Any) -> str:
    """Normalize title punctuation and spacing for merge identity checks."""
    if not isinstance(title, str):
        return ""
    return re.sub(r"[^\w]+", "", title.casefold(), flags=re.UNICODE)


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
        prepare_pdf_rasterized: Callable = None,
        pdf_transport: Optional[PdfTransport] = None,
        runtime_config: Optional[dict] = None,
    ):
        self.client = client
        self.model = model
        self._prepare_pdf = prepare_pdf
        self._prepare_pdf_rasterized = prepare_pdf_rasterized
        self._learner = learner
        self._pdf_transport = pdf_transport or GeminiPdfTransport(client)
        self._runtime_config = runtime_config or {}

    def build_prompt(self, batch_pages: List[int], batch_idx: int, total_batches: int) -> str:
        """Build the prompt for a single batch. Must be overridden."""
        raise NotImplementedError

    def build_merge_prompt(
        self,
        results: List,
        batch_pages: Optional[List[List[int]]] = None,
    ) -> str:
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

    def _get_actionable_batch_issues(
        self,
        result: Any,
        batch_idx: int,
        total_batches: int,
    ) -> Tuple[List[str], List[str]]:
        """Return all validator findings and the non-edge subset."""
        all_issues = self.validate_batch_result(
            result,
            batch_idx,
            total_batches,
        )
        if isinstance(result, list):
            chapters = result
        elif isinstance(result, dict):
            chapters = result.get('chapters', [])
        else:
            chapters = []
        actionable = self._filter_edge_issues(
            all_issues,
            chapters,
            batch_idx,
            total_batches,
        )
        return all_issues, actionable

    def can_defer_batch_issues(
        self,
        result: Any,
        issues: List[str],
        batch_idx: int,
        total_batches: int,
    ) -> bool:
        """Whether downstream recovery can safely handle these batch issues."""
        return False

    def _batch_cache_path(
        self,
        artifacts_dir: Optional[Path],
        batch_pages: List[int],
    ) -> Optional[Path]:
        if artifacts_dir is None:
            return None
        page_key = hashlib.sha256(
            ",".join(str(page) for page in batch_pages).encode("utf-8")
        ).hexdigest()[:16]
        return artifacts_dir / "batch_cache" / f"{page_key}.json"

    def _batch_cache_fingerprint(
        self,
        *,
        prompt: str,
        pdf_data: bytes,
        batch_pages: List[int],
    ) -> str:
        payload = {
            "version": 1,
            "operation": self.operation_name,
            "model": self.model,
            "transport": (
                self._pdf_transport.cache_identity()
                if hasattr(self._pdf_transport, "cache_identity")
                else {
                    "type": type(self._pdf_transport).__qualname__,
                }
            ),
            "prompt": prompt,
            "pages": batch_pages,
            "pdf_sha256": hashlib.sha256(pdf_data).hexdigest(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_cached_batch_result(
        self,
        *,
        cache_path: Optional[Path],
        fingerprint: str,
        batch_idx: int,
        total_batches: int,
    ) -> Any:
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"[{self.operation_name}] Ignoring unreadable batch cache "
                f"{cache_path}: {exc}"
            )
            return None
        if (
            payload.get("version") != 1
            or payload.get("fingerprint") != fingerprint
            or "result" not in payload
        ):
            return None

        result = payload["result"]
        all_issues, actionable = self._get_actionable_batch_issues(
            result,
            batch_idx,
            total_batches,
        )
        if actionable:
            logger.warning(
                f"[{self.operation_name}] Cached batch result no longer "
                f"passes validation ({len(actionable)} actionable issue(s)); "
                "rerunning the model"
            )
            return None
        logger.info(
            f"[{self.operation_name}] Reusing validated batch result from "
            f"{cache_path}"
        )
        if all_issues:
            logger.info(
                f"[{self.operation_name}] Cached result retains "
                f"{len(all_issues)} tolerated edge issue(s)"
            )
        return result

    @staticmethod
    def _save_cached_batch_result(
        cache_path: Optional[Path],
        fingerprint: str,
        result: Any,
    ) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "fingerprint": fingerprint,
                    "result": result,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(cache_path)

    def _get_configured_agent_model(self, config_key: str):
        """
        Get a pydantic-ai Model for the JSON repair agent.

        ``config_key`` lets merge repair use an explicitly stronger model
        without invalidating or rerunning already accepted PDF batches.
        Priority: explicit refine.<config_key> > explicit refine.agent >
        legacy Anthropic Haiku default > legacy Poe fallback.
        """
        config = self._runtime_config
        providers = config.get('credentials', {}).get('providers', {})

        # Explicit selection must win even when legacy provider credentials
        # are also present. Otherwise a requested model upgrade is silently
        # ignored merely because an Anthropic key exists in the same config.
        refine_cfg = config.get('refine', {})
        agent_cfg = (
            refine_cfg.get(config_key, {})
            or refine_cfg.get('agent', {})
        )
        if agent_cfg:
            provider_name = agent_cfg.get('provider')
            model_name = agent_cfg.get('model')
            if not provider_name or not model_name:
                raise ValueError(
                    "refine.agent requires both provider and model"
                )
            if provider_name not in providers:
                raise ValueError(
                    f"refine.agent provider '{provider_name}' is not configured"
                )

            p = providers[provider_name]
            provider_type = p.get('type') or (
                'anthropic' if provider_name == 'anthropic' else 'openai'
            )
            if provider_type == 'anthropic':
                from pydantic_ai.models.anthropic import AnthropicModel
                from pydantic_ai.providers.anthropic import AnthropicProvider
                provider = AnthropicProvider(
                    api_key=os.environ.get('ANTHROPIC_API_KEY') or p.get('api_key'),
                    base_url=p.get('base_url'),
                )
                logger.info(f"[agent-model] Using explicit Anthropic {model_name}")
                return AnthropicModel(model_name, provider=provider)
            if provider_type in {'openai', 'codex'}:
                from .agent_model import build_openai_agent_model
                logger.info(
                    f"[agent-model] Using explicit {provider_type} {model_name}"
                )
                return build_openai_agent_model(model_name, p)
            if provider_type == 'google':
                from pydantic_ai.models.google import GoogleModel
                from pydantic_ai.providers.google import GoogleProvider
                from google.genai import Client
                from google.genai.types import HttpOptions
                client = Client(
                    api_key=p.get('api_key'),
                    http_options=HttpOptions(base_url=p.get('base_url')),
                )
                google_provider = GoogleProvider(client=client)
                logger.info(f"[agent-model] Using explicit Google {model_name}")
                return GoogleModel(model_name, provider=google_provider)
            raise ValueError(
                f"Unsupported refine.agent provider type '{provider_type}'"
            )

        # Backward-compatible default: Anthropic Haiku.
        if 'anthropic' in providers:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
            p = providers['anthropic']
            provider = AnthropicProvider(
                api_key=os.environ.get('ANTHROPIC_API_KEY') or p.get('api_key'),
                base_url=p.get('base_url'),
            )
            model_name = 'claude-haiku-4-5-20251001'
            logger.info(f"[agent-model] Using Anthropic {model_name}")
            return AnthropicModel(model_name, provider=provider)

        # Final backward-compatible fallback.
        if 'poe' in providers:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            p = providers['poe']
            provider = OpenAIProvider(
                api_key=p.get('api_key'),
                base_url=p.get('base_url'),
            )
            model_name = 'Gemini-2.5-Flash'
            logger.info(f"[agent-model] Fallback to Poe {model_name}")
            return OpenAIChatModel(model_name, provider=provider)

        raise ValueError(
            "No suitable provider found for agent model. "
            "Need 'anthropic' or 'poe' in credentials.providers."
        )

    def _get_agent_model(self):
        """Return the normal per-batch JSON repair model."""
        return self._get_configured_agent_model('agent')

    def _get_merge_agent_model(self):
        """Return a merge-only repair model when explicitly configured."""
        refine_cfg = self._runtime_config.get('refine', {})
        if refine_cfg.get('merge_agent'):
            return self._get_configured_agent_model('merge_agent')
        return self._get_agent_model()

    def _build_generate_fn(self, prompt, pdf_data, config, op_name):
        """
        Build a generate_fn closure for the agent loop.

        The returned function follows the contract:
            generate_fn(prefix=None) -> str

        Transport implementations decide how to represent a continuation. The
        prefix is always the JSON agent's validated content, never a partial
        response guessed by this orchestration layer.
        """
        def generate_fn(prefix=None):
            return self._pdf_transport.generate_pdf(
                model=self.model,
                prompt=prompt,
                pdf_data=pdf_data,
                config=config,
                operation_name=op_name,
                prefix=prefix,
            )
        return generate_fn

    def validate_merge(self, merged: Any, original_results: List) -> bool:
        """Validate merged result. Return True if acceptable."""
        return True

    def get_merge_validation_issues(
        self,
        merged: Any,
        original_results: List,
    ) -> List[str]:
        """Return concrete merge problems, or an empty list on success.

        Existing callers can keep overriding ``validate_merge``.  Concrete
        calls should override this method when they can provide actionable
        diagnostics for a model retry.
        """
        if self.validate_merge(merged, original_results):
            return []
        return ["The merged result failed structural validation."]

    def build_merge_repair_prompt(
        self,
        original_prompt: str,
        merged: Any,
        issues: List[str],
    ) -> str:
        """Request a corrected merge using the actual validator feedback."""
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        candidate_json = json.dumps(merged, ensure_ascii=False, indent=2)
        if len(candidate_json) > 16000:
            candidate_json = candidate_json[:16000] + "\n... (truncated)"

        return f"""{original_prompt}

--- YOUR PREVIOUS MERGE FAILED STRUCTURAL VALIDATION ---

Validator findings:
{issues_text}

Previous merged candidate:
{candidate_json}

Return the complete corrected JSON. Re-evaluate the per-batch PDF-page
evidence in the original prompt; do not repeat a page value that the
validator identified as unsupported or overlapping.
"""

    def parse_result(self, response_text: str) -> Any:
        """Parse LLM response. Default: parse as JSON."""
        return parse_llm_json(response_text, operation_name=self.operation_name)

    def _build_merge_generate_fn(self, prompt, config, op_name):
        """Build a generate_fn for merge (text-only, no PDF)."""
        generator_cfg = (
            self._runtime_config.get('refine', {}).get('merge_generator', {})
        )
        if generator_cfg:
            provider_name = generator_cfg.get('provider')
            model_name = generator_cfg.get('model')
            providers = (
                self._runtime_config.get('credentials', {}).get('providers', {})
            )
            if not provider_name or not model_name:
                raise ValueError(
                    "refine.merge_generator requires both provider and model"
                )
            if provider_name not in providers:
                raise ValueError(
                    f"refine.merge_generator provider {provider_name!r} "
                    "is not configured"
                )

            provider_config = dict(providers[provider_name])
            provider_type = provider_config.get('type', 'openai')
            if provider_type == 'codex':
                from pdf2epub.core.whole.model_factory import (
                    _load_codex_openai_provider,
                )

                provider_config = _load_codex_openai_provider(provider_config)
            elif provider_type != 'openai':
                raise ValueError(
                    "refine.merge_generator supports only OpenAI-compatible "
                    f"or Codex providers, got {provider_type!r}"
                )

            from openai import OpenAI

            client = OpenAI(
                api_key=provider_config.get('api_key'),
                base_url=provider_config.get('base_url'),
                timeout=600,
            )

            def generate_fn(prefix=None):
                messages = [{'role': 'user', 'content': prompt}]
                if prefix:
                    messages.extend([
                        {'role': 'assistant', 'content': prefix},
                        {
                            'role': 'user',
                            'content': (
                                "Continue exactly after the existing JSON "
                                "prefix. Output only the remaining JSON."
                            ),
                        },
                    ])
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                return response.choices[0].message.content or ""

            return generate_fn

        def generate_fn(prefix=None):
            return self._pdf_transport.generate_text(
                model=self.model,
                prompt=prompt,
                config=config,
                operation_name=op_name,
                prefix=prefix,
            )
        return generate_fn

    def merge_results(
        self,
        results: List,
        artifacts_dir: Optional[Path] = None,
        batch_pages: Optional[List[List[int]]] = None,
    ) -> Any:
        """
        Merge results from multiple batches.

        Default: LLM-based merge using build_merge_prompt(), with retry.
        Uses agent loop for JSON validation and truncation recovery.
        Override entirely for rule-based merge (e.g. TocDetectionCall).
        """
        if len(results) == 1:
            return results[0]

        # Load agent config
        agent_config = self._runtime_config.get('refine', {})
        request_limit = agent_config.get('agent_request_limit', 100)
        max_continuations = agent_config.get('max_continuations', 5)

        merged = None
        if batch_pages is None:
            # Keep the base class compatible with small external subclasses
            # that still implement the historical one-argument hook.
            base_prompt = self.build_merge_prompt(results)
        else:
            base_prompt = self.build_merge_prompt(results, batch_pages=batch_pages)
        prompt = base_prompt
        for attempt in range(1 + self.merge_max_retries):
            config = self.client.get_default_config(temperature=0.1)
            # NOTE: No response_mime_type="application/json" — agent loop
            # handles JSON validation. Continuation fragments are not valid JSON.

            op_name = f"{self.operation_name} merge"
            if attempt > 0:
                op_name += f" (retry {attempt})"

            generate_fn = self._build_merge_generate_fn(prompt, config, op_name)
            merge_attempt_artifacts = None
            if artifacts_dir:
                merge_attempt_artifacts = artifacts_dir / f"attempt_{attempt+1}"
            try:
                response_text = run_agent_loop_sync(
                    generate_fn=generate_fn,
                    system_prompt=JSON_REFINE_PROMPT,
                    agent_model=self._get_merge_agent_model(),
                    max_continuations=max_continuations,
                    request_limit=request_limit,
                    artifacts_dir=merge_attempt_artifacts,
                )
            except AgentLoopExhausted:
                logger.warning(
                    f"[{self.operation_name}] Agent loop exhausted during merge "
                    f"(attempt {attempt+1}/{1 + self.merge_max_retries})"
                )
                if attempt < self.merge_max_retries:
                    continue
                raise

            # Parse with retry on JSON errors (defense-in-depth)
            try:
                merged = self.parse_result(response_text)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"[{self.operation_name}] JSON parse failed after merge agent loop "
                    f"(attempt {attempt+1}): {e}"
                )
                if attempt < self.merge_max_retries:
                    continue
                raise

            issues = self.get_merge_validation_issues(merged, results)
            if not issues:
                logger.info(
                    f"[{self.operation_name}] Merged {len(results)} batches successfully"
                )
                return merged

            logger.warning(
                f"[{self.operation_name}] Merge validation failed "
                f"(attempt {attempt + 1}/{1 + self.merge_max_retries})"
            )
            for issue in issues[:5]:
                logger.warning(f"  - {issue}")
            if len(issues) > 5:
                logger.warning(f"  ... and {len(issues) - 5} more")
            if attempt < self.merge_max_retries:
                # Repair prompts always start from the same evidence. Building
                # on the previous repair prompt recursively would duplicate
                # the full batch payload and grow the request every round.
                prompt = self.build_merge_repair_prompt(base_prompt, merged, issues)

        message = (
            f"[{self.operation_name}] Merge validation failed after "
            f"{1 + self.merge_max_retries} attempt(s); refusing to continue "
            f"with an untrusted structure result"
        )
        logger.error(message)
        raise MergeValidationError(message)

    def run(self, pdf_path: Path, pages: List[int], artifacts_dir: Optional[Path] = None) -> Any:
        """
        Execute the adaptive PDF→LLM call.

        Args:
            pdf_path: Path to PDF file
            pages: List of 1-indexed page numbers to process
            artifacts_dir: If provided, save agent loop artifacts here for debugging

        Returns:
            Parsed result (single batch) or merged result (multi-batch)
        """
        def process_batch(batch_pages, batch_idx, total_batches, use_rasterized=False):
            # Choose PDF preparation method
            if use_rasterized and self._prepare_pdf_rasterized:
                pdf_data = self._prepare_pdf_rasterized(pdf_path, include_pages=batch_pages)
                if pdf_data is None:
                    logger.warning(
                        f"Rasterization failed for batch {batch_idx+1}, "
                        f"falling back to normal PDF"
                    )
                    pdf_data = self._prepare_pdf(pdf_path, include_pages=batch_pages)
            else:
                pdf_data = self._prepare_pdf(pdf_path, include_pages=batch_pages)

            if pdf_data is None:
                batch_start, batch_end = min(batch_pages), max(batch_pages)
                raise RuntimeError(
                    f"Failed to prepare PDF batch (pages {batch_start}-{batch_end})"
                )

            original_prompt = self.build_prompt(batch_pages, batch_idx, total_batches)
            prompt = original_prompt
            result = None
            cache_path = self._batch_cache_path(
                artifacts_dir,
                batch_pages,
            )
            cache_fingerprint = self._batch_cache_fingerprint(
                prompt=original_prompt,
                pdf_data=pdf_data,
                batch_pages=batch_pages,
            )
            cached_result = self._load_cached_batch_result(
                cache_path=cache_path,
                fingerprint=cache_fingerprint,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )
            if cached_result is not None:
                return cached_result

            # Load agent config
            agent_config = self._runtime_config.get('refine', {})
            request_limit = agent_config.get('agent_request_limit', 100)
            max_continuations = agent_config.get('max_continuations', 5)

            for attempt in range(1 + self.batch_validation_retries):
                config = self.client.get_default_config(temperature=0.1)
                # NOTE: No response_mime_type="application/json" — the agent loop
                # handles JSON validation. Continuation fragments are not valid JSON,
                # so JSON mode would break the Gemini API for continuation calls.

                op_name = f"{self.operation_name} batch {batch_idx+1}/{total_batches}"
                if use_rasterized:
                    op_name += " (rasterized)"
                if attempt > 0:
                    op_name += f" (fix {attempt})"

                # Agent loop: generate → agent inspects → continue/complete
                generate_fn = self._build_generate_fn(prompt, pdf_data, config, op_name)
                batch_artifacts = None
                if artifacts_dir:
                    batch_artifacts = artifacts_dir / f"batch_{batch_idx+1}_attempt_{attempt+1}"
                try:
                    response_text = run_agent_loop_sync(
                        generate_fn=generate_fn,
                        system_prompt=JSON_REFINE_PROMPT,
                        agent_model=self._get_agent_model(),
                        max_continuations=max_continuations,
                        request_limit=request_limit,
                        artifacts_dir=batch_artifacts,
                    )
                except AgentLoopExhausted:
                    logger.warning(
                        f"[{self.operation_name}] Agent loop exhausted for "
                        f"batch {batch_idx+1}/{total_batches} "
                        f"(attempt {attempt+1}/{1 + self.batch_validation_retries})"
                    )
                    if attempt < self.batch_validation_retries:
                        continue
                    raise

                # Parse with retry on JSON errors (defense-in-depth)
                try:
                    result = self.parse_result(response_text)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"[{self.operation_name}] JSON parse failed after agent loop "
                        f"for batch {batch_idx+1}/{total_batches} "
                        f"(attempt {attempt+1}): {e}"
                    )
                    if attempt < self.batch_validation_retries:
                        continue
                    raise

                # Run batch validation hook
                all_issues, actionable_issues = (
                    self._get_actionable_batch_issues(
                        result,
                        batch_idx,
                        total_batches,
                    )
                )
                if not all_issues:
                    self._save_cached_batch_result(
                        cache_path,
                        cache_fingerprint,
                        result,
                    )
                    return result

                if not actionable_issues:
                    logger.info(
                        f"[{self.operation_name}] Batch {batch_idx+1}/{total_batches}: "
                        f"{len(all_issues)} edge issue(s) tolerated"
                    )
                    self._save_cached_batch_result(
                        cache_path,
                        cache_fingerprint,
                        result,
                    )
                    return result

                if self.can_defer_batch_issues(
                    result,
                    actionable_issues,
                    batch_idx,
                    total_batches,
                ):
                    logger.warning(
                        f"[{self.operation_name}] Deferring boundary-only "
                        "issues to downstream OCR boundary verification"
                    )
                    self._save_cached_batch_result(
                        cache_path,
                        cache_fingerprint,
                        result,
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
                    raise BatchValidationError(
                        f"[{self.operation_name}] Batch "
                        f"{batch_idx+1}/{total_batches} remained invalid after "
                        f"{1 + self.batch_validation_retries} attempt(s); "
                        "refusing to continue with an untrusted batch result"
                    )

            return result

        # Check if rasterization is available
        can_rasterize = self._prepare_pdf_rasterized is not None

        results, successful_batches = run_adaptive_batches(
            pages, process_batch, self._learner, is_503_error,
            self.operation_name, overlap=self.overlap,
            can_rasterize=can_rasterize,
            return_batches=True,
        )

        merge_artifacts = artifacts_dir / "merge" if artifacts_dir else None
        return self.merge_results(
            results,
            artifacts_dir=merge_artifacts,
            batch_pages=successful_batches,
        )


# ---------------------------------------------------------------------------
# Concrete call types
# ---------------------------------------------------------------------------

class TocDetectionCall(AdaptivePdfCall):
    """Detect TOC location in PDF."""

    operation_name = "TOC location detection"
    overlap = 0

    def build_prompt(self, batch_pages, batch_idx, total_batches):
        observed = getattr(self, '_observed_pages_by_batch', {})
        observed[batch_idx] = set(batch_pages)
        self._observed_pages_by_batch = observed
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
            raise ValueError(
                f"TOC detection returned {type(result).__name__}, expected object"
            )

        has_toc = result.get('has_toc')
        if not isinstance(has_toc, bool):
            raise ValueError("TOC detection result must contain boolean 'has_toc'")
        if not has_toc:
            return {
                'has_toc': False,
                'toc_start': None,
                'toc_end': None,
            }

        toc_start = result.get('toc_start')
        toc_end = result.get('toc_end')
        if (
            not isinstance(toc_start, int)
            or isinstance(toc_start, bool)
            or not isinstance(toc_end, int)
            or isinstance(toc_end, bool)
            or toc_start < 1
            or toc_end < toc_start
        ):
            raise ValueError(
                "TOC detection with has_toc=true requires positive integer "
                "toc_start/toc_end with toc_end >= toc_start"
            )
        return {
            'has_toc': True,
            'toc_start': toc_start,
            'toc_end': toc_end,
        }

    def validate_batch_result(self, result, batch_idx, total_batches):
        if not result.get('has_toc'):
            return []
        observed = getattr(self, '_observed_pages_by_batch', {}).get(
            batch_idx,
            set(),
        )
        return self._toc_evidence_issues(result, observed)

    @staticmethod
    def _toc_evidence_issues(result, observed):
        start = result['toc_start']
        end = result['toc_end']
        claimed_count = end - start + 1
        if claimed_count > len(observed):
            return [
                f"TOC claims {claimed_count} contiguous pages (p{start}-p{end}) "
                f"but this batch observed only {len(observed)} page(s)"
            ]
        unsupported = [
            page
            for page in range(start, end + 1)
            if page not in observed
        ]
        if unsupported:
            return [
                "TOC range includes page(s) not observed in this PDF batch: "
                + ", ".join(str(page) for page in unsupported[:10])
            ]
        return []

    def merge_results(self, results, artifacts_dir=None, batch_pages=None):
        """Rule-based: pick first result with has_toc=True."""
        if batch_pages is not None and len(batch_pages) != len(results):
            raise MergeValidationError(
                "TOC batch evidence does not align with detection results"
            )
        if batch_pages is not None:
            for index, result in enumerate(results):
                if not isinstance(result, dict) or not result.get('has_toc'):
                    continue
                observed = set(batch_pages[index])
                evidence_issues = self._toc_evidence_issues(
                    result,
                    observed,
                )
                if evidence_issues:
                    raise MergeValidationError(
                        evidence_issues[0]
                    )

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

    def __init__(self, client, model, prepare_pdf, learner, book_title: str,
                 toc_reference: str = None, prepare_pdf_rasterized: Callable = None,
                 pdf_transport: Optional[PdfTransport] = None,
                 overlap_pages: Optional[int] = None,
                 runtime_config: Optional[dict] = None):
        super().__init__(
            client, model, prepare_pdf, learner, prepare_pdf_rasterized,
            pdf_transport=pdf_transport,
            runtime_config=runtime_config,
        )
        self.book_title = book_title
        self.toc_reference = toc_reference
        if overlap_pages is not None:
            self.overlap = max(0, int(overlap_pages))

    def build_prompt(self, batch_pages, batch_idx, total_batches):
        observed = getattr(self, '_observed_pages_by_batch', {})
        observed[batch_idx] = set(batch_pages)
        self._observed_pages_by_batch = observed
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

## Complete Example Output

Below is a complete example for a 353-page academic book ("{self.book_title}" has \
{batch_end - batch_start + 1} pages — scale your output accordingly). \
The key point: your output MUST include metadata AND a complete "chapters" array. \
A short book may have fewer chapters, but every chapter must be listed.

```json
{{
    "author": "Alex Callinicos",
    "language": "chinese",
    "is_vertical_text": false,
    "has_footnotes": true,
    "cover_page": {{"page_number": 1}},
    "table_of_contents": {{"start_page": 10, "end_page": 11}},
    "back_cover": {{"page_number": 2}},
    "chapters": [
        {{"title": "前言與致謝", "level": 1, "start_page": 6, "end_page": 8}},
        {{"title": "導言", "level": 1, "start_page": 12, "end_page": 25}},
        {{
            "title": "第一部分 四條死路", "level": 1, "start_page": 26, "end_page": 203,
            "children": [
                {{
                    "title": "第一章 現代性及其承諾", "level": 2, "start_page": 27, "end_page": 71,
                    "children": [
                        {{"title": "第一節 在社會學式的懷疑與法治之間", "level": 3, "start_page": 27, "end_page": 52}},
                        {{"title": "第二節 對馬克思和羅爾斯的支持與反對", "level": 3, "start_page": 53, "end_page": 71}}
                    ]
                }},
                {{
                    "title": "第二章 在相對主義與普遍主義之間", "level": 2, "start_page": 72, "end_page": 112,
                    "children": [
                        {{"title": "第一節 資本主義與對資本主義的批判", "level": 3, "start_page": 72, "end_page": 97}},
                        {{"title": "第二節 普遍與特殊的辯證法", "level": 3, "start_page": 98, "end_page": 112}}
                    ]
                }},
                {{
                    "title": "第三章 觸摸虛空", "level": 2, "start_page": 113, "end_page": 162,
                    "children": [
                        {{"title": "第一節 例外即規範", "level": 3, "start_page": 113, "end_page": 120}},
                        {{"title": "第二節 巴迪烏的本體論", "level": 3, "start_page": 121, "end_page": 150}},
                        {{"title": "第三節 紀傑克與無產階級", "level": 3, "start_page": 151, "end_page": 162}}
                    ]
                }},
                {{
                    "title": "第四章 存有的慷慨", "level": 2, "start_page": 163, "end_page": 203,
                    "children": [
                        {{"title": "第一節 一切都是神恩", "level": 3, "start_page": 163, "end_page": 165}},
                        {{"title": "第二節 革命主體對陣馬克思主義的客觀主義", "level": 3, "start_page": 166, "end_page": 188}},
                        {{"title": "第三節 拒斥超越", "level": 3, "start_page": 189, "end_page": 203}}
                    ]
                }}
            ]
        }},
        {{
            "title": "第二部分 進步的三個維度", "level": 1, "start_page": 204, "end_page": 336,
            "children": [
                {{
                    "title": "第五章 批判實在論意義上的本體論", "level": 2, "start_page": 205, "end_page": 239,
                    "children": [
                        {{"title": "第一節 迄今為止的故事", "level": 3, "start_page": 205, "end_page": 211}},
                        {{"title": "第二節 實在論的諸維度", "level": 3, "start_page": 212, "end_page": 239}}
                    ]
                }},
                {{
                    "title": "第六章 結構與矛盾", "level": 2, "start_page": 240, "end_page": 283,
                    "children": [
                        {{"title": "第一節 關於結構的實在論", "level": 3, "start_page": 240, "end_page": 249}},
                        {{"title": "第二節 矛盾的首要性", "level": 3, "start_page": 250, "end_page": 273}},
                        {{"title": "第三節 一種自然辯證法?", "level": 3, "start_page": 274, "end_page": 283}}
                    ]
                }},
                {{
                    "title": "第七章 正義與普遍性", "level": 2, "start_page": 284, "end_page": 317,
                    "children": [
                        {{"title": "第一節 從事實到價值", "level": 3, "start_page": 284, "end_page": 291}},
                        {{"title": "第二節 平等與幸福", "level": 3, "start_page": 292, "end_page": 306}},
                        {{"title": "第三節 為何平等重要", "level": 3, "start_page": 307, "end_page": 317}}
                    ]
                }},
                {{"title": "第八章 結論", "level": 2, "start_page": 318, "end_page": 336}}
            ]
        }},
        {{"title": "徵引文獻", "level": 1, "start_page": 337, "end_page": 352, "type": "bibliography"}},
        {{"title": "索引", "level": 1, "start_page": 353, "end_page": 353, "type": "index"}}
    ]
}}
```

**IMPORTANT**:
- Use PDF page numbers from "PDF Page: X" labels, NOT printed page numbers
- Preserve the original language for all titles and author names
- Your output must contain ALL chapters — do not stop after metadata
"""

    @staticmethod
    def _format_observed_page_set(pages: List[int]) -> str:
        """Compact a concrete PDF page set without implying missing pages exist."""
        if not pages:
            return "(no PDF pages observed)"

        ranges = []
        start = previous = pages[0]
        for page in pages[1:]:
            if page == previous + 1:
                previous = page
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = page
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return ", ".join(ranges)

    def build_merge_prompt(self, results, batch_pages=None):
        batch_chapters = [
            r if isinstance(r, list) else r.get('chapters', [])
            for r in results
        ]

        if batch_pages is not None and len(batch_pages) != len(batch_chapters):
            raise ValueError(
                "Merge batch evidence does not align with the batch results "
                f"({len(batch_pages)} page sets for {len(batch_chapters)} results)"
            )

        batch_summaries = []
        for i, chapters in enumerate(batch_chapters):
            page_evidence = "not recorded"
            if batch_pages is not None:
                page_evidence = self._format_observed_page_set(batch_pages[i])
            batch_summaries.append(
                f"=== Batch {i+1}/{len(batch_chapters)} | OBSERVED PDF PAGES: {page_evidence} ===\n"
                f"{json.dumps(chapters, ensure_ascii=False, indent=2)}"
            )

        return f"""You are merging chapter structure results from {len(batch_chapters)} overlapping batches of the same book.

**Book Title**: {self.book_title}

Each batch analyzed a different page range with some overlap. The overlap region may have chapters recognized differently by each batch. Your job is to produce ONE unified, correct chapter list.

Rules:
1. Each chapter should appear exactly ONCE
2. The ``OBSERVED PDF PAGES`` declaration is evidence, not decoration. A batch
   can support a start_page or end_page only when that exact physical PDF page
   is in its observed page set. A value outside that set may have been copied
   from the book TOC or inferred globally; it must never override an in-range
   value from another batch.
3. Reconcile start_page and end_page independently. For each bound, prefer an
   in-range claim from the batch that actually saw that page. Do not use a
   batch merely because it has more nodes or a wider-looking hierarchy.
4. A chapter that appears as a top-level entry in one batch but as a child in
   another should use the hierarchy supported by the batch that observes the
   relevant heading and its surrounding pages.
5. Preserve the hierarchical structure (children nested under parents).
6. Chapters must be ordered by start_page and must not overlap as siblings.
7. Do NOT invent chapters or page numbers. One deterministic boundary repair
   is allowed: when a source-supported later sibling heading starts inside an
   earlier batch's overextended chapter range, end the previous sibling either
   immediately before that heading or on the shared heading page when source
   ranges treat the transition page as belonging to both siblings. If no
   candidate or such adjacent heading supports a required bound, return null
   rather than guessing; the caller will fail closed and request a new analysis.

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
        if not chapters:
            return ["No chapters extracted — output appears truncated or incomplete"]
        issues = validate_chapter_structure(chapters)
        if total_batches == 1:
            observed = getattr(
                self,
                '_observed_pages_by_batch',
                {},
            ).get(batch_idx, set())
            for _node_path, chapter in _iter_chapter_paths(chapters):
                for field in ('start_page', 'end_page'):
                    value = chapter.get(field)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value not in observed
                    ):
                        issues.append(
                            f"Single-batch {field}={value} for "
                            f"{chapter.get('title')!r} was not observed"
                        )
        return issues

    def can_defer_batch_issues(
        self,
        result,
        issues,
        batch_idx,
        total_batches,
    ):
        if total_batches != 1 or not issues:
            return False
        if not all(
            issue.startswith("Single-batch ")
            and issue.endswith(" was not observed")
            for issue in issues
        ):
            return False

        observed = getattr(
            self,
            '_observed_pages_by_batch',
            {},
        ).get(batch_idx, set())
        if not observed:
            return False
        page_min, page_max = min(observed), max(observed)
        chapters = (
            result
            if isinstance(result, list)
            else result.get('chapters', [])
        )
        for _node_path, chapter in _iter_chapter_paths(chapters):
            for field in ('start_page', 'end_page'):
                value = chapter.get(field)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < page_min
                    or value > page_max
                ):
                    return False
        return True

    @staticmethod
    def _single_batch_page_range_issues(result, observed_pages):
        """Reject impossible bounds while allowing intentionally excluded gaps."""
        observed = set(observed_pages)
        if not observed:
            return ["Single-batch result has no observed PDF page range"]
        page_min, page_max = min(observed), max(observed)
        chapters = (
            result
            if isinstance(result, list)
            else result.get('chapters', [])
        )
        issues = []
        for _node_path, chapter in _iter_chapter_paths(chapters):
            for field in ('start_page', 'end_page'):
                value = chapter.get(field)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and not page_min <= value <= page_max
                ):
                    issues.append(
                        f"Single-batch {field}={value} for "
                        f"{chapter.get('title')!r} is outside observed "
                        f"PDF range {page_min}-{page_max}"
                    )
        return issues

    def get_merge_validation_issues(self, merged, _original_results):
        """Validate only deterministic tree-structure invariants."""
        chapters = (
            merged
            if isinstance(merged, list)
            else merged.get('chapters', [])
        )
        return validate_chapter_structure(chapters)

    def merge_results(self, results, artifacts_dir=None, batch_pages=None):
        if len(results) == 1:
            issues = self.get_merge_validation_issues(
                results[0],
                results,
            )
            if batch_pages is not None:
                if len(batch_pages) != 1:
                    issues.append(
                        "Single-batch evidence count does not match result count"
                    )
                else:
                    issues.extend(
                        self._single_batch_page_range_issues(
                            results[0],
                            batch_pages[0],
                        )
                    )
            if issues:
                raise MergeValidationError(
                    f"[{self.operation_name}] Single-batch result failed "
                    f"evidence validation: {issues[:5]}"
                )
            return results[0]

        # Extract metadata from first/last batch (rule-based, no LLM needed)
        # Handle bare list results (LLM may omit the wrapper object)
        metadata = {}
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue
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
        merged = super().merge_results(
            results,
            artifacts_dir=artifacts_dir,
            batch_pages=batch_pages,
        )

        # LLM may return a bare list instead of {"chapters": [...]}
        if isinstance(merged, list):
            chapters = merged
        else:
            chapters = merged.get('chapters', [])

        return {**metadata, 'chapters': chapters}
