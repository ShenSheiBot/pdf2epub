"""
NestedPartProcessor - Handles recursive splitting with shared retry budget.

This module provides:
- NestedPart dataclass for representing ephemeral nested parts
- NestedPartProcessor for processing with automatic splitting on failure
- Tree-based result aggregation

Key design:
- Nested parts are ephemeral (in-memory only)
- Dynamic depth based on token count threshold
- Shared retry budget across entire subtree
- No context injection for nested parts (enables parallelism)
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import tiktoken

from .splitter_strategies import ContentSplitter, MarkdownStructureSplitter
from ...core.tracking import ProcessingTracker, AttemptRecord, SplitRecord, ErrorType


def classify_error(exception: Exception) -> str:
    """Classify an exception into an error type for tracking."""
    error_msg = str(exception).lower()

    # Check for truncation errors
    if any(keyword in error_msg for keyword in ['truncat', 'incomplete', 'cut off', 'token ratio']):
        return ErrorType.TRUNCATION.value

    # Check for content filter/safety errors
    if any(keyword in error_msg for keyword in ['safety', 'content filter', 'blocked', 'harmful', 'inappropriate']):
        return ErrorType.CONTENT_FILTER.value

    # Check for rate limit errors
    if any(keyword in error_msg for keyword in ['rate limit', 'quota', '429', 'too many requests']):
        return ErrorType.RATE_LIMIT.value

    # Check for timeout errors
    if any(keyword in error_msg for keyword in ['timeout', 'timed out', 'deadline']):
        return ErrorType.TIMEOUT.value

    # Check for API errors
    if any(keyword in error_msg for keyword in ['api error', '500', '502', '503', 'service unavailable']):
        return ErrorType.API_ERROR.value

    # Check for validation errors
    if any(keyword in error_msg for keyword in ['validation', 'invalid', 'failed to validate']):
        return ErrorType.VALIDATION.value

    # Check for parse errors
    if any(keyword in error_msg for keyword in ['parse', 'json', 'decode']):
        return ErrorType.PARSE_ERROR.value

    return ErrorType.UNKNOWN.value

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class NestedPart:
    """
    Represents an ephemeral nested part for recursive splitting.

    These parts exist only during processing and are combined back
    to their parent when complete. No files are created for nested parts.
    """
    id: str                           # e.g., "part3.part1.part2"
    sort_key: List[int]               # e.g., [3, 1, 2] for sorting
    content: str                      # Content to process
    token_count: int                  # Token count
    result: Optional[str] = None      # Processed result
    children: List['NestedPart'] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Get the nesting depth (0 for root)."""
        return len(self.sort_key) - 1

    def __lt__(self, other: 'NestedPart') -> bool:
        """Enable sorting by sort_key."""
        return self.sort_key < other.sort_key


class NestedPartProcessor:
    """
    Processes content with automatic splitting on failure.

    Uses a shared retry budget across the entire subtree to prevent
    infinite retries. When processing fails and budget allows, the
    content is split into smaller parts which are processed in parallel.

    Example:
        processor = NestedPartProcessor(
            splitter=SimpleSplitter(),
            total_retries=5,
            min_tokens_to_split=500
        )

        root = NestedPart(
            id="chapter_5.part3",
            sort_key=[3],
            content="...",
            token_count=5000
        )

        result = processor.process_with_splitting(root, my_process_fn)
    """

    def __init__(
        self,
        splitter: ContentSplitter = None,
        total_retries: int = 5,
        min_tokens_to_split: int = 500,
        max_workers: int = 4,
        tracker: ProcessingTracker = None,
        model: str = "unknown"
    ):
        """
        Initialize the processor.

        Args:
            splitter: ContentSplitter instance (defaults to MarkdownStructureSplitter)
            total_retries: Total retry budget for entire subtree
            min_tokens_to_split: Stop splitting if part has fewer tokens
            max_workers: Max parallel workers for processing nested parts
            tracker: ProcessingTracker for recording attempts and splits
            model: Model name for tracking (primary model, may fallback)
        """
        self.splitter = splitter or MarkdownStructureSplitter()
        self.total_retries = total_retries
        self.remaining_retries = total_retries
        self.min_tokens_to_split = min_tokens_to_split
        self.max_workers = max_workers
        self.tracker = tracker
        self.model = model

        # Track statistics
        self.stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "splits_performed": 0,
            "max_depth_reached": 0
        }

    def process_with_splitting(
        self,
        part: NestedPart,
        process_fn: Callable[[str, str], str]
    ) -> str:
        """
        Process a part with automatic splitting on failure.

        Args:
            part: The NestedPart to process
            process_fn: Function that takes (content, part_id) and returns processed result
                        Should raise exception on failure

        Returns:
            Processed and combined result

        Raises:
            Exception: If all retries exhausted and content too small to split
        """
        self.stats["max_depth_reached"] = max(
            self.stats["max_depth_reached"],
            part.depth
        )

        # Try to process directly
        # Note: We only retry at this level if we can split the content.
        # If content is too small to split, we fail immediately since
        # generate_with_validation already tried all models internally.
        try:
            self.remaining_retries -= 1
            self.stats["total_attempts"] += 1

            logger.debug(
                f"Processing {part.id} (tokens: {part.token_count}, "
                f"retries left: {self.remaining_retries})"
            )

            result = process_fn(part.content, part.id)
            part.result = result
            self.stats["successful_attempts"] += 1

            # Record successful attempt
            if self.tracker:
                attempt = AttemptRecord(
                    timestamp=time.time(),
                    status="completed",
                    model=self.model,
                    input_tokens=part.token_count,
                    output_tokens=len(tokenizer.encode(result))
                )
                self.tracker.record_attempt(part.id, attempt)

            logger.debug(f"Successfully processed {part.id}")
            return result

        except Exception as e:
            # Record failed attempt with error classification
            if self.tracker:
                error_type = classify_error(e)

                # Extract error_output_path from exception message if present
                error_output_path = None
                error_msg = str(e)
                if "Error outputs saved to:" in error_msg:
                    # Extract the first (most recent) error output path
                    path_match = re.search(r'Error outputs saved to:\n\s+-\s+(\S+)', error_msg)
                    if path_match:
                        error_output_path = path_match.group(1)

                attempt = AttemptRecord(
                    timestamp=time.time(),
                    status="failed",
                    model=self.model,
                    error_type=error_type,
                    error_message=error_msg[:500],  # Truncate long error messages
                    error_output_path=error_output_path
                )
                self.tracker.record_attempt(part.id, attempt)

            logger.warning(
                f"Failed to process {part.id}: {e} "
                f"(retries left: {self.remaining_retries})"
            )

            # Check if we can split - if not, fail immediately
            # (no point retrying since generate_with_validation already tried all models)
            if part.token_count < self.min_tokens_to_split:
                raise Exception(
                    f"Processing failed for {part.id} and content too small to split "
                    f"({part.token_count} < {self.min_tokens_to_split} tokens)"
                ) from e

            if self.remaining_retries <= 0:
                raise Exception(
                    f"All retries exhausted for {part.id}"
                ) from e

            # Split and process children
            logger.info(
                f"Splitting {part.id} ({part.token_count} tokens) "
                f"into smaller parts"
            )

            result = self._split_and_process(part, process_fn)
            return result

    def _split_and_process(
        self,
        part: NestedPart,
        process_fn: Callable[[str, str], str]
    ) -> str:
        """
        Split a part and process its children.

        Args:
            part: The part to split
            process_fn: Processing function (takes content and part_id)

        Returns:
            Combined result from all children
        """
        # Calculate target size (half of current)
        target_tokens = part.token_count // 2

        # Split content
        try:
            parts_content = self.splitter.split(part.content, target_tokens)
        except Exception as e:
            logger.error(f"Failed to split {part.id}: {e}")
            raise

        if len(parts_content) <= 1:
            logger.warning(
                f"Splitter returned only {len(parts_content)} part(s) for {part.id}, "
                f"cannot split further"
            )
            raise Exception(f"Cannot split {part.id} further")

        self.stats["splits_performed"] += 1

        # Record split event
        if self.tracker:
            split_record = SplitRecord(
                timestamp=time.time(),
                split_points=[],  # Not tracking exact positions for nested splits
                total_tokens=part.token_count,
                part_count=len(parts_content),
                method="nested_split",
                reason="retry_failure"
            )
            self.tracker.record_split(part.id, split_record)

        # Create child NestedParts
        children = []
        for i, content in enumerate(parts_content, 1):
            child_id = f"{part.id}.part{i}"
            child_sort_key = part.sort_key + [i]
            child_tokens = len(tokenizer.encode(content))

            child = NestedPart(
                id=child_id,
                sort_key=child_sort_key,
                content=content,
                token_count=child_tokens
            )
            children.append(child)

            logger.debug(
                f"Created {child_id} with {child_tokens} tokens "
                f"(depth: {child.depth})"
            )

        part.children = children

        # Process children in parallel (no dependencies between them)
        results = self._process_children_parallel(children, process_fn)

        # Combine results in order
        combined = "".join(results)
        part.result = combined

        logger.info(
            f"Combined {len(children)} parts for {part.id} "
            f"(total: {len(combined)} chars)"
        )

        return combined

    def _process_children_parallel(
        self,
        children: List[NestedPart],
        process_fn: Callable[[str, str], str]
    ) -> List[str]:
        """
        Process children in parallel and return results in order.

        Args:
            children: List of child NestedParts
            process_fn: Processing function (takes content and part_id)

        Returns:
            List of results in order
        """
        if len(children) == 1:
            # Single child, process directly
            return [self.process_with_splitting(children[0], process_fn)]

        results = [None] * len(children)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(children))) as executor:
            # Submit all children
            future_to_idx = {}
            for i, child in enumerate(children):
                future = executor.submit(
                    self.process_with_splitting,
                    child,
                    process_fn
                )
                future_to_idx[future] = i

            # Collect results
            for future in as_completed(future_to_idx.keys()):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Child {children[idx].id} failed: {e}")
                    raise

        return results

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            **self.stats,
            "retries_used": self.total_retries - self.remaining_retries,
            "retries_remaining": self.remaining_retries
        }

    def reset(self):
        """Reset processor state for reuse."""
        self.remaining_retries = self.total_retries
        self.stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "splits_performed": 0,
            "max_depth_reached": 0
        }


def create_root_part(
    unit_id: str,
    content: str,
    part_index: Optional[int] = None
) -> NestedPart:
    """
    Create a root NestedPart from a WorkUnit.

    Args:
        unit_id: The unit ID (e.g., "chapter_5" or "chapter_5.part3")
        content: The content to process
        part_index: Part index if this is a part (e.g., 3)

    Returns:
        NestedPart ready for processing
    """
    # Build sort key
    if part_index is not None:
        sort_key = [part_index]
    else:
        # Extract number from unit_id if possible
        import re
        match = re.search(r'(\d+)$', unit_id)
        if match:
            sort_key = [int(match.group(1))]
        else:
            sort_key = [0]

    return NestedPart(
        id=unit_id,
        sort_key=sort_key,
        content=content,
        token_count=len(tokenizer.encode(content))
    )
