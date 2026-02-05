"""
Base processor class for markdown transformations.

This module provides the abstract base class for all markdown processors,
handling common functionality like progress tracking, retry logic, and
concurrent processing.
"""

import json
import time
import threading
import tiktoken
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from pdf2epub.utils.llm_client import LLMClient
from .utils.content_splitter import split_content, get_splitter
from .utils.split_manager import SplitManager
from .utils.work_unit import WorkUnit, WorkUnitDiscovery
from .utils.nested_processor import NestedPartProcessor, create_root_part
from .validation_strategy import ValidationStrategy
from .tracker import ProcessingTracker, AttemptRecord, ErrorType
from ..chapter_identity import ChapterIdentity

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


class BaseMarkdownProcessor(ABC):
    """Abstract base class for markdown processors."""

    def __init_subclass__(cls, **kwargs):
        """
        Prevent subclasses from overriding process_unit.

        This enforces the Template Method pattern - subclasses must implement
        build_prompt() instead of process_unit() to ensure validation and
        error handling are never bypassed.
        """
        super().__init_subclass__(**kwargs)
        if 'process_unit' in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} should not override process_unit. "
                f"Implement build_prompt() instead."
            )

    def __init__(
        self,
        config: Dict,
        book_title: str,
        input_dir: str,
        output_dir: str,
        max_workers: int = 4,
        resume: bool = False,
        use_longest_on_failure: bool = False
    ):
        """
        Initialize the base processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            input_dir: Input directory name (e.g., "polished_markdown")
            output_dir: Output directory name (e.g., "translated")
            max_workers: Maximum number of concurrent workers
            resume: Whether to resume from previous progress
            use_longest_on_failure: If True, use longest response when all attempts fail validation
        """
        self.config = config
        self.book_title = book_title
        # Use config value if default was passed, otherwise use explicit value
        self.max_workers = max_workers if max_workers != 4 else config.get('max_concurrent_workers', 4)
        self.resume = resume
        self.use_longest_on_failure = use_longest_on_failure

        # Validation and saving behavior (can be overridden by subclasses)
        self.auto_save = True  # If False, subclass must handle saving
        self.validation_mode = "inline"  # "inline" or "batch"

        # Setup directories
        self.input_dir = Path("output") / book_title / input_dir
        self.output_dir = Path("output") / book_title / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup splits directory for storing split input content (before processing)
        self.splits_dir = self.output_dir / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Setup error output directory for debugging failed responses
        self.error_output_dir = Path("output") / book_title / "error_outputs"
        self.error_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validation strategy with error output directory
        validation_config = config.get('validation_strategy', {})
        validation_config['use_longest_on_failure'] = use_longest_on_failure
        self.validation_strategy = ValidationStrategy(validation_config, error_output_dir=self.error_output_dir)

        # Initialize LLM client
        self.llm_client = LLMClient(config)

    # ==================== 子类必须实现的方法 ====================

    @abstractmethod
    def build_prompt(self, content: str, unit_key: str, **context) -> Any:
        """
        Build the prompt for LLM processing.

        Subclasses MUST implement this method instead of process_unit().
        This ensures validation and error handling are never bypassed.

        Args:
            content: Content to process
            unit_key: Unit identifier for tracking (e.g., "chapter_5.part2")
            **context: Processor-specific context including:
                - file_name: Original file name
                - part_idx: Part index (1-based)
                - total_parts: Total number of parts
                - previous_context: Context from previous part (if any)
                - Any other processor-specific data

        Returns:
            Either:
            - str: Simple prompt string
            - List[Dict]: Multi-part content with conversation history
        """
        pass

    @abstractmethod
    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate the processed output.

        Subclasses MUST implement this method.
        Even if no validation is needed, return (True, "No validation needed").

        Args:
            original: Original content
            processed: Processed content
            file_name: Name of the file

        Returns:
            Tuple of (is_valid, reason)
        """
        pass

    @abstractmethod
    def post_process(self, result: str, **context) -> str:
        """
        Post-process the LLM result.

        Subclasses MUST implement this method.
        Even if no post-processing is needed, return result unchanged.

        Args:
            result: Cleaned LLM response
            **context: Same context as build_prompt

        Returns:
            Post-processed result
        """
        pass

    @abstractmethod
    def get_context_for_next_part(self, content: str, result: str, **context) -> Optional[Dict]:
        """
        Get context to inject into the next part's build_prompt.

        Subclasses MUST implement this method.
        Return None if no context injection is needed.

        Args:
            content: Original content of this part
            result: Processed result of this part
            **context: Same context as build_prompt

        Returns:
            None: No context injection needed
            Dict: Context dict to pass to next part's build_prompt as 'previous_context'
        """
        pass

    @abstractmethod
    def get_split_strategy(self) -> str:
        """
        Get splitting strategy for this processor.

        Subclasses MUST implement this method.

        Returns:
            Strategy name: 'markdown', 'compressed', 'academic', 'japanese', 'general'
        """
        pass

    @abstractmethod
    def get_model_configs(self) -> List[Dict]:
        """
        Get the model configurations for this processor.

        Subclasses MUST implement this method.

        Returns:
            List of model configuration dictionaries
        """
        pass

    @abstractmethod
    def get_operation_name(self, file_name: str) -> str:
        """
        Get the operation name for logging.

        Subclasses MUST implement this method.

        Args:
            file_name: Name of the file being processed

        Returns:
            Operation name string
        """
        pass

    # ==================== 子类可选覆盖的方法 ====================

    def clean_response(self, response: str) -> str:
        """
        Clean the LLM response before validation and post-processing.

        Override this method to customize response cleaning (e.g., remove markdown code blocks).
        Default implementation calls clean_markdown_response().

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response
        """
        return self.clean_markdown_response(response)

    def on_validation_failure(self, file_name: str, reason: str, response: str) -> None:
        """
        Called when validation fails.

        Override this method to perform additional actions on validation failure
        (e.g., save truncated output for debugging).

        Args:
            file_name: Name of the file being processed
            reason: Validation failure reason
            response: The failed response (already cleaned)
        """
        pass

    def _is_image_only_content(self, content: str, min_text_chars: int = 100) -> bool:
        """
        Check if content is primarily images with minimal text.

        Used to skip LLM processing for pages that are just images (e.g., genealogy charts,
        full-page illustrations) where there's no meaningful text to polish.

        Args:
            content: Content to check
            min_text_chars: Minimum text characters to consider as having meaningful content

        Returns:
            True if content is image-only (should skip processing)
        """
        import re

        # Remove markdown images: ![alt](src)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', content)

        # Remove HTML images with optional wrapping divs
        text = re.sub(r'(?:<div[^>]*>)?\s*<img\s+[^>]*/?\s*>\s*(?:</div>)?', '', text, flags=re.IGNORECASE)

        # Remove markdown headings (they're just titles, not content)
        text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)

        # Remove whitespace
        text = text.strip()

        # If remaining text is very short, it's image-only
        return len(text) < min_text_chars

    # ==================== 基类实现的核心方法 ====================

    def process_unit(self, content: str, unit_key: str, **context) -> str:
        """
        Process a single content unit with validation and error handling.

        DO NOT override this method in subclasses. The __init_subclass__ check
        will raise TypeError if you try to override it.

        Instead, implement build_prompt(), validate_output(), and post_process().

        Args:
            content: Content to process
            unit_key: Unit identifier for tracking
            **context: Processor-specific context

        Returns:
            Processed content
        """
        if not content.strip():
            return content

        # Skip LLM processing for image-only content (minimal text with just images)
        if self._is_image_only_content(content):
            logger.info(f"Skipping {unit_key}: image-only content")
            return content

        file_name = context.get('file_name', unit_key)
        part_idx = context.get('part_idx')
        total_parts = context.get('total_parts')
        nested_part_id = context.get('nested_part_id')

        # Build file_name_with_part for logging and error reporting
        if part_idx and total_parts and total_parts > 1:
            file_name_with_part = f"{file_name} part {part_idx}/{total_parts}"
        else:
            file_name_with_part = file_name

        # Build operation_name with part info
        base_operation_name = self.get_operation_name(file_name)
        if nested_part_id and nested_part_id != unit_key:
            # Nested split occurred
            nested_suffix = nested_part_id.replace(f"{file_name.replace('.md', '')}.", '')
            operation_name = f"{base_operation_name} ({nested_suffix})"
        elif part_idx and total_parts and total_parts > 1:
            operation_name = f"{base_operation_name} part {part_idx}/{total_parts}"
        else:
            operation_name = base_operation_name

        # 1. Subclass builds the prompt
        prompt = self.build_prompt(content, unit_key, **context)

        # 2. Build validator that calls subclass's validate_output
        # Skip validator if validation_mode is "batch"
        if self.validation_mode == "inline":
            def validator(response: str) -> Tuple[bool, str]:
                cleaned = self.clean_response(response)
                is_valid, reason = self.validate_output(content, cleaned, file_name_with_part)
                if not is_valid:
                    self.on_validation_failure(file_name_with_part, reason, cleaned)
                return is_valid, reason
        else:
            # Batch mode: no inline validation
            validator = None

        # 3. Call LLM with validation (auto-saves error outputs)
        result = self.generate_with_retry(
            multi_part_content=prompt,
            model_configs=self.get_model_configs(),
            validator=validator,
            operation_name=operation_name
        )

        # 4. Clean response and call subclass's post_process
        cleaned = self.clean_response(result)
        return self.post_process(cleaned, **context)

    # === Split-Aware Processing ===

    @property
    def can_split(self) -> bool:
        """
        Whether this processor supports content splitting.

        Override to disable splitting for specific processors.
        """
        return True

    def get_model_output_limit(self, model_name: str) -> int:
        """
        Get the practical output token limit for a specific model.

        This returns the reliable output limit based on practical experience,
        not the theoretical maximum.

        Args:
            model_name: The model identifier (e.g., "gemini-2.5-pro")

        Returns:
            Maximum reliable output tokens for this model
        """
        limits = self.config.get('model_output_limits', {})
        return limits.get(model_name, limits.get('_default', 4000))

    def get_max_tokens_per_part(self) -> int:
        """
        Get maximum tokens per part for this processor.

        Uses the model_output_limits config to determine the limit based on
        the first model in the processor's model configs.

        Override to customize based on processor type or model.
        """
        # Get model configs from subclass
        model_configs = self.get_model_configs()
        if model_configs:
            first_model = model_configs[0].get('model', '')
            return self.get_model_output_limit(first_model)
        return self.config.get('model_output_limits', {}).get('_default', 4000)

    def create_split_manager(self, tracker: ProcessingTracker) -> SplitManager:
        """
        Create a SplitManager for this processor.

        Args:
            tracker: ProcessingTracker instance

        Returns:
            Configured SplitManager
        """
        splitting_config = self.config.get('splitting', {})
        return SplitManager(
            tracker=tracker,
            llm_client=self.llm_client,
            model_configs=self.config.get('split_model_configs'),
            output_dir=self.output_dir,
            default_max_tokens=self.get_max_tokens_per_part(),
            max_resplits=splitting_config.get('max_resplits', 3),
            consecutive_failures_threshold=splitting_config.get('consecutive_failures_threshold', 2)
        )

    def _parse_part_key(self, part_key: str) -> Tuple[int, int]:
        """
        Parse part key to extract part index.

        Args:
            part_key: Unit key like "chapter_5.part2" or "chapter_5"

        Returns:
            Tuple of (part_idx, 0) - total_parts is set later by caller
        """
        if '.part' in part_key:
            # Extract part number from key like "chapter_5.part2"
            import re
            match = re.search(r'\.part(\d+)$', part_key)
            if match:
                return int(match.group(1)), 0
        return 1, 1  # Single part

    def _detect_input_part_files(self, file_name: str) -> List[Path]:
        """
        Detect if there are part files from previous stage.

        Args:
            file_name: The file name (e.g., "chapter_5.md")

        Returns:
            Sorted list of part file paths, empty if none found
        """
        base_name = Path(file_name).stem
        part_files = sorted(self.input_dir.glob(f"{base_name}.part*.md"))
        return part_files

    def _classify_error(self, error: Exception) -> str:
        """
        Classify an error into an ErrorType.

        Args:
            error: The exception

        Returns:
            Error type string
        """
        error_str = str(error).lower()

        if 'truncat' in error_str:
            return ErrorType.TRUNCATION.value
        elif 'rate' in error_str or 'limit' in error_str or '429' in error_str:
            return ErrorType.RATE_LIMIT.value
        elif 'timeout' in error_str:
            return ErrorType.TIMEOUT.value
        elif 'filter' in error_str or 'safety' in error_str or 'content' in error_str:
            return ErrorType.CONTENT_FILTER.value
        elif 'parse' in error_str or 'json' in error_str:
            return ErrorType.PARSE_ERROR.value
        elif 'api' in error_str or 'request' in error_str:
            return ErrorType.API_ERROR.value
        elif 'valid' in error_str:
            return ErrorType.VALIDATION.value
        else:
            return ErrorType.UNKNOWN.value

    def get_inject_context(self) -> bool:
        """
        Get whether to inject context between parts.

        Override in subclasses to customize behavior.
        Default is False (parallel processing).

        Returns:
            True if context should be injected between parts
        """
        return False

    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all markdown files in the input directory.

        Uses the flattened parallel processing architecture.

        Returns:
            Summary statistics
        """
        # Use the new unified processing architecture
        inject_context = self.get_inject_context()
        return self.process_all_units(inject_context=inject_context)

    def process_specific_files(self, file_stems: List[str]) -> Dict[str, Any]:
        """
        Process only specific files by their stem names.

        Args:
            file_stems: List of file stems to process (e.g., ["chapter_1", "chapter_2"])

        Returns:
            Summary statistics
        """
        inject_context = self.get_inject_context()
        return self.process_all_units(inject_context=inject_context, file_filter=set(file_stems))

    def process_all_units(
        self,
        inject_context: bool = False,
        file_filter: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        Process all work units with a flat thread pool.

        This method uses a single thread pool for all work units (files and parts),
        with dependency-aware scheduling for context injection support.

        Args:
            inject_context: If True, parts depend on previous parts for context
            file_filter: If provided, only process units whose file_key is in this set

        Returns:
            Summary statistics
        """
        # Discover all work units
        discovery = WorkUnitDiscovery(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            inject_context=inject_context,
            splits_dir=self.splits_dir
        )
        all_units = discovery.discover_all_units()

        # Apply file filter if provided
        if file_filter:
            all_units = [u for u in all_units if u.file_key in file_filter]
            logger.info(f"Filtered to {len(all_units)} units matching file filter")

        if not all_units:
            logger.error(f"No work units found in {self.input_dir}")
            return {"error": "No work units found"}

        # Initialize unified ProcessingTracker
        tracker_path = self.output_dir / "processing_tracker.json"
        if not hasattr(self, 'processing_tracker') or self.processing_tracker is None:
            self.processing_tracker = ProcessingTracker(tracker_path, self.__class__.__name__)

        # Track completion status
        completed_ids: set = set()
        completed_results: Dict[str, str] = {}  # unit_id -> processed content
        failed_ids: set = set()

        # Check for already completed units (resume support) using ProcessingTracker
        for unit in all_units:
            if self.processing_tracker.is_unit_complete(unit.id):
                completed_ids.add(unit.id)
                # Load existing result if available
                if unit.output_path.exists():
                    completed_results[unit.id] = unit.output_path.read_text(encoding='utf-8')
                else:
                    completed_results[unit.id] = ""
                logger.debug(f"Skipping {unit.id} (already completed)")

        # Get units that need processing
        pending_units = [u for u in all_units if u.id not in completed_ids]

        if not pending_units:
            logger.info("All units already completed")
            return self._create_summary(all_units, completed_ids, failed_ids)

        logger.info(f"Processing {len(pending_units)} work units with {self.max_workers} workers")

        # Thread-safe data structures
        lock = threading.Lock()

        # Proactive split large single files before processing
        all_units = self._proactive_split_units(all_units)
        pending_units = [u for u in all_units if u.id not in completed_ids]

        # Process with flat pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[Any, WorkUnit] = {}

            # Submit initially ready units (no dependencies or dependencies completed)
            ready_units = discovery.get_ready_units(pending_units, completed_ids)
            for unit in ready_units:
                future = executor.submit(
                    self._process_work_unit,
                    unit,
                    completed_results if inject_context else None,
                    all_units,
                    inject_context
                )
                futures[future] = unit
                logger.debug(f"Submitted {unit.id} (ready)")

            # Process completed futures and submit newly ready units
            while futures:
                # Wait for at least one future to complete
                done_futures = []
                for future in as_completed(futures.keys()):
                    done_futures.append(future)
                    break  # Process one at a time to check for newly ready units

                for future in done_futures:
                    unit = futures.pop(future)

                    try:
                        result = future.result()

                        with lock:
                            completed_ids.add(unit.id)
                            completed_results[unit.id] = result

                            # Record progress
                            self._record_unit_completion(unit, True, result)

                        logger.info(f"Completed {unit.id}")

                        # Check for newly ready units
                        with lock:
                            newly_ready = []
                            for pending in pending_units:
                                if pending.id not in completed_ids and pending.id not in failed_ids:
                                    if pending.id not in [futures[f].id for f in futures]:
                                        if all(dep in completed_ids for dep in pending.dependencies):
                                            newly_ready.append(pending)

                            for new_unit in newly_ready:
                                new_future = executor.submit(
                                    self._process_work_unit,
                                    new_unit,
                                    completed_results if inject_context else None,
                                    all_units,
                                    inject_context
                                )
                                futures[new_future] = new_unit
                                logger.debug(f"Submitted {new_unit.id} (dependencies satisfied)")

                    except Exception as e:
                        with lock:
                            failed_ids.add(unit.id)
                        self._record_unit_completion(unit, False)
                        logger.error(f"Failed to process {unit.id}: {e}")

        # Batch validation (if enabled)
        if self.validation_mode == "batch" and not self.auto_save:
            # Give subclass a chance to validate and save results
            completed_results, failed_ids = self._batch_validate_and_save(
                all_units, completed_results, failed_ids
            )

        # Aggregate results for multi-part files (only if auto_save is enabled)
        if self.auto_save:
            self._aggregate_file_results(all_units, completed_results)

        return self._create_summary(all_units, completed_ids, failed_ids)

    def _process_work_unit(
        self,
        unit: WorkUnit,
        completed_results: Optional[Dict[str, str]] = None,
        all_units: Optional[List[WorkUnit]] = None,
        inject_context: bool = False
    ) -> str:
        """
        Process a single work unit with automatic splitting on failure.

        Uses NestedPartProcessor for recursive splitting with shared retry budget.

        Args:
            unit: The work unit to process
            completed_results: Dict of completed unit results (for context injection)
            all_units: All work units (unused, kept for compatibility)
            inject_context: Whether to inject context (unused for nested parts)

        Returns:
            Processed content
        """
        logger.info(f"Processing {unit.id}")

        # Build context from previous part if needed
        previous_context = None
        if completed_results and unit.dependencies:
            dep_id = unit.dependencies[-1]
            if dep_id in completed_results:
                previous_context = {
                    "processed": completed_results[dep_id]
                }

        # Get splitting config
        splitting_config = self.config.get('splitting', {})
        total_retries = splitting_config.get('total_retries', 5)
        min_tokens_to_split = splitting_config.get('min_tokens_to_split', 500)

        # Get primary model name for tracking
        model_configs = self.get_model_configs()
        primary_model = model_configs[0].get('model', 'unknown') if model_configs else 'unknown'

        # Create processor with shared retry budget
        processor = NestedPartProcessor(
            splitter=get_splitter("markdown"),
            total_retries=total_retries,
            min_tokens_to_split=min_tokens_to_split,
            max_workers=min(4, self.max_workers),
            tracker=self.processing_tracker,
            model=primary_model
        )

        # Create root nested part
        root = create_root_part(
            unit_id=unit.id,
            content=unit.content,
            part_index=unit.part_index
        )

        # Define the actual processing function
        def do_process(content: str, nested_part_id: str) -> str:
            return self.process_unit(
                content=content,
                unit_key=unit.id,
                file_name=unit.input_path.name,
                part_idx=unit.part_index,
                total_parts=unit.total_parts,
                original_content=unit.content,
                previous_context=previous_context,
                nested_part_id=nested_part_id
            )

        # Process with automatic splitting on failure
        result = processor.process_with_splitting(root, do_process)

        # Log stats
        stats = processor.get_stats()
        if stats['splits_performed'] > 0:
            logger.info(
                f"Processed {unit.id} with {stats['splits_performed']} splits, "
                f"max depth {stats['max_depth_reached']}, "
                f"retries used {stats['retries_used']}/{total_retries}"
            )

        # Save the result (only if auto_save is enabled)
        if self.auto_save:
            with open(unit.output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        else:
            # Batch validation mode: result will be saved after validation
            # Ensure parent directory exists for later saving
            unit.output_path.parent.mkdir(parents=True, exist_ok=True)

        return result

    def _proactive_split_units(self, units: List[WorkUnit]) -> List[WorkUnit]:
        """
        Proactively split large single-file units before processing.

        Only splits files without existing parts that exceed the token threshold.

        Args:
            units: List of work units from discovery

        Returns:
            Updated list with large files split into parts
        """
        max_tokens = self.get_max_tokens_per_part()
        result = []

        for unit in units:
            # Only check single-file units (no parts yet)
            if unit.total_parts == 1 and unit.part_index is None:
                if unit.token_count > max_tokens:
                    logger.info(
                        f"Proactively splitting {unit.id}: "
                        f"{unit.token_count:,} tokens > {max_tokens:,} limit"
                    )

                    # Split using the processor's configured strategy
                    strategy = self.get_split_strategy()
                    splitter = get_splitter(strategy)
                    parts = splitter.split(unit.content, max_tokens)

                    if len(parts) > 1:
                        # Record split in tracker
                        if self.processing_tracker:
                            from .tracker import SplitRecord
                            split_record = SplitRecord(
                                timestamp=time.time(),
                                split_points=[],
                                total_tokens=unit.token_count,
                                part_count=len(parts),
                                method="proactive_markdown",
                                reason="exceeds_token_limit"
                            )
                            self.processing_tracker.record_split(unit.file_key, split_record)

                        # Create new units for each part and write to splits directory
                        for i, part_content in enumerate(parts, 1):
                            # Write part content to splits directory for resume support
                            part_input_path = self.splits_dir / f"{unit.file_key}.part{i}.md"
                            with open(part_input_path, 'w', encoding='utf-8') as f:
                                f.write(part_content)
                            logger.debug(f"Wrote split part to {part_input_path}")

                            new_unit = WorkUnit(
                                id=f"{unit.file_key}.part{i}",
                                file_key=unit.file_key,
                                part_index=i,
                                total_parts=len(parts),
                                content=part_content,
                                input_path=part_input_path,
                                output_path=self.output_dir / f"{unit.file_key}.part{i}.md",
                                dependencies=[],  # No context injection for proactive splits
                                priority=i
                            )
                            result.append(new_unit)

                        logger.info(f"Split {unit.id} into {len(parts)} parts (written to input dir)")
                        continue

            result.append(unit)

        return result

    def _is_unit_completed(self, unit: WorkUnit) -> bool:
        """Check if a work unit is already completed."""
        if self.processing_tracker:
            return self.processing_tracker.is_unit_complete(unit.id)
        return False

    def _record_unit_completion(self, unit: WorkUnit, success: bool, result: str = None):
        """Record completion of a work unit using unified ProcessingTracker."""
        # Get primary model name
        model_configs = self.get_model_configs()
        primary_model = model_configs[0].get('model', 'unknown') if model_configs else 'unknown'

        # Create attempt record
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed" if success else "failed",
            model=primary_model,
            input_tokens=unit.token_count,
            output_tokens=len(tokenizer.encode(result)) if result else 0
        )

        # Record in ProcessingTracker
        self.processing_tracker.record_attempt(unit.id, attempt)

    def _aggregate_file_results(
        self,
        all_units: List[WorkUnit],
        completed_results: Dict[str, str]
    ):
        """
        Aggregate results for multi-part files.

        Creates combined output files from individual part results.
        """
        # Group units by file
        discovery = WorkUnitDiscovery(self.input_dir, self.output_dir, splits_dir=self.splits_dir)
        file_groups = discovery.group_units_by_file(all_units)

        for file_key, units in file_groups.items():
            if len(units) <= 1:
                continue  # Single file, no aggregation needed

            # Check if all parts completed
            if not all(u.id in completed_results for u in units):
                logger.warning(f"Not all parts completed for {file_key}, skipping aggregation")
                continue

            # Aggregate in order
            parts = [completed_results[u.id] for u in units]
            aggregated = "\n\n".join(p for p in parts if p)

            # Save combined file
            combined_path = self.output_dir / f"{file_key}.md"
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write(aggregated)

            logger.info(f"Aggregated {len(units)} parts into {combined_path.name}")

    def _batch_validate_and_save(
        self,
        all_units: List[WorkUnit],
        completed_results: Dict[str, str],
        failed_ids: set
    ) -> Tuple[Dict[str, str], set]:
        """
        Extension point for batch validation and saving.

        Subclasses can override this to implement custom batch validation logic.
        Default implementation just saves all results and returns them as-is.

        Args:
            all_units: All work units
            completed_results: Dict of unit_id -> processed content
            failed_ids: Set of failed unit IDs

        Returns:
            Tuple of (validated_results, updated_failed_ids)
        """
        # Default: save all completed results
        for unit in all_units:
            if unit.id in completed_results:
                with open(unit.output_path, 'w', encoding='utf-8') as f:
                    f.write(completed_results[unit.id])

        return completed_results, failed_ids

    def _create_summary(
        self,
        all_units: List[WorkUnit],
        completed_ids: set,
        failed_ids: set
    ) -> Dict[str, Any]:
        """Create processing summary using unified ProcessingTracker."""
        total = len(all_units)
        completed = len(completed_ids)
        failed = len(failed_ids)

        # Get detailed stats from ProcessingTracker
        tracker_summary = self.processing_tracker.progress.get("summary", {})

        summary = {
            "total_units": total,
            "completed": completed,
            "failed": failed,
            "failed_ids": failed_ids,  # Include failed IDs for retry logic
            "success_rate": completed / total if total > 0 else 0,
            # Include tracker stats
            "total_attempts": tracker_summary.get("total_attempts", 0),
            "total_retries": tracker_summary.get("total_retries", 0),
            "total_input_tokens": tracker_summary.get("total_input_tokens", 0),
            "total_output_tokens": tracker_summary.get("total_output_tokens", 0),
            "errors_by_type": tracker_summary.get("errors_by_type", {}),
            "models_used": tracker_summary.get("models_used", {})
        }

        logger.info(f"\n=== {self.__class__.__name__} Summary ===")
        logger.info(f"Completed: {completed}/{total} units")

        if summary["total_retries"] > 0:
            logger.info(f"Total retries: {summary['total_retries']}")

        if summary["total_output_tokens"] > 0:
            logger.info(f"Total tokens: {summary['total_input_tokens']} in / {summary['total_output_tokens']} out")

        if failed > 0:
            logger.warning(f"Failed units: {', '.join(failed_ids)}")
        elif completed == total:
            logger.success("All units processed successfully!")

        return summary

    def generate_with_retry(
        self,
        multi_part_content: List[Dict],
        model_configs: List[Dict],
        validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
        operation_name: str = "Process"
    ) -> str:
        """
        Generate content with validation and retry logic.

        This is a convenience wrapper around LLMClient.generate_with_validation
        that handles the common pattern of creating a thread-local ValidationStrategy.

        Args:
            multi_part_content: Content to send to LLM
            model_configs: List of model configurations to try
            validator: Optional validation function
            operation_name: Name of operation for logging

        Returns:
            Generated and cleaned content
        """
        # Create thread-local validation strategy for thread safety
        validation_config = self.config.get('validation_strategy', {})
        validation_config['use_longest_on_failure'] = self.use_longest_on_failure
        thread_local_strategy = ValidationStrategy(validation_config, error_output_dir=self.error_output_dir)

        result = self.llm_client.generate_with_validation(
            prompt=multi_part_content,
            model_configs=model_configs,
            validator=validator,
            validation_strategy=thread_local_strategy,
            operation_name=operation_name
        )

        return self.clean_markdown_response(result)

    def save_part_file(self, file_name: str, part_idx: int, content: str) -> Path:
        """
        Save a part file to the output directory.

        Args:
            file_name: Base file name
            part_idx: Part index
            content: Content to save

        Returns:
            Path to saved file
        """
        base_name = Path(file_name).stem
        part_file = self.output_dir / f"{base_name}.part{part_idx}.md"

        with open(part_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.debug(f"Saved part file: {part_file.name}")
        return part_file

    def clean_markdown_response(self, content: str) -> str:
        """
        Clean up markdown response from LLM.

        Args:
            content: Raw response from LLM

        Returns:
            Cleaned markdown content
        """
        lines = content.strip().split('\n')
        
        # Look for code block markers in first 3 non-empty lines
        non_empty_count = 0
        code_block_start = -1
        
        for i, line in enumerate(lines):
            if line.strip():  # Non-empty line
                non_empty_count += 1
                # Check if this line is a code block marker
                if line.strip() in ['```markdown', '```'] or line.strip().startswith('```'):
                    code_block_start = i + 1  # Start from the line after the marker
                    break
                if non_empty_count >= 3:
                    break
        
        # If we found a code block marker, remove everything before and including it
        if code_block_start > 0:
            lines = lines[code_block_start:]
        
        # Rejoin the content
        content = '\n'.join(lines)
        
        # Also handle case where ``` appears at the end
        if content.strip().endswith('```'):
            lines = content.strip().split('\n')
            if lines[-1].strip() == '```':
                lines = lines[:-1]
                content = '\n'.join(lines)
        
        return content.strip()
