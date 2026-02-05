"""
Batch Polish Processor using Gemini Batch API.

Provides asynchronous, high-throughput polish processing at 50% cost reduction
compared to real-time inference.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
import tiktoken

from .polisher import convert_html_images_to_markdown, PolishProcessor
from .prompts import create_polish_prompt
from .utils.split_manager import SplitManager
from .utils.truncation import NGramTruncationDetector
from .utils.image_restore import restore_lost_images
from .tracker import ProcessingTracker, AttemptRecord
from ..utils.batch_utils import (
    GeminiBatchClient,
    BatchRequest,
    BatchResponse,
    BatchJobState,
    BatchJobInfo,
    BATCH_DEFAULTS
)
# Note: Batch mode uses n-gram screening + agent-based verification (no direct LLMClient dependency)
from ..chapter_identity import ChapterIdentity

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class BatchPolishState:
    """Persistent state for batch polish processing."""
    active_job_name: Optional[str] = None
    active_job_requests: List[str] = field(default_factory=list)  # List of request keys in current job
    pending_files: List[str] = field(default_factory=list)
    retry_count: int = 0
    failed_keys: List[str] = field(default_factory=list)
    # Track completed keys across all rounds (for resume support)
    completed_keys: List[str] = field(default_factory=list)
    # Track which keys are being processed in current job (for partial result handling)
    processing_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "active_job_name": self.active_job_name,
            "active_job_requests": self.active_job_requests,
            "pending_files": self.pending_files,
            "retry_count": self.retry_count,
            "failed_keys": self.failed_keys,
            "completed_keys": self.completed_keys,
            "processing_keys": self.processing_keys,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BatchPolishState':
        return cls(
            active_job_name=data.get("active_job_name"),
            active_job_requests=data.get("active_job_requests", []),
            pending_files=data.get("pending_files", []),
            retry_count=data.get("retry_count", 0),
            failed_keys=data.get("failed_keys", []),
            completed_keys=data.get("completed_keys", []),
            processing_keys=data.get("processing_keys", []),
        )


class BatchPolishProcessor:
    """
    Batch processor for polishing OCR-extracted markdown content.

    Uses Gemini Batch API for 50% cost reduction on large-scale processing.
    """

    def __init__(
        self,
        config: Dict,
        book_title: str,
        book_structure: Optional[Dict] = None,
        content_type: str = "auto",
        max_retries: int = 1,
        poll_interval: int = 60,
        resume: bool = False
    ):
        """
        Initialize the batch polish processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            book_structure: Optional book structure from breakdown phase
            content_type: Type of content ("academic", "japanese", "general", "auto")
            max_retries: Maximum retries for validation failures (default: 1)
            poll_interval: Seconds between status polls
            resume: Whether to resume from previous progress
        """
        self.config = config
        self.book_title = book_title
        self.book_structure = book_structure or {}
        self.content_type = content_type
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.resume = resume

        # Track all attempts for failed files (for longest fallback)
        self.attempt_history = {}

        # Setup directories
        self.input_dir = Path("output") / book_title / "ocr_markdown"
        self.output_dir = Path("output") / book_title / "polished_markdown"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State file for persistence
        self.state_file = self.output_dir / "batch_state.json"

        # Initialize processing tracker
        tracker_path = self.output_dir / "processing_tracker.json"
        self.tracker = ProcessingTracker(tracker_path, "BatchPolishProcessor")

        # Initialize split manager
        splitting_config = config.get('splitting', {})
        self.split_manager = SplitManager(
            tracker=self.tracker,
            output_dir=self.output_dir,
            default_max_tokens=self._get_max_tokens_per_part(),
            max_resplits=splitting_config.get('max_resplits', 3),
            consecutive_failures_threshold=splitting_config.get('consecutive_failures_threshold', 2)
        )

        # Verification configuration
        verification_config = config.get('verification', {})
        self.ngram_threshold = verification_config.get('ngram_threshold', 0.6)

        # N-gram detector for fast screening (filters obvious OK cases)
        self.ngram_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=self.ngram_threshold,
            allow_deduplication=True
        )

        logger.info("Using agent-based verification for batch polish")

        # Initialize batch client
        batch_config = config.get('batch', {})
        credentials = config.get('credentials', {}).get('providers', {})

        # Use batch-specific provider or fall back to gemini
        batch_provider = batch_config.get('provider', BATCH_DEFAULTS['provider'])
        provider_config = credentials.get(batch_provider, credentials.get('gemini', {}))

        api_key = provider_config.get('api_key')
        if not api_key:
            raise ValueError(
                f"No API key found for batch provider '{batch_provider}'. "
                "Please configure credentials.providers.gemini.api_key in config.yaml"
            )

        # Default base_url for batch API
        base_url = (
            provider_config.get('base_url') or
            batch_config.get('base_url') or
            BATCH_DEFAULTS['base_url']
        )

        if not base_url:
            raise ValueError(
                f"No base_url configured for batch provider '{batch_provider}'. "
                "Please configure credentials.providers.gemini.base_url in config.yaml"
            )

        logger.info(f"Using Batch API endpoint: {base_url}")

        # Model for batch polish: batch.polish.model > batch.model (legacy) > default
        polish_config = batch_config.get('polish', {})
        batch_model = (
            polish_config.get('model') or
            batch_config.get('model') or
            BATCH_DEFAULTS['polish']['model']
        )
        logger.info(f"Using batch polish model: {batch_model}")

        self.batch_client = GeminiBatchClient(
            api_key=api_key,
            model=batch_model,
            poll_interval=poll_interval,
            base_url=base_url
        )

        # Check for global footnotes
        self.use_global_footnotes = self._has_notes_chapter()

        # Auto-detect content type from book structure
        if content_type == "auto" and self.book_structure:
            self.content_type = self._detect_content_type()
        else:
            self.content_type = content_type

        # Content cache for validation
        self._content_cache: Dict[str, str] = {}

    def _get_max_tokens_per_part(self) -> int:
        """Get max tokens per part from config."""
        batch_config = self.config.get('batch', {})
        model = batch_config.get('model', 'gemini-2.5-flash')
        limits = self.config.get('model_output_limits', {})
        return limits.get(model, limits.get('_default', 4000))

    def _has_notes_chapter(self) -> bool:
        """Check if book structure contains a notes chapter."""
        for chapter in self.book_structure.get('chapters', []):
            if chapter.get('type') == 'notes':
                return True
        return False

    def _detect_content_type(self) -> str:
        """Auto-detect content type from book structure."""
        language = self.book_structure.get('language', '').lower()
        is_vertical = self.book_structure.get('is_vertical_text', False)
        has_footnotes = self.book_structure.get('has_footnotes', False)

        if language == "japanese" and is_vertical:
            logger.info("Auto-detected content type: japanese (vertical Japanese text)")
            return "japanese"
        elif has_footnotes:
            logger.info("Auto-detected content type: academic (has footnotes)")
            return "academic"
        else:
            logger.info("Auto-detected content type: general")
            return "general"

    def _get_chapter_info(self, file_name: str) -> Dict:
        """Get chapter information from book structure."""
        identity = ChapterIdentity.parse(file_name)
        if not identity or not identity.number:
            return {}

        index_path = identity.index_path
        chapters = self.book_structure.get('chapters', [])
        current_level = chapters

        for idx in index_path:
            array_idx = idx - 1
            if 0 <= array_idx < len(current_level):
                current = current_level[array_idx]
                current_level = current.get('children', [])
            else:
                return {}

        return current

    def _load_state(self) -> BatchPolishState:
        """Load batch state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return BatchPolishState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load batch state: {e}")
        return BatchPolishState()

    def _save_state(self, state: BatchPolishState):
        """Save batch state to file."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    def _get_pending_files(self) -> List[Path]:
        """Get list of files that need processing."""
        pending = []

        # Find all input files
        input_files = sorted(self.input_dir.glob("*.md"))

        for input_file in input_files:
            base_name = input_file.stem

            # Check if already completed in tracker
            if self.tracker.is_unit_complete(base_name):
                logger.debug(f"Skipping {base_name} (already completed)")
                continue

            # Check if any parts are completed
            plan = self.tracker.get_processing_plan(base_name, self.output_dir)
            if plan.action == "skip":
                logger.debug(f"Skipping {base_name} ({plan.reason})")
                continue

            pending.append(input_file)

        return pending

    def _build_batch_requests(
        self,
        files: List[Path],
        completed_keys: Optional[List[str]] = None
    ) -> Tuple[List[BatchRequest], Dict[str, Dict]]:
        """
        Build batch requests for all pending files.

        Args:
            files: List of input files to process
            completed_keys: List of keys already completed (for resume support)

        Returns:
            Tuple of (requests, metadata_map)
            metadata_map: key -> {file_stem, part_idx, total_parts, original_content}
        """
        requests = []
        metadata_map = {}
        completed_set = set(completed_keys or [])

        for file_path in files:
            file_stem = file_path.stem
            content = file_path.read_text(encoding='utf-8')

            if not content.strip():
                logger.debug(f"Skipping empty file: {file_stem}")
                continue

            # Preprocess: convert HTML images to markdown
            content = convert_html_images_to_markdown(content)

            # Check if splitting is needed
            max_tokens = self._get_max_tokens_per_part()
            split_result = self.split_manager.get_or_create_split(
                base_key=file_stem,
                content=content,
                max_tokens=max_tokens,
                strategy=self.content_type
            )

            # Build request for each part
            for i, (part_content, part_info) in enumerate(
                zip(split_result.parts, split_result.part_infos), 1
            ):
                # Check if this part is already completed
                if len(split_result.parts) > 1:
                    part_key = f"{file_stem}.part{i}"
                else:
                    part_key = file_stem

                # Skip if already completed in batch state (from previous run)
                if part_key in completed_set:
                    logger.debug(f"Skipping {part_key} (completed in previous batch run)")
                    continue

                # Skip if already completed in tracker
                if self.tracker.is_unit_complete(part_key):
                    logger.debug(f"Skipping {part_key} (already completed in tracker)")
                    continue

                # Build the prompt
                prompt = self._build_prompt(
                    file_stem=file_stem,
                    part_idx=i,
                    total_parts=len(split_result.parts),
                    content=part_content
                )

                # Create batch request
                request = BatchRequest(
                    key=part_key,
                    contents=[
                        {"parts": [{"text": prompt}], "role": "user"},
                    ]
                )
                requests.append(request)

                # Store metadata for validation
                metadata_map[part_key] = {
                    "file_stem": file_stem,
                    "part_idx": i,
                    "total_parts": len(split_result.parts),
                    "original_content": part_content
                }

                # Cache content for validation
                self._content_cache[part_key] = part_content

        logger.info(f"Built {len(requests)} batch requests from {len(files)} files")
        return requests, metadata_map

    def _build_prompt(
        self,
        file_stem: str,
        part_idx: int,
        total_parts: int,
        content: str
    ) -> str:
        """Build the polish prompt for a single part."""
        # Get chapter info
        chapter_info = self._get_chapter_info(file_stem)
        is_notes_chapter = chapter_info.get('type') == 'notes'

        # Chapter name
        chapter_name = chapter_info.get('title') or file_stem

        # Detect notes chapter by content
        if not is_notes_chapter:
            content_start = content[:500].lower()
            if any(marker in content_start for marker in [
                '# notes\n', '# notes \n', '## notes\n',
                '# references\n', '# bibliography\n'
            ]):
                import re
                if '[^' not in content and re.search(r'^\d+\.\s+\w', content, re.MULTILINE):
                    is_notes_chapter = True
                    logger.info(f"Detected {chapter_name} as Notes chapter based on content")

        # Build prompt (no previous_part_context in batch mode)
        prompt = create_polish_prompt(
            chapter_name=chapter_name,
            book_title=self.book_title,
            part_idx=part_idx,
            total_parts=total_parts,
            content=content,
            content_type=self.content_type,
            use_global_footnotes=self.use_global_footnotes,
            is_notes_chapter=is_notes_chapter,
            previous_part_context=None  # Batch mode doesn't support context
        )

        # Append the actual content
        prompt += f"\n\n{content}"

        return prompt

    def _agent_verify_batch(
        self,
        suspicious_responses: List[Dict]
    ) -> Dict[str, bool]:
        """
        Use agent to verify suspicious responses.

        Args:
            suspicious_responses: List of dicts with keys: key, original, processed, ngram_stats

        Returns:
            Dict mapping key to is_valid boolean
        """
        from pdf2epub.processors.utils import VerificationFile, verify_batch

        logger.info(f"Using agent to verify {len(suspicious_responses)} suspicious files")

        # Prepare verification files
        files = {}
        for item in suspicious_responses:
            files[item['key']] = VerificationFile(
                key=item['key'],
                original=item['original'],
                processed=item['processed'],
                metadata={'ngram_stats': item['ngram_stats']}
            )

        # Run agent verification
        try:
            results = verify_batch(files, task_type="polish")

            # Convert to dict
            validation_results = {}
            for result in results:
                is_valid = result.status == "complete"
                validation_results[result.file_key] = is_valid

                if is_valid:
                    logger.info(f"{result.file_key}: Agent verified as complete - {result.reason}")
                else:
                    logger.warning(f"{result.file_key}: Agent confirmed truncation - {result.reason}")

            return validation_results

        except Exception as e:
            logger.error(f"Agent verification failed: {e}")
            logger.warning("Falling back to n-gram results")
            # Fallback: assume all suspicious files are invalid
            return {item['key']: False for item in suspicious_responses}

    def _post_process_result(
        self,
        key: str,
        response_text: str,
        metadata: Dict
    ) -> str:
        """
        Post-process a validated result.

        Args:
            key: Request key
            response_text: The validated response
            metadata: Request metadata

        Returns:
            Post-processed result
        """
        result = response_text.strip()

        # Clean up markdown code blocks
        result = self._clean_markdown_response(result)

        # Apply markdown post-processing
        result = self._post_process_markdown(result)

        # Restore lost images for single-part files
        if metadata.get("total_parts", 1) == 1:
            original = metadata.get("original_content", "")
            result = restore_lost_images(original, result)

        return result

    def _clean_markdown_response(self, content: str) -> str:
        """Clean up markdown response from LLM."""
        lines = content.strip().split('\n')

        # Look for code block markers in first 3 non-empty lines
        non_empty_count = 0
        code_block_start = -1

        for i, line in enumerate(lines):
            if line.strip():
                non_empty_count += 1
                if line.strip() in ['```markdown', '```'] or line.strip().startswith('```'):
                    code_block_start = i + 1
                    break
                if non_empty_count >= 3:
                    break

        if code_block_start > 0:
            lines = lines[code_block_start:]

        content = '\n'.join(lines)

        # Handle trailing ```
        if content.strip().endswith('```'):
            lines = content.strip().split('\n')
            if lines[-1].strip() == '```':
                lines = lines[:-1]
                content = '\n'.join(lines)

        return content.strip()

    def _post_process_markdown(self, markdown: str) -> str:
        """Post-process the polished markdown."""
        import re

        markdown = markdown.strip()

        # Remove trailing empty "Notes" header
        markdown = re.sub(r'\n#{1,6}\s+Notes\s*$', '', markdown, flags=re.IGNORECASE)
        markdown = markdown.strip()

        # Fix common markdown issues
        markdown = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(#{1,6} [^\n]+)\n([^\n#])', r'\1\n\n\2', markdown)
        markdown = re.sub(r'\n{4,}', '\n\n', markdown)
        markdown = re.sub(r'([^\n])\n(!\[)', r'\1\n\n\2', markdown)
        markdown = re.sub(r'(!\[[^\]]*\]\([^\)]*\))\n([^\n])', r'\1\n\n\2', markdown)

        return markdown

    def _save_result(self, key: str, content: str, metadata: Dict):
        """Save a processed result to file."""
        file_stem = metadata["file_stem"]
        part_idx = metadata["part_idx"]
        total_parts = metadata["total_parts"]

        if total_parts > 1:
            output_path = self.output_dir / f"{file_stem}.part{part_idx}.md"
        else:
            output_path = self.output_dir / f"{file_stem}.md"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Saved result: {output_path.name}")

        # Record in tracker
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed",
            model=self.config.get('batch', {}).get('model', 'gemini-2.5-flash'),
            output_tokens=len(tokenizer.encode(content))
        )
        self.tracker.record_attempt(key, attempt)

    def _aggregate_parts(self, file_stem: str):
        """Aggregate parts into a single file if all parts are complete."""
        current_split = self.tracker.get_current_split(file_stem)

        if not current_split:
            return  # Single file, no aggregation needed

        part_count = current_split["part_count"]

        # Check if all parts are complete
        all_complete = True
        parts = []
        for i in range(1, part_count + 1):
            part_key = f"{file_stem}.part{i}"
            if not self.tracker.is_unit_complete(part_key):
                all_complete = False
                break
            part_path = self.output_dir / f"{file_stem}.part{i}.md"
            if part_path.exists():
                parts.append(part_path.read_text(encoding='utf-8'))

        if all_complete and parts:
            # Aggregate
            combined = "\n\n".join(parts)
            combined_path = self.output_dir / f"{file_stem}.md"
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write(combined)
            logger.info(f"Aggregated {part_count} parts into {combined_path.name}")

    def process_all(self) -> Dict[str, Any]:
        """
        Main entry point for batch processing.

        Returns:
            Summary statistics
        """
        # Load or create state
        state = self._load_state() if self.resume else BatchPolishState()

        # Check for active job to resume
        if state.active_job_name and self.resume:
            logger.info(f"Resuming batch job: {state.active_job_name}")
            return self._resume_job(state)

        # Log resume state info
        if self.resume and state.completed_keys:
            logger.info(f"Resuming with {len(state.completed_keys)} previously completed keys")

        # Get pending files
        pending_files = self._get_pending_files()

        if not pending_files:
            logger.info("No files to process")
            return {"total": 0, "completed": 0, "failed": 0}

        logger.info(f"Processing {len(pending_files)} files in batch mode")

        # Build batch requests (will skip completed_keys)
        requests, metadata_map = self._build_batch_requests(pending_files, state.completed_keys)

        if not requests:
            logger.info("No requests to submit (all already completed)")
            return {"total": 0, "completed": 0, "failed": 0}

        # Submit batch job
        job_name = self.batch_client.submit(
            requests=requests,
            display_name=f"polish-{self.book_title}"
        )

        # Update state
        state.active_job_name = job_name
        state.active_job_requests = [r.key for r in requests]
        state.processing_keys = [r.key for r in requests]
        state.pending_files = [f.stem for f in pending_files]
        self._save_state(state)

        # Save metadata for result processing
        metadata_path = self.output_dir / "batch_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_map, f, indent=2, ensure_ascii=False)

        # Wait for completion
        try:
            job_info = self.batch_client.wait_for_completion(
                job_name=job_name,
                poll_interval=self.poll_interval
            )
        except KeyboardInterrupt:
            logger.warning("Processing interrupted. Run with --resume to continue.")
            return {"status": "interrupted", "job_name": job_name}

        if job_info.state != BatchJobState.SUCCEEDED:
            logger.error(f"Batch job failed: {job_info.error}")
            return {"status": "failed", "error": job_info.error}

        # Process results
        return self._process_results(state, metadata_map)

    def _resume_job(self, state: BatchPolishState) -> Dict[str, Any]:
        """Resume a previously started batch job."""
        job_name = state.active_job_name

        # Check job status
        job_info = self.batch_client.get_status(job_name)

        if job_info.state not in self.batch_client.COMPLETED_STATES:
            logger.info(f"Waiting for job {job_name} to complete...")
            try:
                job_info = self.batch_client.wait_for_completion(
                    job_name=job_name,
                    poll_interval=self.poll_interval
                )
            except KeyboardInterrupt:
                logger.warning("Processing interrupted. Run with --resume to continue.")
                return {"status": "interrupted", "job_name": job_name}

        if job_info.state != BatchJobState.SUCCEEDED:
            logger.error(f"Batch job failed: {job_info.error}")
            return {"status": "failed", "error": job_info.error}

        # Load metadata
        metadata_path = self.output_dir / "batch_metadata.json"
        if not metadata_path.exists():
            logger.error("Metadata file not found, cannot process results")
            return {"status": "error", "error": "metadata_missing"}

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_map = json.load(f)

        return self._process_results(state, metadata_map)

    def _process_results(
        self,
        state: BatchPolishState,
        metadata_map: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Process batch job results with validation.

        Args:
            state: Current batch state
            metadata_map: Metadata for each request

        Returns:
            Summary statistics
        """
        job_name = state.active_job_name

        # Get results
        responses = self.batch_client.get_results(job_name)
        logger.info(f"Retrieved {len(responses)} responses")

        # Filter out already completed keys (from previous interrupted run)
        already_completed = set(state.completed_keys)
        responses_to_process = [r for r in responses if r.key not in already_completed]

        if len(responses_to_process) < len(responses):
            logger.info(f"Skipping {len(responses) - len(responses_to_process)} already completed keys")

        # NEW: Two-phase processing for agent-based verification
        # Phase 1: Fast post-process and save all results, track suspicious ones
        completed_this_round = 0
        failed_keys = []
        suspicious_responses = []  # For agent verification

        for resp in responses_to_process:
            key = resp.key
            metadata = metadata_map.get(key, {})

            # Record this attempt for longest fallback strategy
            if resp.text:
                if key not in self.attempt_history:
                    self.attempt_history[key] = []
                self.attempt_history[key].append({
                    'text': resp.text,
                    'length': len(resp.text),
                    'retry_count': state.retry_count
                })

            if resp.error:
                logger.error(f"Request {key} failed: {resp.error}")
                failed_keys.append(key)
                continue

            if not resp.text:
                logger.warning(f"Request {key} returned empty response")
                failed_keys.append(key)
                continue

            original = metadata.get("original_content", "")

            # Skip validation for front/back matter
            if "front_matter" in key.lower() or "back_matter" in key.lower():
                processed = self._post_process_result(key, resp.text, metadata)
                self._save_result(key, processed, metadata)
                completed_this_round += 1
                if key not in state.completed_keys:
                    state.completed_keys.append(key)
                continue

            # Use n-gram for fast screening
            is_truncated, reason, details = self.ngram_detector.detect(
                original=original,
                processed=resp.text
            )

            if not is_truncated:
                # Looks good, save immediately
                logger.info(f"{key}: N-gram validation passed")
                processed = self._post_process_result(key, resp.text, metadata)
                self._save_result(key, processed, metadata)
                completed_this_round += 1
                if key not in state.completed_keys:
                    state.completed_keys.append(key)
            else:
                # Suspicious - queue for agent verification
                logger.warning(f"{key}: N-gram detected possible truncation (recall: {details.get('unique_content_recall', 0):.1%})")
                suspicious_responses.append({
                    'key': key,
                    'original': original,
                    'processed': resp.text,
                    'ngram_stats': details,
                    'metadata': metadata
                })

            # Save state periodically
            if completed_this_round % 10 == 0:
                self._save_state(state)

        # Phase 2: Agent-based batch verification of suspicious responses
        if suspicious_responses:
            logger.info(f"Agent verification: {len(suspicious_responses)} suspicious files")
            validation_results = self._agent_verify_batch(suspicious_responses)

            # Process agent results
            for item in suspicious_responses:
                key = item['key']
                is_valid = validation_results.get(key, False)

                if is_valid:
                    # Agent says it's OK, save it
                    processed = self._post_process_result(key, item['processed'], item['metadata'])
                    self._save_result(key, processed, item['metadata'])
                    completed_this_round += 1
                    if key not in state.completed_keys:
                        state.completed_keys.append(key)
                else:
                    # Agent confirms truncation
                    failed_keys.append(key)

        # Save state after processing all results
        self._save_state(state)

        # Aggregate multi-part files
        processed_files = set()
        for key, metadata in metadata_map.items():
            file_stem = metadata["file_stem"]
            if file_stem not in processed_files:
                self._aggregate_parts(file_stem)
                processed_files.add(file_stem)

        # Handle failed requests
        if failed_keys and state.retry_count < self.max_retries:
            logger.info(f"Retrying {len(failed_keys)} failed requests (attempt {state.retry_count + 1}/{self.max_retries})")
            state.failed_keys = failed_keys
            state.retry_count += 1
            state.active_job_name = None  # Clear for new job
            state.processing_keys = failed_keys  # Track what we're retrying
            self._save_state(state)

            return self._retry_failed(state, metadata_map, failed_keys)

        # Reached max_retries, use longest fallback strategy for remaining failures
        if failed_keys:
            logger.info(f"Applying longest fallback strategy for {len(failed_keys)} failed files")
            saved_count = self._save_longest_attempts(failed_keys, metadata_map)
            logger.info(f"Saved {saved_count} files with longest attempt (may be incomplete)")

            # Try online polish fallback for small number of failures
            still_failed = self._try_online_polish_fallback(failed_keys)

            if len(still_failed) < len(failed_keys):
                logger.info(
                    f"Online polish fallback recovered {len(failed_keys) - len(still_failed)} files"
                )

            # Update failed_keys to only those that still failed after fallback
            failed_keys = still_failed

        # Clear state on completion
        state.active_job_name = None
        state.active_job_requests = []
        state.processing_keys = []
        state.failed_keys = []
        # Keep completed_keys for reference but clear active state
        self._save_state(state)

        # Generate summary
        total_completed = len(state.completed_keys)
        total_failed = len(failed_keys)
        total = total_completed + total_failed

        summary = {
            "total": total,
            "completed": total_completed,
            "failed": total_failed,
            "success_rate": total_completed / total if total > 0 else 0,
            "retry_count": state.retry_count
        }

        logger.info(f"\n=== Batch Polish Summary ===")
        logger.info(f"Total: {total}, Completed: {total_completed}, Failed: {total_failed}")

        if failed_keys:
            logger.warning(f"Failed keys: {', '.join(failed_keys[:10])}")
            if len(failed_keys) > 10:
                logger.warning(f"... and {len(failed_keys) - 10} more")

        return summary

    def _save_longest_attempts(
        self,
        failed_keys: List[str],
        metadata_map: Dict[str, Dict]
    ) -> int:
        """
        Save the longest attempt for files that failed after max retries.

        This is a fallback strategy: when a file fails validation after all retries,
        we save the longest version rather than losing all content. The file is marked
        with a warning note generated by the agent explaining the issue.

        Args:
            failed_keys: List of keys that failed all retries
            metadata_map: Metadata for each request

        Returns:
            Number of files saved
        """
        saved_count = 0

        for key in failed_keys:
            attempts = self.attempt_history.get(key, [])
            if not attempts:
                logger.warning(f"{key}: No attempts recorded, cannot save")
                continue

            # Find the longest attempt
            longest = max(attempts, key=lambda x: x['length'])

            # Get metadata
            metadata = metadata_map.get(key, {})
            if not metadata:
                logger.warning(f"{key}: No metadata found, cannot save")
                continue

            # Generate diagnostic note using agent
            diagnostic_note = self._generate_diagnostic_note(
                key=key,
                original=metadata.get("original_content", ""),
                processed=longest['text'],
                attempts_count=len(attempts)
            )

            # Post-process and save the longest version with diagnostic note
            try:
                processed = self._post_process_result(key, longest['text'], metadata)

                # Prepend diagnostic note
                if diagnostic_note:
                    processed = diagnostic_note + "\n\n---\n\n" + processed

                self._save_result(key, processed, metadata)

                logger.warning(
                    f"{key}: Saved longest attempt ({longest['length']} chars) "
                    f"from {len(attempts)} attempts (retry {longest['retry_count']}) "
                    f"with diagnostic note - content may be incomplete"
                )
                saved_count += 1

            except Exception as e:
                logger.error(f"{key}: Failed to save longest attempt: {e}")
                continue

        return saved_count

    def _generate_diagnostic_note(
        self,
        key: str,
        original: str,
        processed: str,
        attempts_count: int
    ) -> str:
        """
        Use agent to generate a diagnostic note explaining why the file failed.

        The agent analyzes the failure and provides information about:
        - Which pages/sections might have scan issues
        - What type of problem was detected
        - Recommendations for the user

        Args:
            key: File key
            original: Original content
            processed: Processed content (longest attempt)
            attempts_count: Number of attempts made

        Returns:
            Diagnostic note in markdown format
        """
        try:
            from pdf2epub.processors.utils.agent_verifier import get_verification_model
            from pydantic_ai import Agent

            model = get_verification_model()

            # Create a simple agent for diagnostic
            diagnostic_prompt = f"""You are analyzing a file that failed polish validation after {attempts_count} attempts.

Your task: Write a brief diagnostic note (2-3 sentences) explaining:
1. What type of problem was detected (truncation, corruption, etc.)
2. Where in the content the problem likely occurs
3. Brief suggestion for the user

Keep it concise and actionable. Use markdown format.

File: {key}
Original length: {len(original)} chars
Processed length: {len(processed)} chars
Ratio: {len(processed)/len(original)*100:.1f}%

Original ending (last 200 chars):
{original[-200:]}

Processed ending (last 200 chars):
{processed[-200:]}
"""

            agent = Agent(model, output_type=str)
            result = agent.run_sync(diagnostic_prompt)

            # Format the diagnostic note
            note = f"""<!-- DIAGNOSTIC NOTE: Auto-generated by agent verification -->
> ⚠️ **Content may be incomplete** - This file failed validation after {attempts_count} attempts.
> The longest version ({len(processed)} chars) has been saved.

{result.output}

---
"""
            return note

        except Exception as e:
            logger.warning(f"Failed to generate diagnostic note for {key}: {e}")
            # Fallback to simple note
            return f"""<!-- DIAGNOSTIC NOTE -->
> ⚠️ **Content may be incomplete** - This file failed validation after {attempts_count} attempts.
> Original: {len(original)} chars → Processed: {len(processed)} chars ({len(processed)/len(original)*100:.1f}%)
> Please manually review this file for completeness.

---
"""

    def _try_online_polish_fallback(self, failed_keys: List[str]) -> List[str]:
        """
        Try online polish for small number of remaining failures.

        When batch polish exhausts retries and only has a small number of failures
        (< threshold), try using online polish which might succeed due to different
        processing path and retry behavior.

        Args:
            failed_keys: List of file keys that failed batch polish

        Returns:
            List of keys that still failed after online polish
        """
        batch_config = self.config.get('batch', {})
        threshold = batch_config.get('online_polish_fallback_threshold', 5)

        if len(failed_keys) > threshold:
            logger.info(
                f"Skipping online polish fallback: {len(failed_keys)} failures "
                f"> threshold {threshold}"
            )
            return failed_keys

        logger.info(
            f"Attempting online polish fallback for {len(failed_keys)} failed files "
            f"(threshold: {threshold})"
        )

        # Import PolishProcessor (avoid circular import)
        from pdf2epub.processors.polisher import PolishProcessor
        from pdf2epub.processors.utils import VerificationFile, verify_batch

        # Initialize online polisher with same config
        online_polisher = PolishProcessor(
            config=self.config,
            book_title=self.book_title,
            book_structure=self.book_structure,
            content_type=self.content_type
        )

        # Process each failed file with online polish
        still_failed = []
        for key in failed_keys:
            # Skip part files - they don't exist as separate files in input_dir
            # (parts are created during processing from the original file)
            if ".part" in key:
                logger.info(
                    f"Online polish fallback: Skipping {key} (part files cannot be "
                    f"recovered individually - longest fallback has been applied)"
                )
                still_failed.append(key)
                continue

            logger.info(f"Online polish fallback: {key}")

            try:
                # Get original content
                input_file = self.input_dir / f"{key}.md"
                if not input_file.exists():
                    logger.error(f"{key}: Input file not found")
                    still_failed.append(key)
                    continue

                with open(input_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                # Process with online polish (uses process_unit)
                result = online_polisher.process_unit(
                    content=original_content,
                    unit_key=key,
                    file_name=f"{key}.md",
                    original_content=original_content
                )

                # Validate using agent
                verification_file = VerificationFile(
                    file_key=key,
                    original_content=original_content,
                    processed_content=result
                )

                verification_results = verify_batch(
                    files={key: verification_file},
                    task_type="polish"
                )

                if verification_results[0].status == "complete":
                    # Success! Save and overwrite the diagnostic note version
                    output_file = self.output_dir / f"{key}.md"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result)
                    logger.info(
                        f"{key}: Online polish succeeded, overwriting previous version"
                    )
                else:
                    # Still failed, keep longest fallback version
                    logger.warning(
                        f"{key}: Online polish also failed - {verification_results[0].reason}"
                    )
                    still_failed.append(key)

            except Exception as e:
                logger.error(f"{key}: Online polish error: {e}")
                still_failed.append(key)

        logger.info(
            f"Online polish fallback: {len(failed_keys) - len(still_failed)} succeeded, "
            f"{len(still_failed)} still failed"
        )

        return still_failed

    def _retry_failed(
        self,
        state: BatchPolishState,
        metadata_map: Dict[str, Dict],
        failed_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Retry failed requests.

        Args:
            state: Current batch state
            metadata_map: Original metadata map
            failed_keys: List of keys that failed

        Returns:
            Summary statistics
        """
        # Build new requests for failed keys
        requests = []
        new_metadata = {}

        for key in failed_keys:
            if key not in metadata_map:
                continue

            metadata = metadata_map[key]
            original_content = self._content_cache.get(key) or metadata.get("original_content", "")

            if not original_content:
                logger.warning(f"Cannot retry {key}: missing original content")
                continue

            # Rebuild prompt
            prompt = self._build_prompt(
                file_stem=metadata["file_stem"],
                part_idx=metadata["part_idx"],
                total_parts=metadata["total_parts"],
                content=original_content
            )

            request = BatchRequest(
                key=key,
                contents=[
                    {"parts": [{"text": prompt}], "role": "user"},
                ]
            )
            requests.append(request)
            new_metadata[key] = metadata

        if not requests:
            return {"total": 0, "completed": 0, "failed": len(failed_keys)}

        # Submit retry batch
        job_name = self.batch_client.submit(
            requests=requests,
            display_name=f"polish-retry-{self.book_title}"
        )

        state.active_job_name = job_name
        state.active_job_requests = [r.key for r in requests]
        self._save_state(state)

        # Update metadata file
        metadata_path = self.output_dir / "batch_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)

        # Wait and process
        try:
            job_info = self.batch_client.wait_for_completion(
                job_name=job_name,
                poll_interval=self.poll_interval
            )
        except KeyboardInterrupt:
            logger.warning("Retry interrupted. Run with --resume to continue.")
            return {"status": "interrupted", "job_name": job_name}

        if job_info.state != BatchJobState.SUCCEEDED:
            logger.error(f"Retry batch job failed: {job_info.error}")
            return {"status": "failed", "error": job_info.error}

        return self._process_results(state, new_metadata)
