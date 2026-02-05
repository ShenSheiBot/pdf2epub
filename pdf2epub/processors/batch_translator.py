"""
Batch Translate Processor using Gemini Batch API.

Provides asynchronous, high-throughput translation processing at 50% cost reduction
compared to real-time inference.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
import tiktoken

from .utils.truncation import NGramTruncationDetector
from .tracker import ProcessingTracker, AttemptRecord
from ..utils.batch_utils import (
    GeminiBatchClient,
    BatchRequest,
    BatchResponse,
    BatchJobState,
    BatchJobInfo,
    BATCH_DEFAULTS
)
from ..chapter_identity import ChapterIdentity

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class BatchTranslateState:
    """Persistent state for batch translate processing."""
    active_job_name: Optional[str] = None
    active_job_requests: List[str] = field(default_factory=list)  # List of request keys in current job
    pending_files: List[str] = field(default_factory=list)
    retry_count: int = 0
    failed_keys: List[str] = field(default_factory=list)
    # Track completed keys across all rounds (for resume support)
    completed_keys: List[str] = field(default_factory=list)
    # Track which keys are being processed in current job (for partial result handling)
    processing_keys: List[str] = field(default_factory=list)
    # Track attempt history for longest fallback
    attempt_history: Dict[str, List[Dict]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "active_job_name": self.active_job_name,
            "active_job_requests": self.active_job_requests,
            "pending_files": self.pending_files,
            "retry_count": self.retry_count,
            "failed_keys": self.failed_keys,
            "completed_keys": self.completed_keys,
            "processing_keys": self.processing_keys,
            "attempt_history": self.attempt_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BatchTranslateState':
        return cls(
            active_job_name=data.get("active_job_name"),
            active_job_requests=data.get("active_job_requests", []),
            pending_files=data.get("pending_files", []),
            retry_count=data.get("retry_count", 0),
            failed_keys=data.get("failed_keys", []),
            completed_keys=data.get("completed_keys", []),
            processing_keys=data.get("processing_keys", []),
            attempt_history=data.get("attempt_history", {}),
        )


class BatchTranslateProcessor:
    """
    Batch processor for translating markdown content.

    Uses Gemini Batch API for 50% cost reduction on large-scale processing.
    """

    def __init__(
        self,
        config: Dict,
        book_title: str,
        source_language: str = "Japanese",
        target_language: str = "Chinese",
        max_retries: int = 1,
        poll_interval: int = 60,
        resume: bool = False,
        use_entities: Optional[bool] = None,
        book_structure: Optional[Dict] = None
    ):
        """
        Initialize the batch translate processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            source_language: Source language for translation
            target_language: Target language for translation
            max_retries: Maximum retries for validation failures (default: 1)
            poll_interval: Seconds between status polls
            resume: Whether to resume from previous progress
            use_entities: Whether to use extracted entities for consistency
            book_structure: Optional book structure from breakdown phase
        """
        self.config = config
        self.book_title = book_title
        self.source_language = source_language
        self.target_language = target_language
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.resume = resume
        self.book_structure = book_structure or {}

        # Setup directories
        self.input_dir = Path("output") / book_title / "polished_markdown"
        self.output_dir = Path("output") / book_title / "translated_batch"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State file for persistence
        self.state_file = self.output_dir / "batch_state.json"
        self.metadata_file = self.output_dir / "batch_metadata.json"

        # Initialize processing tracker
        tracker_path = self.output_dir / "processing_tracker.json"
        self.tracker = ProcessingTracker(tracker_path, "BatchTranslateProcessor")

        # Load entities if requested
        if use_entities is None:
            # Auto-detect: use entities if the file exists
            entities_file = Path("output") / book_title / "translation_entities.json"
            if entities_file.exists():
                logger.info("Auto-detected translation entities file, will use for consistency")
                self.entities = self._load_entities()
            else:
                self.entities = None
        elif use_entities:
            self.entities = self._load_entities()
        else:
            self.entities = None

        # Use NGramTruncationDetector for fast screening (same as polish)
        # Agent verification will handle accurate detection in Phase 2
        self.truncation_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=0.60,
            allow_deduplication=True
        )

        # Get validation settings
        validation_config = config.get('validation_strategy', {})
        self.validate_chinese = validation_config.get('validate_chinese_translation', True)

        logger.info("Using N-gram detector + agent-based verification for batch translate")

        # Initialize batch client
        batch_config = config.get('batch', {})
        credentials = config.get('credentials', {}).get('providers', {})

        # Determine provider and credentials
        batch_provider = batch_config.get('provider', BATCH_DEFAULTS['provider'])
        provider_config = credentials.get(batch_provider, {})

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

        # Model for batch translate
        translate_batch_config = batch_config.get('translate', {})
        batch_model = (
            translate_batch_config.get('model') or
            batch_config.get('model') or
            BATCH_DEFAULTS['translate']['model']
        )
        logger.info(f"Using batch translate model: {batch_model}")

        # Initialize Gemini Batch client
        self.batch_client = GeminiBatchClient(
            api_key=api_key,
            base_url=base_url,
            model=batch_model
        )

    def _load_entities(self) -> Optional[Dict]:
        """Load translation entities from file."""
        entities_file = Path("output") / self.book_title / "translation_entities.json"
        if not entities_file.exists():
            logger.warning(f"Entities file not found: {entities_file}")
            return None

        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
            logger.info(f"Loaded {len(entities_data)} translation entities")
            return entities_data
        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            return None

    def _get_max_tokens_per_part(self) -> int:
        """Get maximum tokens per part for splitting."""
        # Gemini Batch API has generous limits, use 8000 as default
        return self.config.get('splitting', {}).get('max_tokens_per_part', 8000)

    def _build_prompt(self) -> str:
        """
        Build the translation prompt (batch mode doesn't support conversation history).

        Returns:
            System prompt for translation
        """
        # Use Chinese prompt if target language is Chinese
        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            prompt = f"""你是一位专业的学术和文学文本翻译专家。

请将以下markdown内容从{self.source_language}翻译成简体中文。

重要要求：
1. **保留所有markdown格式**：保持标题(#, ##, ###)、强调(*斜体*, **粗体**)、列表、引用、代码块等格式不变
2. **图片链接保持不变**：不要翻译或修改图片路径，如 ![...](../images/xxx.png) 或 <img src="..." />
3. **脚注处理**：
   - 保持脚注格式 [^1], [^2] 等不变
   - 不要添加原文中不存在的脚注
4. **维持文档结构**：保持相同的段落分隔、章节划分和整体布局
5. **学术文本**：使用准确的中文学术术语
6. **文学文本**：保留原文的风格和语调
7. **不要添加说明**：只返回翻译后的markdown内容，不要添加任何解释或评论
8. **必须输出简体中文**：请确保翻译结果是简体中文，不要返回英文或其他语言"""
        else:
            prompt = f"""You are a professional translator specializing in academic and literary texts.

Translate the following markdown content from {self.source_language} to {self.target_language}.

IMPORTANT REQUIREMENTS:
1. **Preserve ALL markdown formatting**: Keep headers (#, ##, ###), emphasis (*italic*, **bold**), lists, quotes, code blocks, etc.
2. **Keep image links unchanged**: Do not translate or modify image paths like ![...](../images/xxx.png) or <img src="..." />
3. **Don't touch footnote**:
   - Keep the footnote format [^1], [^2], etc. unchanged
   - Don't add footnote that does not exist in the original text
4. **Maintain document structure**: Keep the same paragraph breaks, section divisions, and overall layout
5. **For academic texts**: Use appropriate academic terminology in the target language
6. **For literary texts**: Preserve the style and tone of the original
7. **Do NOT add explanations**: Return ONLY the translated markdown, no explanations or comments"""

        # Add specific rules for Japanese to Chinese translation
        if self.source_language.lower() in ["japanese", "日本語"] and self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            # Add rules in Chinese since the prompt is in Chinese
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                prompt += """

日译中特殊规则：
9. **删除不必要的注音**：不要包含类似 谦逊（けんそん）的日文读音标注，中文读者不需要日文读音
10. **正确处理日文语气词**：
   - 删除或调整句尾的っ
   - 「っ，呜，呜嗯っ……呜，呜」应该翻译为「呜，呜嗯……呜，呜」
   - 不要直接将っ翻译成中文字符
11. **自然的中文表达**：确保译文符合中文表达习惯，没有日文语言痕迹"""
            else:
                prompt += """

SPECIFIC RULES FOR JAPANESE TO CHINESE:
8. **Remove unnecessary ruby annotations**: Do NOT include pronunciation guides like 谦逊（けんそん）. Chinese readers don't need Japanese readings.
9. **Handle Japanese particles properly**:
   - Remove or adapt っ at the end of sentences/exclamations
   - 「っ，呜，呜嗯っ……呜，呜」 should become 「呜，呜嗯……呜，呜」
   - Do not literally translate っ as a character
10. **Natural Chinese expression**: Ensure the translation reads naturally in Chinese without Japanese linguistic artifacts"""

        # Add entity reference if available
        if self.entities:
            prompt += self._create_entity_reference_section()

        # Final instruction in appropriate language
        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            prompt += "\n\n请翻译以下内容："
        else:
            prompt += "\n\nTranslate the following content:"
        return prompt

    def _create_entity_reference_section(self) -> str:
        """Create the entity reference section for the prompt."""
        if not self.entities:
            return ""

        # Use Chinese headers if target language is Chinese
        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            reference = "\n\n**翻译一致性参考：**\n"
            reference += "请使用以下既定译名以保持一致性：\n"
            reference += "重要：请始终使用提供的人物名称及其昵称的翻译。\n\n"
        else:
            reference = "\n\n**TRANSLATION CONSISTENCY REFERENCE:**\n"
            reference += "Use these established translations for consistency:\n"
            reference += "IMPORTANT: Always use the provided translations for character names AND their nicknames.\n\n"

        # Add characters
        if "characters" in self.entities and self.entities["characters"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**人物：**\n"
            else:
                reference += "**Characters:**\n"
            for char in self.entities["characters"]:
                reference += f"- {char['japanese']}"
                if char.get('reading'):
                    reference += f" ({char['reading']})"
                reference += f" → {char['chinese']}"
                if char.get('gender'):
                    reference += f" [{char['gender']}]"
                reference += "\n"

                # Add nicknames if present
                if char.get('nicknames'):
                    for nickname in char['nicknames']:
                        if isinstance(nickname, dict):
                            reference += f"  • {nickname.get('japanese', '')} → {nickname.get('chinese', '')}\n"
                        elif isinstance(nickname, str):
                            # Old format compatibility - just the Japanese nickname
                            reference += f"  • {nickname} (needs translation)\n"
            reference += "\n"

        # Add places
        if "places" in self.entities and self.entities["places"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**地点：**\n"
            else:
                reference += "**Places:**\n"
            for place in self.entities["places"]:
                reference += f"- {place['japanese']} → {place['chinese']}\n"
            reference += "\n"

        # Add important terms
        if "terms" in self.entities and self.entities["terms"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**专有名词：**\n"
            else:
                reference += "**Special Terms:**\n"
            for term in self.entities["terms"]:
                reference += f"- {term['japanese']} → {term['chinese']}\n"
            reference += "\n"

        # Add organizations
        if "organizations" in self.entities and self.entities["organizations"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**组织：**\n"
            else:
                reference += "**Organizations:**\n"
            for org in self.entities["organizations"]:
                reference += f"- {org['japanese']} → {org['chinese']}\n"
            reference += "\n"

        # Add races if present
        if "races" in self.entities and self.entities["races"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**种族/物种：**\n"
            else:
                reference += "**Races/Species:**\n"
            for race in self.entities["races"]:
                reference += f"- {race['japanese']} → {race['chinese']}"
                if race.get('chinese_plural'):
                    reference += f" (plural: {race['chinese_plural']})"
                reference += "\n"
            reference += "\n"

        # Add items if present
        if "items" in self.entities and self.entities["items"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**物品：**\n"
            else:
                reference += "**Items:**\n"
            for item in self.entities["items"]:
                reference += f"- {item['japanese']} → {item['chinese']}\n"
            reference += "\n"

        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            reference += "**重要提示：** 请在整个文本中保持这些译名的一致性。\n"
        else:
            reference += "**IMPORTANT:** Maintain these translations consistently throughout the text.\n"

        return reference

    def _save_state(self, state: BatchTranslateState) -> None:
        """Save state to file."""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved state to {self.state_file}")

    def _load_state(self) -> Optional[BatchTranslateState]:
        """Load state from file."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return BatchTranslateState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def _get_pending_files(self, state: BatchTranslateState) -> List[str]:
        """
        Get list of files that need processing.

        Returns:
            List of file stems (without .md extension)
        """
        # Get all markdown files from input directory
        all_files = []
        for file in sorted(self.input_dir.glob("*.md")):
            file_stem = file.stem
            # Skip already completed files
            if file_stem not in state.completed_keys:
                all_files.append(file_stem)

        logger.info(f"Found {len(all_files)} files pending translation")
        return all_files

    def _read_file_content(self, file_key: str) -> str:
        """Read content from input file."""
        file_path = self.input_dir / f"{file_key}.md"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(tokenizer.encode(text))

    def _record_completion(self, key: str, output_tokens: int = 0, reason: str = None) -> None:
        """Record a completed processing attempt."""
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed",
            model=self.batch_client.model,
            input_tokens=0,
            output_tokens=output_tokens,
            duration_seconds=0.0
        )
        if reason:
            attempt.fallback_reason = reason
        self.tracker.record_attempt(key, attempt)

    def _split_content_if_needed(self, file_key: str, content: str) -> List[Tuple[str, str]]:
        """
        Split content into parts if it exceeds token limit.

        Args:
            file_key: File identifier
            content: Markdown content to split

        Returns:
            List of (part_key, part_content) tuples
        """
        token_count = self._count_tokens(content)
        max_tokens = self._get_max_tokens_per_part()

        if token_count <= max_tokens:
            # No splitting needed
            return [(file_key, content)]

        # Need to split
        logger.info(f"{file_key}: {token_count} tokens exceeds {max_tokens}, splitting")

        # Split by paragraphs (double newline)
        paragraphs = content.split('\n\n')
        parts = []
        current_part = []
        current_tokens = 0
        part_num = 1

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > max_tokens and current_part:
                # Save current part
                part_content = '\n\n'.join(current_part)
                part_key = f"{file_key}.part{part_num}"
                parts.append((part_key, part_content))

                # Start new part
                current_part = [para]
                current_tokens = para_tokens
                part_num += 1
            else:
                current_part.append(para)
                current_tokens += para_tokens

        # Add final part
        if current_part:
            part_content = '\n\n'.join(current_part)
            part_key = f"{file_key}.part{part_num}"
            parts.append((part_key, part_content))

        logger.info(f"{file_key}: Split into {len(parts)} parts")
        return parts

    def _build_batch_requests(self, file_keys: List[str]) -> Tuple[List[BatchRequest], Dict[str, Dict]]:
        """
        Build batch requests for translation.

        Args:
            file_keys: List of file keys to process

        Returns:
            (requests, metadata_map)
            - requests: List of BatchRequest objects
            - metadata_map: {request_key: metadata_dict}
        """
        system_prompt = self._build_prompt()
        requests = []
        metadata_map = {}

        for file_key in file_keys:
            # Read file content
            content = self._read_file_content(file_key)

            # Check if content is image-only (skip translation)
            stripped = content.strip()
            if not stripped or (stripped.startswith('![') and stripped.count('\n') < 3):
                logger.info(f"Skipping {file_key}: image-only content")
                # Mark as completed directly
                self._record_completion(file_key, output_tokens=0, reason="image-only content")
                continue

            # Split if needed
            parts = self._split_content_if_needed(file_key, content)

            # Create request for each part
            for part_key, part_content in parts:
                # Build full prompt
                full_prompt = f"{system_prompt}\n\n{part_content}"

                # Create batch request
                request = BatchRequest(
                    key=part_key,
                    contents=[
                        {"parts": [{"text": full_prompt}], "role": "user"}
                    ]
                )
                requests.append(request)

                # Store metadata for later result processing
                metadata_map[part_key] = {
                    "file_key": file_key,
                    "part_key": part_key,
                    "is_multi_part": len(parts) > 1,
                    "part_index": parts.index((part_key, part_content)),
                    "total_parts": len(parts),
                    "original_content": part_content,
                    "original_token_count": self._count_tokens(part_content)
                }

        logger.info(f"Built {len(requests)} batch requests from {len(file_keys)} files")
        return requests, metadata_map

    def process_all(self) -> Dict[str, Any]:
        """
        Main entry point for batch translate processing.

        Returns:
            Summary statistics
        """
        # Load or initialize state
        state = self._load_state() if self.resume else BatchTranslateState()

        # If resuming and job is active, continue from there
        if state.active_job_name and self.resume:
            logger.info(f"Resuming batch job: {state.active_job_name}")
            return self._resume_job(state)

        # Get pending files
        pending = self._get_pending_files(state)
        if not pending:
            logger.info("No files pending translation")
            return {"total": 0, "completed": 0, "failed": 0}

        state.pending_files = pending
        state.retry_count = 0
        state.failed_keys = []
        self._save_state(state)

        # Process with retry loop
        total_files = len(pending)
        retry_files = pending

        while retry_files and state.retry_count <= self.max_retries:
            if state.retry_count > 0:
                logger.info(f"Retry {state.retry_count}/{self.max_retries} for {len(retry_files)} files")

            # Process batch
            result = self._process_batch_round(state, retry_files)

            # Check if any failed
            if not result["failed_keys"]:
                logger.info("All files translated successfully")
                break

            # Prepare for retry
            retry_files = [k for k in result["failed_keys"]]
            state.retry_count += 1
            state.failed_keys = retry_files
            self._save_state(state)

        # After exhausting retries, use longest fallback for remaining failures
        if state.failed_keys:
            logger.info(f"Applying longest fallback for {len(state.failed_keys)} failed files")
            saved_count = self._save_longest_attempts(state.failed_keys, state.attempt_history)
            logger.info(f"Saved {saved_count} files with longest attempt (may be incomplete)")

        # Aggregate multi-part files
        self._aggregate_all_parts()

        # Translate TOC
        self._translate_toc()

        # Final summary
        completed_count = len([k for k in state.completed_keys if k not in state.failed_keys])
        return {
            "total": total_files,
            "completed": completed_count,
            "failed": len(state.failed_keys)
        }

    def _process_batch_round(self, state: BatchTranslateState, file_keys: List[str]) -> Dict[str, Any]:
        """
        Process one round of batch translation.

        Args:
            state: Current processing state
            file_keys: List of file keys to process

        Returns:
            Dict with "completed_keys" and "failed_keys"
        """
        # Build batch requests
        requests, metadata_map = self._build_batch_requests(file_keys)

        if not requests:
            return {"completed_keys": file_keys, "failed_keys": []}

        # Submit batch job
        logger.info(f"Submitting batch job with {len(requests)} requests")
        job_name = self.batch_client.submit(requests)
        logger.info(f"Batch job submitted: {job_name}")

        # Update state
        state.active_job_name = job_name
        state.active_job_requests = list(metadata_map.keys())
        state.processing_keys = file_keys
        self._save_state(state)

        # Save metadata
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_map, f, ensure_ascii=False, indent=2)

        # Wait for completion and process results
        result = self._wait_for_completion(state, metadata_map)

        return {
            "completed_keys": [k for k in state.completed_keys],
            "failed_keys": state.failed_keys
        }

    def _wait_for_completion(self, state: BatchTranslateState, metadata_map: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Wait for batch job to complete and process results.

        Args:
            state: Current processing state
            metadata_map: Metadata for each request

        Returns:
            Summary statistics
        """
        job_name = state.active_job_name

        while True:
            # Get job status
            job_info = self.batch_client.get_status(job_name)

            logger.info(
                f"Job {job_name}: {job_info.state.value} "
                f"({job_info.total_requests} total, {job_info.completed_requests} completed)"
            )

            if job_info.state == BatchJobState.SUCCEEDED:
                logger.info(f"Job {job_name} completed successfully")
                break
            elif job_info.state in [BatchJobState.FAILED, BatchJobState.CANCELLED, BatchJobState.EXPIRED]:
                logger.error(f"Job {job_name} failed with state: {job_info.state.value}")
                # Still try to get partial results
                break
            elif job_info.state in [BatchJobState.PENDING, BatchJobState.RUNNING]:
                logger.info(f"Job in progress, waiting {self.poll_interval}s...")
                time.sleep(self.poll_interval)
            else:
                logger.warning(f"Unknown job state: {job_info.state.value}, waiting...")
                time.sleep(self.poll_interval)

        # Get results
        logger.info(f"Fetching results for job {job_name}")
        responses = self.batch_client.get_results(job_name)

        # Process results
        return self._process_results(state, responses, metadata_map)

    def _process_results(
        self,
        state: BatchTranslateState,
        responses: List[BatchResponse],
        metadata_map: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Process batch results and validate.

        Args:
            state: Current processing state
            responses: List of batch responses
            metadata_map: Metadata for each request

        Returns:
            Summary statistics
        """
        logger.info(f"Processing {len(responses)} batch results")

        # Group responses by success/failure
        successful = {}
        failed_keys = []

        for response in responses:
            request_key = response.key
            metadata = metadata_map.get(request_key)

            if not metadata:
                logger.warning(f"No metadata found for {request_key}, skipping")
                continue

            if response.error:
                logger.error(f"{request_key}: API error - {response.error}")
                failed_keys.append(request_key)
                continue

            if not response.text:
                logger.error(f"{request_key}: Empty response")
                failed_keys.append(request_key)
                continue

            # Store successful result
            successful[request_key] = {
                "text": response.text,
                "metadata": metadata
            }

            # Track in attempt history for longest fallback
            file_key = metadata["file_key"]
            if file_key not in state.attempt_history:
                state.attempt_history[file_key] = []
            state.attempt_history[file_key].append({
                "text": response.text,
                "length": len(response.text),
                "retry_count": state.retry_count
            })

        logger.info(f"Batch results: {len(successful)} successful, {len(failed_keys)} failed")

        # Validate successful results
        validated_keys, still_failed = self._validate_batch_results(successful)

        # Combine all failures
        all_failed_keys = list(set(failed_keys + still_failed))

        # Update state
        state.completed_keys.extend(validated_keys)
        state.failed_keys = all_failed_keys
        state.active_job_name = None
        state.active_job_requests = []
        state.processing_keys = []
        self._save_state(state)

        return {
            "total": len(metadata_map),
            "completed": len(validated_keys),
            "failed": len(all_failed_keys)
        }

    def _validate_batch_results(
        self,
        successful: Dict[str, Dict]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate batch results using two-phase approach.

        Args:
            successful: {request_key: {"text": ..., "metadata": ...}}

        Returns:
            (validated_keys, failed_keys)
        """
        from .utils.verification_tools import VerificationFile

        passed_keys = []
        suspicious = {}

        # Phase 1: LLM truncation detection
        logger.info(f"Phase 1: LLMTruncationDetector screening for {len(successful)} results")

        for request_key, result_data in successful.items():
            text = result_data["text"]
            metadata = result_data["metadata"]
            original = metadata["original_content"]

            # Check for truncation
            is_truncated, reason, details = self.truncation_detector.detect(
                original=original,
                processed=text
            )

            if not is_truncated:
                # Passed Phase 1, save immediately
                self._save_result(request_key, text, metadata)
                passed_keys.append(metadata["file_key"])
            else:
                # Suspicious, queue for agent verification
                logger.warning(f"{request_key}: LLM detector suspicious - {reason}")
                suspicious[request_key] = VerificationFile(
                    key=request_key,
                    original=original,
                    processed=text
                )

        logger.info(f"{len(passed_keys)} results passed LLM screening, {len(suspicious)} suspicious")

        if not suspicious:
            return passed_keys, []

        # Phase 2: Agent verification (optional, based on validation config)
        if not self.validate_chinese:
            logger.info("Chinese validation disabled, accepting suspicious results")
            # Save suspicious results anyway
            for request_key, verification_file in suspicious.items():
                result_data = successful[request_key]
                self._save_result(request_key, verification_file.processed, result_data["metadata"])
                passed_keys.append(result_data["metadata"]["file_key"])
            return passed_keys, []

        logger.info(f"Phase 2: Agent verification for {len(suspicious)} suspicious results")

        # Import agent verifier
        from .utils.agent_verifier import TranslationVerificationAgent

        agent = TranslationVerificationAgent(
            config=self.config,
            source_language=self.source_language,
            target_language=self.target_language
        )

        verification_results = agent.verify_batch(
            files=suspicious,
            task_type="translate"
        )

        failed_keys = []
        for result in verification_results:
            if result.status == "complete":
                # Agent confirmed complete
                result_data = successful[result.key]
                self._save_result(result.key, suspicious[result.key].processed, result_data["metadata"])
                passed_keys.append(result_data["metadata"]["file_key"])
                logger.info(f"{result.key}: Agent verified as complete - {result.reason}")
            else:
                # Agent confirmed truncation
                failed_keys.append(result.key)
                logger.warning(f"{result.key}: Agent confirmed truncation - {result.reason}")

        return passed_keys, failed_keys

    def _save_result(self, request_key: str, text: str, metadata: Dict) -> None:
        """Save translation result to file."""
        file_key = metadata["file_key"]
        part_key = metadata["part_key"]

        # Save to output directory
        output_file = self.output_dir / f"{part_key}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        logger.debug(f"Saved translation: {output_file}")

        # Update tracker
        self._record_completion(request_key, output_tokens=self._count_tokens(text))

    def _aggregate_parts(self, file_key: str, parts: List[str]) -> None:
        """Aggregate multi-part translation into single file."""
        combined_content = []

        for part_key in sorted(parts):
            part_file = self.output_dir / f"{part_key}.md"
            if part_file.exists():
                with open(part_file, 'r', encoding='utf-8') as f:
                    combined_content.append(f.read())

        # Write aggregated file
        output_file = self.output_dir / f"{file_key}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(combined_content))

        logger.info(f"Aggregated {len(parts)} parts into {file_key}.md")

    def _save_longest_attempts(self, failed_keys: List[str], attempt_history: Dict[str, List[Dict]]) -> int:
        """
        Save longest attempts with diagnostic notes for failed files.

        Args:
            failed_keys: List of file keys that failed validation
            attempt_history: {file_key: [{"text": ..., "length": ..., "retry_count": ...}]}

        Returns:
            Number of files saved
        """
        saved_count = 0

        for file_key in failed_keys:
            attempts = attempt_history.get(file_key, [])
            if not attempts:
                logger.error(f"{file_key}: No attempts recorded")
                continue

            # Find longest attempt
            longest = max(attempts, key=lambda x: x['length'])

            # Get original content for diagnostic
            original = self._read_file_content(file_key)

            # Generate diagnostic note
            diagnostic_note = self._generate_diagnostic_note(
                file_key=file_key,
                original=original,
                processed=longest['text'],
                attempts_count=len(attempts)
            )

            # Prepend diagnostic note
            content_with_note = diagnostic_note + "\n\n---\n\n" + longest['text']

            # Save with note
            output_file = self.output_dir / f"{file_key}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content_with_note)

            logger.warning(
                f"{file_key}: Saved longest attempt ({longest['length']} chars) "
                f"from {len(attempts)} attempts (retry {longest['retry_count']}) "
                f"with diagnostic note - content may be incomplete"
            )
            saved_count += 1

        return saved_count

    def _generate_diagnostic_note(self, file_key: str, original: str, processed: str, attempts_count: int) -> str:
        """Generate diagnostic note using agent analysis."""
        try:
            from pydantic_ai import Agent

            # Use translation verification model
            translate_config = self.config.get('translation', {})
            models = translate_config.get('models', [])
            if models:
                provider = models[0].get('provider', 'gemini')
                model_name = models[0].get('model', 'gemini-2.5-flash')
                model = f"{provider}:{model_name}"
            else:
                model = "gemini:gemini-2.5-flash"

            diagnostic_prompt = f"""Analyze this file that failed translation validation after {attempts_count} attempts.

File: {file_key}
Original length: {len(original)} chars
Translated length: {len(processed)} chars
Ratio: {len(processed)/len(original)*100:.1f}%

Original ending: ...{original[-200:]}
Translated ending: ...{processed[-200:]}

Provide:
1. **Issue Detected**: What type of problem (truncation, corruption, etc.)
2. **Likely Cause**: Why translation failed (be specific about location if possible)
3. **Recommendation**: What user should do to fix

Be concise but specific."""

            agent = Agent(model, output_type=str)
            result = agent.run_sync(diagnostic_prompt)

            note = f"""<!-- DIAGNOSTIC NOTE: Auto-generated by agent verification -->
> ⚠️ **Content may be incomplete** - This file failed validation after {attempts_count} attempts.
> The longest version ({len(processed)} chars) has been saved.

{result.output}

---"""
            return note

        except Exception as e:
            logger.error(f"Failed to generate diagnostic note: {e}")
            # Fallback to simple note
            return f"""<!-- DIAGNOSTIC NOTE -->
> ⚠️ **Content may be incomplete** - This file failed validation after {attempts_count} attempts.
> The longest version ({len(processed)} chars) has been saved.
> Please manually review this file.

---"""

    def _aggregate_all_parts(self) -> None:
        """Aggregate all multi-part translations."""
        # Group files by base name
        file_groups = {}
        for file in self.output_dir.glob("*.md"):
            stem = file.stem
            # Check if it's a part file
            if ".part" in stem:
                base_name = stem.split(".part")[0]
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(stem)

        # Aggregate each group
        for base_name, parts in file_groups.items():
            if len(parts) > 1:
                self._aggregate_parts(base_name, parts)

    def _translate_toc(self) -> None:
        """Translate TOC titles if book structure exists."""
        toc_file = Path("output") / self.book_title / "toc_tree.json"
        if not toc_file.exists():
            logger.info("No TOC file found, skipping TOC translation")
            return

        logger.info("Translating TOC titles...")

        try:
            with open(toc_file, 'r', encoding='utf-8') as f:
                toc_tree = json.load(f)

            # Collect all titles
            titles_to_translate = []

            def collect_titles(node):
                if isinstance(node, dict):
                    if "title" in node and node["title"]:
                        titles_to_translate.append(node["title"])
                    if "children" in node:
                        for child in node["children"]:
                            collect_titles(child)
                elif isinstance(node, list):
                    for item in node:
                        collect_titles(item)

            collect_titles(toc_tree)

            if not titles_to_translate:
                logger.info("No titles to translate in TOC")
                return

            # Build batch request for TOC translation
            system_prompt = f"""Translate the following table of contents entries from {self.source_language} to {self.target_language}.

Return ONLY the translations, one per line, in the same order.
Do not add numbering or explanations."""

            content = "\n".join(titles_to_translate)

            # Use online LLM for TOC (small content, fast)
            from .translator import TranslateProcessor
            temp_processor = TranslateProcessor(
                config=self.config,
                book_title=self.book_title,
                source_language=self.source_language,
                target_language=self.target_language,
                max_workers=1,
                resume=False
            )

            # Generate translation
            result = temp_processor.llm_client.generate_with_retry(
                prompt=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
                validator=None,
                max_retries=2
            )

            # Parse translations
            translated_lines = result.strip().split('\n')
            title_map = dict(zip(titles_to_translate, translated_lines))

            # Update TOC tree
            def update_titles(node):
                if isinstance(node, dict):
                    if "title" in node and node["title"] in title_map:
                        node["title"] = title_map[node["title"]]
                    if "children" in node:
                        for child in node["children"]:
                            update_titles(child)
                elif isinstance(node, list):
                    for item in node:
                        update_titles(item)

            update_titles(toc_tree)

            # Save translated TOC
            output_toc = Path("output") / self.book_title / "toc_tree_translated.json"
            with open(output_toc, 'w', encoding='utf-8') as f:
                json.dump(toc_tree, f, ensure_ascii=False, indent=2)

            logger.success(f"Saved translated TOC to {output_toc}")
            logger.success("TOC translation completed")

        except Exception as e:
            logger.error(f"Failed to translate TOC: {e}")

    def _resume_job(self, state: BatchTranslateState) -> Dict[str, Any]:
        """Resume an active job."""
        logger.info(f"Checking status of job: {state.active_job_name}")

        # Load metadata
        if not self.metadata_file.exists():
            logger.error("Metadata file not found, cannot resume")
            return {"total": 0, "completed": 0, "failed": 0}

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata_map = json.load(f)

        # Continue waiting for completion
        return self._wait_for_completion(state, metadata_map)
