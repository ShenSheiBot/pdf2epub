"""
Agent-based content verification using Pydantic AI.

Provides intelligent verification of processed content (polish, translation, etc.)
by giving an LLM agent tools to inspect and judge content quality.

Implements BatchValidator protocol for use with new validation architecture.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from loguru import logger
import yaml

from .verification_tools import VerificationTools, VerificationFile
from .._protocol import ValidationResult as CoreValidationResult


def load_config() -> dict:
    """Load config from config.yaml"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Default model names (used when not specified in config)
DEFAULT_ANTHROPIC_MODEL = 'claude-haiku-4-5-20251001'
DEFAULT_POE_MODEL = 'Gemini-2.5-Flash'


def get_verification_model():
    """
    Get the model for agent-based verification.

    Reads from config.yaml:
    - batch.verification.provider and batch.verification.model (if specified)
    - Falls back to anthropic > poe with default models

    Returns:
        Pydantic AI model
    """
    config = load_config()
    providers = config.get('credentials', {}).get('providers', {})

    # Check for explicit verification config
    batch_config = config.get('batch', {})
    verification_config = batch_config.get('verification', {})
    configured_provider = verification_config.get('provider')
    configured_model = verification_config.get('model')

    # If explicit config exists, use it
    if configured_provider and configured_model:
        if configured_provider in providers:
            p = providers[configured_provider]
            # Determine provider type from config
            provider_type = p.get('type', 'openai')
            if provider_type == 'anthropic':
                provider = AnthropicProvider(
                    api_key=p.get('api_key'),
                    base_url=p.get('base_url'),
                )
                model = AnthropicModel(configured_model, provider=provider)
            else:
                provider = OpenAIProvider(
                    api_key=p.get('api_key'),
                    base_url=p.get('base_url'),
                )
                model = OpenAIChatModel(configured_model, provider=provider)
            logger.info(f"Using {configured_model} for verification (from config)")
            return model

    # Fallback: Try Anthropic first (Haiku for speed/cost)
    if 'anthropic' in providers:
        p = providers['anthropic']
        provider = AnthropicProvider(
            api_key=p.get('api_key'),
            base_url=p.get('base_url'),
        )
        model_name = DEFAULT_ANTHROPIC_MODEL
        model = AnthropicModel(model_name, provider=provider)
        logger.info(f"Using {model_name} for verification (default)")
        return model

    # Fallback to POE (Gemini)
    if 'poe' in providers:
        p = providers['poe']
        provider = OpenAIProvider(
            api_key=p.get('api_key'),
            base_url=p.get('base_url'),
        )
        model_name = DEFAULT_POE_MODEL
        model = OpenAIChatModel(model_name, provider=provider)
        logger.info(f"Using {model_name} for verification (default)")
        return model

    raise ValueError("No suitable provider found in config.yaml (need anthropic or poe)")


class VerificationResult(BaseModel):
    """Result of verification for a single file."""
    file_key: str
    status: str  # "complete" or "truncated"
    reason: str  # Brief explanation
    confidence: str  # "high", "medium", "low"


class VerificationState(BaseModel):
    """State for verification agent."""
    files_to_verify: List[str]  # List of file keys
    task_type: str  # "polish" or "translate"


class AgentVerifier:
    """
    Base class for agent-based verification.

    Subclasses define task-specific prompts and logic.
    """

    def __init__(
        self,
        tools: VerificationTools,
        task_type: str = "polish"
    ):
        """
        Initialize verifier.

        Args:
            tools: VerificationTools instance with files to verify
            task_type: Type of verification ("polish" or "translate")
        """
        self.tools = tools
        self.task_type = task_type
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent with tools."""
        model = get_verification_model()

        agent = Agent(
            model,
            output_type=List[VerificationResult],
            deps_type=VerificationState,
            system_prompt=self._get_system_prompt()
        )

        # Register tools
        @agent.tool
        def read_segment(
            ctx: RunContext[VerificationState],
            file_key: str,
            source: str = "processed",
            start: int = 0,
            length: int = 1000
        ) -> str:
            """
            Read a segment of content from a file.

            Args:
                file_key: File identifier
                source: "original" or "processed"
                start: Starting character position
                length: Number of characters to read

            Returns:
                Content segment with line numbers
            """
            return self.tools.read_segment(file_key, source, start, length)

        @agent.tool
        def get_stats(ctx: RunContext[VerificationState], file_key: str) -> Dict:
            """
            Get statistics about a file.

            Args:
                file_key: File identifier

            Returns:
                Dict with length, ratio, and metadata
            """
            return self.tools.get_stats(file_key)

        @agent.tool
        def compare_segments(
            ctx: RunContext[VerificationState],
            file_key: str,
            position: str = "end",
            length: int = 500
        ) -> Dict:
            """
            Compare corresponding segments from original and processed.

            Args:
                file_key: File identifier
                position: "start", "middle", or "end"
                length: Length of segment to compare

            Returns:
                Dict with both segments
            """
            return self.tools.compare_segments(file_key, position, length)

        @agent.tool
        def detect_content_type(ctx: RunContext[VerificationState], file_key: str) -> str:
            """
            Detect the type of content (table, index, toc, prose, list).

            Args:
                file_key: File identifier

            Returns:
                Content type string
            """
            return self.tools.detect_content_type(file_key)

        return agent

    def _get_system_prompt(self) -> str:
        """Get system prompt (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement _get_system_prompt")

    async def verify_async(self, file_keys: List[str], batch_size: int = 15) -> List[VerificationResult]:
        """
        Verify files asynchronously in batches to avoid context overflow.

        Args:
            file_keys: List of file keys to verify
            batch_size: Max files per agent call (default 15 to stay within context limits)

        Returns:
            List of VerificationResult
        """
        all_results = []
        total_batches = (len(file_keys) + batch_size - 1) // batch_size

        # Process in batches to avoid context overflow
        for batch_idx, i in enumerate(range(0, len(file_keys), batch_size)):
            batch_keys = file_keys[i:i + batch_size]
            logger.info(f"Verifying batch {batch_idx + 1}/{total_batches}: {len(batch_keys)} files")

            state = VerificationState(
                files_to_verify=batch_keys,
                task_type=self.task_type
            )

            try:
                result = await self.agent.run(
                    f"Verify the following {len(batch_keys)} files: {', '.join(batch_keys)}. "
                    f"Be concise - only use tools when necessary, summarize findings briefly.",
                    deps=state
                )
                all_results.extend(result.output)
            except Exception as e:
                # If batch fails (e.g., context overflow), try smaller batches
                logger.warning(f"Batch {batch_idx + 1} failed: {e}, trying smaller batches")
                if batch_size > 5:
                    # Retry with smaller batch size
                    sub_results = await self.verify_async(batch_keys, batch_size=batch_size // 2)
                    all_results.extend(sub_results)
                else:
                    # Fail fast: mark as truncated so they get retried on --resume
                    logger.error(f"Verification failed for batch, marking as truncated: {batch_keys}")
                    for key in batch_keys:
                        all_results.append(VerificationResult(
                            file_key=key,
                            status="truncated",
                            reason=f"Verification agent error: {str(e)[:100]}",
                            confidence="low"
                        ))

        return all_results

    def verify(self, file_keys: List[str], batch_size: int = 15) -> List[VerificationResult]:
        """
        Verify files (synchronous wrapper).

        Args:
            file_keys: List of file keys to verify
            batch_size: Max files per agent call

        Returns:
            List of VerificationResult
        """
        return asyncio.run(self.verify_async(file_keys, batch_size))


class PolishVerificationAgent(AgentVerifier):
    """Agent for verifying polish results."""

    def __init__(self, tools: VerificationTools):
        super().__init__(tools, task_type="polish")

    def _get_system_prompt(self) -> str:
        return """You are a content verification expert for polish (formatting/cleanup) operations.

Your task: Verify if polish results are complete or truncated.

**CRITICAL: Split Files (.partN)**

Files with names like `chapter_9.part2` are SPLIT FILES - they are INTENTIONALLY partial:
- Split files contain only a PORTION of the original chapter (e.g., part2 of 3)
- They are SUPPOSED to start mid-chapter and end mid-chapter
- Do NOT compare them against the full original chapter
- Only verify the content WITHIN the split is complete (no mid-sentence cuts, proper formatting)
- A split file starting with a section heading like "## Boys' Love" is NORMAL - it's where the split was made
- A split file ending before the chapter conclusion is NORMAL - the conclusion is in the next part

**Judging Criteria:**

ACCEPTABLE (status="complete"):
- Format transformations: table → list, deduplication, OCR error cleanup, standardization
- Content reorganization: paragraph merging, list formatting, heading cleanup
- Whitespace normalization, punctuation fixes
- The KEY is: all meaningful content within THIS file/split is preserved, just reformatted
- Split files that start/end at logical section boundaries

TRUNCATION (status="truncated"):
- Sentences cut off mid-way (not at punctuation)
- Paragraphs ending abruptly with incomplete thoughts
- Sudden stop WITHOUT logical section boundary
- Missing content that should be WITHIN this specific file/split

**Tools Available:**
- get_stats(file_key): Get length ratios, metadata
- read_segment(file_key, source, start, length): Read any part
- compare_segments(file_key, position, length): Compare original vs processed
- detect_content_type(file_key): Detect if table/index/prose/etc

**Strategy Suggestions:**

1. Start with get_stats() to see the overall picture
2. Check if file_key contains ".part" - if so, it's a split file, be lenient about missing start/end content
3. Use detect_content_type() to understand what you're dealing with
4. Based on severity (length ratio) and type, decide how much to read:
   - Ratio >70%: Likely OK, check end only
   - Ratio 50-70%: Check start, end, maybe middle
   - Ratio <50%: More suspicious, check multiple points
5. For tables/indexes: Format changes are expected, focus on content preservation
6. For prose: Check logical flow and sentence completion WITHIN the file

**Important:**
- Tables/indexes often have EXTREME length reductions (50-95%) due to format cleanup - this is NORMAL
  - Raw OCR text → clean markdown tables can result in 90%+ reduction
  - Example: verbose index "明石志津子 ... 76, 77, 80" → clean table row
- Focus on CONTENT preservation, NOT length ratios
- Only mark as truncated if you see:
  - Sentences cut off mid-word or mid-thought
  - Structural corruption (garbled text, broken tables)
  - Missing expected sections WITHIN this specific file (not in other parts)
- When uncertain, read more segments to be sure

**Output Format:**
Return a list of VerificationResult with:
- file_key: The file identifier
- status: "complete" or "truncated"
- reason: One sentence explaining your judgment
- confidence: "high", "medium", or "low"
"""


class TranslationVerificationAgent(AgentVerifier):
    """Agent for verifying translation results."""

    def __init__(self, tools: VerificationTools):
        super().__init__(tools, task_type="translate")

    def _get_system_prompt(self) -> str:
        return """You are a translation completeness verification expert.

Your task: Verify if translations are complete and correctly formatted.

**CRITICAL: Bilingual Output Detection**

A common translation error is producing BILINGUAL output instead of pure translation:
- The model outputs Japanese original paragraph, then Chinese translation paragraph, alternating
- This results in output length ~150-200% of input (nearly double)
- Look for Japanese text (hiragana ぁ-ん, katakana ァ-ン) OUTSIDE of quote markers (> or 〈〉)
- Japanese inside quotes (> 〈...〉) is ACCEPTABLE - these are intentional source quotes
- Japanese OUTSIDE quotes in the body text is a BILINGUAL FORMAT ERROR

If output ratio > 150% AND you see Japanese outside quotes, mark as "truncated" with reason "bilingual format error".

**CRITICAL: Split Files (.partN)**

Files with names like `chapter_9.part2` are SPLIT FILES - they are INTENTIONALLY partial:
- Split files contain only a PORTION of the original chapter (e.g., part2 of 3)
- They are SUPPOSED to start mid-chapter and end mid-chapter
- Do NOT expect them to contain the full chapter's beginning or ending
- Only verify the content WITHIN the split is fully translated (no mid-sentence cuts)
- A split file starting with a section heading is NORMAL - it's where the split was made

**Judging Criteria:**

COMPLETE:
- Translation is in TARGET LANGUAGE ONLY (Chinese), except for intentional quotes
- All source content WITHIN THIS FILE has corresponding translation
- Translation ends at a logical point (end of paragraph, section, etc.)
- No mid-sentence cuts

TRUNCATED (or ERROR):
- BILINGUAL FORMAT: Source language paragraphs mixed with translation paragraphs
- Translation stops mid-sentence or mid-paragraph
- Missing sections that exist in the original FILE (not other parts)
- Sudden stop without proper ending WITHIN this file

**Tools Available:**
- get_stats(file_key): Get length ratios, metadata
- read_segment(file_key, source, start, length): Read any part of original or translation
- compare_segments(file_key, position, length): Compare original vs translation
- detect_content_type(file_key): Detect content type

**Strategy Suggestions:**

1. Get stats to see length ratio - if >150%, check for bilingual format
2. Read segments of the TRANSLATION output, look for Japanese outside quotes
3. Check if file_key contains ".part" - if so, it's a split file, be lenient on start/end
4. Check the end of translation - does it end properly?
5. Compare start/middle/end segments to ensure coverage WITHIN this file

**Output Format:**
Return a list of VerificationResult with:
- file_key: The file identifier
- status: "complete" or "truncated"
- reason: One sentence explaining your judgment
- confidence: "high", "medium", or "low"
"""


def verify_batch(
    files: Dict[str, VerificationFile],
    task_type: str = "polish"
) -> List[VerificationResult]:
    """
    Convenience function to verify a batch of files.

    Args:
        files: Dict mapping file_key to VerificationFile
        task_type: "polish" or "translate"

    Returns:
        List of VerificationResult
    """
    tools = VerificationTools(files)

    if task_type == "polish":
        verifier = PolishVerificationAgent(tools)
    elif task_type == "translate":
        verifier = TranslationVerificationAgent(tools)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    file_keys = list(files.keys())
    logger.info(f"Verifying {len(file_keys)} files with agent-based verification")

    return verifier.verify(file_keys)


# === BatchValidator Protocol Implementations ===


class PolishBatchValidator:
    """
    Batch validator for polish results.

    Implements BatchValidator protocol for use with new validation architecture.
    Uses batched verification to avoid context overflow.
    """

    def __init__(self, batch_size: int = 15):
        """
        Initialize validator.

        Args:
            batch_size: Max files per agent call (default 15)
        """
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "PolishBatchValidator"

    def validate_batch(
        self,
        files: Dict[str, VerificationFile]
    ) -> Dict[str, CoreValidationResult]:
        """
        Validate multiple polish results in batch.

        Args:
            files: Dict mapping file_key to VerificationFile(original, processed)

        Returns:
            Dict mapping file_key to ValidationResult
        """
        if not files:
            return {}

        tools = VerificationTools(files)
        verifier = PolishVerificationAgent(tools)
        file_keys = list(files.keys())

        logger.info(f"PolishBatchValidator: Verifying {len(file_keys)} files (batch_size={self._batch_size})")

        results = verifier.verify(file_keys, batch_size=self._batch_size)

        # Convert VerificationResult to CoreValidationResult
        return {
            r.file_key: CoreValidationResult(
                key=r.file_key,
                is_valid=(r.status == "complete"),
                reason=r.reason,
                confidence=r.confidence
            )
            for r in results
        }


class TranslationBatchValidator:
    """
    Batch validator for translation results.

    Implements BatchValidator protocol for use with new validation architecture.
    Uses batched verification to avoid context overflow.
    """

    def __init__(self, batch_size: int = 15):
        """
        Initialize validator.

        Args:
            batch_size: Max files per agent call (default 15)
        """
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "TranslationBatchValidator"

    def validate_batch(
        self,
        files: Dict[str, VerificationFile]
    ) -> Dict[str, CoreValidationResult]:
        """
        Validate multiple translation results in batch.

        Args:
            files: Dict mapping file_key to VerificationFile(original, processed)

        Returns:
            Dict mapping file_key to ValidationResult
        """
        if not files:
            return {}

        tools = VerificationTools(files)
        verifier = TranslationVerificationAgent(tools)
        file_keys = list(files.keys())

        logger.info(f"TranslationBatchValidator: Verifying {len(file_keys)} files (batch_size={self._batch_size})")

        results = verifier.verify(file_keys, batch_size=self._batch_size)

        # Convert VerificationResult to CoreValidationResult
        return {
            r.file_key: CoreValidationResult(
                key=r.file_key,
                is_valid=(r.status == "complete"),
                reason=r.reason,
                confidence=r.confidence
            )
            for r in results
        }
