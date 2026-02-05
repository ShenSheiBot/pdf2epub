"""
Agent-based content verification using Pydantic AI.

Provides intelligent verification of processed content (polish, translation, etc.)
by giving an LLM agent tools to inspect and judge content quality.
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


def load_config() -> dict:
    """Load config from config.yaml"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_verification_model():
    """
    Get the model for agent-based verification.

    Priority: Anthropic (Haiku 4.5) > POE (Gemini)

    Returns:
        Pydantic AI model
    """
    config = load_config()
    providers = config.get('credentials', {}).get('providers', {})

    # Try Anthropic first (Haiku for speed/cost)
    if 'anthropic' in providers:
        p = providers['anthropic']
        provider = AnthropicProvider(
            api_key=p.get('api_key'),
            base_url=p.get('base_url'),
        )
        model_name = 'claude-haiku-4-5-20251001'
        model = AnthropicModel(model_name, provider=provider)
        logger.info(f"Using {model_name} for verification")
        return model

    # Fallback to POE (Gemini)
    if 'poe' in providers:
        p = providers['poe']
        provider = OpenAIProvider(
            api_key=p.get('api_key'),
            base_url=p.get('base_url'),
        )
        model_name = 'Gemini-2.5-Flash'
        model = OpenAIChatModel(model_name, provider=provider)
        logger.info(f"Using {model_name} for verification")
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

    async def verify_async(self, file_keys: List[str]) -> List[VerificationResult]:
        """
        Verify files asynchronously.

        Args:
            file_keys: List of file keys to verify

        Returns:
            List of VerificationResult
        """
        import json

        state = VerificationState(
            files_to_verify=file_keys,
            task_type=self.task_type
        )

        result = await self.agent.run(
            f"Verify the following {len(file_keys)} files: {', '.join(file_keys)}",
            deps=state
        )

        # With output_type specified, output contains the structured data
        return result.output

    def verify(self, file_keys: List[str]) -> List[VerificationResult]:
        """
        Verify files (synchronous wrapper).

        Args:
            file_keys: List of file keys to verify

        Returns:
            List of VerificationResult
        """
        return asyncio.run(self.verify_async(file_keys))


class PolishVerificationAgent(AgentVerifier):
    """Agent for verifying polish results."""

    def __init__(self, tools: VerificationTools):
        super().__init__(tools, task_type="polish")

    def _get_system_prompt(self) -> str:
        return """You are a content verification expert for polish (formatting/cleanup) operations.

Your task: Verify if polish results are complete or truncated.

**Judging Criteria:**

ACCEPTABLE (status="complete"):
- Format transformations: table → list, deduplication, OCR error cleanup, standardization
- Content reorganization: paragraph merging, list formatting, heading cleanup
- Whitespace normalization, punctuation fixes
- The KEY is: all meaningful content is preserved, just reformatted

TRUNCATION (status="truncated"):
- Sentences cut off mid-way (not at punctuation)
- Paragraphs ending abruptly with incomplete thoughts
- Sudden stop without logical conclusion
- Missing significant content that should be there

**Tools Available:**
- get_stats(file_key): Get length ratios, metadata
- read_segment(file_key, source, start, length): Read any part
- compare_segments(file_key, position, length): Compare original vs processed
- detect_content_type(file_key): Detect if table/index/prose/etc

**Strategy Suggestions:**

1. Start with get_stats() to see the overall picture
2. Use detect_content_type() to understand what you're dealing with
3. Based on severity (length ratio) and type, decide how much to read:
   - Ratio >70%: Likely OK, check end only
   - Ratio 50-70%: Check start, end, maybe middle
   - Ratio <50%: More suspicious, check multiple points
4. For tables/indexes: Format changes are expected, focus on content preservation
5. For prose: Check logical flow and sentence completion

**Important:**
- Tables/indexes often have EXTREME length reductions (50-95%) due to format cleanup - this is NORMAL
  - Raw OCR text → clean markdown tables can result in 90%+ reduction
  - Example: verbose index "明石志津子 ... 76, 77, 80" → clean table row
- Focus on CONTENT preservation, NOT length ratios
- Only mark as truncated if you see:
  - Sentences cut off mid-word or mid-thought
  - Structural corruption (garbled text, broken tables)
  - Missing expected sections (e.g., index missing entire alphabetical sections)
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

Your task: Verify if translations are complete or truncated.

**Judging Criteria:**

COMPLETE:
- All source content has corresponding translation
- Translation ends at a logical point (end of paragraph, section, etc.)
- No mid-sentence cuts

TRUNCATED:
- Translation stops mid-sentence or mid-paragraph
- Missing sections that exist in the original
- Sudden stop without proper ending

**Tools Available:**
- get_stats(file_key): Get length ratios, metadata
- read_segment(file_key, source, start, length): Read any part of original or translation
- compare_segments(file_key, position, length): Compare original vs translation
- detect_content_type(file_key): Detect content type

**Strategy Suggestions:**

1. Get stats to see length ratio (translations may be longer or shorter)
2. Check the end of translation - does it end properly?
3. Compare start/middle/end segments to ensure coverage
4. For long files, sample multiple positions

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
