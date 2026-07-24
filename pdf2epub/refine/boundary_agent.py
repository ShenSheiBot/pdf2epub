"""
Boundary verification agent using Pydantic AI.

Uses an LLM agent with tools to verify and adjust chapter boundaries.
Supports recursive verification of nested TOC structures.
"""

import json
import asyncio
import re
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from loguru import logger
import yaml
import tiktoken

from .toc_tree import TOCNode

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def load_config() -> dict:
    """Load config from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_model_and_limits(runtime_config: Optional[dict[str, Any]] = None):
    """Get the model for boundary verification and its token limits.

    Priority: explicit refine.agent > Anthropic (Haiku) > POE (Gemini)

    Returns:
        Tuple of (model, model_name, max_tokens)
    """
    config = runtime_config if runtime_config is not None else load_config()
    providers = config.get('credentials', {}).get('providers', {})
    model_limits = config.get('model_output_limits', {})
    default_limit = model_limits.get('_default', 4000)

    agent_config = config.get('refine', {}).get('agent', {})
    if agent_config:
        provider_name = agent_config.get('provider')
        model_name = agent_config.get('model')
        if not provider_name or not model_name:
            raise ValueError("refine.agent requires both provider and model")
        if provider_name not in providers:
            raise ValueError(
                f"refine.agent provider '{provider_name}' is not configured"
            )

        provider_config = providers[provider_name]
        provider_type = provider_config.get('type') or (
            'anthropic' if provider_name == 'anthropic' else 'openai'
        )
        if provider_type == 'anthropic':
            provider = AnthropicProvider(
                api_key=provider_config.get('api_key'),
                base_url=provider_config.get('base_url'),
            )
            model = AnthropicModel(model_name, provider=provider)
        elif provider_type == 'openai':
            provider = OpenAIProvider(
                api_key=provider_config.get('api_key'),
                base_url=provider_config.get('base_url'),
            )
            model = OpenAIChatModel(model_name, provider=provider)
        elif provider_type == 'google':
            from google.genai import Client
            from google.genai.types import HttpOptions
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            client_kwargs = {'api_key': provider_config.get('api_key')}
            if provider_config.get('base_url'):
                client_kwargs['http_options'] = HttpOptions(
                    base_url=provider_config['base_url']
                )
            client = Client(**client_kwargs)
            model = GoogleModel(
                model_name,
                provider=GoogleProvider(client=client),
            )
        else:
            raise ValueError(
                f"Unsupported refine.agent provider type '{provider_type}'"
            )

        max_tokens = model_limits.get(model_name, default_limit)
        return model, model_name, max_tokens

    # Try Anthropic first (Haiku for speed/cost)
    if 'anthropic' in providers:
        p = providers['anthropic']
        provider = AnthropicProvider(
            api_key=p.get('api_key'),
            base_url=p.get('base_url'),
        )
        model_name = 'claude-haiku-4-5-20251001'
        model = AnthropicModel(model_name, provider=provider)
        max_tokens = model_limits.get(model_name, default_limit)
        return model, model_name, max_tokens

    # Fallback to POE (OpenAI-compatible, can use Gemini-2.5-Flash)
    if 'poe' in providers:
        p = providers['poe']
        provider = OpenAIProvider(
            api_key=p.get('api_key'),
            base_url=p.get('base_url'),
        )
        model_name = 'Gemini-2.5-Flash'
        model = OpenAIChatModel(model_name, provider=provider)
        max_tokens = model_limits.get(model_name, default_limit)
        return model, model_name, max_tokens

    raise ValueError("No suitable provider found (need anthropic or poe)")


class Section(BaseModel):
    """A section in the TOC with boundary information."""
    title: str
    start_page: int
    end_page: int
    start_line: Optional[int] = None  # None = start of page
    end_line: Optional[int] = None    # None = end of page
    verified: bool = False
    children_count: int = 0
    estimated_tokens: int = 0
    original_index: Optional[int] = None  # Index in original children, None if inserted
    original_start_page: Optional[int] = None  # Original start page, for cumulative drift cap


# Hard cap on page adjustments to prevent catastrophic boundary moves
MAX_ADJUSTMENT_PAGES = 10


class ChapterState(BaseModel):
    """State for boundary verification of a chapter's sections."""
    sections: list[Section]
    pages_dir: str
    total_pages: int
    insert_rejections: int = 0  # Consecutive insert_section rejections
    # Ancestor and sibling labels may be visible near a page boundary, but they
    # must not be rediscovered as children of the current node.
    forbidden_insert_titles: list[str] = Field(default_factory=list)


def _normalize_title(title: str) -> str:
    """Normalize a heading enough to catch OCR spacing/punctuation variants."""
    return re.sub(r"[^\w]+", "", title.casefold(), flags=re.UNICODE)


def _titles_equivalent(left: str, right: str) -> bool:
    """Whether two headings are equal after OCR-neutral punctuation cleanup."""
    normalized_left = _normalize_title(left)
    normalized_right = _normalize_title(right)
    return bool(normalized_left and normalized_left == normalized_right)


def _title_supported_as_standalone_line(title: str, page_text: str) -> bool:
    """Match a proposed heading against complete OCR lines from any backend."""
    normalized_title = _normalize_title(title)
    if not normalized_title:
        return False
    for line in page_text.splitlines():
        candidate = line.strip().lstrip("#").strip()
        if _normalize_title(candidate) == normalized_title:
            return True
    return False


# Create the agent (lazy initialization)
_agent = None
_model_max_tokens = None  # Token limit from config
_agent_config_key = None
_model_limit_config_key = None


def _config_cache_key(runtime_config: Optional[dict[str, Any]]) -> object:
    """Keep lazy model caches scoped to the config passed for this run."""
    return id(runtime_config) if runtime_config is not None else "default-config"


def get_model_max_tokens(runtime_config: Optional[dict[str, Any]] = None) -> int:
    """Get the model's max token limit from config.

    This is used as the threshold for deciding whether to process children.
    """
    global _model_max_tokens, _model_limit_config_key
    config_key = _config_cache_key(runtime_config)
    if _model_max_tokens is None or _model_limit_config_key != config_key:
        _, _, _model_max_tokens = get_model_and_limits(runtime_config)
        _model_limit_config_key = config_key
    return _model_max_tokens


def get_agent(runtime_config: Optional[dict[str, Any]] = None):
    global _agent, _model_max_tokens, _agent_config_key, _model_limit_config_key
    config_key = _config_cache_key(runtime_config)
    if _agent is None or _agent_config_key != config_key:
        model, model_name, max_tokens = get_model_and_limits(runtime_config)
        _model_max_tokens = max_tokens
        _model_limit_config_key = config_key
        _agent_config_key = config_key
        logger.info(f"Using model {model_name} with max_tokens={max_tokens}")
        _agent = Agent(
            model,
            deps_type=ChapterState,
            system_prompt="""You are a boundary verification expert for book chapter segmentation.

Your task is to verify and fix section boundaries, ensuring complete coverage with no gaps or overlaps.

IMPORTANT: Be flexible with title matching!
- OCR is imperfect: titles may have wrong spaces ("L' ATTENTAT" vs "L'ATTENTAT"),
  accent errors ("é" vs "e"), or minor typos
- Container sections like "Première partie", "Huitième partie" often DON'T appear literally
  on the page - they're just TOC labels. The page may only show the first subsection's title.
- What matters is: does the PAGE CONTENT make sense as the START of this section?

Available tools:
- read_page: Read a page's content (with line numbers)
- adjust_start: Adjust a section's start page (automatically adjusts previous section's end)
- set_split: Set a split point within a page (for when a section starts mid-page)
- mark_verified: Mark a section as verified once you confirm the boundary is correct
- insert_section: Insert a new section when you find a gap with distinct content
- check_issues: Check for gaps/overlaps/missing splits/possible mid-sentence cuts - call before saying "Done"

Process:
1. For each unverified section, read its start_page
2. Check if the section title/heading appears on that page
3. If the title is at line 1, just mark_verified
4. If the title is NOT at line 1 (there's content before it), you MUST use set_split to set the exact line number where the section starts - this is critical for correct page merging!
5. If title is on wrong page, use adjust_start to fix it
6. If you find gaps between sections, read those pages and either:
   - Extend an adjacent section to cover the gap, OR
   - Insert a new section if there's a distinct chapter/heading
7. If a section seems wrong, verify it by reading the page — it may just have a variant title

Never use insert_section for the current parent, an ancestor, or a sibling
heading. Those headings may appear at the page boundary you inspect, but they
are not missing children.

IMPORTANT: When a section title appears mid-page (not line 1), you MUST call set_split before mark_verified. Otherwise the content before the title will be lost!

Before saying "Done", call check_issues to verify:
- No gaps between sections (missing pages)
- No overlaps (end_page > next start_page)
- No shared pages without split points defined
- No POSSIBLE SPLIT warnings (page starts with lowercase/non-heading text)
If issues remain, fix them or confirm they are intentional before finishing.
""",
        )

        # Register tools
        @_agent.tool
        def read_page(ctx: RunContext[ChapterState], page_num: int) -> str:
            """Read a page's content with line numbers.

            Args:
                page_num: The page number to read (1-indexed)

            Returns:
                The page content with line numbers, or error message if page doesn't exist.
            """
            logger.debug(f"Tool: read_page(page_num={page_num})")
            pages_dir = Path(ctx.deps.pages_dir)
            page_file = pages_dir / f"page_{page_num:03d}.md"

            if not page_file.exists():
                return f"Error: Page {page_num} does not exist (valid range: 1-{ctx.deps.total_pages})"

            content = page_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            numbered_lines = [f"{i+1:3d}| {line}" for i, line in enumerate(lines)]
            return '\n'.join(numbered_lines)

        @_agent.tool
        def adjust_start(ctx: RunContext[ChapterState], section_index: int, new_start_page: int) -> str:
            """Adjust a section's start page. Automatically adjusts the previous section's end page.

            Args:
                section_index: Index of the section to adjust (0-indexed)
                new_start_page: The new start page for this section

            Returns:
                Updated sections state as JSON.
            """
            logger.debug(f"Tool: adjust_start(section_index={section_index}, new_start_page={new_start_page})")
            sections = ctx.deps.sections

            if section_index < 0 or section_index >= len(sections):
                return f"Error: Invalid section index {section_index}"

            section = sections[section_index]
            old_start = section.start_page

            # Hard cap: reject adjustments larger than ±10 pages from ORIGINAL position
            # This prevents cumulative drift via chained adjustments
            reference = old_start if section.original_start_page is None else section.original_start_page
            delta = abs(new_start_page - reference)
            if delta > MAX_ADJUSTMENT_PAGES:
                logger.warning(
                    f"Rejected adjustment for '{section.title}': "
                    f"p{reference} (original) -> p{new_start_page} (delta={delta})"
                )
                return (
                    f"Error: Cumulative adjustment of {delta} pages from original "
                    f"position p{reference} is too large (max ±{MAX_ADJUSTMENT_PAGES}). "
                    f"Proposed: p{new_start_page}. "
                    f"Keep original start page if uncertain."
                )

            section.start_page = new_start_page
            section.start_line = None  # Clear any split
            section.verified = False   # Need to re-verify

            # Adjust previous section's end
            if section_index > 0:
                prev_section = sections[section_index - 1]
                prev_section.end_page = new_start_page - 1
                prev_section.end_line = None  # Clear any split

            logger.info(f"Adjusted '{section.title}' start: {old_start} -> {new_start_page}")
            return _format_sections(sections)

        @_agent.tool
        def set_split(ctx: RunContext[ChapterState], section_index: int, page_num: int, line_num: int) -> str:
            """Set a split point: this section starts at the given line on the given page.
            The previous section ends just before this line on the same page.

            Args:
                section_index: Index of the section that starts at this split point
                page_num: The page number where the split occurs
                line_num: The line number where this section starts (1-indexed)

            Returns:
                Updated sections state as JSON.
            """
            logger.debug(f"Tool: set_split(section_index={section_index}, page_num={page_num}, line_num={line_num})")
            sections = ctx.deps.sections

            if section_index < 0 or section_index >= len(sections):
                return f"Error: Invalid section index {section_index}"

            if section_index == 0:
                return "Error: Cannot split before the first section"

            section = sections[section_index]
            prev_section = sections[section_index - 1]

            # Hard cap: reject splits too far from section's ORIGINAL start page
            # This prevents cumulative drift via chained adjustments
            reference = section.start_page if section.original_start_page is None else section.original_start_page
            delta = abs(page_num - reference)
            if delta > MAX_ADJUSTMENT_PAGES:
                logger.warning(
                    f"Rejected split for '{section.title}': "
                    f"page {page_num} vs original start p{reference} (delta={delta})"
                )
                return (
                    f"Error: Split page {page_num} is too far from section's original "
                    f"start page {reference} (delta={delta} pages, "
                    f"max ±{MAX_ADJUSTMENT_PAGES}). "
                    f"Use adjust_start first if the section start needs correction."
                )

            # Set this section's start
            section.start_page = page_num
            section.start_line = line_num
            section.verified = False

            # Set previous section's end
            prev_section.end_page = page_num
            prev_section.end_line = line_num

            logger.info(f"Set split at page {page_num} line {line_num}: '{prev_section.title}' ends, '{section.title}' starts")
            return _format_sections(sections)

        @_agent.tool
        def mark_verified(ctx: RunContext[ChapterState], section_index: int) -> str:
            """Mark a section as verified after confirming its boundary is correct.

            Args:
                section_index: Index of the section to mark as verified

            Returns:
                Updated sections state as JSON.
            """
            logger.debug(f"Tool: mark_verified(section_index={section_index})")
            sections = ctx.deps.sections

            if section_index < 0 or section_index >= len(sections):
                return f"Error: Invalid section index {section_index}"

            section = sections[section_index]
            section.verified = True
            logger.info(f"Verified: '{section.title}' at page {section.start_page}")
            return _format_sections(sections)

        @_agent.tool
        def insert_section(ctx: RunContext[ChapterState], after_index: int, title: str, start_page: int, end_page: int) -> str:
            """Insert a new section after the specified index.

            Use when:
            - You discover a gap between sections that contains a distinct chapter/section
            - The TOC missed a section that clearly exists on the page

            Args:
                after_index: Insert after this section index (-1 to insert at beginning)
                title: Title of the new section (as it appears on the page)
                start_page: Start page of the new section
                end_page: End page of the new section

            Returns:
                Updated sections state as JSON.
            """
            logger.debug(f"Tool: insert_section(after_index={after_index}, title={title}, start_page={start_page}, end_page={end_page})")
            sections = ctx.deps.sections

            # Circuit breaker: disable after 3 consecutive rejections
            MAX_INSERT_REJECTIONS = 3
            if ctx.deps.insert_rejections >= MAX_INSERT_REJECTIONS:
                return (
                    "Error: insert_section is disabled after repeated rejections. "
                    "Focus on verifying existing sections with mark_verified."
                )

            if after_index < -1 or after_index >= len(sections):
                return f"Error: Invalid after_index {after_index}"

            prohibited_titles = (
                ctx.deps.forbidden_insert_titles
                + [section.title for section in sections]
            )
            matching_title = next(
                (known for known in prohibited_titles if _titles_equivalent(title, known)),
                None,
            )
            if matching_title is not None:
                ctx.deps.insert_rejections += 1
                remaining = MAX_INSERT_REJECTIONS - ctx.deps.insert_rejections
                return (
                    f"Error: '{title}' duplicates or aliases known heading "
                    f"'{matching_title}'. Do not insert parent, ancestor, sibling, "
                    f"or existing child headings as new sections. "
                    f"({remaining} attempts remaining before insert_section is disabled)"
                )

            page_file = Path(ctx.deps.pages_dir) / f"page_{start_page:03d}.md"
            if not page_file.exists():
                ctx.deps.insert_rejections += 1
                return f"Error: Start page {start_page} does not exist for inserted section '{title}'."
            page_text = page_file.read_text(encoding='utf-8')
            if not _title_supported_as_standalone_line(title, page_text):
                ctx.deps.insert_rejections += 1
                remaining = MAX_INSERT_REJECTIONS - ctx.deps.insert_rejections
                return (
                    f"Error: OCR page {start_page} has no standalone line "
                    f"matching '{title}'. Do not infer a new section from body "
                    f"prose. ({remaining} attempts remaining before "
                    "insert_section is disabled)"
                )

            # Validate: new section must fit in a gap (no overlap with existing sections)
            if after_index >= 0:
                prev = sections[after_index]
                if start_page <= prev.end_page:
                    ctx.deps.insert_rejections += 1
                    remaining = MAX_INSERT_REJECTIONS - ctx.deps.insert_rejections
                    return (
                        f"Error: Start page {start_page} overlaps with previous section "
                        f"'{prev.title}' ending at page {prev.end_page}. "
                        f"New sections can only fill gaps between existing sections. "
                        f"({remaining} attempts remaining before insert_section is disabled)"
                    )

            insert_pos = after_index + 1
            if insert_pos < len(sections):
                next_sec = sections[insert_pos]
                if end_page >= next_sec.start_page:
                    ctx.deps.insert_rejections += 1
                    remaining = MAX_INSERT_REJECTIONS - ctx.deps.insert_rejections
                    return (
                        f"Error: End page {end_page} overlaps with next section "
                        f"'{next_sec.title}' starting at page {next_sec.start_page}. "
                        f"New sections can only fill gaps between existing sections. "
                        f"({remaining} attempts remaining before insert_section is disabled)"
                    )

            # Success — reset rejection counter
            ctx.deps.insert_rejections = 0

            # Create new section (with drift anchor so cumulative cap applies)
            new_section = Section(
                title=title,
                start_page=start_page,
                end_page=end_page,
                verified=False,
                original_start_page=start_page,
            )

            # Tighten adjacent boundaries to abut the new section
            if after_index >= 0:
                sections[after_index].end_page = start_page - 1
                sections[after_index].end_line = None

            if insert_pos < len(sections):
                next_sec = sections[insert_pos]
                if next_sec.start_page == end_page + 2:
                    # Close 1-page gap between new section and next
                    next_sec.start_page = end_page + 1

            sections.insert(insert_pos, new_section)
            logger.info(f"Inserted section '{title}' at index {insert_pos} (p{start_page}-p{end_page})")
            return _format_sections(sections)

        @_agent.tool
        def check_issues(ctx: RunContext[ChapterState]) -> str:
            """Check for boundary issues (gaps, overlaps, missing splits).

            Call this before saying "Done" to verify everything is correct.

            Returns:
                List of issues found, or "No issues found" if clean.
            """
            logger.debug("Tool: check_issues()")
            sections = ctx.deps.sections
            pages_dir = Path(ctx.deps.pages_dir)
            issues = detect_boundary_issues(sections, pages_dir)

            if not issues:
                return "No issues found. All sections have valid boundaries."

            return f"Found {len(issues)} issues:\n" + "\n".join(f"- {issue}" for issue in issues)

    return _agent


def _sort_sections_by_page(sections: list[Section]) -> None:
    """Sort sections by start_page in place."""
    sections.sort(key=lambda s: (s.start_page, s.start_line or 0))


def _format_sections(sections: list[Section]) -> str:
    """Format sections as a readable JSON summary."""
    result = []
    for i, s in enumerate(sections):
        status = "✓" if s.verified else "?"
        start = f"p{s.start_page}"
        if s.start_line:
            start += f":L{s.start_line}"
        end = f"p{s.end_page}"
        if s.end_line:
            end += f":L{s.end_line}"
        result.append(f"  [{i}] {status} {s.title[:40]:<40} {start}-{end}")
    return "Current sections:\n" + "\n".join(result)


def detect_boundary_issues(sections: list[Section], pages_dir: Path = None) -> list[str]:
    """Detect gaps, overlaps, missing splits, and possible mid-sentence splits.

    Args:
        sections: List of sections to check
        pages_dir: Directory containing page files (optional, for content-based checks)

    Returns:
        List of issue descriptions, empty if no issues found.
    """
    issues = []

    # Check for invalid ranges (end_page < start_page)
    for i, s in enumerate(sections):
        if s.end_page < s.start_page:
            issues.append(
                f"INVALID RANGE: [{i}] '{s.title[:30]}' has end_page={s.end_page} < "
                f"start_page={s.start_page}. Use adjust_start to fix."
            )

    for i in range(len(sections) - 1):
        curr = sections[i]
        next_sec = sections[i + 1]

        # Check for gap: curr.end_page + 1 < next.start_page
        if curr.end_page + 1 < next_sec.start_page:
            gap_start = curr.end_page + 1
            gap_end = next_sec.start_page - 1
            issues.append(
                f"GAP: Pages {gap_start}-{gap_end} between "
                f"[{i}] '{curr.title[:30]}' (ends p{curr.end_page}) and "
                f"[{i+1}] '{next_sec.title[:30]}' (starts p{next_sec.start_page}). "
                f"Read these pages and either extend an adjacent section or insert_section if there's a distinct heading."
            )

        # Check for overlap: curr.end_page > next.start_page
        elif curr.end_page > next_sec.start_page:
            issues.append(
                f"OVERLAP: [{i}] '{curr.title[:30]}' ends p{curr.end_page} > "
                f"[{i+1}] '{next_sec.title[:30]}' starts p{next_sec.start_page}. "
                f"Use adjust_start or set_split to fix."
            )

        # Check for shared page without split
        elif curr.end_page == next_sec.start_page:
            if not curr.end_line and not next_sec.start_line:
                issues.append(
                    f"MISSING SPLIT: [{i}] '{curr.title[:30]}' and [{i+1}] '{next_sec.title[:30]}' "
                    f"share page {curr.end_page} but no split is defined. "
                    f"Read page {curr.end_page} and use set_split to define the boundary line."
                )

    # Check for possible mid-sentence splits (content-based)
    if pages_dir:
        for i, section in enumerate(sections):
            # Skip if start_line is already set
            if section.start_line:
                continue

            page_file = pages_dir / f"page_{section.start_page:03d}.md"
            if not page_file.exists():
                continue

            content = page_file.read_text(encoding='utf-8')
            first_line = content.split('\n')[0].strip() if content else ""

            # Check if first line looks like a heading
            if first_line.startswith('#'):
                continue  # Has markdown heading, OK

            # Check if first character is uppercase (A-Z)
            if first_line and first_line[0].isupper():
                continue  # Starts with uppercase, likely a title

            # First line doesn't start with # or uppercase - possible split problem
            if first_line:
                issues.append(
                    f"POSSIBLE SPLIT: [{i}] '{section.title[:30]}' starts at p{section.start_page} "
                    f"but first line doesn't look like a heading: \"{first_line[:50]}...\". "
                    f"Read the page and use set_split if the title appears later on the page."
                )

    return issues


def estimate_tokens(pages_dir: Path, start_page: int, end_page: int) -> int:
    """Estimate token count for a page range."""
    total = 0
    for page_num in range(start_page, end_page + 1):
        page_file = pages_dir / f"page_{page_num:03d}.md"
        if page_file.exists():
            content = page_file.read_text(encoding='utf-8')
            total += len(tokenizer.encode(content))
    return total


def toc_node_to_sections(node: TOCNode) -> list[Section]:
    """Convert TOCNode children to Section list for verification."""
    sections = []
    for i, child in enumerate(node.children):
        start_page = child.start_page
        end_page = child.end_page

        # Auto-fix invalid ranges
        if end_page < start_page:
            logger.warning(f"Fixed invalid range for '{child.title}': p{start_page}-p{end_page} -> p{start_page}-p{start_page}")
            end_page = start_page

        sections.append(Section(
            title=child.title,
            start_page=start_page,
            end_page=end_page,
            children_count=len(child.children),
            estimated_tokens=child.estimated_tokens,
            original_index=i,
            original_start_page=start_page,
        ))
    return sections


def sections_to_toc_children(sections: list[Section], original_children: list[TOCNode]) -> list[TOCNode]:
    """Update TOCNode children from verified sections.

    Handles:
    - Updated sections: sync start_page, end_page, boundary_info
    - Removed sections: skip (not in sections list)
    - Inserted sections: create new TOCNode
    """
    new_children = []

    for section in sections:
        if section.original_index is not None:
            # Existing section - update it
            child = original_children[section.original_index]
            child.start_page = section.start_page
            child.end_page = section.end_page
            # Always reset boundary_info to new format (clears old content_before_title etc.)
            if section.start_line or section.end_line:
                child.boundary_info = {
                    'start_line': section.start_line,
                    'end_line': section.end_line,
                }
            else:
                child.boundary_info = None  # Clear old format
            new_children.append(child)
        else:
            # Inserted section - create new TOCNode
            new_node = TOCNode(
                title=section.title,
                level=original_children[0].level if original_children else 1,
                start_page=section.start_page,
                end_page=section.end_page,
                children=[],
                gap_filled=True,  # Mark as inserted by agent
            )
            if section.start_line or section.end_line:
                new_node.boundary_info = {
                    'start_line': section.start_line,
                    'end_line': section.end_line,
                }
            new_children.append(new_node)

    return new_children


def _enforce_boundary_invariants(
    sections: list[Section],
    parent_start: int,
    parent_end: int,
) -> None:
    """Enforce structural invariants after agent verification.

    The boundary agent makes best-effort adjustments, but can leave structural
    issues: children outside parent bounds, end_page < start_page, last child
    not extending to parent's end, gaps between siblings. This function fixes
    them deterministically.

    Fixes:
    0. Clamp all children to parent bounds (no child outside parent range)
    1. end_page < start_page (logically impossible)
    2. First child's start_page should match parent's start_page
    3. Last child's end_page should match parent's end_page
    4. Gaps between consecutive sections (extend previous section)
    """
    if not sections:
        return

    # Fix 0: Clamp all children to parent bounds [parent_start, parent_end]
    for s in sections:
        clamped_start = max(parent_start, min(s.start_page, parent_end))
        clamped_end = max(parent_start, min(s.end_page, parent_end))
        if s.start_page != clamped_start:
            logger.warning(
                f"Boundary invariant: '{s.title}' start_page={s.start_page} "
                f"outside parent [{parent_start}, {parent_end}], clamping to {clamped_start}"
            )
            s.start_page = clamped_start
            s.start_line = None
        if s.end_page != clamped_end:
            logger.warning(
                f"Boundary invariant: '{s.title}' end_page={s.end_page} "
                f"outside parent [{parent_start}, {parent_end}], clamping to {clamped_end}"
            )
            s.end_page = clamped_end
            s.end_line = None

    # Fix 1: end_page >= start_page (may happen after clamping)
    for s in sections:
        if s.end_page < s.start_page:
            logger.warning(
                f"Boundary invariant: '{s.title}' end_page={s.end_page} < "
                f"start_page={s.start_page}, setting end_page={s.start_page}"
            )
            s.end_page = s.start_page
            s.end_line = None

    # Fix 2: First child should start at parent's start_page
    if sections[0].start_page > parent_start:
        logger.info(
            f"Boundary invariant: first child '{sections[0].title}' start_page "
            f"{sections[0].start_page} -> {parent_start} (parent start)"
        )
        sections[0].start_page = parent_start
        sections[0].start_line = None

    # Fix 3: Last child should end at parent's end_page
    if sections[-1].end_page < parent_end:
        logger.info(
            f"Boundary invariant: last child '{sections[-1].title}' end_page "
            f"{sections[-1].end_page} -> {parent_end} (parent end)"
        )
        sections[-1].end_page = parent_end
        sections[-1].end_line = None

    # Fix 4: Close gaps between consecutive sections
    for i in range(len(sections) - 1):
        curr = sections[i]
        next_sec = sections[i + 1]
        if curr.end_page + 1 < next_sec.start_page:
            old_end = curr.end_page
            curr.end_page = next_sec.start_page - 1
            curr.end_line = None
            logger.info(
                f"Boundary invariant: closed gap after '{curr.title}' "
                f"end_page {old_end} -> {curr.end_page}"
            )


async def verify_node_boundaries(
    node: TOCNode,
    pages_dir: Path,
    total_pages: int,
    max_retries: int = 3,
    runtime_config: Optional[dict[str, Any]] = None,
    forbidden_insert_titles: Optional[list[str]] = None,
) -> None:
    """Verify boundaries for a single node's children.

    Args:
        node: TOCNode whose children need verification
        pages_dir: Directory containing page_XXX.md files
        total_pages: Total number of pages in the book
        max_retries: Maximum retries for API errors
    """
    if not node.children:
        return

    sections = toc_node_to_sections(node)

    state = ChapterState(
        sections=sections,
        pages_dir=str(pages_dir),
        total_pages=total_pages,
        forbidden_insert_titles=forbidden_insert_titles or [],
    )

    logger.info(f"Verifying {len(sections)} children of '{node.title}'")
    logger.debug(_format_sections(sections))

    agent = get_agent(runtime_config)

    forbidden_titles_text = ""
    if state.forbidden_insert_titles:
        titles = "; ".join(state.forbidden_insert_titles)
        forbidden_titles_text = (
            "\nDo NOT insert a section with any parent, ancestor, or sibling "
            f"heading from this list: {titles}.\n"
        )

    # First round: normal verification
    await _run_agent_with_retries(
        agent, state,
        f"Verify the boundaries for these {len(sections)} sections. "
        f"Check each section's start_page to ensure the title appears there.\n\n"
        f"{_format_sections(state.sections)}{forbidden_titles_text}",
        max_retries,
        runtime_config,
    )

    # Check for issues after first round
    issues = detect_boundary_issues(state.sections, pages_dir)
    if issues:
        logger.warning(f"Found {len(issues)} boundary issues, running second round...")
        for issue in issues:
            logger.warning(f"  {issue}")

        # Second round: fix detected issues
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        await _run_agent_with_retries(
            agent, state,
            f"The following boundary issues were detected:\n\n{issues_text}\n\n"
            f"Please fix these issues. For each issue:\n"
            f"- GAP: Read the gap pages and either extend an adjacent section's end_page (adjust_start), or use insert_section if there's a distinct chapter/heading.\n"
            f"- OVERLAP: Use adjust_start or set_split to fix the overlap.\n"
            f"- MISSING SPLIT: Read the shared page and use set_split to define where one section ends and the next begins.\n"
            f"- POSSIBLE SPLIT: Read the page and check if the section title appears later on the page. If so, use set_split.\n\n"
            f"If an issue is intentional (e.g., blank pages, appendix), you can ignore it.\n\n"
            f"Current state:\n{_format_sections(state.sections)}{forbidden_titles_text}",
            max_retries,
            runtime_config,
        )
        # Log final state after second round
        final_issues = detect_boundary_issues(state.sections, pages_dir)
        if final_issues:
            logger.info(f"After second round, {len(final_issues)} issues remain (may be intentional)")

    # Sort by page, enforce invariants, then update the node's children
    _sort_sections_by_page(state.sections)
    parent_start = getattr(node, 'start_page', 1)
    parent_end = getattr(node, 'end_page', total_pages)
    _enforce_boundary_invariants(state.sections, parent_start, parent_end)
    node.children = sections_to_toc_children(state.sections, node.children)


async def _run_agent_with_retries(
    agent,
    state: ChapterState,
    prompt: str,
    max_retries: int,
    runtime_config: Optional[dict[str, Any]] = None,
):
    """Run agent with retry logic for transient errors."""
    config = runtime_config if runtime_config is not None else load_config()
    request_limit = config.get('refine', {}).get('agent_request_limit', 100)
    last_error = None
    for attempt in range(max_retries):
        try:
            await agent.run(prompt, deps=state, usage_limits=UsageLimits(request_limit=request_limit))
            return
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Check if it's a retryable error
            if any(term in error_msg for term in ['empty', 'timeout', 'connection', '429', '500', '502', '503', '504']):
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

            # Non-retryable error
            raise

    # All retries exhausted
    raise last_error


async def verify_toc_recursive(
    toc_tree: list[TOCNode],
    pages_dir: Path,
    total_pages: int,
    max_tokens: int = None,
    max_retries: int = 3,
    runtime_config: Optional[dict[str, Any]] = None,
) -> list[TOCNode]:
    """Recursively verify all boundaries in TOC tree.

    Process:
    1. Verify top-level chapters (sequential for stability)
    2. For each chapter, if tokens < max_tokens, remove children
    3. Otherwise, verify children (parallel within each level)
    4. Recurse into children

    Args:
        toc_tree: List of top-level TOCNodes
        pages_dir: Directory containing page_XXX.md files
        total_pages: Total number of pages
        max_tokens: Token threshold - nodes below this have children removed.
                    If None, uses the model's limit from config.yaml.
        max_retries: Maximum retries for API errors

    Returns:
        Updated TOC tree with verified boundaries
    """
    # Get max_tokens from config if not specified
    if max_tokens is None:
        max_tokens = get_model_max_tokens(runtime_config)
        logger.info(f"Using max_tokens={max_tokens} from config")

    # First estimate tokens for all nodes
    def estimate_all(nodes: list[TOCNode]):
        for node in nodes:
            node.estimated_tokens = estimate_tokens(pages_dir, node.start_page, node.end_page)
            estimate_all(node.children)

    estimate_all(toc_tree)

    # Create a virtual root to verify top-level chapters
    class VirtualRoot:
        def __init__(self, children):
            self.title = "ROOT"
            self.children = children

    root = VirtualRoot(toc_tree)

    # Step 1: Verify top-level chapters
    logger.info("=== Verifying top-level chapters ===")
    await verify_node_boundaries(
        root, pages_dir, total_pages, max_retries, runtime_config
    )
    # verify_node_boundaries replaces node.children with a new list (via
    # sections_to_toc_children), so root.children may differ from toc_tree
    # if sections were inserted. Capture the updated reference.
    toc_tree = root.children
    # Re-estimate tokens for any newly inserted chapters (gap_filled=True)
    for node in toc_tree:
        if not node.estimated_tokens:
            node.estimated_tokens = estimate_tokens(pages_dir, node.start_page, node.end_page)

    # Step 2: Process each chapter recursively
    async def process_node(
        node: TOCNode,
        depth: int = 1,
        ancestor_titles: Optional[list[str]] = None,
        sibling_nodes: Optional[list[TOCNode]] = None,
    ):
        indent = "  " * depth
        ancestor_titles = ancestor_titles or []
        sibling_nodes = sibling_nodes or []

        # Check if below token threshold
        if node.estimated_tokens < max_tokens:
            if node.children:
                logger.info(f"{indent}'{node.title}' has {node.estimated_tokens} tokens < {max_tokens}, removing {len(node.children)} children")
                node.children = []
            return

        # Has children and above threshold - verify them
        if node.children:
            logger.info(f"{indent}=== Verifying children of '{node.title}' ({node.estimated_tokens} tokens) ===")
            forbidden_titles = [
                *ancestor_titles,
                node.title,
                *(sibling.title for sibling in sibling_nodes if sibling is not node),
            ]
            await verify_node_boundaries(
                node,
                pages_dir,
                total_pages,
                max_retries,
                runtime_config,
                forbidden_insert_titles=forbidden_titles,
            )

            # Recurse into children (parallel)
            tasks = [
                process_node(
                    child,
                    depth + 1,
                    ancestor_titles=[*ancestor_titles, node.title],
                    sibling_nodes=node.children,
                )
                for child in node.children
            ]
            await asyncio.gather(*tasks)

    # Process all top-level chapters in parallel
    tasks = [process_node(chapter, sibling_nodes=toc_tree) for chapter in toc_tree]
    await asyncio.gather(*tasks)

    return toc_tree


def load_toc_tree(toc_path: Path) -> tuple[list[TOCNode], dict]:
    """Load TOC tree from JSON file.

    Returns:
        Tuple of (toc_tree, metadata)
    """
    with open(toc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chapters = data.get('chapters', [])
    metadata = {k: v for k, v in data.items() if k != 'chapters'}

    toc_tree = [TOCNode.from_dict(ch) for ch in chapters]
    return toc_tree, metadata


def save_toc_tree(toc_tree: list[TOCNode], metadata: dict, toc_path: Path):
    """Save TOC tree to JSON file."""
    data = {
        **metadata,
        'chapters': [node.to_dict() for node in toc_tree]
    }
    with open(toc_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# CLI entry point for testing
if __name__ == "__main__":
    import sys
    from pdf2epub.utils.logging_config import configure_logging

    configure_logging()

    async def main():
        output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/Boniface VIII - Un pape hérétique")
        toc_path = output_dir / "toc_tree_original.json"
        pages_dir = output_dir / "pages"

        # Count pages
        total_pages = len(list(pages_dir.glob("page_*.md")))

        # Load TOC
        toc_tree, metadata = load_toc_tree(toc_path)

        # Verify recursively (uses max_tokens from config)
        toc_tree = await verify_toc_recursive(
            toc_tree, pages_dir, total_pages,
        )

        # Save result
        output_path = output_dir / "toc_tree_verified.json"
        save_toc_tree(toc_tree, metadata, output_path)

        print(f"\n=== Verification Complete ===")
        print(f"Saved to: {output_path}")

        # Print summary
        def count_nodes(nodes):
            total = len(nodes)
            for n in nodes:
                total += count_nodes(n.children)
            return total

        print(f"Total nodes: {count_nodes(toc_tree)}")

    asyncio.run(main())
