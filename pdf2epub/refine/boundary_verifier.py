"""
Boundary verification for TOC nodes.

Verifies that section titles appear on expected pages and extracts
the content before/after the title for precise page cutting.

DEPRECATED: This module has been replaced by boundary_agent.py which uses
a Pydantic AI agent for more flexible boundary verification, gap filling,
and overlap detection. This file is kept for reference and will be removed
in a future version.
"""

import warnings
warnings.warn(
    "boundary_verifier is deprecated, use boundary_agent instead",
    DeprecationWarning,
    stacklevel=2
)

from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from ..utils.common import parse_llm_json
from ..utils.llm_client import BoundLLMClient
from .toc_tree import TOCNode


class BoundaryVerifier:
    """
    Verifies boundaries for TOC nodes.

    Sends single page content to LLM and asks if the title is present,
    extracting content before and after the title.
    """

    def __init__(self, client: BoundLLMClient, model: str = "gemini-2.5-flash"):
        """
        Initialize the boundary verifier.

        Args:
            client: BoundLLMClient for API calls
            model: Model to use for verification (should be fast/cheap)
        """
        self.client = client
        self.model = model

    def verify_boundary(self, node: TOCNode, pages_dir: Path) -> Dict:
        """
        Verify the boundary for a single node.

        Sends only the start page content to LLM and asks:
        1. Is the title on this page?
        2. What content is before the title?
        3. What content is after the title?

        Args:
            node: TOCNode to verify
            pages_dir: Directory containing page_*.md files

        Returns:
            Dict with keys: found, title_at_start, content_before_title, content_after_title
        """
        # Read the start page
        page_file = pages_dir / f"page_{node.start_page:03d}.md"
        if not page_file.exists():
            logger.warning(f"Page file not found: {page_file}")
            return {"found": False, "error": "page_not_found"}

        page_content = page_file.read_text(encoding='utf-8')

        prompt = f"""
Find the title "{node.title}" in the following page content.

**Page Content (Page {node.start_page}):**
{page_content}

**Task:**
1. Determine if the title "{node.title}" is on this page
2. Determine if the title is at the start of the page (title_at_start)
3. If found and not at start, extract content before and after the title

Return JSON:
{{
    "found": bool,
    "title_at_start": bool,
    "content_before_title": string,
    "content_after_title": string
}}

**Important:**
- Use LENIENT matching. Return found=true if:
  - The page contains a title with roughly the same meaning
  - Minor differences are acceptable: OCR errors, punctuation, spacing, shortened forms, Roman numeral variations
  - Examples: "List of contributors" matches "Contributors", "73 Modern drama" matches "Modern drama", "PART II" matches "PART III" (OCR error)
- Only return found=false if:
  - No similar title exists on the page at all
  - The title meaning is completely different
- Chapter numbers are part of the title, not content_before_title
- If title_at_start is true, both content fields are empty strings
- If found is false, all content fields are empty strings
"""

        generation_config = self.client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        # Debug logging
        logger.debug(f"Verify request for '{node.title}' on page {node.start_page}")
        logger.debug(f"Page content length: {len(page_content)} chars")
        logger.debug(f"Prompt length: {len(prompt)} chars")

        try:
            response_text = self.client.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=generation_config,
                operation_name=f"Verify: {node.title}"
            )

            logger.debug(f"Response for '{node.title}': {response_text[:200]}...")
            result = parse_llm_json(response_text, operation_name=f"Verify: {node.title}")

            if result.get('found', False):
                logger.debug(f"Verified '{node.title}' on page {node.start_page}")
            else:
                logger.warning(f"Title '{node.title}' not found on page {node.start_page}")

            return result

        except Exception as e:
            logger.error(f"Error verifying '{node.title}': {e}")
            logger.error(f"Page content preview: {page_content[:500]}...")
            return {"found": False, "error": str(e)}

    def search_nearby_pages(
        self,
        node: TOCNode,
        pages_dir: Path,
        search_range: int = 5,
        prev_title: str = None,
        next_title: str = None
    ) -> Optional[int]:
        """
        Search for a title in nearby pages using LLM.

        Used when verification fails on the expected page. Uses a better model
        and provides comprehensive context for accurate search.

        Args:
            node: TOCNode to search for
            pages_dir: Directory containing page files
            search_range: How many pages to search in each direction
            prev_title: Title of previous section (for context)
            next_title: Title of next section (for context)

        Returns:
            Page number where title was found, or None
        """
        # Find total page count
        page_files = list(pages_dir.glob("page_*.md"))
        if not page_files:
            return None
        total_pages = len(page_files)

        # Collect all pages in search range
        pages_content = []
        start_search = max(1, node.start_page - search_range)
        end_search = min(total_pages, node.start_page + search_range)

        for page_num in range(start_search, end_search + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if page_file.exists():
                content = page_file.read_text(encoding='utf-8')
                # Truncate if too long
                if len(content) > 3000:
                    content = content[:3000] + "\n... [truncated]"
                pages_content.append(f"=== 第 {page_num} 页 ===\n{content}")

        if not pages_content:
            return None

        all_pages = "\n\n".join(pages_content)

        # Build context about surrounding sections
        context_info = ""
        if prev_title:
            context_info += f"- Previous section: {prev_title}\n"
        if next_title:
            context_info += f"- Next section: {next_title}\n"

        prompt = f"""Find which page contains the title "{node.title}".

**Context:**
- Expected page: {node.start_page}
- Search range: pages {start_search} to {end_search}
{context_info}

**Page Contents:**

{all_pages}

**Task:**
Find which page actually contains the title "{node.title}".

**Important:**
- Use LENIENT matching. Accept if:
  - The page contains a title with roughly the same meaning
  - Minor differences are acceptable: OCR errors, shortened forms, punctuation
  - Examples: "List of contributors" matches "Contributors"
- Only return found=false if no similar title exists in any page
- Chapter titles usually appear as markdown headings (# prefix) at page start

Return JSON:
{{
    "found": bool,
    "page_number": int or null,
    "confidence": "high" | "medium" | "low",
    "reason": string
}}
"""

        # Use a better model for this complex task
        generation_config = self.client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.client.generate_content_stream(
                self.model,
                prompt,
                generation_config,
                f"Search: {node.title}"
            )

            result = parse_llm_json(response_text, operation_name=f"Search: {node.title}")

            if result.get('found') and result.get('page_number'):
                page_num = result['page_number']
                confidence = result.get('confidence', 'unknown')
                reason = result.get('reason', '')
                logger.info(f"Found '{node.title}' on page {page_num} (confidence: {confidence})")
                logger.debug(f"Reason: {reason}")
                return page_num

            logger.warning(f"Could not find '{node.title}' in pages {start_search}-{end_search}")
            return None

        except Exception as e:
            logger.error(f"Error searching for '{node.title}': {e}")
            return None
