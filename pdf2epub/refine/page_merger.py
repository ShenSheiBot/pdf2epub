"""
Page merging with precise boundary cutting.

Merges page content for TOC nodes, using boundary_info to
precisely cut content at section boundaries.
"""

import re
from pathlib import Path
from typing import List
from loguru import logger

from .toc_tree import TOCNode


class PageMerger:
    """
    Merges pages for TOC nodes with precise boundary cutting.
    """

    def merge_node_content(
        self,
        node: TOCNode,
        pages_dir: Path,
        next_node: TOCNode = None
    ) -> str:
        """
        Merge page content for a node.

        Uses boundary_info to exclude content that doesn't belong to this node.

        Args:
            node: TOCNode to merge content for
            pages_dir: Directory containing page files
            next_node: Next sibling node (to get its content_before_title for end boundary)

        Returns:
            Merged content string
        """
        content_parts = []
        boundary = node.boundary_info or {}

        for page_num in range(node.start_page, node.end_page + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if not page_file.exists():
                logger.warning(f"Page file not found: {page_file}")
                continue

            page_content = page_file.read_text(encoding='utf-8')

            # Handle first page - remove content before title
            if page_num == node.start_page:
                content_before = boundary.get('content_before_title', '')
                if content_before:
                    page_content = self._remove_prefix(page_content, content_before)

            # Handle last page - remove content that belongs to next section
            if page_num == node.end_page and next_node:
                next_boundary = next_node.boundary_info or {}
                # If next section starts on the same page, use its content_before_title as suffix
                if next_node.start_page == node.end_page:
                    content_before_next = next_boundary.get('content_before_title', '')
                    if content_before_next:
                        page_content = self._remove_suffix(page_content, content_before_next)

            if page_content.strip():
                content_parts.append(page_content)

        return '\n\n'.join(content_parts)

    def merge_nodes_content(
        self,
        nodes: List[TOCNode],
        pages_dir: Path,
        next_node: TOCNode = None
    ) -> str:
        """
        Merge content for multiple consecutive nodes.

        Used when a parent node is treated as a single unit.

        Args:
            nodes: List of TOCNodes to merge
            pages_dir: Directory containing page files
            next_node: Next sibling node (to get its content_before_title for end boundary)

        Returns:
            Merged content string
        """
        if not nodes:
            return ""

        # Get the full page range
        start_page = nodes[0].start_page
        end_page = nodes[-1].end_page

        content_parts = []

        for page_num in range(start_page, end_page + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if not page_file.exists():
                continue

            page_content = page_file.read_text(encoding='utf-8')

            # Handle first page of first node
            if page_num == start_page and nodes[0].boundary_info:
                content_before = nodes[0].boundary_info.get('content_before_title', '')
                if content_before:
                    page_content = self._remove_prefix(page_content, content_before)

            # Handle last page - remove content that belongs to next section
            if page_num == end_page and next_node:
                next_boundary = next_node.boundary_info or {}
                if next_node.start_page == end_page:
                    content_before_next = next_boundary.get('content_before_title', '')
                    if content_before_next:
                        page_content = self._remove_suffix(page_content, content_before_next)

            if page_content.strip():
                content_parts.append(page_content)

        return '\n\n'.join(content_parts)

    def _remove_prefix(self, text: str, prefix: str) -> str:
        """
        Remove prefix content from text.

        Tries exact match first, then fuzzy match.

        Args:
            text: Full text
            prefix: Prefix to remove

        Returns:
            Text with prefix removed
        """
        if not prefix or not prefix.strip():
            return text

        # Try exact match first
        idx = text.find(prefix)
        if idx != -1:
            result = text[idx + len(prefix):].lstrip()
            logger.debug(f"Removed {len(prefix)} chars from start (exact match)")
            return result

        # Try fuzzy match - match first few words
        prefix_words = prefix.split()[:10]
        if prefix_words:
            # Build a pattern that matches the words with flexible whitespace
            pattern = r'\s*'.join(re.escape(w) for w in prefix_words)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Find the end of the prefix in the original text
                # Look for where the prefix content ends
                result = text[match.end():].lstrip()
                logger.debug(f"Removed prefix using fuzzy match")
                return result

        # If no match found, log warning and return original
        logger.warning(f"Could not find prefix to remove (first 50 chars): {prefix[:50]}...")
        return text

    def _remove_suffix(self, text: str, suffix: str) -> str:
        """
        Remove suffix content from text.

        Args:
            text: Full text
            suffix: Suffix to remove

        Returns:
            Text with suffix removed
        """
        if not suffix or not suffix.strip():
            return text

        # Try exact match
        idx = text.rfind(suffix)
        if idx != -1:
            result = text[:idx].rstrip()
            logger.debug(f"Removed {len(suffix)} chars from end")
            return result

        # Try fuzzy match
        suffix_words = suffix.split()[:10]
        if suffix_words:
            pattern = r'\s*'.join(re.escape(w) for w in suffix_words)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = text[:match.start()].rstrip()
                logger.debug(f"Removed suffix using fuzzy match")
                return result

        logger.warning(f"Could not find suffix to remove (first 50 chars): {suffix[:50]}...")
        return text
