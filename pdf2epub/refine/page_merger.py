"""
Page merging with precise boundary cutting.

Merges page content for TOC nodes, using boundary_info (start_line/end_line)
to precisely cut content at section boundaries.
"""

from pathlib import Path
from typing import List
from loguru import logger

from .toc_tree import TOCNode


class PageMerger:
    """
    Merges pages for TOC nodes with precise line-based boundary cutting.

    Uses start_line/end_line from boundary_info to handle mid-page splits.
    """

    def merge_node_content(
        self,
        node: TOCNode,
        pages_dir: Path,
        next_node: TOCNode = None
    ) -> str:
        """
        Merge page content for a node.

        Uses boundary_info.start_line/end_line for precise cutting when
        sections share a page.

        Args:
            node: TOCNode to merge content for
            pages_dir: Directory containing page files
            next_node: Next sibling node (to get its start_line for end boundary)

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
            lines = page_content.split('\n')

            # Handle first page - start from start_line if set
            if page_num == node.start_page:
                start_line = boundary.get('start_line')
                if start_line is not None and start_line > 1:
                    # start_line is 1-indexed, so we slice from start_line-1
                    lines = lines[start_line - 1:]
                    logger.debug(f"Node '{node.title}' starts at line {start_line}")

            # Handle last page - end at end_line if set, or at next_node's start_line
            if page_num == node.end_page:
                end_line = boundary.get('end_line')
                if end_line is not None:
                    # end_line is 1-indexed, we want lines before this line
                    lines = lines[:end_line - 1]
                    logger.debug(f"Node '{node.title}' ends at line {end_line}")
                elif next_node and next_node.start_page == node.end_page:
                    # Next section starts on same page - cut before it
                    next_boundary = next_node.boundary_info or {}
                    next_start_line = next_boundary.get('start_line')
                    if next_start_line is not None:
                        lines = lines[:next_start_line - 1]
                        logger.debug(f"Cutting before next section at line {next_start_line}")

            page_content = '\n'.join(lines)
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
            next_node: Next sibling node (to get its start_line for end boundary)

        Returns:
            Merged content string
        """
        if not nodes:
            return ""

        # Get the full page range
        start_page = nodes[0].start_page
        end_page = nodes[-1].end_page
        first_boundary = nodes[0].boundary_info or {}

        content_parts = []

        for page_num in range(start_page, end_page + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if not page_file.exists():
                continue

            page_content = page_file.read_text(encoding='utf-8')
            lines = page_content.split('\n')

            # Handle first page of first node
            if page_num == start_page:
                start_line = first_boundary.get('start_line')
                if start_line is not None and start_line > 1:
                    lines = lines[start_line - 1:]

            # Handle last page - end at next_node's start_line if on same page
            if page_num == end_page and next_node and next_node.start_page == end_page:
                next_boundary = next_node.boundary_info or {}
                next_start_line = next_boundary.get('start_line')
                if next_start_line is not None:
                    lines = lines[:next_start_line - 1]

            page_content = '\n'.join(lines)
            if page_content.strip():
                content_parts.append(page_content)

        return '\n\n'.join(content_parts)
