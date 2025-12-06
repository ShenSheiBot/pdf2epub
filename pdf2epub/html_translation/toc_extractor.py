"""
TOC (Table of Contents) Extractor Module

This module extracts and parses TOC from EPUB files.
Supports:
- EPUB 2.0 NCX (toc.ncx)
- EPUB 3.x Navigation Document (nav.xhtml)
- Fallback: Extract from spine structure

Handles multi-level hierarchical TOC and anchor links.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from .epub_parser import EPUBParser


class TOCEntry:
    """Represents a single TOC entry with hierarchy."""

    def __init__(
        self,
        title: str,
        href: str,
        level: int = 1,
        anchor: Optional[str] = None
    ):
        """
        Initialize a TOC entry.

        Args:
            title: Display title
            href: File path (e.g., 'Text/chapter1.xhtml')
            level: Hierarchy level (1-6)
            anchor: Anchor ID (from #anchor in href)
        """
        self.title = title
        self.href = href
        self.level = level
        self.anchor = anchor
        self.children: List[TOCEntry] = []

    def add_child(self, child: 'TOCEntry'):
        """Add a child entry."""
        self.children.append(child)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'href': self.href,
            'level': self.level,
            'anchor': self.anchor,
            'children': [child.to_dict() for child in self.children]
        }

    def flatten(self) -> List[Dict]:
        """Flatten hierarchy to list."""
        result = [self.to_dict()]
        for child in self.children:
            result.extend(child.flatten())
        return result


class TOCExtractor:
    """Extract TOC from EPUB files."""

    def __init__(self, parser: 'EPUBParser'):
        """
        Initialize TOC extractor.

        Args:
            parser: EPUBParser instance
        """
        self.parser = parser
        self._toc_entries = None

    @property
    def toc(self) -> List[TOCEntry]:
        """
        Get TOC entries (cached).

        Returns:
            List of TOCEntry objects (hierarchical)
        """
        if self._toc_entries is None:
            self._toc_entries = self._extract_toc()
        return self._toc_entries

    def _extract_toc(self) -> List[TOCEntry]:
        """
        Extract TOC using EPUBParser.

        Returns:
            List of TOCEntry objects
        """
        raw_toc = self.parser.toc
        if raw_toc:
            entries = self._convert_toc_entries(raw_toc)
            if entries:
                logger.info(f"Extracted {len(entries)} TOC entries")
                return entries

        # Fallback to spine
        logger.warning("No TOC found, using spine fallback")
        return self._fallback_spine_toc()

    def _convert_toc_entries(self, raw_toc: List[Dict], level: int = 1) -> List[TOCEntry]:
        """
        Convert raw TOC from EPUBParser to TOCEntry objects.

        Args:
            raw_toc: List of dicts with title, href, children
            level: Current hierarchy level

        Returns:
            List of TOCEntry objects
        """
        entries = []

        for item in raw_toc:
            href, anchor = self._split_href_anchor(item.get('href', ''))

            entry = TOCEntry(
                title=item.get('title', 'Untitled'),
                href=href,
                level=level,
                anchor=anchor
            )

            # Process children recursively
            children = item.get('children', [])
            if children:
                entry.children = self._convert_toc_entries(children, level + 1)

            entries.append(entry)

        return entries

    def _fallback_spine_toc(self) -> List[TOCEntry]:
        """
        Fallback: Create TOC from spine structure.
        Assigns generic titles like "Chapter 1", "Chapter 2".

        Returns:
            List of TOCEntry objects
        """
        entries = []
        chapter_num = 1

        for item in self.parser.spine:
            href = item.get('href', '')

            # Skip cover, titlepage, etc.
            name_lower = href.lower()
            if any(skip in name_lower for skip in ['cover', 'title', 'copy', 'toc']):
                continue

            title = f"Chapter {chapter_num}"
            entry = TOCEntry(
                title=title,
                href=href,
                level=1,
                anchor=None
            )
            entries.append(entry)
            chapter_num += 1

        logger.warning(f"Generated {len(entries)} generic TOC entries from spine")
        return entries

    def _split_href_anchor(self, href: str) -> Tuple[str, Optional[str]]:
        """
        Split href into (path, anchor).

        Args:
            href: e.g., 'Text/ch1.xhtml#section-1'

        Returns:
            ('Text/ch1.xhtml', 'section-1')
        """
        if '#' in href:
            path, anchor = href.split('#', 1)
            return (path, anchor)
        return (href, None)

    def get_flat_toc(self) -> List[Dict]:
        """
        Get flattened TOC (all levels in one list).

        Returns:
            List of dictionaries with title, href, level, anchor
        """
        flat = []
        for entry in self.toc:
            flat.extend(entry.flatten())
        return flat

    def get_toc_by_href(self) -> Dict[str, Dict]:
        """
        Get TOC as a href -> TOCEntry mapping.

        Returns:
            Dictionary mapping href (with anchor) to TOCEntry dict
        """
        mapping = {}
        for entry_dict in self.get_flat_toc():
            key = entry_dict['href']
            if entry_dict['anchor']:
                key += f"#{entry_dict['anchor']}"
            mapping[key] = entry_dict
        return mapping

    def get_summary(self) -> Dict:
        """
        Get TOC summary statistics.

        Returns:
            Dictionary with counts per level
        """
        flat = self.get_flat_toc()
        level_counts = {}
        for entry in flat:
            level = entry['level']
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            'total_entries': len(flat),
            'max_depth': max(level_counts.keys()) if level_counts else 0,
            'level_counts': level_counts
        }
