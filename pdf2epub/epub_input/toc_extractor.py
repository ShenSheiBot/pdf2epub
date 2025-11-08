"""
TOC (Table of Contents) Extractor Module

This module extracts and parses TOC from EPUB files.
Supports:
- EPUB 2.0 NCX (toc.ncx)
- EPUB 3.x Navigation Document (nav.xhtml)
- Fallback: Extract from spine structure

Handles multi-level hierarchical TOC and anchor links.
"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import re


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

    def __init__(self, book: epub.EpubBook):
        """
        Initialize TOC extractor.

        Args:
            book: Parsed EpubBook object from ebooklib
        """
        self.book = book
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
        Extract TOC using multiple strategies.

        Priority:
        1. ebooklib's built-in TOC parser
        2. Parse nav.xhtml (EPUB 3)
        3. Parse toc.ncx (EPUB 2)
        4. Fallback: Analyze spine
        """
        # Strategy 1: Use ebooklib's TOC
        try:
            ebooklib_toc = self.book.toc
            if ebooklib_toc:
                logger.info("Using ebooklib's built-in TOC parser")
                entries = self._parse_ebooklib_toc(ebooklib_toc)
                if entries:
                    logger.info(f"Extracted {len(entries)} TOC entries via ebooklib")
                    return entries
        except Exception as e:
            logger.warning(f"ebooklib TOC parsing failed: {e}")

        # Strategy 2: Parse nav.xhtml (EPUB 3)
        try:
            nav_entries = self._parse_nav_document()
            if nav_entries:
                logger.info(f"Extracted {len(nav_entries)} TOC entries from nav.xhtml")
                return nav_entries
        except Exception as e:
            logger.debug(f"nav.xhtml parsing failed: {e}")

        # Strategy 3: Parse toc.ncx (EPUB 2)
        try:
            ncx_entries = self._parse_ncx()
            if ncx_entries:
                logger.info(f"Extracted {len(ncx_entries)} TOC entries from toc.ncx")
                return ncx_entries
        except Exception as e:
            logger.debug(f"toc.ncx parsing failed: {e}")

        # Strategy 4: Fallback to spine analysis
        logger.warning("All TOC parsing methods failed, using spine fallback")
        return self._fallback_spine_toc()

    def _parse_ebooklib_toc(self, toc, level: int = 1) -> List[TOCEntry]:
        """
        Parse ebooklib's TOC structure recursively.

        Args:
            toc: ebooklib TOC object (nested tuples/lists)
            level: Current hierarchy level

        Returns:
            List of TOCEntry objects
        """
        entries = []

        for item in toc:
            if isinstance(item, tuple):
                # It's a (Section, SubItems) tuple
                section, subitems = item
                if isinstance(section, epub.Link):
                    entry = self._create_entry_from_link(section, level)
                    if subitems:
                        entry.children = self._parse_ebooklib_toc(subitems, level + 1)
                    entries.append(entry)
                elif isinstance(section, epub.Section):
                    # Section has title and href, create TOCEntry for it
                    href, anchor = self._split_href_anchor(section.href)
                    entry = TOCEntry(
                        title=section.title,
                        href=href,
                        level=level,
                        anchor=anchor
                    )
                    # Subitems are children
                    if subitems:
                        entry.children = self._parse_ebooklib_toc(subitems, level + 1)
                    entries.append(entry)
            elif isinstance(item, epub.Link):
                # Direct link
                entry = self._create_entry_from_link(item, level)
                entries.append(entry)
            elif isinstance(item, list):
                # Nested list
                entries.extend(self._parse_ebooklib_toc(item, level))

        return entries

    def _create_entry_from_link(self, link: epub.Link, level: int) -> TOCEntry:
        """Create TOCEntry from ebooklib Link."""
        href, anchor = self._split_href_anchor(link.href)
        return TOCEntry(
            title=link.title,
            href=href,
            level=level,
            anchor=anchor
        )

    def _parse_nav_document(self) -> Optional[List[TOCEntry]]:
        """
        Parse EPUB 3 Navigation Document (nav.xhtml).

        Returns:
            List of TOCEntry objects or None
        """
        # Find nav document
        nav_item = None
        for item in self.book.get_items_of_type(ebooklib.ITEM_NAVIGATION):
            nav_item = item
            break

        if not nav_item:
            # Try to find by filename
            for item in self.book.get_items():
                if 'nav' in item.get_name().lower() and item.get_name().endswith('.xhtml'):
                    nav_item = item
                    break

        if not nav_item:
            return None

        logger.debug(f"Found nav document: {nav_item.get_name()}")

        # Parse HTML
        content = nav_item.get_content().decode('utf-8')
        soup = BeautifulSoup(content, 'lxml')

        # Find <nav epub:type="toc"> or <nav id="toc">
        toc_nav = soup.find('nav', attrs={'epub:type': 'toc'})
        if not toc_nav:
            toc_nav = soup.find('nav', id='toc')
        if not toc_nav:
            # Fallback: first <nav> with <ol>
            toc_nav = soup.find('nav')

        if not toc_nav:
            return None

        # Extract from <ol> structure
        ol = toc_nav.find('ol')
        if not ol:
            return None

        return self._parse_nav_ol(ol, base_path=Path(nav_item.get_name()).parent)

    def _parse_nav_ol(self, ol, level: int = 1, base_path: Path = Path()) -> List[TOCEntry]:
        """
        Recursively parse <ol> structure from nav document.

        Args:
            ol: BeautifulSoup <ol> element
            level: Current level
            base_path: Base path for resolving relative hrefs

        Returns:
            List of TOCEntry objects
        """
        entries = []

        for li in ol.find_all('li', recursive=False):
            # Find <a> tag
            a = li.find('a')
            if not a:
                continue

            title = a.get_text(strip=True)
            href_raw = a.get('href', '')

            # Resolve relative path
            href = self._resolve_href(href_raw, base_path)
            href, anchor = self._split_href_anchor(href)

            entry = TOCEntry(title=title, href=href, level=level, anchor=anchor)

            # Check for nested <ol>
            nested_ol = li.find('ol')
            if nested_ol:
                entry.children = self._parse_nav_ol(nested_ol, level + 1, base_path)

            entries.append(entry)

        return entries

    def _parse_ncx(self) -> Optional[List[TOCEntry]]:
        """
        Parse EPUB 2 NCX (toc.ncx).

        Returns:
            List of TOCEntry objects or None
        """
        # Find NCX
        ncx_item = None
        for item in self.book.get_items():
            if item.get_name().endswith('.ncx'):
                ncx_item = item
                break

        if not ncx_item:
            return None

        logger.debug(f"Found NCX: {ncx_item.get_name()}")

        # Parse XML
        content = ncx_item.get_content().decode('utf-8')
        soup = BeautifulSoup(content, 'lxml-xml')  # Use XML parser

        # Find <navMap>
        nav_map = soup.find('navMap')
        if not nav_map:
            return None

        base_path = Path(ncx_item.get_name()).parent
        return self._parse_ncx_navpoint(nav_map, base_path=base_path)

    def _parse_ncx_navpoint(
        self,
        parent,
        level: int = 1,
        base_path: Path = Path()
    ) -> List[TOCEntry]:
        """
        Recursively parse <navPoint> from NCX.

        Args:
            parent: Parent element (navMap or navPoint)
            level: Current level
            base_path: Base path for resolving hrefs

        Returns:
            List of TOCEntry objects
        """
        entries = []

        for navpoint in parent.find_all('navPoint', recursive=False):
            # Extract title from <navLabel><text>
            nav_label = navpoint.find('navLabel')
            if nav_label:
                text_elem = nav_label.find('text')
                title = text_elem.get_text(strip=True) if text_elem else 'Untitled'
            else:
                title = 'Untitled'

            # Extract href from <content src="..."/>
            content_elem = navpoint.find('content')
            if content_elem:
                href_raw = content_elem.get('src', '')
                href = self._resolve_href(href_raw, base_path)
                href, anchor = self._split_href_anchor(href)
            else:
                href = ''
                anchor = None

            entry = TOCEntry(title=title, href=href, level=level, anchor=anchor)

            # Recursive: find nested navPoints
            entry.children = self._parse_ncx_navpoint(navpoint, level + 1, base_path)

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

        for item_id, linear in self.book.spine:
            item = self.book.get_item_with_id(item_id)
            if not item:
                continue

            # Skip cover, titlepage, etc.
            name_lower = item.get_name().lower()
            if any(skip in name_lower for skip in ['cover', 'title', 'copy', 'toc']):
                continue

            title = f"Chapter {chapter_num}"
            entry = TOCEntry(
                title=title,
                href=item.get_name(),
                level=1,
                anchor=None
            )
            entries.append(entry)
            chapter_num += 1

        logger.warning(f"Generated {len(entries)} generic TOC entries from spine")
        return entries

    def _resolve_href(self, href: str, base_path: Path) -> str:
        """
        Resolve relative href to absolute path within EPUB.

        Args:
            href: Raw href (may be relative, e.g., '../Text/ch1.xhtml')
            base_path: Base directory path

        Returns:
            Resolved href (e.g., 'Text/ch1.xhtml')
        """
        if not href:
            return ''

        # Remove anchor temporarily
        href_clean, _ = self._split_href_anchor(href)

        # Resolve relative path
        resolved = (base_path / href_clean).as_posix()

        # Normalize (remove ../)
        resolved = Path(resolved).as_posix()

        return resolved

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

    def get_toc_by_href(self) -> Dict[str, TOCEntry]:
        """
        Get TOC as a href -> TOCEntry mapping.

        Returns:
            Dictionary mapping href (with anchor) to TOCEntry
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
