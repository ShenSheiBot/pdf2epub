"""
EPUB Parser Module

This module provides functionality to parse EPUB files and extract:
- Metadata (title, author, language, etc.)
- Spine order (reading sequence)
- Content files (XHTML documents)
- Resources (CSS, images, fonts)

Uses ebooklib for EPUB parsing.
"""

import ebooklib
from ebooklib import epub
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger


class EPUBParser:
    """Parse EPUB files and extract structure and resources."""

    def __init__(self, epub_path: str):
        """
        Initialize the EPUB parser.

        Args:
            epub_path: Path to the EPUB file
        """
        self.epub_path = Path(epub_path)
        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")

        logger.info(f"Loading EPUB: {self.epub_path.name}")
        self.book = epub.read_epub(str(self.epub_path))
        self._spine_items = None
        self._metadata = None
        self._resources = None

    @property
    def metadata(self) -> Dict:
        """
        Extract and cache metadata from EPUB.

        Returns:
            Dictionary containing:
            - title: Book title
            - author: Author name(s)
            - language: Language code
            - identifier: Unique identifier
            - publisher: Publisher name
            - description: Book description
        """
        if self._metadata is None:
            self._metadata = self._extract_metadata()
        return self._metadata

    def _extract_metadata(self) -> Dict:
        """Extract metadata from EPUB."""
        meta = {}

        # Title
        title = self.book.get_metadata('DC', 'title')
        meta['title'] = title[0][0] if title else 'Unknown'

        # Author
        authors = self.book.get_metadata('DC', 'creator')
        if authors:
            # Handle multiple authors
            author_names = [author[0] for author in authors]
            meta['author'] = ', '.join(author_names)
        else:
            meta['author'] = 'Unknown'

        # Language
        language = self.book.get_metadata('DC', 'language')
        meta['language'] = language[0][0] if language else 'unknown'

        # Identifier (ISBN, DOI, etc.)
        identifier = self.book.get_metadata('DC', 'identifier')
        meta['identifier'] = identifier[0][0] if identifier else None

        # Publisher
        publisher = self.book.get_metadata('DC', 'publisher')
        meta['publisher'] = publisher[0][0] if publisher else None

        # Description
        description = self.book.get_metadata('DC', 'description')
        meta['description'] = description[0][0] if description else None

        logger.info(f"Extracted metadata: {meta['title']} by {meta['author']}")
        return meta

    @property
    def spine(self) -> List[Dict]:
        """
        Get the spine (reading order) of the EPUB.

        Returns:
            List of dictionaries with:
            - id: Item ID
            - href: File path within EPUB
            - media_type: MIME type
            - content: Raw content bytes
        """
        if self._spine_items is None:
            self._spine_items = self._extract_spine()
        return self._spine_items

    def _extract_spine(self) -> List[Dict]:
        """Extract spine items in reading order."""
        spine_items = []

        # Get spine (ordered list of content files)
        for item_id, linear in self.book.spine:
            # Find the item in the book
            item = self.book.get_item_with_id(item_id)

            if item is None:
                logger.warning(f"Spine item not found: {item_id}")
                continue

            spine_item = {
                'id': item.get_id(),
                'href': item.get_name(),  # e.g., 'Text/chapter1.xhtml'
                'media_type': item.get_type(),
                'linear': linear,  # 'yes' or 'no'
                'content': item.get_content(),  # bytes
            }

            spine_items.append(spine_item)

        logger.info(f"Extracted {len(spine_items)} spine items")
        return spine_items

    @property
    def resources(self) -> Dict[str, List[Dict]]:
        """
        Get all resources (CSS, images, fonts) from EPUB.

        Returns:
            Dictionary with categories:
            - css: CSS stylesheets
            - images: Image files
            - fonts: Font files
            - other: Other resources
        """
        if self._resources is None:
            self._resources = self._extract_resources()
        return self._resources

    def _extract_resources(self) -> Dict[str, List[Dict]]:
        """Categorize and extract resources."""
        resources = {
            'css': [],
            'images': [],
            'fonts': [],
            'other': []
        }

        for item in self.book.get_items():
            item_type = item.get_type()

            resource = {
                'id': item.get_id(),
                'href': item.get_name(),
                'media_type': item_type,
                'content': item.get_content()
            }

            # Categorize by media type
            if item_type == ebooklib.ITEM_STYLE:
                resources['css'].append(resource)
            elif item_type == ebooklib.ITEM_IMAGE:
                resources['images'].append(resource)
            elif item_type == ebooklib.ITEM_FONT:
                resources['fonts'].append(resource)
            elif item_type == ebooklib.ITEM_DOCUMENT:
                # Skip document items (they're in spine)
                continue
            else:
                resources['other'].append(resource)

        logger.info(
            f"Extracted resources: "
            f"{len(resources['css'])} CSS, "
            f"{len(resources['images'])} images, "
            f"{len(resources['fonts'])} fonts, "
            f"{len(resources['other'])} other"
        )

        return resources

    def get_item_by_href(self, href: str) -> Optional[ebooklib.epub.EpubItem]:
        """
        Get an EPUB item by its href.

        Args:
            href: File path within EPUB (e.g., 'Text/chapter1.xhtml')

        Returns:
            EpubItem or None if not found
        """
        return self.book.get_item_with_href(href)

    def get_css_content(self, css_href: str) -> Optional[str]:
        """
        Get CSS content as string.

        Args:
            css_href: Path to CSS file within EPUB

        Returns:
            CSS content as string or None
        """
        item = self.get_item_by_href(css_href)
        if item:
            try:
                return item.get_content().decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to decode CSS {css_href}: {e}")
        return None

    def extract_cover_image(self) -> Optional[Tuple[bytes, str]]:
        """
        Extract cover image from EPUB.

        Returns:
            Tuple of (image_bytes, extension) or None
        """
        # Method 1: Try metadata
        cover_id = self.book.get_metadata('OPF', 'cover')
        if cover_id:
            cover_item = self.book.get_item_with_id(cover_id[0][1]['content'])
            if cover_item:
                content = cover_item.get_content()
                # Determine extension from media type
                media_type = cover_item.get_type()
                ext = media_type.split('/')[-1] if '/' in str(media_type) else 'jpg'
                logger.info(f"Found cover image via metadata: {cover_item.get_name()}")
                return (content, ext)

        # Method 2: Look for items with 'cover' in name
        for item in self.book.get_items_of_type(ebooklib.ITEM_IMAGE):
            name = item.get_name().lower()
            if 'cover' in name:
                content = item.get_content()
                media_type = item.get_type()
                ext = media_type.split('/')[-1] if '/' in str(media_type) else 'jpg'
                logger.info(f"Found cover image via filename: {item.get_name()}")
                return (content, ext)

        logger.warning("No cover image found")
        return None

    def get_summary(self) -> Dict:
        """
        Get a summary of the EPUB structure.

        Returns:
            Dictionary with:
            - metadata: Book metadata
            - spine_count: Number of spine items
            - resource_counts: Counts of each resource type
        """
        return {
            'metadata': self.metadata,
            'spine_count': len(self.spine),
            'resource_counts': {
                category: len(items)
                for category, items in self.resources.items()
            }
        }
