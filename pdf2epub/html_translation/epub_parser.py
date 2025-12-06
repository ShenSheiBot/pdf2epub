"""
EPUB Parser Module

This module provides functionality to parse EPUB files and extract:
- Metadata (title, author, language, etc.)
- Spine order (reading sequence)
- Content files (XHTML documents)
- Resources (CSS, images, fonts)

Uses zipfile + lxml for direct EPUB parsing (no ebooklib dependency).
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from lxml import etree


# Namespaces used in EPUB
NAMESPACES = {
    'opf': 'http://www.idpf.org/2007/opf',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'ncx': 'http://www.daisy.org/z3986/2005/ncx/',
    'container': 'urn:oasis:names:tc:opendocument:xmlns:container',
    'xhtml': 'http://www.w3.org/1999/xhtml',
}

# Media type to category mapping
MEDIA_TYPE_CATEGORIES = {
    'text/css': 'css',
    'application/x-dtbncx+xml': 'ncx',
    'application/xhtml+xml': 'document',
    'text/html': 'document',
    'image/jpeg': 'images',
    'image/png': 'images',
    'image/gif': 'images',
    'image/svg+xml': 'images',
    'application/font-woff': 'fonts',
    'application/font-woff2': 'fonts',
    'font/woff': 'fonts',
    'font/woff2': 'fonts',
    'font/otf': 'fonts',
    'font/ttf': 'fonts',
    'application/vnd.ms-opentype': 'fonts',
    'application/x-font-ttf': 'fonts',
    'application/x-font-otf': 'fonts',
}


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

        # Open ZIP and parse structure
        self._zf = zipfile.ZipFile(str(self.epub_path), 'r')
        self._rootfile_path = self._find_rootfile()
        self._rootfile_dir = str(Path(self._rootfile_path).parent)
        if self._rootfile_dir == '.':
            self._rootfile_dir = ''

        # Parse OPF (content.opf)
        self._opf_tree = self._parse_opf()
        self._manifest = self._parse_manifest()

        # Cached properties
        self._spine_items = None
        self._metadata = None
        self._resources = None
        self._toc = None

    def __del__(self):
        """Close ZIP file on cleanup."""
        if hasattr(self, '_zf') and self._zf:
            self._zf.close()

    def _find_rootfile(self) -> str:
        """Find the root OPF file path from container.xml."""
        container_path = 'META-INF/container.xml'
        try:
            container_xml = self._zf.read(container_path)
            tree = etree.fromstring(container_xml)
            rootfile = tree.find('.//container:rootfile', NAMESPACES)
            if rootfile is not None:
                return rootfile.get('full-path', 'content.opf')
        except Exception as e:
            logger.warning(f"Failed to parse container.xml: {e}")
        return 'content.opf'

    def _parse_opf(self) -> etree._Element:
        """Parse the OPF (content.opf) file."""
        try:
            opf_content = self._zf.read(self._rootfile_path)
            return etree.fromstring(opf_content)
        except Exception as e:
            raise ValueError(f"Failed to parse OPF file: {e}")

    def _parse_manifest(self) -> Dict[str, Dict]:
        """Parse manifest to build id -> item mapping."""
        manifest = {}
        for item in self._opf_tree.findall('.//opf:manifest/opf:item', NAMESPACES):
            item_id = item.get('id', '')
            href = item.get('href', '')
            media_type = item.get('media-type', '')

            # Resolve href relative to OPF location
            if self._rootfile_dir:
                full_href = f"{self._rootfile_dir}/{href}"
            else:
                full_href = href

            manifest[item_id] = {
                'id': item_id,
                'href': href,
                'full_href': full_href,
                'media_type': media_type,
            }
        return manifest

    def _resolve_href(self, href: str) -> str:
        """Resolve href relative to OPF directory."""
        if self._rootfile_dir:
            return f"{self._rootfile_dir}/{href}"
        return href

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
        """Extract metadata from OPF."""
        meta = {}

        # Helper to get DC element text
        def get_dc(name: str) -> Optional[str]:
            el = self._opf_tree.find(f'.//dc:{name}', NAMESPACES)
            return el.text if el is not None and el.text else None

        def get_all_dc(name: str) -> List[str]:
            els = self._opf_tree.findall(f'.//dc:{name}', NAMESPACES)
            return [el.text for el in els if el.text]

        # Title
        meta['title'] = get_dc('title') or 'Unknown'

        # Author(s)
        authors = get_all_dc('creator')
        meta['author'] = ', '.join(authors) if authors else 'Unknown'

        # Language
        meta['language'] = get_dc('language') or 'unknown'

        # Identifier
        meta['identifier'] = get_dc('identifier')

        # Publisher
        meta['publisher'] = get_dc('publisher')

        # Description
        meta['description'] = get_dc('description')

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
            - linear: Whether item is in linear reading order
        """
        if self._spine_items is None:
            self._spine_items = self._extract_spine()
        return self._spine_items

    def _extract_spine(self) -> List[Dict]:
        """Extract spine items in reading order."""
        spine_items = []

        spine_el = self._opf_tree.find('.//opf:spine', NAMESPACES)
        if spine_el is None:
            logger.warning("No spine found in OPF")
            return spine_items

        for itemref in spine_el.findall('opf:itemref', NAMESPACES):
            idref = itemref.get('idref', '')
            linear = itemref.get('linear', 'yes')

            if idref not in self._manifest:
                logger.warning(f"Spine item not found in manifest: {idref}")
                continue

            item = self._manifest[idref]
            spine_items.append({
                'id': item['id'],
                'href': item['href'],
                'full_href': item['full_href'],
                'media_type': item['media_type'],
                'linear': linear,
            })

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

        # Get spine hrefs to exclude documents
        spine_hrefs = {item['href'] for item in self.spine}

        for item_id, item in self._manifest.items():
            href = item['href']
            media_type = item['media_type']

            # Skip spine documents
            if href in spine_hrefs:
                continue

            # Determine category
            category = MEDIA_TYPE_CATEGORIES.get(media_type, 'other')
            if category in ('document', 'ncx'):
                continue

            # Read content
            try:
                content = self._zf.read(item['full_href'])
            except KeyError:
                logger.warning(f"Resource not found in ZIP: {item['full_href']}")
                continue

            resource = {
                'id': item_id,
                'href': href,
                'full_href': item['full_href'],
                'media_type': media_type,
                'content': content
            }
            resources[category].append(resource)

        logger.info(
            f"Extracted resources: "
            f"{len(resources['css'])} CSS, "
            f"{len(resources['images'])} images, "
            f"{len(resources['fonts'])} fonts, "
            f"{len(resources['other'])} other"
        )

        return resources

    @property
    def toc(self) -> List[Dict]:
        """
        Get table of contents.

        Returns:
            List of TOC entries with title, href, and children
        """
        if self._toc is None:
            self._toc = self._extract_toc()
        return self._toc

    def _extract_toc(self) -> List[Dict]:
        """Extract TOC from NCX or NAV document."""
        # Try NCX first
        ncx_toc = self._extract_ncx_toc()
        if ncx_toc:
            return ncx_toc

        # Try NAV (EPUB3)
        nav_toc = self._extract_nav_toc()
        if nav_toc:
            return nav_toc

        logger.warning("No TOC found")
        return []

    def _extract_ncx_toc(self) -> List[Dict]:
        """
        Extract TOC from NCX file.

        NCX files may use default namespace (no prefix) or explicit prefix.
        This method handles both cases by using local-name() matching.
        """
        # Find NCX in manifest by media-type
        ncx_item = None
        for item in self._manifest.values():
            if item['media_type'] == 'application/x-dtbncx+xml':
                ncx_item = item
                break

        # Fallback: find by .ncx extension
        if not ncx_item:
            for item in self._manifest.values():
                if item['href'].lower().endswith('.ncx'):
                    ncx_item = item
                    break

        if not ncx_item:
            return []

        try:
            ncx_content = self._zf.read(ncx_item['full_href'])
            ncx_tree = etree.fromstring(ncx_content)

            # Helper: find element by local name (ignores namespace)
            def find_by_local(parent, local_name):
                """Find first child element by local name, ignoring namespace."""
                for el in parent:
                    if etree.QName(el.tag).localname == local_name:
                        return el
                return None

            def findall_by_local(parent, local_name):
                """Find all child elements by local name, ignoring namespace."""
                return [el for el in parent
                        if etree.QName(el.tag).localname == local_name]

            def parse_navpoint(navpoint) -> Optional[Dict]:
                # Find navLabel/text
                nav_label = find_by_local(navpoint, 'navLabel')
                if nav_label is None:
                    return None
                text_el = find_by_local(nav_label, 'text')
                title = text_el.text if text_el is not None and text_el.text else ''

                # Find content/@src
                content_el = find_by_local(navpoint, 'content')
                href = content_el.get('src', '') if content_el is not None else ''

                entry = {
                    'title': title,
                    'href': href,
                    'children': []
                }

                # Recursively parse nested navPoints
                for child in findall_by_local(navpoint, 'navPoint'):
                    child_entry = parse_navpoint(child)
                    if child_entry:
                        entry['children'].append(child_entry)

                return entry

            # Find navMap (could be at root level or nested)
            navmap = None
            for el in ncx_tree.iter():
                if etree.QName(el.tag).localname == 'navMap':
                    navmap = el
                    break

            if navmap is None:
                return []

            toc = []
            for navpoint in findall_by_local(navmap, 'navPoint'):
                entry = parse_navpoint(navpoint)
                if entry:
                    toc.append(entry)

            return toc
        except Exception as e:
            logger.warning(f"Failed to parse NCX: {e}")
            return []

    def _extract_nav_toc(self) -> List[Dict]:
        """
        Extract TOC from EPUB3 NAV document.

        NAV document is identified by properties="nav" in manifest, not by filename.
        """
        # Find NAV document by properties="nav" in OPF manifest
        nav_item = None
        for item in self._opf_tree.findall('.//opf:manifest/opf:item', NAMESPACES):
            props = item.get('properties', '')
            if 'nav' in props.split():
                item_id = item.get('id', '')
                if item_id in self._manifest:
                    nav_item = self._manifest[item_id]
                    break

        if not nav_item:
            return []

        try:
            nav_content = self._zf.read(nav_item['full_href'])
            # Use HTML parser for more tolerance
            from lxml import html as lxml_html
            nav_tree = lxml_html.fromstring(nav_content)

            def parse_li(li) -> Optional[Dict]:
                # Find <a> (direct child or nested)
                a = li.find('.//a')
                if a is None:
                    return None

                entry = {
                    'title': ''.join(a.itertext()).strip(),
                    'href': a.get('href', ''),
                    'children': []
                }

                # Find nested <ol> for children
                ol = li.find('ol')
                if ol is not None:
                    for child_li in ol.findall('li'):
                        child_entry = parse_li(child_li)
                        if child_entry:
                            entry['children'].append(child_entry)

                return entry

            # Find <nav> with epub:type="toc" or id containing "toc"
            nav_el = None

            # Try epub:type="toc" first
            for nav in nav_tree.iter('nav'):
                epub_type = nav.get('{http://www.idpf.org/2007/ops}type', '')
                if 'toc' in epub_type:
                    nav_el = nav
                    break

            # Fallback: any nav with id/class containing 'toc'
            if nav_el is None:
                for nav in nav_tree.iter('nav'):
                    nav_id = (nav.get('id', '') + nav.get('class', '')).lower()
                    if 'toc' in nav_id:
                        nav_el = nav
                        break

            # Fallback: first nav with ol
            if nav_el is None:
                for nav in nav_tree.iter('nav'):
                    if nav.find('ol') is not None:
                        nav_el = nav
                        break

            if nav_el is None:
                return []

            ol = nav_el.find('ol')
            if ol is None:
                return []

            toc = []
            for li in ol.findall('li'):
                entry = parse_li(li)
                if entry:
                    toc.append(entry)

            return toc
        except Exception as e:
            logger.warning(f"Failed to parse NAV: {e}")
            return []

    def get_raw_content(self, href: str) -> Optional[bytes]:
        """
        Get raw content directly from EPUB ZIP.

        Args:
            href: File path within EPUB (e.g., 'Text/chapter1.xhtml')

        Returns:
            Raw bytes content or None if not found
        """
        try:
            # Try with rootfile directory prefix first
            full_href = self._resolve_href(href)
            if full_href in self._zf.namelist():
                return self._zf.read(full_href)

            # Try direct path
            if href in self._zf.namelist():
                return self._zf.read(href)

            # Search for matching filename
            for name in self._zf.namelist():
                if name.endswith(href) or name.endswith('/' + href):
                    return self._zf.read(name)

            logger.warning(f"File not found in EPUB ZIP: {href}")
            return None
        except Exception as e:
            logger.warning(f"Failed to read content for {href}: {e}")
            return None

    def get_css_content(self, css_href: str) -> Optional[str]:
        """
        Get CSS content as string.

        Args:
            css_href: Path to CSS file within EPUB

        Returns:
            CSS content as string or None
        """
        content = self.get_raw_content(css_href)
        if content:
            try:
                return content.decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to decode CSS {css_href}: {e}")
        return None

    def extract_cover_image(self) -> Optional[Tuple[bytes, str]]:
        """
        Extract cover image from EPUB.

        Returns:
            Tuple of (image_bytes, extension) or None
        """
        # Method 1: Check metadata for cover
        meta_el = self._opf_tree.find('.//opf:meta[@name="cover"]', NAMESPACES)
        if meta_el is not None:
            cover_id = meta_el.get('content', '')
            if cover_id in self._manifest:
                item = self._manifest[cover_id]
                try:
                    content = self._zf.read(item['full_href'])
                    ext = item['media_type'].split('/')[-1]
                    logger.info(f"Found cover image via metadata: {item['href']}")
                    return (content, ext)
                except Exception:
                    pass

        # Method 2: Look for items with 'cover' in name
        for item in self._manifest.values():
            category = MEDIA_TYPE_CATEGORIES.get(item['media_type'], 'other')
            if category == 'images' and 'cover' in item['href'].lower():
                try:
                    content = self._zf.read(item['full_href'])
                    ext = item['media_type'].split('/')[-1]
                    logger.info(f"Found cover image via filename: {item['href']}")
                    return (content, ext)
                except Exception:
                    pass

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
