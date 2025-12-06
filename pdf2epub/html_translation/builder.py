"""
HTML EPUB Builder.

Rebuilds EPUB by replacing translated XHTML content while preserving
all other resources (CSS, fonts, images, metadata).
"""

import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from loguru import logger
from xml.etree import ElementTree as ET

from .epub_parser import EPUBParser


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    import re
    # Remove or replace characters that are problematic in filenames
    # Windows: \ / : * ? " < > |
    # Also handle other common issues
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', name)
    # Replace multiple underscores/spaces with single
    sanitized = re.sub(r'[_\s]+', ' ', sanitized)
    # Trim and limit length
    sanitized = sanitized.strip()[:200]
    return sanitized


@dataclass
class BuildConfig:
    """Configuration for EPUB building."""
    original_epub: Path
    translated_dir: Path
    output_path: Path
    book_title: str
    translated_metadata: Optional[Dict] = None  # Contains translated_title and toc


class HTMLEpubBuilder:
    """
    Build translated EPUB by replacing XHTML content.

    Process:
    1. Extract original EPUB to temp directory
    2. Replace XHTML files with translated versions
    3. Repackage as new EPUB

    Preserves:
    - mimetype (uncompressed, first file)
    - META-INF/container.xml
    - content.opf (manifest, spine, metadata)
    - toc.ncx / nav.xhtml (navigation)
    - CSS, fonts, images
    """

    def __init__(self, config: BuildConfig):
        self.config = config
        self.original_epub = config.original_epub
        self.translated_dir = config.translated_dir
        self.output_path = config.output_path

    def build(self) -> Path:
        """
        Build the translated EPUB.

        Returns:
            Path to the output EPUB file
        """
        logger.info(f"Building translated EPUB from {self.original_epub}")

        # Create temp directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extract_dir = temp_path / "epub_content"

            # Step 1: Extract original EPUB
            self._extract_epub(extract_dir)

            # Step 2: Find and replace XHTML files
            replaced_count = self._replace_xhtml_files(extract_dir)
            logger.info(f"Replaced {replaced_count} XHTML files")

            # Step 3: Update metadata if provided
            if self.config.translated_metadata:
                self._update_content_opf(extract_dir, self.config.translated_metadata)
                self._update_toc_ncx(extract_dir, self.config.translated_metadata)
                self._update_nav_xhtml(extract_dir, self.config.translated_metadata)

            # Step 4: Repackage as EPUB
            self._package_epub(extract_dir)

        logger.info(f"Built EPUB: {self.output_path}")
        return self.output_path

    def _extract_epub(self, extract_dir: Path):
        """Extract original EPUB to directory."""
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.original_epub, 'r') as zf:
            zf.extractall(extract_dir)

        logger.debug(f"Extracted EPUB to {extract_dir}")

    def _replace_xhtml_files(self, extract_dir: Path) -> int:
        """
        Replace XHTML files with translated versions.

        Returns:
            Number of files replaced
        """
        replaced = 0

        # Get list of translated files
        translated_files = {f.name: f for f in self.translated_dir.glob("*.xhtml")}
        translated_files.update({f.name: f for f in self.translated_dir.glob("*.html")})

        if not translated_files:
            logger.warning("No translated files found in translated_dir")
            return 0

        # Find all XHTML/HTML files in extracted EPUB
        for xhtml_file in extract_dir.rglob("*.xhtml"):
            if xhtml_file.name in translated_files:
                self._replace_file(xhtml_file, translated_files[xhtml_file.name])
                replaced += 1

        for html_file in extract_dir.rglob("*.html"):
            if html_file.name in translated_files:
                self._replace_file(html_file, translated_files[html_file.name])
                replaced += 1

        return replaced

    def _replace_file(self, target: Path, source: Path):
        """Replace target file with source content."""
        content = source.read_text(encoding='utf-8')
        target.write_text(content, encoding='utf-8')
        logger.debug(f"Replaced {target.name}")

    def _package_epub(self, content_dir: Path):
        """
        Package directory contents as EPUB.

        EPUB packaging rules:
        1. mimetype must be first file, stored uncompressed
        2. Other files use DEFLATE compression
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.output_path, 'w') as zf:
            # 1. Write mimetype first, uncompressed
            mimetype_path = content_dir / "mimetype"
            if mimetype_path.exists():
                zf.write(
                    mimetype_path,
                    "mimetype",
                    compress_type=zipfile.ZIP_STORED
                )
            else:
                # Create mimetype if missing
                zf.writestr(
                    "mimetype",
                    "application/epub+zip",
                    compress_type=zipfile.ZIP_STORED
                )

            # 2. Write all other files with compression
            for file_path in content_dir.rglob("*"):
                if file_path.is_file() and file_path.name != "mimetype":
                    arcname = file_path.relative_to(content_dir).as_posix()
                    zf.write(
                        file_path,
                        arcname,
                        compress_type=zipfile.ZIP_DEFLATED
                    )

        logger.debug(f"Packaged EPUB to {self.output_path}")

    def _find_opf_path(self, extract_dir: Path) -> Optional[Path]:
        """
        Find the OPF package document by reading META-INF/container.xml.

        The container.xml is the standard entry point for EPUB - it specifies
        where the actual OPF file is located (could be content.opf, package.opf,
        or any other name in any subdirectory).
        """
        container_path = extract_dir / "META-INF" / "container.xml"

        if not container_path.exists():
            # Fallback: glob for any .opf file
            logger.debug("container.xml not found, falling back to glob")
            opf_candidates = list(extract_dir.rglob("*.opf"))
            return opf_candidates[0] if opf_candidates else None

        try:
            tree = ET.parse(container_path)
            root = tree.getroot()

            # container.xml uses the container namespace
            ns = {'container': 'urn:oasis:names:tc:opendocument:xmlns:container'}

            # Find rootfile element
            rootfile = root.find('.//container:rootfile', ns)
            if rootfile is None:
                # Try without namespace
                for elem in root.iter():
                    if elem.tag.endswith('}rootfile') or elem.tag == 'rootfile':
                        rootfile = elem
                        break

            if rootfile is not None:
                full_path = rootfile.get('full-path')
                if full_path:
                    opf_path = extract_dir / full_path
                    if opf_path.exists():
                        return opf_path

            # Fallback to glob
            opf_candidates = list(extract_dir.rglob("*.opf"))
            return opf_candidates[0] if opf_candidates else None

        except Exception as e:
            logger.debug(f"Error parsing container.xml: {e}, falling back to glob")
            opf_candidates = list(extract_dir.rglob("*.opf"))
            return opf_candidates[0] if opf_candidates else None

    def _find_toc_files(self, extract_dir: Path, opf_path: Path) -> Dict[str, Optional[Path]]:
        """
        Find NCX and Nav files by parsing the OPF manifest.

        Returns dict with keys:
        - 'ncx': Path to NCX file (EPUB 2 TOC, media-type="application/x-dtbncx+xml")
        - 'nav': Path to Nav document (EPUB 3 TOC, properties="nav")
        """
        result = {'ncx': None, 'nav': None}
        opf_dir = opf_path.parent

        try:
            tree = ET.parse(opf_path)
            root = tree.getroot()

            # OPF namespace
            ns = {'opf': 'http://www.idpf.org/2007/opf'}

            # Find manifest element
            manifest = root.find('.//opf:manifest', ns)
            if manifest is None:
                # Try without namespace
                for elem in root.iter():
                    if elem.tag.endswith('}manifest') or elem.tag == 'manifest':
                        manifest = elem
                        break

            if manifest is None:
                logger.debug("No manifest found in OPF")
                return result

            # Search all items in manifest
            for item in manifest:
                if not (item.tag.endswith('}item') or item.tag == 'item'):
                    continue

                media_type = item.get('media-type', '')
                properties = item.get('properties', '')
                href = item.get('href', '')

                if not href:
                    continue

                # NCX: identified by media-type
                if media_type == 'application/x-dtbncx+xml':
                    ncx_path = opf_dir / href
                    if ncx_path.exists():
                        result['ncx'] = ncx_path
                        logger.debug(f"Found NCX via manifest: {ncx_path}")

                # Nav: identified by properties="nav"
                if 'nav' in properties.split():
                    nav_path = opf_dir / href
                    if nav_path.exists():
                        result['nav'] = nav_path
                        logger.debug(f"Found Nav via manifest: {nav_path}")

            # Also check spine for toc attribute (alternative NCX reference)
            if result['ncx'] is None:
                spine = root.find('.//opf:spine', ns)
                if spine is None:
                    for elem in root.iter():
                        if elem.tag.endswith('}spine') or elem.tag == 'spine':
                            spine = elem
                            break

                if spine is not None:
                    toc_id = spine.get('toc')
                    if toc_id:
                        # Find the item with this id
                        for item in manifest:
                            if item.get('id') == toc_id:
                                href = item.get('href', '')
                                if href:
                                    ncx_path = opf_dir / href
                                    if ncx_path.exists():
                                        result['ncx'] = ncx_path
                                        logger.debug(f"Found NCX via spine toc attr: {ncx_path}")
                                break

        except Exception as e:
            logger.debug(f"Error parsing OPF for TOC files: {e}")

        return result

    def _update_content_opf(self, extract_dir: Path, metadata: Dict):
        """Update content.opf with translated title and language."""
        opf_path = self._find_opf_path(extract_dir)
        if not opf_path:
            logger.warning("No OPF file found")
            return

        translated_title = metadata.get('translated_title')
        target_lang_code = metadata.get('target_language_code')

        if not translated_title and not target_lang_code:
            return

        try:
            # Parse with namespace handling
            tree = ET.parse(opf_path)
            root = tree.getroot()

            # Handle namespaces - OPF uses default namespace
            namespaces = {
                'opf': 'http://www.idpf.org/2007/opf',
                'dc': 'http://purl.org/dc/elements/1.1/'
            }

            # Find and update dc:title
            if translated_title:
                # Try with namespace first
                title_elem = root.find('.//dc:title', namespaces)
                if title_elem is None:
                    # Try without namespace (some EPUBs don't use namespaces properly)
                    for elem in root.iter():
                        if elem.tag.endswith('}title') or elem.tag == 'title':
                            title_elem = elem
                            break

                if title_elem is not None:
                    title_elem.text = translated_title
                    logger.debug(f"Updated content.opf title: {translated_title}")
                else:
                    logger.warning("dc:title element not found in content.opf")

            # Find and update dc:language
            if target_lang_code:
                lang_elem = root.find('.//dc:language', namespaces)
                if lang_elem is None:
                    # Try without namespace
                    for elem in root.iter():
                        if elem.tag.endswith('}language') or elem.tag == 'language':
                            lang_elem = elem
                            break

                if lang_elem is not None:
                    old_lang = lang_elem.text
                    lang_elem.text = target_lang_code
                    logger.debug(f"Updated content.opf language: {old_lang} -> {target_lang_code}")
                else:
                    logger.warning("dc:language element not found in content.opf")

            tree.write(opf_path, encoding='utf-8', xml_declaration=True)

        except Exception as e:
            logger.warning(f"Failed to update content.opf: {e}")

    def _update_toc_ncx(self, extract_dir: Path, metadata: Dict):
        """Update toc.ncx with translated chapter titles."""
        # Find NCX file via OPF manifest (proper EPUB way)
        opf_path = self._find_opf_path(extract_dir)
        if not opf_path:
            logger.debug("No OPF found, cannot locate NCX")
            return

        toc_files = self._find_toc_files(extract_dir, opf_path)
        ncx_path = toc_files.get('ncx')

        if not ncx_path:
            # Fallback to glob (for malformed EPUBs)
            ncx_candidates = list(extract_dir.rglob("*.ncx"))
            if not ncx_candidates:
                logger.debug("No NCX file found (EPUB 3 only?)")
                return
            ncx_path = ncx_candidates[0]

        ncx_dir = ncx_path.parent  # For resolving relative paths
        toc_entries = metadata.get('toc', [])
        if not toc_entries:
            return

        # Build multiple mappings for robust matching
        # 1. Full href -> title (exact match)
        # 2. Basename + fragment -> title (for when paths differ)
        # 3. Normalized href -> title
        href_to_title = {}
        basename_to_title = {}
        for entry in toc_entries:
            href = entry.get('href', '')
            translated = entry.get('translated', '')
            if href and translated:
                # Full href
                href_to_title[href] = translated
                # Basename with fragment (e.g., "chapter1.html#section1")
                basename = Path(href.split('#')[0]).name
                if '#' in href:
                    basename += '#' + href.split('#')[1]
                basename_to_title[basename] = translated
                # Also store without fragment for base file matching
                basename_no_frag = Path(href.split('#')[0]).name
                if basename_no_frag not in basename_to_title:
                    basename_to_title[basename_no_frag] = translated

        def find_translation(src: str) -> str:
            """Try multiple strategies to find translation for src."""
            # 1. Exact match
            if src in href_to_title:
                return href_to_title[src]

            # 2. Resolve relative to NCX dir and try again
            if ncx_dir != extract_dir:
                resolved = (ncx_dir / src).relative_to(extract_dir)
                resolved_str = resolved.as_posix()
                if resolved_str in href_to_title:
                    return href_to_title[resolved_str]

            # 3. Match by basename + fragment
            basename = Path(src.split('#')[0]).name
            if '#' in src:
                basename += '#' + src.split('#')[1]
            if basename in basename_to_title:
                return basename_to_title[basename]

            return None

        try:
            tree = ET.parse(ncx_path)
            root = tree.getroot()

            # Update docTitle (try multiple namespace approaches)
            translated_title = metadata.get('translated_title')
            if translated_title:
                # Try with namespace
                ns = {'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}
                doc_title = root.find('.//ncx:docTitle/ncx:text', ns)
                if doc_title is None:
                    # Try without namespace
                    for elem in root.iter():
                        if elem.tag.endswith('}text') or elem.tag == 'text':
                            parent = elem.getparent() if hasattr(elem, 'getparent') else None
                            if parent is not None and (parent.tag.endswith('}docTitle') or parent.tag == 'docTitle'):
                                doc_title = elem
                                break
                if doc_title is not None:
                    doc_title.text = translated_title

            # Update navPoints
            updated_count = 0
            for nav_point in root.iter():
                if nav_point.tag.endswith('}navPoint') or nav_point.tag == 'navPoint':
                    # Find content src
                    content = None
                    for child in nav_point:
                        if child.tag.endswith('}content') or child.tag == 'content':
                            content = child
                            break

                    if content is not None:
                        src = content.get('src', '')
                        translated = find_translation(src)

                        if translated:
                            # Find navLabel/text and update
                            for child in nav_point:
                                if child.tag.endswith('}navLabel') or child.tag == 'navLabel':
                                    for text_elem in child:
                                        if text_elem.tag.endswith('}text') or text_elem.tag == 'text':
                                            text_elem.text = translated
                                            updated_count += 1
                                            break
                                    break

            tree.write(ncx_path, encoding='utf-8', xml_declaration=True)
            logger.debug(f"Updated {updated_count} navPoints in toc.ncx")

        except Exception as e:
            logger.warning(f"Failed to update toc.ncx: {e}")

    def _update_nav_xhtml(self, extract_dir: Path, metadata: Dict):
        """Update nav.xhtml with translated chapter titles (EPUB 3)."""
        # Find nav document via OPF manifest (proper EPUB 3 way)
        opf_path = self._find_opf_path(extract_dir)
        nav_path = None

        if opf_path:
            toc_files = self._find_toc_files(extract_dir, opf_path)
            nav_path = toc_files.get('nav')

        if not nav_path:
            # Fallback to glob (for malformed EPUBs or when OPF doesn't specify nav)
            nav_candidates = (
                list(extract_dir.rglob("nav.xhtml")) +
                list(extract_dir.rglob("nav.html")) +
                list(extract_dir.rglob("*nav*.xhtml")) +
                list(extract_dir.rglob("*nav*.html"))
            )
            # Deduplicate
            nav_candidates = list(dict.fromkeys(nav_candidates))

            if not nav_candidates:
                logger.debug("No nav document found (EPUB 2 only?)")
                return

            nav_path = nav_candidates[0]
        nav_dir = nav_path.parent
        toc_entries = metadata.get('toc', [])
        if not toc_entries:
            return

        # Build multiple mappings for robust matching (like NCX)
        href_to_title = {}
        basename_to_title = {}
        for entry in toc_entries:
            href = entry.get('href', '')
            translated = entry.get('translated', '')
            if href and translated:
                href_to_title[href] = translated
                # Basename with fragment
                basename = Path(href.split('#')[0]).name
                if '#' in href:
                    basename += '#' + href.split('#')[1]
                basename_to_title[basename] = translated

        try:
            # Use BeautifulSoup for more robust HTML parsing
            from bs4 import BeautifulSoup

            content = nav_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'html.parser')

            updated_count = 0
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href', '')

                # Try multiple matching strategies
                translated = None

                # 1. Exact match
                if href in href_to_title:
                    translated = href_to_title[href]

                # 2. Resolve relative to nav dir
                if not translated and nav_dir != extract_dir:
                    try:
                        resolved = (nav_dir / href.split('#')[0]).relative_to(extract_dir)
                        resolved_str = resolved.as_posix()
                        if '#' in href:
                            resolved_str += '#' + href.split('#')[1]
                        if resolved_str in href_to_title:
                            translated = href_to_title[resolved_str]
                    except ValueError:
                        pass

                # 3. Match by basename + fragment
                if not translated:
                    basename = Path(href.split('#')[0]).name
                    if '#' in href:
                        basename += '#' + href.split('#')[1]
                    if basename in basename_to_title:
                        translated = basename_to_title[basename]

                if translated:
                    # Update the link text, handling both simple and nested cases
                    if a_tag.string is not None:
                        # Simple case: direct text content
                        a_tag.string = translated
                        updated_count += 1
                    else:
                        # Complex case: nested elements - replace all text content
                        # Clear existing content and set new text
                        a_tag.clear()
                        a_tag.string = translated
                        updated_count += 1

            # Write back, preserving original structure as much as possible
            nav_path.write_text(str(soup), encoding='utf-8')
            logger.debug(f"Updated {updated_count} entries in nav.xhtml")

        except ImportError:
            # Fallback to regex if BeautifulSoup not available
            logger.debug("BeautifulSoup not available, using regex fallback")
            import re
            content = nav_path.read_text(encoding='utf-8')
            updated_count = 0
            for href, translated in href_to_title.items():
                pattern = rf'(<a[^>]*href=["\']){re.escape(href)}(["\'][^>]*>)([^<]*)(<\/a>)'
                new_content = re.sub(pattern, rf'\g<1>{href}\g<2>{translated}\g<4>', content)
                if new_content != content:
                    updated_count += 1
                    content = new_content
            nav_path.write_text(content, encoding='utf-8')
            logger.debug(f"Updated {updated_count} entries in nav.xhtml (regex)")

        except Exception as e:
            logger.warning(f"Failed to update nav.xhtml: {e}")


def build_html_epub(
    original_epub: Path,
    translated_dir: Path,
    output_path: Optional[Path] = None,
    book_title: Optional[str] = None,
    translated_metadata: Optional[Dict] = None
) -> Path:
    """
    Convenience function to build translated EPUB.

    Args:
        original_epub: Path to original EPUB file
        translated_dir: Directory containing translated XHTML files
        output_path: Optional output path (defaults to translated_{original}.epub)
        book_title: Optional book title
        translated_metadata: Optional dict with translated_title and toc entries

    Returns:
        Path to the built EPUB file
    """
    if output_path is None:
        output_path = original_epub.parent / f"translated_{original_epub.name}"

    config = BuildConfig(
        original_epub=original_epub,
        translated_dir=translated_dir,
        output_path=output_path,
        book_title=book_title or original_epub.stem,
        translated_metadata=translated_metadata
    )

    builder = HTMLEpubBuilder(config)
    return builder.build()


class HTMLEpubPipeline:
    """
    Complete HTML translation pipeline using HTMLCompressor.

    Orchestrates the full workflow:
    1. Extract XHTML from EPUB
    2. Compress HTML (strip outer structure, save mapping)
    3. Translate compressed content (one line per translation unit)
    4. Translate title and TOC
    5. Decompress (reconstruct full HTML from translated lines + mapping)
    6. Build new EPUB

    Directory structure:
    - compressed_units/  : .md (compressed content) + .mapping.json
    - translated_compressed/ : .md (translated lines)
    - final_xhtml/ : .xhtml (fully reconstructed HTML)
    """

    def __init__(
        self,
        epub_path: Path,
        output_dir: Path,
        config: Dict
    ):
        self.epub_path = epub_path
        self.output_dir = output_dir
        self.config = config

        # Parse EPUB
        self.parser = EPUBParser(str(epub_path))

        # Extract metadata
        self.metadata = self.parser.metadata
        self.book_title = self.metadata.get('title', epub_path.stem)
        self.source_language = self._detect_language()

        # Setup directories (using new compressor-based workflow)
        self.compressed_units_dir = output_dir / "compressed_units"  # .md + .mapping.json
        self.translated_dir = output_dir / "translated_compressed"   # .md
        self.final_dir = output_dir / "final_xhtml"                  # .xhtml

        for d in [self.compressed_units_dir, self.translated_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _detect_language(self) -> str:
        """Detect source language from EPUB metadata."""
        lang_code = self.metadata.get('language', 'en')

        # Map language codes to full names
        lang_map = {
            'en': 'English',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'ko': 'Korean',
            'ru': 'Russian',
        }

        # Handle codes like 'en-US', 'zh-CN'
        base_code = lang_code.split('-')[0].lower()
        return lang_map.get(base_code, 'English')

    def _get_language_code(self, language: str) -> str:
        """Convert language name to ISO 639-1 code for EPUB metadata."""
        # Map full names to language codes (reverse of _detect_language)
        name_to_code = {
            'english': 'en',
            'japanese': 'ja',
            'chinese': 'zh',
            'german': 'de',
            'french': 'fr',
            'spanish': 'es',
            'korean': 'ko',
            'russian': 'ru',
            '中文': 'zh',
            '日本語': 'ja',
            '한국어': 'ko',
        }

        lang_lower = language.lower()
        return name_to_code.get(lang_lower, 'zh')  # Default to 'zh' for Chinese

    def extract_and_preprocess(self) -> int:
        """
        Extract XHTML from EPUB and compress for translation.

        Uses HTMLCompressor to strip outer structure and save mapping.
        Only translatable content is output (no empty placeholders).

        Returns:
            Number of files extracted
        """
        import json
        from .compressor import HTMLCompressor
        from .verified_compactor import VerifiedCompactor

        # Get all CSS content for compactor
        css_content = ""
        for css_item in self.parser.resources.get('css', []):
            try:
                content = css_item.get('content', b'')
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                css_content += content + "\n"
            except Exception as e:
                logger.warning(f"Failed to read CSS: {e}")

        # Create compactor with CSS for DOM optimization (oracle-verified)
        compactor = VerifiedCompactor(css_content) if css_content else None
        compressor = HTMLCompressor(compactor=compactor)

        if compactor:
            logger.info(
                f"VerifiedCompactor: {compactor.oracle.selector_count} selectors, "
                f"{len(compactor.oracle.failed_selectors)} failed"
            )

        extracted = 0

        for item in self.parser.spine:
            href = item.get('href', '')
            if not href:
                continue

            try:
                # Get raw content directly from EPUB ZIP
                # (bypasses ebooklib which modifies HTML - removes body class, head content)
                content = self.parser.get_raw_content(href)
                if content is None:
                    logger.warning(f"Item not found: {href}")
                    continue

                # Decode if bytes
                if isinstance(content, bytes):
                    content = content.decode('utf-8')

                # Compress HTML
                compressed, mapping = compressor.compress(content)

                # Save compressed content (.md - one translation unit per line)
                href_path = Path(href)
                file_stem = href_path.stem
                original_extension = href_path.suffix  # .html or .xhtml
                compressed_path = self.compressed_units_dir / f"{file_stem}.md"
                compressed_path.write_text(compressed, encoding='utf-8')

                # Save mapping for decompression (.mapping.json)
                # Include original extension for correct output filename
                mapping['original_extension'] = original_extension
                mapping_path = self.compressed_units_dir / f"{file_stem}.mapping.json"
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False)

                extracted += 1
                lines = len(compressed.splitlines()) if compressed else 0
                logger.debug(f"Compressed: {file_stem} ({lines} translatable lines)")

            except Exception as e:
                logger.warning(f"Failed to extract {href}: {e}")

        logger.info(f"Compressed {extracted} XHTML files to {self.compressed_units_dir}")
        return extracted

    def _merge_part_files(self) -> Dict[str, str]:
        """
        Merge split part files (*.part1.md, *.part2.md, etc.) into combined content.

        When files are split for translation due to size limits, they produce
        files like split_023.part1.md, split_023.part2.md. This method merges
        them back together.

        Returns:
            Dict mapping base_name -> merged_content
        """
        import re
        from collections import defaultdict

        # Group files by base name
        part_files = defaultdict(list)
        part_pattern = re.compile(r'^(.+)\.part(\d+)\.md$')

        for f in self.translated_dir.glob("*.part*.md"):
            match = part_pattern.match(f.name)
            if match:
                base_name = match.group(1)
                part_num = int(match.group(2))
                part_files[base_name].append((part_num, f))

        # Merge each group in order
        merged = {}
        for base_name, parts in part_files.items():
            # Sort by part number
            parts.sort(key=lambda x: x[0])

            # Concatenate content
            content_parts = []
            for part_num, part_file in parts:
                content_parts.append(part_file.read_text(encoding='utf-8'))

            merged[base_name] = '\n'.join(content_parts)
            logger.debug(f"Merged {len(parts)} parts for {base_name}")

        return merged

    def postprocess_and_build(self, output_epub: Optional[Path] = None) -> Path:
        """
        Decompress translated content and build final EPUB.

        Uses HTMLCompressor to reconstruct full HTML from translated
        compressed content and saved mappings.

        Args:
            output_epub: Optional output path

        Returns:
            Path to built EPUB
        """
        import json
        import re
        from .compressor import HTMLCompressor

        compressor = HTMLCompressor()

        # First, merge any split part files
        merged_parts = self._merge_part_files()
        if merged_parts:
            logger.info(f"Merged {len(merged_parts)} split files")

        # Track which base names have been processed (to avoid duplicates)
        processed_bases = set()

        # Decompress each translated file
        for translated_file in self.translated_dir.glob("*.md"):
            # Skip part files (they've been merged)
            if re.match(r'.*\.part\d+\.md$', translated_file.name):
                continue

            base_stem = translated_file.stem
            mapping_path = self.compressed_units_dir / f"{base_stem}.mapping.json"

            if mapping_path.exists():
                # Load mapping
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)

                # Check if we have merged content for this file
                if base_stem in merged_parts:
                    # Use merged content from parts
                    content = merged_parts[base_stem]
                    logger.debug(f"Using merged content for {base_stem}")
                else:
                    # Read translated compressed content directly
                    content = translated_file.read_text(encoding='utf-8')

                # Decompress to full HTML
                restored = compressor.decompress(content, mapping)

                # Save to final directory with original extension (.html or .xhtml)
                original_ext = mapping.get('original_extension', '.xhtml')
                final_path = self.final_dir / f"{base_stem}{original_ext}"
                final_path.write_text(restored, encoding='utf-8')

                processed_bases.add(base_stem)
                logger.debug(f"Decompressed: {base_stem}")
            else:
                # Check if this is a base name that only has part files
                if base_stem in merged_parts:
                    # We have merged parts but no direct file - need to find mapping
                    # This shouldn't happen normally, but handle it
                    logger.warning(f"Found merged parts for {base_stem} but no base file")
                else:
                    logger.warning(f"No mapping found for {translated_file.name}")

        # Handle cases where we have merged parts but no base .md file
        for base_stem, content in merged_parts.items():
            if base_stem in processed_bases:
                continue

            mapping_path = self.compressed_units_dir / f"{base_stem}.mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)

                restored = compressor.decompress(content, mapping)
                original_ext = mapping.get('original_extension', '.xhtml')
                final_path = self.final_dir / f"{base_stem}{original_ext}"
                final_path.write_text(restored, encoding='utf-8')

                logger.debug(f"Decompressed from merged parts: {base_stem}")
            else:
                logger.warning(f"No mapping found for merged parts: {base_stem}")

        # Load translated metadata if available
        metadata_path = self.output_dir / "translated_metadata.json"
        translated_metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                translated_metadata = json.load(f)

        # Build EPUB with translated title as filename
        if output_epub is None:
            if translated_metadata and translated_metadata.get('translated_title'):
                safe_title = sanitize_filename(translated_metadata['translated_title'])
                output_epub = self.output_dir / f"{safe_title}.epub"
            else:
                output_epub = self.output_dir / f"translated_{self.epub_path.name}"

        return build_html_epub(
            original_epub=self.epub_path,
            translated_dir=self.final_dir,
            output_path=output_epub,
            book_title=self.book_title,
            translated_metadata=translated_metadata
        )

    def translate_metadata(
        self,
        target_language: str = "Chinese",
        llm_client=None,
        force: bool = False,
        batch_size: int = 50
    ) -> Dict:
        """
        Translate book title and TOC using JSON format and batch processing.

        Args:
            target_language: Target language
            llm_client: LLM client for translation
            force: Force re-translation even if cached
            batch_size: Number of TOC entries per batch

        Returns:
            Dict with translated_title and translated_toc
        """
        import json

        # Check for existing translated metadata (resume support)
        metadata_path = self.output_dir / "translated_metadata.json"
        if metadata_path.exists() and not force:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            logger.info(f"Using existing translated metadata: {existing['translated_title']}")
            return existing

        from .toc_extractor import TOCExtractor

        if llm_client is None:
            from pdf2epub.utils.llm_client import LLMClient
            llm_client = LLMClient(self.config)

        # Get TOC (flattened to include all levels)
        toc_extractor = TOCExtractor(self.parser)
        flat_toc = toc_extractor.get_flat_toc()

        # Translate book title first
        translated_title = self._translate_title(
            self.book_title, self.source_language, target_language, llm_client
        )

        # Translate TOC entries in batches
        translated_toc = self._translate_toc_batched(
            flat_toc, self.source_language, target_language, llm_client, batch_size
        )

        # Map target language to ISO 639-1 code for EPUB metadata
        target_lang_code = self._get_language_code(target_language)

        # Build result
        result = {
            'original_title': self.book_title,
            'translated_title': translated_title,
            'target_language': target_language,
            'target_language_code': target_lang_code,
            'toc': translated_toc
        }

        # Save to file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Translated title: {result['translated_title']}")
        logger.info(f"Translated {len(result['toc'])} TOC entries")

        return result

    def _translate_title(
        self,
        title: str,
        source_lang: str,
        target_lang: str,
        llm_client
    ) -> str:
        """Translate book title."""
        prompt = self._create_title_prompt(title, source_lang, target_lang)

        response = llm_client.generate(
            prompt=prompt,
            model_configs=self.config.get('translation_models', [
                {"provider": "gemini", "model": "gemini-2.5-pro"}
            ]),
            operation_name="Translate title"
        )

        # Clean up response
        translated = response.strip()
        # Remove quotes if present
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1]
        if translated.startswith('《') and translated.endswith('》'):
            translated = translated[1:-1]

        return translated or title

    def _translate_toc_batched(
        self,
        flat_toc: List[Dict],
        source_lang: str,
        target_lang: str,
        llm_client,
        batch_size: int
    ) -> List[Dict]:
        """Translate TOC entries in batches using JSON format."""
        import json

        if not flat_toc:
            return []

        # Build original -> entry mapping for later
        results = []
        total_batches = (len(flat_toc) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(flat_toc))
            batch = flat_toc[start:end]

            logger.info(f"Translating TOC batch {batch_idx + 1}/{total_batches} ({len(batch)} entries)")

            # Create batch prompt
            prompt = self._create_toc_batch_prompt(batch, source_lang, target_lang)

            # Call LLM
            response = llm_client.generate(
                prompt=prompt,
                model_configs=self.config.get('translation_models', [
                    {"provider": "gemini", "model": "gemini-2.5-pro"}
                ]),
                operation_name=f"Translate TOC batch {batch_idx + 1}/{total_batches}"
            )

            # Parse JSON response
            translations = self._parse_toc_json_response(response, batch)

            # Build result entries
            for entry, translated in zip(batch, translations):
                href = entry['href']
                if entry.get('anchor'):
                    href = f"{href}#{entry['anchor']}"

                results.append({
                    'original': entry['title'],
                    'translated': translated,
                    'level': entry['level'],
                    'href': href
                })

        return results

    def _create_title_prompt(self, title: str, source_lang: str, target_lang: str) -> str:
        """Create prompt for translating book title."""
        if target_lang.lower() in ['chinese', '中文', 'zh']:
            return f"""请将以下书名从{source_lang}翻译成简体中文。只返回翻译结果，不要添加任何解释或标点。

书名：{title}

翻译："""
        else:
            return f"""Translate the following book title from {source_lang} to {target_lang}. Return only the translation, no explanation.

Title: {title}

Translation:"""

    def _create_toc_batch_prompt(
        self,
        batch: List[Dict],
        source_lang: str,
        target_lang: str
    ) -> str:
        """Create prompt for translating a batch of TOC entries using JSON format."""
        import json

        # Build input JSON
        input_entries = [{"original": entry['title']} for entry in batch]
        input_json = json.dumps(input_entries, ensure_ascii=False, indent=2)

        if target_lang.lower() in ['chinese', '中文', 'zh']:
            return f"""请将以下目录条目从{source_lang}翻译成简体中文。

要求：
1. 返回JSON数组格式
2. 每个对象包含 "original"（原文）和 "translated"（译文）两个字段
3. 保持条目顺序不变
4. 保持学术著作的专业性和准确性
5. 只返回JSON，不要添加任何解释

输入：
{input_json}

输出JSON："""
        else:
            return f"""Translate the following TOC entries from {source_lang} to {target_lang}.

Requirements:
1. Return as JSON array
2. Each object should have "original" and "translated" fields
3. Maintain the same order
4. Maintain academic professionalism
5. Return only JSON, no explanation

Input:
{input_json}

Output JSON:"""

    def _parse_toc_json_response(self, response: str, batch: List[Dict]) -> List[str]:
        """Parse JSON response from LLM for TOC translation."""
        import json
        import re

        # Try to extract JSON from response
        response = response.strip()

        # Remove markdown code block if present
        if response.startswith('```'):
            # Find the end of code block
            lines = response.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith('```') and not in_block:
                    in_block = True
                    continue
                elif line.startswith('```') and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response = '\n'.join(json_lines)

        # Try to parse as JSON
        try:
            parsed = json.loads(response)

            if isinstance(parsed, list):
                # Build original -> translated mapping
                translation_map = {}
                for item in parsed:
                    if isinstance(item, dict):
                        orig = item.get('original', '')
                        trans = item.get('translated', '')
                        if orig and trans:
                            translation_map[orig] = trans

                # Match back to batch order
                results = []
                for entry in batch:
                    title = entry['title']
                    # Try exact match first
                    if title in translation_map:
                        results.append(translation_map[title])
                    else:
                        # Try fuzzy match (strip whitespace, case-insensitive)
                        found = False
                        for orig, trans in translation_map.items():
                            if orig.strip().lower() == title.strip().lower():
                                results.append(trans)
                                found = True
                                break
                        if not found:
                            logger.warning(f"No translation found for: {title[:50]}")
                            results.append(title)  # Fallback to original

                return results

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: try line-by-line parsing
        logger.warning("Falling back to line-by-line parsing")
        lines = response.strip().split('\n')
        translations = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering prefix
            cleaned = re.sub(r'^\d+[\.\、\)\]]\s*', '', line)
            # Remove JSON-like formatting
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
            if cleaned:
                translations.append(cleaned)

        # Pad or trim to match batch size
        while len(translations) < len(batch):
            translations.append(batch[len(translations)]['title'])

        return translations[:len(batch)]
