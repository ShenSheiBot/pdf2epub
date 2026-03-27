"""
Novel Extractor: Convert EPUB XHTML to plain text for translation.

Converts HTML to plain text preserving:
- Ruby annotations as parenthesis format: 経緯(いきさつ)
- Image references as [Image: filename.jpg]
- Paragraph structure (one line per paragraph)
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from lxml import etree

logger = logging.getLogger(__name__)

XHTML_NS = "http://www.w3.org/1999/xhtml"
XLINK_NS = "http://www.w3.org/1999/xlink"


@dataclass
class NovelUnit:
    """A single translatable unit from the EPUB."""
    spine_index: int
    file_name: str
    text_path: Optional[Path]
    has_content: bool
    image_refs: List[str] = field(default_factory=list)
    toc_title: Optional[str] = None  # Chapter title from TOC (for glossary injection)
    source_href: Optional[str] = None  # Original XHTML href in EPUB


class NovelExtractor:
    """Extract EPUB XHTML files to plain text for novel translation."""

    def __init__(self, parser):
        """
        Args:
            parser: EPUBParser instance with spine and content access.
        """
        self.parser = parser

    def extract_all(self, output_dir: Path) -> List[NovelUnit]:
        """Extract all spine items to plain text files in reading order.

        Args:
            output_dir: Directory to write .txt files to.

        Returns:
            List of NovelUnit in spine order.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        units = []

        for idx, item in enumerate(self.parser.spine):
            href = item.get('full_href') or item.get('href', '')
            file_name = Path(href).stem

            try:
                xhtml_bytes = self.parser.get_raw_content(href)
                if isinstance(xhtml_bytes, bytes):
                    xhtml_content = xhtml_bytes.decode('utf-8')
                else:
                    xhtml_content = xhtml_bytes
            except Exception as e:
                logger.warning(f"Failed to read {href}: {e}")
                units.append(NovelUnit(
                    spine_index=idx,
                    file_name=file_name,
                    text_path=None,
                    has_content=False,
                    source_href=href,
                ))
                continue

            text, image_refs = self._convert_xhtml_to_text(xhtml_content)

            # Check if there's actual translatable text (not just image placeholders)
            text_stripped = text.strip()
            # Remove image placeholders before checking
            text_no_images = re.sub(r'\[Image:[^\]]*\]', '', text_stripped).strip()
            has_content = bool(text_no_images) and any(
                '\u3000' <= c <= '\u9fff' or  # CJK
                '\u3040' <= c <= '\u30ff' or  # Hiragana/Katakana
                c.isalpha()
                for c in text_no_images
            )

            # Always write a file for every spine item (1:1 mapping)
            out_path = output_dir / f"{idx:03d}_{file_name}.txt"
            if has_content:
                out_path.write_text(text_stripped, encoding='utf-8')
                logger.info(f"Extracted {href} → {out_path.name} ({len(text_stripped)} chars, {len(image_refs)} images)")
            elif image_refs:
                out_path.write_text(text_stripped if text_stripped else "\n".join(f"[Image: {r}]" for r in image_refs), encoding='utf-8')
                logger.info(f"Extracted {href} → {out_path.name} (image-only, {len(image_refs)} images)")
            else:
                # Empty/decorative spine item — write placeholder
                out_path.write_text("", encoding='utf-8')
                logger.info(f"Extracted {href} → {out_path.name} (empty placeholder)")

            units.append(NovelUnit(
                spine_index=idx,
                file_name=file_name,
                text_path=out_path,
                has_content=has_content,
                image_refs=image_refs,
                source_href=href,
            ))

        content_count = sum(1 for u in units if u.has_content)
        logger.info(f"Extracted {len(units)} spine items, {content_count} with translatable content")
        return units

    def _convert_xhtml_to_text(self, xhtml_content: str) -> tuple:
        """Convert a single XHTML file to plain text.

        Returns:
            (text, image_refs) tuple.
        """
        image_refs = []

        # Parse HTML
        try:
            # Try as XML first
            tree = etree.fromstring(xhtml_content.encode('utf-8'))
        except etree.XMLSyntaxError:
            try:
                parser = etree.HTMLParser(encoding='utf-8')
                tree = etree.fromstring(xhtml_content.encode('utf-8'), parser)
            except Exception as e:
                logger.warning(f"Failed to parse XHTML: {e}")
                return "", []

        # Find body element
        body = tree.find(f'.//{{{XHTML_NS}}}body')
        if body is None:
            body = tree.find('.//body')
        if body is None:
            body = tree

        raw_items = []
        self._extract_text_recursive(body, raw_items, image_refs)

        # Merge consecutive inline items into single lines, blocks stay separate
        lines = []
        inline_buf = []
        for kind, text in raw_items:
            if kind == 'block':
                if inline_buf:
                    lines.append(''.join(inline_buf))
                    inline_buf = []
                lines.append(text)
            else:  # inline
                inline_buf.append(text)
        if inline_buf:
            lines.append(''.join(inline_buf))

        return "\n".join(lines), image_refs

    def _extract_text_recursive(self, element, lines: list, image_refs: list):
        """Recursively extract text from element tree."""
        # Skip comment nodes (e.g. <!-- <script ...> -->)
        if not isinstance(element.tag, str):
            return

        tag = etree.QName(element.tag).localname

        # Skip non-content elements
        if tag in ('script', 'style', 'head', 'meta', 'link'):
            return

        # Handle specific elements
        if tag == 'ruby':
            text = self._ruby_to_parentheses(element)
            if text:
                # Append to current line context - handled by caller
                lines.append(('inline', text))
            return

        if tag == 'img':
            src = element.get('src', '')
            if src:
                src_name = Path(src).name
                image_refs.append(src_name)
                lines.append(('block', f'[Image: {src_name}]'))
            return

        if tag == 'image':
            # SVG image element
            href = element.get(f'{{{XLINK_NS}}}href') or element.get('href', '')
            if href:
                href_name = Path(href).name
                image_refs.append(href_name)
                lines.append(('block', f'[Image: {href_name}]'))
            return

        if tag == 'svg':
            # Look for image inside SVG
            for child in element.iter():
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'image':
                    href = child.get(f'{{{XLINK_NS}}}href') or child.get('href', '')
                    if href:
                        href_name = Path(href).name
                        image_refs.append(href_name)
                        lines.append(('block', f'[Image: {href_name}]'))
            return

        # Block elements - each gets its own line
        is_block = tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                          'blockquote', 'li', 'tr', 'section', 'article',
                          'nav', 'header', 'footer', 'aside', 'main')

        if is_block:
            # Check if this block contains child block elements
            child_has_block = any(
                (etree.QName(c.tag).localname if isinstance(c.tag, str) else '')
                in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'blockquote', 'li', 'tr', 'section', 'article',
                    'nav', 'header', 'footer', 'aside', 'main', 'ul', 'ol', 'table')
                for c in element
            )

            if child_has_block:
                # Recurse into children so nested blocks get their own lines
                if element.text and element.text.strip():
                    lines.append(('inline', element.text.strip()))
                for child in element:
                    self._extract_text_recursive(child, lines, image_refs)
                    if child.tail and child.tail.strip():
                        lines.append(('inline', child.tail.strip()))
            else:
                # Leaf block — collect inline content
                inline_parts = []
                self._collect_inline(element, inline_parts, image_refs)
                text = ''.join(inline_parts).strip()
                if text:
                    lines.append(('block', text))
            return

        # For non-block elements, recurse into children
        # Handle text before first child
        if element.text and element.text.strip():
            lines.append(('inline', element.text.strip()))

        for child in element:
            self._extract_text_recursive(child, lines, image_refs)
            # Handle tail text
            if child.tail and child.tail.strip():
                lines.append(('inline', child.tail.strip()))

    def _collect_inline(self, element, parts: list, image_refs: list):
        """Collect inline text content from an element."""
        if not isinstance(element.tag, str):
            return
        tag = etree.QName(element.tag).localname

        if tag == 'ruby':
            parts.append(self._ruby_to_parentheses(element))
            return

        if tag == 'img':
            src = element.get('src', '')
            if src:
                src_name = Path(src).name
                image_refs.append(src_name)
                parts.append(f'[Image: {src_name}]')
            return

        if tag == 'br':
            parts.append('\n')
            return

        # Add element's text
        if element.text:
            parts.append(element.text)

        # Process children
        for child in element:
            self._collect_inline(child, parts, image_refs)
            # Tail text after child
            if child.tail:
                parts.append(child.tail)

    def _ruby_to_parentheses(self, ruby_element) -> str:
        """Convert <ruby> element to text(reading) format.

        <ruby><rb>経緯</rb><rt>いきさつ</rt></ruby> → 経緯(いきさつ)
        """
        base_text = ""
        reading_text = ""

        for child in ruby_element:
            child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''

            if child_tag == 'rb':
                base_text += (child.text or '')
                # Also collect any nested text
                for sub in child:
                    base_text += (sub.text or '') + (sub.tail or '')
            elif child_tag == 'rt':
                reading_text += (child.text or '')
            elif child_tag == 'rp':
                pass  # Skip ruby parentheses markers

        # Also handle direct text in ruby element (no <rb> wrapper)
        if not base_text and ruby_element.text:
            base_text = ruby_element.text

        if base_text and reading_text:
            return f"{base_text}({reading_text})"
        return base_text or reading_text or ''
