"""
HTML Compressor for translation optimization.

Compresses HTML by stripping outer block elements and recording structure mapping.
Saves ~40% tokens for LLM translation while enabling perfect reconstruction.

Uses lxml for all DOM operations, outputs XHTML-compliant XML.
"""

import re
from html import escape
from typing import Dict, List, Tuple, Any, Optional
from lxml import etree
from lxml import html as lxml_html
from loguru import logger
from .display_resolver import resolve_display


class HTMLCompressor:
    """
    HTML compressor that strips outer structure for translation efficiency.

    Compression rules:
    1. Block elements: Strip outer wrapper, preserve content
    2. Single-chain nesting: Strip completely to plain text
    3. Multi-sibling: Keep inner structure, strip attrs
    4. Naked text: Preserve directly
    5. Void elements: Record position, empty placeholder
    """

    def __init__(self, compactor=None):
        """
        Initialize HTMLCompressor.

        Args:
            compactor: Optional HTMLCompactor instance for pre-processing
                      nested HTML to reduce element count before stripping.
        """
        self.compactor = compactor

    VOID_ELEMENTS = {
        'img', 'br', 'hr', 'input', 'meta', 'link', 'area', 'base',
        'col', 'embed', 'source', 'track', 'wbr'
    }

    # Types that need translation (have content to translate)
    TRANSLATABLE_TYPES = {'naked', 'block', 'inline', 'inline_run'}

    def _parse_html(self, html: str):
        """Parse HTML string, handling XML declarations properly.

        lxml_html.fromstring() fails on unicode strings with XML declarations.
        This method encodes to bytes first when needed.
        """
        # Check for XML declaration or encoding declaration
        if html.lstrip().startswith('<?xml') or 'encoding=' in html[:200]:
            return lxml_html.fromstring(html.encode('utf-8'))
        return lxml_html.fromstring(html)

    def compress(self, html: str, author_css: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        Compress HTML for translation.

        Args:
            html: Input HTML string

        Returns:
            Tuple of (compressed_text, mapping):
            - compressed_text: One translation unit per line (only translatable content)
            - mapping: Dict with 'wrapper' (for full HTML reconstruction) and 'units' (for body content)

        Note:
            Only translatable types (naked, block, inline) produce output lines.
            Non-translatable types (void, comment, block_open/close, empty_*) are
            reconstructed from mapping without consuming translation lines.
        """
        # Parse with lxml HTML parser (tolerant mode)
        # Encode to bytes if string contains XML declaration (lxml requirement)
        root = self._parse_html(html)

        # Build CSS display map for block/inline detection
        self._display_map = resolve_display(root, author_css)

        units: List[str] = []
        unit_mappings: List[Dict[str, Any]] = []

        # Extract wrapper info for full HTML reconstruction
        wrapper = self._extract_wrapper(html, root)

        # Find the body or use entire root
        body = root.find('.//body')
        if body is None:
            body = root

        # Process body children, grouping consecutive inline content
        self._process_mixed_children(body, units, unit_mappings)

        mapping = {
            'wrapper': wrapper,
            'units': unit_mappings
        }

        # Mark units with actual content and only output those
        for i, m in enumerate(unit_mappings):
            if m.get('type') in self.TRANSLATABLE_TYPES:
                m['has_content'] = bool(units[i].strip())

        translatable_units = [
            units[i] for i, m in enumerate(unit_mappings)
            if m.get('type') in self.TRANSLATABLE_TYPES and m.get('has_content')
        ]

        return '\n'.join(translatable_units), mapping

    def _extract_wrapper(self, html: str, root) -> Dict[str, Any]:
        """Extract HTML wrapper info (everything except body content)."""
        wrapper: Dict[str, Any] = {}

        # Extract XML declaration if present
        xml_match = re.match(r'^(<\?xml[^?]*\?>)\s*', html)
        if xml_match:
            wrapper['xml_declaration'] = xml_match.group(1)

        # Keep the publication's declared HTML dialect. Reconstructing every
        # document with a hard-coded doctype can turn a valid EPUB 3 document
        # into invalid XHTML 1.1 (or vice versa).
        doctype_match = re.search(r'<!DOCTYPE[^>]*>', html, re.IGNORECASE)
        if doctype_match:
            wrapper['doctype'] = doctype_match.group(0)

        # Extract html tag attributes
        html_tag = root if root.tag == 'html' else root.find('.//html')
        if html_tag is not None:
            wrapper['html_attrs'] = dict(html_tag.attrib)

        # Extract head content
        head_tag = root.find('.//head')
        if head_tag is not None:
            # Temporarily clear tail to avoid including whitespace after </head>
            original_tail = head_tag.tail
            head_tag.tail = None
            head_html = etree.tostring(head_tag, encoding='unicode', method='xml')
            head_tag.tail = original_tail
            # Remove CR characters that cause lxml HTML parser issues
            head_html = head_html.replace('&#13;', '')
            wrapper['head_html'] = head_html

        # Extract body attributes
        body_tag = root.find('.//body')
        if body_tag is not None:
            wrapper['body_attrs'] = dict(body_tag.attrib)

        return wrapper

    def decompress(self, translated: str, mapping: Dict[str, Any]) -> str:
        """
        Reconstruct full HTML from translated content and mapping.

        Args:
            translated: Translated content (one unit per line, only translatable types)
            mapping: Mapping from compress() with 'wrapper' and 'units'

        Returns:
            Reconstructed full HTML (XHTML-compliant)

        Note:
            Only translatable types (naked, block, inline) consume lines from translated.
            Non-translatable types are reconstructed directly from mapping.
        """
        # 过滤空行（合并 parts 时 \n\n 可能产生空行）
        lines = [line for line in translated.split('\n') if line.strip()]
        body_parts: List[str] = []
        line_idx = 0  # Index into translated lines (only for translatable types)

        # Get unit mappings
        unit_mappings = mapping.get('units', [])

        for unit_map in unit_mappings:
            unit_type = unit_map.get('type', 'unknown')

            if unit_type == 'naked':
                # Translatable: consume a line only if has_content
                if unit_map.get('has_content', True):
                    line = lines[line_idx] if line_idx < len(lines) else ''
                    line_idx += 1
                    body_parts.append(line)
                else:
                    body_parts.append('')  # Empty naked text

            elif unit_type == 'comment':
                # Non-translatable: restore from mapping
                body_parts.append(f'<!--{unit_map["content"]}-->')

            elif unit_type == 'void':
                # Non-translatable: restore from mapping (use XML self-closing)
                tag = unit_map['tag']
                attrs_str = self._format_attrs(unit_map.get('attrs', {}))
                body_parts.append(f'<{tag}{attrs_str}/>')

            elif unit_type == 'empty_block':
                # Non-translatable: restore from mapping
                outer_path = unit_map.get('outer_path', [])
                content = ''
                for tag, attrs in reversed(outer_path):
                    attrs_str = self._format_attrs(attrs)
                    content = f'<{tag}{attrs_str}>{content}</{tag}>'
                body_parts.append(content)

            elif unit_type == 'empty_inline':
                # Non-translatable: restore from mapping
                tag = unit_map['tag']
                attrs_str = self._format_attrs(unit_map.get('attrs', {}))
                body_parts.append(f'<{tag}{attrs_str}></{tag}>')

            elif unit_type == 'block_open':
                # Non-translatable: restore from mapping
                tag = unit_map['tag']
                attrs_str = self._format_attrs(unit_map.get('attrs', {}))
                body_parts.append(f'<{tag}{attrs_str}>')

            elif unit_type == 'block_close':
                # Non-translatable: restore from mapping
                tag = unit_map['tag']
                body_parts.append(f'</{tag}>')

            elif unit_type in ('block', 'inline'):
                # Translatable: consume a line only if has_content
                if unit_map.get('has_content', True):
                    content = lines[line_idx] if line_idx < len(lines) else ''
                    line_idx += 1
                else:
                    content = ''  # No content, don't consume line

                # Restore inner attrs if present
                if unit_map.get('inner_tags') and unit_map.get('inner_attr_map'):
                    content = self._restore_inner_attrs(content, unit_map['inner_attr_map'])

                # Wrap with outer tags
                outer_path = unit_map.get('outer_path', [])
                for tag, attrs in reversed(outer_path):
                    attrs_str = self._format_attrs(attrs)
                    content = f'<{tag}{attrs_str}>{content}</{tag}>'

                body_parts.append(content)

            elif unit_type == 'inline_run':
                # Translatable inline run: consume a line only if has_content
                if unit_map.get('has_content', True):
                    content = lines[line_idx] if line_idx < len(lines) else ''
                    line_idx += 1
                else:
                    content = ''

                # Restore inner attrs if present
                if unit_map.get('inner_tags') and unit_map.get('inner_attr_map'):
                    content = self._restore_inner_attrs(content, unit_map['inner_attr_map'])

                body_parts.append(content)

            else:
                # Unknown type - skip (shouldn't happen)
                pass

        # Reconstruct body content
        body_content = ''.join(body_parts)

        # Reconstruct full HTML with wrapper
        return self._reconstruct_html(body_content, mapping.get('wrapper', {}))

    def _reconstruct_html(self, body_content: str, wrapper: Dict[str, Any]) -> str:
        """Reconstruct full HTML document from body content and wrapper info."""
        parts: List[str] = []

        # XML declaration
        if wrapper.get('xml_declaration'):
            parts.append(wrapper['xml_declaration'])
            parts.append('\n')

        if wrapper.get('doctype'):
            parts.append(wrapper['doctype'])
            parts.append('\n')

        # Opening html tag
        html_attrs = self._format_attrs(wrapper.get('html_attrs', {}))
        parts.append(f'<html{html_attrs}>')
        parts.append('\n')

        # Head
        if wrapper.get('head_html'):
            parts.append(wrapper['head_html'])
            parts.append('\n')

        # Body
        body_attrs = self._format_attrs(wrapper.get('body_attrs', {}))
        parts.append(f'<body{body_attrs}>')
        parts.append(body_content)
        parts.append('</body>')

        # Closing html
        parts.append('</html>')
        parts.append('\n')

        return ''.join(parts)

    def _process_element(
        self,
        elem,
        units: List[str],
        mapping: List[Dict[str, Any]]
    ) -> None:
        """Process a single element recursively."""

        # Handle Comment nodes
        if isinstance(elem, etree._Comment):
            units.append('')
            mapping.append({
                'type': 'comment',
                'content': str(elem.text) if elem.text else ''
            })
            return

        # Handle ProcessingInstruction and other special nodes
        if not isinstance(elem, etree._Element) or elem.tag is etree.Comment:
            return

        tag_name = elem.tag

        # Handle void elements
        if tag_name in self.VOID_ELEMENTS:
            units.append('')  # Empty placeholder
            mapping.append({
                'type': 'void',
                'tag': tag_name,
                'attrs': dict(elem.attrib)
            })
            return

        # Handle block elements (determined by CSS display). Some EPUBs contain
        # invalid-but-common markup like <span><div class="para">...</div></span>.
        # Treat inline wrappers that contain block descendants as containers;
        # otherwise whole page sections collapse into one huge translation unit.
        if self._is_block_container(elem):
            # Check if contains nested block children
            if self._has_block_children(elem):
                # Container block: add open marker, recurse, add close marker
                units.append('')
                mapping.append({
                    'type': 'block_open',
                    'tag': tag_name,
                    'attrs': dict(elem.attrib)
                })

                self._process_mixed_children(elem, units, mapping)

                units.append('')
                mapping.append({
                    'type': 'block_close',
                    'tag': tag_name
                })
            else:
                # Leaf block: extract content
                self._process_leaf_block(elem, units, mapping)
        else:
            # Non-block element (e.g., bare span)
            children = list(elem)
            if not children and not elem.text:
                # Empty inline element - preserve it
                units.append('')
                mapping.append({
                    'type': 'empty_inline',
                    'tag': tag_name,
                    'attrs': dict(elem.attrib)
                })
            elif self._is_single_chain(elem):
                # Single-chain inline: extract text and save path
                text, path = self._extract_chain(elem)
                units.append(self._normalize_whitespace(text))
                mapping.append({
                    'type': 'inline',
                    'outer_path': path,
                    'inner_tags': False
                })
            else:
                # Multi-sibling inline: keep inner structure, strip attrs
                inner = self._get_inner_html(elem)
                stripped, attr_map = self._strip_inner_attrs(inner)
                units.append(self._normalize_whitespace(stripped))
                mapping.append({
                    'type': 'inline',
                    'outer_path': [(elem.tag, dict(elem.attrib))],
                    'inner_tags': True,
                    'inner_attr_map': attr_map
                })

    def _process_leaf_block(
        self,
        elem,
        units: List[str],
        mapping: List[Dict[str, Any]]
    ) -> None:
        """Process a leaf block element (no block children)."""

        # Check for empty element
        inner = self._get_inner_html(elem)
        if not inner.strip():
            units.append('')
            mapping.append({
                'type': 'empty_block',
                'outer_path': [(elem.tag, dict(elem.attrib))]
            })
            return

        # Check if single-chain nesting
        if self._is_single_chain(elem):
            # Extract text and full path
            text, path = self._extract_chain(elem)
            units.append(self._normalize_whitespace(text))
            mapping.append({
                'type': 'block',
                'outer_path': path,
                'inner_tags': False
            })
        else:
            # Multi-sibling: keep inner structure, strip attrs
            stripped, attr_map = self._strip_inner_attrs(inner)
            units.append(self._normalize_whitespace(stripped))
            mapping.append({
                'type': 'block',
                'outer_path': [(elem.tag, dict(elem.attrib))],
                'inner_tags': True,
                'inner_attr_map': attr_map
            })

    def _is_single_chain(self, elem) -> bool:
        """Check if element has single-chain nesting (each level has one meaningful child)."""
        children = list(elem)

        # Filter out empty text-only situations
        has_significant_text = elem.text and elem.text.strip()

        if len(children) == 0:
            return True
        if len(children) == 1 and not has_significant_text:
            child = children[0]
            # Check if child has significant tail text
            if child.tail and child.tail.strip():
                return False
            return self._is_single_chain(child)
        return False

    def _is_inline(self, elem) -> bool:
        """Check if element has inline-level display based on CSS cascade."""
        if not isinstance(elem, etree._Element) or not isinstance(elem.tag, str):
            return False
        display = self._display_map.get(elem, 'inline')
        return display.startswith('inline') or display in ('contents', 'none')

    def _is_block(self, elem) -> bool:
        """Check if element has block-level display based on CSS cascade."""
        if not isinstance(elem, etree._Element) or not isinstance(elem.tag, str):
            return False
        return not self._is_inline(elem)

    def _has_block_descendants(self, elem) -> bool:
        """Check if element contains any block-level descendant."""
        for child in elem.iterdescendants():
            if isinstance(child, etree._Element) and self._is_block(child):
                return True
        return False

    def _is_block_container(self, elem) -> bool:
        """Treat inline wrappers around block descendants as block containers."""
        return self._is_block(elem) or self._has_block_descendants(elem)

    def _has_block_children(self, elem) -> bool:
        """Check if element contains any block-level children."""
        for child in elem:
            if isinstance(child, etree._Element) and self._is_block_container(child):
                return True
        return False

    def _group_children(self, elem):
        """Group element's children into block elements and inline runs.

        Returns:
            List of tuples:
            - ('block', element) for block-level children
            - ('inline_run', [(type, data), ...]) for consecutive inline content
            - ('void', element) for void elements
            - ('comment', element) for comment nodes
        """
        groups = []
        current_run = []

        # elem.text (text before first child)
        if elem.text:
            current_run.append(('text', elem.text))

        for child in elem:
            if isinstance(child, etree._Comment):
                if current_run:
                    groups.append(('inline_run', current_run))
                    current_run = []
                groups.append(('comment', child))
                if child.tail:
                    current_run.append(('text', child.tail))
            elif isinstance(child, etree._Element) and child.tag in self.VOID_ELEMENTS:
                if current_run:
                    groups.append(('inline_run', current_run))
                    current_run = []
                groups.append(('void', child))
                if child.tail:
                    current_run.append(('text', child.tail))
            elif isinstance(child, etree._Element) and self._is_block_container(child):
                if current_run:
                    groups.append(('inline_run', current_run))
                    current_run = []
                groups.append(('block', child))
                if child.tail:
                    current_run.append(('text', child.tail))
            elif isinstance(child, etree._Element):
                # Inline element
                current_run.append(('element', child))
                if child.tail:
                    current_run.append(('text', child.tail))
            # Skip non-element, non-comment nodes

        if current_run:
            groups.append(('inline_run', current_run))

        return groups

    def _process_mixed_children(self, elem, units, mapping):
        """Process a container's children, grouping consecutive inline content.

        Used for both body-level and container block processing.
        """
        groups = self._group_children(elem)
        for group_type, group_data in groups:
            if group_type == 'block':
                self._process_element(group_data, units, mapping)
            elif group_type == 'inline_run':
                self._process_inline_run(group_data, units, mapping)
            elif group_type == 'void':
                units.append('')
                mapping.append({
                    'type': 'void',
                    'tag': group_data.tag,
                    'attrs': dict(group_data.attrib)
                })
            elif group_type == 'comment':
                units.append('')
                mapping.append({
                    'type': 'comment',
                    'content': str(group_data.text) if group_data.text else ''
                })

    def _process_inline_run(self, run_items, units, mapping):
        """Process a group of consecutive inline content as a single unit.

        Serializes text nodes and inline elements into one HTML string,
        strips inner attributes for translation, and records mapping.
        """
        parts = []
        for item_type, item_data in run_items:
            if item_type == 'text':
                parts.append(escape(item_data, quote=False))
            elif item_type == 'element':
                parts.append(etree.tostring(
                    item_data, encoding='unicode', method='xml', with_tail=False
                ))

        inner_html = ''.join(parts)
        normalized = self._normalize_whitespace(inner_html)

        if not normalized.strip():
            return  # Empty run, skip

        has_elements = any(t == 'element' for t, _ in run_items)

        if not has_elements:
            # Pure text run - use naked type (compatible with existing behavior)
            units.append(normalized)
            mapping.append({'type': 'naked'})
        else:
            # Contains inline elements - strip attrs and save mapping
            stripped, attr_map = self._strip_inner_attrs(inner_html)
            stripped = self._normalize_whitespace(stripped)
            if not stripped.strip():
                return
            units.append(stripped)
            mapping.append({
                'type': 'inline_run',
                'has_content': bool(stripped.strip()),
                'inner_tags': bool(attr_map),
                'inner_attr_map': attr_map
            })

    def _extract_chain(self, elem) -> Tuple[str, List[Tuple[str, Dict]]]:
        """Extract text and full tag path from single-chain element."""
        path: List[Tuple[str, Dict]] = [(elem.tag, dict(elem.attrib))]
        current = elem

        while True:
            children = list(current)

            if len(children) == 0:
                # Return text content
                text = current.text or ''
                return escape(text, quote=False), path

            child = children[0]
            path.append((child.tag, dict(child.attrib)))
            current = child

    def _get_inner_html(self, elem) -> str:
        """Get innerHTML of an element using lxml."""
        parts = []
        if elem.text:
            parts.append(escape(elem.text, quote=False))
        for child in elem:
            # Use method='xml' for XHTML-compliant output
            parts.append(etree.tostring(child, encoding='unicode', method='xml'))
        return ''.join(parts)

    def _parse_fragment(self, s: str):
        """Parse HTML fragment, wrapping in a div container."""
        return lxml_html.fragment_fromstring(s, create_parent="div")

    def _normalize_html(self, html: str) -> str:
        """
        使用 lxml 规范化 HTML（容错解析）。

        对规范文本无害，可能修复：
        - 未闭合的标签
        - 错误嵌套
        - 实体编码问题
        """
        if not html or not html.strip():
            return html

        try:
            wrapped = f'<div>{html}</div>'
            parser = lxml_html.HTMLParser(recover=True, encoding='utf-8')
            doc = lxml_html.document_fromstring(wrapped.encode('utf-8'), parser=parser)

            body = doc.find('.//body')
            if body is not None and len(body) > 0:
                wrapper = body[0]
                parts = []
                if wrapper.text:
                    parts.append(escape(wrapper.text, quote=False))
                for child in wrapper:
                    parts.append(etree.tostring(child, encoding='unicode', method='xml'))
                return ''.join(parts)
            return html
        except Exception as e:
            logger.debug(f"HTML normalization failed: {e}")
            return html

    def _extract_text_only(self, html: str) -> str:
        """
        从 HTML 提取纯文本（降级策略）。

        保留所有文本内容，只移除标签。
        """
        if not html:
            return html
        try:
            root = self._parse_fragment(html)
            return root.text_content()
        except Exception:
            # 最后的回退：正则移除标签
            return re.sub(r'<[^>]+>', '', html).strip()

    def _serialize_fragment(self, wrapper) -> str:
        """Serialize fragment contents back to HTML string (XML mode for XHTML)."""
        parts = []
        if wrapper.text:
            # Escape HTML special chars in text (lxml decodes entities like &lt; to <)
            parts.append(escape(wrapper.text, quote=False))
        for child in wrapper:
            # Use method='xml' for XHTML-compliant output (preserves <br/> etc)
            parts.append(etree.tostring(child, encoding="unicode", method="xml"))
        return "".join(parts)

    def _strip_inner_attrs(self, html: str) -> Tuple[str, List[Dict]]:
        """
        Strip attributes from all tags in HTML using lxml DOM operations.

        If a compactor is configured, it will first compact the HTML by
        merging nested spans to reduce element count.

        Returns:
            Tuple of (stripped_html, attr_mapping)
            attr_mapping is a list of {index, tag, attrs} for restoration
        """
        if not html or not html.strip():
            return html, []

        # Pre-process with compactor if available
        if self.compactor:
            html = self.compactor.compact(html)

        root = self._parse_fragment(html)
        mapping: List[Dict] = []
        i = 0

        for el in root.iter():
            if el is root:
                continue
            # Skip Comment/PI nodes — their .tag is a cython callable, not a string,
            # which causes json.dump to fail mid-write and truncate mapping files
            if not isinstance(el.tag, str):
                continue
            mapping.append({
                "index": i,
                "tag": el.tag,
                "attrs": dict(el.attrib)
            })
            el.attrib.clear()
            i += 1

        return self._serialize_fragment(root), mapping

    def _restore_inner_attrs(self, html: str, attr_map: List[Dict]) -> str:
        """
        Restore attributes to tags using lxml DOM operations.

        Supports two mapping formats:
        - New format: {'index': N, 'tag': 'span', 'attrs': {'class': 'x'}}
        - Legacy format: {'index': N, 'tag': 'span', 'original': '<span class="x">'}

        包含容错降级策略（局部降级）：
        1. 先尝试 HTML 规范化修复
        2. 如果完全匹配，直接还原属性
        3. 如果不匹配，尝试局部降级：保留能匹配的外层 tag，内部降级为纯文本
        """
        if not attr_map:
            return html

        # Check for legacy format (backward compatibility)
        if attr_map and 'original' in attr_map[0]:
            return self._restore_inner_attrs_legacy(html, attr_map)

        if not html or not html.strip():
            return html

        # 步骤 1：尝试 HTML 规范化修复
        normalized = self._normalize_html(html)

        # 步骤 2：解析并计数（skip Comment/PI nodes to match _strip_inner_attrs）
        root = self._parse_fragment(normalized)
        elements = [el for el in root.iter() if el is not root and isinstance(el.tag, str)]

        # 步骤 3：完全匹配 → 直接还原
        if len(elements) == len(attr_map):
            all_match = True
            for el, m in zip(elements, attr_map):
                if el.tag != m["tag"]:
                    all_match = False
                    break
            if all_match:
                for el, m in zip(elements, attr_map):
                    el.attrib.clear()
                    el.attrib.update(m["attrs"])
                return self._serialize_fragment(root)

        # 步骤 4：不匹配 → 尝试局部降级
        return self._partial_restore_attrs(normalized, attr_map, len(elements))

    def _partial_restore_attrs(self, html: str, attr_map: List[Dict], actual_count: int) -> str:
        """
        局部降级：尽量保留能匹配的外层 tag，内部降级为纯文本。

        策略：按原始 attr_map 的 tag 序列重建，内容从翻译结果提取纯文本。
        """
        expected_count = len(attr_map)
        logger.warning(
            f"Tag count mismatch: {actual_count} vs {expected_count}. "
            f"Attempting partial restoration."
        )

        # 提取纯文本内容
        text_content = self._extract_text_only(html)
        if not text_content:
            return html

        # 分析 attr_map 的结构：找出顶层 tag 和嵌套 tag
        # 通过观察：attr_map 是深度优先遍历的结果
        # 例如 <a><span>x</span><span>y</span></a> → [a, span, span]
        # 我们需要找到顶层 tag（没有父级的 tag）

        if not attr_map:
            return text_content

        # 简化策略：只保留第一个 tag（通常是 <a> 等容器），内部用纯文本
        first_tag = attr_map[0]
        tag_name = first_tag["tag"]
        attrs = first_tag["attrs"]

        # 构建属性字符串
        attr_str = ""
        for k, v in attrs.items():
            # 转义属性值中的特殊字符
            escaped_v = v.replace("&", "&amp;").replace('"', "&quot;")
            attr_str += f' {k}="{escaped_v}"'

        # 包装纯文本
        result = f"<{tag_name}{attr_str}>{text_content}</{tag_name}>"
        logger.debug(f"Partial restore: kept <{tag_name}>, inner content as plain text")
        return result

    def _restore_inner_attrs_legacy(self, html: str, attr_map: List[Dict]) -> str:
        """Restore original tags with attributes (legacy regex-based implementation)."""
        if not attr_map:
            return html

        # Build restoration mapping: index -> original tag
        restore_map = {item['index']: item['original'] for item in attr_map}

        tag_index = 0
        result_parts = []
        last_end = 0

        pattern = r'<(/)?(\w+)(?:[^>]*?)(/)?>'
        for match in re.finditer(pattern, html):
            is_closing = match.group(1) is not None

            if is_closing:
                # Closing tags don't count in index
                result_parts.append(html[last_end:match.end()])
                last_end = match.end()
                continue

            # Opening or self-closing tag
            result_parts.append(html[last_end:match.start()])

            if tag_index in restore_map:
                result_parts.append(restore_map[tag_index])
            else:
                result_parts.append(match.group(0))

            last_end = match.end()
            tag_index += 1

        result_parts.append(html[last_end:])
        return ''.join(result_parts)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: newlines to spaces, collapse multiple spaces.

        Preserves non-breaking spaces (\xa0), ideographic space (\u3000),
        and other special Unicode whitespace.
        """
        # Only normalize ASCII whitespace (space, tab, newline, etc.)
        # Preserve NBSP (\xa0), ideographic space (\u3000), and other Unicode spaces
        normalized = re.sub(r'[ \t\n\r\f\v]+', ' ', text)
        # Only strip ASCII spaces from ends, not Unicode spaces
        return normalized.strip(' \t\n\r\f\v')

    def _format_attrs(self, attrs: Dict) -> str:
        """Format attributes dict as HTML string."""
        if not attrs:
            return ''

        parts = []
        for key, value in attrs.items():
            if isinstance(value, list):
                value = ' '.join(value)
            if value is True:
                parts.append(f' {key}')
            elif value is not None and value is not False:
                # Escape quotes in value
                escaped = str(value).replace('"', '&quot;')
                parts.append(f' {key}="{escaped}"')

        return ''.join(parts)
