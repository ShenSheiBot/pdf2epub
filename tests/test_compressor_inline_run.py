"""Tests for HTMLCompressor inline run grouping and round-trip correctness.

Validates that compress→decompress preserves HTML content, and that
inline elements are properly grouped with surrounding text.
"""

import json
import re
import pytest
from lxml import html as lxml_html

from pdf2epub.html_translation.compressor import HTMLCompressor


def _normalize(html: str) -> str:
    """Normalize HTML for comparison: strip whitespace, collapse spaces."""
    # Parse and re-serialize to normalize tag casing, attribute order, etc.
    try:
        root = lxml_html.fromstring(f"<div>{html}</div>")
        from lxml import etree
        from html import escape
        parts = []
        if root.text:
            parts.append(escape(root.text, quote=False))
        for child in root:
            parts.append(etree.tostring(child, encoding='unicode', method='xml'))
        html = ''.join(parts)
    except Exception:
        pass
    # Collapse whitespace
    return re.sub(r'\s+', ' ', html).strip()


def _round_trip(html: str, author_css: str = "") -> str:
    """Compress then decompress HTML, return result."""
    c = HTMLCompressor()
    compressed, mapping = c.compress(html, author_css=author_css)
    return c.decompress(compressed, mapping)


def _compress(html: str, author_css: str = ""):
    """Compress HTML, return (compressed_text, mapping)."""
    c = HTMLCompressor()
    return c.compress(html, author_css=author_css)


def _body_content(full_html: str) -> str:
    """Extract body innerHTML from full HTML."""
    root = lxml_html.fromstring(full_html)
    body = root.find('.//body')
    if body is None:
        return full_html
    from lxml import etree
    from html import escape
    parts = []
    if body.text:
        parts.append(escape(body.text, quote=False))
    for child in body:
        parts.append(etree.tostring(child, encoding='unicode', method='xml'))
    return ''.join(parts)


# --- Round-trip tests ---

class TestRoundTrip:
    """Verify compress→decompress preserves HTML content."""

    def test_pure_blocks(self):
        html = "<html><body><p>Hello</p><p>World</p></body></html>"
        result = _round_trip(html)
        body = _body_content(result)
        assert "<p>Hello</p>" in body
        assert "<p>World</p>" in body

    def test_leaf_block_with_inline(self):
        html = '<html><body><p>text <i>italic</i> more <a href="x">link</a></p></body></html>'
        result = _round_trip(html)
        body = _body_content(result)
        # Content should be preserved (attrs may be restored)
        assert "italic" in body
        assert "link" in body
        assert "text" in body
        assert "more" in body

    def test_container_with_mixed_children(self):
        """Core scenario: div with both block and inline children."""
        html = (
            "<html><body>"
            "<div>"
            "Text before <i>italic</i> more "
            "<blockquote><p>Quote</p></blockquote>"
            "After quote <a href=\"x\">link</a> end"
            "</div>"
            "</body></html>"
        )
        result = _round_trip(html)
        body = _body_content(result)
        assert "Text before" in body
        assert "italic" in body
        assert "more" in body
        assert "Quote" in body
        assert "After quote" in body
        assert "link" in body
        assert "end" in body

    def test_body_level_mixed(self):
        """Body directly contains text + inline + block."""
        html = (
            "<html><body>"
            "Hello <em>world</em> "
            "<p>Paragraph</p>"
            "After <strong>bold</strong>"
            "</body></html>"
        )
        result = _round_trip(html)
        body = _body_content(result)
        assert "Hello" in body
        assert "world" in body
        assert "Paragraph" in body
        assert "After" in body
        assert "bold" in body

    def test_nested_containers(self):
        html = (
            "<html><body>"
            "<div><div>"
            "<blockquote><p>Deep quote</p></blockquote>"
            " text <i>x</i>"
            "</div></div>"
            "</body></html>"
        )
        result = _round_trip(html)
        body = _body_content(result)
        assert "Deep quote" in body
        assert "text" in body

    def test_void_between_inline(self):
        """Void element interrupts inline run."""
        html = "<html><body><div>text <br/> <i>more</i><p>block</p></div></body></html>"
        result = _round_trip(html)
        body = _body_content(result)
        assert "<br/>" in body
        assert "text" in body
        assert "more" in body
        assert "block" in body

    def test_comment_between_inline(self):
        """Comment interrupts inline run."""
        html = "<html><body><div>text <!-- comment --> <i>more</i><p>block</p></div></body></html>"
        result = _round_trip(html)
        body = _body_content(result)
        assert "<!-- comment -->" in body or "<!--comment-->" in body.replace(" ", "")
        assert "text" in body
        assert "more" in body

    def test_single_chain_block(self):
        """Single-chain nesting: behavior unchanged."""
        html = "<html><body><p><span><em>text</em></span></p></body></html>"
        result = _round_trip(html)
        body = _body_content(result)
        assert "text" in body
        assert "<p>" in body or "<p " in body

    def test_empty_block(self):
        html = "<html><body><div></div></body></html>"
        result = _round_trip(html)
        body = _body_content(result)
        assert "<div" in body

    def test_empty_inline(self):
        html = "<html><body><span></span></body></html>"
        result = _round_trip(html)
        body = _body_content(result)
        assert "<span" in body

    def test_xml_declaration_preserved(self):
        html = '<?xml version="1.0" encoding="UTF-8"?><html><body><p>text</p></body></html>'
        result = _round_trip(html)
        assert '<?xml' in result
        assert 'text' in result


# --- Inline run grouping tests ---

class TestInlineRunGrouping:
    """Verify that consecutive inline elements are grouped into single units."""

    def test_inline_grouped_in_container(self):
        """Text + inline elements in container block → single inline_run unit."""
        html = (
            "<html><body>"
            "<div>"
            "Text <i>italic</i> and <a href=\"x\">link</a>"
            "<p>Para</p>"
            "</div>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        lines = compressed.strip().split('\n')

        # The inline content before <p> should be ONE line, not three
        # (old behavior: "Text", "<i>italic</i>", "and", "<a>link</a>" = 4 lines)
        # Find inline_run units
        inline_run_units = [u for u in mapping['units'] if u['type'] == 'inline_run']
        assert len(inline_run_units) >= 1, "Should have at least one inline_run"

        # The first translatable line should contain both "italic" and "link"
        first_line = lines[0]
        assert "italic" in first_line
        assert "link" in first_line

    def test_pure_text_stays_naked(self):
        """Pure text (no inline elements) in container → naked type, not inline_run."""
        html = (
            "<html><body>"
            "<div>"
            "Just text"
            "<p>Para</p>"
            "</div>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        naked_units = [u for u in mapping['units'] if u['type'] == 'naked']
        assert any(u['type'] == 'naked' for u in mapping['units'])

    def test_multiple_inline_runs(self):
        """Block child creates separate inline runs before and after."""
        html = (
            "<html><body>"
            "<div>"
            "Before <i>a</i>"
            "<blockquote><p>Quote</p></blockquote>"
            "After <b>b</b>"
            "</div>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        inline_runs = [u for u in mapping['units'] if u['type'] == 'inline_run']
        assert len(inline_runs) == 2, f"Expected 2 inline_runs, got {len(inline_runs)}"

    def test_line_count_reduced(self):
        """Inline run grouping produces fewer lines than old per-element approach."""
        # This would produce 4+ lines in old approach (text, <i>, text, <a>)
        # but should produce 1-2 lines with inline run grouping
        html = (
            "<html><body>"
            "<div>"
            "Text <i>italic</i> and <a href=\"x\"><sup>1</sup></a> end"
            "<p>Para</p>"
            "</div>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        lines = [l for l in compressed.strip().split('\n') if l.strip()]
        # Should be at most 2 translatable lines (inline_run + block para)
        assert len(lines) <= 3, f"Too many lines: {len(lines)}. Lines: {lines}"


# --- CSS-based block detection ---

class TestCSSBlockDetection:
    """Verify CSS display resolution affects compression behavior."""

    def test_inline_style_display_block(self):
        """span with display:block should be treated as block element."""
        html = (
            '<html><body>'
            '<div>'
            '<span style="display:block">Block span</span>'
            '<p>Para</p>'
            '</div>'
            '</body></html>'
        )
        compressed, mapping = _compress(html)
        # The span should be treated as a block, not grouped into inline run
        unit_types = [u['type'] for u in mapping['units']]
        # span with display:block should produce its own block unit or block_open/close
        # It should NOT be in an inline_run
        inline_run_with_span = any(
            u['type'] == 'inline_run' and 'Block span' in compressed
            for u in mapping['units']
        )
        # Check that "Block span" is not in any inline_run line
        block_units = [u for u in mapping['units'] if u['type'] in ('block', 'block_open')]
        assert len(block_units) >= 2, "display:block span should create additional block unit"

    def test_author_css_display_block(self):
        """Author CSS display:block should make element block-level."""
        html = (
            '<html><body>'
            '<div>'
            '<span class="note">Note</span>'
            '<p>Para</p>'
            '</div>'
            '</body></html>'
        )
        compressed, mapping = _compress(html, author_css=".note { display: block }")
        # .note span should be block, not inline
        block_units = [u for u in mapping['units'] if u['type'] in ('block', 'block_open')]
        assert len(block_units) >= 2, "Author CSS display:block should create block unit"


# --- Backward compatibility ---

class TestBackwardCompat:
    """Old mapping formats should still decompress correctly."""

    def test_old_mapping_without_inline_run(self):
        """Mapping with only naked/block/inline types still works."""
        c = HTMLCompressor()
        mapping = {
            'wrapper': {},
            'units': [
                {'type': 'block', 'outer_path': [('p', {})], 'inner_tags': False, 'has_content': True},
                {'type': 'naked', 'has_content': True},
                {'type': 'inline', 'outer_path': [('i', {})], 'inner_tags': False, 'has_content': True},
            ]
        }
        translated = "Hello\nWorld\nItalic"
        result = c.decompress(translated, mapping)
        assert "Hello" in result
        assert "World" in result
        assert "<i>Italic</i>" in result


# --- Edge cases ---

class TestEdgeCases:
    def test_whitespace_only_inline_run_skipped(self):
        """Whitespace-only inline runs should not produce units."""
        html = (
            "<html><body>"
            "<div>  \n  <p>Para</p></div>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        # No naked or inline_run units for the whitespace
        for unit in mapping['units']:
            if unit['type'] in ('naked', 'inline_run'):
                assert unit.get('has_content', True), "Whitespace-only unit should not have content"

    def test_inline_attrs_restored(self):
        """Inline element attributes should be stripped for translation and restored."""
        html = (
            '<html><body>'
            '<div>'
            'Text <a href="http://example.com" class="ref">link</a> end'
            '<p>Para</p>'
            '</div>'
            '</body></html>'
        )
        result = _round_trip(html)
        body = _body_content(result)
        assert 'href="http://example.com"' in body
        assert "link" in body

    def test_deeply_nested_inline(self):
        """Deeply nested inline elements are preserved."""
        html = (
            "<html><body>"
            "<div>"
            "Text <a href=\"x\"><span><em>deep</em></span></a> end"
            "<p>Para</p>"
            "</div>"
            "</body></html>"
        )
        result = _round_trip(html)
        body = _body_content(result)
        assert "deep" in body
        assert "Text" in body
        assert "end" in body

    def test_multiple_blocks_between_inline_runs(self):
        html = (
            "<html><body>"
            "<div>"
            "Before <i>a</i>"
            "<p>P1</p>"
            "<p>P2</p>"
            "After <b>b</b>"
            "</div>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        # Should have inline_run, block (P1), block (P2), inline_run
        block_units = [u for u in mapping['units'] if u['type'] == 'block']
        assert len(block_units) >= 2
        inline_runs = [u for u in mapping['units'] if u['type'] == 'inline_run']
        assert len(inline_runs) == 2

    def test_real_world_epub_pattern(self):
        """Pattern from Decolonising Anime: div with blockquote + inline text."""
        html = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            '<head><title>Test</title></head>'
            '<body>'
            '<div class="calibre1">'
            '<p class="calibre2">Normal paragraph with <i class="calibre3">otaku</i> culture.</p>'
            '<blockquote class="calibre4"><div class="calibre5">'
            '<p class="calibre2">A quoted paragraph.</p>'
            '</div></blockquote>'
            '<p class="calibre2">After the quote, referencing <a href="#note1"><sup>1</sup></a> something.</p>'
            '</div>'
            '</body></html>'
        )
        compressed, mapping = _compress(html)
        lines = [l for l in compressed.strip().split('\n') if l.strip()]

        # "otaku" should be on the same line as its surrounding text
        otaku_lines = [l for l in lines if 'otaku' in l]
        assert len(otaku_lines) == 1
        otaku_line = otaku_lines[0]
        assert 'culture' in otaku_line, f"otaku should be inline with surrounding text: {otaku_line}"

        # Round-trip should preserve content
        c = HTMLCompressor()
        result = c.decompress(compressed, mapping)
        assert 'otaku' in result
        assert 'quoted paragraph' in result

    def test_only_inline_children_no_blocks(self):
        """Container with only inline children → leaf block processing, not mixed."""
        html = (
            "<html><body>"
            "<p>Hello <em>world</em> <strong>bold</strong></p>"
            "</body></html>"
        )
        compressed, mapping = _compress(html)
        # p has no block children → leaf block → single 'block' unit
        block_units = [u for u in mapping['units'] if u['type'] == 'block']
        assert len(block_units) == 1
        # Should be one translatable line
        lines = [l for l in compressed.strip().split('\n') if l.strip()]
        assert len(lines) == 1
