"""Tests for CSS display property resolver."""

import pytest
from lxml import html as lxml_html

from pdf2epub.html_translation.display_resolver import resolve_display


def _make_tree(body_html, head_css=""):
    """Helper: build lxml tree from body HTML and optional <style> CSS."""
    if head_css:
        full = f"<html><head><style>{head_css}</style></head><body>{body_html}</body></html>"
    else:
        full = f"<html><body>{body_html}</body></html>"
    return lxml_html.fromstring(full)


def _get(root, tag_or_xpath):
    """Helper: find element by tag name or XPath."""
    el = root.find(f".//{tag_or_xpath}")
    assert el is not None, f"Element {tag_or_xpath!r} not found"
    return el


# --- UA defaults ---

class TestUADefaults:
    def test_div_is_block(self):
        root = _make_tree("<div>hello</div>")
        dm = resolve_display(root)
        assert dm[_get(root, "div")] == "block"

    def test_p_is_block(self):
        root = _make_tree("<p>hello</p>")
        dm = resolve_display(root)
        assert dm[_get(root, "p")] == "block"

    def test_span_is_inline(self):
        root = _make_tree("<span>hello</span>")
        dm = resolve_display(root)
        assert dm[_get(root, "span")] == "inline"

    def test_em_is_inline(self):
        root = _make_tree("<em>hello</em>")
        dm = resolve_display(root)
        assert dm[_get(root, "em")] == "inline"

    def test_table_is_table(self):
        root = _make_tree("<table><tr><td>x</td></tr></table>")
        dm = resolve_display(root)
        assert dm[_get(root, "table")] == "table"
        assert dm[_get(root, "tr")] == "table-row"
        assert dm[_get(root, "td")] == "table-cell"

    def test_li_is_list_item(self):
        root = _make_tree("<ul><li>x</li></ul>")
        dm = resolve_display(root)
        assert dm[_get(root, "li")] == "list-item"

    def test_blockquote_is_block(self):
        root = _make_tree("<blockquote>x</blockquote>")
        dm = resolve_display(root)
        assert dm[_get(root, "blockquote")] == "block"

    def test_headings_are_block(self):
        for i in range(1, 7):
            tag = f"h{i}"
            root = _make_tree(f"<{tag}>heading</{tag}>")
            dm = resolve_display(root)
            assert dm[_get(root, tag)] == "block", f"{tag} should be block"


# --- Unknown elements ---

class TestUnknownElements:
    def test_custom_tag_defaults_to_inline(self):
        root = _make_tree("<custom-tag>hello</custom-tag>")
        dm = resolve_display(root)
        el = root.find(".//custom-tag")
        # lxml may lowercase or not find custom tags; skip if not found
        if el is not None:
            assert dm[el] == "inline"

    def test_a_is_inline(self):
        root = _make_tree('<a href="#">link</a>')
        dm = resolve_display(root)
        assert dm[_get(root, "a")] == "inline"

    def test_sup_is_inline(self):
        root = _make_tree("<sup>1</sup>")
        dm = resolve_display(root)
        assert dm[_get(root, "sup")] == "inline"

    def test_i_is_inline(self):
        root = _make_tree("<i>italic</i>")
        dm = resolve_display(root)
        assert dm[_get(root, "i")] == "inline"


# --- Author CSS override ---

class TestAuthorCSS:
    def test_class_override_to_flex(self):
        root = _make_tree('<div class="flex">hello</div>')
        dm = resolve_display(root, author_css=".flex { display: flex }")
        assert dm[_get(root, "div")] == "flex"

    def test_class_override_to_inline_block(self):
        root = _make_tree('<div class="ib">hello</div>')
        dm = resolve_display(root, author_css=".ib { display: inline-block }")
        assert dm[_get(root, "div")] == "inline-block"

    def test_author_css_overrides_ua(self):
        """Author CSS has higher specificity than UA defaults."""
        root = _make_tree("<span>hello</span>")
        dm = resolve_display(root, author_css="span { display: block }")
        assert dm[_get(root, "span")] == "block"


# --- Inline style override ---

class TestInlineStyle:
    def test_inline_style_block(self):
        root = _make_tree('<span style="display:block">hello</span>')
        dm = resolve_display(root)
        assert dm[_get(root, "span")] == "block"

    def test_inline_style_overrides_author(self):
        root = _make_tree('<span style="display:inline">hello</span>')
        dm = resolve_display(root, author_css="span { display: block }")
        assert dm[_get(root, "span")] == "inline"


# --- !important ---

class TestImportant:
    def test_author_important_overrides_inline(self):
        root = _make_tree('<span style="display:inline">hello</span>')
        dm = resolve_display(root, author_css="span { display: block !important }")
        assert dm[_get(root, "span")] == "block"

    def test_inline_important_wins(self):
        root = _make_tree('<span style="display:block !important">hello</span>')
        dm = resolve_display(root, author_css="span { display: inline }")
        assert dm[_get(root, "span")] == "block"


# --- hidden attribute ---

class TestHidden:
    def test_hidden_div_is_none(self):
        root = _make_tree('<div hidden="">hello</div>')
        dm = resolve_display(root)
        assert dm[_get(root, "div")] == "none"
