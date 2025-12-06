import re
import pytest
from lxml import etree

from pdf2epub.html_translation.verified_compactor import VerifiedCompactor
from pdf2epub.html_translation.oracle import (
    StylesheetOracle,
    XHTML_NS,
    EPUB_NS,
    DEFAULT_NAMESPACES,
)

# -----------------------
# helpers
# -----------------------

def _wrap_fragment(html: str) -> etree._Element:
    wrapped = f'<div xmlns="{XHTML_NS}" xmlns:epub="{EPUB_NS}">{html}</div>'
    return etree.fromstring(wrapped.encode("utf-8"))

def _element_count(root: etree._Element) -> int:
    return sum(1 for el in root.iter() if el is not root and isinstance(el.tag, str))

def _first_child_element(root: etree._Element) -> etree._Element:
    for ch in root:
        if isinstance(ch.tag, str):
            return ch
    raise AssertionError("no element child")


# -----------------------
# Tests
# -----------------------

def test_T1_whitespace_parent_text_and_child_tail_allows_merge():
    """Whitespace-only parent.text and child.tail should allow merge."""
    css = """.a { color: red; } .b { color: blue; }"""
    html = '<div class="a">\n  <div class="b">X</div>\n</div>'

    comp = VerifiedCompactor(css_content=css, conservative_mode=False)
    out, stats = comp.compact_with_stats(html)
    assert stats["phase2"] == 1, "whitespace-only text should allow merge"

    root = _wrap_fragment(out)
    assert _element_count(root) == 1
    el = _first_child_element(root)
    assert el.tag.endswith("div")
    classes = set((el.get("class") or "").split())
    assert {"a", "b"} <= classes


def test_T2_whitespace_child_tail_only_allows_merge():
    """Whitespace-only child.tail should allow merge."""
    css = """.a { } .b { }"""
    html = '<div class="a"><div class="b">X</div>\n</div>'  # parent.text=None, child.tail="\n"

    comp = VerifiedCompactor(css_content=css, conservative_mode=False)
    out, stats = comp.compact_with_stats(html)
    assert stats["phase2"] == 1, "whitespace-only child.tail should allow merge"

    root = _wrap_fragment(out)
    assert _element_count(root) == 1
    el = _first_child_element(root)
    classes = set((el.get("class") or "").split())
    assert {"a", "b"} <= classes


def test_T3_namespace_url_svg_compiles_correctly():
    """@namespace with url(...) syntax should compile correctly."""
    css = """
    @namespace svg url("http://www.w3.org/2000/svg");
    svg|text { fill: red; }
    .a { } .b { }
    """
    html = '<div class="a"><div class="b">X</div></div>'

    comp = VerifiedCompactor(css_content=css, conservative_mode=True)
    assert len(comp.oracle.failed_selectors) == 0, "url(...) namespace should parse correctly"
    out, stats = comp.compact_with_stats(html)
    assert stats["phase2"] == 1


def test_T4_namespace_url_mathml_compiles_correctly():
    """@namespace with url(...) syntax for MathML should compile correctly."""
    css = """
    @namespace m url("http://www.w3.org/1998/Math/MathML");
    m|math { display: block; }
    .a { } .b { }
    """
    html = '<div class="a"><div class="b">X</div></div>'

    comp = VerifiedCompactor(css_content=css, conservative_mode=True)
    assert len(comp.oracle.failed_selectors) == 0, "url(...) namespace should parse correctly"
    out, stats = comp.compact_with_stats(html)
    assert stats["phase2"] == 1
