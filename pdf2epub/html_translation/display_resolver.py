"""
CSS Display Property Resolver.

Resolves the computed `display` value for every element in an HTML tree
using the CSS cascade (UA defaults + author CSS + inline styles).

Zero new dependencies — uses tinycss2 (CSS parsing) and cssselect2
(selector matching with specificity sorting), both already in the project.

UA stylesheet display rules sourced from:
https://html.spec.whatwg.org/multipage/rendering.html
"""

import cssselect2
import tinycss2
from lxml import etree


# HTML5 User-Agent stylesheet (display rules only).
# Source: https://html.spec.whatwg.org/multipage/rendering.html
# Note: :heading pseudo-class expanded to h1-h6 (not real CSS).
_UA_CSS = """
[hidden], area, base, basefont, datalist, head, link, meta, noembed,
noframes, param, rp, script, source, style, template, title, track
  { display: none }
address, article, aside, blockquote, body, center, dd, details, dir,
div, dl, dt, fieldset, figure, figcaption, footer, form,
h1, h2, h3, h4, h5, h6, header, hgroup, hr, html, legend, listing,
main, menu, nav, ol, p, plaintext, pre, search, section, summary, ul, xmp
  { display: block }
li { display: list-item }
table { display: table }
caption { display: table-caption }
colgroup { display: table-column-group }
col { display: table-column }
thead { display: table-header-group }
tbody { display: table-row-group }
tfoot { display: table-footer-group }
tr { display: table-row }
td, th { display: table-cell }
"""


def _add_rules(matcher, css_text):
    """Parse CSS and add rules to the matcher."""
    rules = tinycss2.parse_stylesheet(css_text, skip_comments=True, skip_whitespace=True)
    for rule in rules:
        if rule.type != "qualified-rule":
            continue
        try:
            selectors = cssselect2.compile_selector_list(rule.prelude)
        except cssselect2.SelectorError:
            continue
        declarations = tinycss2.parse_declaration_list(rule.content)
        for sel in selectors:
            matcher.add_selector(sel, declarations)


def resolve_display(html_root, author_css=""):
    """Resolve CSS display property for every element in an HTML tree.

    Uses cssselect2's specificity-sorted matching to implement the CSS cascade.
    Handles: UA defaults + author CSS + inline style + !important.

    Args:
        html_root: lxml etree Element (the root element, typically <html>).
        author_css: Optional author stylesheet text (e.g. from EPUB CSS files).

    Returns:
        dict mapping lxml Elements to their resolved display value string.
    """
    matcher = cssselect2.Matcher()
    _add_rules(matcher, _UA_CSS)
    if author_css:
        _add_rules(matcher, author_css)

    wrapper = cssselect2.ElementWrapper.from_html_root(html_root)
    result = {}

    for el in wrapper.iter_subtree():
        etree_el = el.etree_element
        if not isinstance(etree_el.tag, str):
            continue

        display = "inline"  # CSS default for unknown elements
        important_display = None

        # Matched CSS rules (UA + author), sorted by specificity (low → high)
        for _spec, _order, pseudo, declarations in matcher.match(el):
            if pseudo is not None:
                continue
            for decl in declarations:
                if hasattr(decl, "name") and decl.name == "display":
                    val = tinycss2.serialize(decl.value).strip()
                    if decl.important:
                        important_display = val
                    else:
                        display = val

        # Inline style= attribute (higher priority than selectors, except !important)
        inline_style = etree_el.get("style")
        if inline_style:
            for decl in tinycss2.parse_declaration_list(inline_style):
                if hasattr(decl, "name") and decl.name == "display":
                    val = tinycss2.serialize(decl.value).strip()
                    if decl.important:
                        important_display = val
                    else:
                        display = val

        result[etree_el] = important_display if important_display is not None else display

    return result
