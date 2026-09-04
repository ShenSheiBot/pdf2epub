import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from pdf2epub.epub.converter import ContentConverter
from pdf2epub.epub.math_renderer import (
    LatexSvgRenderer,
    MathRenderingError,
    _wrap_formula_for_tex,
)
from pdf2epub.markdown_to_html import convert_markdown_to_html


def test_duplicate_title_cleanup_preserves_repeated_symbolic_section_breaks(
    tmp_path,
) -> None:
    chapter = tmp_path / "chapter_1.md"
    original = "# Chapter\n\n## *\n\nFirst section.\n\n## *\n\nSecond section.\n"
    chapter.write_text(original, encoding="utf-8")

    converter = ContentConverter(SimpleNamespace(markdown_dir=tmp_path))

    assert converter.remove_duplicate_titles() == 0
    assert chapter.read_text(encoding="utf-8") == original


def test_markdown_output_normalizes_epub2_ids_and_list_starts() -> None:
    html = convert_markdown_to_html(
        "## 2. Heading\n\n2. Second item\n\n[Jump](#2-heading)\n",
        standalone=False,
    )

    assert 'id="epub-id-2-heading"' in html
    assert 'href="#epub-id-2-heading"' in html
    assert 'start="2"' not in html
    assert 'class="epub-continued-list"' in html
    assert 'style="counter-reset: epub-list 1;"' in html


def test_japanese_ruby_uses_epub2_compatible_spans() -> None:
    html = convert_markdown_to_html("玄関(げんかん)", standalone=False)

    assert '<span class="ruby">玄関<span class="rt">げんかん</span></span>' in html
    assert "<ruby>" not in html
    assert "<rt>" not in html


def test_url_query_values_are_not_rewritten_as_html_attributes() -> None:
    html = convert_markdown_to_html(
        "<https://youtu.be/example?t=186>",
        standalone=True,
    )

    ET.fromstring(html)
    assert 'href="https://youtu.be/example?t=186"' in html
    assert '?t="186"' not in html


def test_tables_are_wrapped_and_only_wide_tables_disable_cell_wrapping() -> None:
    ordinary = convert_markdown_to_html(
        "| Name | 1 | 2 |\n| --- | --- | --- |\n| Animator | x | x |",
        standalone=False,
    )
    wide = convert_markdown_to_html(
        "| A | B | C | D | E | F | G | H |\n"
        "| - | - | - | - | - | - | - | - |\n"
        "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |",
        standalone=False,
    )

    assert '<div class="table-scroll"><table>' in ordinary
    assert "table-scroll--wide" not in ordinary
    assert '<div class="table-scroll table-scroll--wide"><table>' in wide
    assert "</table></div>" in wide


def test_currency_prefixed_inline_math_preserves_delimiters_and_xml() -> None:
    html = convert_markdown_to_html(
        r"Cost was $\$1.1 \times 10^9$ in 1920.",
        standalone=True,
    )

    ET.fromstring(html)
    assert "PDF2EPUBESCAPEDDOLLARTOKEN" not in html
    assert "$1.1" in html
    assert "×" in html


def test_complex_math_uses_epub2_compatible_text_fallback() -> None:
    html = convert_markdown_to_html(
        r"Inline $\frac{x}{y}$ and display: $$\frac{a}{b}$$",
        standalone=True,
    )

    ET.fromstring(html)
    assert "http://www.w3.org/1998/Math/MathML" not in html
    assert r"$\frac{x}{y}$" in html
    assert 'class="math-display"' in html
    assert r"\frac{a}{b}" in html


def test_raw_math_wrappers_become_epub2_compatible_text() -> None:
    html = convert_markdown_to_html(
        r"A value is <math>F^* = \frac{dF}{dt}</math>.",
        standalone=True,
    )

    ET.fromstring(html)
    assert "<math" not in html
    assert 'class="math-inline"' in html
    assert r"F^&#42; = \frac{dF}{dt}" in html


def test_math_asterisks_are_not_parsed_as_markdown_emphasis() -> None:
    html = convert_markdown_to_html(
        r"$R^*(t) \propto 10^{at}$, so $F^*(t) \propto t$.",
        standalone=True,
    )

    ET.fromstring(html)
    assert "<em>" not in html
    assert html.count("&#42;") == 2
    assert "𝑅" in html
    assert "𝐹" in html


def test_math_renderer_receives_asterisk_tex_syntax_unchanged() -> None:
    rendered = []

    def render_svg(source: str, display: bool) -> str:
        rendered.append((source, display))
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1pt" height="1pt" '
            'viewBox="0 0 1 1"><path d="M0 0" /></svg>'
        )

    convert_markdown_to_html(
        r"$\operatorname*{argmax} Q^*$ and $$\begin{align*}a&=b\end{align*}$$",
        standalone=False,
        math_renderer=render_svg,
    )

    assert rendered == [
        (r"\begin{align*}a&=b\end{align*}", True),
        (r"\operatorname*{argmax} Q^*", False),
    ]


def test_standalone_display_environment_is_not_nested_in_display_math() -> None:
    standalone = r"\begin{align*}a&=b\\c&=d\end{align*}"
    nested = r"\begin{aligned}a&=b\\c&=d\end{aligned}"

    assert _wrap_formula_for_tex(standalone, True) == standalone
    assert _wrap_formula_for_tex(nested, True) == rf"\[\displaystyle {nested}\]"


def test_raw_math_display_attribute_reaches_renderer() -> None:
    rendered = []

    def render_svg(source: str, display: bool) -> str:
        rendered.append((source, display))
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1pt" height="1pt" '
            'viewBox="0 0 1 1"><path d="M0 0" /></svg>'
        )

    html = convert_markdown_to_html(
        '<math>x=1</math>\n\n<math display="block">y=2</math>',
        standalone=False,
        math_renderer=render_svg,
    )

    assert rendered == [("x=1", False), ("y=2", True)]
    assert '<span class="math-inline"><svg' in html
    assert '<div class="math-display"><svg' in html


def test_svg_placeholder_cannot_replace_literal_book_text() -> None:
    literal = "PDF2EPUBMATHSVGTOKEN0END"

    def render_svg(source: str, display: bool) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1pt" height="1pt" '
            'viewBox="0 0 1 1"><path d="M0 0" /></svg>'
        )

    html = convert_markdown_to_html(
        f"Keep {literal} unchanged beside $\\frac{{1}}{{2}}$.",
        standalone=False,
        math_renderer=render_svg,
    )

    assert literal in html
    assert html.count("<svg") == 1


def test_formula_syntax_inside_code_is_not_rendered() -> None:
    rendered = []

    def render_svg(source: str, display: bool) -> str:
        rendered.append((source, display))
        return "<svg />"

    html = convert_markdown_to_html(
        "Inline `$x$ <math>y</math>`.\n\n"
        "```tex\n$$z$$\n<math display=\"block\">w</math>\n```",
        standalone=False,
        math_renderer=render_svg,
    )

    assert rendered == []
    text = "".join(ET.fromstring(f"<root>{html}</root>").itertext())
    assert "$x$ <math>y</math>" in text
    assert "$$z$$" in text
    assert '<math display="block">w</math>' in text


def test_epub_math_renderer_receives_raw_complex_and_display_formulas() -> None:
    rendered = []

    def render_svg(source: str, display: bool) -> str:
        rendered.append((source, display))
        css_class = "math-svg-display" if display else "math-svg-inline"
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'class="math-svg {css_class}" fill="currentColor" '
            'width="1pt" height="1pt" viewBox="0 0 1 1">'
            '<path d="M0 0" /></svg>'
        )

    html = convert_markdown_to_html(
        r"Raw <math>F^* &gt; \frac{dF}{dt}</math>, inline $\frac{x}{y}$, "
        r"and display $$\begin{aligned}a&=b\\c&=d\end{aligned}$$.",
        standalone=True,
        math_renderer=render_svg,
    )

    ET.fromstring(html)
    assert html.count("<svg") == 3
    assert "math-latex" not in html
    assert r"\frac{x}{y}" not in "".join(ET.fromstring(html).itertext())
    assert rendered == [
        (r"F^* > \frac{dF}{dt}", False),
        (r"\begin{aligned}a&=b\\c&=d\end{aligned}", True),
        (r"\frac{x}{y}", False),
    ]


def test_svg_math_is_restored_after_markdown_table_parsing() -> None:
    def render_svg(source: str, display: bool) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" class="math-svg" '
            'fill="currentColor" width="1pt" height="1pt" viewBox="0 0 1 1">'
            '<path d="M0 0" /></svg>'
        )

    html = convert_markdown_to_html(
        "| Value |\n| --- |\n| <math>\\frac{1}{2}</math> |",
        standalone=True,
        math_renderer=render_svg,
    )

    ET.fromstring(html)
    assert html.count("<svg") == 1
    assert "&lt;svg" not in html
    assert "PDF2EPUBMATHSVGTOKEN" not in html


def test_latex_svg_renderer_gives_repeated_formulas_unique_ids(
    tmp_path, monkeypatch
) -> None:
    renderer = LatexSvgRenderer(tmp_path)
    svg = """<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="1pt" height="1pt" viewBox="0 0 1 1">
  <defs><path id="glyph" d="M0 0" /></defs>
  <g id="page"><use x="2" y="3" xlink:href="#glyph" /></g>
</svg>"""
    template = renderer._prepare_svg(svg, display=False)
    assert "\n" not in template
    assert "<use" not in template
    assert "<defs" not in template
    assert 'transform="translate(2 3)"' in template
    monkeypatch.setattr(renderer, "_compile", lambda source, display, digest: template)

    first = renderer.render(r"\frac{x}{y}")
    second = renderer.render(r"\frac{x}{y}")
    first_root = ET.fromstring(first)
    second_root = ET.fromstring(second)
    first_ids = {node.attrib["id"] for node in first_root.iter() if "id" in node.attrib}
    second_ids = {node.attrib["id"] for node in second_root.iter() if "id" in node.attrib}

    assert first_ids
    assert first_ids.isdisjoint(second_ids)
    assert first_root.attrib["fill"] == "currentColor"
    assert renderer.cache_hit_count == 1


def test_latex_svg_renderer_reports_missing_native_tools(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pdf2epub.epub.math_renderer.shutil.which", lambda name: None)
    renderer = LatexSvgRenderer(tmp_path)

    with pytest.raises(MathRenderingError, match="xelatex, dvisvgm, kpsewhich"):
        renderer.render(r"\frac{x}{y}")


@pytest.mark.parametrize(
    "payload, expected",
    [
        ("<script>alert(1)</script>", "unsupported <script>"),
        ('<a href="https://example.invalid"><path d="M0 0" /></a>', "unsupported <a>"),
        ('<path d="M0 0" onload="alert(1)" />', "unsupported attribute"),
    ],
)
def test_latex_svg_renderer_rejects_executable_raw_specials(
    tmp_path, payload, expected
) -> None:
    renderer = LatexSvgRenderer(tmp_path)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1pt" height="1pt" '
        f'viewBox="0 0 1 1">{payload}</svg>'
    )

    with pytest.raises(MathRenderingError, match=expected):
        renderer._prepare_svg(svg, display=False)
