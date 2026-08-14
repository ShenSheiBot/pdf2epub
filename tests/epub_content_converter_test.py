from types import SimpleNamespace

from pdf2epub.epub.converter import ContentConverter
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
