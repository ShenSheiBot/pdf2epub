from types import SimpleNamespace

from pdf2epub.epub.converter import ContentConverter


def test_duplicate_title_cleanup_preserves_repeated_symbolic_section_breaks(
    tmp_path,
) -> None:
    chapter = tmp_path / "chapter_1.md"
    original = "# Chapter\n\n## *\n\nFirst section.\n\n## *\n\nSecond section.\n"
    chapter.write_text(original, encoding="utf-8")

    converter = ContentConverter(SimpleNamespace(markdown_dir=tmp_path))

    assert converter.remove_duplicate_titles() == 0
    assert chapter.read_text(encoding="utf-8") == original
