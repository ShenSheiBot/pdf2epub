from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from pdf2epub.build_epub import generate_hierarchical_toc_ncx
from pdf2epub.epub.builder import EpubBuilder


def test_ncx_and_opf_share_publication_identifier(tmp_path: Path) -> None:
    config = SimpleNamespace(
        book_title="Test Book",
        author="Test Author",
        language="en",
    )
    builder = EpubBuilder(config)
    ncx_path = tmp_path / "toc.ncx"
    opf_path = tmp_path / "content.opf"

    assert builder.create_toc_ncx({"chapters": []}, ncx_path)
    assert builder.create_content_opf(
        {"chapters": []},
        tmp_path,
        opf_path,
        all_html_files=[],
    )

    ncx = ET.parse(ncx_path)
    opf = ET.parse(opf_path)
    ncx_uid = ncx.find(
        ".//{http://www.daisy.org/z3986/2005/ncx/}meta[@name='dtb:uid']"
    ).attrib["content"]
    opf_uid = opf.find(
        ".//{http://purl.org/dc/elements/1.1/}identifier"
    ).text
    assert ncx_uid == opf_uid == builder.uid


def test_hierarchical_ncx_reuses_play_order_for_same_target(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "toc.ncx"
    structure = [
        {
            "title": "Parent",
            "children": [
                {
                    "title": "First child",
                    "unit_id": "chapter_1",
                    "file_path": tmp_path / "chapter_1.md",
                    "part_files": [],
                }
            ],
        }
    ]

    assert generate_hierarchical_toc_ncx(
        structure,
        "Test Book",
        output_path,
        uid="test-publication-id",
    )

    root = ET.parse(output_path)
    ns = {"n": "http://www.daisy.org/z3986/2005/ncx/"}
    points = root.findall(".//n:navPoint", ns)
    targets = [
        point.find("n:content", ns).attrib["src"]
        for point in points
    ]
    play_orders = [point.attrib["playOrder"] for point in points]
    same_target_orders = [
        order
        for target, order in zip(targets, play_orders)
        if target == "text/chapter_1.html"
    ]

    assert same_target_orders == ["2", "2"]
    assert len({point.attrib["id"] for point in points}) == len(points)


def test_epub_stylesheet_uses_a_valid_quote_string() -> None:
    stylesheet = (
        Path(__file__).parents[1]
        / "pdf2epub"
        / "epub"
        / "resources"
        / "stylesheet.css"
    ).read_text()

    assert 'content: "“";' in stylesheet
    assert 'content: """;' not in stylesheet
