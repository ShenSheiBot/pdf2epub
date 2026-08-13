import json
from pathlib import Path

import fitz
from PIL import Image

from pdf2epub.ocr.artifacts import OCRPageResult
from pdf2epub.ocr.backends.chandra import ChandraClient, layout_to_markdown, materialize_layout
from pdf2epub.ocr_pages import ocr_full_book_pagewise, save_page_artifacts


def test_materializer_preserves_blocks_and_crops_nested_table_images(tmp_path):
    raw = """
<div data-bbox="0 0 1000 100" data-label="Page-Header"><p>RUNNING</p></div>
<div data-bbox="100 100 900 800" data-label="Table">
  <table><tr><td>A</td><td><img data-bbox="600 300 800 600" alt="diagram"></td></tr></table>
</div>
<div data-bbox="50 50 950 950" data-label="Blank-Page"><p>blank white page</p></div>
<div data-bbox="0 900 1000 1000" data-label="Page-Footer"><p>42</p></div>
""".strip()
    image = Image.new("RGB", (1000, 2000), "white")

    result = materialize_layout(raw, image, tmp_path / "images", 7)

    assert [block["label"] for block in result.blocks] == [
        "Page-Header", "Table", "Blank-Page", "Page-Footer"
    ]
    assert result.assets == [
        {
            "name": "page_007_img_001.png",
            "path": "../images/page_007_img_001.png",
            "bbox": [600, 300, 800, 600],
            "bbox_px": [600, 600, 800, 1200],
            "block_order": 1,
            "nested": True,
            "alt": "diagram",
        }
    ]
    assert (tmp_path / "images" / "page_007_img_001.png").stat().st_size > 0
    assert 'src="../images/page_007_img_001.png"' in result.html
    assert 'data-label="Blank-Page"' in result.html

    markdown = layout_to_markdown(result.html, include_headers_footers=False)
    assert "RUNNING" not in markdown
    assert "42" not in markdown
    # Blank-page content is not discarded by a hard-coded semantic rule.
    assert "blank white page" in markdown
    assert "<table>" in markdown


def test_save_page_artifacts_writes_all_views_and_raw_layout(tmp_path):
    result = OCRPageResult(
        markdown="Body",
        html='<div data-label="Text"><p>Body</p></div>',
        raw_html='<div data-bbox="1 2 3 4" data-label="Text"><p>Body</p></div>',
        blocks=[{"order": 0, "label": "Text", "bbox_px": [1, 2, 3, 4]}],
        page_box=[0, 0, 100, 200],
        model_input_size=[112, 196],
        token_count=3,
        backend="chandra",
        model="chandra",
        model_revision="revision",
    )

    markdown_path = save_page_artifacts(result, tmp_path, 12)

    assert markdown_path.read_text() == "Body"
    assert (tmp_path / "page_012.html").read_text().startswith("<div")
    assert "data-bbox" in (tmp_path / "page_012.raw.html").read_text()
    sidecar = json.loads((tmp_path / "page_012.ocr.json").read_text())
    assert sidecar["formats"] == {
        "markdown": "page_012.md",
        "html": "page_012.html",
        "raw_html": "page_012.raw.html",
    }
    assert sidecar["raw_html"] == result.raw_html
    assert sidecar["blocks"] == result.blocks


def test_pagewise_ocr_marks_progress_only_after_rich_artifacts_exist(tmp_path, monkeypatch):
    pdf_path = tmp_path / "input.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    output_dir = tmp_path / "output"

    def fake_ocr_page(*args, **kwargs):
        return OCRPageResult(
            markdown="page body",
            html='<div data-label="Text"><p>page body</p></div>',
            raw_html='<div data-bbox="0 0 1000 1000" data-label="Text"><p>page body</p></div>',
            blocks=[{"order": 0, "label": "Text", "bbox_px": [0, 0, 10, 10]}],
            page_box=[0, 0, 10, 10],
            model_input_size=[28, 28],
            token_count=2,
            backend="chandra",
            model="chandra",
        )

    monkeypatch.setattr("pdf2epub.ocr_pages.ocr_pdf_page", fake_ocr_page)
    ocr_full_book_pagewise(
        pdf_path=pdf_path,
        output_dir=output_dir,
        backend="chandra",
        config={"ocr": {"backends": {"chandra": {}}}},
        max_workers=1,
    )

    pages = output_dir / "pages"
    assert (pages / "page_001.md").exists()
    assert (pages / "page_001.html").exists()
    assert (pages / "page_001.raw.html").exists()
    assert (pages / "page_001.ocr.json").exists()
    progress = json.loads((pages / "ocr_progress.json").read_text())
    assert progress["pages_processed"] == [1]
    stats = json.loads((pages / "page_stats.json").read_text())
    assert stats["1"]["html_file"] == "pages/page_001.html"
    assert stats["1"]["artifact_file"] == "pages/page_001.ocr.json"


def test_chandra_rejects_token_limited_partial_layout(monkeypatch):
    client = ChandraClient(
        {"ocr": {"backends": {"chandra": {"max_retries": 1}}}}
    )
    monkeypatch.setattr(
        "pdf2epub.ocr.backends.chandra.render_pdf_page",
        lambda *args, **kwargs: Image.new("RGB", (100, 100), "white"),
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: (
            '<div data-bbox="0 0 1000 1000" data-label="Table"><table><tr>',
            12384,
            "length",
        ),
    )

    try:
        client.process_pdf_page(
            b"unused",
            page_number=1,
            images_dir=None,
            image_counter=0,
        )
    except RuntimeError as error:
        assert "finish_reason='length'" in str(error)
    else:
        raise AssertionError("token-limited Chandra output must not be accepted")
