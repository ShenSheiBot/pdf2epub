from pdf2epub.core.book_structure import BookStructure


def test_short_text_without_images_is_not_image_only(tmp_path):
    structure = BookStructure(tmp_path)

    assert not structure.is_image_only_content("## 目次\n\n第一章 短い章題")


def test_image_with_only_a_short_caption_is_image_only(tmp_path):
    structure = BookStructure(tmp_path)

    assert structure.is_image_only_content("![cover](images/cover.jpg)\n\n表紙")


def test_html_image_detection_is_case_insensitive(tmp_path):
    structure = BookStructure(tmp_path)

    assert structure.is_image_only_content('<IMG SRC="images/cover.jpg">\n\n表紙')
