from pdf2epub.processors.utils.image_restore import (
    _snap_out_of_spans,
    restore_lost_images_fast,
)


def test_restore_skips_same_image_src_with_different_markup() -> None:
    original = (
        "PART I\n\n"
        '<div style="text-align: center;"><img src="../images/page_31_img_3100.jpeg" '
        'alt="Image" width="18%" /></div>\n\n'
        "# 1 TITULATURE\n\nBody text."
    )
    polished = (
        "![Image](../images/page_31_img_3100.jpeg)\n\n"
        "# 1 TITULATURE\n\nBody text."
    )

    restored = restore_lost_images_fast(original, polished)

    assert restored == polished
    assert restored.count("page_31_img_3100.jpeg") == 1


def test_restore_inserts_truly_missing_image() -> None:
    image = (
        '<div style="text-align: center;"><img src="../images/page_60_img_6000.jpeg" '
        'alt="Image" width="72%" /></div>'
    )
    following = (
        "Following paragraph with enough trailing content to keep the image away "
        "from the end-of-unit fallback path."
    )
    original = f"Intro paragraph.\n\n{image}\n\n{following}"
    polished = f"Intro paragraph.\n\n{following}"

    restored = restore_lost_images_fast(original, polished)

    assert image in restored
    assert restored.index("Intro paragraph.") < restored.index(image)
    assert restored.index(image) < restored.index(following)


def test_restore_insertion_position_snaps_out_of_image_span() -> None:
    text = "![Image](../images/page_31_img_3100.jpeg)\n\n# 1 TITULATURE"
    image_span = [(0, text.index("\n\n"))]

    assert _snap_out_of_spans(2, image_span, len(text)) == image_span[0][1]
