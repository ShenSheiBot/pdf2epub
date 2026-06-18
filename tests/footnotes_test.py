from pathlib import Path

import pytest

from pdf2epub.build_epub import process_chapter_content
from pdf2epub.epub.footnotes import FootnoteManager, FootnoteStyle
import pdf2epub.epub.footnotes.manager as manager_module
from pdf2epub.markdown_to_html import convert_file, convert_markdown_to_html


def test_local_footnotes_resolve_across_refined_sibling_units(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.1.md").write_text(
        "Early section text with a note [^1].\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_1.2.md").write_text(
        "Final section text with its own note [^2].\n\n"
        "### Notes\n\n"
        "[^1]: Note for the early section.\n\n"
        "[^2]: Note for the final section.\n",
        encoding="utf-8",
    )

    manager = FootnoteManager(tmp_path)
    assert manager.get_style() == FootnoteStyle.LOCAL

    manager.configure_from_structure(
        [
            {
                "unit_id": "chapter_1",
                "children": [
                    {
                        "unit_id": "chapter_1.1",
                        "file_path": tmp_path / "chapter_1.1.md",
                        "part_files": [tmp_path / "chapter_1.1.md"],
                    },
                    {
                        "unit_id": "chapter_1.2",
                        "file_path": tmp_path / "chapter_1.2.md",
                        "part_files": [tmp_path / "chapter_1.2.md"],
                    },
                ],
            }
        ]
    )

    early_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.1",
    )
    final_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.2.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.2",
    )

    assert 'href="chapter_1.2.html#fn:1:1"' in early_html
    assert 'id="fn:1:1"' in final_html
    assert 'href="chapter_1.2.html#fn:2:1"' in final_html
    assert 'id="fn:2:1"' in final_html


def test_local_footnotes_use_occurrence_when_build_inserts_heading(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.part1.md").write_text(
        "First repeated note [^1].\n\nSecond repeated note [^1].\n\n"
        "Local note keeps this chapter in local mode [^9].\n\n"
        "[^9]: Local definition.\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_1.part2.md").write_text(
        "[^1]: First definition.\n\n[^1]: Second definition.\n",
        encoding="utf-8",
    )

    manager = FootnoteManager(tmp_path)
    manager.configure_from_structure(
        [
            {
                "unit_id": "chapter_1",
                "file_path": tmp_path / "chapter_1.md",
                "part_files": [
                    tmp_path / "chapter_1.part1.md",
                    tmp_path / "chapter_1.part2.md",
                ],
            }
        ]
    )

    first_part = process_chapter_content(
        "Inserted Title",
        1,
        (tmp_path / "chapter_1.part1.md").read_text(encoding="utf-8"),
        is_first_part=True,
    )
    second_part = process_chapter_content(
        "Inserted Title",
        1,
        (tmp_path / "chapter_1.part2.md").read_text(encoding="utf-8"),
        is_first_part=False,
    )

    first_html = convert_markdown_to_html(
        first_part,
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part1",
    )
    second_html = convert_markdown_to_html(
        second_part,
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part2",
    )

    assert 'href="chapter_1_part2.html#fn:1:1"' in first_html
    assert 'href="chapter_1_part2.html#fn:1:2"' in first_html
    assert 'id="fn:1:1"' in second_html
    assert 'id="fn:1:2"' in second_html
    assert 'id="fnref-chapter_1.part1-1"' in first_html
    assert 'id="fnref-chapter_1.part1-1-2"' in first_html
    assert 'href="chapter_1_part1.html#fnref-chapter_1.part1-1-2"' in second_html


def test_local_split_part_linked_and_unlinked_repeated_refs_have_distinct_ids(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.part1.md").write_text(
        "Earlier part uses the same key [^1].\n\n"
        "[^9]: Local-only definition keeps this fixture in local mode.\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_1.part2.md").write_text(
        "This part has a linkable repeat [^1].\n\n"
        "This part has an unlinked repeat [^1].\n\n"
        "[^9]: Local-only definition keeps this fixture in local mode.\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_1.part3.md").write_text(
        "[^1]: First definition.\n\n"
        "[^1]: Second definition.\n",
        encoding="utf-8",
    )

    structure = [
        {
            "unit_id": "chapter_1",
            "file_path": tmp_path / "chapter_1.md",
            "part_files": [
                tmp_path / "chapter_1.part1.md",
                tmp_path / "chapter_1.part2.md",
                tmp_path / "chapter_1.part3.md",
            ],
        }
    ]
    manager = FootnoteManager(tmp_path, epub_structure=structure)
    manager.configure_from_structure(structure)

    second_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.part2.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part2",
    )
    third_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.part3.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part3",
    )

    assert 'id="fnref-chapter_1.part2-1"' in second_html
    assert 'id="fnref-chapter_1.part2-1-2"' in second_html
    assert second_html.count('id="fnref-chapter_1.part2-1-2"') == 1
    assert 'href="chapter_1_part3.html#fn:1:2"' in second_html
    assert '<sup id="fnref-chapter_1.part2-1-2">[1]</sup>' in second_html
    assert 'href="chapter_1_part2.html#fnref-chapter_1.part2-1"' in third_html


def test_global_notes_use_occurrence_anchor_for_repeated_keys(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text("First chapter note [^1].\n", encoding="utf-8")
    (tmp_path / "chapter_2.md").write_text("Second chapter note [^1].\n", encoding="utf-8")
    (tmp_path / "chapter_3.md").write_text(
        "[^1]: First chapter definition.\n\n[^1]: Second chapter definition.\n",
        encoding="utf-8",
    )

    structure = [
        {"unit_id": "chapter_1", "file_path": tmp_path / "chapter_1.md", "part_files": [tmp_path / "chapter_1.md"]},
        {"unit_id": "chapter_2", "file_path": tmp_path / "chapter_2.md", "part_files": [tmp_path / "chapter_2.md"]},
        {"unit_id": "chapter_3", "file_path": tmp_path / "chapter_3.md", "part_files": [tmp_path / "chapter_3.md"]},
    ]
    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    manager.configure_from_structure(structure)
    assert manager.get_style() == FootnoteStyle.GLOBAL

    first_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )
    second_html = convert_markdown_to_html(
        (tmp_path / "chapter_2.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_2",
    )
    notes_html = convert_markdown_to_html(
        (tmp_path / "chapter_3.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_3",
    )

    assert 'href="chapter_3.html#fn:1:1"' in first_html
    assert 'href="chapter_3.html#fn:1:2"' in second_html
    assert 'id="fn:1:1"' in notes_html
    assert 'id="fn:1:2"' in notes_html


def test_global_notes_keep_definitions_after_unknown_page_placeholder(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "First note [^1]. Second note [^2].\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_2.md").write_text(
        "[^1]: See above p. $$$.\n"
        "[^2]: Definition after placeholder.\n",
        encoding="utf-8",
    )

    structure = [
        {"unit_id": "chapter_1", "file_path": tmp_path / "chapter_1.md", "part_files": [tmp_path / "chapter_1.md"]},
        {"unit_id": "chapter_2", "file_path": tmp_path / "chapter_2.md", "part_files": [tmp_path / "chapter_2.md"]},
    ]
    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    manager.configure_from_structure(structure)

    text_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )
    notes_html = convert_markdown_to_html(
        (tmp_path / "chapter_2.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_2",
    )

    assert 'href="chapter_2.html#fn:1:1"' in text_html
    assert 'href="chapter_2.html#fn:2:1"' in text_html
    assert 'id="fn:1:1"' in notes_html
    assert 'id="fn:2:1"' in notes_html
    assert "$$$" in notes_html


def test_no_colon_footnote_definitions_only_in_notes_chapters(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "Body note [^1].\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_2.1.md").write_text(
        "[^1] Legacy notes definition without colon.\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_3.md").write_text(
        "[^2] Body-leading marker is still a reference.\n",
        encoding="utf-8",
    )

    structure = [
        {"unit_id": "chapter_1", "file_path": tmp_path / "chapter_1.md", "part_files": [tmp_path / "chapter_1.md"]},
        {
            "unit_id": "chapter_2",
            "type": "notes",
            "children": [
                {
                    "unit_id": "chapter_2.1",
                    "file_path": tmp_path / "chapter_2.1.md",
                    "part_files": [tmp_path / "chapter_2.1.md"],
                },
            ],
        },
        {"unit_id": "chapter_3", "file_path": tmp_path / "chapter_3.md", "part_files": [tmp_path / "chapter_3.md"]},
    ]
    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    manager.configure_from_structure(structure)

    assert "chapter_2.1" in manager.no_colon_definition_chapters
    assert "chapter_2.1" in manager.primary_definition_chapters
    assert "chapter_3" not in manager.no_colon_definition_chapters
    assert manager.references["1"][0].chapter == "chapter_1"
    assert manager.definitions["1"][0].chapter == "chapter_2.1"
    assert "2" in manager.references
    assert "2" not in manager.definitions

    body_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )
    notes_html = convert_markdown_to_html(
        (tmp_path / "chapter_2.1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_2.1",
    )
    body_marker_html = convert_markdown_to_html(
        (tmp_path / "chapter_3.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_3",
    )

    assert 'href="chapter_2.1.html#fn:1:1"' in body_html
    assert 'id="fn:1:1"' in notes_html
    assert '<strong>[1]:</strong> Legacy notes definition without colon.' in notes_html
    assert '<sup id="fnref-chapter_3-2">[2]</sup>' in body_marker_html


def test_local_orphan_definition_does_not_create_broken_backref(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "Referenced note [^1].\n\n"
        "[^1]: Referenced definition.\n"
        "[^2]: Definition with no text reference.\n",
        encoding="utf-8",
    )

    manager = FootnoteManager(tmp_path)
    html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )

    assert 'href="chapter_1.html#fnref-chapter_1-1"' in html
    assert 'id="fn:2"' in html
    assert 'href="chapter_1.html#fnref-chapter_1-2"' not in html


def test_local_single_file_reference_and_definition_ids_are_consistent(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "Body note [^1].\n\n"
        "[^1]: Local definition.\n",
        encoding="utf-8",
    )

    manager = FootnoteManager(tmp_path)
    direct_ref = manager.get_footnote_html(
        "1",
        "chapter_1",
        line_num=1,
        occurrence_in_file=1,
    )
    html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )

    assert 'href="chapter_1.html#fn:1"' in direct_ref
    assert 'href="chapter_1.html#fn:1"' in html
    assert 'id="fn:1"' in html
    assert "#fn:1:1" not in html


def test_local_single_file_repeated_key_links_by_file_occurrence(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "First repeated note [^1]. Second repeated note [^1].\n\n"
        "[^1]: First definition.\n\n"
        "[^1]: Second definition.\n",
        encoding="utf-8",
    )

    manager = FootnoteManager(tmp_path)
    html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )

    assert 'id="fnref-chapter_1-1"' in html
    assert 'id="fnref-chapter_1-1-2"' in html
    assert 'href="chapter_1.html#fn:1"' in html
    assert 'href="chapter_1.html#fn:1:2"' in html
    assert 'id="fn:1"' in html
    assert 'id="fn:1:2"' in html
    assert 'href="chapter_1.html#fnref-chapter_1-1"' in html
    assert 'href="chapter_1.html#fnref-chapter_1-1-2"' in html


def test_global_repeated_key_in_same_file_uses_file_occurrence(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "First note [^1]. Second note [^1].\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_2.md").write_text(
        "[^1]: First definition.\n\n"
        "[^1]: Second definition.\n",
        encoding="utf-8",
    )

    structure = [
        {"unit_id": "chapter_1", "file_path": tmp_path / "chapter_1.md", "part_files": [tmp_path / "chapter_1.md"]},
        {"unit_id": "chapter_2", "file_path": tmp_path / "chapter_2.md", "part_files": [tmp_path / "chapter_2.md"]},
    ]
    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    manager.configure_from_structure(structure)

    text_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )
    notes_html = convert_markdown_to_html(
        (tmp_path / "chapter_2.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_2",
    )

    assert 'href="chapter_2.html#fn:1:1"' in text_html
    assert 'href="chapter_2.html#fn:1:2"' in text_html
    assert 'id="fnref-chapter_1-1"' in text_html
    assert 'id="fnref-chapter_1-1-2"' in text_html
    assert 'id="fn:1:1"' in notes_html
    assert 'id="fn:1:2"' in notes_html


def test_page_note_reference_displays_original_key_and_links_normalized_key(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "Page-note style reference [^197n67].\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_2.md").write_text(
        "[^67]: Page note definition.\n",
        encoding="utf-8",
    )

    structure = [
        {"unit_id": "chapter_1", "file_path": tmp_path / "chapter_1.md", "part_files": [tmp_path / "chapter_1.md"]},
        {"unit_id": "chapter_2", "file_path": tmp_path / "chapter_2.md", "part_files": [tmp_path / "chapter_2.md"]},
    ]
    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    manager.configure_from_structure(structure)

    text_html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )
    notes_html = convert_markdown_to_html(
        (tmp_path / "chapter_2.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_2",
    )

    assert 'href="chapter_2.html#fn:67:1"' in text_html
    assert "[197n67]" in text_html
    assert 'id="fn:67:1"' in notes_html


def test_replaced_first_heading_reference_does_not_shift_local_occurrence(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.part1.md").write_text(
        "# Raw title note [^1] [^2]\n\n"
        "Body note still exists [^1].\n\n"
        "Local note keeps this in local mode [^9].\n\n"
        "[^9]: Local definition.\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_1.part2.md").write_text(
        "[^1]: Body definition.\n"
        "[^2]: Heading-only definition.\n",
        encoding="utf-8",
    )

    structure = [
        {
            "unit_id": "chapter_1",
            "title": "TOC Title",
            "level": 1,
            "file_path": tmp_path / "chapter_1.part1.md",
            "part_files": [
                tmp_path / "chapter_1.part1.md",
                tmp_path / "chapter_1.part2.md",
            ],
        }
    ]
    manager = FootnoteManager(tmp_path, epub_structure=structure)
    manager.configure_from_structure(structure)

    first_part = process_chapter_content(
        "TOC Title",
        1,
        (tmp_path / "chapter_1.part1.md").read_text(encoding="utf-8"),
        is_first_part=True,
    )
    second_part = process_chapter_content(
        "TOC Title",
        1,
        (tmp_path / "chapter_1.part2.md").read_text(encoding="utf-8"),
        is_first_part=False,
    )

    first_html = convert_markdown_to_html(
        first_part,
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part1",
    )
    second_html = convert_markdown_to_html(
        second_part,
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part2",
    )

    assert 'href="chapter_1_part2.html#fn:1:1"' in first_html
    assert 'id="fn:1:1"' in second_html
    assert 'href="chapter_1_part1.html#fnref-chapter_1.part1-1"' in second_html
    assert 'href="chapter_1_part1.html#fnref-chapter_1.part1-2"' not in second_html


def test_late_structure_config_suppresses_replaced_heading_reference(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.part1.md").write_text(
        "# Raw title note [^1] [^2]\n\n"
        "Body note still exists [^1].\n\n"
        "Local note keeps this in local mode [^9].\n\n"
        "[^9]: Local definition.\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_1.part2.md").write_text(
        "[^1]: Body definition.\n"
        "[^2]: Heading-only definition.\n",
        encoding="utf-8",
    )

    structure = [
        {
            "unit_id": "chapter_1",
            "title": "TOC Title",
            "level": 1,
            "file_path": tmp_path / "chapter_1.part1.md",
            "part_files": [
                tmp_path / "chapter_1.part1.md",
                tmp_path / "chapter_1.part2.md",
            ],
        }
    ]
    manager = FootnoteManager(tmp_path)
    manager.configure_from_structure(structure)

    first_part = process_chapter_content(
        "TOC Title",
        1,
        (tmp_path / "chapter_1.part1.md").read_text(encoding="utf-8"),
        is_first_part=True,
    )
    second_part = process_chapter_content(
        "TOC Title",
        1,
        (tmp_path / "chapter_1.part2.md").read_text(encoding="utf-8"),
        is_first_part=False,
    )

    first_html = convert_markdown_to_html(
        first_part,
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part1",
    )
    second_html = convert_markdown_to_html(
        second_part,
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1.part2",
    )

    assert 'href="chapter_1_part2.html#fn:1:1"' in first_html
    assert 'id="fn:1:1"' in second_html
    assert 'href="chapter_1_part1.html#fnref-chapter_1.part1-1"' in second_html
    assert 'href="chapter_1_part1.html#fnref-chapter_1.part1-2"' not in second_html


def test_configure_from_structure_rescans_current_markdown(tmp_path: Path) -> None:
    chapter = tmp_path / "chapter_1.md"
    chapter.write_text(
        "Referenced note [^1].\n\n"
        "[^1]: Definition.\n",
        encoding="utf-8",
    )
    structure = [
        {"unit_id": "chapter_1", "file_path": chapter, "part_files": [chapter]},
    ]
    manager = FootnoteManager(tmp_path, epub_structure=structure)

    chapter.write_text(
        "Reference was removed by cleanup.\n\n"
        "[^1]: Definition.\n",
        encoding="utf-8",
    )
    manager.configure_from_structure(structure)

    html = convert_markdown_to_html(
        chapter.read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )

    assert 'id="fn:1"' in html
    assert 'footnote-backref' not in html


def test_configure_from_structure_clears_stale_section_matcher_state(tmp_path: Path) -> None:
    chapter = tmp_path / "chapter_1.md"
    chapter.write_text(
        "Referenced note [^1].\n\n"
        "[^1]: Definition.\n",
        encoding="utf-8",
    )
    structure = [
        {"unit_id": "chapter_1", "file_path": chapter, "part_files": [chapter]},
    ]
    manager = FootnoteManager(tmp_path, epub_structure=structure)

    class StaleMatcher:
        notes_sections = [object()]
        chapter_to_section = {"chapter_1": object()}

    manager.llm_matcher = StaleMatcher()
    manager.mapper.section_definition_by_occurrence[(0, "1", 1)] = object()
    manager.mapper.section_definition_occurrence_by_line[("1", "chapter_1", 1)] = (0, 1)
    manager.mapper.section_definition_occurrence_in_file[("1", "chapter_1", 1)] = (0, 1)

    manager.configure_from_structure(structure)

    assert manager.llm_matcher is None
    assert manager.mapper.section_definition_by_occurrence == {}
    assert manager.mapper.section_definition_occurrence_by_line == {}
    assert manager.mapper.section_definition_occurrence_in_file == {}


def test_configure_from_structure_rebuilds_llm_matcher_when_enabled(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "Referenced global note [^1].\n",
        encoding="utf-8",
    )
    (tmp_path / "chapter_2.md").write_text(
        "[^1]: Global definition.\n",
        encoding="utf-8",
    )
    structure = [
        {"unit_id": "chapter_1", "file_path": tmp_path / "chapter_1.md", "part_files": [tmp_path / "chapter_1.md"]},
        {"unit_id": "chapter_2", "file_path": tmp_path / "chapter_2.md", "part_files": [tmp_path / "chapter_2.md"]},
    ]

    class FakeMatcher:
        created = 0

        def __init__(self, markdown_dir, config):
            FakeMatcher.created += 1
            self.markdown_dir = markdown_dir
            self.config = config
            self.notes_sections = []
            self.chapter_to_section = {}

        def load_toc_tree(self):
            return True

        def match_sections(self, primary_definition_chapters):
            return False

    monkeypatch.setattr(manager_module, "LLMSectionMatcher", FakeMatcher)
    manager = FootnoteManager(tmp_path, auto_global=True, config={"provider": "fake"}, epub_structure=structure)
    created_after_init = FakeMatcher.created

    manager.llm_matcher = None
    manager.configure_from_structure(structure)

    assert FakeMatcher.created == created_after_init + 1
    assert isinstance(manager.llm_matcher, FakeMatcher)


def test_definition_text_marker_is_not_counted_as_body_reference(tmp_path: Path) -> None:
    (tmp_path / "chapter_1.md").write_text(
        "Body note [^1].\n\n"
        "[^1]: This definition mentions literal [^2].\n"
        "[^2]: Definition with no body reference.\n",
        encoding="utf-8",
    )

    manager = FootnoteManager(tmp_path)
    assert "2" not in manager.references

    html = convert_markdown_to_html(
        (tmp_path / "chapter_1.md").read_text(encoding="utf-8"),
        "Book",
        standalone=False,
        footnote_manager=manager,
        source_chapter="chapter_1",
    )

    assert 'href="chapter_1.html#fnref-chapter_1-1"' in html
    assert 'href="chapter_1.html#fnref-chapter_1-2"' not in html


def test_markdown_without_footnotes_does_not_require_manager() -> None:
    html = convert_markdown_to_html(
        "Plain paragraph without notes.",
        "Book",
        standalone=False,
    )

    assert "Plain paragraph without notes." in html


def test_markdown_footnotes_require_manager() -> None:
    with pytest.raises(ValueError, match="Footnote markdown requires FootnoteManager"):
        convert_markdown_to_html(
            "Body note [^1].\n\n[^1]: Definition.\n",
            "Book",
            standalone=False,
        )


def test_convert_file_uses_manager_for_footnotes(tmp_path: Path) -> None:
    input_path = tmp_path / "chapter_1.md"
    output_path = tmp_path / "chapter_1.html"
    input_path.write_text(
        "Body note [^1].\n\n[^1]: Definition.\n",
        encoding="utf-8",
    )

    assert convert_file(input_path, output_path, standalone=False)
    html = output_path.read_text(encoding="utf-8")

    assert 'id="fnref-chapter_1-1"' in html
    assert 'href="chapter_1.html#fnref-chapter_1-1"' in html
