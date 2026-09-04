import json
from pathlib import Path

import pytest

from pdf2epub.build_epub import process_chapter_content
from pdf2epub.epub.footnotes import (
    FootnoteGraphError,
    FootnoteManager,
    FootnoteStyle,
    inspect_footnote_graph,
    validate_footnote_graph,
)
from pdf2epub.epub.footnotes.content_index import ContentAddressIndex
from pdf2epub.markdown_to_html import convert_markdown_to_html


def _entry(unit_id: str, *part_files: Path, children=None, entry_type=None) -> dict:
    result = {"unit_id": unit_id}
    if part_files:
        result["file_path"] = part_files[0]
        result["part_files"] = list(part_files)
    if children is not None:
        result["children"] = children
    if entry_type:
        result["type"] = entry_type
    return result


def _render(structure: list[dict], manager: FootnoteManager) -> dict[str, str]:
    html_files: dict[str, str] = {}

    def walk(entries: list[dict]) -> None:
        for entry in entries:
            part_files = entry.get("part_files") or []
            for part_index, part_file in enumerate(part_files, 1):
                content = Path(part_file).read_text(encoding="utf-8")
                processed = process_chapter_content(
                    entry.get("title", entry["unit_id"]),
                    entry.get("level", 1),
                    content,
                    is_first_part=(part_index == 1),
                )
                html = convert_markdown_to_html(
                    processed,
                    "Book",
                    standalone=False,
                    footnote_manager=manager,
                    source_chapter=Path(part_file).stem,
                )
                html_name = (
                    f"{entry['unit_id']}_part{part_index}.html"
                    if len(part_files) > 1
                    else f"{entry['unit_id']}.html"
                )
                html_files[html_name] = html
            walk(entry.get("children", []))

    walk(structure)
    return html_files


def _write_section_cache(markdown_dir: Path, matches: list[dict]) -> None:
    book_dir = markdown_dir.parent.parent
    (book_dir / "toc_tree.json").write_text(
        json.dumps(
            {
                "chapters": [
                    {"title": "Body one"},
                    {"title": "Body two"},
                    {"title": "Notes", "type": "notes"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (markdown_dir.parent / "footnote_section_matches.json").write_text(
        json.dumps(matches, ensure_ascii=False),
        encoding="utf-8",
    )


def test_content_address_index_uses_structure_for_leaf_ancestry_and_html_names(
    tmp_path: Path,
) -> None:
    first = tmp_path / "chapter_9.part3.part1.md"
    second = tmp_path / "chapter_9.part3.part2.md"
    structure = [
        _entry(
            "chapter_1",
            children=[
                _entry(
                    "chapter_1.1",
                    children=[_entry("chapter_1.1.1", first, second)],
                )
            ],
        )
    ]

    index = ContentAddressIndex.from_structure(structure)

    assert index.ancestry_for_source(first.stem) == (
        "chapter_1",
        "chapter_1.1",
        "chapter_1.1.1",
    )
    assert index.nearest_scope(first.stem, {"chapter_1", "chapter_1.1"}) == "chapter_1.1"
    assert index.html_for_source(first.stem) == "chapter_1.1.1_part1.html"
    assert index.html_for_source(second.stem) == "chapter_1.1.1_part2.html"


def test_structure_limits_scanning_to_files_that_will_be_built(tmp_path: Path) -> None:
    included = tmp_path / "chapter_1.md"
    stale = tmp_path / "chapter_99.md"
    included.write_text("Body [^1].\n\n[^1]: Definition.\n", encoding="utf-8")
    stale.write_text("Stale marker [^99].\n", encoding="utf-8")
    structure = [_entry("chapter_1", included)]

    manager = FootnoteManager(tmp_path, epub_structure=structure)

    assert "1" in manager.references
    assert "99" not in manager.references


def test_local_notes_cross_refined_siblings_form_a_valid_graph(tmp_path: Path) -> None:
    early = tmp_path / "chapter_1.1.md"
    late = tmp_path / "chapter_1.2.md"
    early.write_text("Early note [^1].\n", encoding="utf-8")
    late.write_text(
        "Late note [^2].\n\n[^1]: Early definition.\n\n[^2]: Late definition.\n",
        encoding="utf-8",
    )
    structure = [
        _entry(
            "chapter_1",
            children=[_entry("chapter_1.1", early), _entry("chapter_1.2", late)],
        )
    ]

    manager = FootnoteManager(tmp_path, epub_structure=structure)
    html_files = _render(structure, manager)
    report = validate_footnote_graph(html_files)

    assert manager.get_style() == FootnoteStyle.LOCAL
    assert manager.get_local_group_id(early.stem) == "chapter_1"
    assert report["forward_hrefs"] == 2
    assert report["backref_hrefs"] == 2


def test_unbalanced_local_occurrences_degrade_to_visible_unlinked_refs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "chapter_1.part1.md"
    second = tmp_path / "chapter_1.part2.md"
    first.write_text("One [^1]. Two [^1].\n", encoding="utf-8")
    second.write_text("[^1]: Only one definition.\n", encoding="utf-8")
    structure = [_entry("chapter_1", first, second)]

    manager = FootnoteManager(tmp_path, epub_structure=structure)
    report = validate_footnote_graph(_render(structure, manager))

    assert report["forward_hrefs"] == 1
    assert report["unlinked_sup_count"] == 1


def test_global_occurrence_mapping_without_semantic_sections_is_valid(
    tmp_path: Path,
) -> None:
    body = tmp_path / "chapter_1.md"
    notes = tmp_path / "chapter_2.md"
    body.write_text("One [^1]. Two [^1].\n", encoding="utf-8")
    notes.write_text(
        "[^1]: First definition.\n\n[^1]: Second definition.\n",
        encoding="utf-8",
    )
    structure = [_entry("chapter_1", body), _entry("chapter_2", notes)]

    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    report = validate_footnote_graph(_render(structure, manager))

    assert manager.notes_sections == []
    assert report["forward_hrefs"] == 2
    assert report["backref_hrefs"] == 0


def test_repeated_notes_fragments_merge_into_hierarchical_semantic_scope(
    tmp_path: Path,
) -> None:
    markdown_dir = tmp_path / "translated" / "validated"
    markdown_dir.mkdir(parents=True)
    body_a = markdown_dir / "chapter_1.1.md"
    body_b = markdown_dir / "chapter_1.2.md"
    notes_a = markdown_dir / "chapter_3.part10.md"
    notes_b = markdown_dir / "chapter_3.part2.md"
    body_a.write_text("First [^1].\n", encoding="utf-8")
    body_b.write_text("Second [^2]. False OCR [^3].\n", encoding="utf-8")
    notes_a.write_text("## Chapter One\n\n[^1]: First.\n", encoding="utf-8")
    notes_b.write_text("### False Heading\n\n[^2]: Second.\n", encoding="utf-8")
    structure = [
        _entry(
            "chapter_1",
            children=[_entry("chapter_1.1", body_a), _entry("chapter_1.2", body_b)],
        ),
        _entry("chapter_3", notes_a, notes_b, entry_type="notes"),
    ]
    _write_section_cache(
        markdown_dir,
        [
            {"header": "## Chapter One", "unit_id": "chapter_1"},
            {"header": "### False Heading", "unit_id": "chapter_1"},
        ],
    )

    manager = FootnoteManager(
        markdown_dir,
        auto_global=True,
        epub_structure=structure,
    )
    report = validate_footnote_graph(_render(structure, manager))

    assert len(manager.notes_sections) == 1
    assert len(manager.notes_sections[0].definitions) == 2
    assert report["forward_hrefs"] == 2
    assert report["backref_hrefs"] == 2
    assert report["unlinked_sup_count"] == 1


def test_semantic_scopes_prevent_reset_keys_from_colliding_or_cross_linking(
    tmp_path: Path,
) -> None:
    markdown_dir = tmp_path / "translated" / "validated"
    markdown_dir.mkdir(parents=True)
    body_one = markdown_dir / "chapter_1.md"
    body_two = markdown_dir / "chapter_2.md"
    notes = markdown_dir / "chapter_3.md"
    body_one.write_text("Real [^1]. Extra [^1].\n", encoding="utf-8")
    body_two.write_text("Other chapter [^1].\n", encoding="utf-8")
    notes.write_text(
        "## One\n\n[^1]: Definition one.\n\n"
        "## Two\n\n[^1]: Definition two.\n",
        encoding="utf-8",
    )
    structure = [
        _entry("chapter_1", body_one),
        _entry("chapter_2", body_two),
        _entry("chapter_3", notes, entry_type="notes"),
    ]
    _write_section_cache(
        markdown_dir,
        [
            {"header": "## One", "unit_id": "chapter_1"},
            {"header": "## Two", "unit_id": "chapter_2"},
        ],
    )

    manager = FootnoteManager(
        markdown_dir,
        auto_global=True,
        epub_structure=structure,
    )
    html_files = _render(structure, manager)
    report = validate_footnote_graph(html_files)

    assert "fn-chapter_1-1-1" in html_files["chapter_3.html"]
    assert "fn-chapter_2-1-1" in html_files["chapter_3.html"]
    assert report["forward_hrefs"] == 2
    assert report["backref_hrefs"] == 2
    assert report["unlinked_sup_count"] == 1


def test_resumed_semantic_scope_keeps_physical_definition_occurrences_aligned(
    tmp_path: Path,
) -> None:
    markdown_dir = tmp_path / "translated" / "validated"
    markdown_dir.mkdir(parents=True)
    body_one = markdown_dir / "chapter_1.md"
    body_two = markdown_dir / "chapter_2.md"
    notes = markdown_dir / "chapter_3.md"
    body_one.write_text("First [^1]. Resumed [^1].\n", encoding="utf-8")
    body_two.write_text("Middle [^1].\n", encoding="utf-8")
    notes.write_text(
        "## One\n\n[^1]: One first.\n\n"
        "## Two\n\n[^1]: Two.\n\n"
        "### One continued\n\n[^1]: One second.\n",
        encoding="utf-8",
    )
    structure = [
        _entry("chapter_1", body_one),
        _entry("chapter_2", body_two),
        _entry("chapter_3", notes, entry_type="notes"),
    ]
    _write_section_cache(
        markdown_dir,
        [
            {"header": "## One", "unit_id": "chapter_1"},
            {"header": "## Two", "unit_id": "chapter_2"},
            {"header": "### One continued", "unit_id": "chapter_1"},
        ],
    )

    manager = FootnoteManager(
        markdown_dir,
        auto_global=True,
        epub_structure=structure,
    )
    report = validate_footnote_graph(_render(structure, manager))

    assert report["forward_hrefs"] == 3
    assert report["backref_hrefs"] == 3
    assert report["duplicate_footnote_id_count"] == 0


def test_no_colon_definitions_are_limited_to_structural_notes_scope(
    tmp_path: Path,
) -> None:
    body = tmp_path / "chapter_1.md"
    notes = tmp_path / "chapter_2.md"
    unrelated = tmp_path / "chapter_3.md"
    body.write_text("Body [^1].\n", encoding="utf-8")
    notes.write_text("[^1] Legacy definition.\n", encoding="utf-8")
    unrelated.write_text("[^2] This remains a reference.\n", encoding="utf-8")
    structure = [
        _entry("chapter_1", body),
        _entry("chapter_2", notes, entry_type="notes"),
        _entry("chapter_3", unrelated),
    ]

    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)

    assert manager.definitions["1"][0].chapter == notes.stem
    assert "2" not in manager.definitions
    assert manager.references["2"][0].chapter == unrelated.stem


def test_page_note_keys_preserve_display_while_using_normalized_definition(
    tmp_path: Path,
) -> None:
    body = tmp_path / "chapter_1.md"
    notes = tmp_path / "chapter_2.md"
    body.write_text("Page note [^197n67].\n", encoding="utf-8")
    notes.write_text("[^67]: Definition.\n", encoding="utf-8")
    structure = [_entry("chapter_1", body), _entry("chapter_2", notes)]

    manager = FootnoteManager(tmp_path, auto_global=True, epub_structure=structure)
    html_files = _render(structure, manager)
    report = validate_footnote_graph(html_files)

    assert "[197n67]" in html_files["chapter_1.html"]
    assert report["forward_hrefs"] == 1


def test_replaced_source_heading_markers_do_not_create_phantom_backlinks(
    tmp_path: Path,
) -> None:
    body = tmp_path / "chapter_1.part1.md"
    notes = tmp_path / "chapter_1.part2.md"
    body.write_text(
        "# Raw title [^1] [^2]\n\n"
        "Body [^1]. Local [^9].\n\n[^9]: Local definition.\n",
        encoding="utf-8",
    )
    notes.write_text(
        "[^1]: Body definition.\n\n[^2]: Replaced-heading definition.\n",
        encoding="utf-8",
    )
    structure = [
        {
            **_entry("chapter_1", body, notes),
            "title": "TOC title",
            "level": 1,
        }
    ]

    manager = FootnoteManager(tmp_path, epub_structure=structure)
    report = validate_footnote_graph(_render(structure, manager))

    assert report["forward_hrefs"] == 2
    assert report["backref_hrefs"] == 2
    assert report["unlinked_sup_count"] == 0


def test_definition_text_markers_are_not_scanned_as_body_references(
    tmp_path: Path,
) -> None:
    chapter = tmp_path / "chapter_1.md"
    chapter.write_text(
        "Body [^1].\n\n"
        "[^1]: *Styled* $\\frac{x}{y}$ mentions literal [^2] and <Book>.\n\n"
        "[^2]: Orphan definition.\n",
        encoding="utf-8",
    )
    structure = [_entry("chapter_1", chapter)]

    manager = FootnoteManager(tmp_path, epub_structure=structure)
    html_files = _render(structure, manager)
    report = validate_footnote_graph(html_files)

    assert "2" not in manager.references
    assert "<em>Styled</em>" in html_files["chapter_1.html"]
    assert "<math" not in html_files["chapter_1.html"]
    assert r"$\frac{x}{y}$" in html_files["chapter_1.html"]
    assert "&lt;Book&gt;" in html_files["chapter_1.html"]
    assert report["forward_hrefs"] == 1
    assert report["backref_hrefs"] == 1


def test_reconfiguration_rescans_markdown_and_rebuilds_structural_state(
    tmp_path: Path,
) -> None:
    chapter = tmp_path / "chapter_1.md"
    chapter.write_text("Body [^1].\n\n[^1]: Definition.\n", encoding="utf-8")
    structure = [_entry("chapter_1", chapter)]
    manager = FootnoteManager(tmp_path, epub_structure=structure)

    chapter.write_text("[^1]: Definition without a reference.\n", encoding="utf-8")
    manager.configure_from_structure(structure)
    report = validate_footnote_graph(_render(structure, manager))

    assert manager.references == {}
    assert report["backref_hrefs"] == 0


@pytest.mark.parametrize(
    "html_files",
    [
        {
            "one.html": '<sup id="fnref-one-1"><a href="two.html#fn-1">[1]</a></sup>',
            "two.html": "<p>No target</p>",
        },
        {
            "one.html": '<div id="fn-1"></div><div id="fn-1"></div>',
        },
        {
            "one.html": '<div id="fn:legacy"></div>',
        },
    ],
)
def test_graph_validator_rejects_broken_targets_and_duplicate_ids(
    html_files: dict[str, str],
) -> None:
    with pytest.raises(FootnoteGraphError):
        validate_footnote_graph(html_files)


def test_graph_inspection_allows_explicit_unlinked_reference() -> None:
    report = inspect_footnote_graph(
        {"one.html": '<p><sup id="fnref-one-1">[1]</sup></p>'}
    )

    assert report["unlinked_sup_count"] == 1
    assert report["forward_broken_count"] == 0
    assert report["backref_broken_count"] == 0
