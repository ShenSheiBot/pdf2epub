#!/usr/bin/env python3
"""Audit generated footnote links against existing book outputs.

This is an opt-in regression tool for local book archives. It intentionally is
not a pytest test: it depends on the caller's output/ corpus. The script is
read-only and builds chapter HTML in memory using the same footnote conversion
path as build_epub.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from pdf2epub.build_epub import (
    build_epub_structure,
    flatten_toc_tree,
    has_notes_chapter,
    load_toc_tree,
    markdown_to_html,
    process_chapter_content,
)
from pdf2epub.epub.footnotes import FootnoteManager, inspect_footnote_graph


STAGE_CANDIDATES = (
    ("translated_validated", ("translated", "validated"), "toc_tree_translated.json", True),
    ("translated", ("translated",), "toc_tree_translated.json", True),
    ("polished_validated", ("polished_markdown", "validated"), "toc_tree.json", False),
    ("polished", ("polished_markdown",), "toc_tree.json", False),
    ("ocr_markdown", ("ocr_markdown",), "toc_tree.json", False),
    ("translated_original_toc", ("translated",), "toc_tree.json", True),
)

BOOK_MARKERS = (
    "input.pdf",
    "input.epub",
    "input_original.pdf",
    "pages",
    "translated",
    "polished_markdown",
    "compressed_units",
    "toc_tree.json",
    "toc_tree_translated.json",
    "book_structure.json",
)

NON_BOOK_DIRS = {
    "pages",
    "logs",
    "translated",
    "translated_compressed",
    "polished_markdown",
    "compressed_units",
    "ocr_markdown",
    "images",
    "epub_build",
    "validated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=["output", "output/archive"],
        help="Book output roots to scan, default: output output/archive",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/pdf2epub_footnote_audit.json"),
        help="JSON report path",
    )
    parser.add_argument(
        "--book",
        action="append",
        default=[],
        help="Only scan books whose path contains this substring. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N discovered book dirs")
    parser.add_argument(
        "--fail-on-unlinked",
        action="store_true",
        help="Exit non-zero when visible unlinked footnote refs remain",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show warnings from the underlying build/footnote pipeline",
    )
    return parser.parse_args()


def path_from_parts(root: Path, parts: tuple[str, ...]) -> Path:
    path = root
    for part in parts:
        path /= part
    return path


def has_chapter_markdown(path: Path) -> bool:
    return path.is_dir() and any(path.glob("chapter_*.md"))


def select_stage(book_dir: Path) -> tuple[str, Path, Path, bool] | None:
    for stage_name, md_parts, toc_name, translated in STAGE_CANDIDATES:
        markdown_dir = path_from_parts(book_dir, md_parts)
        toc_path = book_dir / toc_name
        if toc_path.exists() and has_chapter_markdown(markdown_dir):
            return stage_name, markdown_dir, toc_path, translated
    return None


def discover_book_dirs(roots: list[Path], filters: list[str], limit: int) -> list[Path]:
    seen: set[Path] = set()
    books: list[Path] = []

    def looks_like_book_dir(path: Path) -> bool:
        return any((path / marker).exists() for marker in BOOK_MARKERS)

    def add_candidate(path: Path) -> None:
        if not path.is_dir() or path.name in NON_BOOK_DIRS or path.name == "archive":
            return
        path_text = str(path)
        resolved = path.resolve()
        if filters and not any(f in path_text for f in filters):
            return
        if resolved in seen:
            return
        seen.add(resolved)
        books.append(path)

    for root in roots:
        if not root.exists():
            continue

        if looks_like_book_dir(root):
            add_candidate(root)
        else:
            for child in sorted(root.iterdir(), key=lambda p: str(p)):
                add_candidate(child)
                if limit and len(books) >= limit:
                    break
        if limit and len(books) >= limit:
            break

    if limit:
        books = books[:limit]
    return sorted(books, key=lambda p: str(p))


def walk_entries(entries: list[dict[str, Any]]):
    for entry in entries:
        yield entry
        yield from walk_entries(entry.get("children", []))


def render_html_files(
    book_title: str,
    language: str,
    epub_structure: list[dict[str, Any]],
    footnote_manager: FootnoteManager,
) -> dict[str, str]:
    html_files: dict[str, str] = {}

    for entry in walk_entries(epub_structure):
        if "file_path" not in entry:
            continue

        unit_id = entry.get("unit_id")
        if not unit_id:
            continue

        part_files = entry.get("part_files") or [entry["file_path"]]
        for part_idx, part_file in enumerate(part_files):
            part_path = Path(part_file)
            content = part_path.read_text(encoding="utf-8")
            processed = process_chapter_content(
                entry["title"],
                entry["level"],
                content,
                is_first_part=(part_idx == 0),
            )
            html = markdown_to_html(
                processed,
                book_title,
                language,
                footnote_manager=footnote_manager,
                image_mapping={},
                source_chapter=part_path.stem,
            )
            if len(part_files) > 1:
                html_name = f"{unit_id}_part{part_idx + 1}.html"
            else:
                html_name = f"{unit_id}.html"
            html_files[html_name] = html

    return html_files


def audit_book(book_dir: Path) -> dict[str, Any]:
    selected = select_stage(book_dir)
    if not selected:
        return {"book": str(book_dir), "status": "skipped_no_compatible_markdown_or_toc"}

    stage, markdown_dir, toc_path, translated = selected
    try:
        toc_tree = load_toc_tree(toc_path)
        toc_structure = flatten_toc_tree(toc_tree["chapters"])
        epub_structure = build_epub_structure(toc_structure, markdown_dir)
        auto_global = has_notes_chapter(toc_tree.get("chapters", []))

        manager = FootnoteManager(
            markdown_dir,
            auto_global=auto_global,
            epub_structure=epub_structure,
        )

        references = sum(len(refs) for refs in manager.references.values())
        definitions = sum(len(defs) for defs in manager.definitions.values())
        if references == 0 and definitions == 0:
            return {
                "book": str(book_dir),
                "stage": stage,
                "status": "no_markdown_footnotes",
            }

        book_title = toc_tree.get("book_title") or book_dir.name
        language = "zh" if translated else toc_tree.get("language", "en")
        language = {"english": "en", "japanese": "ja", "chinese": "zh"}.get(
            str(language).lower(),
            language,
        )

        html_files = render_html_files(book_title, language, epub_structure, manager)
        link_report = inspect_footnote_graph(html_files)

        missing_keys = sorted(k for k in manager.references if k not in manager.definitions)
        true_missing_ref_count = sum(len(manager.references[k]) for k in missing_keys)

        coverage = [manager.get_style().value]
        if any(len((entry.get("part_files") or [])) > 1 for entry in walk_entries(epub_structure)):
            coverage.append("same_unit_split_part")
        if auto_global:
            coverage.append("global_notes")
        if link_report["unlinked_sup_count"]:
            coverage.append("unlinked_refs")
        if true_missing_ref_count:
            coverage.append("missing_def")

        return {
            "book": str(book_dir),
            "stage": stage,
            "style": manager.get_style().value,
            "auto_global": auto_global,
            "html_files": len(html_files),
            "references": references,
            "definitions": definitions,
            **link_report,
            "true_missing_key_count": len(missing_keys),
            "true_missing_ref_count": true_missing_ref_count,
            "true_missing_keys_sample": missing_keys[:50],
            "coverage": coverage,
            "status": "ok",
        }
    except Exception as exc:  # pragma: no cover - diagnostic tool
        return {
            "book": str(book_dir),
            "stage": stage,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.verbose else "ERROR")

    roots = [Path(root) for root in args.roots]
    books = discover_book_dirs(roots, args.book, args.limit)
    results = [audit_book(book) for book in books]

    summary = Counter(result["status"] for result in results)
    coverage = Counter()
    for result in results:
        coverage.update(result.get("coverage", []))

    ok_results = [r for r in results if r.get("status") == "ok"]
    totals = {
        "books_scanned": len(results),
        "compatible_books": sum(1 for r in results if r.get("status") in {"ok", "no_markdown_footnotes"}),
        "references": sum(int(r.get("references", 0) or 0) for r in ok_results),
        "definitions": sum(int(r.get("definitions", 0) or 0) for r in ok_results),
        "forward_broken": sum(int(r.get("forward_broken_count", 0) or 0) for r in ok_results),
        "backref_broken": sum(int(r.get("backref_broken_count", 0) or 0) for r in ok_results),
        "duplicate_footnote_ids": sum(int(r.get("duplicate_footnote_id_count", 0) or 0) for r in ok_results),
        "invalid_footnote_ids": sum(int(r.get("invalid_footnote_id_count", 0) or 0) for r in ok_results),
        "unlinked": sum(int(r.get("unlinked_sup_count", 0) or 0) for r in ok_results),
        "true_missing_refs": sum(int(r.get("true_missing_ref_count", 0) or 0) for r in ok_results),
    }

    report = {
        "summary": dict(summary),
        "coverage": dict(coverage),
        "totals": totals,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(args.report), **totals, "summary": dict(summary)}, ensure_ascii=False))

    has_broken = (
        totals["forward_broken"]
        or totals["backref_broken"]
        or totals["duplicate_footnote_ids"]
        or totals["invalid_footnote_ids"]
    )
    has_error = bool(summary.get("error"))
    has_unlinked = bool(totals["unlinked"])
    if has_broken or has_error or (args.fail_on_unlinked and has_unlinked):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
