"""Whole-book invariants for generated footnote links."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from bs4 import BeautifulSoup


_FOOTNOTE_ID_PREFIXES = ("fn-", "fnref-", "fn:", "fnref:")
_REFERENCE_ID_PREFIXES = ("fnref-", "fnref:")


class FootnoteGraphError(ValueError):
    """Raised when generated footnote anchors do not form a valid link graph."""


def inspect_footnote_graph(html_files: Mapping[str, str]) -> dict[str, Any]:
    """Inspect forward links, backlinks, duplicate IDs, and conservative refs."""
    ids_by_file: dict[str, set[str]] = {}
    duplicate_ids: list[tuple[str, str, int]] = []
    invalid_ids: list[tuple[str, str]] = []
    hrefs: list[tuple[str, str, str]] = []
    unlinked: list[tuple[str, str, str]] = []

    for filename, html in html_files.items():
        soup = BeautifulSoup(html, "html.parser")
        id_counts = Counter(
            tag["id"]
            for tag in soup.find_all(attrs={"id": True})
            if tag.get("id")
        )
        ids_by_file[filename] = set(id_counts)
        duplicate_ids.extend(
            (filename, id_value, count)
            for id_value, count in id_counts.items()
            if count > 1 and id_value.startswith(_FOOTNOTE_ID_PREFIXES)
        )
        invalid_ids.extend(
            (filename, id_value)
            for id_value in id_counts
            if id_value.startswith(_FOOTNOTE_ID_PREFIXES)
            and (":" in id_value or any(character.isspace() for character in id_value))
        )

        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if "#" not in href:
                continue
            before_hash, target_id = href.split("#", 1)
            if not target_id.startswith(_FOOTNOTE_ID_PREFIXES):
                continue
            target_file = Path(before_hash).name if before_hash else filename
            hrefs.append((filename, target_file, target_id))

        for sup in soup.find_all("sup"):
            sup_id = sup.get("id", "")
            if not sup_id.startswith(_REFERENCE_ID_PREFIXES):
                continue
            if not sup.find("a", href=True):
                unlinked.append((filename, sup_id, sup.get_text("", strip=True)))

    forward_broken: list[tuple[str, str]] = []
    backref_broken: list[tuple[str, str]] = []
    forward_hrefs = 0
    backref_hrefs = 0
    for source_file, target_file, target_id in hrefs:
        if target_id.startswith(("fn-", "fn:")) and not target_id.startswith(
            _REFERENCE_ID_PREFIXES
        ):
            forward_hrefs += 1
            broken_bucket = forward_broken
        else:
            backref_hrefs += 1
            broken_bucket = backref_broken
        if target_id not in ids_by_file.get(target_file, set()):
            broken_bucket.append((source_file, f"{target_file}#{target_id}"))

    return {
        "hrefs": len(hrefs),
        "forward_hrefs": forward_hrefs,
        "backref_hrefs": backref_hrefs,
        "forward_broken_count": len(forward_broken),
        "backref_broken_count": len(backref_broken),
        "forward_broken_samples": forward_broken[:20],
        "backref_broken_samples": backref_broken[:20],
        "duplicate_footnote_id_count": len(duplicate_ids),
        "duplicate_footnote_id_samples": duplicate_ids[:20],
        "invalid_footnote_id_count": len(invalid_ids),
        "invalid_footnote_id_samples": invalid_ids[:20],
        "unlinked_sup_count": len(unlinked),
        "unlinked_samples": unlinked[:20],
    }


def validate_footnote_graph(html_files: Mapping[str, str]) -> dict[str, Any]:
    """Fail when generated links are broken or footnote IDs are ambiguous."""
    report = inspect_footnote_graph(html_files)
    hard_failures = (
        report["forward_broken_count"]
        + report["backref_broken_count"]
        + report["duplicate_footnote_id_count"]
        + report["invalid_footnote_id_count"]
    )
    if hard_failures:
        raise FootnoteGraphError(
            "Invalid generated footnote graph: "
            f"{report['forward_broken_count']} broken forward link(s), "
            f"{report['backref_broken_count']} broken backlink(s), "
            f"{report['duplicate_footnote_id_count']} duplicate footnote ID(s), "
            f"{report['invalid_footnote_id_count']} invalid footnote ID(s). "
            f"Samples: forward={report['forward_broken_samples'][:3]}, "
            f"back={report['backref_broken_samples'][:3]}, "
            f"duplicates={report['duplicate_footnote_id_samples'][:3]}, "
            f"invalid={report['invalid_footnote_id_samples'][:3]}"
        )
    return report
