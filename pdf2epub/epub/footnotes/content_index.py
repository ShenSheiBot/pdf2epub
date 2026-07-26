"""Structural addresses for chapter sources used by the footnote pipeline.

The EPUB builder has three different namespaces:

* markdown source stems (``chapter_4.3.1.part2``)
* generated HTML filenames (``chapter_4.3.1_part2.html``)
* semantic TOC units (``chapter_4.3``)

They are related by the refined EPUB structure, not by filename surgery.  This
module builds that relationship once and exposes the small set of structural
queries needed by footnote mapping.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ...chapter_identity import ChapterIdentity, strip_part_suffix


@dataclass(frozen=True)
class ContentAddress:
    """The physical and semantic address of one markdown source file."""

    source_stem: str
    html_filename: str
    unit_id: str
    ancestry: Tuple[str, ...]
    order: int


class ContentAddressIndex:
    """Index physical chapter files against the authoritative TOC hierarchy."""

    def __init__(
        self,
        addresses: Iterable[ContentAddress],
        unit_ancestry: Optional[Mapping[str, Tuple[str, ...]]] = None,
    ) -> None:
        ordered = sorted(addresses, key=lambda address: address.order)
        source_counts = Counter(address.source_stem for address in ordered)
        duplicate_sources = sorted(
            source for source, count in source_counts.items() if count > 1
        )
        html_counts = Counter(address.html_filename for address in ordered)
        duplicate_html = sorted(
            filename for filename, count in html_counts.items() if count > 1
        )
        if duplicate_sources or duplicate_html:
            raise ValueError(
                "EPUB structure has ambiguous content addresses: "
                f"duplicate sources={duplicate_sources}, "
                f"duplicate HTML files={duplicate_html}"
            )

        self._by_source: Dict[str, ContentAddress] = {
            address.source_stem: address for address in ordered
        }
        self._ordered_sources = tuple(address.source_stem for address in ordered)
        self._unit_ancestry = dict(unit_ancestry or {})
        self._sources_by_unit: Dict[str, Tuple[str, ...]] = {}

        unit_sources: Dict[str, List[str]] = {}
        for address in ordered:
            for unit_id in address.ancestry:
                unit_sources.setdefault(unit_id, []).append(address.source_stem)
        self._sources_by_unit = {
            unit_id: tuple(dict.fromkeys(stems))
            for unit_id, stems in unit_sources.items()
        }

    @classmethod
    def from_structure(cls, epub_structure: Sequence[Dict]) -> "ContentAddressIndex":
        """Build addresses from the same hierarchy used to write EPUB files."""
        addresses: List[ContentAddress] = []
        unit_ancestry: Dict[str, Tuple[str, ...]] = {}
        order = 0

        def walk(entries: Sequence[Dict], parents: Tuple[str, ...]) -> None:
            nonlocal order
            for entry in entries:
                unit_id = entry.get("unit_id")
                ancestry = parents
                if unit_id:
                    ancestry = (*parents, unit_id)
                    unit_ancestry[unit_id] = ancestry

                part_files = entry.get("part_files") or []
                if not part_files and entry.get("file_path"):
                    part_files = [entry["file_path"]]

                if unit_id and part_files:
                    for part_index, part_file in enumerate(part_files, 1):
                        source_stem = Path(part_file).stem
                        html_filename = (
                            f"{unit_id}_part{part_index}.html"
                            if len(part_files) > 1
                            else f"{unit_id}.html"
                        )
                        addresses.append(
                            ContentAddress(
                                source_stem=source_stem,
                                html_filename=html_filename,
                                unit_id=unit_id,
                                ancestry=ancestry,
                                order=order,
                            )
                        )
                        order += 1

                walk(entry.get("children", []), ancestry)

        walk(epub_structure, ())
        return cls(addresses, unit_ancestry)

    @classmethod
    def from_files(cls, files: Sequence[Path]) -> "ContentAddressIndex":
        """Create a compatibility index when no refined structure is available."""
        addresses: List[ContentAddress] = []
        for order, file_path in enumerate(sorted(files, key=lambda path: cls._natural_key(path.stem))):
            source_stem = file_path.stem
            unit_id = strip_part_suffix(source_stem)
            identity = ChapterIdentity.parse(source_stem)
            html_filename = (
                identity.html_name
                if identity
                else f"{source_stem.replace('.part', '_part')}.html"
            )
            addresses.append(
                ContentAddress(
                    source_stem=source_stem,
                    html_filename=html_filename,
                    unit_id=unit_id,
                    ancestry=(unit_id,),
                    order=order,
                )
            )
        return cls(addresses)

    @staticmethod
    def _natural_key(value: str) -> Tuple:
        return tuple(
            int(piece) if piece.isdigit() else piece
            for piece in re.split(r"(\d+)", value)
        )

    @property
    def sources(self) -> Tuple[str, ...]:
        return self._ordered_sources

    def html_for_source(self, source_stem: str) -> str:
        address = self._by_source.get(source_stem)
        if address:
            return address.html_filename
        identity = ChapterIdentity.parse(source_stem)
        if identity:
            return identity.html_name
        return f"{source_stem.replace('.part', '_part')}.html"

    def order_key(self, source_stem: str) -> Tuple[int, object]:
        address = self._by_source.get(source_stem)
        if address:
            return (0, address.order)
        return (1, self._natural_key(source_stem))

    def ancestry_for_source(self, source_stem: str) -> Tuple[str, ...]:
        address = self._by_source.get(source_stem)
        return address.ancestry if address else ()

    def nearest_scope(self, source_stem: str, scope_unit_ids: Set[str]) -> Optional[str]:
        """Return the closest ancestor explicitly present in ``scope_unit_ids``."""
        for unit_id in reversed(self.ancestry_for_source(source_stem)):
            if unit_id in scope_unit_ids:
                return unit_id
        return None

    def sources_in_scope(self, unit_id: str) -> Tuple[str, ...]:
        return self._sources_by_unit.get(unit_id, ())

    def build_local_groups(
        self,
        references: Mapping[str, Sequence],
        definitions: Mapping[str, Sequence],
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Infer local scopes from structural branches and unmatched counts.

        A TOC parent becomes a local footnote scope only when one immediate
        branch has unmatched references and another has matching unmatched
        definitions.  This captures notes collected at the end of a refined
        chapter without grouping otherwise self-contained sibling chapters.
        """
        refs_by_source: Dict[str, Counter] = {}
        defs_by_source: Dict[str, Counter] = {}
        for key, ref_list in references.items():
            for reference in ref_list:
                refs_by_source.setdefault(reference.chapter, Counter())[key] += 1
        for key, def_list in definitions.items():
            for definition in def_list:
                defs_by_source.setdefault(definition.chapter, Counter())[key] += 1

        groups: Dict[str, List[str]] = {}
        part_to_group: Dict[str, str] = {}

        # Every physical unit is a valid base scope.  Split files therefore work
        # without any inference.
        direct_by_unit: Dict[str, List[str]] = {}
        for source in self._ordered_sources:
            address = self._by_source[source]
            direct_by_unit.setdefault(address.unit_id, []).append(source)
        for unit_id, stems in direct_by_unit.items():
            groups[unit_id] = list(stems)
            for stem in stems:
                part_to_group[stem] = unit_id

        # Process deepest units first. A balanced child remains self-contained;
        # only a genuine deficit/surplus crossing a child boundary promotes the
        # scope to its parent.
        units_by_depth = sorted(
            self._unit_ancestry,
            key=lambda unit_id: len(self._unit_ancestry[unit_id]),
            reverse=True,
        )
        for unit_id in units_by_depth:
            ancestry = self._unit_ancestry[unit_id]
            descendant_stems = list(self.sources_in_scope(unit_id))
            if len(descendant_stems) < 2:
                continue

            branch_stems: Dict[str, List[str]] = {}
            for stem in descendant_stems:
                stem_ancestry = self.ancestry_for_source(stem)
                if len(stem_ancestry) <= len(ancestry):
                    branch = f"{unit_id}::__direct__"
                else:
                    branch = stem_ancestry[len(ancestry)]
                branch_stems.setdefault(branch, []).append(stem)
            if len(branch_stems) < 2:
                continue

            branch_balances: Dict[str, Counter] = {}
            for branch, stems in branch_stems.items():
                balance = Counter()
                for stem in stems:
                    balance.update(refs_by_source.get(stem, Counter()))
                    balance.subtract(defs_by_source.get(stem, Counter()))
                branch_balances[branch] = balance

            keys = {
                key
                for balance in branch_balances.values()
                for key in balance
            }
            crosses_boundary = any(
                any(balance[key] > 0 for balance in branch_balances.values())
                and any(balance[key] < 0 for balance in branch_balances.values())
                for key in keys
            )
            if not crosses_boundary:
                continue

            groups[unit_id] = descendant_stems
            for stem in descendant_stems:
                part_to_group[stem] = unit_id

        return groups, part_to_group
