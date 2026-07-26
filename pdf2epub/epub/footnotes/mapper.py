"""
Footnote mapping functionality for building occurrence-based mappings.
"""

from typing import Dict, List, Set, Tuple
from loguru import logger

from .content_index import ContentAddressIndex
from .models import FootnoteDefinition, NotesSection


class FootnoteMapper:
    """
    Builds various mappings between footnote references and definitions.
    """

    def __init__(self, content_index: ContentAddressIndex):
        self.content_index = content_index

        # For occurrence-based mapping in GLOBAL mode
        self.reference_occurrence_count: Dict[Tuple[str, str], int] = {}  # (key, chapter) -> occurrence number
        self.reference_occurrence_in_file: Dict[Tuple[str, str, int], int] = {}  # (key, chapter, occurrence_in_file) -> occurrence number
        self.definition_by_occurrence: Dict[Tuple[str, int], FootnoteDefinition] = {}  # (key, occurrence_num) -> definition
        self.definition_occurrence_in_file: Dict[Tuple[str, str, int], int] = {}  # (key, chapter, occurrence_in_file) -> occurrence number

        # For LOCAL mode with multi-part chapters
        self.local_chapter_groups: Dict[str, List[str]] = {}  # base_chapter -> list of part files
        self.local_part_to_group: Dict[str, str] = {}  # part file -> logical local footnote group
        self.local_occurrence_mapping: Dict[str, Dict] = {}  # base_chapter -> occurrence mappings

        # For section-based mapping in GLOBAL mode (LLM-based)
        self.section_definition_by_occurrence: Dict[Tuple[str, str, int], FootnoteDefinition] = {}
        self.section_definition_occurrence_by_line: Dict[Tuple[str, str, int], Tuple[str, int]] = {}
        self.section_definition_occurrence_in_file: Dict[Tuple[str, str, int], Tuple[str, int]] = {}

    def build_chapter_groups(
        self,
        references: Dict[str, List],
        definitions: Dict[str, List[FootnoteDefinition]],
    ) -> None:
        """Build local scopes from the authoritative content address index."""
        (
            self.local_chapter_groups,
            self.local_part_to_group,
        ) = self.content_index.build_local_groups(references, definitions)
        logger.debug(f"Built {len(self.local_chapter_groups)} structural chapter groups")

    def get_local_group_for_part(self, part_name: str) -> str:
        """Return the logical local footnote group for a markdown stem."""
        return self.local_part_to_group.get(part_name, part_name)

    def build_occurrence_mapping(
        self,
        references: Dict[str, List],
        definitions: Dict[str, List[FootnoteDefinition]],
        primary_definition_chapters: Set[str],
        force_global: bool,
        auto_global: bool
    ) -> None:
        """
        Build a mapping from references to definitions based on occurrence order.

        Args:
            references: Dictionary of key -> list of FootnoteReference
            definitions: Dictionary of key -> list of FootnoteDefinition
            primary_definition_chapters: Set of primary definition chapter names
            force_global: If True, force global style
            auto_global: If True, auto-detected global mode
        """
        self.reference_occurrence_count = {}
        self.reference_occurrence_in_file = {}
        self.definition_by_occurrence = {}
        self.definition_occurrence_in_file = {}

        # Sort all references by chapter and line number
        all_refs = []
        for key, ref_list in references.items():
            for ref in ref_list:
                all_refs.append((key, ref.chapter, ref.line_num))
        all_refs.sort(key=lambda x: (self._chapter_sort_key(x[1]), x[2]))

        # Count occurrences of each key in references
        ref_counts = {}
        occurrence_in_file = {}
        for key, chapter, line_num in all_refs:
            count = ref_counts.get(key, 0) + 1
            ref_counts[key] = count
            # Store the occurrence number for this specific reference
            self.reference_occurrence_count[(key, chapter)] = count
            file_counter_key = (key, chapter)
            occurrence_in_file[file_counter_key] = occurrence_in_file.get(file_counter_key, 0) + 1
            self.reference_occurrence_in_file[(key, chapter, occurrence_in_file[file_counter_key])] = count

        # Sort all definitions by chapter and line number
        all_defs = []
        for key, def_list in definitions.items():
            for defn in def_list:
                # Filter to primary chapters if force_global
                if not (force_global or auto_global) or not primary_definition_chapters or defn.chapter in primary_definition_chapters:
                    all_defs.append((key, defn))
        all_defs.sort(key=lambda x: (self._chapter_sort_key(x[1].chapter), x[1].line_num))

        # Map definitions by occurrence number
        def_counts = {}
        definition_in_file = {}
        for key, defn in all_defs:
            count = def_counts.get(key, 0) + 1
            def_counts[key] = count
            self.definition_by_occurrence[(key, count)] = defn
            file_counter_key = (key, defn.chapter)
            definition_in_file[file_counter_key] = definition_in_file.get(file_counter_key, 0) + 1
            self.definition_occurrence_in_file[
                (key, defn.chapter, definition_in_file[file_counter_key])
            ] = count

        logger.debug(f"Built occurrence mapping: {len(ref_counts)} unique ref keys, {len(def_counts)} unique def keys")

    def build_local_occurrence_mappings(
        self,
        references: Dict[str, List],
        definitions: Dict[str, List[FootnoteDefinition]],
        chapter_definitions: Dict[str, Set[str]],
        chapter_references: Dict[str, Set[str]]
    ) -> None:
        """
        Build occurrence mappings for LOCAL mode with multi-part chapters.

        Args:
            references: Dictionary of key -> list of FootnoteReference
            definitions: Dictionary of key -> list of FootnoteDefinition
            chapter_definitions: Dictionary of chapter -> set of defined keys
            chapter_references: Dictionary of chapter -> set of referenced keys
        """
        self.local_occurrence_mapping = {}

        for base_chapter, part_files in self.local_chapter_groups.items():
            active_parts = [
                part_file for part_file in part_files
                if self.local_part_to_group.get(part_file, base_chapter) == base_chapter
            ]
            if len(active_parts) != len(part_files):
                continue

            if len(part_files) <= 1:
                # Single file chapter, no cross-part references needed
                continue

            # Build position-based mapping for this chapter group
            chapter_mapping = {
                'reference_positions': {},  # (key, part_file, line_num) -> position
                'reference_occurrence_in_file': {},  # (key, part_file, occurrence_in_file) -> position
                'definition_positions': {},  # (key, part_file, line_num) -> position
                'definition_occurrence_in_file': {},  # (key, part_file, occurrence_in_file) -> position
                # Kept for manager-internal compatibility while callers use manager APIs.
                'reference_occurrence_count': {},
                'definition_by_occurrence': {},
            }

            # Group references by key
            refs_by_key = {}
            for part_file in part_files:
                if part_file in chapter_references:
                    for key in chapter_references[part_file]:
                        if key in references:
                            if key not in refs_by_key:
                                refs_by_key[key] = []
                            for ref in references[key]:
                                if ref.chapter == part_file:
                                    refs_by_key[key].append((part_file, ref.line_num))

            # Sort references within each key and assign positions
            for key in refs_by_key:
                refs_by_key[key].sort(key=lambda x: (self._chapter_sort_key(x[0]), x[1]))
                occurrence_in_file = {}
                for position, (part_file, line_num) in enumerate(refs_by_key[key], 1):
                    chapter_mapping['reference_positions'][(key, part_file, line_num)] = position
                    file_counter_key = (key, part_file)
                    occurrence_in_file[file_counter_key] = occurrence_in_file.get(file_counter_key, 0) + 1
                    chapter_mapping['reference_occurrence_in_file'][
                        (key, part_file, occurrence_in_file[file_counter_key])
                    ] = position
                    chapter_mapping['reference_occurrence_count'][(key, part_file)] = position

            # Group definitions by key
            defs_by_key = {}
            for part_file in part_files:
                if part_file in chapter_definitions:
                    for key in chapter_definitions[part_file]:
                        if key in definitions:
                            if key not in defs_by_key:
                                defs_by_key[key] = []
                            for defn in definitions[key]:
                                if defn.chapter == part_file:
                                    defs_by_key[key].append(defn)

            # Sort definitions within each key and assign positions
            for key in defs_by_key:
                defs_by_key[key].sort(key=lambda x: (self._chapter_sort_key(x.chapter), x.line_num))
                occurrence_in_file = {}
                for position, defn in enumerate(defs_by_key[key], 1):
                    chapter_mapping['definition_positions'][(key, defn.chapter, defn.line_num)] = position
                    file_counter_key = (key, defn.chapter)
                    occurrence_in_file[file_counter_key] = occurrence_in_file.get(file_counter_key, 0) + 1
                    chapter_mapping['definition_occurrence_in_file'][
                        (key, defn.chapter, occurrence_in_file[file_counter_key])
                    ] = position
                    chapter_mapping['definition_by_occurrence'][(key, position)] = defn

            # Validate ref/def counts match
            unique_ref_keys = set(refs_by_key.keys())
            unique_def_keys = set(defs_by_key.keys())
            for key in unique_ref_keys | unique_def_keys:
                ref_count = len(refs_by_key.get(key, []))
                def_count = len(defs_by_key.get(key, []))
                if ref_count != def_count:
                    logger.warning(
                        f"Footnote count mismatch in {base_chapter} for [{key}]: "
                        f"{ref_count} references, {def_count} definitions"
                    )

            self.local_occurrence_mapping[base_chapter] = chapter_mapping
            total_refs = sum(len(refs) for refs in refs_by_key.values())
            total_defs = sum(len(defs) for defs in defs_by_key.values())
            logger.debug(f"Built position-based mapping for {base_chapter}: {total_refs} refs, {total_defs} defs")

    def build_section_occurrence_mapping(self, notes_sections: List[NotesSection]) -> None:
        """
        Build occurrence-based mapping within each section.

        Args:
            notes_sections: List of NotesSection objects
        """
        if not notes_sections:
            return

        self.section_definition_by_occurrence = {}
        self.section_definition_occurrence_by_line = {}
        self.section_definition_occurrence_in_file = {}
        definition_assignments = []

        # Build mapping for each semantic section. ``matched_unit_id`` is the
        # stable scope identity; list position is not.
        for section in notes_sections:
            scope_unit_id = section.matched_unit_id
            if not scope_unit_id:
                continue

            # Group definitions by key within this section
            defs_by_key = {}
            for defn in section.definitions:
                if defn.key not in defs_by_key:
                    defs_by_key[defn.key] = []
                defs_by_key[defn.key].append(defn)

            # Sort definitions by line number
            for key in defs_by_key:
                defs_by_key[key].sort(
                    key=lambda definition: (
                        self._chapter_sort_key(definition.chapter),
                        definition.line_num,
                    )
                )

            # Store in section_definition_by_occurrence
            for key, def_list in defs_by_key.items():
                for occurrence, defn in enumerate(def_list, 1):
                    self.section_definition_by_occurrence[(scope_unit_id, key, occurrence)] = defn
                    self.section_definition_occurrence_by_line[
                        (key, defn.chapter, defn.line_num)
                    ] = (scope_unit_id, occurrence)
                    definition_assignments.append(
                        (defn, scope_unit_id, occurrence)
                    )

        # Markdown conversion reports occurrence within the physical file. Build
        # that index in physical line order, independently of semantic-section
        # iteration order (a scope may resume after another scope).
        assignments_by_file: Dict[Tuple[str, str], List] = {}
        for defn, scope_unit_id, scope_occurrence in definition_assignments:
            assignments_by_file.setdefault((defn.key, defn.chapter), []).append(
                (defn, scope_unit_id, scope_occurrence)
            )
        for (key, chapter), assignments in assignments_by_file.items():
            assignments.sort(key=lambda item: item[0].line_num)
            for occurrence_in_file, (_, scope_unit_id, scope_occurrence) in enumerate(
                assignments,
                1,
            ):
                self.section_definition_occurrence_in_file[
                    (key, chapter, occurrence_in_file)
                ] = (scope_unit_id, scope_occurrence)

        logger.debug(f"Built section occurrence mapping for {len(notes_sections)} sections")

    def _chapter_sort_key(self, chapter_name: str) -> tuple:
        """
        Generate a sort key for chapter names to maintain proper order.

        Args:
            chapter_name: The chapter name to sort

        Returns:
            Tuple for sorting
        """
        return self.content_index.order_key(chapter_name)
