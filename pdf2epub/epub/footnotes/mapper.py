"""
Footnote mapping functionality for building occurrence-based mappings.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
from loguru import logger

from .models import FootnoteDefinition, NotesSection
from ...chapter_identity import ChapterIdentity, strip_part_suffix, part_path


class FootnoteMapper:
    """
    Builds various mappings between footnote references and definitions.
    """

    def __init__(self):
        # For occurrence-based mapping in GLOBAL mode
        self.reference_occurrence_count: Dict[Tuple[str, str], int] = {}  # (key, chapter) -> occurrence number
        self.definition_by_occurrence: Dict[Tuple[str, int], FootnoteDefinition] = {}  # (key, occurrence_num) -> definition

        # For LOCAL mode with multi-part chapters
        self.local_chapter_groups: Dict[str, List[str]] = {}  # base_chapter -> list of part files
        self.local_occurrence_mapping: Dict[str, Dict] = {}  # base_chapter -> occurrence mappings

        # For section-based mapping in GLOBAL mode (LLM-based)
        self.section_definition_by_occurrence: Dict[Tuple[int, str, int], FootnoteDefinition] = {}

    def build_chapter_groups(self, files: List[Path]) -> None:
        """
        Build groups of files that belong to the same chapter.

        Args:
            files: List of markdown file paths
        """
        for file_path in files:
            chapter_name = file_path.stem
            # Use ChapterIdentity to extract base chapter name. ChapterIdentity.parse
            # only handles a single .partN, so fall back to strip_part_suffix for
            # multiply-nested parts (e.g. chapter_7.4.part3.part1).
            identity = ChapterIdentity.parse(chapter_name)
            if identity:
                base_chapter = identity.base_name
            else:
                base_chapter = strip_part_suffix(chapter_name)
                if not ChapterIdentity.parse(base_chapter):
                    continue
            if base_chapter not in self.local_chapter_groups:
                self.local_chapter_groups[base_chapter] = []
            self.local_chapter_groups[base_chapter].append(chapter_name)

        # Sort the part files within each group
        for base_chapter in self.local_chapter_groups:
            self.local_chapter_groups[base_chapter].sort(key=self._chapter_sort_key)

        logger.debug(f"Built {len(self.local_chapter_groups)} chapter groups")

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
        # Sort all references by chapter and line number
        all_refs = []
        for key, ref_list in references.items():
            for ref in ref_list:
                all_refs.append((key, ref.chapter, ref.line_num))
        all_refs.sort(key=lambda x: (self._chapter_sort_key(x[1]), x[2]))

        # Count occurrences of each key in references
        ref_counts = {}
        for key, chapter, line_num in all_refs:
            count = ref_counts.get(key, 0) + 1
            ref_counts[key] = count
            # Store the occurrence number for this specific reference
            self.reference_occurrence_count[(key, chapter)] = count

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
        for key, defn in all_defs:
            count = def_counts.get(key, 0) + 1
            def_counts[key] = count
            self.definition_by_occurrence[(key, count)] = defn

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
        for base_chapter, part_files in self.local_chapter_groups.items():
            if len(part_files) <= 1:
                # Single file chapter, no cross-part references needed
                continue

            # Build position-based mapping for this chapter group
            chapter_mapping = {
                'reference_positions': {},  # (key, part_file, line_num) -> position
                'definition_positions': {},  # (key, part_file, line_num) -> position
                'position_to_definition': {},  # (key, position) -> definition
                'reference_to_position': {},  # (key, part_file) -> position (for first ref in file)
                # Keep legacy fields for compatibility
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
                for position, (part_file, line_num) in enumerate(refs_by_key[key], 1):
                    chapter_mapping['reference_positions'][(key, part_file, line_num)] = position
                    # Store first occurrence for each part file (for get_footnote_html)
                    if (key, part_file) not in chapter_mapping['reference_to_position']:
                        chapter_mapping['reference_to_position'][(key, part_file)] = position
                    # Legacy field
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
                for position, defn in enumerate(defs_by_key[key], 1):
                    chapter_mapping['definition_positions'][(key, defn.chapter, defn.line_num)] = position
                    chapter_mapping['position_to_definition'][(key, position)] = defn
                    # Legacy field
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

        # Build mapping for each section
        for section_idx, section in enumerate(notes_sections):
            if not section.matched_unit_id:
                continue

            # Group definitions by key within this section
            defs_by_key = {}
            for defn in section.definitions:
                if defn.key not in defs_by_key:
                    defs_by_key[defn.key] = []
                defs_by_key[defn.key].append(defn)

            # Sort definitions by line number
            for key in defs_by_key:
                defs_by_key[key].sort(key=lambda d: d.line_num)

            # Store in section_definition_by_occurrence
            for key, def_list in defs_by_key.items():
                for occurrence, defn in enumerate(def_list, 1):
                    self.section_definition_by_occurrence[(section_idx, key, occurrence)] = defn

        logger.debug(f"Built section occurrence mapping for {len(notes_sections)} sections")

    def _chapter_sort_key(self, chapter_name: str) -> tuple:
        """
        Generate a sort key for chapter names to maintain proper order.

        Args:
            chapter_name: The chapter name to sort

        Returns:
            Tuple for sorting
        """
        # Use ChapterIdentity for the base (category + hierarchical index), and the
        # full part chain for ordering. The part component is always a tuple so that
        # single parts (.part1) and nested parts (.part3.part1) sort consistently.
        base = strip_part_suffix(chapter_name)
        base_id = ChapterIdentity.parse(base)
        parts = part_path(chapter_name)
        if base_id:
            category, idx_path, _ = base_id.sort_key
            return (category, list(idx_path), parts)
        return (999, [], parts)  # Put unrecognized files at the end
