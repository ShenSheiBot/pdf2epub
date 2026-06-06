"""
Main FootnoteManager class that coordinates all footnote processing.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

from .models import FootnoteStyle, FootnoteDefinition, NotesSection
from .scanner import FootnoteScanner
from .mapper import FootnoteMapper
from .llm_matcher import LLMSectionMatcher
from ...chapter_identity import ChapterIdentity, strip_part_suffix


def _parse_footnote_key(key: str) -> Tuple:
    """
    Parse a footnote key for sorting.

    Numeric keys are sorted numerically, others alphabetically.

    Args:
        key: The footnote key

    Returns:
        Tuple for sorting
    """
    try:
        return (0, int(key))  # Numeric keys first
    except ValueError:
        return (1, key)  # Then alphabetic


class FootnoteManager:
    """
    Manages footnote processing for EPUB generation.

    Automatically detects whether footnotes are organized locally (per-chapter)
    or globally (centralized in specific chapters) and handles them appropriately.
    """

    def __init__(self, markdown_dir: Path, force_global: bool = False, auto_global: bool = False, config=None):
        """
        Initialize the footnote manager.

        Args:
            markdown_dir: Directory containing markdown files
            force_global: If True, force global footnote style via CLI flag
            auto_global: If True, auto-detected global mode (e.g., notes chapter found)
            config: Configuration object for LLM calls (optional)
        """
        self.markdown_dir = Path(markdown_dir)
        self.force_global = force_global
        self.auto_global = auto_global
        self.config = config

        # Initialize components
        self.scanner = FootnoteScanner()
        self.mapper = FootnoteMapper()
        self.llm_matcher: Optional[LLMSectionMatcher] = None

        # Enable global mode if either forced or auto-detected
        use_global = force_global or auto_global
        self.style = FootnoteStyle.GLOBAL if use_global else FootnoteStyle.LOCAL

        # Primary definition chapters (for force_global)
        self.primary_definition_chapters: Set[str] = set()

        # Runtime counter for section-based occurrence tracking
        self._section_occurrence_counter: Dict[Tuple[str, str], int] = {}

        # Mapping from markdown stems to HTML filenames (for nested parts)
        # e.g., {"chapter_25.part1.part1": "chapter_25_part1.html"}
        self._html_filename_mapping: Dict[str, str] = {}

        # Analyze the footnote structure
        self._analyze_footnote_structure()

    # Expose scanner data for external access
    @property
    def definitions(self) -> Dict[str, List[FootnoteDefinition]]:
        return self.scanner.definitions

    @property
    def references(self) -> Dict[str, List]:
        return self.scanner.references

    @property
    def chapter_definitions(self) -> Dict[str, Set[str]]:
        return self.scanner.chapter_definitions

    @property
    def chapter_references(self) -> Dict[str, Set[str]]:
        return self.scanner.chapter_references

    @property
    def definition_chapters(self) -> Set[str]:
        return self.scanner.definition_chapters

    @property
    def reference_only_chapters(self) -> Set[str]:
        return self.scanner.reference_only_chapters

    # Expose mapper data for external access
    @property
    def reference_occurrence_count(self) -> Dict[Tuple[str, str], int]:
        return self.mapper.reference_occurrence_count

    @property
    def definition_by_occurrence(self) -> Dict[Tuple[str, int], FootnoteDefinition]:
        return self.mapper.definition_by_occurrence

    @property
    def local_chapter_groups(self) -> Dict[str, List[str]]:
        return self.mapper.local_chapter_groups

    @property
    def local_occurrence_mapping(self) -> Dict[str, Dict]:
        return self.mapper.local_occurrence_mapping

    @property
    def section_definition_by_occurrence(self) -> Dict[Tuple[int, str, int], FootnoteDefinition]:
        return self.mapper.section_definition_by_occurrence

    # Expose LLM matcher data for external access
    @property
    def notes_sections(self) -> List[NotesSection]:
        return self.llm_matcher.notes_sections if self.llm_matcher else []

    @property
    def chapter_to_section(self) -> Dict[str, NotesSection]:
        return self.llm_matcher.chapter_to_section if self.llm_matcher else {}

    def _analyze_footnote_structure(self) -> None:
        """
        Analyze all markdown files to determine footnote style.

        Sets self.style to LOCAL or GLOBAL based on the detected pattern.
        """
        logger.info("Analyzing footnote structure in markdown files...")

        # Get all chapter markdown files
        chapter_files = sorted(self.markdown_dir.glob("chapter_*.md"))

        if not chapter_files:
            logger.warning("No chapter files found")
            return

        # Filter out original files if split parts exist
        filtered_files = []
        for md_file in chapter_files:
            if '.part' not in md_file.name:
                base_name = md_file.stem
                has_parts = any(self.markdown_dir.glob(f"{base_name}.part*.md"))
                if has_parts:
                    logger.debug(f"Skipping {md_file.name} because split parts exist")
                    continue
            filtered_files.append(md_file)

        # Scan files for footnotes
        self.scanner.scan_files(self.markdown_dir, filtered_files)

        # Build chapter groups
        self.mapper.build_chapter_groups(filtered_files)

        # Determine style
        self.style = self.scanner.determine_style(self.force_global, self.auto_global)

        # Build appropriate mappings based on style
        if self.style == FootnoteStyle.LOCAL:
            self.mapper.build_local_occurrence_mappings(
                self.scanner.references,
                self.scanner.definitions,
                self.scanner.chapter_definitions,
                self.scanner.chapter_references
            )
        elif self.style == FootnoteStyle.GLOBAL:
            # If force_global or auto_global, identify primary definition chapters
            if self.force_global or self.auto_global:
                self.primary_definition_chapters = self.scanner.identify_primary_definition_chapters()

            # Build occurrence mapping for all global styles
            self.mapper.build_occurrence_mapping(
                self.scanner.references,
                self.scanner.definitions,
                self.primary_definition_chapters,
                self.force_global,
                self.auto_global
            )

            # Try LLM-based section matching for better accuracy
            if self.config and (self.force_global or self.auto_global):
                self.llm_matcher = LLMSectionMatcher(self.markdown_dir, self.config)
                if self.llm_matcher.load_toc_tree():
                    if self.llm_matcher.match_sections(self.primary_definition_chapters):
                        self.mapper.build_section_occurrence_mapping(self.llm_matcher.notes_sections)
                        logger.info("Using LLM-based section matching for footnotes")
                    else:
                        logger.info("LLM section matching failed, using occurrence-based mapping")
                else:
                    logger.info("Could not load TOC, using occurrence-based mapping")

        # Log the analysis results
        self._log_analysis_results()

    def _log_analysis_results(self) -> None:
        """Log the results of the footnote analysis."""
        if self.force_global or self.auto_global:
            mode_type = "FORCED" if self.force_global else "AUTO"
            logger.info(f"Footnote style: {mode_type} GLOBAL")
            logger.info(f"Primary definition chapters: {sorted(self.primary_definition_chapters)}")
            total_defs = sum(len(self.scanner.chapter_definitions.get(ch, set())) for ch in self.primary_definition_chapters)
            logger.info(f"Total definitions in primary chapters: {total_defs}")
        else:
            logger.info(f"Footnote style detected: {self.style.value.upper()}")

        if self.style == FootnoteStyle.GLOBAL:
            logger.info(f"Found {len(self.scanner.definition_chapters)} chapters with definitions")
            logger.info(f"Found {len(self.scanner.reference_only_chapters)} chapters with references only")

            # Log which chapters contain definitions
            for chapter in sorted(self.scanner.definition_chapters):
                count = len(self.scanner.chapter_definitions.get(chapter, set()))
                is_primary = " (PRIMARY)" if (self.force_global or self.auto_global) and chapter in self.primary_definition_chapters else ""
                logger.debug(f"  {chapter}: {count} definitions{is_primary}")

            # Log chapters with references only
            for chapter in sorted(self.scanner.reference_only_chapters):
                count = len(self.scanner.chapter_references.get(chapter, set()))
                logger.debug(f"  {chapter}: {count} references (no definitions)")

            # If force_global, log which definitions will be used
            if (self.force_global or self.auto_global) and self.scanner.definitions:
                logger.info("Footnote consolidation summary:")
                definitions_used = {}
                for key in sorted(self.scanner.definitions.keys()):
                    # Find which definition will be used
                    for def_obj in reversed(self.scanner.definitions[key]):
                        if def_obj.chapter in self.primary_definition_chapters:
                            definitions_used[key] = def_obj.chapter
                            break
                    else:
                        # Fallback
                        definitions_used[key] = self.scanner.definitions[key][-1].chapter

                # Group by chapter for summary
                by_chapter = {}
                for key, chapter in definitions_used.items():
                    if chapter not in by_chapter:
                        by_chapter[chapter] = []
                    by_chapter[chapter].append(key)

                for chapter in sorted(by_chapter.keys()):
                    logger.debug(f"  {chapter}: Will contain definitions for keys {sorted(by_chapter[chapter])[:10]}{'...' if len(by_chapter[chapter]) > 10 else ''}")

    def get_style(self) -> FootnoteStyle:
        """Get the detected footnote style."""
        return self.style

    def configure_from_structure(self, epub_structure: List[Dict]) -> None:
        """
        Configure HTML filename mapping from epub_structure.

        This builds the mapping from markdown stems to HTML filenames,
        needed for nested part files where ChapterIdentity can't parse
        the filename (e.g., chapter_25.part1.part1 -> chapter_25_part1.html).

        Args:
            epub_structure: The hierarchical structure from build_epub_structure()
        """
        mapping = {}

        def walk(entries):
            for entry in entries:
                if 'file_path' in entry and 'unit_id' in entry:
                    unit_id = entry['unit_id']
                    part_files = entry.get('part_files', [entry['file_path']])
                    for idx, part_file in enumerate(part_files):
                        stem = Path(part_file).stem
                        if len(part_files) > 1:
                            mapping[stem] = f"{unit_id}_part{idx + 1}.html"
                        else:
                            mapping[stem] = f"{unit_id}.html"
                if 'children' in entry:
                    walk(entry['children'])

        walk(epub_structure)
        self._html_filename_mapping = mapping
        logger.debug(f"Configured HTML filename mapping with {len(mapping)} entries")

    def get_html_filename(self, markdown_stem: str) -> str:
        """
        Get the HTML filename for a markdown file stem.

        Uses the explicit mapping first, then falls back to ChapterIdentity parsing,
        then to simple string replacement.

        Args:
            markdown_stem: The markdown file stem (e.g., "chapter_25.part1.part1")

        Returns:
            HTML filename (e.g., "chapter_25_part1.html")
        """
        # Try explicit mapping first (for nested parts)
        if markdown_stem in self._html_filename_mapping:
            return self._html_filename_mapping[markdown_stem]

        # Try ChapterIdentity parsing
        identity = ChapterIdentity.parse(markdown_stem)
        if identity:
            return identity.html_name

        # Fallback: simple replacement (may be wrong for nested parts)
        return f"{markdown_stem.replace('.part', '_part')}.html"

    def _chapter_sort_key(self, chapter_name: str) -> tuple:
        """
        Generate a sort key for chapter names to maintain proper order.

        Args:
            chapter_name: The chapter name to sort

        Returns:
            Tuple for sorting
        """
        return self.mapper._chapter_sort_key(chapter_name)

    def get_footnote_html(self, key: str, source_chapter: str) -> Optional[str]:
        """
        Get the HTML for a footnote reference.

        Args:
            key: The footnote key (e.g., "1", "note", or "197n67" for page-note format)
            source_chapter: The chapter containing the reference

        Returns:
            HTML string for the footnote reference, or None if not found
        """
        # Handle special page-note format like [^197n67]
        original_key = key
        page_note_match = re.match(r'^(\d+)n(\d+)$', key)
        if page_note_match:
            key = page_note_match.group(2)

        if self.style == FootnoteStyle.LOCAL:
            # Check if this chapter is part of a multi-part chapter. ChapterIdentity.parse
            # only handles a single .partN, so fall back to strip_part_suffix for
            # multiply-nested parts (e.g. chapter_7.4.part3.part1).
            base_chapter = strip_part_suffix(source_chapter)
            if ChapterIdentity.parse(base_chapter):

                # Check if this is a multi-part chapter with local mappings
                if base_chapter in self.mapper.local_occurrence_mapping and len(self.mapper.local_chapter_groups.get(base_chapter, [])) > 1:
                    # Multi-part chapter - use occurrence-based mapping
                    chapter_mapping = self.mapper.local_occurrence_mapping[base_chapter]
                    occurrence_num = chapter_mapping['reference_occurrence_count'].get((key, source_chapter))

                    if occurrence_num and (key, occurrence_num) in chapter_mapping['definition_by_occurrence']:
                        definition = chapter_mapping['definition_by_occurrence'][(key, occurrence_num)]
                        target_chapter = definition.chapter

                        if target_chapter == source_chapter:
                            # Same file, use local anchor with unique ID
                            fnref_id = f"fnref-{source_chapter}-{key}"
                            source_html = self.get_html_filename(source_chapter)
                            return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{source_html}#fn:{key}:{occurrence_num}">[{key}]</a></sup>'
                        else:
                            # Cross-part reference within the same chapter
                            fn_id = f"fn:{key}:{occurrence_num}"
                            fnref_id = f"fnref-{source_chapter}-{key}"
                            html_target = self.get_html_filename(target_chapter)
                            return (
                                f'<sup id="{fnref_id}">'
                                f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{key}]</a>'
                                f'</sup>'
                            )

            # Single file chapter or no multi-part mapping
            fnref_id = f"fnref-{source_chapter}-{key}"
            source_html = self.get_html_filename(source_chapter)
            return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{source_html}#fn:{key}:1">[{key}]</a></sup>'

        # Global style: try section-based mapping first, then fall back to occurrence mapping

        # Try section-based mapping if available
        if self.notes_sections and self.chapter_to_section:
            # Find the base chapter name (without .part suffix)
            base_chapter = source_chapter.split('.part')[0] if '.part' in source_chapter else source_chapter

            if base_chapter in self.chapter_to_section:
                section = self.chapter_to_section[base_chapter]
                section_idx = self.notes_sections.index(section)

                # Dynamic occurrence counting - increment counter for each call
                counter_key = (key, base_chapter)
                current_count = self._section_occurrence_counter.get(counter_key, 0) + 1
                self._section_occurrence_counter[counter_key] = current_count
                occurrence = current_count

                lookup_key = (section_idx, key, occurrence)
                if lookup_key in self.mapper.section_definition_by_occurrence:
                    definition = self.mapper.section_definition_by_occurrence[lookup_key]
                    target_chapter = definition.chapter

                    # Generate HTML link
                    # Use base_chapter (unit_id) for fnref_id to match the backref in definition
                    fnref_id = f"fnref-{base_chapter}-{key}-{occurrence}"
                    html_target = self.get_html_filename(target_chapter)

                    # Use occurrence number for the anchor
                    fn_id = f"fn:{key}:{occurrence}"

                    return (
                        f'<sup id="{fnref_id}">'
                        f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{original_key}]</a>'
                        f'</sup>'
                    )
                else:
                    # Section mapping exists but definition not found - fall through to global mapping
                    pass

        # Fall back to global occurrence-based mapping
        if key in self.scanner.definitions:
            # Use occurrence-based mapping if available
            occurrence_num = self.mapper.reference_occurrence_count.get((key, source_chapter))
            if occurrence_num and (key, occurrence_num) in self.mapper.definition_by_occurrence:
                definition = self.mapper.definition_by_occurrence[(key, occurrence_num)]
            else:
                # Fallback to old logic
                if self.force_global or self.auto_global:
                    # Find the last definition in primary definition chapters
                    definition = None
                    for def_obj in reversed(self.scanner.definitions[key]):
                        if def_obj.chapter in self.primary_definition_chapters:
                            definition = def_obj
                            break

                    # Fallback to last definition if none in primary chapters
                    if not definition:
                        definition = self.scanner.definitions[key][-1]
                else:
                    definition = self.scanner.definitions[key][0]  # Use first definition

            target_chapter = definition.chapter

            # Use original_key for display text, but key for linking
            display_key = original_key

            if target_chapter == source_chapter:
                # Same file reference
                fnref_id = f"fnref-{source_chapter}-{original_key}"
                html_target = self.get_html_filename(source_chapter)
                occ_num = occurrence_num if occurrence_num else 1
                return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{html_target}#fn:{key}:{occ_num}">[{display_key}]</a></sup>'
            else:
                # Cross-file reference
                source_identity = ChapterIdentity.parse(source_chapter)
                base_source_chapter = source_identity.base_name if source_identity else source_chapter

                if base_source_chapter in self.mapper.local_occurrence_mapping:
                    mapping = self.mapper.local_occurrence_mapping[base_source_chapter]

                    if 'reference_to_position' in mapping:
                        ref_position = mapping['reference_to_position'].get((key, source_chapter), 1)
                        fn_id = f"fn:{key}:{ref_position}"
                    else:
                        ref_occurrence = mapping['reference_occurrence_count'].get((key, source_chapter), 1)
                        fn_id = f"fn:{key}:{ref_occurrence}"
                else:
                    fn_id = f"fn:{key}:1"
                    logger.debug(f"No mapping found for {base_source_chapter}, using default ID")

                fnref_id = f"fnref-{source_chapter}-{original_key}"
                html_target = self.get_html_filename(target_chapter)

                return (
                    f'<sup id="{fnref_id}">'
                    f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{display_key}]</a>'
                    f'</sup>'
                )

        # Footnote not found
        logger.warning(f"Footnote '{original_key}' referenced in {source_chapter} but not defined")
        fnref_id = f"fnref-{source_chapter}-{original_key}"
        return f'<sup id="{fnref_id}">[{original_key}]</sup>'

    def get_definition_content(self, key: str) -> Optional[str]:
        """
        Get the content of a footnote definition.

        Args:
            key: The footnote key

        Returns:
            The footnote content, or None if not found
        """
        if key in self.scanner.definitions and self.scanner.definitions[key]:
            # If force_global or auto_global, use definition from primary definition chapters
            if self.force_global or self.auto_global:
                # Find the last definition in primary definition chapters
                for def_obj in reversed(self.scanner.definitions[key]):
                    if def_obj.chapter in self.primary_definition_chapters:
                        return def_obj.content

                # Fallback to last definition if none in primary chapters
                return self.scanner.definitions[key][-1].content
            else:
                return self.scanner.definitions[key][0].content
        return None

    def should_include_definition(self, key: str, chapter: str) -> bool:
        """
        Determine if a footnote definition should be included in a chapter's HTML.

        Args:
            key: The footnote key
            chapter: The chapter being processed

        Returns:
            True if the definition should be included, False otherwise
        """
        if self.style == FootnoteStyle.LOCAL:
            # Always include definitions in local style
            return True

        # Global style: only include if this is where it's defined
        if key in self.scanner.definitions:
            if self.force_global or self.auto_global:
                # Only include definitions in primary definition chapters
                if chapter not in self.primary_definition_chapters:
                    return False

                # Include ALL definitions in primary definition chapters
                for def_obj in self.scanner.definitions[key]:
                    if def_obj.chapter == chapter:
                        return True

                return False
            else:
                # Normal global style: include any definition in this chapter
                for definition in self.scanner.definitions[key]:
                    if definition.chapter == chapter:
                        return True
        return False

    def get_chapter_definitions(self, chapter: str) -> List[Tuple[str, str]]:
        """
        Get all footnote definitions for a chapter.

        Args:
            chapter: The chapter name (without .md extension)

        Returns:
            List of (key, content) tuples for footnotes defined in this chapter
        """
        result = []
        for key, definitions in self.scanner.definitions.items():
            for definition in definitions:
                if definition.chapter == chapter:
                    result.append((key, definition.content))
        return sorted(result, key=lambda x: _parse_footnote_key(x[0]))
