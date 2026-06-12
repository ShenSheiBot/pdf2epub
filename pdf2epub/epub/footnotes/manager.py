"""
Main FootnoteManager class that coordinates all footnote processing.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from loguru import logger

from .models import FootnoteStyle, FootnoteDefinition, NotesSection
from .scanner import FootnoteScanner
from .mapper import FootnoteMapper
from .llm_matcher import LLMSectionMatcher
from ...chapter_identity import ChapterIdentity, strip_part_suffix


class FootnoteManager:
    """
    Manages footnote processing for EPUB generation.

    Automatically detects whether footnotes are organized locally (per-chapter)
    or globally (centralized in specific chapters) and handles them appropriately.
    """

    def __init__(self, markdown_dir: Path, force_global: bool = False, auto_global: bool = False, config=None, epub_structure=None):
        """
        Initialize the footnote manager.

        Args:
            markdown_dir: Directory containing markdown files
            force_global: If True, force global footnote style via CLI flag
            auto_global: If True, auto-detected global mode (e.g., notes chapter found)
            config: Configuration object for LLM calls (optional)
            epub_structure: Refined EPUB structure for local footnote scopes
        """
        self.markdown_dir = Path(markdown_dir)
        self.force_global = force_global
        self.auto_global = auto_global
        self.config = config
        self.epub_structure = epub_structure
        self._chapter_files: List[Path] = []

        # Initialize components
        self.scanner = FootnoteScanner()
        self.mapper = FootnoteMapper()
        self.llm_matcher: Optional[LLMSectionMatcher] = None

        # Enable global mode if either forced or auto-detected
        use_global = force_global or auto_global
        self.style = FootnoteStyle.GLOBAL if use_global else FootnoteStyle.LOCAL

        # Primary definition chapters (for force_global)
        self.primary_definition_chapters: Set[str] = set()

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

        self._chapter_files = self._discover_chapter_files()
        if not self._chapter_files:
            logger.warning("No chapter files found")
            return

        # Scan files for footnotes
        self.scanner.scan_files(self._chapter_files)
        self._drop_replaced_heading_references()

        # Determine style
        self.style = self.scanner.determine_style(self.force_global, self.auto_global)

        self._build_style_mappings(run_llm=True)

        # Log the analysis results
        self._log_analysis_results()

    def _discover_chapter_files(self) -> List[Path]:
        """Return chapter markdown files, excluding unsplit originals with parts."""
        chapter_files = sorted(self.markdown_dir.glob("chapter_*.md"))
        filtered_files = []
        for md_file in chapter_files:
            if '.part' not in md_file.name:
                base_name = md_file.stem
                has_parts = any(self.markdown_dir.glob(f"{base_name}.part*.md"))
                if has_parts:
                    logger.debug(f"Skipping {md_file.name} because split parts exist")
                    continue
            filtered_files.append(md_file)
        return filtered_files

    def _build_style_mappings(self, run_llm: bool = False) -> None:
        """Build mapper state from the current scanner data and style."""
        self.mapper.build_chapter_groups(self._chapter_files)

        if self.style == FootnoteStyle.LOCAL:
            self.primary_definition_chapters = set()
            if self.epub_structure:
                self.mapper.augment_chapter_groups_from_structure(self.epub_structure)
            self.mapper.build_local_occurrence_mappings(
                self.scanner.references,
                self.scanner.definitions,
                self.scanner.chapter_definitions,
                self.scanner.chapter_references
            )
        elif self.style == FootnoteStyle.GLOBAL:
            self.primary_definition_chapters = set()
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
            if run_llm and self.config and (self.force_global or self.auto_global):
                self.llm_matcher = LLMSectionMatcher(self.markdown_dir, self.config)
                if self.llm_matcher.load_toc_tree():
                    if self.llm_matcher.match_sections(self.primary_definition_chapters):
                        self.mapper.build_section_occurrence_mapping(self.llm_matcher.notes_sections)
                        logger.info("Using LLM-based section matching for footnotes")
                    else:
                        logger.info("LLM section matching failed, using occurrence-based mapping")
                else:
                    logger.info("Could not load TOC, using occurrence-based mapping")

    def _drop_replaced_heading_references(self) -> bool:
        """
        Remove references from raw first headings that build_epub will replace.

        build_epub replaces the first heading in the first part of each unit with
        the TOC title before markdown conversion. A footnote marker present only
        in that raw heading therefore never creates a fnref anchor in the final
        HTML, and must not participate in occurrence mapping or backrefs.
        """
        if not self.epub_structure:
            return False

        suppressed_positions = set()

        def walk(entries):
            for entry in entries:
                part_files = entry.get('part_files') or []
                if not part_files and entry.get('file_path'):
                    part_files = [entry['file_path']]

                if part_files:
                    first_part = Path(part_files[0])
                    try:
                        lines = first_part.read_text(encoding='utf-8').splitlines()
                    except OSError:
                        lines = []

                    for line_idx, line in enumerate(lines[:3], 1):
                        if line.strip().startswith('#'):
                            raw_keys = set(re.findall(r'\[\^(\w+)\](?!:)', line))
                            title_keys = set(re.findall(r'\[\^(\w+)\](?!:)', entry.get('title', '')))
                            for key in raw_keys - title_keys:
                                suppressed_positions.add((first_part.stem, line_idx, key))
                            break

                walk(entry.get('children', []))

        walk(self.epub_structure)
        if not suppressed_positions:
            return False

        removed = 0
        for key in list(self.scanner.references.keys()):
            kept_refs = []
            for ref in self.scanner.references[key]:
                if (ref.chapter, ref.line_num, key) in suppressed_positions:
                    removed += 1
                    continue
                kept_refs.append(ref)

            if kept_refs:
                self.scanner.references[key] = kept_refs
            else:
                del self.scanner.references[key]

        if not removed:
            return False

        chapter_references = {}
        for key, refs in self.scanner.references.items():
            for ref in refs:
                chapter_references.setdefault(ref.chapter, set()).add(key)

        self.scanner.chapter_references = chapter_references
        self.scanner.reference_only_chapters = {
            chapter for chapter in chapter_references
            if not self.scanner.chapter_definitions.get(chapter)
        }
        logger.debug(f"Dropped {removed} footnote references from replaced first headings")
        return True

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
        self.epub_structure = epub_structure
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

        self.llm_matcher = None
        self.mapper.section_definition_by_occurrence = {}
        self.mapper.section_definition_occurrence_by_line = {}
        self.mapper.section_definition_occurrence_in_file = {}
        self._chapter_files = self._discover_chapter_files()
        self.scanner = FootnoteScanner()
        self.scanner.scan_files(self._chapter_files)
        self._drop_replaced_heading_references()
        self.style = self.scanner.determine_style(self.force_global, self.auto_global)
        run_llm = bool(
            self.config
            and self.style == FootnoteStyle.GLOBAL
            and (self.force_global or self.auto_global)
        )
        self._build_style_mappings(run_llm=run_llm)

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

    def get_local_group_id(self, source_chapter: str) -> str:
        """Return the logical local footnote scope for a markdown stem."""
        return self.mapper.get_local_group_for_part(source_chapter)

    def is_definition_chapter(self, source_chapter: str) -> bool:
        """Return whether definitions in this source should be rendered."""
        if self.style == FootnoteStyle.LOCAL:
            return True
        if self.primary_definition_chapters:
            return source_chapter in self.primary_definition_chapters
        return source_chapter in self.definition_chapters

    def get_definition_html(
        self,
        key: str,
        content: str,
        source_chapter: str,
        line_num: Optional[int] = None,
        occurrence_in_file: Optional[int] = None,
    ) -> str:
        """Render a footnote definition using the manager's resolved mapping."""
        if self.style == FootnoteStyle.LOCAL:
            return self._get_local_definition_html(
                key,
                content,
                source_chapter,
                line_num=line_num,
                occurrence_in_file=occurrence_in_file,
            )
        return self._get_global_definition_html(
            key,
            content,
            source_chapter,
            line_num=line_num,
            occurrence_in_file=occurrence_in_file,
        )

    def _get_local_definition_html(
        self,
        key: str,
        content: str,
        source_chapter: str,
        line_num: Optional[int] = None,
        occurrence_in_file: Optional[int] = None,
    ) -> str:
        base_chapter = self.get_local_group_id(source_chapter)
        local_mapping = self.mapper.local_occurrence_mapping.get(base_chapter)
        is_multi_part = bool(
            local_mapping and len(self.mapper.local_chapter_groups.get(base_chapter, [])) > 1
        )

        if is_multi_part:
            occurrence_num = self._local_definition_occurrence(
                local_mapping,
                key,
                source_chapter,
                line_num=line_num,
                occurrence_in_file=occurrence_in_file,
            )
            fn_id = f"fn:{key}:{occurrence_num}"
            backref_html = self._local_definition_backref_html(
                local_mapping,
                key,
                occurrence_num,
            )
        else:
            if occurrence_in_file and occurrence_in_file > 1:
                fn_id = f"fn:{key}:{occurrence_in_file}"
            else:
                fn_id = f"fn:{key}"
            backref_html = ""
            ref_occurrence = occurrence_in_file if occurrence_in_file is not None else 1
            matching_refs = [
                ref for ref in self.scanner.references.get(key, [])
                if ref.chapter == source_chapter
            ]
            matching_refs.sort(key=lambda ref: ref.line_num)
            if 1 <= ref_occurrence <= len(matching_refs):
                fnref_id = self._fnref_id(source_chapter, key, ref_occurrence)
                backref_link = f"{self.get_html_filename(source_chapter)}#{fnref_id}"
                backref_html = self._backref_html(backref_link, key)

        return self._definition_html(fn_id, key, content, backref_html)

    def _local_definition_occurrence(
        self,
        local_mapping: Dict,
        key: str,
        source_chapter: str,
        line_num: Optional[int] = None,
        occurrence_in_file: Optional[int] = None,
    ) -> int:
        if occurrence_in_file is not None:
            occurrence_num = local_mapping.get('definition_occurrence_in_file', {}).get(
                (key, source_chapter, occurrence_in_file)
            )
            if occurrence_num is not None:
                return occurrence_num

        if line_num is not None:
            occurrence_num = local_mapping.get('definition_positions', {}).get(
                (key, source_chapter, line_num)
            )
            if occurrence_num is not None:
                return occurrence_num

        for (mapped_key, occurrence_num), defn in local_mapping.get('definition_by_occurrence', {}).items():
            if mapped_key == key and defn.chapter == source_chapter:
                return occurrence_num

        logger.debug(f"Using default occurrence 1 for footnote {key} in {source_chapter}")
        return 1

    def _local_definition_backref_html(
        self,
        local_mapping: Dict,
        key: str,
        occurrence_num: int,
    ) -> str:
        reference_entries = sorted(
            local_mapping.get('reference_occurrence_in_file', {}).items(),
            key=lambda item: (self._chapter_sort_key(item[0][1]), item[0][2]),
        )
        for (ref_key, ref_chapter, ref_occurrence_in_file), ref_position in reference_entries:
            if ref_key == key and ref_position == occurrence_num:
                fnref_id = self._fnref_id(ref_chapter, key, ref_occurrence_in_file)
                backref_link = f"{self.get_html_filename(ref_chapter)}#{fnref_id}"
                return self._backref_html(backref_link, key)
        return ""

    def _get_global_definition_html(
        self,
        key: str,
        content: str,
        source_chapter: str,
        line_num: Optional[int] = None,
        occurrence_in_file: Optional[int] = None,
    ) -> str:
        section_occurrence = None
        if occurrence_in_file is not None:
            section_occurrence = self.mapper.section_definition_occurrence_in_file.get(
                (key, source_chapter, occurrence_in_file)
            )
        if section_occurrence is None and line_num is not None:
            section_occurrence = self.mapper.section_definition_occurrence_by_line.get(
                (key, source_chapter, line_num)
            )

        if section_occurrence is not None:
            section_idx, occurrence_num = section_occurrence
            fn_id = f"fn:{key}:{occurrence_num}"
            backref_html = self._section_definition_backref_html(section_idx, key, occurrence_num)
            return self._definition_html(fn_id, key, content, backref_html)

        occurrence_num = None
        if occurrence_in_file is not None:
            occurrence_num = self.mapper.definition_occurrence_in_file.get(
                (key, source_chapter, occurrence_in_file)
            )
        if occurrence_num is None and line_num is not None:
            for (mapped_key, mapped_occurrence), definition in self.mapper.definition_by_occurrence.items():
                if mapped_key == key and definition.chapter == source_chapter and definition.line_num == line_num:
                    occurrence_num = mapped_occurrence
                    break
        if occurrence_num is None:
            occurrence_num = 1

        return self._definition_html(f"fn:{key}:{occurrence_num}", key, content, "")

    def _section_definition_backref_html(self, section_idx: int, key: str, occurrence_num: int) -> str:
        if section_idx >= len(self.notes_sections):
            return ""

        ref_unit_id = self.notes_sections[section_idx].matched_unit_id
        if not ref_unit_id:
            return ""

        fnref_id = f"fnref-{ref_unit_id}-{key}-{occurrence_num}"
        ref_file = self._reference_chapter_for_base_occurrence(key, ref_unit_id, occurrence_num)
        if ref_file:
            backref_link = f"{self.get_html_filename(ref_file)}#{fnref_id}"
        elif ref_unit_id in self.mapper.local_chapter_groups and self.mapper.local_chapter_groups[ref_unit_id]:
            first_part = self.mapper.local_chapter_groups[ref_unit_id][0]
            backref_link = f"{self.get_html_filename(first_part)}#{fnref_id}"
        else:
            backref_link = f"{self.get_html_filename(ref_unit_id)}#{fnref_id}"
        return self._backref_html(backref_link, key)

    def _definition_html(self, fn_id: str, key: str, content: str, backref_html: str) -> str:
        return (
            f'<div class="footnote-def" id="{fn_id}">\n'
            f'<p><strong>[{key}]:</strong> {content}{backref_html}</p>\n'
            f'</div>'
        )

    def _backref_html(self, backref_link: str, key: str) -> str:
        return (
            f' <a class="footnote-backref" href="{backref_link}" '
            f'title="Jump back to footnote {key} in the text">↩</a>'
        )

    def _fnref_id(
        self,
        source_chapter: str,
        key: str,
        occurrence_marker: Optional[int] = None,
    ) -> str:
        """Return a stable reference anchor id, suffixing repeated refs."""
        if occurrence_marker and occurrence_marker > 1:
            return f"fnref-{source_chapter}-{key}-{occurrence_marker}"
        return f"fnref-{source_chapter}-{key}"

    def _base_chapter_for_reference(self, source_chapter: str) -> str:
        return strip_part_suffix(source_chapter)

    def _reference_occurrence_for_base(
        self,
        key: str,
        base_chapter: str,
        source_chapter: str,
        occurrence_in_file: Optional[int] = None,
        line_num: Optional[int] = None,
    ) -> Optional[int]:
        refs = [
            ref for ref in self.scanner.references.get(key, [])
            if self._base_chapter_for_reference(ref.chapter) == base_chapter
        ]
        refs.sort(key=lambda ref: (self._chapter_sort_key(ref.chapter), ref.line_num))

        source_seen = 0
        for occurrence, ref in enumerate(refs, 1):
            if ref.chapter != source_chapter:
                continue
            source_seen += 1
            if occurrence_in_file is not None and source_seen == occurrence_in_file:
                return occurrence
            if occurrence_in_file is None and line_num is not None and ref.line_num == line_num:
                return occurrence
        return None

    def _reference_chapter_for_base_occurrence(
        self,
        key: str,
        base_chapter: str,
        occurrence_num: int,
    ) -> Optional[str]:
        refs = [
            ref for ref in self.scanner.references.get(key, [])
            if self._base_chapter_for_reference(ref.chapter) == base_chapter
        ]
        refs.sort(key=lambda ref: (self._chapter_sort_key(ref.chapter), ref.line_num))
        if 1 <= occurrence_num <= len(refs):
            return refs[occurrence_num - 1].chapter
        return None

    def get_footnote_html(
        self,
        key: str,
        source_chapter: str,
        line_num: Optional[int] = None,
        occurrence_in_file: Optional[int] = None,
    ) -> Optional[str]:
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

        def unlinked_reference() -> str:
            logger.warning(f"Footnote '{original_key}' referenced in {source_chapter} but not defined")
            fnref_id = self._fnref_id(source_chapter, original_key, occurrence_in_file)
            return f'<sup id="{fnref_id}">[{original_key}]</sup>'

        if self.style == FootnoteStyle.LOCAL:
            base_chapter = self.get_local_group_id(source_chapter)
            if ChapterIdentity.parse(base_chapter):

                # Check if this is a multi-part chapter with local mappings
                if base_chapter in self.mapper.local_occurrence_mapping and len(self.mapper.local_chapter_groups.get(base_chapter, [])) > 1:
                    # Multi-part chapter - use occurrence-based mapping
                    chapter_mapping = self.mapper.local_occurrence_mapping[base_chapter]
                    occurrence_num = None
                    if occurrence_in_file is not None and 'reference_occurrence_in_file' in chapter_mapping:
                        occurrence_num = chapter_mapping['reference_occurrence_in_file'].get(
                            (key, source_chapter, occurrence_in_file)
                        )
                    if occurrence_num is None and occurrence_in_file is None and line_num is not None and 'reference_positions' in chapter_mapping:
                        occurrence_num = occurrence_num or chapter_mapping['reference_positions'].get((key, source_chapter, line_num))
                    if occurrence_num is None and occurrence_in_file is None:
                        occurrence_num = chapter_mapping['reference_occurrence_count'].get((key, source_chapter))

                    if occurrence_num and (key, occurrence_num) in chapter_mapping['definition_by_occurrence']:
                        definition = chapter_mapping['definition_by_occurrence'][(key, occurrence_num)]
                        target_chapter = definition.chapter
                        fnref_marker = occurrence_in_file if occurrence_in_file is not None else occurrence_num

                        if target_chapter == source_chapter:
                            # Same file, use local anchor with unique ID
                            fnref_id = self._fnref_id(source_chapter, key, fnref_marker)
                            source_html = self.get_html_filename(source_chapter)
                            return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{source_html}#fn:{key}:{occurrence_num}">[{key}]</a></sup>'
                        else:
                            # Cross-part reference within the same chapter
                            fn_id = f"fn:{key}:{occurrence_num}"
                            fnref_id = self._fnref_id(source_chapter, key, fnref_marker)
                            html_target = self.get_html_filename(target_chapter)
                            return (
                                f'<sup id="{fnref_id}">'
                                f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{key}]</a>'
                                f'</sup>'
                            )
                    if occurrence_num:
                        return unlinked_reference()

            # Single file chapter or no multi-part mapping
            definitions_in_source = [
                defn for defn in self.scanner.definitions.get(key, [])
                if defn.chapter == source_chapter
            ]
            definitions_in_source.sort(key=lambda defn: defn.line_num)
            target_occurrence = occurrence_in_file if occurrence_in_file is not None else 1
            if not (1 <= target_occurrence <= len(definitions_in_source)):
                return unlinked_reference()
            fnref_id = self._fnref_id(source_chapter, key, occurrence_in_file)
            fn_id = f"fn:{key}" if target_occurrence == 1 else f"fn:{key}:{target_occurrence}"
            source_html = self.get_html_filename(source_chapter)
            return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{source_html}#{fn_id}">[{key}]</a></sup>'

        # Global style: try section-based mapping first, then fall back to occurrence mapping

        # Try section-based mapping if available
        if self.notes_sections and self.chapter_to_section:
            base_chapter = self._base_chapter_for_reference(source_chapter)

            if base_chapter in self.chapter_to_section:
                section = self.chapter_to_section[base_chapter]
                section_idx = self.notes_sections.index(section)
                occurrence = self._reference_occurrence_for_base(
                    key,
                    base_chapter,
                    source_chapter,
                    occurrence_in_file=occurrence_in_file,
                    line_num=line_num,
                )

                lookup_key = (section_idx, key, occurrence)
                if occurrence and lookup_key in self.mapper.section_definition_by_occurrence:
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
                elif occurrence:
                    return unlinked_reference()
                else:
                    # Section mapping exists but definition not found - fall through to global mapping
                    pass

        # Fall back to global occurrence-based mapping
        if key in self.scanner.definitions:
            # Use occurrence-based mapping if available
            occurrence_num = None
            if occurrence_in_file is not None:
                occurrence_num = self.mapper.reference_occurrence_in_file.get(
                    (key, source_chapter, occurrence_in_file)
                )
            if occurrence_num is None:
                refs_in_source = [
                    ref for ref in self.scanner.references.get(key, [])
                    if ref.chapter == source_chapter
                ]
                if occurrence_in_file is None and len(refs_in_source) == 1:
                    occurrence_num = self.mapper.reference_occurrence_count.get((key, source_chapter))
            if occurrence_num and (key, occurrence_num) in self.mapper.definition_by_occurrence:
                definition = self.mapper.definition_by_occurrence[(key, occurrence_num)]
            else:
                return unlinked_reference()

            target_chapter = definition.chapter

            # Use original_key for display text, but key for linking
            display_key = original_key

            if target_chapter == source_chapter:
                # Same file reference
                fnref_id = self._fnref_id(source_chapter, original_key, occurrence_in_file)
                html_target = self.get_html_filename(source_chapter)
                occ_num = occurrence_num if occurrence_num else 1
                return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{html_target}#fn:{key}:{occ_num}">[{display_key}]</a></sup>'
            else:
                # Cross-file reference
                fn_id = f"fn:{key}:{occurrence_num}"

                fnref_id = self._fnref_id(source_chapter, original_key, occurrence_in_file)
                html_target = self.get_html_filename(target_chapter)

                return (
                    f'<sup id="{fnref_id}">'
                    f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{display_key}]</a>'
                    f'</sup>'
                )

        # Footnote not found
        return unlinked_reference()
