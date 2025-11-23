"""
Footnote management system for EPUB generation.

This module handles both local (per-chapter) and global (cross-chapter) footnote styles.
It automatically detects which style is appropriate based on the book's structure.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..chapter_identity import ChapterIdentity


class FootnoteStyle(Enum):
    """Footnote organization style."""
    LOCAL = "local"    # Each chapter has its own footnotes (default)
    GLOBAL = "global"  # Footnotes are centralized in specific chapters


@dataclass
class FootnoteDefinition:
    """Represents a footnote definition."""
    key: str           # The footnote key (e.g., "1", "note")
    content: str       # The footnote text
    chapter: str       # The chapter file where defined
    line_num: int      # Line number in the file


@dataclass
class FootnoteReference:
    """Represents a footnote reference."""
    key: str           # The footnote key
    chapter: str       # The chapter file where referenced
    line_num: int      # Line number in the file


@dataclass
class NotesSection:
    """Represents a section in the Notes chapter."""
    header_text: str                      # Original header text (e.g., "CHAPTER ONE", "前言")
    start_line: int                       # Start line number in the file
    end_line: int                         # End line number (exclusive)
    source_file: str                      # Which part file contains this section
    definitions: List[FootnoteDefinition] # Definitions in this section
    matched_unit_id: Optional[str] = None # Matched chapter unit_id (e.g., "chapter_5")


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
        # Enable global mode if either forced or auto-detected
        use_global = force_global or auto_global
        self.style = FootnoteStyle.GLOBAL if use_global else FootnoteStyle.LOCAL  # Default or forced style
        self.definitions: Dict[str, List[FootnoteDefinition]] = {}  # key -> list of definitions
        self.references: Dict[str, List[FootnoteReference]] = {}    # key -> list of references
        self.chapter_definitions: Dict[str, Set[str]] = {}  # chapter -> set of defined keys
        self.chapter_references: Dict[str, Set[str]] = {}   # chapter -> set of referenced keys
        self.definition_chapters: Set[str] = set()  # Chapters that contain definitions
        self.reference_only_chapters: Set[str] = set()  # Chapters with refs but no defs
        self.primary_definition_chapters: Set[str] = set()  # Chapters with the most definitions (for force_global)

        # For occurrence-based mapping in GLOBAL mode
        self.reference_occurrence_count: Dict[Tuple[str, str], int] = {}  # (key, chapter) -> occurrence number
        self.definition_by_occurrence: Dict[Tuple[str, int], FootnoteDefinition] = {}  # (key, occurrence_num) -> definition

        # For LOCAL mode with multi-part chapters
        self.local_chapter_groups: Dict[str, List[str]] = {}  # base_chapter -> list of part files
        self.local_occurrence_mapping: Dict[str, Dict] = {}  # base_chapter -> occurrence mappings

        # For section-based mapping in GLOBAL mode (LLM-based)
        self.notes_sections: List[NotesSection] = []  # Parsed sections from Notes chapter
        self.chapter_to_section: Dict[str, NotesSection] = {}  # chapter unit_id -> NotesSection
        self.toc_chapters: List[Dict] = []  # Loaded from toc_tree.json
        # (section_idx, key, occurrence) -> FootnoteDefinition
        self.section_definition_by_occurrence: Dict[Tuple[int, str, int], FootnoteDefinition] = {}
        # Runtime counter for section-based occurrence tracking
        # (key, chapter) -> current occurrence count
        self._section_occurrence_counter: Dict[Tuple[str, str], int] = {}

        # Analyze the footnote structure
        self._analyze_footnote_structure()
    
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
        # e.g., if chapter_5.part1.md exists, ignore chapter_5.md
        filtered_files = []
        for md_file in chapter_files:
            # Check if this is an original file with split parts
            if '.part' not in md_file.name:
                # Check if split parts exist
                base_name = md_file.stem
                has_parts = any(self.markdown_dir.glob(f"{base_name}.part*.md"))
                if has_parts:
                    logger.debug(f"Skipping {md_file.name} because split parts exist")
                    continue
            filtered_files.append(md_file)
        
        # Scan each file for footnotes
        for md_file in filtered_files:
            self._scan_file_for_footnotes(md_file)
        
        # Build chapter groups for multi-part handling
        self._build_chapter_groups(filtered_files)

        # Determine style based on the pattern
        self._determine_footnote_style()

        # Build appropriate mappings based on style
        if self.style == FootnoteStyle.LOCAL:
            self._build_local_occurrence_mappings()
        elif self.style == FootnoteStyle.GLOBAL:
            # If force_global or auto_global, identify primary definition chapters
            if self.force_global or self.auto_global:
                self._identify_primary_definition_chapters()
            # Build occurrence mapping for all global styles
            self._build_occurrence_mapping()

            # Try LLM-based section matching for better accuracy
            if self.config and (self.force_global or self.auto_global):
                if self._load_toc_tree():
                    if self._match_sections_with_llm():
                        self._build_section_occurrence_mapping()
                        logger.info("Using LLM-based section matching for footnotes")
                    else:
                        logger.info("LLM section matching failed, using occurrence-based mapping")
                else:
                    logger.info("Could not load TOC, using occurrence-based mapping")

        # Log the analysis results
        self._log_analysis_results()
    
    def _scan_file_for_footnotes(self, file_path: Path) -> None:
        """
        Scan a single file for footnote definitions and references.
        
        Args:
            file_path: Path to the markdown file
        """
        chapter_name = file_path.stem
        defined_keys = set()
        referenced_keys = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Check for footnote definitions [^key]:
                def_matches = re.findall(r'^\[\^(\w+)\]:', line)
                for key in def_matches:
                    defined_keys.add(key)
                    if key not in self.definitions:
                        self.definitions[key] = []
                    # Extract the content after the definition marker
                    content_match = re.match(r'^\[\^(\w+)\]:\s*(.*)', line)
                    content = content_match.group(2) if content_match else ""
                    self.definitions[key].append(
                        FootnoteDefinition(key, content, chapter_name, line_num)
                    )
                
                # Check for footnote references [^key] (not followed by colon)
                ref_matches = re.findall(r'\[\^(\w+)\](?!:)', line)
                for key in ref_matches:
                    # Handle special page-note format like [^197n67]
                    # Convert to just the note number (67) for matching with definitions
                    page_note_match = re.match(r'^(\d+)n(\d+)$', key)
                    if page_note_match:
                        key = page_note_match.group(2)

                    referenced_keys.add(key)
                    if key not in self.references:
                        self.references[key] = []
                    self.references[key].append(
                        FootnoteReference(key, chapter_name, line_num)
                    )
        
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            return
        
        # Store chapter-level information
        if defined_keys:
            self.chapter_definitions[chapter_name] = defined_keys
            self.definition_chapters.add(chapter_name)
        
        if referenced_keys:
            self.chapter_references[chapter_name] = referenced_keys
            # Check if this chapter has references but no definitions
            if not defined_keys:
                self.reference_only_chapters.add(chapter_name)
    
    def _determine_footnote_style(self) -> None:
        """
        Determine whether to use LOCAL or GLOBAL footnote style.
        
        GLOBAL style is used when:
        - force_global is True, OR
        - The following conditions are met:
          1. There are chapters with only definitions (like a Notes chapter)
          2. There are other chapters with references but no definitions
          3. The referenced keys match defined keys across chapters
        """
        # If forced or auto global, skip analysis
        if self.force_global or self.auto_global:
            self.style = FootnoteStyle.GLOBAL
            if self.force_global:
                logger.info("Using forced GLOBAL footnote style")
            else:
                logger.info("Using auto-detected GLOBAL footnote style (notes chapter found)")
            return
        
        # Check if we have the pattern for global footnotes
        has_definition_only_chapters = False
        has_reference_only_chapters = bool(self.reference_only_chapters)
        
        # Check for chapters that have definitions but no references
        for chapter, defined_keys in self.chapter_definitions.items():
            if chapter not in self.chapter_references or not self.chapter_references[chapter]:
                # This chapter has definitions but no references
                has_definition_only_chapters = True
                break
            # Also check if definitions far outnumber references (like a Notes chapter)
            ref_keys = self.chapter_references.get(chapter, set())
            if len(defined_keys) > 10 and len(defined_keys) > len(ref_keys) * 3:
                has_definition_only_chapters = True
                break
        
        # Determine if we should use global style
        if has_definition_only_chapters and has_reference_only_chapters:
            # Verify that referenced keys have definitions somewhere
            all_defined_keys = set()
            for keys in self.chapter_definitions.values():
                all_defined_keys.update(keys)
            
            unmatched_refs = set()
            for keys in self.chapter_references.values():
                unmatched_refs.update(keys - all_defined_keys)
            
            if len(unmatched_refs) / max(1, len(all_defined_keys)) < 0.1:  # Less than 10% unmatched
                self.style = FootnoteStyle.GLOBAL
            else:
                logger.warning(
                    f"Found {len(unmatched_refs)} unmatched footnote references. "
                    "Using LOCAL style for safety."
                )
        
        # Additional check: if any chapter has duplicate footnote keys with different content
        if self.style == FootnoteStyle.GLOBAL:
            if self._has_conflicting_definitions():
                logger.warning("Found conflicting footnote definitions. Using LOCAL style for safety.")
                self.style = FootnoteStyle.LOCAL
    
    def _build_occurrence_mapping(self) -> None:
        """
        Build a mapping from references to definitions based on occurrence order.
        Maps the Nth occurrence of key X in references to the Nth occurrence of key X in definitions.
        """
        # Sort all references by chapter and line number
        all_refs = []
        for key, ref_list in self.references.items():
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
        for key, def_list in self.definitions.items():
            for defn in def_list:
                # Filter to primary chapters if force_global
                if not (self.force_global or self.auto_global) or not self.primary_definition_chapters or defn.chapter in self.primary_definition_chapters:
                    all_defs.append((key, defn))
        all_defs.sort(key=lambda x: (self._chapter_sort_key(x[1].chapter), x[1].line_num))
        
        # Map definitions by occurrence number
        def_counts = {}
        for key, defn in all_defs:
            count = def_counts.get(key, 0) + 1
            def_counts[key] = count
            self.definition_by_occurrence[(key, count)] = defn
        
        logger.debug(f"Built occurrence mapping: {len(ref_counts)} unique ref keys, {len(def_counts)} unique def keys")
    
    def _build_chapter_groups(self, files: List[Path]) -> None:
        """
        Build groups of files that belong to the same chapter.
        For example, chapter_5.part1.md and chapter_5.part2.md belong to the same chapter group.
        """
        for file_path in files:
            chapter_name = file_path.stem
            # Use ChapterIdentity to extract base chapter name
            identity = ChapterIdentity.parse(chapter_name)
            if identity:
                base_chapter = identity.base_name
                if base_chapter not in self.local_chapter_groups:
                    self.local_chapter_groups[base_chapter] = []
                self.local_chapter_groups[base_chapter].append(chapter_name)

        # Sort the part files within each group
        for base_chapter in self.local_chapter_groups:
            self.local_chapter_groups[base_chapter].sort(key=self._chapter_sort_key)

        logger.debug(f"Built {len(self.local_chapter_groups)} chapter groups")

    def _build_local_occurrence_mappings(self) -> None:
        """
        Build occurrence mappings for LOCAL mode with multi-part chapters using position-based mapping.
        Maps each reference to its corresponding definition by position (1st ref -> 1st def, etc).
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
                if part_file in self.chapter_references:
                    for key in self.chapter_references[part_file]:
                        if key in self.references:
                            if key not in refs_by_key:
                                refs_by_key[key] = []
                            for ref in self.references[key]:
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
                if part_file in self.chapter_definitions:
                    for key in self.chapter_definitions[part_file]:
                        if key in self.definitions:
                            if key not in defs_by_key:
                                defs_by_key[key] = []
                            for defn in self.definitions[key]:
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

    def _chapter_sort_key(self, chapter_name: str) -> tuple:
        """
        Generate a sort key for chapter names to maintain proper order.
        Handles chapter_N and chapter_N.partM or chapter_N_partM formats.
        """
        # Use ChapterIdentity for consistent sorting
        identity = ChapterIdentity.parse(chapter_name)
        if identity:
            return identity.sort_key
        return (999, 0, 0)  # Put unrecognized files at the end
    
    def _identify_primary_definition_chapters(self) -> None:
        """
        Identify chapters with the most footnote definitions.
        These will be used as the primary definition chapters when force_global is True.
        """
        if not self.chapter_definitions:
            return
        
        # Count definitions per chapter (including part files)
        chapter_def_counts = {}
        for chapter, keys in self.chapter_definitions.items():
            # Group by base chapter name (e.g., chapter_7.part1 -> chapter_7)
            base_chapter = chapter.split('.part')[0] if '.part' in chapter else chapter
            if base_chapter not in chapter_def_counts:
                chapter_def_counts[base_chapter] = 0
            chapter_def_counts[base_chapter] += len(keys)
        
        # Find the maximum number of definitions
        if chapter_def_counts:
            max_defs = max(chapter_def_counts.values())
            
            # Identify chapters with the most definitions (could be multiple)
            # Use a threshold - chapters with at least 80% of max definitions
            threshold = max_defs * 0.8
            
            for chapter, count in chapter_def_counts.items():
                if count >= threshold:
                    # Add all parts of this chapter
                    for full_chapter in self.chapter_definitions.keys():
                        if full_chapter.startswith(chapter):
                            self.primary_definition_chapters.add(full_chapter)
            
            logger.info(f"Identified primary definition chapters: {sorted(self.primary_definition_chapters)}")
            logger.info(f"These chapters contain {sum(len(self.chapter_definitions.get(ch, set())) for ch in self.primary_definition_chapters)} footnote definitions")
    
    def _has_conflicting_definitions(self) -> bool:
        """
        Check if there are conflicting definitions for the same footnote key.
        
        Returns:
            True if conflicts found, False otherwise
        """
        for key, definitions in self.definitions.items():
            if len(definitions) > 1:
                # Check if definitions have different content
                contents = set(d.content.strip() for d in definitions)
                if len(contents) > 1:
                    # Check if all definitions are in definition-only chapters
                    # (chapters that have definitions but no references)
                    definition_chapters = set(d.chapter for d in definitions)
                    all_in_definition_only = all(
                        ch in self.chapter_definitions and 
                        (ch not in self.chapter_references or not self.chapter_references[ch])
                        for ch in definition_chapters
                    )
                    
                    if all_in_definition_only:
                        # All definitions are in definition-only chapters
                        # This is expected for global mode with multi-part definitions
                        logger.debug(f"Footnote key '{key}' has multiple definitions in definition-only chapters (valid for global mode)")
                        continue
                    
                    # Definitions in mixed chapters or reference chapters = conflict
                    logger.debug(f"Footnote key '{key}' has conflicting definitions")
                    return True
        return False
    
    def _log_analysis_results(self) -> None:
        """Log the results of the footnote analysis."""
        if self.force_global or self.auto_global:
            mode_type = "FORCED" if self.force_global else "AUTO"
            logger.info(f"Footnote style: {mode_type} GLOBAL")
            logger.info(f"Primary definition chapters: {sorted(self.primary_definition_chapters)}")
            total_defs = sum(len(self.chapter_definitions.get(ch, set())) for ch in self.primary_definition_chapters)
            logger.info(f"Total definitions in primary chapters: {total_defs}")
        else:
            logger.info(f"Footnote style detected: {self.style.value.upper()}")
        
        if self.style == FootnoteStyle.GLOBAL:
            logger.info(f"Found {len(self.definition_chapters)} chapters with definitions")
            logger.info(f"Found {len(self.reference_only_chapters)} chapters with references only")
            
            # Log which chapters contain definitions
            for chapter in sorted(self.definition_chapters):
                count = len(self.chapter_definitions.get(chapter, set()))
                is_primary = " (PRIMARY)" if (self.force_global or self.auto_global) and chapter in self.primary_definition_chapters else ""
                logger.debug(f"  {chapter}: {count} definitions{is_primary}")
            
            # Log chapters with references only
            for chapter in sorted(self.reference_only_chapters):
                count = len(self.chapter_references.get(chapter, set()))
                logger.debug(f"  {chapter}: {count} references (no definitions)")
            
            # If force_global, log which definitions will be used
            if (self.force_global or self.auto_global) and self.definitions:
                logger.info("Footnote consolidation summary:")
                definitions_used = {}
                for key in sorted(self.definitions.keys()):
                    # Find which definition will be used
                    for def_obj in reversed(self.definitions[key]):
                        if def_obj.chapter in self.primary_definition_chapters:
                            definitions_used[key] = def_obj.chapter
                            break
                    else:
                        # Fallback
                        definitions_used[key] = self.definitions[key][-1].chapter
                
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
        # Convert to just the note number (67) for matching with definitions
        original_key = key
        page_note_match = re.match(r'^(\d+)n(\d+)$', key)
        if page_note_match:
            key = page_note_match.group(2)
        if self.style == FootnoteStyle.LOCAL:
            # Check if this chapter is part of a multi-part chapter
            source_identity = ChapterIdentity.parse(source_chapter)
            if source_identity:
                base_chapter = source_identity.base_name

                # Check if this is a multi-part chapter with local mappings
                if base_chapter in self.local_occurrence_mapping and len(self.local_chapter_groups.get(base_chapter, [])) > 1:
                    # Multi-part chapter - use occurrence-based mapping
                    chapter_mapping = self.local_occurrence_mapping[base_chapter]
                    occurrence_num = chapter_mapping['reference_occurrence_count'].get((key, source_chapter))

                    if occurrence_num and (key, occurrence_num) in chapter_mapping['definition_by_occurrence']:
                        definition = chapter_mapping['definition_by_occurrence'][(key, occurrence_num)]
                        target_chapter = definition.chapter

                        if target_chapter == source_chapter:
                            # Same file, use local anchor with unique ID
                            fnref_id = f"fnref-{source_chapter}-{key}"
                            # Use file name even for same-file references in LOCAL style, with occurrence number
                            source_html = source_identity.html_name
                            return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{source_html}#fn:{key}:{occurrence_num}">[{key}]</a></sup>'
                        else:
                            # Cross-part reference within the same chapter
                            fn_id = f"fn:{key}:{occurrence_num}"
                            fnref_id = f"fnref-{source_chapter}-{key}"
                            # Use ChapterIdentity for HTML name conversion
                            target_identity = ChapterIdentity.parse(target_chapter)
                            html_target = target_identity.html_name if target_identity else f"{target_chapter}.html"
                            return (
                                f'<sup id="{fnref_id}">'
                                f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{key}]</a>'
                                f'</sup>'
                            )

            # Single file chapter or no multi-part mapping - still use file name for consistency
            fnref_id = f"fnref-{source_chapter}-{key}"
            # Use ChapterIdentity for HTML name, fallback if not recognized
            source_html = source_identity.html_name if source_identity else f"{source_chapter.replace('.part', '_part')}.html"
            # For single-file chapters, occurrence is always 1
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
                counter_key = (key, source_chapter)
                current_count = self._section_occurrence_counter.get(counter_key, 0) + 1
                self._section_occurrence_counter[counter_key] = current_count
                occurrence = current_count

                lookup_key = (section_idx, key, occurrence)
                if lookup_key in self.section_definition_by_occurrence:
                    definition = self.section_definition_by_occurrence[lookup_key]
                    target_chapter = definition.chapter

                    # Generate HTML link
                    fnref_id = f"fnref-{source_chapter}-{original_key}-{occurrence}"
                    target_identity = ChapterIdentity.parse(target_chapter)
                    html_target = target_identity.html_name if target_identity else f"{target_chapter.replace('.part', '_part')}.html"

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
        if key in self.definitions:
            # Use occurrence-based mapping if available
            occurrence_num = self.reference_occurrence_count.get((key, source_chapter))
            if occurrence_num and (key, occurrence_num) in self.definition_by_occurrence:
                definition = self.definition_by_occurrence[(key, occurrence_num)]
            else:
                # Fallback to old logic
                if self.force_global or self.auto_global:
                    # Find the last definition in primary definition chapters
                    definition = None
                    for def_obj in reversed(self.definitions[key]):
                        if def_obj.chapter in self.primary_definition_chapters:
                            definition = def_obj
                            break

                    # Fallback to last definition if none in primary chapters
                    if not definition:
                        definition = self.definitions[key][-1]
                else:
                    definition = self.definitions[key][0]  # Use first definition

            target_chapter = definition.chapter

            # Use original_key for display text, but key for linking
            display_key = original_key

            if target_chapter == source_chapter:
                # Same file reference - still use file name for consistency
                fnref_id = f"fnref-{source_chapter}-{original_key}"
                # Use ChapterIdentity for HTML name
                source_identity = ChapterIdentity.parse(source_chapter)
                html_target = source_identity.html_name if source_identity else f"{source_chapter.replace('.part', '_part')}.html"
                # Use occurrence number if available, otherwise default to 1
                occ_num = occurrence_num if occurrence_num else 1
                return f'<sup id="{fnref_id}"><a class="footnote-ref" href="{html_target}#fn:{key}:{occ_num}">[{display_key}]</a></sup>'
            else:
                # Cross-file reference in LOCAL mode with multi-part chapters
                # Use position-based mapping to find the correct occurrence number
                source_identity = ChapterIdentity.parse(source_chapter)
                base_source_chapter = source_identity.base_name if source_identity else source_chapter

                if base_source_chapter in self.local_occurrence_mapping:
                    mapping = self.local_occurrence_mapping[base_source_chapter]

                    # Use new position-based mapping if available
                    if 'reference_to_position' in mapping:
                        # Get the position for this reference
                        ref_position = mapping['reference_to_position'].get((key, source_chapter), 1)
                        fn_id = f"fn:{key}:{ref_position}"
                    else:
                        # Fallback to legacy mapping
                        ref_occurrence = mapping['reference_occurrence_count'].get((key, source_chapter), 1)
                        fn_id = f"fn:{key}:{ref_occurrence}"
                else:
                    # Fallback for non-mapped chapters
                    fn_id = f"fn:{key}:1"
                    logger.debug(f"No mapping found for {base_source_chapter}, using default ID")

                fnref_id = f"fnref-{source_chapter}-{original_key}"

                # Use ChapterIdentity for HTML name conversion
                target_identity = ChapterIdentity.parse(target_chapter)
                html_target = target_identity.html_name if target_identity else f"{target_chapter.replace('.part', '_part')}.html"

                return (
                    f'<sup id="{fnref_id}">'
                    f'<a class="footnote-ref" href="{html_target}#{fn_id}">[{display_key}]</a>'
                    f'</sup>'
                )

        # Footnote not found - return a plain reference with unique ID
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
        if key in self.definitions and self.definitions[key]:
            # If force_global or auto_global, use definition from primary definition chapters
            if self.force_global or self.auto_global:
                # Find the last definition in primary definition chapters
                for def_obj in reversed(self.definitions[key]):
                    if def_obj.chapter in self.primary_definition_chapters:
                        return def_obj.content

                # Fallback to last definition if none in primary chapters
                return self.definitions[key][-1].content
            else:
                return self.definitions[key][0].content  # Use first definition
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
        if key in self.definitions:
            if self.force_global or self.auto_global:
                # Only include definitions in primary definition chapters
                if chapter not in self.primary_definition_chapters:
                    return False

                # Include ALL definitions in primary definition chapters
                # This is important for occurrence-based mapping where we have
                # multiple definitions with the same key (e.g., multiple [^1]s)
                for def_obj in self.definitions[key]:
                    if def_obj.chapter == chapter:
                        return True

                return False
            else:
                # Normal global style: include any definition in this chapter
                for definition in self.definitions[key]:
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
        for key, definitions in self.definitions.items():
            for definition in definitions:
                if definition.chapter == chapter:
                    result.append((key, definition.content))
        return sorted(result, key=lambda x: _parse_footnote_key(x[0]))

    def _load_toc_tree(self) -> bool:
        """
        Load toc_tree.json from the output directory.

        Returns:
            True if loaded successfully, False otherwise
        """
        import json

        # toc_tree.json is in the parent of markdown_dir (output_dir)
        toc_path = self.markdown_dir.parent / "toc_tree.json"
        if not toc_path.exists():
            logger.warning(f"toc_tree.json not found at {toc_path}")
            return False

        try:
            with open(toc_path, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)

            # Flatten the TOC tree to get all chapters with unit_ids
            def flatten_chapters(chapters, parent_path=None):
                result = []
                for i, ch in enumerate(chapters):
                    path = (parent_path or []) + [i + 1]
                    unit_id = f"chapter_{path[0]}" if len(path) == 1 else f"chapter_{'_'.join(map(str, path))}"
                    result.append({
                        "unit_id": unit_id,
                        "title": ch.get("title", ""),
                        "type": ch.get("type")
                    })
                    if "children" in ch and ch["children"]:
                        result.extend(flatten_chapters(ch["children"], path))
                return result

            self.toc_chapters = flatten_chapters(toc_data.get("chapters", []))
            logger.debug(f"Loaded {len(self.toc_chapters)} chapters from toc_tree.json")
            return True

        except Exception as e:
            logger.error(f"Error loading toc_tree.json: {e}")
            return False

    def _get_notes_structure_for_llm(self) -> str:
        """
        Get the Notes chapter structure with footnote definitions removed.

        Returns:
            String with only headers/titles (footnote content removed)
        """
        # Read all primary definition chapter files
        all_lines = []
        for chapter_name in sorted(self.primary_definition_chapters):
            file_path = self.markdown_dir / f"{chapter_name}.md"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Remove footnote definition lines
                    lines = content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        # Skip footnote definitions [^key]: content
                        if re.match(r'^\[\^\w+\]:', line):
                            continue
                        # Keep non-empty lines that might be headers
                        stripped = line.strip()
                        if stripped:
                            filtered_lines.append(stripped)
                    all_lines.extend(filtered_lines)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        return '\n'.join(all_lines)

    def _match_sections_with_llm(self) -> bool:
        """
        Use LLM to match Notes section headers to TOC chapters.

        Returns:
            True if matching succeeded, False otherwise
        """
        import json

        if not self.config:
            logger.warning("No config provided, cannot use LLM for section matching")
            return False

        if not self.toc_chapters:
            logger.warning("No TOC chapters loaded")
            return False

        # Get Notes structure
        notes_structure = self._get_notes_structure_for_llm()
        if not notes_structure.strip():
            logger.warning("Notes structure is empty")
            return False

        # Filter TOC to exclude notes chapter itself
        toc_entries = [
            {"unit_id": ch["unit_id"], "title": ch["title"]}
            for ch in self.toc_chapters
            if ch.get("type") != "notes"
        ]

        # Build prompt
        prompt = f"""分析以下 Notes 章节的结构（已删除脚注内容，只保留标题），并将每个标题匹配到对应的 TOC 章节。

Notes 章节结构：
```
{notes_structure}
```

TOC 章节列表：
```json
{json.dumps(toc_entries, ensure_ascii=False, indent=2)}
```

请返回一个 JSON 数组，每个元素包含：
- "header": Notes 中的标题文本（完全匹配原文）
- "unit_id": 对应的 TOC 章节 unit_id

注意：
1. Notes 中可能有 OCR 错误或格式问题，请基于语义理解进行匹配
2. 标题可能是任何格式（如 "## CHAPTER ONE"、"PREFACE"、"第一章" 等）
3. 只返回 JSON 数组，不要其他内容

示例返回格式：
```json
[
  {{"header": "PREFACE", "unit_id": "chapter_3"}},
  {{"header": "INTRODUCTION", "unit_id": "chapter_4"}},
  {{"header": "CHAPTER ONE", "unit_id": "chapter_5"}}
]
```
"""

        try:
            from ..utils.llm_client import LLMClient

            llm_client = LLMClient(self.config)

            # Get model config
            model_configs = self.config.get("translation.models", [
                {"provider": "gemini", "model": "gemini-2.5-flash", "max_retries": 2}
            ])

            response = llm_client.generate(
                prompt=prompt,
                model_configs=model_configs,
                operation_name="Match notes sections to chapters"
            )

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block
                lines = response.split('\n')
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response = '\n'.join(json_lines)

            matches = json.loads(response)
            logger.info(f"LLM returned {len(matches)} section matches")

            # Parse sections from LLM result
            return self._parse_sections_from_llm_result(matches)

        except Exception as e:
            logger.error(f"Error in LLM section matching: {e}")
            return False

    def _parse_sections_from_llm_result(self, matches: List[Dict]) -> bool:
        """
        Parse Notes sections based on LLM matching results.

        Args:
            matches: List of {"header": str, "unit_id": str} from LLM

        Returns:
            True if parsing succeeded, False otherwise
        """
        if not matches:
            return False

        # Read all primary definition chapter content with line numbers
        all_content = []  # [(line_num, line_text, source_file), ...]
        global_line_num = 0

        for chapter_name in sorted(self.primary_definition_chapters):
            file_path = self.markdown_dir / f"{chapter_name}.md"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for local_line_num, line in enumerate(lines, 1):
                        all_content.append((global_line_num, line, chapter_name, local_line_num))
                        global_line_num += 1
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        if not all_content:
            return False

        # Find header positions in the content
        header_positions = []  # [(global_line_num, header_text, unit_id, source_file, local_line_num), ...]

        # Create a mapping from header text to unit_id
        header_to_unit_id = {}
        for match in matches:
            header = match.get("header", "").strip()
            unit_id = match.get("unit_id", "")
            if header and unit_id:
                # Normalize header for matching
                header_normalized = re.sub(r'^#+\s*', '', header).strip()
                header_to_unit_id[header_normalized] = unit_id

        # Scan content to find headers in order
        for global_ln, line, source_file, local_ln in all_content:
            line_stripped = line.strip()
            # Remove markdown header markers
            line_clean = re.sub(r'^#+\s*', '', line_stripped)

            # Check if this line matches any header
            if line_clean in header_to_unit_id:
                unit_id = header_to_unit_id[line_clean]
                header_positions.append((global_ln, line_clean, unit_id, source_file, local_ln))
                # Remove from dict to avoid duplicates
                del header_to_unit_id[line_clean]

        # Log any unmatched headers
        for header in header_to_unit_id:
            logger.warning(f"Header '{header}' not found in Notes content")

        if not header_positions:
            logger.warning("No headers found in Notes content")
            return False

        # Sort by position
        header_positions.sort(key=lambda x: x[0])

        # Create sections
        self.notes_sections = []

        for i, (global_ln, header, unit_id, source_file, local_ln) in enumerate(header_positions):
            # Determine end position
            if i + 1 < len(header_positions):
                end_global_ln = header_positions[i + 1][0]
            else:
                end_global_ln = len(all_content)

            # Collect definitions in this section
            section_definitions = []
            for g_ln, line, src_file, loc_ln in all_content:
                if global_ln <= g_ln < end_global_ln:
                    # Check for footnote definition
                    def_match = re.match(r'^\[\^(\w+)\]:\s*(.*)', line)
                    if def_match:
                        key = def_match.group(1)
                        content = def_match.group(2)
                        section_definitions.append(
                            FootnoteDefinition(key, content, src_file, loc_ln)
                        )

            section = NotesSection(
                header_text=header,
                start_line=local_ln,
                end_line=local_ln + (end_global_ln - global_ln),
                source_file=source_file,
                definitions=section_definitions,
                matched_unit_id=unit_id
            )
            self.notes_sections.append(section)

            # Build chapter to section mapping
            self.chapter_to_section[unit_id] = section
            logger.debug(f"Section '{header}' -> {unit_id}: {len(section_definitions)} definitions")

        logger.info(f"Parsed {len(self.notes_sections)} sections from Notes chapter")
        return True

    def _build_section_occurrence_mapping(self) -> None:
        """
        Build occurrence-based mapping within each section.

        Maps references in each chapter to definitions in its matched section.
        """
        if not self.notes_sections:
            return

        # Build mapping for each section
        for section_idx, section in enumerate(self.notes_sections):
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

        logger.debug(f"Built section occurrence mapping for {len(self.notes_sections)} sections")


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