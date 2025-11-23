"""
Footnote scanning and analysis functionality.
"""

import re
from pathlib import Path
from typing import Dict, List, Set
from loguru import logger

from .models import FootnoteDefinition, FootnoteReference, FootnoteStyle


class FootnoteScanner:
    """
    Scans markdown files for footnote definitions and references.
    """

    def __init__(self):
        self.definitions: Dict[str, List[FootnoteDefinition]] = {}  # key -> list of definitions
        self.references: Dict[str, List[FootnoteReference]] = {}    # key -> list of references
        self.chapter_definitions: Dict[str, Set[str]] = {}  # chapter -> set of defined keys
        self.chapter_references: Dict[str, Set[str]] = {}   # chapter -> set of referenced keys
        self.definition_chapters: Set[str] = set()  # Chapters that contain definitions
        self.reference_only_chapters: Set[str] = set()  # Chapters with refs but no defs

    def scan_files(self, markdown_dir: Path, files: List[Path]) -> None:
        """
        Scan all provided files for footnotes.

        Args:
            markdown_dir: Directory containing markdown files
            files: List of file paths to scan
        """
        for md_file in files:
            self._scan_file(md_file)

    def _scan_file(self, file_path: Path) -> None:
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

    def determine_style(self, force_global: bool, auto_global: bool) -> FootnoteStyle:
        """
        Determine whether to use LOCAL or GLOBAL footnote style.

        Args:
            force_global: If True, force global style via CLI flag
            auto_global: If True, auto-detected global mode

        Returns:
            The determined footnote style
        """
        # If forced or auto global, skip analysis
        if force_global or auto_global:
            if force_global:
                logger.info("Using forced GLOBAL footnote style")
            else:
                logger.info("Using auto-detected GLOBAL footnote style (notes chapter found)")
            return FootnoteStyle.GLOBAL

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
        style = FootnoteStyle.LOCAL
        if has_definition_only_chapters and has_reference_only_chapters:
            # Verify that referenced keys have definitions somewhere
            all_defined_keys = set()
            for keys in self.chapter_definitions.values():
                all_defined_keys.update(keys)

            unmatched_refs = set()
            for keys in self.chapter_references.values():
                unmatched_refs.update(keys - all_defined_keys)

            if len(unmatched_refs) / max(1, len(all_defined_keys)) < 0.1:  # Less than 10% unmatched
                style = FootnoteStyle.GLOBAL
            else:
                logger.warning(
                    f"Found {len(unmatched_refs)} unmatched footnote references. "
                    "Using LOCAL style for safety."
                )

        # Additional check: if any chapter has duplicate footnote keys with different content
        if style == FootnoteStyle.GLOBAL:
            if self._has_conflicting_definitions():
                logger.warning("Found conflicting footnote definitions. Using LOCAL style for safety.")
                style = FootnoteStyle.LOCAL

        return style

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

    def identify_primary_definition_chapters(self) -> Set[str]:
        """
        Identify chapters with the most footnote definitions.

        Returns:
            Set of chapter names that are primary definition chapters
        """
        if not self.chapter_definitions:
            return set()

        # Count definitions per chapter (including part files)
        chapter_def_counts = {}
        for chapter, keys in self.chapter_definitions.items():
            # Group by base chapter name (e.g., chapter_7.part1 -> chapter_7)
            base_chapter = chapter.split('.part')[0] if '.part' in chapter else chapter
            if base_chapter not in chapter_def_counts:
                chapter_def_counts[base_chapter] = 0
            chapter_def_counts[base_chapter] += len(keys)

        # Find the maximum number of definitions
        primary_chapters = set()
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
                            primary_chapters.add(full_chapter)

            logger.info(f"Identified primary definition chapters: {sorted(primary_chapters)}")
            total_defs = sum(len(self.chapter_definitions.get(ch, set())) for ch in primary_chapters)
            logger.info(f"These chapters contain {total_defs} footnote definitions")

        return primary_chapters
