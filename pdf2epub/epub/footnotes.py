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
    line_number: int   # Line number in the file


@dataclass
class FootnoteReference:
    """Represents a footnote reference."""
    key: str           # The footnote key
    chapter: str       # The chapter file where referenced
    line_number: int   # Line number in the file


class FootnoteManager:
    """
    Manages footnote processing for EPUB generation.
    
    Automatically detects whether footnotes are organized locally (per-chapter)
    or globally (centralized in specific chapters) and handles them appropriately.
    """
    
    def __init__(self, markdown_dir: Path):
        """
        Initialize the footnote manager.
        
        Args:
            markdown_dir: Directory containing markdown files
        """
        self.markdown_dir = Path(markdown_dir)
        self.style = FootnoteStyle.LOCAL  # Default to local style
        self.definitions: Dict[str, List[FootnoteDefinition]] = {}  # key -> list of definitions
        self.references: Dict[str, List[FootnoteReference]] = {}    # key -> list of references
        self.chapter_definitions: Dict[str, Set[str]] = {}  # chapter -> set of defined keys
        self.chapter_references: Dict[str, Set[str]] = {}   # chapter -> set of referenced keys
        self.definition_chapters: Set[str] = set()  # Chapters that contain definitions
        self.reference_only_chapters: Set[str] = set()  # Chapters with refs but no defs
        
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
        
        # Scan each file for footnotes
        for md_file in chapter_files:
            self._scan_file_for_footnotes(md_file)
        
        # Determine style based on the pattern
        self._determine_footnote_style()
        
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
        
        GLOBAL style is used only when:
        1. There are chapters with only definitions (like a Notes chapter)
        2. There are other chapters with references but no definitions
        3. The referenced keys match defined keys across chapters
        """
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
                    logger.debug(f"Footnote key '{key}' has conflicting definitions")
                    return True
        return False
    
    def _log_analysis_results(self) -> None:
        """Log the results of the footnote analysis."""
        logger.info(f"Footnote style detected: {self.style.value.upper()}")
        
        if self.style == FootnoteStyle.GLOBAL:
            logger.info(f"Found {len(self.definition_chapters)} chapters with definitions")
            logger.info(f"Found {len(self.reference_only_chapters)} chapters with references only")
            
            # Log which chapters contain definitions
            for chapter in sorted(self.definition_chapters):
                count = len(self.chapter_definitions.get(chapter, set()))
                logger.debug(f"  {chapter}: {count} definitions")
            
            # Log chapters with references only
            for chapter in sorted(self.reference_only_chapters):
                count = len(self.chapter_references.get(chapter, set()))
                logger.debug(f"  {chapter}: {count} references (no definitions)")
    
    def get_style(self) -> FootnoteStyle:
        """Get the detected footnote style."""
        return self.style
    
    def get_footnote_html(self, key: str, source_chapter: str) -> Optional[str]:
        """
        Get the HTML for a footnote reference.
        
        Args:
            key: The footnote key (e.g., "1", "note")
            source_chapter: The chapter containing the reference
            
        Returns:
            HTML string for the footnote reference, or None if not found
        """
        if self.style == FootnoteStyle.LOCAL:
            # Use local anchors within the same file
            return f'<sup id="fnref{key}"><a class="footnote-ref" href="#fn:{key}">[{key}]</a></sup>'
        
        # Global style: find where the footnote is defined
        if key in self.definitions:
            # Get the first definition (should typically be only one in global style)
            definition = self.definitions[key][0]
            target_chapter = definition.chapter
            
            if target_chapter == source_chapter:
                # Same file, use local anchor
                return f'<sup id="fnref{key}"><a class="footnote-ref" href="#fn:{key}">[{key}]</a></sup>'
            else:
                # Cross-file reference
                return (
                    f'<sup id="fnref{key}">'
                    f'<a class="footnote-ref" href="{target_chapter}.html#fn:{key}">[{key}]</a>'
                    f'</sup>'
                )
        
        # Footnote not found - return a plain reference
        logger.warning(f"Footnote '{key}' referenced in {source_chapter} but not defined")
        return f'<sup>[{key}]</sup>'
    
    def get_definition_content(self, key: str) -> Optional[str]:
        """
        Get the content of a footnote definition.
        
        Args:
            key: The footnote key
            
        Returns:
            The footnote content, or None if not found
        """
        if key in self.definitions and self.definitions[key]:
            return self.definitions[key][0].content
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