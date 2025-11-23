"""
Data models for the footnote management system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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
