"""
Footnote management system for EPUB generation.

This module handles both local (per-chapter) and global (cross-chapter) footnote styles.
It automatically detects which style is appropriate based on the book's structure.
"""

from .models import FootnoteStyle, FootnoteDefinition, FootnoteReference, NotesSection
from .manager import FootnoteManager

__all__ = [
    'FootnoteStyle',
    'FootnoteDefinition',
    'FootnoteReference',
    'NotesSection',
    'FootnoteManager',
]
