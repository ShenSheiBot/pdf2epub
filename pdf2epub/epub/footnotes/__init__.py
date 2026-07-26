"""
Footnote management system for EPUB generation.

This module handles both local (per-chapter) and global (cross-chapter) footnote styles.
It automatically detects which style is appropriate based on the book's structure.
"""

from .models import FootnoteStyle, FootnoteDefinition, FootnoteReference, NotesSection
from .manager import FootnoteManager
from .validator import FootnoteGraphError, inspect_footnote_graph, validate_footnote_graph

__all__ = [
    'FootnoteStyle',
    'FootnoteDefinition',
    'FootnoteReference',
    'NotesSection',
    'FootnoteManager',
    'FootnoteGraphError',
    'inspect_footnote_graph',
    'validate_footnote_graph',
]
