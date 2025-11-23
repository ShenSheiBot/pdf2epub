"""
Footnote management system for EPUB generation.

This module is a facade that re-exports from the refactored footnotes package
for backward compatibility.
"""

# Re-export all public symbols from the footnotes package
from .footnotes import (
    FootnoteStyle,
    FootnoteDefinition,
    FootnoteReference,
    NotesSection,
    FootnoteManager,
)

__all__ = [
    'FootnoteStyle',
    'FootnoteDefinition',
    'FootnoteReference',
    'NotesSection',
    'FootnoteManager',
]
