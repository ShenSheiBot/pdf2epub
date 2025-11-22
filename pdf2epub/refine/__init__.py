"""
Refined Breakdown module - Structure analysis with boundary verification.

This module provides precise splitting of PDF books based on TOC structure,
with boundary verification to handle sections that don't start at page breaks.
"""

from .main import RefinedBreakdown
from .toc_tree import TOCNode
from .refiner_state import RefinerState

__all__ = ['RefinedBreakdown', 'TOCNode', 'RefinerState']
