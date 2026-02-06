"""
Phase module - composable processing stages.

Core design:
- Any phase can follow any other (polish -> translate -> polish)
- No aggregation between phases
- Aggregation only happens at build-epub
- Each phase reads parts and outputs parts

Components:
- Phase: A processing stage that reads from input_dir, writes to output_dir
- WorkUnitLoader: Protocol for loading work units
- PartBasedLoader: Load part files (chapter_1.part1.md)
- ChapterBasedLoader: Load chapter files (chapter_1.md)
- HTMLLoader: Load HTML files (for EPUB translation)
"""

from ._protocol import PhaseResult, WorkUnitLoader

from .loader import (
    PartBasedLoader,
    ChapterBasedLoader,
    HTMLLoader,
)

from .phase import Phase

__all__ = [
    # Protocol and types
    'PhaseResult',
    'WorkUnitLoader',
    # Loaders
    'PartBasedLoader',
    'ChapterBasedLoader',
    'HTMLLoader',
    # Phase
    'Phase',
]
