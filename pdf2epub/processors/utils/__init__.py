"""
Utilities for markdown processors.

Contains:
- Content splitting utilities
- Image restoration
- Nested processor for recursive splitting
- Split management
"""

from .content_splitter import (
    split_content,
    fuzzy_find_sentence
)
from .image_restore import (
    extract_images,
    extract_images_from_markdown,
    restore_lost_images,
    restore_lost_images_fast,
    find_best_insertion_point,
)
from .nested_processor import (
    NestedPartProcessor,
    NestedPart,
    create_root_part,
    classify_error,
)
from .split_manager import (
    SplitManager,
    SplitResult,
    PartInfo,
)
from .splitter_strategies import (
    ContentSplitter,
    SimpleSplitter,
    MarkdownStructureSplitter,
)

__all__ = [
    # Content splitter
    'split_content',
    'fuzzy_find_sentence',
    # Image restore
    'extract_images',
    'extract_images_from_markdown',
    'restore_lost_images',
    'restore_lost_images_fast',
    'find_best_insertion_point',
    # Nested processor
    'NestedPartProcessor',
    'NestedPart',
    'create_root_part',
    'classify_error',
    # Split manager
    'SplitManager',
    'SplitResult',
    'PartInfo',
    # Splitter strategies
    'ContentSplitter',
    'SimpleSplitter',
    'MarkdownStructureSplitter',
]
