"""
Unified unit ID generation for the pdf2epub pipeline.

Unit IDs use hierarchical index format:
- chapter_7       (top-level, 7th entry)
- chapter_7.1     (first child of 7th entry)
- chapter_7.1.1   (first grandchild)
- chapter_7.1.1.part2  (with part suffix)
"""

from typing import List


def generate_unit_id(index_path: List[int]) -> str:
    """
    Generate a hierarchical index-based unit ID.

    Args:
        index_path: List of 1-based indices from root to current node.
                   e.g., [7, 1, 1] for the first grandchild of the 7th top-level entry.

    Returns:
        Unit ID string like "chapter_7.1.1"

    Examples:
        >>> generate_unit_id([7])
        'chapter_7'
        >>> generate_unit_id([7, 1])
        'chapter_7.1'
        >>> generate_unit_id([7, 1, 1])
        'chapter_7.1.1'
    """
    if not index_path:
        raise ValueError("index_path cannot be empty")

    if len(index_path) == 1:
        return f"chapter_{index_path[0]}"

    # Join all indices after the first with dots
    sub_indices = ".".join(str(i) for i in index_path[1:])
    return f"chapter_{index_path[0]}.{sub_indices}"


def parse_unit_id(unit_id: str) -> List[int]:
    """
    Parse a unit ID back into an index path.

    Args:
        unit_id: Unit ID string like "chapter_7.1.1"

    Returns:
        List of indices like [7, 1, 1]

    Examples:
        >>> parse_unit_id("chapter_7")
        [7]
        >>> parse_unit_id("chapter_7.1.1")
        [7, 1, 1]
    """
    if not unit_id.startswith("chapter_"):
        raise ValueError(f"Invalid unit_id format: {unit_id}")

    # Remove "chapter_" prefix
    rest = unit_id[8:]  # len("chapter_") == 8

    # Split by dots and convert to integers
    parts = rest.split(".")
    return [int(p) for p in parts]
