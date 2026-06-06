#!/usr/bin/env python3
"""
ChapterIdentity - Core abstraction for chapter/unit file naming.

Provides a unified interface for parsing and generating file names
across the pdf2epub system, supporting any prefix (chapter_, unit_, etc.)
and handling special cases like front_matter/back_matter.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class ChapterIdentity:
    """
    Represents the identity of a chapter/unit file.

    Attributes:
        prefix: The type prefix (e.g., "chapter", "unit", "front_matter")
        number: The chapter/unit hierarchical index as string (e.g., "7.1.1", None for front_matter/back_matter)
        part: The part number if split (None for single files)

    Examples:
        "chapter_5" -> ChapterIdentity("chapter", "5", None)
        "chapter_7.1.1" -> ChapterIdentity("chapter", "7.1.1", None)
        "chapter_7.1.1.part2" -> ChapterIdentity("chapter", "7.1.1", 2)
        "front_matter" -> ChapterIdentity("front_matter", None, None)
        "back_matter.part1" -> ChapterIdentity("back_matter", None, 1)
    """

    prefix: str
    number: Optional[str]  # Changed from int to str to support "7.1.1"
    part: Optional[int]

    # Regex patterns for parsing
    # Pattern 1: prefix_number[.subindices][.partN] (e.g., chapter_7, chapter_7.1.1, chapter_7.1.1.part2)
    _NUMBERED_PATTERN = re.compile(
        r'^([a-zA-Z_]+)_(\d+(?:\.\d+)*)(?:\.part(\d+))?$'
    )

    # Pattern 2: special_matter[.partN] (e.g., front_matter, back_matter.part1)
    _SPECIAL_PATTERN = re.compile(
        r'^(front_matter|back_matter)(?:\.part(\d+))?$'
    )

    # Known prefixes (can be extended)
    KNOWN_PREFIXES = {'chapter', 'unit', 'section', 'front_matter', 'back_matter'}

    @classmethod
    def parse(cls, filename: str) -> Optional['ChapterIdentity']:
        """
        Parse a filename (with or without extension) into ChapterIdentity.

        Args:
            filename: The filename to parse (e.g., "chapter_7.1.1.part1.md" or "chapter_7.1.1")

        Returns:
            ChapterIdentity if parsing succeeds, None otherwise

        Examples:
            >>> ChapterIdentity.parse("chapter_5.md")
            ChapterIdentity(prefix='chapter', number='5', part=None)

            >>> ChapterIdentity.parse("chapter_7.1.1.part2")
            ChapterIdentity(prefix='chapter', number='7.1.1', part=2)

            >>> ChapterIdentity.parse("front_matter")
            ChapterIdentity(prefix='front_matter', number=None, part=None)
        """
        # Remove file extension if present, but preserve .partN suffix
        # Path.stem incorrectly treats .partN as an extension
        path = Path(filename)
        if path.suffix.lower() in {'.md', '.html', '.txt', '.json', '.xml'}:
            stem = path.stem
        else:
            # No recognized extension, use the full name
            # This handles "chapter_5.part2" where .part2 is NOT an extension
            stem = path.name

        # Try special matter pattern first (front_matter, back_matter)
        match = cls._SPECIAL_PATTERN.match(stem)
        if match:
            prefix = match.group(1)
            part = int(match.group(2)) if match.group(2) else None
            return cls(prefix=prefix, number=None, part=part)

        # Try numbered pattern (chapter_N, chapter_N.M.K, etc.)
        match = cls._NUMBERED_PATTERN.match(stem)
        if match:
            prefix = match.group(1)
            number = match.group(2)  # Keep as string to preserve "7.1.1"
            part = int(match.group(3)) if match.group(3) else None
            return cls(prefix=prefix, number=number, part=part)

        return None

    @property
    def base_name(self) -> str:
        """
        Get the base name without part suffix.

        Examples:
            ChapterIdentity("chapter", 5, 2).base_name -> "chapter_5"
            ChapterIdentity("front_matter", None, 1).base_name -> "front_matter"
        """
        if self.number is not None:
            return f"{self.prefix}_{self.number}"
        else:
            return self.prefix

    @property
    def full_name(self) -> str:
        """
        Get the full name including part suffix.

        Examples:
            ChapterIdentity("chapter", 5, 2).full_name -> "chapter_5.part2"
            ChapterIdentity("chapter", 5, None).full_name -> "chapter_5"
        """
        if self.part is not None:
            return f"{self.base_name}.part{self.part}"
        else:
            return self.base_name

    @property
    def html_name(self) -> str:
        """
        Get the HTML filename (with underscores instead of dots for parts).

        Examples:
            ChapterIdentity("chapter", 5, 2).html_name -> "chapter_5_part2.html"
            ChapterIdentity("chapter", 5, None).html_name -> "chapter_5.html"
        """
        if self.part is not None:
            return f"{self.base_name}_part{self.part}.html"
        else:
            return f"{self.base_name}.html"

    @property
    def is_part(self) -> bool:
        """Check if this represents a part of a split file."""
        return self.part is not None

    @property
    def is_special_matter(self) -> bool:
        """Check if this is front_matter or back_matter."""
        return self.prefix in ('front_matter', 'back_matter')

    @property
    def is_front_matter(self) -> bool:
        """Check if this is front_matter."""
        return self.prefix == 'front_matter'

    @property
    def is_back_matter(self) -> bool:
        """Check if this is back_matter."""
        return self.prefix == 'back_matter'

    @property
    def index_path(self) -> List[int]:
        """
        Get the hierarchical index as a list of integers.

        Examples:
            ChapterIdentity("chapter", "7", None).index_path -> [7]
            ChapterIdentity("chapter", "7.1.1", None).index_path -> [7, 1, 1]
            ChapterIdentity("front_matter", None, None).index_path -> []
        """
        if self.number is None:
            return []
        return [int(x) for x in self.number.split('.')]

    @property
    def sort_key(self) -> Tuple:
        """
        Get a sort key for ordering chapters.

        Order: front_matter < numbered chapters (by hierarchical index) < back_matter
        Within each: parts ordered by part number

        Returns:
            Tuple for sorting (category, index_path, part)
        """
        if self.is_front_matter:
            category = 0
        elif self.is_back_matter:
            category = 2
        else:
            category = 1

        # Use index_path for hierarchical sorting
        idx_path = self.index_path if self.number is not None else []
        part = self.part if self.part is not None else 0

        return (category, idx_path, part)

    @staticmethod
    def make_part_name(base: str, part_num: int) -> str:
        """
        Create a part filename from base name and part number.

        Args:
            base: Base name (e.g., "chapter_5")
            part_num: Part number (1-indexed)

        Returns:
            Part filename (e.g., "chapter_5.part1")

        Examples:
            >>> ChapterIdentity.make_part_name("chapter_5", 1)
            'chapter_5.part1'

            >>> ChapterIdentity.make_part_name("front_matter", 2)
            'front_matter.part2'
        """
        return f"{base}.part{part_num}"

    @staticmethod
    def get_base_from_part(part_name: str) -> str:
        """
        Extract base name from a part name.

        Args:
            part_name: Part filename (e.g., "chapter_5.part1")

        Returns:
            Base name (e.g., "chapter_5")

        Examples:
            >>> ChapterIdentity.get_base_from_part("chapter_5.part1")
            'chapter_5'
        """
        identity = ChapterIdentity.parse(part_name)
        if identity:
            return identity.base_name
        # Fallback: remove .partN suffix
        return re.sub(r'\.part\d+$', '', part_name)

    def __str__(self) -> str:
        return self.full_name

    def __repr__(self) -> str:
        return f"ChapterIdentity(prefix={self.prefix!r}, number={self.number}, part={self.part})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ChapterIdentity):
            return False
        return (self.prefix == other.prefix and
                self.number == other.number and
                self.part == other.part)

    def __hash__(self) -> int:
        return hash((self.prefix, self.number, self.part))

    def __lt__(self, other: 'ChapterIdentity') -> bool:
        """Enable sorting of ChapterIdentity objects."""
        return self.sort_key < other.sort_key


_PART_SUFFIX_RE = re.compile(r'(?:\.part\d+)+$')
_PART_NUM_RE = re.compile(r'\.part(\d+)')


def strip_part_suffix(stem: str) -> str:
    """Strip a trailing chain of .partN suffixes to get the base chapter name.

    Unlike ChapterIdentity.parse, this handles multiply-nested parts.

    Examples:
        "chapter_7.4.part3.part1" -> "chapter_7.4"
        "chapter_5.part2"         -> "chapter_5"
        "chapter_5"               -> "chapter_5"
    """
    # Remove a recognized file extension first (but not .partN)
    path = Path(stem)
    if path.suffix.lower() in {'.md', '.html', '.txt', '.json', '.xml'}:
        stem = path.stem
    else:
        stem = path.name
    return _PART_SUFFIX_RE.sub('', stem)


def part_path(stem: str) -> Tuple[int, ...]:
    """Return the chain of part numbers from a (possibly nested) part name.

    Examples:
        "chapter_7.4.part3.part1" -> (3, 1)
        "chapter_5.part2"         -> (2,)
        "chapter_5"               -> ()
    """
    path = Path(stem)
    if path.suffix.lower() in {'.md', '.html', '.txt', '.json', '.xml'}:
        stem = path.stem
    else:
        stem = path.name
    suffix_match = _PART_SUFFIX_RE.search(stem)
    if not suffix_match:
        return ()
    return tuple(int(x) for x in _PART_NUM_RE.findall(suffix_match.group(0)))


def group_by_base(identities: List[ChapterIdentity]) -> dict:
    """
    Group ChapterIdentity objects by their base name.

    Args:
        identities: List of ChapterIdentity objects

    Returns:
        Dict mapping base_name to list of ChapterIdentity (sorted by part)

    Example:
        >>> ids = [
        ...     ChapterIdentity.parse("chapter_5.part2"),
        ...     ChapterIdentity.parse("chapter_5.part1"),
        ...     ChapterIdentity.parse("chapter_6"),
        ... ]
        >>> grouped = group_by_base(ids)
        >>> grouped["chapter_5"]
        [ChapterIdentity(chapter, 5, 1), ChapterIdentity(chapter, 5, 2)]
    """
    groups = {}

    for identity in identities:
        base = identity.base_name
        if base not in groups:
            groups[base] = []
        groups[base].append(identity)

    # Sort parts within each group
    for base in groups:
        groups[base].sort(key=lambda x: x.part or 0)

    return groups


def discover_parts(directory: Path, base_name: str) -> List[ChapterIdentity]:
    """
    Discover all part files for a given base name in a directory.

    Args:
        directory: Directory to search
        base_name: Base name to find parts for (e.g., "chapter_5")

    Returns:
        List of ChapterIdentity for found parts, sorted by part number
    """
    parts = []

    # Look for both .md and other extensions
    for pattern in [f"{base_name}.part*.md", f"{base_name}.part*"]:
        for file_path in directory.glob(pattern):
            identity = ChapterIdentity.parse(file_path.name)
            if identity and identity.is_part:
                parts.append(identity)

    # Remove duplicates and sort
    unique_parts = list(set(parts))
    unique_parts.sort(key=lambda x: x.part or 0)

    return unique_parts


def is_valid_chapter_file(filename: str) -> bool:
    """
    Check if a filename is a valid chapter/unit file.

    Args:
        filename: Filename to check

    Returns:
        True if the file is a recognized chapter/unit type
    """
    identity = ChapterIdentity.parse(filename)
    return identity is not None


if __name__ == "__main__":
    # Quick tests
    test_cases = [
        "chapter_1",
        "chapter_5.part2",
        "chapter_10.part1.md",
        "unit_001",
        "unit_001.part3",
        "front_matter",
        "back_matter.part1",
        "invalid_file",
        "random.txt",
    ]

    print("ChapterIdentity Parsing Tests:")
    print("=" * 60)

    for test in test_cases:
        identity = ChapterIdentity.parse(test)
        if identity:
            print(f"{test:30} -> {identity}")
            print(f"  base_name: {identity.base_name}")
            print(f"  full_name: {identity.full_name}")
            print(f"  html_name: {identity.html_name}")
            print(f"  is_part: {identity.is_part}")
            print(f"  sort_key: {identity.sort_key}")
        else:
            print(f"{test:30} -> None (not recognized)")
        print()
