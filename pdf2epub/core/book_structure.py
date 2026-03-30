"""
BookStructure: Book metadata and content-aware processing.

This module handles:
1. Loading book_structure.json
2. Chapter type detection (notes, appendix, front/back matter)
3. Content type auto-detection (Japanese, academic, general)
4. Image-only content detection

CRITICAL: Without chapter type detection, validation will fail on
front/back matter that legitimately has different content ratios.
Without content type detection, wrong prompts will be used.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

from ._frozen import Frozen, final, check_final_methods


@dataclass(frozen=True)
class ChapterInfo:
    """Information about a chapter."""
    file_key: str
    chapter_type: str  # "chapter", "notes", "appendix", "front_matter", "back_matter", "toc"
    title: Optional[str] = None
    number: Optional[str] = None  # "5" or "7.1.1"
    toc_path: Optional[str] = None
    page_range: Optional[Tuple[int, int]] = None
    has_footnotes: bool = False

    @property
    def is_front_back_matter(self) -> bool:
        return self.chapter_type in ("front_matter", "back_matter", "notes", "appendix", "toc")

    @property
    def skip_validation(self) -> bool:
        """Whether to skip strict validation for this chapter."""
        return self.is_front_back_matter


# Chapter number pattern
CHAPTER_NUMBER_PATTERN = re.compile(r'^chapter[_\s]*(\d+(?:\.\d+)*)', re.IGNORECASE)

# Japanese character pattern (Hiragana, Katakana, CJK)
JAPANESE_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')  # Hiragana + Katakana

# Academic indicators
ACADEMIC_INDICATORS = [
    r'\[\^?\d+\]',  # Footnote references
    r'et al\.',
    r'\(\d{4}\)',  # Year citations
    r'doi:',
    r'pp\.\s*\d+',
    r'Vol\.\s*\d+',
    r'ibid\.',
    r'op\.\s*cit\.',
    r'cf\.',
]


@check_final_methods
class BookStructure(Frozen, frozen=True):
    """
    Book structure manager.

    FROZEN: Cannot be inherited or modified.

    Features:
    1. Load and cache book_structure.json
    2. Chapter info resolution by file key
    3. Content type auto-detection
    4. Special content detection (image-only, notes chapter)
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save', 'build_prompt'}

    def __init__(self, book_dir: Path):
        """
        Initialize book structure.

        Args:
            book_dir: Directory containing book output (output/{title}/)
        """
        self._book_dir = Path(book_dir)
        self._structure_path = self._book_dir / "book_structure.json"
        self._toc_path = self._book_dir / "toc_tree.json"

        # Load structure
        self._structure: Optional[Dict] = None
        self._toc: Optional[Dict] = None
        self._chapters: Dict[str, ChapterInfo] = {}

        self._load_structure()

    def _load_structure(self) -> None:
        """Load book structure from JSON."""
        # Try book_structure.json
        if self._structure_path.exists():
            try:
                self._structure = json.loads(
                    self._structure_path.read_text(encoding='utf-8')
                )
                self._parse_chapters()
                logger.debug(f"Loaded book structure with {len(self._chapters)} chapters")
            except Exception as e:
                logger.warning(f"Could not load book_structure.json: {e}")

        # Try toc_tree.json
        if self._toc_path.exists():
            try:
                self._toc = json.loads(
                    self._toc_path.read_text(encoding='utf-8')
                )
            except Exception as e:
                logger.warning(f"Could not load toc_tree.json: {e}")

    def _parse_chapters(self) -> None:
        """Parse chapters from structure."""
        if not self._structure:
            return

        chapters = self._structure.get("chapters", [])

        def process_chapter(ch: Dict, parent_path: str = "") -> None:
            file_key = ch.get("file_key", ch.get("id", ""))
            if not file_key:
                return

            # Determine chapter type
            ch_type = ch.get("type", "chapter")
            title = ch.get("title", "")

            # Auto-detect type from title if not specified
            # NOTE: CJK keywords are intentional for i18n support (Japanese: 注釈, 目次, etc.)
            if ch_type == "chapter":
                title_lower = title.lower()
                if any(w in title_lower for w in ["notes", "注", "注釈", "endnotes"]):
                    ch_type = "notes"
                elif any(w in title_lower for w in ["appendix", "付録", "附录"]):
                    ch_type = "appendix"
                elif any(w in title_lower for w in ["contents", "目次", "目录", "toc"]):
                    ch_type = "toc"
                elif any(w in title_lower for w in ["preface", "foreword", "introduction", "前言", "序"]):
                    ch_type = "front_matter"
                elif any(w in title_lower for w in ["afterword", "epilogue", "後記", "后记"]):
                    ch_type = "back_matter"

            # Parse chapter number
            number = None
            match = CHAPTER_NUMBER_PATTERN.match(file_key)
            if match:
                number = match.group(1)
            elif "chapter" in ch:
                number = str(ch.get("chapter"))

            # Build TOC path
            toc_path = f"{parent_path}/{title}" if parent_path else title

            # Page range
            page_range = None
            if "start_page" in ch and "end_page" in ch:
                page_range = (ch["start_page"], ch["end_page"])

            self._chapters[file_key] = ChapterInfo(
                file_key=file_key,
                chapter_type=ch_type,
                title=title,
                number=number,
                toc_path=toc_path,
                page_range=page_range,
                has_footnotes=ch.get("has_footnotes", False)
            )

            # Process children
            for child in ch.get("children", []):
                process_chapter(child, toc_path)

        for chapter in chapters:
            process_chapter(chapter)

    @final
    def get_chapter_info(self, file_key: str) -> ChapterInfo:
        """
        Get chapter information for a file.

        Args:
            file_key: File identifier (e.g., "chapter_5")

        Returns:
            ChapterInfo (defaults to "chapter" type if not found)
        """
        # Try exact match
        if file_key in self._chapters:
            return self._chapters[file_key]

        # Try without .partN suffix
        base_key = file_key.split('.part')[0] if '.part' in file_key else file_key
        if base_key in self._chapters:
            return self._chapters[base_key]

        # Try to parse chapter number from file key
        number = None
        match = CHAPTER_NUMBER_PATTERN.match(file_key)
        if match:
            number = match.group(1)

        # Return default
        return ChapterInfo(
            file_key=file_key,
            chapter_type="chapter",
            number=number
        )

    @final
    def detect_content_type(self, content: str) -> str:
        """
        Auto-detect content type from content.

        Args:
            content: Text content to analyze

        Returns:
            "japanese", "academic", or "general"
        """
        if not content:
            return "general"

        # Count Japanese characters
        japanese_chars = len(JAPANESE_PATTERN.findall(content))
        total_chars = len(content)

        # Check Japanese ratio (more than 10%)
        if total_chars > 0 and japanese_chars / total_chars > 0.10:
            return "japanese"

        # Check academic indicators
        academic_count = 0
        for pattern in ACADEMIC_INDICATORS:
            if re.search(pattern, content, re.IGNORECASE):
                academic_count += 1

        if academic_count >= 2:
            return "academic"

        return "general"

    @final
    def has_notes_chapter(self) -> bool:
        """
        Check if book has a dedicated notes chapter.

        This affects footnote handling - global vs per-chapter.
        """
        for ch_info in self._chapters.values():
            if ch_info.chapter_type == "notes":
                return True

        # Also check structure directly
        if self._structure:
            for ch in self._structure.get("chapters", []):
                if ch.get("type") == "notes":
                    return True

        return False

    @final
    def is_vertical_text(self) -> bool:
        """Check if book uses vertical text layout."""
        if self._structure:
            return self._structure.get("is_vertical_text", False)
        return False

    @final
    def get_language(self) -> Optional[str]:
        """Get book's primary language."""
        if self._structure:
            return self._structure.get("language")
        return None

    @final
    def is_image_only_content(self, content: str, min_text_chars: int = 100) -> bool:
        """
        Check if content is primarily images.

        Args:
            content: Content to check
            min_text_chars: Minimum text characters required

        Returns:
            True if content is mostly images
        """
        if not content:
            return True

        # Remove markdown images
        text_only = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # Remove HTML images
        text_only = re.sub(r'<img[^>]*>', '', text_only)
        # Remove whitespace
        text_only = text_only.strip()

        return len(text_only) < min_text_chars

    @final
    def detect_notes_chapter_by_content(self, content: str) -> bool:
        """
        Detect if content is a notes chapter by its content.

        Fallback when chapter type is not in structure.

        Args:
            content: Content to check

        Returns:
            True if this looks like a notes chapter
        """
        if not content:
            return False

        # Check for notes/references header at start
        first_500 = content[:500].lower()
        note_headers = ["# notes", "# 注", "# 注釈", "## notes", "# references", "# endnotes"]
        if any(h in first_500 for h in note_headers):
            return True

        # Check for numbered list pattern (typical of notes)
        # e.g., "1. ", "[1] ", "^1 "
        lines = content.split('\n')
        numbered_lines = 0
        for line in lines[:50]:  # Check first 50 lines
            if re.match(r'^\s*(\d+\.|\[\d+\]|\^\d+)\s+', line):
                numbered_lines += 1

        # If more than 30% are numbered, likely notes
        if len(lines) > 0 and numbered_lines / min(len(lines), 50) > 0.3:
            return True

        return False

    @final
    def get_all_chapters(self) -> Dict[str, ChapterInfo]:
        """Get all loaded chapter info."""
        return dict(self._chapters)

    @final
    def get_toc(self) -> Optional[Dict]:
        """Get table of contents structure."""
        return self._toc


