"""
EPUB configuration data class.

This module provides a structured configuration object for EPUB generation,
encapsulating all settings and metadata needed for the process.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class EpubConfig:
    """Configuration for EPUB generation."""
    
    # Basic metadata
    book_title: str
    author: str = "Unknown Author"
    language: str = "en"
    
    # Directories
    input_dir: Path = None
    output_dir: Path = None
    markdown_dir: Path = None
    images_dir: Path = None
    epub_dir: Path = None
    
    # File paths
    cover_path: Optional[Path] = None
    input_pdf_path: Optional[Path] = None
    output_epub_path: Optional[Path] = None
    
    # Structure
    book_structure: Optional[Dict[str, Any]] = None
    reference_structure: Optional[Dict[str, Any]] = None
    
    # Processing options
    use_translated: bool = False
    use_relevel: bool = False
    create_zip: bool = False
    
    # API configuration
    api_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize paths based on book title if not provided."""
        # Convert string paths to Path objects first
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.markdown_dir, str):
            self.markdown_dir = Path(self.markdown_dir)
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        if isinstance(self.epub_dir, str):
            self.epub_dir = Path(self.epub_dir)
        if isinstance(self.output_epub_path, str):
            self.output_epub_path = Path(self.output_epub_path)
        if self.cover_path and isinstance(self.cover_path, str):
            self.cover_path = Path(self.cover_path)
        if self.input_pdf_path and isinstance(self.input_pdf_path, str):
            self.input_pdf_path = Path(self.input_pdf_path)

        # Set default paths if not provided
        if self.input_dir is None:
            self.input_dir = Path("output") / self.book_title

        if self.output_dir is None:
            self.output_dir = self.input_dir

        if self.markdown_dir is None:
            if self.use_translated:
                self.markdown_dir = self.input_dir / "translated"
            else:
                self.markdown_dir = self.input_dir / "polished_markdown"

        if self.images_dir is None:
            self.images_dir = self.input_dir / "images"

        if self.epub_dir is None:
            self.epub_dir = self.output_dir / "epub"

        if self.output_epub_path is None:
            self.output_epub_path = self.output_dir / f"{self.book_title}.epub"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "EpubConfig":
        """
        Create an EpubConfig from a dictionary (e.g., from config.yaml).

        Args:
            config_dict: Dictionary with configuration values
            **kwargs: Additional keyword arguments to override config

        Returns:
            EpubConfig instance
        """
        # Extract relevant fields
        config = cls(
            book_title=config_dict.get("title", "Untitled"),
            author=config_dict.get("author", "Unknown Author"),
            language=config_dict.get("language", "en"),
            input_dir=config_dict.get("input_dir"),
            output_dir=config_dict.get("output_dir"),
            api_config=config_dict,
            **kwargs
        )

        return config
