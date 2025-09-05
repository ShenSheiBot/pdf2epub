"""
EPUB generation package.

This package provides modular components for EPUB file generation:
- EpubConfig: Configuration data class
- ContentConverter: Content transformation and preparation
- EpubBuilder: EPUB file structure creation and packaging
"""

from .config import EpubConfig
from .converter import ContentConverter
from .builder import EpubBuilder

__all__ = ['EpubConfig', 'ContentConverter', 'EpubBuilder']