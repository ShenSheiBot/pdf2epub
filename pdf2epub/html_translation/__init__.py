"""
HTML Translation Pipeline.

This package provides direct HTML translation for EPUB content,
using HTMLCompressor for structure-preserving translation.
"""

from .compressor import HTMLCompressor
from .splitter import HTMLSplitter, CompressedSplitter
from .translator import HTMLTranslateProcessor
from .builder import HTMLEpubBuilder, HTMLEpubPipeline, build_html_epub

__all__ = [
    # Compression
    "HTMLCompressor",
    # Splitting
    "HTMLSplitter",
    "CompressedSplitter",
    # Translation
    "HTMLTranslateProcessor",
    # Building
    "HTMLEpubBuilder",
    "HTMLEpubPipeline",
    "build_html_epub",
]
