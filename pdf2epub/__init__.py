"""PDF to EPUB converter with Japanese OCR support."""

__version__ = "0.2.0"

from .ocr import extract_illustrations, inject_illustrations_into_text

__all__ = ["extract_illustrations", "inject_illustrations_into_text"]
