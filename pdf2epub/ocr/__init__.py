"""OCR module for Japanese text extraction from PDFs."""

from .illustration_extractor import extract_illustrations, inject_illustrations_into_text

__all__ = ['extract_illustrations', 'inject_illustrations_into_text']