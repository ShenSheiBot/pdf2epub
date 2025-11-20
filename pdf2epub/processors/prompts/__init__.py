"""
Prompt generation functions for processors.

This module contains pure functions for generating prompts,
making them easy to test and reuse.
"""

from .polish_prompts import (
    create_academic_polish_prompt,
    create_academic_global_prompt,
    create_notes_chapter_prompt,
    create_japanese_polish_prompt,
    create_general_polish_prompt,
    create_polish_prompt,
    detect_content_type,
)

__all__ = [
    "create_academic_polish_prompt",
    "create_academic_global_prompt",
    "create_notes_chapter_prompt",
    "create_japanese_polish_prompt",
    "create_general_polish_prompt",
    "create_polish_prompt",
    "detect_content_type",
]
