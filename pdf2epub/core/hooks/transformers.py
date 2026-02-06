"""
Transformers - modify processing results.

Transformers run after LLM response and before validation.
They are chained: the output of one becomes the input of the next.
"""

import re
from typing import List


class RestoreImagesTransformer:
    """Restore images that were removed by LLM during processing."""

    @property
    def name(self) -> str:
        return "RestoreImages"

    def transform(self, key: str, original: str, result: str) -> str:
        """Restore lost images from original to result."""
        from pdf2epub.processors.utils.image_restore import restore_lost_images_fast
        return restore_lost_images_fast(original, result)


class RemoveArtifactsTransformer:
    """Remove common LLM artifacts from output."""

    # Patterns to remove
    PATTERNS: List[str] = [
        r'^```\w*\n',           # Opening code block at start
        r'\n```$',              # Closing code block at end
        r'^```\n',              # Opening code block without language
        r'^Here is the .*?:\n*',  # "Here is the translation:"
        r'^Here\'s the .*?:\n*',  # "Here's the polished version:"
        r'^\*\*Translation:\*\*\n*',  # "**Translation:**"
        r'^\*\*Polished:\*\*\n*',     # "**Polished:**"
    ]

    @property
    def name(self) -> str:
        return "RemoveArtifacts"

    def transform(self, key: str, original: str, result: str) -> str:
        """Remove artifacts from result."""
        cleaned = result
        for pattern in self.PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned.strip()


class NormalizeWhitespaceTransformer:
    """Normalize excessive whitespace in output."""

    @property
    def name(self) -> str:
        return "NormalizeWhitespace"

    def transform(self, key: str, original: str, result: str) -> str:
        """Normalize whitespace."""
        # Replace 4+ newlines with 2
        cleaned = re.sub(r'\n{4,}', '\n\n\n', result)
        # Replace 3+ spaces with 1
        cleaned = re.sub(r' {3,}', ' ', cleaned)
        return cleaned


class StripTransformer:
    """Strip leading and trailing whitespace."""

    @property
    def name(self) -> str:
        return "Strip"

    def transform(self, key: str, original: str, result: str) -> str:
        """Strip whitespace."""
        return result.strip()
