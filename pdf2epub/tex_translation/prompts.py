"""Stable prompts for TeX translation and compile repair."""

from __future__ import annotations

TRANSLATION_PROMPT_VERSION = "tex-translation-v1"

TRANSLATION_TEMPLATE = """\
Translate the LaTeX fragment below from {source_language} into natural, precise
{target_language} suitable for a graduate-level academic paper.

Return only the complete translated LaTeX fragment. Do not add Markdown fences,
a preamble, explanations, or commentary.

Rules:
- Translate prose, headings, quotations, footnotes, and captions.
- Preserve LaTeX commands, environments, braces, labels, references, citations,
  bibliography keys, filenames, and mathematical notation.
- Do not omit, summarize, reorder, or invent source content.
- Keep formulas unchanged.
- Leave text in the source language when translating it would risk changing a
  technical name or breaking TeX.

SOURCE FRAGMENT:

{fragment}"""

CONTINUATION_INSTRUCTION = """\
Continue the same translation from exactly where the previous response stopped.
Return only the missing suffix of the translated LaTeX fragment."""

REPAIR_SYSTEM = """\
You repair a single translated TeX unit. The only hard completion criterion is
that the complete project compiles with the supplied latexmk/XeLaTeX command.
Make the smallest possible change inside the marked translation unit. Do not
polish prose, change formulas for style, or edit unrelated files. If translated
prose cannot be repaired safely, restore that passage from the original source.
"""


def build_translation_messages(
    fragment: str,
    *,
    source_language: str,
    target_language: str,
    prefix: str | None = None,
) -> list[dict]:
    """Keep the same unit byte-identical across immediate retries/continuations."""
    first_message = TRANSLATION_TEMPLATE.format(
        source_language=source_language,
        target_language=target_language,
        fragment=fragment,
    )
    messages: list[dict] = [{"role": "user", "content": first_message}]
    if prefix is not None:
        messages.extend(
            [
                {"role": "assistant", "content": prefix},
                {"role": "user", "content": CONTINUATION_INSTRUCTION},
            ]
        )
    return messages
