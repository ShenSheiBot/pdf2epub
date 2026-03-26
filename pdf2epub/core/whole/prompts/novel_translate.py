"""
Prompts and utilities for the novel translation verification agent.

The agent verifies chunked translations: merges chunks, checks head/tail
alignment with source, detects hallucination, and decides continue/complete.
"""

NOVEL_TRANSLATE_SYSTEM = """\
You are a novel translation verification agent. \
You verify chunked translation output against Japanese source text, \
checking head/tail alignment and detecting hallucination.\
"""

NOVEL_TRANSLATE_INSTRUCTIONS = """\
## Background

A translation system translates a Japanese chapter into Chinese, one line at a time (1 source line → 1 translated line). It works in chunks. Your job is to verify the translation and decide whether it's done.

## Files

- `originals/raw_output.txt` — first translation chunk (read-only)
- `originals/continuation_001.txt` — incremental continuation from where the last chunk left off (read-only)
- `originals/source.txt` — full Japanese source (read-only)
- `workspace/translated.txt` — the accumulated translation you maintain (writable)

## Merge

First, merge the translation chunks:
```python
from _utils import merge_to_translated
print(merge_to_translated())
```
This handles round 1 (copy raw_output) and round 2+ (append continuation) automatically, and shows the seam context if a continuation was appended.

## Verification

After merging, check three things:

1. **First 5 lines**: translated lines 1-5 vs source lines 1-5. Should be translations, not meta-comments like "以下是翻译：". If line 1 is a meta-comment (not a translation of source line 1), delete it.

2. **Last 5 lines**: translated lines N-4 to N vs source lines N-4 to N (where N = translated line count). Should correspond to each other.

3. **Seam check** (round 2+ only): if continuation was appended, check the join point — the last few lines before the seam and first few lines after. Look for duplicated lines or gaps.

## Decision

If first 5 and last 5 both match, decide immediately — do not check anything else:
- |N - S| <= 3 (within tolerance) → **complete**
- N < S - 3 (more than 3 lines short) → **continue**

Only if last 5 DON'T match source at position N → suspect **hallucination**. In that case (and only that case), search backwards (check N-50, N-100, etc.) to find where hallucination starts, truncate translated.txt there. → **continue**

## Rules

- Use `wc -l` for counts, `tail`/`head`/`sed -n` for spot checks. Don't read entire files.
- originals/ is read-only. Write only to workspace/.
- Do not check middle positions if first 5 and last 5 already match.
"""

# Plain-text utilities injected into work_dir root
NOVEL_UTILS_PY = '''\
"""Utilities for novel translation verification agent.

All paths are relative to cwd (work_dir root, enforced by sandbox).
Import with: from _utils import ...
"""

import pathlib
import os


def line_count(path):
    """Count non-empty lines in a file."""
    return len([l for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()])


def head(path_or_lines, n=5):
    """Return first n non-empty lines."""
    if isinstance(path_or_lines, list):
        return path_or_lines[:n]
    return [l for l in pathlib.Path(path_or_lines).read_text(encoding="utf-8").splitlines() if l.strip()][:n]


def tail(path_or_lines, n=10):
    """Return last n non-empty lines."""
    if isinstance(path_or_lines, list):
        return path_or_lines[-n:]
    return [l for l in pathlib.Path(path_or_lines).read_text(encoding="utf-8").splitlines() if l.strip()][-n:]


def _resolve_path(rel_path):
    """Resolve a relative path. Works whether cwd is work_dir or workspace/."""
    p = pathlib.Path(rel_path)
    if p.exists():
        return str(p)
    parent = pathlib.Path.cwd().parent / rel_path
    if parent.exists():
        return str(parent)
    return str(p)


def originals_path(filename=''):
    """Get the path to a file in originals/. Works from any cwd."""
    p = pathlib.Path('originals') / filename if filename else pathlib.Path('originals')
    resolved = _resolve_path(str(p))
    return resolved


def workspace_path(filename=''):
    """Get the path to a file in workspace/. Works from any cwd."""
    p = pathlib.Path('workspace') / filename if filename else pathlib.Path('workspace')
    resolved = _resolve_path(str(p))
    return resolved


def merge_to_translated(originals_dir=None, workspace_dir=None):
    """Merge raw_output + continuations into workspace/translated.txt.

    Round 1 (no translated.txt): copies raw_output.txt.
    Round 2+: appends continuation_001.txt.
    Ensures clean newline between files.
    Auto-detects paths whether cwd is work_dir or workspace/.
    """
    # Auto-detect paths: look for originals/ relative to cwd or parent
    cwd = pathlib.Path.cwd()
    if originals_dir is None:
        if (cwd / "originals").is_dir():
            originals_dir = "originals"
            workspace_dir = workspace_dir or "workspace"
        elif (cwd.parent / "originals").is_dir():
            originals_dir = str(cwd.parent / "originals")
            workspace_dir = workspace_dir or str(cwd)
        else:
            originals_dir = "originals"
            workspace_dir = workspace_dir or "workspace"
    elif workspace_dir is None:
        workspace_dir = "workspace"

    raw = pathlib.Path(originals_dir) / "raw_output.txt"
    cont = pathlib.Path(originals_dir) / "continuation_001.txt"
    translated = pathlib.Path(workspace_dir) / "translated.txt"

    seam_info = ""
    if not translated.exists():
        # Round 1: copy raw_output
        text = raw.read_text(encoding="utf-8")
        if text and not text.endswith("\\n"):
            text += "\\n"
        translated.write_text(text, encoding="utf-8")
    elif cont.exists() and cont.stat().st_size > 0:
        # Round 2+: append continuation
        cont_text = cont.read_text(encoding="utf-8")
        cont_lines = [l for l in cont_text.splitlines() if l.strip()]

        existing = translated.read_text(encoding="utf-8")
        if existing and not existing.endswith("\\n"):
            existing += "\\n"
        existing_lines = [l for l in existing.splitlines() if l.strip()]
        seam_line = len(existing_lines)

        translated.write_text(existing + cont_text, encoding="utf-8")

        # Show seam context
        before = existing_lines[-3:] if len(existing_lines) >= 3 else existing_lines
        after = cont_lines[:3] if len(cont_lines) >= 3 else cont_lines
        seam_info = f"\\nSeam at line {seam_line}:"
        for i, l in enumerate(before):
            seam_info += f"\\n  {seam_line - len(before) + i + 1}: {l[:60]}"
        seam_info += f"\\n  --- continuation joined here ---"
        for i, l in enumerate(after):
            seam_info += f"\\n  {seam_line + i + 1}: {l[:60]}"

    lines = [l for l in translated.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(lines)
    result = f"Translated: {n} lines"
    if seam_info:
        result += seam_info
    return result
'''
