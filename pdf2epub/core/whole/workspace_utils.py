"""
Pre-built utilities placed into agent workspace/_utils.py.

Agent can `from _utils import ...` in any Python script it writes.
"""

WORKSPACE_UTILS_CODE = '''\
"""Pre-built utilities for agent scripts. Usage: from _utils import ..."""
import re
import json
import pathlib as _pathlib


def _resolve_path(rel_path):
    """Resolve a relative path that may reference originals/ or workspace/.
    Works whether cwd is work_dir or workspace/."""
    p = _pathlib.Path(rel_path)
    if p.exists():
        return str(p)
    # Try from parent (if we're in workspace/)
    parent = _pathlib.Path.cwd().parent / rel_path
    if parent.exists():
        return str(parent)
    return str(p)  # return as-is, let caller handle the error


def originals_path(filename=''):
    """Get the path to a file in originals/. Works from any cwd."""
    return _resolve_path(_pathlib.Path('originals') / filename if filename else 'originals')


def workspace_path(filename=''):
    """Get the path to a file in workspace/. Works from any cwd."""
    return _resolve_path(_pathlib.Path('workspace') / filename if filename else 'workspace')


def extract_divs(file_path):
    """Extract <div> contents from LLM output file. Returns list of lines."""
    file_path = _resolve_path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Strip markdown code fences
    content = re.sub(r'```[a-z]*\\n?', '', content)
    content = re.sub(r'```', '', content)
    # Remove real newlines (LLM may add formatting breaks)
    content = content.replace('\\n', '').replace('\\r', '')
    # Extract div contents
    lines = re.findall(r'<div>(.*?)</div>', content, re.DOTALL)
    # Filter empty divs
    return [l for l in lines if l.strip()]


def load_source_lines(path='originals/source.txt'):
    """Load source lines (non-empty) from file."""
    path = _resolve_path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return [l for l in f.read().splitlines() if l.strip()]


def get_tag_seq(text):
    """Extract tag name sequence from HTML text. E.g. ['span', 'a', '/a', '/span']."""
    return [t.lower() for t in re.findall(r'<(/?[a-zA-Z0-9]+)', text)]


def check_tags(source_lines, translated_lines):
    """Compare tag sequences between source and translated lines.
    Returns list of (line_index, source_tags, translated_tags) for mismatches."""
    mismatches = []
    for i, (src, tgt) in enumerate(zip(source_lines, translated_lines)):
        st = get_tag_seq(src)
        tt = get_tag_seq(tgt)
        if st != tt:
            mismatches.append((i, st, tt))
    return mismatches


def repair_tags_from_source(source_line, translated_line):
    """Repair a translated line to match source tag structure.

    Strategy: extract plain text from translated line, then rebuild
    using the source line's tag skeleton.

    This is a safe fallback that preserves the translated text content
    while ensuring tag structure matches exactly.
    """
    # Extract plain text from translated line
    plain_text = re.sub(r'<[^>]+>', '', translated_line).strip()

    # Extract tag skeleton from source line
    parts = []
    last_end = 0
    text_positions = []
    for m in re.finditer(r'<[^>]+>', source_line):
        if m.start() > last_end:
            text_positions.append(len(parts))
            parts.append(source_line[last_end:m.start()])  # original text placeholder
        parts.append(m.group(0))  # tag
        last_end = m.end()
    if last_end < len(source_line):
        text_positions.append(len(parts))
        parts.append(source_line[last_end:])

    if not text_positions:
        return translated_line  # No text in source, nothing to do

    # Simple strategy: put all translated text in the first text position
    for i, pos in enumerate(text_positions):
        if i == 0:
            parts[pos] = plain_text
        else:
            # Check if source text at this position is a number (like footnote)
            orig_text = parts[pos].strip()
            if orig_text.isdigit():
                pass  # Keep the original number
            else:
                parts[pos] = ''  # Clear other text positions

    return ''.join(parts)


def merge_originals(originals_dir='originals'):
    """Extract and merge all translation outputs (raw + continuations).
    Returns list of translated lines."""
    import os
    originals_dir = _resolve_path(originals_dir)
    all_lines = extract_divs(f'{originals_dir}/raw_output.txt')

    # Find continuation files
    cont_files = sorted(
        f for f in os.listdir(originals_dir)
        if f.startswith('continuation_') and f.endswith('.txt')
    )

    for cont_file in cont_files:
        cont_lines = extract_divs(f'{originals_dir}/{cont_file}')
        if not cont_lines:
            continue
        # Simple overlap detection
        merged = False
        for overlap_len in range(min(5, len(all_lines)), 0, -1):
            if all_lines[-overlap_len:] == cont_lines[:overlap_len]:
                all_lines.extend(cont_lines[overlap_len:])
                merged = True
                break
        if not merged:
            all_lines.extend(cont_lines)

    return all_lines
'''
