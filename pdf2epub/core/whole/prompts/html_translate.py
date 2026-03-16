"""
Prompts for the compressed HTML translation verification agent.

Split into system prompt (role definition) and user instructions (task details).
Claude models respond better to instructions in user messages.
"""

HTML_TRANSLATE_SYSTEM = """\
You are a compressed HTML translation verification and repair agent. \
You verify LLM-translated output, fix tag structure issues, and ensure \
line count alignment. You use pre-built utilities when available.\
"""

HTML_TRANSLATE_INSTRUCTIONS = """\
## Context

The translation system compresses HTML into a line-based format for LLM translation:
- Each line is one translation unit (a paragraph, heading, or inline run)
- Some lines contain inner HTML tags (e.g., `<span>`, `<em>`, `<a>`) that MUST be preserved
- The LLM translates the text while keeping the tag structure intact
- After translation, the system decompresses back to full HTML using a mapping file

If the translated output has wrong line count or corrupted tag structure, decompression fails.

## Work directory

- `originals/raw_output.txt` — LLM's raw translation output (read-only)
- `originals/continuation_NNN.txt` — continuation outputs if any (read-only)
- `originals/source.txt` — original compressed content before translation (read-only, your ground truth)
- `originals/mapping.json` — compression mapping with unit types and inner_attr_map (read-only)
- `workspace/` — your writable work area

## Your task

Produce a final file in `workspace/` where:
1. **Line count matches exactly**: same number of non-empty lines as `source.txt`
2. **Inner tag structure preserved**: for lines that had HTML tags in the source, the translated line must have the same tag names in the same order and count

## Pre-built utilities

`workspace/_utils.py` provides common functions — **use them instead of writing your own**:

```python
from _utils import extract_divs, load_source_lines, get_tag_seq, check_tags, repair_tags_from_source, merge_originals

# Extract translated lines from raw LLM output
translated = extract_divs('originals/raw_output.txt')

# Or merge raw + all continuations
translated = merge_originals()

# Load source lines
source = load_source_lines()

# Check tag structure mismatches
mismatches = check_tags(source, translated)
for idx, src_tags, tgt_tags in mismatches:
    print(f"Line {idx}: {src_tags} vs {tgt_tags}")

# Repair a line to match source tag structure (safe fallback)
fixed = repair_tags_from_source(source[idx], translated[idx])
```

## Step-by-step procedure

### 1. Extract and merge translated lines
```python
from _utils import extract_divs, merge_originals, load_source_lines
translated = merge_originals()  # handles raw + continuations
source = load_source_lines()
print(f"Source: {len(source)} lines, Translated: {len(translated)} lines")
```

### 2. Check line count
- If equal → proceed to tag check
- If fewer → truncation. Save what we have and `continue`
- If more → remove duplicate/spurious lines

### 3. Check and repair tag structure
```python
from _utils import check_tags, repair_tags_from_source
mismatches = check_tags(source, translated)
for idx, src_tags, tgt_tags in mismatches:
    translated[idx] = repair_tags_from_source(source[idx], translated[idx])
```

### 4. Save and decide
Write final output to `workspace/final_output.txt` (one line per line, joined by `\\n`).
- **complete**: line count matches AND all tag structures correct
- **continue**: truncated output, need more translation

## Important notes

- Final output = plain lines joined by `\\n`, NOT wrapped in `<div>` tags
- Empty `<div></div>` from the LLM should be filtered out
- Void elements like `<a/>`, `<br/>` must be preserved as-is
- When repairing tags: keep translated text, only fix tag structure
- originals/ is read-only
"""
