# Novel Translation v4: Haiku + Agent Loop + Glossary Memory

## Background

v3 used Murasaki-14b (a fine-tuned 14B model) with sliding window chunking. This caused:
- CoT token overhead consuming output budget → silent truncation
- Context line duplication at every chunk boundary (model re-translates multi-turn context)
- Fragile prompt format (术语表 format mismatch → model echoes glossary and stops)
- Unable to handle structured output (TOC/metadata) → needed separate model

v4 drops Murasaki entirely. Uses Haiku (200k context) for everything — translation, glossary, metadata. Reuses the existing `run_agent_loop_sync` continuation system from the HTML translation pipeline.

## Architecture Overview

```
Extract EPUB → translate_metadata (Haiku) → for each chapter:
  1. recall glossary (long-term + short-term)
  2. translate entire chapter (Haiku, via run_agent_loop_sync)
     - agent checks line count via head/tail (not full read)
     - if truncated → continuation
     - if preamble → strip
  3. extract glossary from translated chapter (Haiku, cache hit — same prefix)
  4. update long-term glossary store
→ convert txt→xhtml → HTMLEpubBuilder
```

## Component 1: GlossaryManager (new module)

**File**: `pdf2epub/html_translation/glossary_manager.py`

Independent module, not coupled to any translation flow. Can be reused by PDF/EPUB pipelines for cross-chapter concept alignment (e.g., legal/academic texts).

### Storage Format (`glossary_store.json`)

```json
{
  "宮崎薰": {
    "zh_name": "宫崎薰",
    "aliases": ["スミレ", "すみれ"],
    "description": "女主角，星蘭高中二年级，与主角岛崎苍同班。性格内向，只能对特定的人说出十个字以内的话。",
    "updated_by": "010_c75",
    "timestamp": "2026-03-24T02:00:00",
    "history": [
      {
        "description": "女主角，星蘭高中学生，与主角同班",
        "updated_by": "008_c2V",
        "timestamp": "2026-03-24T01:50:00"
      }
    ]
  }
}
```

Notes on storage:
- **No cap on entries or history** — a multi-million-word web novel may have thousands of terms, this is fine. Recall is filtered by exact match against chapter text, so injection volume is naturally bounded.
- `updated_by` and `timestamp` are managed by code, not by Haiku.
- History is append-only: when a term's description changes, the old version is pushed to `history` with the chapter that last set it.

### Interface

```python
class GlossaryManager:
    def __init__(self, output_dir: Path, llm_client, model: str, max_tokens: int = 1000):
        self.store_path = output_dir / "glossary_store.json"
        self.prev_chapter_path = output_dir / "glossary_prev.txt"
        self.log_dir = output_dir / "logs" / "glossary"
        self.store: Dict[str, dict] = {}  # long-term
        self.prev_chapter: str = ""        # short-term

    def load(self):
        """Load store and prev_chapter from disk. Tolerant of missing files."""

    def save(self):
        """Persist store and prev_chapter to disk."""

    def recall(self, source_text: str) -> str:
        """Recall relevant glossary entries for a chapter.

        1. Exact string match all keys + aliases against source_text
        2. Format matched entries as natural language for prompt injection
        3. Append prev_chapter (short-term) if available
        4. Return combined glossary string for system prompt
        """

    def extract_and_update(self, source_text: str, translated_text: str, chapter_id: str) -> str:
        """Generate glossary from completed translation, update store.

        1. Call Haiku with source + translated text
           (cache hit guaranteed: same system prompt + source text prefix as translation call)
        2. Haiku outputs JSON array — parsed via existing parse_llm_json (utils/common.py)
           which handles markdown fences, json_repair, etc.
        3. Compress if > max_tokens (re-call Haiku to prioritize)
        4. Update store: merge new entries, push old description to history
        5. Set prev_chapter = this chapter's glossary summary
        6. Save per-chapter log to logs/glossary/{chapter_id}.json
        7. Return the glossary text (for logging)
        """
```

### Haiku Glossary Extraction Prompt

```
你是轻小说术语表管理器。根据以下已完成的翻译，提取/更新术语表。

输出JSON数组，每个条目：
- key: 日文全名（越完整越好，如"宮崎薰"而非"宮崎"）
- zh_name: 中文翻译名
- aliases: 该角色/术语的其他日文称呼（昵称、简称等）
- description: 简要描述，包括：身份、与主角关系、当前状态

规则：
- 只收录重要角色、地名、专有名词
- description 控制在一两句话
- 如果和已有术语表有冲突，以本章翻译为准

已有术语表（供参考，可更新）：
{existing_entries}

本章原文：
{source_text}

本章译文：
{translated_text}
```

### Recall Format (injected into translation prompt)

```
术语表：
宮崎薰 → 宫崎薰（别名：スミレ/堇）：女主角，星蘭高中二年级...
島崎蒼 → 岛崎苍：男主角，...
高千穂弥生 → 高千穂弥生：...

上一章术语：
...（prev_chapter 内容，≤1k tokens）
```

## Component 2: Novel Translation (rewrite novel_translator.py)

### Simplified Flow

```python
class NovelTranslator:
    def __init__(self, config, book_title, output_dir, glossary_manager, ...):
        self.glossary_manager = glossary_manager

    def translate_all(self, units: List[NovelUnit]) -> dict:
        for unit in content_units:
            source_text = unit.text_path.read_text()

            # 1. Recall glossary
            glossary_prompt = self.glossary_manager.recall(source_text)

            # 2. Translate entire chapter via agent loop
            translated = self._translate_chapter(unit, glossary_prompt)

            # 3. Post-process: repair image placeholders
            translated = self._repair_images(source_text, translated)

            # 4. Extract glossary from completed translation (cache hit)
            self.glossary_manager.extract_and_update(source_text, translated, chapter_id)

    def _translate_chapter(self, unit, glossary_prompt) -> str:
        """Translate using run_agent_loop_sync with continuation support."""

        def generate_fn(prefix=None):
            if prefix is None:
                return llm_client.generate(prompt=user_prompt, ...)
            else:
                return llm_client.generate(prompt=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": prefix},
                    {"role": "user", "content": "继续翻译，从上次截断处接着。"},
                ], ...)

        # content_validator needs source_text — use closure
        source_text = unit.text_path.read_text()
        def content_validator(result_text: str) -> Optional[str]:
            return validate_novel_content(result_text, source_text)

        return run_agent_loop_sync(
            generate_fn=generate_fn,
            system_prompt=system_prompt_with_glossary,
            agent_model=agent_model,
            max_continuations=10,
            content_validator=content_validator,
            extra_originals={"source.txt": source_text},
            user_instructions=NOVEL_TRANSLATE_INSTRUCTIONS,
        )
```

### Input Length Guard

Before translating, check if chapter exceeds reasonable length:

```python
tokens = count_tokens(source_text)
if tokens > 50_000:
    raise ValueError(
        f"Chapter {unit.file_name} is {tokens} tokens — too long. "
        f"Please split the chapter before translating."
    )
```

This is a fail-fast for user error (unsplit input), not our problem to solve.

### Image Placeholder Repair (post-processing)

Image placeholders (`[Image: filename.jpg]`) pass through translation. The model usually preserves them, but may occasionally drop or hallucinate them. Repair after translation, before writing output:

```python
def _repair_images(self, source_text: str, translated_text: str) -> str:
    """Ensure image placeholders match source exactly."""
    IMAGE_RE = re.compile(r'\[Image:\s*[^\]]+\]')

    source_images = {}  # {line_index: placeholder_text}
    for i, line in enumerate(source_text.splitlines()):
        m = IMAGE_RE.search(line)
        if m:
            source_images[i] = m.group(0)

    translated_lines = translated_text.splitlines()

    # Remove hallucinated image placeholders (not in source)
    source_filenames = {img for img in source_images.values()}
    for i, line in enumerate(translated_lines):
        m = IMAGE_RE.search(line)
        if m and m.group(0) not in source_filenames:
            translated_lines[i] = IMAGE_RE.sub('', line).strip()

    # Re-insert missing image placeholders at original line positions
    translated_images = set()
    for line in translated_lines:
        m = IMAGE_RE.search(line)
        if m:
            translated_images.add(m.group(0))

    for line_idx, placeholder in source_images.items():
        if placeholder not in translated_images:
            # Insert at same line position (clamped to bounds)
            insert_at = min(line_idx, len(translated_lines))
            translated_lines.insert(insert_at, placeholder)

    return '\n'.join(translated_lines)
```

This is deterministic — doesn't depend on model behavior for correctness.

### What Gets Deleted

- Cursor/chunk/sliding window logic (`_take_chunk`, cursor state)
- Multi-turn context (`recent_ja`, `recent_zh`)
- `strip_thinking` (Haiku doesn't output `<think>`)
- `_detect_overlap` / dedup logic
- `glossary_to_murasaki_format` (Haiku doesn't need special format)
- `GlossaryOverflowError` handling in translator (moved to GlossaryManager)
- `_generate_glossary` / `_compress_glossary` (moved to GlossaryManager)
- `_build_translated_metadata` (already deleted in v3)
- `NovelState.cursor` field (no longer needed)
- `clean_glossary`, `extract_glossary_ids`, `glossary_value_to_translation` (v3 glossary utils)

### What Stays

- `NovelState` (current_unit_index, completed_units, for resume)
- `NovelUnit` dataclass
- `NovelExtractor`
- SIGTERM handler for state persistence
- Per-chapter glossary logging

## Component 3: Agent Workspace Utilities (new, plain-text version)

**File**: `pdf2epub/core/whole/prompts/novel_translate.py`

The existing `_utils.py` for HTML agent uses `extract_divs` (div-wrapped output parsing). Novel output is plain text — need a separate set of utils.

### Novel `_utils.py` (injected into agent workspace)

```python
def merge_originals():
    """Merge raw_output.txt + continuation_NNN.txt files into a single list of lines."""
    import pathlib, re
    lines = []
    raw = pathlib.Path('originals/raw_output.txt')
    if raw.exists():
        lines = [l for l in raw.read_text().splitlines() if l.strip()]
    # Merge continuations in order
    cont_files = sorted(pathlib.Path('originals').glob('continuation_*.txt'))
    for f in cont_files:
        cont_lines = [l for l in f.read_text().splitlines() if l.strip()]
        lines.extend(cont_lines)
    return lines

def load_source_lines():
    """Load source text lines."""
    import pathlib
    return [l for l in pathlib.Path('originals/source.txt').read_text().splitlines() if l.strip()]

def line_count(path):
    """Count non-empty lines in a file."""
    import pathlib
    return len([l for l in pathlib.Path(path).read_text().splitlines() if l.strip()])

def head(path_or_lines, n=5):
    """Return first n non-empty lines."""
    if isinstance(path_or_lines, list):
        return path_or_lines[:n]
    import pathlib
    lines = [l for l in pathlib.Path(path_or_lines).read_text().splitlines() if l.strip()]
    return lines[:n]

def tail(path_or_lines, n=10):
    """Return last n non-empty lines."""
    if isinstance(path_or_lines, list):
        return path_or_lines[-n:]
    import pathlib
    lines = [l for l in pathlib.Path(path_or_lines).read_text().splitlines() if l.strip()]
    return lines[-n:]
```

### Agent System Prompt + Instructions

```python
NOVEL_TRANSLATE_SYSTEM = """\
You are a novel translation verification agent. \
You verify translated output, check line count alignment, \
strip preamble text, and handle truncation.\
"""

NOVEL_TRANSLATE_INSTRUCTIONS = """\
## Context

The translation system feeds an entire chapter of Japanese text to an LLM.
The LLM translates line-by-line (one output line per input line).
The output may be truncated (chapter too long for output window) or have preamble.

## Work directory

- `originals/raw_output.txt` — LLM's raw translation output (read-only)
- `originals/continuation_NNN.txt` — continuation outputs if any (read-only)
- `originals/source.txt` — original Japanese text (read-only, ground truth)
- `workspace/` — your writable work area

## Your task

Produce a final file in `workspace/` where:
1. **No preamble**: remove any lines before the actual translation
   (e.g., "下面是翻译：", "以下为中文译文：", "Translation:" etc.)
2. **Line count alignment**: aim for same number of non-empty lines as source

## Procedure

### 1. Count lines and merge (DO NOT read full files)
```python
from _utils import merge_originals, load_source_lines
translated = merge_originals()
source_count = len(load_source_lines())
print(f"Source: {source_count}, Translated: {len(translated)}")
```

### 2. Check for preamble (read only first 5 lines)
```python
from _utils import head
for i, line in enumerate(head(translated, 5)):
    print(f"  {i}: {line[:80]}")
```
If first line(s) are meta-commentary (not translation), remove them.

### 3. Check for truncation (read only last 10 lines of each)
```python
from _utils import tail, load_source_lines
source_tail = tail('originals/source.txt', 10)
trans_tail = tail(translated, 10)
for s, t in zip(source_tail, trans_tail):
    print(f"  SRC: {s[:60]}")
    print(f"  TGT: {t[:60]}")
```

### 4. Decide
- Line count matches (within tolerance of 3) → save to workspace/ → **complete**
- Translated lines < source lines → truncated → save what we have → **continue**
- Translated lines > source lines → likely preamble or duplication → fix → **complete**

## Important
- Do NOT read entire source or translated files — use line counts and head/tail only
- originals/ is read-only, write to workspace/
"""
```

## Component 4: Content Validator

```python
def validate_novel_content(result_text: str, source_text: str) -> Optional[str]:
    """Returns None if valid, error string if not.

    Note: run_agent_loop_sync expects ContentValidator = Callable[[str], Optional[str]]
    (single arg). The caller must use a closure to capture source_text.
    """
    source_lines = len([l for l in source_text.splitlines() if l.strip()])
    result_lines = len([l for l in result_text.splitlines() if l.strip()])

    if result_lines == 0:
        return "Empty translation output"

    diff = abs(result_lines - source_lines)
    if diff <= 3:
        return None  # acceptable

    if result_lines < source_lines:
        return f"Truncated: {result_lines}/{source_lines} lines. Continue translating."

    # More lines than source — possible preamble or duplication
    return f"Line count mismatch: {result_lines} translated vs {source_lines} source. Check for preamble or duplicated lines."
```

## Component 5: CLI Changes

`translate_novel_command` simplified:

```python
def translate_novel_command(args):
    # 1. Extract EPUB
    parser = EPUBParser(epub_path)
    extractor = NovelExtractor(parser)
    units = extractor.extract_all(novel_units_dir)

    # 2. Translate metadata (Haiku, via novel.agent provider config)
    pipeline = HTMLEpubPipeline(epub_path, output_dir, metadata_config)
    translated_metadata = pipeline.translate_metadata(target_language=target_language)

    # 3. Init GlossaryManager (uses same Haiku model)
    glossary_manager = GlossaryManager(output_dir, llm_client, model="claude-haiku-4-5-20251001")
    glossary_manager.load()

    # 4. Translate chapters
    translator = NovelTranslator(
        config=config,
        book_title=epub_title,
        glossary_manager=glossary_manager,
        output_dir=output_dir,
        ...
    )
    summary = translator.translate_all(content_units)

    # 5. Build EPUB (same as v3: txt→xhtml via _convert_txt_to_xhtml, then HTMLEpubBuilder)
    ...
```

## Config Changes

```yaml
title: "その10文字を僕は忘れない"
credentials:
  providers:
    anthropic:
      type: anthropic
      api_key: ...
      base_url: ...

translation:
  source_language: Japanese
  target_language: Chinese
  models:
    - provider: anthropic
      model: claude-haiku-4-5-20251001

novel:
  glossary_max_tokens: 1000
  # No more: chunk_max_tokens, context_lines, extra_body, agent section
```

## JSON Parsing

Glossary extraction returns JSON. Use existing `parse_llm_json` from `pdf2epub/utils/common.py` which already handles:
- Markdown fence stripping (```json ... ```)
- `json_repair` for minor syntax issues
- Graceful error messages

No need to reinvent JSON parsing. Same function used by TOC translation, refine pipeline, etc.

## Migration from v3

### Files to rewrite
- `novel_translator.py` — remove chunking/sliding window, integrate GlossaryManager + agent loop
- `novel_prompts.py` — new prompts for Haiku translation

### Files to create
- `glossary_manager.py` — GlossaryManager class
- `core/whole/prompts/novel_translate.py` — agent system prompt + instructions + _utils

### Files to keep as-is
- `novel_extractor.py` — no changes
- `builder.py` — no changes (HTMLEpubBuilder reused)
- `cli.py` — minor changes (init GlossaryManager, pass to translator)

### Files to delete
- Nothing new to delete (novel_builder.py already deleted in v3)

## Testing Strategy

### Unit tests
- GlossaryManager: recall with key match, alias match, no match
- GlossaryManager: extract_and_update with mock Haiku response (via parse_llm_json)
- GlossaryManager: history versioning — update existing entry, verify old pushed to history
- GlossaryManager: recall returns empty when no terms match
- GlossaryManager: load/save round-trip
- Content validator: truncation detection, preamble detection, exact match, empty output

### Integration tests
- Extract → translate (mocked Haiku) → glossary extract → verify glossary store updated
- Resume from interrupted state (NovelState persistence)
- Continuation on truncated output (mocked generate_fn returns partial)

### End-to-end
- Translate sono10moji with --limit 5, verify:
  - translated_metadata.json has correct title + TOC (via Haiku, no think residue)
  - glossary_store.json has entries with aliases and history
  - c2V and c75 translated with correct line count (no duplication)
  - No preamble in output
  - EPUB builds correctly with images preserved (not squished)
  - Output dir uses original Japanese title

## Success Criteria

1. Full book translation completes with zero failures
2. Line count diff ≤ 3 per chapter (no duplication, truncation handled by continuation)
3. Glossary entries persist across chapters with version history
4. Character names consistent across chapters (verified by glossary alias recall)
5. No `<think>` residue, no preamble in output
6. EPUB has correct title, TOC, metadata, images preserved with original styling
7. Cache hit on glossary extraction calls (verified in logs: prompt_tokens should show cached)
