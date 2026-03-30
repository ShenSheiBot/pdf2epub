# Novel Translation Pipeline v2 Audit Criteria

## Environment preconditions

- Use a clean temporary directory for test output so `output/` artifacts do not interact with prior runs; delete `output/<title>/` before each full run.
- Prepare a test config with `title`, `translation.models`, and `novel` entries, including `novel.agent.provider`/`novel.agent.model` and valid fallback providers in `credentials.providers`.
- Prepare at least two EPUB fixtures:
  - one small valid EPUB with Japanese text and multiple spine items,
  - one EPUB with image-only spine items and non-text content.
- Install deps and run tests inside an isolated virtual env (`.venv` or `uv` environment).
- For automated tests, mock both translation and glossary API clients unless you have a controlled staging API; real API calls are for manual smoke tests only.

## User scenario checks

- `pdf2epub/cli.py` `translate-novel` must support all documented behaviors:
  - `-i` present uses that EPUB directly,
  - missing `-i` falls back to `output/<title>/input.epub`,
  - missing title in config fails fast with a clear error,
  - `--limit` restricts processing to the first N content units,
  - `--glossary` copies the external file to `output/<title>/glossary.txt`,
  - `--skip-build` runs translation only and does not call EPUB packager.
- The command must reject missing input files immediately and return non-zero instead of creating partial output silently.
- Extraction must read EPUB files in parser spine order and emit per-spine text files in `output/<title>/novel_units`, preserving filename-to-unit mapping for later rebuild (`pdf2epub/html_translation/novel_extractor.py`).
- Extraction must convert ruby and image tags into the expected plain format (ruby to `base(reading)`, images to `[Image: ...]`) and write one file per spine item in output units (`pdf2epub/html_translation/novel_extractor.py`).
- `NovelExtractor` must not mark image-only units with decorative text as translatable content; those units should still be preserved so builder can place placeholders in output EPUB.
- `NovelTranslator.translate_all(...)` must process units sequentially in extracted order and update stats/summary only from actual execution path.
- Per chapter, the translator must build glossary context before translation, not after chunking, and must pass reading-order context across chapters via persisted state (`pdf2epub/html_translation/novel_translator.py`).
- Per chapter glossary generation must:
  - load accumulated glossary state,
  - extract exact IDs from stored entries in the format `【...】`,
  - select relevant terms only when ID text appears in chapter text,
  - invoke the glossary model through `AnthropicClient` with config-driven provider/model (`novel.agent.provider/model`),
  - append/update accumulated glossary with deduplicated ID keys.
- If glossary text exceeds token budget, glossary compression must repeat up to 3 times; on 4th exceedance, translation must stop with a hard failure (no “best effort” silent overflow).
- Glossary generation failures after retry exhaustion must propagate an exception and abort translation; no partial success marker should report success.
- Sliding-window translation must use chunking by token budget and context window settings from `novel` config, and translation messages must be built in strict multi-turn format:
  - system message carries glossary (and no glossary duplication in user messages),
  - user/assistant replay only provides recent source+translated context,
  - first turn includes only source chunk.
- For each chunk, translator must:
  - call the translation model with `messages` (not prompt concatenation only),
  - strip `<think>...</think>` blocks from output,
  - append translated lines to chapter output incrementally,
  - persist partial chapter progress after each chunk.
- On any chunk with `finish_reason == 'length'`, translator must trim potentially incomplete final line before persisting/appending.
- After chapter complete, translator must run `LineCountValidator` in screener mode against original vs translated chapter and only log warning/retry strategy guidance, never hard-fail the whole chapter on merge/split differences.
- The chapter loop must handle empty lines/chunks safely (skip or advance cursor without throwing or spinning).
- `NovelState` persistence must include at least `current_unit_index`, `cursor`, glossary state, and completed-unit tracking needed for resume correctness; cursor must be updated per chunk (`pdf2epub/html_translation/novel_translator.py`).
- KeyboardInterrupt or process termination signal during translation must persist `NovelState` and allow restart from exact cursor and glossary context.
- `--resume` must resume from saved state without reprocessing completed units and continue from saved cursor when mid-chapter; completed units must not be duplicated.
- Resume should still behave correctly if `novel_state.json` exists but `glossary.txt` is missing; glossary should be reconstructed from state or continue gracefully without crash.
- The translation output directory must contain final chapter files in `output/<title>/translated_novel` with filenames matching extractor outputs (`pdf2epub/html_translation/novel_builder.py` input contract).
- Rebuild path (`translate-novel` → `build`) must produce a valid EPUB (`mimetype` stored, `content.opf`, `toc.ncx`, text entries) and preserve image placeholders as `<img>` elements (`pdf2epub/html_translation/novel_builder.py`).
- Builder fallback behavior must copy untranslated source text for chapters where translated file is missing, so output is complete and ordered.
- Large-book behavior: with huge chapters, translator must chunk through all lines without dropping or reordering lines.
- Edge content behavior: if a unit is tiny/trivial, it should still produce deterministic output and not depend on agent-trigger heuristics that are removed in v2.
- Failure path checks:
  - translation API retry exhaustion aborts immediately after max attempts,
  - glossary overflow/invalid format aborts with explicit reason,
  - missing/invalid generated content should not leave `current_unit_index` advanced past failed unit.

## Technical supplement

- Static checks (grep/read) to enforce v2 design:
  - `grep` in `pdf2epub/html_translation/novel_translator.py` for `run_agent_loop_sync`, `NOVEL_AGENT_SYSTEM`, `NOVEL_AGENT_INSTRUCTIONS`, `agent` workflow imports; none should remain in the novel path.
  - `grep` for `_generate_glossary`, `_compress_glossary`, `_extract_glossary_ids` (or equivalent clearly named methods) and assert they exist.
  - verify `pdf2epub/core/whole/prompts/novel_agent.py` is deleted and not imported anywhere.
  - verify `LineCountValidator` is used in chapter-end validation path and not as a hard-stop.
  - verify `NOVEL_TRANSLATE_SYSTEM` and glossary prompt builders remain in `pdf2epub/html_translation/novel_prompts.py`.
- Code quality checks:
  - `NovelState` schema should not regress to old agent-only fields; include new glossary-tracking fields required for continuity.
  - ensure CLI wiring still calls `NovelTranslator`, `NovelExtractor`, `NovelBuilder`, and preserves `translate-novel` options.
- Test quality checks:
  - unit tests for glossary parsing (`【...】` extraction) and ID-based relevance filtering,
  - unit tests for token budget compression loop and stop-on-3-attempt overflow,
  - unit tests for chunking cursor updates and finish-reason truncation behavior,
  - unit tests for resume from mid-chunk and `KeyboardInterrupt` persistence,
  - integration test for a 2–3 chapter EPUB end-to-end with `--limit`, `--glossary`, `--skip-build`, and full build path,
  - assert that `LineCountValidator` mismatch is non-blocking and only warning-level for chapter-level validation.
- Run and report:
  - `pytest -q tests/test_*.py` (or at minimum the newly added novel test module) before sign-off.