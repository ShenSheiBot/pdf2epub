# Novel Translation Pipeline v3 Audit Criteria

## Environment preconditions

- Prepare a disposable test workspace (no reuse of prior `output/` contents); create fresh per run:
  - remove `output/<title>/` before each full pipeline test.
- Prepare a valid config in temp path with:
  - `title`, `translation.models`, `novel.chunk_max_tokens`, `novel.context_lines`, `novel.glossary_max_tokens`.
  - `novel.agent.provider` and `novel.agent.model` so glossary generation can be exercised without ambiguity.
  - valid credentials section entries for referenced providers (or stable mocks in tests).
- Prepare at least three EPUB fixtures:
  - EPUB A: multi-spine, mixed block structure (`<p>`, `<h1>`, nested tags), ruby tags, and `head` content for style references.
  - EPUB B: valid TOC with real chapter titles and at least one deeper TOC entry (nested anchor/href).
  - EPUB C: image-heavy or image-only units and spine entries without textual content.
- Prepare a “limit” fixture where first two chapters are translatable and remaining are nontrivial content for `--limit` checks.
- For API-dependent checks, predefine deterministic mocks for glossary and translation clients; keep one run path with mocked failures (timeouts, malformed responses) and one path with valid responses.
- Ensure command test runner environment has pytest available and can execute CLI entrypoint (`pdf2epub`).

## User scenario checks

- User runs `translate-novel` with a valid `--input` EPUB:
  - command should honor explicit input path, parse config title, and create `output/{title}` structure before processing.
  - failure path: missing `--input` should load `output/{title}/input.epub` if present; if missing both, must return non-zero and not proceed.

- User starts `translate-novel` without `--glossary` on a clean output directory:
  - pipeline should initialize translation state cleanly, translate chapters sequentially by spine order, and produce `translated_novel` files mapped to extractor units.

- User runs with an invalid `--glossary` path:
  - command should fail fast, return non-zero, and avoid partial translation/build side effects.

- User runs `--limit` on a large EPUB:
  - only first N content units should be passed to translator; skipped units should still be structurally represented in final EPUB build through fallback logic.
  - failure path: ensure translation output count equals requested limit, not total units.

- User passes `--skip-build`:
  - translation should complete and write translated text, state, and glossary artifacts, but must not invoke EPUB rebuild.
  - failure path: if translation fails mid-run, `--skip-build` must not hide build attempts or create a false-success EPUB.

- User runs `translate-novel` with interruption/resume behavior:
  - `--resume` should load `novel_state.json` and continue from exact unit/cursor position, not re-translate completed units.
  - if run is interrupted mid-chapter, next run should continue from saved cursor with glossary state aligned.

- User provides EPUB with rich block structure:
  - `pdf2epub/html_translation/novel_extractor.py` must emit `novel_units/*.txt` with one logical line per block boundary so paragraph boundaries are preserved.
  - failure path: no paragraph flattening where two separate `<p>/<h1>/<div>` blocks collapse onto one line.

- User includes images and ruby text in source units:
  - `novel_extractor` should preserve ruby as reading format (e.g., `経緯(いきさつ)`) and image placeholders as `[Image: xxx]`.
  - failure path: image-only units should not be dropped; they must remain in unit output and be represented during rebuild.

- User sees glossary contamination in assistant context in prior versions:
  - `strip_thinking` must remove all prior model reasoning tails by taking text after the last `</think>` and dropping preceding reasoning residue.
  - failure case: malformed output containing only `<think>...` with no closing tag must not leak stale text into translation.
  - failure case: orphan `</think>` before real translation must not corrupt result; translated text after closing tag must be kept.

- User expects glossary consistency chapter over chapter:
  - `_generate_glossary` must clean raw model output with `clean_glossary`-style filtering before update/usage so only bracketed lines (`【...】`) are accumulated and reused.
  - failure case: raw model remarks, markdown bullets, headings, or comments must not enter the glossary passed to subsequent chapters.

- User expects per-chapter glossary persistence:
  - `novel_translator` should write each chapter glossary to `output/{title}/logs/glossary/{idx:03d}_{name}.txt` (or equivalent deterministic unit filename key).
  - failure case: previously, only final chapter glossary remained; now should retain per-chapter debug artifacts.

- User expects cumulative glossary snapshot:
  - `output/{title}/logs/glossary/accumulated.txt` should be written after run (or on successful completion) as a complete snapshot of accumulated entries used for translation consistency.

- User expects chapter-title consistency:
  - per chapter, title from EPUB TOC/spine mapping must be injected into glossary context as a `【原文标题】...` entry.
  - failure case: if TOC missing, fallback strategy should still produce deterministic per-unit glossary and not crash translation.

- User expects title/TOK translation consistency between body and TOC:
  - `translate-novel` pipeline must generate `output/{title}/translated_metadata.json` with:
    - `translated_title`
    - `toc: [{original, translated}, ...]`
  - `toc` entries should preserve order and map back to original EPUB units.

- User expects rebuilt EPUB metadata/title updates:
  - during build, `translated_metadata` must be passed into `HTMLEpubBuilder` and reflected in OPF/Toc files where supported.
  - failure path: missing translated_title or toc should not silently produce incomplete metadata updates without surfacing clear build diagnostics.

- User expects `.txt` to `.xhtml` conversion correctness:
  - converter should output one `.xhtml` per original spine XHTML file with preserved `<head>` (or copied style-link context) and one `<p>` per non-empty source line.
  - `[Image: xxx]` should convert to `<img src="xxx"/>` with original relative path semantics.
  - empty or placeholder-only inputs should still produce valid XHTML replacement candidates without blank-page corruption.

- User expects rebuild correctness:
  - only matching translated files should be replaced; untouched XHTML entries should remain byte-safe from original if no corresponding translated XHTML exists.
  - EPUB packaging should preserve required package files and structure (`mimetype`, container, manifest/spine consistency).

- User expects no stale output overwrite:
  - output EPUB should be deterministic with translated filenames and safe fallback behavior when partial translation exists.

- User expects removal of deprecated custom novel builder path:
  - `translate-novel` should not route through `novel_builder.py` workflow anymore; this path must be absent from CLI command logic.
  - failure path: if old builder references remain, build will retain old title/TOC behavior and fail v3 acceptance.

- User expects TOC/title mapping consistency check across TOC formats:
  - title extraction should work from EPUB toc entries and map to corresponding chapter content so `translated_metadata.json` and builder updates are not using `"Chapter 1"` placeholders.

## Technical supplement

- Static/API-call checks (grep/read)
  - `grep -n "strip_thinking" pdf2epub/html_translation/novel_translator.py` and verify implementation matches last-`</think>` behavior, not permissive pair-only regex.
  - `grep -n "clean_glossary" pdf2epub/html_translation/novel_translator.py` must return an implemented filter function and all glossary-read/write call sites should pass filtered output.
  - `grep -n "logs/glossary" pdf2epub/html_translation/novel_translator.py` must show per-chapter and accumulated writes to `output/{title}/logs/glossary/`.
  - `grep -R "novel_builder" pdf2epub` should not show references in `cli.py` or translator/build flow; `novel_builder.py` can be absent for this feature path.
  - `grep -R "HTMLEpubBuilder|BuildConfig|txt_to_xhtml" pdf2epub` should confirm `translate-novel` build path uses the shared HTML builder and a text→xhtml conversion step.
  - `grep -R "translate-metadata|translated_metadata|translated_title|toc" pdf2epub/html_translation` to ensure metadata file is generated and consumed in build.
  - `grep -R "core/whole/prompts/novel_agent.py\|NOVEL_AGENT" pdf2epub` to confirm no residual legacy agent wiring in novel flow.

- Test quality checks
  - Add/update unit tests in `tests/test_novel_translation.py` for:
    - orphan `</think>` and mixed think/translation outputs in `strip_thinking`.
    - `clean_glossary` filtering with noise lines and bracket-only entries.
    - per-chapter glossary file creation and `logs/glossary/accumulated.txt` snapshot.
    - title injection into per-chapter glossary from TOC extraction.
  - Add/update integration tests for:
    - extraction linebreak behavior for mixed block elements.
    - TOC-title-driven translation metadata generation (`translated_metadata.json`) and build-time TOC update path.
    - `--limit` + `--skip-build` and full build path with mixed translated/untranslated units.
    - interruption + `--resume` from mid-chapter cursor and completed units skip.
    - EPUB output from `translate-novel` build including OPF/NCX/nav updates when translated metadata is present.
  - Run:
    - `uv run pytest -q tests/test_novel_translation.py`
    - plus any newly added targeted tests for txt→xhtml conversion and translated_metadata/build integration.
---
## Goal

Finish the `translate-novel` pipeline migration from the v2 implementation to **novel-translation-v3**, where:

- extraction preserves block line breaks correctly,
- glossary is cleaned/filter-stored per chapter,
- per-chapter and accumulated glossary are persisted for debug,
- chapter title IDs from EPUB TOC are injected for terminology consistency,
- translated metadata (title + TOC) is generated and reused,
- EPUB rebuild uses the shared `HTMLEpubBuilder` instead of the custom `NovelBuilder`,
- text units are converted back into XHTML with `<head>` preserved and image placeholders restored.

---

## Instructions

- Follow the v3 plan in `docs/plans/novel-translation-v3.md` (and reconcile with prior v2 spec/tests as needed).
- Existing key constraints to keep:
  - `translate-novel` is currently documented to be text-only light-novel mode with glossary-based consistency.
  - Use existing v3-capable builder infrastructure in `pdf2epub/html_translation/builder.py` rather than bespoke novel builder logic.
  - Preserve existing behavior for resume/state handling and non-blocking chapter validation.
  - Keep tests in sync with implementation changes.
- Relevant plan requirements already stated by user:
  1. Extractor line break fix in `novel_extractor.py` (block elements as separate lines).
  2. `strip_thinking` should use last-`</think>` strategy.
  3. Glossary cleaned to pure `【...】` entries and saved per chapter.
  4. Chapter titles injected into glossary via TOC mapping.
  5. Build with `HTMLEpubBuilder` + `translated_metadata` updates to OPF/NCX/NAV.
  6. Delete/use-out `novel_builder.py` in translate-novel flow.
  7. Add `output/{title}/logs/glossary/{idx}_{name}.txt`, `accumulated.txt`, and `translated_metadata.json`.
- Non-code request from previous session: no code modifications were requested explicitly by user in this conversation; we were in discovery/investigation phase.

---

## Discoveries

- **Current v2 implementation status**:
  - `pdf2epub/html_translation/novel_translator.py` already has:
    - chunked per-line translation flow,
    - glossary generation with haiku via `GLOSSARY_GENERATE_PROMPT`,
    - resume state (`NovelState`) with `current_unit_index`, `cursor`, `prev_chapter_glossary`, `accumulated_glossary`, `completed_units`,
    - line-count validation via `LineCountValidator` in non-blocking warning mode.
  - But it does **not** yet implement v3-specific persistence and metadata flow.
- `strip_thinking` is still regex-based:
  - `_THINK_CLOSED_RE`, `_THINK_UNCLOSED_RE` and removal logic can fail on malformed thinking tags.
- Glossary filtering is currently missing:
  - `extract_glossary_ids` can parse bracketed terms for IDs, but `prev_glossary`, generated glossary, and persisted glossary are not sanitized to discard non-`【...】` output prior to passing onward.
- Glossary persistence currently only writes:
  - `output/{title}/glossary.txt` (single file), not per-chapter or accumulated snapshots in `logs/glossary/`.
- There is currently **no** `translated_metadata.json` generation in translator path; no build-time metadata handoff.
- `translate-novel` CLI path (`pdf2epub/cli.py`) still uses:
  - `NovelBuilder` from `pdf2epub/html_translation/novel_builder.py`.
- `pdf2epub/html_translation/builder.py` already includes:
  - `HTMLEpubBuilder`,
  - metadata-updating helpers for OPF/NCX/NAV via provided metadata,
  - `_replace_xhtml_files`,
  - and convenience `build_html_epub`.
- `novel_extractor.py` currently appears to already keep block elements separated (each block via `_collect_inline` then block append and later joined by `\n`), but this should be verified against the v3 requirement and test cases.
- `html_translation/__init__.py` already exports `HTMLEpubBuilder` / pipeline.
- No `core/whole/prompts/novel_agent.py` file exists in filesystem; docs indicate it should be gone and legacy references absent (good).
- Existing tests in `tests/test_novel_translation.py` are mostly v2-focused and still import/assert `NovelBuilder`; they’ll need updates for v3 behavior and new files/output artifacts.
- `.claude/audit-state/.../passed` indicates a prior run reported v2 implementation aligned with earlier audit; and `uv run pytest -q tests/test_novel_translation.py` passed 26 tests (historical, no new edits yet in this session).

---

## Accomplished

### Done
- Read and mapped:
  - user-provided v3 plan (`docs/plans/novel-translation-v3.md`),
  - related historical plans and audit criteria docs,
  - all relevant code paths:
    - `novel_extractor.py`,
    - `novel_translator.py`,
    - `novel_prompts.py`,
    - `cli.py`,
    - `epub_parser.py`,
    - `toc_extractor.py`,
    - shared HTML builder module (`html_translation/builder.py`),
    - test suite (`tests/test_novel_translation.py`).
- Verified current imports and references via grep:
  - `NovelBuilder` still wired in `translate-novel` CLI path.
  - `HTMLEpubBuilder` currently not used by `translate-novel`.
- Confirmed legacy novelty:
  - no active `novel_agent.py` file; old agent module absent.

### In progress / not done yet
- No code edits were actually applied in this session.
- No new tests were added/updated in this session.
- No build/translation run executed.

### Left to do next
1. Implement `strip_thinking` v3 behavior (`rfind("</think>")`).
2. Add glossary cleaning helper (pure `【...】` lines) and apply it:
   - before state updates,
   - before glossary passed in `system` prompt.
3. Add per-chapter glossary file output to:
   - `output/{title}/logs/glossary/{idx:03d}_{name}.txt`,
   - `output/{title}/logs/glossary/accumulated.txt`.
4. Inject TOC chapter title mappings into per-chapter glossary.
5. In translator, construct and persist `translated_metadata.json` from accumulated glossary title mappings.
6. Update `translate-novel` CLI:
   - remove `novel_builder` import/usage,
   - build a file map from EPUB TOC and translator units,
   - convert translated `.txt` to `.xhtml` (preserve original `<head>`),
   - build with `BuildConfig` + `HTMLEpubBuilder`.
7. Decide whether to delete or deprecate `novel_builder.py` path for novel flow.
8. Add/update tests for:
   - think stripping edge cases,
   - glossary filtering + per-chapter logs + accumulated snapshot,
   - TOC-title consistency,
   - metadata propagation + HTMLEpubBuilder integration,
   - CLI contract with `--skip-build`, `--limit`, missing inputs, and resume.
9. Run focused tests: `uv run pytest -q tests/test_novel_translation.py` plus any new v3 integration tests.

---

## Relevant files / directories

### Specs & planning
- `docs/plans/novel-translation-v3.md` (primary implementation target)
- `docs/archive/novel/plans/novel-translation.md` (v2 baseline + architecture)
- `docs/archive/audit/plans/audit-prompt.md` (audit criteria checklist)

### Novel extraction/translation core
- `pdf2epub/html_translation/novel_extractor.py`
- `pdf2epub/html_translation/novel_translator.py`
- `pdf2epub/html_translation/novel_prompts.py`

### EPUB parser / TOC
- `pdf2epub/html_translation/epub_parser.py`
- `pdf2epub/html_translation/toc_extractor.py`

### Builder / packaging
- `pdf2epub/html_translation/builder.py`
- `pdf2epub/html_translation/__init__.py`
- `pdf2epub/html_translation/novel_builder.py` (likely to be removed from translate-novel flow)

### CLI entrypoint
- `pdf2epub/cli.py` (`translate_novel_command`)

### Tests
- `tests/test_novel_translation.py` (needs v3 updates)
- `.claude/audit-state/ce691ff1-c977-4566-9aa6-736da9267273/` (historical audit notes/results)

### Config / runner
- `pyproject.toml` (dependencies/test runner settings)
Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.
## Goal

Migrate `translate-novel` to the v3 plan in `docs/plans/novel-translation-v3.md` so glossary handling is clean/filer-friendly, chapter titles are injected into glossary and reflected in translated metadata/TOC, and EPUB build is moved onto `HTMLEpubBuilder` with preserved XHTML heads + image placeholders restored.

## Instructions

- Primary execution target: `docs/plans/novel-translation-v3.md` and its ordered checklist.
- Preserve existing behavior where required:
  - Keep resume/state handling in translator.
  - Keep non-blocking `LineCountValidator` chapter validation warnings.
  - Keep existing CLI argument flow unless explicitly changed by v3 requirements.
- Key v3 requirements to keep enforcing:
  - `strip_thinking` should use “last `</think>` only”.
  - Glossary persistence should split into per-chapter logs + `accumulated.txt`.
  - Glossary passed to translator should be cleaned to pure `【...】` lines.
  - Chapter titles from TOC should be injected into glossary flow.
  - Build should reuse shared `HTMLEpubBuilder` and `translated_metadata` (title + TOC updates).
- If user confirms broader scope again, next concrete work should include both implementation + tests.

## Discoveries

- The repo already had v2 implementation mostly in place:
  - `novel_translator.py` already has `_generate_glossary`, `strip_thinking` (old regex version), state resume, and chapter validation.
  - `novel_extractor.py` already outputs one line per block via block/non-block merge logic; no explicit `novel_extractor` edits were completed yet in this run.
  - `cli.py` still used `NovelBuilder` from `pdf2epub/html_translation/novel_builder.py` in `translate_novel_command`.
- `html_translation/builder.py` already includes `HTMLEpubBuilder`, `BuildConfig`, and metadata update helpers (`_update_content_opf`, `_update_toc_ncx`, `_update_nav_xhtml`) that can be leveraged directly.
- `TOCExtractor` and `EPUBParser` expose enough TOC info for chapter title mapping, but metadata construction is not wired in `translate_novel_command`.
- Tests in `tests/test_novel_translation.py` are v2-focused and still import/expect `NovelBuilder`, plus old assumptions (single glossary file only, old `strip_thinking` behavior, etc.).
- I started editing `pdf2epub/html_translation/novel_translator.py` and introduced partial v3 behavior, but translator workflow integration is incomplete.

## Accomplished

### What was changed
- File modified: `pdf2epub/html_translation/novel_translator.py` (partial, in-progress):
  - Updated module docstring toward v3 wording.
  - Replaced regex-based `strip_thinking` with last-`</think>` strategy:
    - Uses `text.rfind("</think>")` and returns everything after it, else full stripped text.
  - Added glossary cleaning helpers:
    - `clean_glossary(raw)` to keep only lines containing `【...】`.
    - `glossary_value_to_translation(line)` to extract translated text after `=` for metadata mapping.
  - Added output paths in `__init__`:
    - `translated_metadata_path = output_dir / "translated_metadata.json"`
    - `glossary_log_dir = output_dir / "logs" / "glossary"` (created).
  - `_generate_glossary` updated (partially):
    - New optional parameter `chapter_title`.
    - Prompt now accepts `chapter_title_section`.
    - Applies `clean_glossary` to generated/compressed glossary text.
    - Falls back to cleaned previous glossary when no parseable entries.
  - `_persist_state` now writes cleaned glossary content and calls `_save_accumulated_glossary`.
  - Added methods:
    - `_target_language_code()`
    - `_save_per_chapter_glossary()`
    - `_save_accumulated_glossary()`
    - `_build_translated_metadata()`
- No changes were completed in `cli.py`, `novel_extractor.py`, or tests yet in this pass.
- No deletion of `novel_builder.py` yet.

### What is currently broken/incomplete (important)
- `translate_all` and `_translate_chapter` were not yet wired to:
  - pass chapter titles into glossary generation,
  - call per-chapter glossary logging,
  - write `accumulated.txt` at the right cadence,
  - generate/write `translated_metadata.json` after translation pass.
- `NovelUnit` currently lacks attributes referenced by new helper logic:
  - `source_href` and `toc_title` were referenced in `novel_translator.py` helper methods, but dataclass does not define them yet.
- Existing static diagnostics seen during edit:
  - `translation_models: list = None` type mismatch with `None` unless typed as `Optional[list]`.
  - `NovelUnit` unknown attributes (`toc_title`, `source_href`) in translator.
  - Optional field warnings around `unit.text_path`/`unit.file_name` downstream are pre-existing risk amplified by new usage paths.

## Next actions to continue

### Required code steps to finish v3
1. **Complete `NovelUnit` metadata**
   - Update `pdf2epub/html_translation/novel_extractor.py`:
     - add `source_href` and `toc_title` fields (optional defaults).
     - preserve existing constructor/usage compatibility.

2. **Finish translator integration**
   - In `pdf2epub/html_translation/novel_translator.py`:
     - update `translate_all` to optionally accept/use chapter title map (or trust `unit.toc_title`).
     - pass `chapter_title` into `_generate_glossary`.
     - after each chapter call `_save_per_chapter_glossary`.
     - after each save/update call `_save_accumulated_glossary`.
     - ensure final `translated_metadata` is generated via `_build_translated_metadata` and persisted.
     - normalize `translation_models: Optional[list]` for type clarity.

3. **Implement build pipeline migration in CLI**
   - Update `pdf2epub/cli.py` `translate_novel_command`:
     - replace `NovelBuilder` import/usage with `HTMLEpubBuilder + BuildConfig`.
     - derive toc title mapping from EPUB parser/`TOCExtractor`.
     - assign unit-level chapter titles for translator.
     - add `translated_novel_xhtml` conversion step:
       - convert translated `.txt` → `.xhtml`,
       - preserve original `<head>` from EPUB source files,
       - replace `[Image: ...]` with `<img src="..."/>`,
       - keep filenames aligned for `_replace_xhtml_files`.
     - pass `translated_metadata` into builder config and build EPUB.
     - keep `--skip-build` behavior.
   
4. **Align/remove legacy builder path**
   - Keep `novel_builder.py` file for compatibility only if needed externally, but remove `translate-novel` usage from CLI.
   - Update tests to no longer rely on `NovelBuilder` path in novel flow.

5. **Update tests**
   - `tests/test_novel_translation.py`:
     - add assertions for `strip_thinking` edge case with trailing open think (no close) handled by rfind logic.
     - add glossary cleaning/per-chapter/accumulated file expectations.
     - update integration/build tests to use `HTMLEpubBuilder` and translated metadata file path expectations.
     - preserve resume/state contract tests.

6. **Verification**
   - Run `uv run pytest -q tests/test_novel_translation.py` (or equivalent test runner), then iterate on failing assertions.

## Relevant files / directories

- `docs/plans/novel-translation-v3.md` (primary spec)
- `pdf2epub/html_translation/novel_translator.py` (partially modified; main work in progress)
- `pdf2epub/html_translation/novel_extractor.py` (likely needs `NovelUnit` enhancement)
- `pdf2epub/html_translation/builder.py` (target for shared builder integration)
- `pdf2epub/html_translation/toc_extractor.py` (for chapter title mapping)
- `pdf2epub/html_translation/novel_builder.py` (legacy builder, should be removed from translate flow)
- `pdf2epub/cli.py` (`translate_novel_command` rewrite pending)
- `tests/test_novel_translation.py` (must be brought in line with v3 behavior)
Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.
Maximum steps reached for this task.

What I completed before the limit:
- Reviewed the v3 plan (`docs/plans/novel-translation-v3.md`) and mapped required behaviors into implementation checks.
- Audited current `translate-novel` flow in code (`pdf2epub/cli.py`, `pdf2epub/html_translation/novel_translator.py`) and confirmed:
  - The command still builds via legacy `NovelBuilder`, not `HTMLEpubBuilder`.
  - Glossary/title metadata propagation is not fully wired end-to-end yet.
  - Several v3 behaviors are partially present in `novel_translator.py` (e.g., `strip_thinking` via last `</think>`, glossary cleaning/persistence helpers), but not fully integrated into chapter loop.
- Inspected tests and found `tests/test_novel_translation.py` still expects legacy `NovelBuilder` paths and old assumptions.
- Verified open worktree is already dirty with modified/untracked novel-related files from the partial refactor.

What remains unfinished (high-confidence gaps):
- `NovelUnit`/extractor integration:
  - Ensure chapter/title metadata is attached for each unit (`toc_title`, and any source href/file mapping needed for TOC metadata).
- Translator integration in `novel_translator.py`:
  - Pass chapter title into glossary generation.
  - Persist per-chapter glossary files under `logs/glossary/{idx}_{name}.txt`.
  - Update and persist `accumulated.txt` consistently.
  - Build and write `translated_metadata.json` from translated-title and TOC mapping.
- `translate_novel_command` in `cli.py`:
  - Replace `NovelBuilder` with `HTMLEpubBuilder` + `BuildConfig`.
  - Add TXT→XHTML conversion (preserve source `<head>`, map image placeholders back to `<img src="..."/>`, and align filenames).
  - Pass `translated_metadata.json` into build config and support `--skip-build`.
- Cleanup/compatibility:
  - Stop using `novel_builder.py` in the novel flow; keep file only if needed externally.
- Tests:
  - Update `tests/test_novel_translation.py` to cover v3 behaviors: glossary filtering, per-chapter/accumulated logs, strip-thinking edge cases, translated metadata, and HTMLEpubBuilder-based build output.
- Verification:
  - Run test suite for novel translation path and fix any regressions.

Recommended next step:
- If you want, I can continue and finish the refactor now in this order: `novel_extractor.py` metadata → `novel_translator.py` state/metadata wiring → `cli.py` HTMLEpubBuilder pipeline → tests + validation run.