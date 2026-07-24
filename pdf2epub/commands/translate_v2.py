"""
Translate V2 command - uses new Phase + Pipeline V2 architecture.

Key features:
- Uses Phase for composable stages
- Uses Pipeline V2 with Executor + Hooks
- Configuration-driven (reads from config.yaml)
- No intermediate aggregation
- No retry loops (re-queue instead)
- Batch + Online simultaneous execution (if batch client provided)
- TOC translation after content (needs translated titles as reference)
"""

import json
import re
from pathlib import Path
from typing import Any, List, Dict, Tuple, Union
from loguru import logger

from ..utils.common import load_config, parse_llm_json
from ..utils.logging_config import configure_logging
from ..utils.llm_client import LLMClient
from ..processors import TranslateProcessor
from ..core.phase import Phase, PartBasedLoader
from ..core.factory_v2 import (
    create_processing_pipeline_v2,
    get_task_model_configs,
)
from ..core.book_structure import BookStructure
from ..chapter_identity import ChapterIdentity


def translate_v2_command(args):
    """
    Handle the translate-v2 subcommand using new architecture.

    Configuration is read from config.yaml under:
    - translation.models: Model chain configuration
    - translation.source_language / target_language: Languages
    - translation.truncation_check_lines: Truncation detection
    - validation_v2.*: V2-specific configuration
    - validation_strategy.*: General validation settings
    """
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "translate_v2")

    # Get language settings from config (with CLI override)
    translation_config = config.get('translation', {})
    source_language = (
        getattr(args, 'source_language', None) or
        translation_config.get('source_language', 'Japanese')
    )
    target_language = (
        getattr(args, 'target_language', None) or
        translation_config.get('target_language', 'Chinese')
    )

    logger.info(f"[v2] Starting translation for: {book_title}")
    logger.info(f"[v2] Translation: {source_language} -> {target_language}")

    # Setup directories
    output_dir = Path("output") / book_title
    from pdf2epub.utils.network_utils import set_llm_trace_path
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")

    # Input can be polished (V2 validated or legacy) or merged pages
    # Priority: polished_markdown/validated > polished_markdown > pages_merged
    input_dir = output_dir / "polished_markdown" / "validated"
    if not input_dir.exists():
        input_dir = output_dir / "polished_markdown"
    if not input_dir.exists():
        input_dir = output_dir / "pages_merged"
    if not input_dir.exists():
        logger.error(f"No input directory found. Tried:")
        logger.error(f"  - {output_dir / 'polished_markdown' / 'validated'}")
        logger.error(f"  - {output_dir / 'polished_markdown'}")
        logger.error(f"  - {output_dir / 'pages_merged'}")
        return 1

    logger.info(f"[v2] Using input from: {input_dir}")

    translate_dir = output_dir / "translated"
    translate_dir.mkdir(parents=True, exist_ok=True)

    # Load book structure
    book_structure = BookStructure(output_dir)
    book_structure_data = None
    toc_tree_file = output_dir / "toc_tree.json"
    if toc_tree_file.exists():
        with open(toc_tree_file, 'r', encoding='utf-8') as f:
            book_structure_data = json.load(f)
        logger.info("Loaded book structure from toc_tree.json")

    # Create LLM client
    llm_client = LLMClient(config)
    translation_models = get_task_model_configs(config, "translate")

    # Determine use_entities
    if getattr(args, 'no_entities', False):
        use_entities = False
    elif getattr(args, 'use_entities', False):
        use_entities = True
    else:
        use_entities = None  # Auto-detect

    # Create the old TranslateProcessor (for its prompt/response logic)
    old_processor = TranslateProcessor(
        config=config,
        book_title=book_title,
        source_language=source_language,
        target_language=target_language,
        max_workers=1,  # Not used
        resume=False,   # Not used
        translation_models=translation_models,
        use_entities=use_entities,
        use_longest_on_failure=False,  # Not used
        book_structure=book_structure_data,
    )

    # TranslateProcessor now directly implements ProcessorProtocol
    processor = old_processor

    # Get configuration overrides from args
    max_workers = getattr(args, 'max_workers', None)
    # CLI --parallel overrides config; otherwise use config's processing_mode
    if getattr(args, 'parallel', False):
        sequential_mode = False
    else:
        # Check translation section first, then top-level
        translation_config = config.get('translation', {})
        processing_mode = translation_config.get('processing_mode', config.get('processing_mode', 'parallel'))
        sequential_mode = processing_mode != 'parallel'
    skip_validation = getattr(args, 'skip_validation', False)

    # Create Pipeline V2 using config-based factory
    pipeline = create_processing_pipeline_v2(
        processor=processor,
        output_dir=translate_dir,
        llm_client=llm_client,
        config=config,  # Pass full config for reading all settings
        book_structure=book_structure,
        task_type="translate",
        use_batch_validation=not skip_validation,
        sequential_mode=sequential_mode,
        max_workers=max_workers,
        restore_images=True,
    )

    # Create Phase
    phase = Phase(
        name="translate_v2",
        input_dir=input_dir,
        output_dir=translate_dir,
        pipeline=pipeline,
        loader=PartBasedLoader(),
        file_pattern="*.md",
    )

    # Run
    resume = getattr(args, 'resume', False)
    result = phase.run(resume=resume)

    # Report
    logger.info(f"[v2] Translation completed: {result.completed}/{result.total} succeeded")
    if result.failed > 0:
        logger.warning(f"[v2] Failed: {result.failed} files")
        for key in result.failed_keys[:10]:
            logger.warning(f"  - {key}")

    # Translate TOC titles (after content, needs translated titles as reference)
    if toc_tree_file.exists():
        logger.info("[v2] Translating TOC titles...")
        toc_translated = _translate_toc(
            output_dir=output_dir,
            translate_dir=translate_dir,
            llm_client=llm_client,
            translation_models=translation_models,
            source_language=source_language,
            target_language=target_language,
            config=config,
        )
        if not toc_translated:
            logger.error("[v2] TOC translation failed; translated content was preserved")
            return 1
        logger.success("[v2] TOC translation completed")

    return 0 if result.failed == 0 else 1


def _translate_toc(
    output_dir: Path,
    translate_dir: Path,
    llm_client: LLMClient,
    translation_models: List[Dict],
    source_language: str,
    target_language: str,
    config: Dict[str, Any],
) -> bool:
    """Translate and persist TOC while one process owns its lifecycle."""
    from ..core.executor.batch_state import BatchRunLock

    with BatchRunLock(output_dir / ".toc_translation_lifecycle"):
        return _translate_toc_locked(
            output_dir,
            translate_dir,
            llm_client,
            translation_models,
            source_language,
            target_language,
            config,
        )


def _translate_toc_locked(
    output_dir: Path,
    translate_dir: Path,
    llm_client: LLMClient,
    translation_models: List[Dict],
    source_language: str,
    target_language: str,
    config: Dict[str, Any],
) -> bool:
    """
    Translate TOC titles using already translated content as reference.

    Args:
        output_dir: Book output directory (contains toc_tree.json)
        translate_dir: Translation output directory (contains translated markdown)
        llm_client: LLM client for API calls
        translation_models: Model configurations
        source_language: Source language
        target_language: Target language
    """
    # Build reference JSON from toc_tree.json + translated markdown titles
    reference_json = _build_toc_reference_json(output_dir, translate_dir)
    if not reference_json:
        logger.warning("No TOC entries to translate")
        return False

    state_path = output_dir / "toc_translation_batch_state.json"
    translated_path = output_dir / "toc_tree_translated.json"
    if state_path.exists() and translated_path.exists():
        try:
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state_data = {}
        if state_data.get("job_state") == "FINALIZED":
            from ..core.executor import PersistedSingleRequestBatch
            from ..utils.batch_utils import BatchRequest

            batch_model = next(
                (
                    model
                    for model in translation_models
                    if model.get("mode") == "batch"
                ),
                None,
            )
            if not batch_model:
                logger.error(
                    "Finalized TOC batch state has no matching current "
                    "batch model; state was retained"
                )
                return False
            provider = batch_model.get("provider", "gemini")
            model = batch_model.get("model")
            request = BatchRequest(
                key="toc_translation",
                contents=[{
                    "role": "user",
                    "parts": [{
                        "text": _build_toc_translation_prompt(
                            reference_json,
                            source_language,
                            target_language,
                        )
                    }],
                }],
                config={"response_mime_type": "application/json"},
            )
            expected_sha = PersistedSingleRequestBatch._fingerprint(
                provider,
                model,
                request,
            )
            expected_identity = (
                provider,
                model,
                request.key,
                expected_sha,
            )
            actual_identity = (
                state_data.get("provider"),
                state_data.get("model"),
                state_data.get("request_key"),
                state_data.get("input_sha256"),
            )
            if (
                actual_identity != expected_identity
                or not isinstance(state_data.get("job_name"), str)
                or not state_data["job_name"].strip()
            ):
                logger.error(
                    "Finalized TOC batch state does not match the current "
                    "provider, model, request, or input; state was retained"
                )
                return False
            # The durable output and remote cleanup both completed; only a
            # crash between deleting the response cache and state remains.
            state_path.with_suffix(".response.txt").unlink(
                missing_ok=True
            )
            state_path.unlink(missing_ok=True)
            return True

    # Translate
    translations = _translate_toc_batch(
        reference_json,
        llm_client,
        translation_models,
        source_language,
        target_language,
        config=config,
        output_dir=output_dir,
    )
    if not translations:
        logger.error("TOC translation failed")
        return False

    # Save toc_tree_translated.json
    if not _save_toc_tree_translated(
        output_dir,
        translations,
        source_language,
        target_language,
    ):
        return False
    if not _finalize_toc_batch_state(
        config,
        output_dir,
        translation_models,
    ):
        return False
    return True


def _build_toc_reference_json(output_dir: Path, translate_dir: Path) -> List[Dict]:
    """
    Build TOC reference JSON including book_title and all chapter titles.

    Extracts titles from translated markdown files as references.
    """
    result = []

    # Load toc_tree.json
    toc_tree_path = output_dir / "toc_tree.json"
    if not toc_tree_path.exists():
        logger.error(f"toc_tree.json not found at {toc_tree_path}")
        return result

    with open(toc_tree_path, 'r', encoding='utf-8') as f:
        toc_tree = json.load(f)

    # Add book_title as first item (level 0)
    book_title = toc_tree.get('book_title', '')
    result.append({
        "id": "book_title",
        "level": 0,
        "original": book_title,
        "reference": ""  # No reference for book title
    })

    # Build mapping from chapter_id to translated title
    title_references = _extract_titles_from_translated(translate_dir)

    # Recursively process chapters
    chapter_counter = [0]  # Use list for mutable counter in nested function

    def process_chapters(chapters: List[Dict], parent_id: str = ""):
        for chapter in chapters:
            chapter_counter[0] += 1

            # Generate chapter_id based on structure
            if parent_id:
                chapter_id = f"{parent_id}.{chapter_counter[0]}"
            else:
                chapter_id = f"chapter_{chapter_counter[0]}"

            original_title = chapter.get('title', '')
            level = chapter.get('level', 1)

            # Try to find reference from translated files
            reference = title_references.get(chapter_id, "")

            result.append({
                "id": chapter_id,
                "level": level,
                "original": original_title,
                "reference": reference
            })

            # Process children
            if 'children' in chapter:
                old_counter = chapter_counter[0]
                chapter_counter[0] = 0
                process_chapters(chapter['children'], chapter_id)
                chapter_counter[0] = old_counter

    process_chapters(toc_tree.get('chapters', []))

    return result


def _extract_titles_from_translated(translate_dir: Path) -> Dict[str, str]:
    """
    Extract first # heading from each translated markdown file.

    Returns:
        Dict mapping chapter_id to translated title
    """
    title_map = {}

    # Check validated subdirectory first (V2 architecture)
    validated_dir = translate_dir / "validated"
    if validated_dir.exists():
        search_dir = validated_dir
    else:
        search_dir = translate_dir

    if not search_dir.exists():
        return title_map

    for md_file in sorted(search_dir.glob("*.md")):
        # Parse filename to get chapter_id
        stem = md_file.stem
        identity = ChapterIdentity.parse(stem)

        if identity is None:
            # Fallback for files that don't match chapter pattern
            continue

        # Only use part1 or non-part files
        if identity.part and identity.part > 1:
            continue

        # Read file and extract first # heading
        try:
            content = md_file.read_text(encoding='utf-8')

            # Find first # heading (not ##)
            match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                # Use base_name as key (e.g., "chapter_3.1")
                title_map[identity.base_name] = title
        except Exception as e:
            logger.warning(f"Failed to extract title from {md_file}: {e}")

    return title_map


def _build_toc_translation_prompt(
    reference_json: List[Dict],
    source_language: str,
    target_language: str,
) -> str:
    """Build the exact request text used by online and batch TOC paths."""
    input_json = json.dumps(reference_json, ensure_ascii=False, indent=2)
    return f"""翻译以下书籍目录结构从{source_language}到简体中文。

**要求**：
1. 参考"reference"字段保持术语一致（如果有参考的话）
2. 统一全书序号格式（如发现"1."和"一、"混用，请统一为同一种格式）
3. 全部使用简体中文（不要繁体）
4. 可修正OCR导致的明显错误
5. 保持原文中的序号格式（如原文是"1."则保持阿拉伯数字）

**输入**：
```json
{input_json}
```

**返回格式**（只返回JSON数组，不要其他内容）：
```json
[
  {{"id": "book_title", "translated": "翻译后的书名"}},
  {{"id": "chapter_1", "translated": "翻译后的标题"}},
  ...
]
```
"""


def _translate_toc_batch(
    reference_json: List[Dict],
    llm_client: LLMClient,
    translation_models: List[Dict],
    source_language: str,
    target_language: str,
    *,
    config: Dict[str, Any],
    output_dir: Path,
) -> List[Dict]:
    """
    Translate entire TOC in one batch using LLM.
    """
    if not reference_json:
        return []

    prompt = _build_toc_translation_prompt(
        reference_json,
        source_language,
        target_language,
    )

    expected_count = len(reference_json)
    expected_ids = [item["id"] for item in reference_json]
    expected_id_set = set(expected_ids)

    def _validate_toc_json(response: str) -> Tuple[bool, str]:
        """Validate TOC translation response is valid JSON with correct structure."""
        try:
            translations = parse_llm_json(response, operation_name="TOC translation")
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

        if not isinstance(translations, list):
            return False, f"Expected JSON array, got {type(translations).__name__}"

        if len(translations) != expected_count:
            return False, (
                f"Translation count mismatch: expected {expected_count}, "
                f"got {len(translations)}"
            )

        response_ids = []
        for i, entry in enumerate(translations):
            if not isinstance(entry, dict):
                return False, f"Entry {i} is not an object"
            if "id" not in entry or "translated" not in entry:
                return False, f"Entry {i} missing 'id' or 'translated' field"
            if not isinstance(entry["translated"], str) or not entry["translated"].strip():
                return False, f"Entry {i} has an empty or non-string translation"
            response_ids.append(entry["id"])

        if len(set(response_ids)) != len(response_ids):
            return False, "Response contains duplicate IDs"
        response_id_set = set(response_ids)
        if response_id_set != expected_id_set:
            missing = sorted(expected_id_set - response_id_set)
            unexpected = sorted(response_id_set - expected_id_set)
            return False, (
                f"ID set mismatch: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )

        return True, ""

    def _build_repair_prompt(
        original_prompt: str,
        failed_response: str,
        error_reason: str,
    ) -> List[Dict]:
        """Build multi-turn repair prompt for TOC JSON."""
        return [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": failed_response},
            {"role": "user", "content": (
                f"你返回的JSON有错误：{error_reason}\n\n"
                f"请修复并重新返回完整的、语法正确的JSON数组。"
                f"只返回JSON，不要其他内容。"
            )},
        ]

    # Ensure validation retries for TOC translation (repair prompt needs ≥1).
    # ``generate_with_validation`` is an online API path, so batch-only models
    # need a separate, persisted batch round-trip below.
    toc_models = [
        {**m, "validation_retries": max(m.get("validation_retries", 0), 2)}
        for m in (translation_models or [])
    ]
    if toc_models and toc_models[0].get("mode", "online") == "batch":
        return _translate_toc_with_batch(
            prompt=prompt,
            validator=_validate_toc_json,
            config=config,
            output_dir=output_dir,
            batch_models=toc_models,
        )

    online_models = [
        m for m in toc_models
        if m.get("mode", "online") != "batch"
    ]
    if not online_models:
        logger.error("No model configured for TOC translation")
        return []

    try:
        response = llm_client.generate_with_validation(
            prompt=prompt,
            model_configs=online_models,
            validator=_validate_toc_json,
            operation_name="TOC batch translation",
            repair_prompt_builder=_build_repair_prompt,
        )

        translations = parse_llm_json(response, operation_name="TOC translation")
        return translations

    except Exception as e:
        logger.error(f"TOC batch translation failed: {e}")
        return []


def _translate_toc_with_batch(
    *,
    prompt: str,
    validator,
    config: Dict[str, Any],
    output_dir: Path,
    batch_models: List[Dict],
) -> List[Dict]:
    """Translate a TOC through the generic persisted single-request runner."""
    from ..core.executor import PersistedSingleRequestBatch
    from ..utils.batch_utils import (
        BatchRequest,
        create_batch_client_from_config,
    )

    batch_model = next((m for m in batch_models if m.get("mode") == "batch"), None)
    if not batch_model:
        logger.error("No batch model configured for TOC translation")
        return []

    provider = batch_model.get("provider", "gemini")
    model = batch_model.get("model")
    poll_interval = config.get("batch", {}).get("poll_interval", 60)
    state_path = output_dir / "toc_translation_batch_state.json"

    try:
        client = create_batch_client_from_config(
            config,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        logger.error(f"Cannot create TOC batch client: {exc}")
        return []

    request = BatchRequest(
        key="toc_translation",
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"response_mime_type": "application/json"},
    )
    runner = PersistedSingleRequestBatch(
        client=client,
        provider=provider,
        model=model,
        state_path=state_path,
        poll_interval=poll_interval,
    )
    try:
        response_text = runner.run(
            request,
            validator,
            display_name="pdf2epub-toc-translation",
        )
        translations = parse_llm_json(
            response_text,
            operation_name="TOC translation",
        )
        return translations
    except Exception as e:
        logger.error(f"TOC batch translation failed: {e}")
        return []


def _finalize_toc_batch_state(
    config: Dict[str, Any],
    output_dir: Path,
    translation_models: List[Dict],
) -> bool:
    """Finalize a validated TOC batch only after its JSON is durable."""
    state_path = output_dir / "toc_translation_batch_state.json"
    if not state_path.exists():
        return True

    from ..core.executor import PersistedSingleRequestBatch
    from ..utils.batch_utils import create_batch_client_from_config

    batch_model = next(
        (
            model
            for model in translation_models
            if model.get("mode") == "batch"
        ),
        None,
    )
    if not batch_model:
        logger.error(
            "TOC batch state exists, but no batch model is configured; "
            "state was retained"
        )
        return False

    provider = batch_model.get("provider", "gemini")
    model = batch_model.get("model")
    try:
        client = create_batch_client_from_config(
            config,
            provider=provider,
            model=model,
        )
        runner = PersistedSingleRequestBatch(
            client=client,
            provider=provider,
            model=model,
            state_path=state_path,
            poll_interval=config.get("batch", {}).get(
                "poll_interval",
                60,
            ),
        )
        runner.finalize()
    except Exception as exc:
        logger.error(
            f"TOC output was saved, but batch finalization failed: {exc}. "
            "State and response cache were retained for --resume."
        )
        return False
    return True


def _save_toc_tree_translated(
    output_dir: Path,
    translations: List[Dict],
    source_language: str,
    target_language: str,
) -> bool:
    """
    Save toc_tree_translated.json with translated titles.
    """
    # Load original toc_tree
    toc_tree_path = output_dir / "toc_tree.json"
    if not toc_tree_path.exists():
        logger.error(f"toc_tree.json not found at {toc_tree_path}")
        return False

    with open(toc_tree_path, 'r', encoding='utf-8') as f:
        toc_tree = json.load(f)

    # Build translation lookup
    trans_map = {t['id']: t['translated'] for t in translations if 'id' in t and 'translated' in t}

    # Update book_title
    if 'book_title' in trans_map:
        toc_tree['book_title'] = trans_map['book_title']

    # Update language to target
    toc_tree['language'] = target_language.lower()
    toc_tree['source_language'] = source_language.lower()

    # Recursively update chapter titles
    chapter_counter = [0]

    def update_chapters(chapters: List[Dict], parent_id: str = ""):
        for chapter in chapters:
            chapter_counter[0] += 1

            if parent_id:
                chapter_id = f"{parent_id}.{chapter_counter[0]}"
            else:
                chapter_id = f"chapter_{chapter_counter[0]}"

            # Store original title
            if 'original_title' not in chapter:
                chapter['original_title'] = chapter.get('title', '')

            # Update with translation
            if chapter_id in trans_map:
                chapter['title'] = trans_map[chapter_id]

            # Process children
            if 'children' in chapter:
                old_counter = chapter_counter[0]
                chapter_counter[0] = 0
                update_chapters(chapter['children'], chapter_id)
                chapter_counter[0] = old_counter

    update_chapters(toc_tree.get('chapters', []))

    # Save translated toc_tree
    output_path = output_dir / "toc_tree_translated.json"
    temp_path = output_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(toc_tree, file, ensure_ascii=False, indent=2)
        temp_path.replace(output_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        logger.error(f"Failed to save translated TOC: {exc}")
        return False

    logger.success(f"Saved translated TOC to {output_path}")
    return True
