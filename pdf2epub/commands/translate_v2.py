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
from typing import List, Dict
from loguru import logger

from ..utils.common import load_config, parse_llm_json
from ..utils.logging_config import configure_logging
from ..utils.llm_client import LLMClient
from ..processors import TranslateProcessor
from ..core.phase import Phase, PartBasedLoader
from ..core.factory_v2 import create_processing_pipeline_v2
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
        translation_models=translation_config.get('models'),
        use_entities=use_entities,
        use_longest_on_failure=False,  # Not used
        book_structure=book_structure_data,
    )

    # TranslateProcessor now directly implements ProcessorProtocol
    processor = old_processor

    # Get configuration overrides from args
    max_workers = getattr(args, 'max_workers', None)
    sequential_mode = not getattr(args, 'parallel', False)
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
        _translate_toc(
            output_dir=output_dir,
            translate_dir=translate_dir,
            llm_client=llm_client,
            translation_models=translation_config.get('models'),
            source_language=source_language,
            target_language=target_language,
        )
        logger.success("[v2] TOC translation completed")

    return 0 if result.failed == 0 else 1


def _translate_toc(
    output_dir: Path,
    translate_dir: Path,
    llm_client: LLMClient,
    translation_models: List[Dict],
    source_language: str,
    target_language: str,
) -> None:
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
        return

    # Translate
    translations = _translate_toc_batch(
        reference_json, llm_client, translation_models, source_language, target_language
    )
    if not translations:
        logger.error("TOC translation failed")
        return

    # Save toc_tree_translated.json
    _save_toc_tree_translated(output_dir, translations, source_language, target_language)


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


def _translate_toc_batch(
    reference_json: List[Dict],
    llm_client: LLMClient,
    translation_models: List[Dict],
    source_language: str,
    target_language: str,
) -> List[Dict]:
    """
    Translate entire TOC in one batch using LLM.
    """
    if not reference_json:
        return []

    # Build prompt
    input_json = json.dumps(reference_json, ensure_ascii=False, indent=2)

    prompt = f"""翻译以下书籍目录结构从{source_language}到简体中文。

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

    try:
        # Call LLM
        response = llm_client.generate(
            prompt=prompt,
            model_configs=translation_models,
            operation_name="TOC batch translation"
        )

        # Parse response - extract JSON from possible markdown code block
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block
            lines = response.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response = '\n'.join(json_lines)

        translations = parse_llm_json(response, operation_name="TOC translation")

        # Validate count
        if len(translations) != len(reference_json):
            logger.warning(
                f"Translation count mismatch: expected {len(reference_json)}, "
                f"got {len(translations)}"
            )

        return translations

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse TOC translation response: {e}")
        logger.error(f"Response was: {response[:500]}...")
        return []
    except Exception as e:
        logger.error(f"TOC batch translation failed: {e}")
        return []


def _save_toc_tree_translated(
    output_dir: Path,
    translations: List[Dict],
    source_language: str,
    target_language: str,
) -> None:
    """
    Save toc_tree_translated.json with translated titles.
    """
    # Load original toc_tree
    toc_tree_path = output_dir / "toc_tree.json"
    if not toc_tree_path.exists():
        logger.error(f"toc_tree.json not found at {toc_tree_path}")
        return

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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(toc_tree, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved translated TOC to {output_path}")
