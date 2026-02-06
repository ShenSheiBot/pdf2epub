"""
Translate V2 command - uses new Phase + Pipeline V2 architecture.

Key features:
- Uses Phase for composable stages
- Uses Pipeline V2 with Executor + Hooks
- Configuration-driven (reads from config.yaml)
- No intermediate aggregation
- No retry loops (re-queue instead)
- Batch + Online simultaneous execution (if batch client provided)
"""

import json
from pathlib import Path
from loguru import logger

from ..utils.common import load_config
from ..utils.logging_config import configure_logging
from ..utils.llm_client import LLMClient
from ..processors import TranslateProcessor
from ..core.phase import Phase, PartBasedLoader
from ..core.factory_v2 import create_processing_pipeline_v2
from ..core.book_structure import BookStructure


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

    return 0 if result.failed == 0 else 1
