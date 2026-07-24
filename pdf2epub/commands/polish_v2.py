"""
Polish V2 command - uses new Phase + Pipeline V2 architecture.

Key features:
- Uses Phase for composable stages
- Uses Pipeline V2 with Executor + Hooks
- Configuration-driven (reads from config.yaml)
- No intermediate aggregation
- No retry loops (re-queue instead)
"""

import json
from pathlib import Path
from loguru import logger

from ..utils.common import load_config
from ..utils.logging_config import configure_logging
from ..utils.llm_client import LLMClient
from ..processors import PolishProcessor
from ..core.phase import Phase, PartBasedLoader
from ..core.factory_v2 import (
    create_processing_pipeline_v2,
    get_task_model_configs,
)
from ..core.book_structure import BookStructure


def polish_v2_command(args):
    """
    Handle the polish-v2 subcommand using new architecture.

    Configuration is read from config.yaml under:
    - polish.models: Model chain configuration
    - polish.truncation_check_lines: Truncation detection
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
    configure_logging(book_title, "polish_v2")

    logger.info(f"[v2] Starting polish for: {book_title}")

    # Setup directories
    output_dir = Path("output") / book_title
    from pdf2epub.utils.network_utils import set_llm_trace_path
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
    # Try ocr_markdown first (refine output), then pages_merged (legacy)
    input_dir = output_dir / "ocr_markdown"
    if not input_dir.exists():
        input_dir = output_dir / "pages_merged"
    polish_dir = output_dir / "polished_markdown"

    if not input_dir.exists():
        logger.error(f"Input directory not found. Tried:")
        logger.error(f"  - {output_dir / 'ocr_markdown'}")
        logger.error(f"  - {output_dir / 'pages_merged'}")
        logger.info("Run 'pdf2epub refine' first to generate merged pages")
        return 1

    polish_dir.mkdir(parents=True, exist_ok=True)

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

    # Create the old PolishProcessor (for its prompt/response logic)
    content_type = getattr(args, 'content_type', 'auto')
    old_processor = PolishProcessor(
        config=config,
        book_title=book_title,
        max_workers=1,  # Not used, Pipeline handles concurrency
        resume=False,   # Not used, Phase handles resume
        skip_truncation_check=getattr(args, 'skip_truncation_check', False),
        polish_models=get_task_model_configs(config, "polish"),
        content_type=content_type,
        use_longest_on_failure=False,  # Not used
        book_structure=book_structure_data,
    )

    # PolishProcessor now directly implements ProcessorProtocol
    processor = old_processor

    # Get configuration overrides from args
    max_workers = getattr(args, 'max_workers', None)
    # CLI --parallel overrides config; otherwise use config's processing_mode
    if getattr(args, 'parallel', False):
        sequential_mode = False
    else:
        # Check polish section first, then top-level
        polish_config = config.get('polish', {})
        processing_mode = polish_config.get('processing_mode', config.get('processing_mode', 'parallel'))
        sequential_mode = processing_mode != 'parallel'
    skip_validation = getattr(args, 'skip_validation', False)

    # Create Pipeline V2 using config-based factory
    pipeline = create_processing_pipeline_v2(
        processor=processor,
        output_dir=polish_dir,
        llm_client=llm_client,
        config=config,  # Pass full config for reading all settings
        book_structure=book_structure,
        task_type="polish",
        use_batch_validation=not skip_validation,
        sequential_mode=sequential_mode,
        max_workers=max_workers,
        restore_images=True,
    )

    # Create Phase
    phase = Phase(
        name="polish_v2",
        input_dir=input_dir,
        output_dir=polish_dir,
        pipeline=pipeline,
        loader=PartBasedLoader(),
        file_pattern="*.md",
    )

    # Run
    resume = getattr(args, 'resume', False)
    result = phase.run(resume=resume)

    # Report
    logger.info(f"[v2] Polish completed: {result.completed}/{result.total} succeeded")
    if result.failed > 0:
        logger.warning(f"[v2] Failed: {result.failed} files")
        for key in result.failed_keys[:10]:  # Show first 10
            logger.warning(f"  - {key}")

    return 0 if result.failed == 0 else 1
