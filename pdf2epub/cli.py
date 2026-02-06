#!/usr/bin/env python3
"""
Unified CLI for pdf2epub markdown processing.

This module provides a single entrypoint for all markdown processing operations
including polishing OCR output and translating content.
"""

import yaml
import argparse
import sys
from pathlib import Path
from loguru import logger
from pdf2epub.utils.logging_config import configure_logging
from pdf2epub.utils.common import load_config
from pdf2epub.utils.safety import check_output_directory_conflict
from pdf2epub.processors import PolishProcessor, TranslateProcessor

# Configure logger
logger = configure_logging()


def polish_command(args):
    """Handle the polish subcommand - uses V2 pipeline."""
    from .commands import polish_v2_command
    return polish_v2_command(args)


def breakdown_command(args):
    """Handle the breakdown subcommand.

    DEPRECATED: This command is part of the legacy workflow.
    Please use: ocr-pages → refine → polish → build-epub
    """
    # ============================================================
    # ⚠️  DEPRECATED COMMAND - 已弃用命令
    # ============================================================
    # This command is part of the LEGACY workflow and may be removed.
    # 此命令属于旧版工作流，可能会被移除。
    #
    # RECOMMENDED workflow / 推荐的新工作流：
    #   pdf2epub ocr-pages → refine → polish → build-epub
    # ============================================================
    logger.warning("=" * 60)
    logger.warning("⚠️  DEPRECATED: 'breakdown' is part of the legacy workflow")
    logger.warning("   此命令已弃用，属于旧版工作流")
    logger.warning("")
    logger.warning("   Recommended workflow / 推荐工作流:")
    logger.warning("   pdf2epub ocr-pages -i <pdf>")
    logger.warning("   pdf2epub refine")
    logger.warning("   pdf2epub polish")
    logger.warning("   pdf2epub build-epub")
    logger.warning("=" * 60)

    from pdf2epub.breakdown import (
        load_config as load_breakdown_config,
        preprocess_pdf,
        analyze_pdf_structure,
        detect_front_and_back_matter
    )
    from pdf2epub.utils.network_utils import create_gemini_client_from_config
    import json

    # Load configuration
    config = load_breakdown_config(args.config)
    
    # Get input PDF path
    input_pdf = Path(args.input)
    if not input_pdf.exists():
        logger.error(f"Input PDF not found: {input_pdf}")
        return 1
    
    # Get book title from config
    book_title = config.get("title")
    if not book_title:
        book_title = input_pdf.stem
        logger.warning(f"No title found in config, using PDF filename: {book_title}")

    # Configure file logging
    configure_logging(book_title, "breakdown")

    logger.info(f"Processing PDF breakdown for: {book_title}")

    # Define output directory
    output_dir = Path("output") / Path(book_title)

    # Check for conflicts with existing output
    output_dir = check_output_directory_conflict(output_dir, input_pdf)

    # If directory was renamed, update book_title
    actual_book_title = output_dir.name
    if actual_book_title != book_title:
        logger.info(f"Using renamed directory: {actual_book_title}")
        book_title = actual_book_title

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Gemini API
    breakdown_config = config.get("breakdown", {})
    provider_name = breakdown_config.get("provider", "gemini")
    try:
        gemini_client = create_gemini_client_from_config(config, provider_name)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Use preprocess_pdf which handles page number patches and multi-round compression
    processed_pdf = preprocess_pdf(input_pdf, output_dir)
    
    # Analyze book structure
    logger.info("Analyzing book structure...")
    structure = analyze_pdf_structure(gemini_client, processed_pdf, book_title, config)
    
    # Add book title to the structure
    structure['book_title'] = book_title
    
    # Detect front matter and back matter automatically
    structure = detect_front_and_back_matter(structure, processed_pdf)
    
    # Save the structured output to the output directory
    output_file = output_dir / "book_structure.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)
    
    logger.success(f"Book structure saved to {output_file}")
    
    return 0


def refine_command(args):
    """Handle the refine subcommand (refined breakdown with boundary verification)."""
    from pdf2epub.refine import RefinedBreakdown
    from pathlib import Path

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "refine")

    # Get refine config
    refine_config = config.get('refine', {})
    max_tokens = args.max_tokens or refine_config.get('max_tokens', 8000)

    # Determine PDF path
    output_dir = Path("output") / book_title
    if args.input:
        pdf_path = Path(args.input)
    else:
        # Try to find processed PDF in output directory
        pdf_path = output_dir / "input.pdf"
        if not pdf_path.exists():
            pdf_path = output_dir / "input_original.pdf"

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.info("Specify --input <pdf_path> to provide the PDF file")
        return 1

    # Check for pages
    pages_dir = output_dir / "pages"
    if not pages_dir.exists() or not list(pages_dir.glob("page_*.md")):
        logger.error(f"OCR pages not found in {pages_dir}")
        logger.info("Run 'pdf2epub ocr-pages' first to generate page-level OCR")
        return 1

    logger.info(f"Starting refined breakdown for: {book_title}")
    logger.info(f"Max tokens per unit: {max_tokens}")

    try:
        refiner = RefinedBreakdown(
            config=config,
            max_tokens=max_tokens
        )

        unit_metadata = refiner.process(
            pdf_path=pdf_path,
            output_dir=output_dir,
            book_title=book_title,
            resume=args.resume
        )

        logger.success(f"Refined breakdown complete: {len(unit_metadata)} units generated")
        return 0

    except Exception as e:
        logger.error(f"Refined breakdown failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def ocr_pages_command(args):
    """Handle the ocr-pages subcommand (page-level OCR)."""
    from pdf2epub.ocr_pages import ocr_full_book_pagewise
    from pathlib import Path
    import yaml

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "ocr-pages")

    # Setup paths
    output_dir = Path("output") / book_title

    # Find PDF
    if args.input:
        pdf_path = Path(args.input)
    else:
        pdf_path = output_dir / "input.pdf"
        if not pdf_path.exists():
            pdf_path = output_dir / "input_original.pdf"

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.info("Specify --input with the path to your PDF file")
        return 1

    logger.info(f"Starting page-level OCR for: {book_title}")

    # Get OCR settings from config
    ocr_config = config.get('ocr', {})
    backend = ocr_config.get('backend', 'mistral')
    max_workers = args.max_workers or ocr_config.get('vision', {}).get('max_workers', 5)

    # Get credentials
    credentials = config.get('credentials', {}).get('providers', {})

    # Setup backend-specific parameters
    api_key = None
    base_url = None

    if backend == 'mistral':
        mistral_config = credentials.get('mistral', {})
        api_key = mistral_config.get('api_key')
        base_url = mistral_config.get('base_url')
    elif backend == 'azure':
        azure_config = credentials.get('azure', {})
        api_key = azure_config.get('api_key')
        base_url = azure_config.get('endpoint')

    try:
        ocr_full_book_pagewise(
            pdf_path=pdf_path,
            output_dir=output_dir,
            start_page=args.start_page or 1,
            end_page=args.end_page,
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            resume=args.resume,
            config=config,
            max_workers=max_workers
        )

        logger.success(f"Page-level OCR complete!")
        logger.info(f"Output: {output_dir / 'pages'}")
        logger.info("Next step: pdf2epub refine")
        return 0

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def ocr_command(args):
    """Handle the OCR subcommand (legacy chapter-based workflow).

    DEPRECATED: This command is part of the legacy workflow.
    Please use: ocr-pages → refine → polish → build-epub
    """
    # ============================================================
    # ⚠️  DEPRECATED COMMAND - 已弃用命令
    # ============================================================
    # This command is part of the LEGACY workflow and may be removed.
    # 此命令属于旧版工作流，可能会被移除。
    #
    # RECOMMENDED workflow / 推荐的新工作流：
    #   pdf2epub ocr-pages → refine → polish → build-epub
    # ============================================================
    logger.warning("=" * 60)
    logger.warning("⚠️  DEPRECATED: 'ocr' is part of the legacy workflow")
    logger.warning("   此命令已弃用，属于旧版工作流")
    logger.warning("")
    logger.warning("   Use 'ocr-pages' instead / 请使用 'ocr-pages':")
    logger.warning("   pdf2epub ocr-pages -i <pdf>")
    logger.warning("   pdf2epub refine")
    logger.warning("   pdf2epub polish")
    logger.warning("   pdf2epub build-epub")
    logger.warning("=" * 60)

    import sys
    original_argv = sys.argv

    try:
        from pdf2epub.ocr_chapters import main as ocr_main

        # Build argv for ocr_chapters main
        new_argv = ['ocr_chapters.py']
        if hasattr(args, 'resume') and args.resume:
            new_argv.append('--resume')
        if hasattr(args, 'aggregate_only') and args.aggregate_only:
            new_argv.append('--aggregate-only')

        sys.argv = new_argv
        return ocr_main()
    finally:
        sys.argv = original_argv


def extract_entities_command(args):
    """Handle the extract-entities subcommand."""
    from pdf2epub.entity_extractor import (
        load_config as load_entity_config,
        extract_entities_from_pdf,
        save_entities
    )
    from pdf2epub.utils.network_utils import create_gemini_client_from_config
    
    # Load configuration
    config = load_entity_config(args.config)
    book_title = config.get("title")

    if not book_title:
        if args.input:
            # Use PDF filename as fallback
            book_title = Path(args.input).stem
            logger.warning(f"No title in config, using: {book_title}")
        else:
            logger.error("No title found in config.yaml and no input file specified")
            return 1

    # Configure file logging
    configure_logging(book_title, "extract-entities")

    logger.info(f"Extracting entities from: {book_title}")
    logger.info(f"Language pair: {args.source_lang} → {args.target_lang}")

    # Initialize Gemini client
    translation_config = config.get("translation", {})
    provider_name = translation_config.get("provider", "gemini")
    try:
        gemini_client = create_gemini_client_from_config(config, provider_name)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Setup paths
    output_dir = Path("output") / book_title
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine PDF path
    if args.input:
        pdf_path = Path(args.input)
    else:
        # Default to input.pdf in the book's output directory
        pdf_path = output_dir / "input.pdf"

    # Check if PDF exists
    if not pdf_path.exists():
        # Try alternative paths in output directory
        processed_path = output_dir / "input.pdf"
        original_path = output_dir / "input_original.pdf"

        if processed_path.exists() and pdf_path != processed_path:
            pdf_path = processed_path
            logger.info(f"Using processed PDF from: {pdf_path}")
        elif original_path.exists():
            pdf_path = original_path
            logger.info(f"Using original PDF from: {pdf_path}")
        else:
            if args.input:
                logger.error(f"PDF not found: {args.input}")
            else:
                logger.error(f"PDF not found in {output_dir}/. Expected input.pdf or input_original.pdf")
            return 1
    
    try:
        # Extract entities
        entities = extract_entities_from_pdf(
            pdf_path=pdf_path,
            book_title=book_title,
            gemini_client=gemini_client,
            config=config,
            language_pair=(args.source_lang, args.target_lang)
        )
        
        # Save results
        save_entities(entities, output_dir)
        
        logger.success("Entity extraction completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return 1


def epub_command(args):
    """Handle the epub generation subcommand.

    DEPRECATED: This command is part of the legacy workflow.
    Please use: build-epub (which uses toc_tree.json)
    """
    # ============================================================
    # ⚠️  DEPRECATED COMMAND - 已弃用命令
    # ============================================================
    # This command is part of the LEGACY workflow and may be removed.
    # 此命令属于旧版工作流，可能会被移除。
    #
    # RECOMMENDED command / 推荐的新命令：
    #   pdf2epub build-epub (uses toc_tree.json)
    # ============================================================
    logger.warning("=" * 60)
    logger.warning("⚠️  DEPRECATED: 'epub' is part of the legacy workflow")
    logger.warning("   此命令已弃用，属于旧版工作流")
    logger.warning("")
    logger.warning("   Use 'build-epub' instead / 请使用 'build-epub':")
    logger.warning("   pdf2epub build-epub [--translated]")
    logger.warning("=" * 60)

    # Import the main function from generate_epub
    from pdf2epub.generate_epub import main as generate_epub_main
    import sys

    # Prepare arguments for generate_epub main function
    # Save original sys.argv and replace it
    original_argv = sys.argv
    try:
        # Build new argv for generate_epub
        new_argv = ['generate_epub.py', '-c', args.config]
        if args.input:
            new_argv.extend(['-i', args.input])
        if args.translated:
            new_argv.append('--translated')
        if args.zip:
            new_argv.append('--zip')
        if args.relevel:
            new_argv.append('--relevel')
        if args.global_footnotes:
            new_argv.append('--global-footnotes')
        
        sys.argv = new_argv
        
        # Call the main function
        if args.translated:
            logger.info("Generating translated EPUB...")
        else:
            logger.info("Generating EPUB...")
        generate_epub_main()
        
        return 0
    except Exception as e:
        logger.error(f"EPUB generation failed: {e}")
        return 1
    finally:
        # Restore original argv
        sys.argv = original_argv


def build_epub_command(args):
    """Handle the build-epub subcommand (toc_tree.json driven)."""
    import asyncio
    from pathlib import Path
    from .build_epub import build_epub, BuildEpubConfig

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "build-epub")

    # Set up paths
    output_dir = Path("output") / book_title
    toc_tree_path = output_dir / "toc_tree.json"

    if not toc_tree_path.exists():
        logger.error(f"toc_tree.json not found at {toc_tree_path}")
        logger.info("Run 'refine' command first to generate toc_tree.json")
        return 1

    # Determine markdown directory
    # V2 architecture stores results in validated/ subdirectory
    if args.translated:
        markdown_dir = output_dir / "translated" / "validated"
        if not markdown_dir.exists():
            # Fallback to old path for backwards compatibility
            markdown_dir = output_dir / "translated"
        logger.info("Building EPUB from translated markdown...")
    else:
        markdown_dir = output_dir / "polished_markdown" / "validated"
        if not markdown_dir.exists():
            # Fallback to old path for backwards compatibility
            markdown_dir = output_dir / "polished_markdown"
        logger.info("Building EPUB from polished markdown...")

    if not markdown_dir.exists():
        logger.error(f"Markdown directory not found: {markdown_dir}")
        return 1

    # Set up images directory
    images_dir = output_dir / "images"
    if not images_dir.exists():
        images_dir = None

    # Set up cover image
    cover_image = None
    if args.cover:
        cover_path = Path(args.cover)
        if cover_path.exists():
            cover_image = cover_path
        else:
            logger.warning(f"Cover image not found: {args.cover}")
    else:
        # Auto-detect cover in images directory
        if images_dir:
            for cover_name in ["cover.jpg", "cover.jpeg", "cover.png", "cover.gif"]:
                cover_path = images_dir / cover_name
                if cover_path.exists():
                    cover_image = cover_path
                    logger.info(f"Auto-detected cover image: {cover_path}")
                    break

    # Get target language from config
    target_language = config.get("translation", {}).get("target_language", "Chinese")

    # Create config
    build_config = BuildEpubConfig(
        book_title=book_title,
        output_dir=output_dir,
        markdown_dir=markdown_dir,
        toc_tree_path=toc_tree_path,
        images_dir=images_dir,
        cover_image=cover_image,
        translated=args.translated,
        target_language=target_language,
        config=config
    )

    try:
        # Build EPUB
        epub_path = build_epub(build_config)
        logger.success(f"EPUB created: {epub_path}")
        return 0
    except Exception as e:
        logger.error(f"EPUB build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def translate_html_command(args):
    """Handle the translate-html subcommand (direct HTML translation)."""
    from pathlib import Path
    from pdf2epub.html_translation import HTMLEpubPipeline, HTMLTranslateProcessor

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "translate-html")

    # Setup paths
    output_dir = Path("output") / book_title
    epub_path = Path(args.input) if args.input else None

    # If no epub specified, look for original epub in output dir
    if epub_path is None:
        for candidate in ["input.epub", "original.epub"]:
            candidate_path = output_dir / candidate
            if candidate_path.exists():
                epub_path = candidate_path
                break

    if epub_path is None or not epub_path.exists():
        logger.error("Input file not found. Use -i to specify EPUB/AZW3/MOBI file.")
        return 1

    # 格式转换或复制到 output 目录
    from pdf2epub.utils.ebook_converter import needs_conversion, convert_to_epub
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    input_epub = output_dir / "input.epub"

    if needs_conversion(epub_path):
        try:
            epub_path, _ = convert_to_epub(epub_path, output_dir)
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return 1
    elif epub_path.resolve() != input_epub.resolve():
        # EPUB 输入也复制到 output 目录，方便 build-html-epub 找到
        shutil.copy2(epub_path, input_epub)
        logger.info(f"Copied input EPUB to: {input_epub}")
        epub_path = input_epub

    try:
        # Create pipeline
        pipeline = HTMLEpubPipeline(
            epub_path=epub_path,
            output_dir=output_dir,
            config=config
        )

        # Auto-detect source language from EPUB metadata
        source_language = args.source_language or pipeline.source_language
        target_language = args.target_language or config.get("translation", {}).get("target_language", "Chinese")

        logger.info(f"Starting HTML translation for: {pipeline.book_title}")
        logger.info(f"Source EPUB: {epub_path}")
        logger.info(f"Translation: {source_language} → {target_language}")

        # Step 1: Extract and preprocess
        if not args.skip_extract:
            extracted = pipeline.extract_and_preprocess()
            logger.info(f"Extracted {extracted} XHTML files")

        # Step 2: Translate metadata (title + TOC)
        if not args.skip_translate:
            logger.info("Translating book title and TOC...")
            metadata = pipeline.translate_metadata(target_language=target_language)
            logger.info(f"Translated title: {metadata['translated_title']}")

        # Step 3: Translate content
        if not args.skip_translate:
            # Initialize translator
            use_entities = None
            if args.use_entities:
                use_entities = True
            elif args.no_entities:
                use_entities = False

            processor = HTMLTranslateProcessor(
                config=config,
                book_title=book_title,
                source_language=source_language,
                target_language=target_language,
                max_workers=args.max_workers or config.get('max_concurrent_workers', 4),
                resume=args.resume,
                translation_models=config.get('translation', {}).get('models'),
                use_entities=use_entities,
                use_longest_on_failure=config.get('validation_strategy', {}).get('use_longest_on_failure', False)
            )

            # Handle --limit: only translate first N files, copy rest
            if args.limit:
                import shutil
                all_files = sorted(pipeline.compressed_units_dir.glob("*.md"))
                files_to_translate = all_files[:args.limit]
                files_to_copy = all_files[args.limit:]

                logger.info(f"Limit mode: translating {len(files_to_translate)} files, copying {len(files_to_copy)} untranslated")

                # Copy untranslated files directly
                for f in files_to_copy:
                    dest = pipeline.translated_dir / f.name
                    shutil.copy(f, dest)
                    logger.debug(f"Copied untranslated: {f.name}")

                # Process only limited files (pass specific files to processor)
                summary = processor.process_specific_files([f.stem for f in files_to_translate])
            else:
                # Process all files
                summary = processor.process_all_files()

            if summary.get("error"):
                logger.error(f"Translation failed: {summary['error']}")
                return 1

            logger.info(f"Translation complete: {summary.get('successful', 0)} files")

        logger.success("HTML translation complete!")
        logger.info(f"Output: {output_dir / 'translated_compressed'}")
        logger.info("Next step: pdf2epub build-html-epub")
        return 0

    except Exception as e:
        logger.error(f"HTML translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def build_html_epub_command(args):
    """Handle the build-html-epub subcommand (rebuild EPUB with translated HTML)."""
    from pathlib import Path
    from pdf2epub.html_translation import HTMLEpubPipeline, build_html_epub

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "build-html-epub")

    # Setup paths
    output_dir = Path("output") / book_title
    epub_path = Path(args.input) if args.input else None

    # If no epub specified, look for original epub in output dir
    if epub_path is None:
        for candidate in ["input.epub", "original.epub"]:
            candidate_path = output_dir / candidate
            if candidate_path.exists():
                epub_path = candidate_path
                break

    if epub_path is None or not epub_path.exists():
        logger.error("Input file not found. Use -i to specify EPUB/AZW3/MOBI file.")
        return 1

    # 格式转换或复制到 output 目录
    from pdf2epub.utils.ebook_converter import needs_conversion, convert_to_epub
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    input_epub = output_dir / "input.epub"

    if needs_conversion(epub_path):
        try:
            epub_path, _ = convert_to_epub(epub_path, output_dir)
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return 1
    elif epub_path.resolve() != input_epub.resolve():
        shutil.copy2(epub_path, input_epub)
        logger.info(f"Copied input EPUB to: {input_epub}")
        epub_path = input_epub

    logger.info(f"Building translated EPUB for: {book_title}")

    try:
        # Create pipeline
        pipeline = HTMLEpubPipeline(
            epub_path=epub_path,
            output_dir=output_dir,
            config=config
        )

        # Determine output path (None = let postprocess_and_build use translated title)
        output_epub = Path(args.output) if args.output else None

        # Build EPUB (restore attrs + repackage)
        result_path = pipeline.postprocess_and_build(output_epub)

        logger.success(f"EPUB created: {result_path}")
        return 0

    except Exception as e:
        logger.error(f"EPUB build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def translate_command(args):
    """Handle the translate subcommand - uses V2 pipeline."""
    from .commands import translate_v2_command
    return translate_v2_command(args)


def patch_paper_command(args):
    """Execute the patch-paper command."""
    from .patch_paper_structure import patch_paper_structure
    from .utils.common import load_config

    # Load config to get book title
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No book title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "patch-paper")

    logger.info(f"Patching structure for: {book_title}")

    success = patch_paper_structure(
        book_title=book_title,
        chapter_name=args.chapter_name,
        preserve_toc=not args.no_preserve_toc
    )

    if success:
        logger.success("Structure patched successfully!")
        logger.info("You can now run 'pdf2epub ocr' to process the document as a single chapter")

    return 0 if success else 1


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="PDF to EPUB markdown processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
===============================================================================
RECOMMENDED WORKFLOW / 推荐工作流 (uses toc_tree.json):
===============================================================================

  # Complete pipeline for a PDF book:
  pdf2epub ocr-pages -i mybook.pdf   # Page-level OCR
  pdf2epub refine                    # Generate toc_tree.json with boundary verification
  pdf2epub polish                    # Clean up OCR output
  pdf2epub build-epub                # Generate EPUB from toc_tree.json

  # With translation:
  pdf2epub ocr-pages -i mybook.pdf
  pdf2epub refine
  pdf2epub polish --content-type japanese
  pdf2epub translate --target-language Chinese
  pdf2epub build-epub --translated

  # EPUB Translation (preserves original formatting):
  pdf2epub translate-html -i mybook.epub     # Extract + translate HTML
  pdf2epub build-html-epub                    # Build translated EPUB

  # Test with limited files first:
  pdf2epub translate-html -i mybook.epub --limit 5
  pdf2epub build-html-epub

===============================================================================
⚠️  DEPRECATED WORKFLOW / 已弃用工作流 (still works, but not recommended):
===============================================================================

  # Legacy pipeline (uses book_structure.json):
  pdf2epub breakdown -i mybook.pdf   # [DEPRECATED]
  pdf2epub ocr                       # [DEPRECATED]
  pdf2epub polish
  pdf2epub epub                      # [DEPRECATED]

===============================================================================
        """
    )
    
    # Global arguments
    parser.add_argument("-c", "--config", default="config.yaml", 
                        help="Path to config file")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Breakdown subcommand (DEPRECATED)
    breakdown_parser = subparsers.add_parser(
        "breakdown",
        help="[DEPRECATED] Analyze PDF structure (use ocr-pages → refine instead)",
        description="⚠️ DEPRECATED: This command is part of the legacy workflow. Use 'ocr-pages' + 'refine' instead."
    )
    breakdown_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input PDF file"
    )
    breakdown_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF compression (default: 150)"
    )
    breakdown_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if files exist"
    )
    breakdown_parser.set_defaults(func=breakdown_command)

    # OCR Pages subcommand (new workflow)
    ocr_pages_parser = subparsers.add_parser(
        "ocr-pages",
        help="Page-level OCR (for refined breakdown workflow)",
        description="Extract text from each PDF page individually for refined breakdown"
    )
    ocr_pages_parser.add_argument(
        "-i", "--input",
        help="Path to PDF file (default: auto-detect from output directory)"
    )
    ocr_pages_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    ocr_pages_parser.add_argument(
        "--start-page",
        type=int,
        help="First page to process (default: 1)"
    )
    ocr_pages_parser.add_argument(
        "--end-page",
        type=int,
        help="Last page to process (default: all pages)"
    )
    ocr_pages_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel OCR workers (default: from config or 5)"
    )
    ocr_pages_parser.set_defaults(func=ocr_pages_command)

    # OCR subcommand (DEPRECATED - legacy chapter-based workflow)
    ocr_parser = subparsers.add_parser(
        "ocr",
        help="[DEPRECATED] Chapter-based OCR (use ocr-pages → refine instead)",
        description="⚠️ DEPRECATED: This command is part of the legacy workflow. Use 'ocr-pages' + 'refine' instead."
    )
    ocr_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    ocr_parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing pages into chapters (skip OCR)"
    )
    ocr_parser.set_defaults(func=ocr_command)

    # Refine subcommand (refined breakdown with boundary verification)
    refine_parser = subparsers.add_parser(
        "refine",
        help="Refine structure with boundary verification",
        description="Analyze TOC structure and verify section boundaries for precise splitting"
    )
    refine_parser.add_argument(
        "-i", "--input",
        help="Path to PDF file (default: auto-detect from output directory)"
    )
    refine_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    refine_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per unit (default: from config or 8000)"
    )
    refine_parser.set_defaults(func=refine_command)
    
    # Polish subcommand
    polish_parser = subparsers.add_parser(
        "polish",
        help="Polish OCR-extracted markdown files",
        description="Clean up and format OCR-extracted markdown content"
    )
    polish_parser.add_argument(
        "--skip-truncation-check",
        action="store_true",
        help="Skip truncation detection"
    )
    polish_parser.add_argument(
        "--content-type",
        choices=["academic", "japanese", "general", "auto"],
        default="auto",
        help="Type of content to polish (default: auto-detect)"
    )
    polish_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    polish_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent workers (default: from config or 4)"
    )
    polish_parser.add_argument(
        "--use-longest-on-failure",
        action="store_true",
        default=None,
        help="Use longest response when all validation attempts fail (default: from config.yaml)"
    )
    polish_parser.add_argument(
        "--no-use-longest-on-failure",
        dest="use_longest_on_failure",
        action="store_false",
        help="Don't use longest response on failure (overrides config.yaml)"
    )
    polish_parser.set_defaults(func=polish_command)

    # Translate subcommand
    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate polished markdown files",
        description="Translate markdown content to another language"
    )
    translate_parser.add_argument(
        "--source-language",
        help="Source language (default: from config or English)"
    )
    translate_parser.add_argument(
        "--target-language",
        help="Target language (default: from config or Chinese)"
    )
    translate_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    translate_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent workers (default: from config or 4)"
    )
    translate_parser.add_argument(
        "--use-entities",
        action="store_true",
        default=None,
        help="Force use of extracted entities (auto-detects by default)"
    )
    translate_parser.add_argument(
        "--no-entities",
        action="store_true",
        help="Force disable entity usage even if file exists"
    )
    translate_parser.add_argument(
        "--use-longest-on-failure",
        action="store_true",
        default=None,
        help="Use longest response when all validation attempts fail (default: from config.yaml)"
    )
    translate_parser.add_argument(
        "--no-use-longest-on-failure",
        dest="use_longest_on_failure",
        action="store_false",
        help="Don't use longest response on failure (overrides config.yaml)"
    )
    translate_parser.set_defaults(func=translate_command)

    # Entity extraction subcommand
    entity_parser = subparsers.add_parser(
        "extract-entities",
        help="Extract characters, places, and terms for translation consistency",
        description="Analyze PDF to extract entities that need consistent translation"
    )
    entity_parser.add_argument(
        "-i", "--input",
        default=None,  # Will be resolved to book_folder/input.pdf in the command
        help="Path to input PDF file (default: output/<book_title>/input.pdf)"
    )
    entity_parser.add_argument(
        "--source-lang",
        default="Japanese",
        help="Source language (default: Japanese)"
    )
    entity_parser.add_argument(
        "--target-lang",
        default="Chinese",
        help="Target language for translation (default: Chinese)"
    )
    entity_parser.set_defaults(func=extract_entities_command)
    
    # EPUB generation subcommand (DEPRECATED)
    epub_parser = subparsers.add_parser(
        "epub",
        help="[DEPRECATED] Generate EPUB (use build-epub instead)",
        description="⚠️ DEPRECATED: This command is part of the legacy workflow. Use 'build-epub' instead."
    )
    epub_parser.add_argument(
        "-i", "--input",
        help="Path to PDF file for cover extraction (optional)"
    )
    epub_parser.add_argument(
        "--translated",
        action="store_true",
        help="Generate EPUB from translated markdown instead of polished"
    )
    epub_parser.add_argument(
        "--zip",
        action="store_true",
        help="Create password-protected ZIP file after EPUB generation"
    )
    epub_parser.add_argument(
        "--relevel",
        action="store_true",
        help="Use LLM to analyze and re-level the book structure (adjust heading hierarchy)"
    )
    epub_parser.add_argument(
        "--global-footnotes",
        action="store_true",
        help="Force global footnotes (use last definition, ignore previous definitions)"
    )
    epub_parser.set_defaults(func=epub_command)

    # Build EPUB subcommand (toc_tree.json driven - new approach)
    build_epub_parser = subparsers.add_parser(
        "build-epub",
        help="Build EPUB from toc_tree.json structure (recommended)",
        description="Create EPUB file using toc_tree.json as the structure authority"
    )
    build_epub_parser.add_argument(
        "--translated",
        action="store_true",
        help="Build EPUB from translated markdown instead of polished"
    )
    build_epub_parser.add_argument(
        "--cover",
        help="Path to cover image file"
    )
    build_epub_parser.set_defaults(func=build_epub_command)

    # HTML Translation subcommand (direct HTML translation for EPUB/AZW3/MOBI)
    translate_html_parser = subparsers.add_parser(
        "translate-html",
        help="Translate EPUB/AZW3/MOBI content directly (preserves HTML structure)",
        description="Translate ebook XHTML content directly. AZW3/MOBI files are auto-converted to EPUB."
    )
    translate_html_parser.add_argument(
        "-i", "--input",
        help="Input file: EPUB, AZW3, or MOBI (default: output/<book_title>/input.epub)"
    )
    translate_html_parser.add_argument(
        "--source-language",
        help="Source language (default: from config or Japanese)"
    )
    translate_html_parser.add_argument(
        "--target-language",
        help="Target language (default: from config or Chinese)"
    )
    translate_html_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    translate_html_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent workers (default: from config or 4)"
    )
    translate_html_parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (use existing html_units/)"
    )
    translate_html_parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip translation step (only extract)"
    )
    translate_html_parser.add_argument(
        "--use-entities",
        action="store_true",
        default=None,
        help="Force use of extracted entities"
    )
    translate_html_parser.add_argument(
        "--no-entities",
        action="store_true",
        help="Force disable entity usage"
    )
    translate_html_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only translate first N files (rest are copied untranslated for testing)"
    )
    translate_html_parser.set_defaults(func=translate_html_command)

    # Build HTML EPUB subcommand (rebuild EPUB with translated HTML)
    build_html_epub_parser = subparsers.add_parser(
        "build-html-epub",
        help="Build EPUB from translated HTML (preserves original formatting)",
        description="Rebuild EPUB by replacing XHTML content with translations. AZW3/MOBI auto-converted."
    )
    build_html_epub_parser.add_argument(
        "-i", "--input",
        help="Input file: EPUB, AZW3, or MOBI (default: output/<book_title>/input.epub)"
    )
    build_html_epub_parser.add_argument(
        "-o", "--output",
        help="Path to output EPUB file (default: <book_title>_translated.epub)"
    )
    build_html_epub_parser.set_defaults(func=build_html_epub_command)

    # Patch paper structure subcommand
    patch_parser = subparsers.add_parser(
        "patch-paper",
        help="Patch book structure for academic papers (single chapter mode)",
        description="Modify book_structure.json to treat entire document as one chapter"
    )
    patch_parser.add_argument(
        "--chapter-name",
        help="Name for the single chapter (default: use document title)"
    )
    patch_parser.add_argument(
        "--no-preserve-toc",
        action="store_true",
        help="Don't preserve original TOC entries as metadata"
    )
    patch_parser.set_defaults(func=patch_paper_command)

    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
