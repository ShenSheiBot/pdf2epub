#!/usr/bin/env python3
"""
Unified CLI for pdf2epub markdown processing.

This module provides a single entrypoint for all markdown processing operations
including polishing OCR output and translating content.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from pdf2epub.utils.logging_config import configure_logging
from pdf2epub.utils.common import load_config
from pdf2epub.utils.network_utils import set_llm_trace_path

# Configure logger
logger = configure_logging()


def polish_command(args):
    """Handle the polish subcommand - uses V2 pipeline."""
    from .commands import polish_v2_command
    return polish_v2_command(args)



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
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
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
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")

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

    # Preprocess PDF: copy to output dir + add page stamps + compress
    from pdf2epub.utils.pdf_utils import preprocess_pdf
    pdf_path = preprocess_pdf(pdf_path, output_dir)

    logger.info(f"Starting page-level OCR for: {book_title}")

    # Get OCR settings from config
    ocr_config = config.get('ocr', {})
    backend = ocr_config.get('backend', 'mistral')
    backend_config = ocr_config.get('backends', {}).get(backend, {})
    max_workers = args.max_workers or backend_config.get(
        'max_workers',
        ocr_config.get('vision', {}).get('max_workers', 5),
    )

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
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
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
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
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
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
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
            pipeline.write_translation_report(phase="translation")

        logger.success("HTML translation complete!")
        logger.info(f"Output: {output_dir / 'translated_compressed'}")
        logger.info("Next step: pdf2epub build-html-epub")
        return 0

    except Exception as e:
        logger.error(f"HTML translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def translate_novel_command(args):
    """Handle the translate-novel subcommand (light novel translation)."""
    from pathlib import Path
    import shutil
    from pdf2epub.html_translation.epub_parser import EPUBParser
    from pdf2epub.html_translation.novel_extractor import NovelExtractor
    from pdf2epub.html_translation.novel_translator import NovelTranslator
    from pdf2epub.html_translation.glossary_manager import GlossaryManager
    from pdf2epub.html_translation.builder import BuildConfig, HTMLEpubBuilder, sanitize_filename
    from pdf2epub.utils.llm_client import LLMClient

    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config.yaml")
        return 1

    # Configure file logging
    configure_logging(book_title, "translate-novel")

    output_dir = Path("output") / book_title

    # Set up unified LLM trace
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")

    # Validate input file before creating output directories
    epub_path = Path(args.input) if args.input else None
    if epub_path is None:
        candidate_path = output_dir / "input.epub"
        if candidate_path.exists():
            epub_path = candidate_path

    if epub_path is None or not epub_path.exists():
        logger.error("Input file not found. Use -i to specify EPUB file.")
        return 1

    # Validate glossary early
    if args.glossary:
        glossary_path = Path(args.glossary)
        if not glossary_path.exists():
            logger.error(f"Glossary file not found: {glossary_path}")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Format conversion if needed
    from pdf2epub.utils.ebook_converter import needs_conversion, convert_to_epub

    input_epub = output_dir / "input.epub"
    if needs_conversion(epub_path):
        try:
            epub_path, _ = convert_to_epub(epub_path, output_dir)
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return 1
    elif epub_path.resolve() != input_epub.resolve():
        shutil.copy2(epub_path, input_epub)
        epub_path = input_epub

    # Language settings
    translation_config = config.get("translation", {})
    source_language = args.source_language or translation_config.get("source_language", "Japanese")
    target_language = args.target_language or translation_config.get("target_language", "Chinese")

    logger.info(f"Novel translation: {book_title} ({source_language} → {target_language})")

    try:
        # Step 1: Extract EPUB to plain text
        parser = EPUBParser(str(epub_path))
        extractor = NovelExtractor(parser)
        novel_units_dir = output_dir / "novel_units"
        units = extractor.extract_all(novel_units_dir)

        content_units = [u for u in units if u.has_content]
        logger.info(f"Extracted {len(content_units)} content units from {len(units)} spine items")

        # Apply --limit
        if args.limit:
            content_units = content_units[:args.limit]
            logger.info(f"Limiting to first {args.limit} content units")

        # Use EPUB's actual title for translation
        epub_title = parser.metadata.get('title', book_title)
        logger.info(f"EPUB title: {epub_title}")

        # Step 2: Translate metadata (title + TOC) using Haiku
        from pdf2epub.html_translation.builder import HTMLEpubPipeline
        metadata_models = translation_config.get("models", [
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
        ])
        metadata_config = {**config, "translation_models": metadata_models}
        pipeline = HTMLEpubPipeline(epub_path, output_dir, metadata_config)
        translated_metadata = pipeline.translate_metadata(target_language=target_language)
        logger.info(
            f"Translated metadata: title='{translated_metadata.get('translated_title')}', "
            f"{len(translated_metadata.get('toc', []))} TOC entries"
        )

        # Step 3: Init GlossaryManager
        llm_client = LLMClient(config)
        novel_config = config.get("novel", {})
        glossary_manager = GlossaryManager(
            output_dir=output_dir,
            llm_client=llm_client,
            model_configs=metadata_models,
            max_tokens=novel_config.get("glossary_max_tokens", 1000),
            extract_retries=novel_config.get("glossary_extract_retries", 2),
            dedup_retries=novel_config.get("glossary_dedup_retries", 2),
        )
        glossary_manager.load()

        # Load initial glossary if provided
        if args.glossary:
            glossary_manager.load_initial_glossary(glossary_path)
            glossary_manager.save()

        # Step 4: Translate chapters
        translator = NovelTranslator(
            config=config,
            book_title=epub_title,
            source_language=source_language,
            target_language=target_language,
            glossary_manager=glossary_manager,
            resume=args.resume,
            output_dir=output_dir,
        )

        # Handle --retranslate: single chapter retranslation
        if args.retranslate is not None:
            from pdf2epub.html_translation.novel_translator import NOVEL_TRANSLATE_PROMPT
            spine_idx = args.retranslate
            target_unit = None
            for u in content_units:
                if u.spine_index == spine_idx:
                    target_unit = u
                    break
            if target_unit is None:
                logger.error(f"No content unit found at spine index {spine_idx}")
                return 1

            source_text_raw = target_unit.text_path.read_text(encoding="utf-8")
            chapter_id = f"{target_unit.spine_index:03d}_{target_unit.file_name}"
            logger.info(f"Retranslating chapter {chapter_id} ({len(source_text_raw.splitlines())} lines)")

            # Degeneration guard: truncate repetitive kana sequences
            from pdf2epub.html_translation.chunked_translator import compress_repetitive_source
            source_text = compress_repetitive_source(source_text_raw)

            # Backup current translation
            existing = output_dir / "translated_novel" / f"{chapter_id}.txt"
            if existing.exists():
                backup = existing.with_suffix(".txt.bak")
                import shutil
                shutil.copy2(existing, backup)
                logger.info(f"Backed up existing translation to {backup.name}")

            # Recall glossary (current state — before rollback, in case translation fails)
            glossary_prompt = glossary_manager.recall(source_text)

            # Translate first (if this fails, glossary is untouched)
            translated, exhausted = translator._run_translation(target_unit, source_text, glossary_prompt)
            if exhausted:
                logger.warning(f"  Chapter {chapter_id} exhausted retries (hallucinated)")

            # Image repair
            from pdf2epub.html_translation.novel_translator import repair_images
            translated = repair_images(source_text_raw, translated)

            # Translation succeeded — now rollback + re-extract glossary (atomic)
            glossary_manager.rollback_chapter(chapter_id)
            system_prompt = NOVEL_TRANSLATE_PROMPT
            if glossary_prompt:
                system_prompt = f"{NOVEL_TRANSLATE_PROMPT}\n\n{glossary_prompt}"
            glossary_manager.extract_and_update(
                source_text, translated, chapter_id,
                translation_system_prompt=system_prompt,
            )

            # Write output
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_text(translated, encoding="utf-8")
            logger.info(f"Wrote retranslated chapter to {existing.name}")

            tl_lines = len([l for l in translated.splitlines() if l.strip()])
            src_lines = len([l for l in source_text.splitlines() if l.strip()])
            logger.info(f"Retranslation complete: {src_lines} source → {tl_lines} translated")
            return 0

        summary = translator.translate_all(content_units)
        logger.info(f"Translation summary: {summary}")

        # Step 5: Build EPUB
        if not args.skip_build:
            import json

            translated_xhtml_dir = output_dir / "translated_xhtml"
            translated_xhtml_dir.mkdir(parents=True, exist_ok=True)
            _convert_txt_to_xhtml(
                units=units,
                translated_dir=output_dir / "translated_novel",
                xhtml_dir=translated_xhtml_dir,
                parser=parser,
            )

            metadata_path = output_dir / "translated_metadata.json"
            translated_metadata = None
            if metadata_path.exists():
                translated_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

            if translated_metadata and translated_metadata.get('translated_title'):
                safe_title = sanitize_filename(translated_metadata['translated_title'])
                output_epub = output_dir / f"{safe_title}.epub"
            else:
                output_epub = output_dir / f"{book_title}_translated.epub"

            build_config = BuildConfig(
                original_epub=epub_path,
                translated_dir=translated_xhtml_dir,
                output_path=output_epub,
                book_title=book_title,
                translated_metadata=translated_metadata,
                epubcheck_mode=config.get("html_translation", {}).get(
                    "epubcheck_mode", "warn"
                ),
                epubcheck_path=config.get("html_translation", {}).get(
                    "epubcheck_path"
                ),
            )
            builder = HTMLEpubBuilder(build_config)
            builder.build()
            logger.success(f"Translated EPUB: {output_epub}")

        return 0

    except Exception as e:
        logger.error(f"Novel translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def build_novel_epub_command(args):
    """Handle the build-novel-epub subcommand (rebuild EPUB from translated novel text)."""
    import json
    from pathlib import Path
    from pdf2epub.html_translation.epub_parser import EPUBParser
    from pdf2epub.html_translation.novel_extractor import NovelExtractor
    from pdf2epub.html_translation.builder import BuildConfig, HTMLEpubBuilder, sanitize_filename

    config = load_config(args.config)
    book_title = config.get("title")

    if not book_title:
        logger.error("No title found in config")
        return 1

    configure_logging(book_title, "build-novel-epub")

    output_dir = Path("output") / book_title
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
    epub_path = output_dir / "input.epub"

    if not epub_path.exists():
        logger.error(f"Input EPUB not found: {epub_path}")
        return 1

    try:
        parser_obj = EPUBParser(str(epub_path))
        units = NovelExtractor(parser_obj).extract_all(output_dir / "novel_units")

        translated_dir = output_dir / "translated_novel"
        xhtml_dir = output_dir / "final_xhtml"
        xhtml_dir.mkdir(parents=True, exist_ok=True)

        # Convert txt to xhtml (partial OK — untranslated chapters keep original)
        _convert_txt_to_xhtml(units, translated_dir, xhtml_dir, parser_obj)

        translated_count = sum(1 for u in units if u.has_content and (translated_dir / u.text_path.name).exists())
        total_content = sum(1 for u in units if u.has_content)
        logger.info(f"Translated {translated_count}/{total_content} content units")

        metadata_path = output_dir / "translated_metadata.json"
        translated_metadata = None
        if metadata_path.exists():
            translated_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

        if translated_metadata and translated_metadata.get('translated_title'):
            safe_title = sanitize_filename(translated_metadata['translated_title'])
            output_epub = output_dir / f"{safe_title}.epub"
        else:
            output_epub = output_dir / f"{book_title}_translated.epub"

        build_config = BuildConfig(
            original_epub=epub_path,
            translated_dir=xhtml_dir,
            output_path=output_epub,
            book_title=book_title,
            translated_metadata=translated_metadata,
            epubcheck_mode=config.get("html_translation", {}).get(
                "epubcheck_mode", "warn"
            ),
            epubcheck_path=config.get("html_translation", {}).get(
                "epubcheck_path"
            ),
        )
        builder = HTMLEpubBuilder(build_config)
        builder.build()
        logger.success(f"Built EPUB: {output_epub}")
        return 0

    except Exception as e:
        logger.error(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _convert_txt_to_xhtml(units, translated_dir, xhtml_dir, parser):
    """Restore translated novel text into the original XHTML structure."""
    import html
    import re
    from pathlib import Path

    from pdf2epub.html_translation.compressor import HTMLCompressor
    from pdf2epub.html_translation.novel_extractor import NovelExtractor

    IMAGE_PATTERN = r'\[Image:\s*([^\]]+)\]'

    def nonempty_lines(text):
        return [line.strip() for line in text.splitlines() if line.strip()]

    def normalize_alignment_text(text):
        return re.sub(r'\s+', '', html.unescape(text))

    def regroup_formatting_lines(aligned_pairs, compressed_lines, extractor):
        """Collapse XHTML formatting newlines back into compressor-sized units."""
        grouped_pairs = []
        pair_idx = 0

        for compressed_line in compressed_lines:
            wrapped = (
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f'<body><p>{compressed_line}</p></body></html>'
            )
            expected_text, _ = extractor._convert_xhtml_to_text(wrapped)
            expected = normalize_alignment_text(expected_text)
            source_parts = []
            translated_parts = []
            accumulated = ''

            while pair_idx < len(aligned_pairs) and len(accumulated) < len(expected):
                source_line, translated_line = aligned_pairs[pair_idx]
                candidate = accumulated + normalize_alignment_text(source_line)
                if not expected.startswith(candidate):
                    break
                source_parts.append(source_line)
                translated_parts.append(translated_line)
                accumulated = candidate
                pair_idx += 1

            if not source_parts or accumulated != expected:
                return None

            grouped_pairs.append((''.join(source_parts), ''.join(translated_parts)))

        if pair_idx != len(aligned_pairs):
            return None
        return grouped_pairs

    def prepare_structured_line(source_line, translated_line, compressed_line, compressor):
        """Retain inline tag topology while inserting a plain-text translation."""
        source_images = re.findall(IMAGE_PATTERN, source_line)
        translated_images = re.findall(IMAGE_PATTERN, translated_line)
        if source_images != translated_images:
            raise ValueError(
                "Inline image placeholders changed during translation: "
                f"source={source_images!r}, translated={translated_images!r}"
            )

        if '<' not in compressed_line:
            return html.escape(translated_line, quote=False)

        # Novel translation intentionally uses plain text rather than exposing
        # inline markup to the model. Reuse the compressor's original tag
        # skeleton, clear only its source-language text nodes, and let
        # decompress() restore every recorded attribute.
        fragment = compressor._parse_fragment(compressed_line)
        for element in fragment.iter():
            element.text = None
            if element is not fragment:
                element.tail = None

        if not source_images:
            fragment.text = translated_line
            return compressor._serialize_fragment(fragment)

        image_elements = [
            element
            for element in fragment.iter()
            if element is not fragment
            and isinstance(element.tag, str)
            and element.tag.rsplit('}', 1)[-1].lower() in {'img', 'image'}
        ]
        if len(image_elements) != len(source_images):
            raise ValueError(
                "Cannot align inline image placeholders with original XHTML: "
                f"placeholders={len(source_images)}, elements={len(image_elements)}"
            )

        translated_parts = re.split(IMAGE_PATTERN, translated_line)
        text_parts = translated_parts[::2]
        fragment.text = text_parts[0]
        for image_element, trailing_text in zip(image_elements, text_parts[1:]):
            image_element.tail = trailing_text
        return compressor._serialize_fragment(fragment)

    css_content = ""
    for css_item in getattr(parser, 'resources', {}).get('css', []):
        content = css_item.get('content', b'')
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        css_content += content + "\n"

    compressor = HTMLCompressor()
    extractor = NovelExtractor(parser)
    xhtml_dir.mkdir(parents=True, exist_ok=True)

    for unit in units:
        if not unit.text_path:
            continue

        # Skip image-only pages — let HTMLEpubBuilder preserve original XHTML
        if not unit.has_content:
            continue

        txt_path = translated_dir / unit.text_path.name
        if not txt_path.exists():
            continue

        if not unit.source_href:
            raise ValueError(f"Missing source XHTML href for {unit.file_name}")

        raw_xhtml = parser.get_raw_content(unit.source_href)
        if isinstance(raw_xhtml, bytes):
            raw_xhtml = raw_xhtml.decode('utf-8')

        source_text = unit.text_path.read_text(encoding='utf-8')
        translated_text = txt_path.read_text(encoding='utf-8')
        source_lines = nonempty_lines(source_text)
        translated_lines = nonempty_lines(translated_text)
        if len(source_lines) != len(translated_lines):
            raise ValueError(
                f"Novel line count mismatch for {unit.file_name}: "
                f"source={len(source_lines)}, translated={len(translated_lines)}"
            )

        compressed_text, mapping = compressor.compress(
            raw_xhtml,
            author_css=css_content,
        )
        compressed_lines = nonempty_lines(compressed_text)

        aligned_pairs = [
            (source_line, translated_line)
            for source_line, translated_line in zip(source_lines, translated_lines)
            if not re.fullmatch(IMAGE_PATTERN, source_line)
        ]
        if len(aligned_pairs) != len(compressed_lines):
            regrouped_pairs = regroup_formatting_lines(
                aligned_pairs,
                compressed_lines,
                extractor,
            )
            if regrouped_pairs is None:
                raise ValueError(
                    f"Novel structure mapping mismatch for {unit.file_name}: "
                    f"translated_units={len(aligned_pairs)}, "
                    f"original_xhtml_units={len(compressed_lines)}"
                )
            aligned_pairs = regrouped_pairs

        prepared_lines = []
        for (source_line, translated_line), compressed_line in zip(
            aligned_pairs,
            compressed_lines,
        ):
            prepared_lines.append(
                prepare_structured_line(
                    source_line,
                    translated_line,
                    compressed_line,
                    compressor,
                )
            )

        xhtml = compressor.decompress('\n'.join(prepared_lines), mapping)

        # Use the original XHTML filename
        if unit.source_href:
            out_name = Path(unit.source_href).name
        else:
            out_name = f"{unit.file_name}.xhtml"

        (xhtml_dir / out_name).write_text(xhtml, encoding='utf-8')


def build_html_epub_command(args):
    """Handle the build-html-epub subcommand (rebuild EPUB with translated HTML)."""
    from pathlib import Path
    from pdf2epub.html_translation import HTMLEpubPipeline

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
    set_llm_trace_path(output_dir / "logs" / "llm_trace.jsonl")
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


def translate_arxiv_command(args):
    """Translate an arXiv/local TeX project with compile-gated whole mode."""
    from pdf2epub.tex_translation import (
        TexTranslationOptions,
        TexTranslationPipeline,
    )
    from pdf2epub.tex_translation.arxiv import (
        ArxivSourceResolver,
        slugify_source_id,
    )
    from pdf2epub.tex_translation.compiler import TexCompiler

    config = load_config(args.config)
    tex_config = config.get("tex_translation", {})
    model_config = tex_config.get("model", {})
    repair_config = tex_config.get("repair", {})
    compile_config = tex_config.get("compiler", {})

    options = TexTranslationOptions(
        provider=args.provider or model_config.get("provider", "gemini"),
        model=args.model or model_config.get("model", "gemini-3.1-pro-preview"),
        source_language=(
            args.source_language
            or tex_config.get("source_language", "English")
        ),
        target_language=(
            args.target_language
            or tex_config.get("target_language", "Simplified Chinese")
        ),
        unit_chars=args.unit_chars or tex_config.get("unit_chars", 12_000),
        max_retries=model_config.get("max_retries", 2),
        use_local_cache=not args.no_local_cache,
        repair_enabled=(
            repair_config.get("enabled", True) and not args.no_repair
        ),
        repair_provider=(
            args.repair_provider
            or repair_config.get("provider", "codex")
        ),
        repair_model=(
            args.repair_model
            or repair_config.get("model", "gpt-5.6-luna")
        ),
        retry_fallbacks=args.retry_fallbacks,
        retry_repaired=args.retry_repaired,
    )
    resolver = ArxivSourceResolver()
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        source_id = resolver.source_id(args.source)
        run_dir = Path("output") / "arxiv" / slugify_source_id(source_id)
    run_dir = run_dir.resolve()
    set_llm_trace_path(run_dir / ".pdf2epub" / "logs" / "llm_trace.jsonl")

    pipeline = TexTranslationPipeline(
        config=config,
        options=options,
        compiler=TexCompiler(
            timeout_seconds=(
                args.compile_timeout
                or compile_config.get("timeout_seconds", 180)
            )
        ),
        source_resolver=resolver,
    )
    try:
        result = pipeline.run(
            args.source,
            run_dir=run_dir,
            main_tex=args.main_tex,
            limit=args.limit,
        )
    except Exception as exc:
        logger.error(f"arXiv/TeX translation failed: {exc}")
        return 1

    logger.success(f"Translated TeX project: {result.project_dir}")
    logger.success(f"Compiled PDF: {result.pdf_path}")
    logger.info(f"State summary: {result.summary}")
    return 0



def cancel_batch_command(args):
    """Cancel active batch jobs."""
    from .commands.cancel_batch import run
    return run(args)


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

  # Novel Translation (text mode for light novels):
  pdf2epub translate-novel -i mybook.epub     # Translate with glossary
  pdf2epub build-novel-epub                   # Rebuild EPUB from translations

  # arXiv/TeX Translation (compile-gated whole mode):
  pdf2epub translate-arxiv 2503.01800
  pdf2epub translate-arxiv ./latex-source --main-tex paper.tex

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

    # arXiv / TeX whole-mode translation
    translate_arxiv_parser = subparsers.add_parser(
        "translate-arxiv",
        help="Translate an arXiv or local TeX project and recompile it",
        description=(
            "Download or copy a TeX source tree, translate it unit by unit, "
            "commit only replacements that pass a full XeLaTeX build, and "
            "produce a resumable translated project."
        ),
    )
    translate_arxiv_parser.add_argument(
        "source",
        help="arXiv ID/URL, local source directory, archive, or main .tex file",
    )
    translate_arxiv_parser.add_argument(
        "--main-tex",
        help="Compilation entry point relative to the source root (auto-detected)",
    )
    translate_arxiv_parser.add_argument(
        "--output-dir",
        help="Run directory (default: output/arxiv/<source-id>)",
    )
    translate_arxiv_parser.add_argument(
        "--provider",
        help="Translation provider name (default: tex_translation.model.provider)",
    )
    translate_arxiv_parser.add_argument(
        "--model",
        help="Translation model ID (default: gemini-3.1-pro-preview)",
    )
    translate_arxiv_parser.add_argument(
        "--source-language",
        help="Source language (default: English)",
    )
    translate_arxiv_parser.add_argument(
        "--target-language",
        help="Target language (default: Simplified Chinese)",
    )
    translate_arxiv_parser.add_argument(
        "--unit-chars",
        type=int,
        help="Approximate characters per TeX transaction (default: 12000)",
    )
    translate_arxiv_parser.add_argument(
        "--limit",
        type=int,
        help="Process only the next N pending units; the partial project still compiles",
    )
    translate_arxiv_parser.add_argument(
        "--no-repair",
        action="store_true",
        help="On compile failure, keep the original unit without invoking an agent",
    )
    translate_arxiv_parser.add_argument(
        "--repair-provider",
        help="Whole-mode repair provider (default: codex)",
    )
    translate_arxiv_parser.add_argument(
        "--repair-model",
        help="Whole-mode repair model (default: gpt-5.6-luna)",
    )
    translate_arxiv_parser.add_argument(
        "--retry-fallbacks",
        action="store_true",
        help="Retry units previously kept in the source language",
    )
    translate_arxiv_parser.add_argument(
        "--retry-repaired",
        action="store_true",
        help="Retranslate repaired units while retaining the last compile-safe version",
    )
    translate_arxiv_parser.add_argument(
        "--no-local-cache",
        action="store_true",
        help="Disable the content-addressed local translation response cache",
    )
    translate_arxiv_parser.add_argument(
        "--compile-timeout",
        type=int,
        help="Seconds allowed for each full-project compile (default: 180)",
    )
    translate_arxiv_parser.set_defaults(func=translate_arxiv_command)

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

    # Novel Translation subcommand (text-mode for light novels)
    translate_novel_parser = subparsers.add_parser(
        "translate-novel",
        help="Translate light novel EPUB (text mode + glossary)",
        description="Translate light novel EPUB using sliding window with glossary management. "
                    "Optimized for small-context models like murasaki-14b."
    )
    translate_novel_parser.add_argument(
        "-i", "--input",
        help="Input EPUB file"
    )
    translate_novel_parser.add_argument(
        "--source-language",
        help="Source language (default: Japanese)"
    )
    translate_novel_parser.add_argument(
        "--target-language",
        help="Target language (default: from config or Chinese)"
    )
    translate_novel_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    translate_novel_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only translate first N content chapters (for testing)"
    )
    translate_novel_parser.add_argument(
        "--retranslate",
        type=int,
        default=None,
        help="Retranslate a single chapter by spine index (rolls back glossary, retranslates, re-extracts)"
    )
    translate_novel_parser.add_argument(
        "--glossary",
        help="Path to initial glossary file (for series continuation)"
    )
    translate_novel_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip EPUB building step (only translate)"
    )
    translate_novel_parser.set_defaults(func=translate_novel_command)

    # Build Novel EPUB subcommand (rebuild from translated text, no re-translation)
    build_novel_epub_parser = subparsers.add_parser(
        "build-novel-epub",
        help="Build EPUB from translated novel text (no re-translation)",
        description="Rebuild EPUB from existing translated .txt files. Supports partial translation — untranslated chapters keep original content."
    )
    build_novel_epub_parser.set_defaults(func=build_novel_epub_command)

    # Patch paper structure subcommand

    # Cancel batch subcommand
    cancel_batch_parser = subparsers.add_parser(
        "cancel-batch",
        help="Cancel active batch jobs",
        description="Cancel all active batch jobs for the current project"
    )
    cancel_batch_parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    cancel_batch_parser.add_argument(
        "--all",
        action="store_true",
        help="Cancel ALL batch jobs from API (not just from state files)"
    )
    cancel_batch_parser.set_defaults(func=cancel_batch_command)

    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
