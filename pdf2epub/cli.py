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
    """Handle the polish subcommand."""
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    
    if not book_title:
        logger.error("No title found in config.yaml")
        return 1
    
    logger.info(f"Starting polish process for: {book_title}")
    if args.content_type != "auto":
        logger.info(f"Content type: {args.content_type}")
    
    # Load book structure if available
    output_dir = Path("output") / book_title
    structure_file = output_dir / "book_structure.json"
    book_structure = None
    
    if structure_file.exists():
        import json
        with open(structure_file, 'r', encoding='utf-8') as f:
            book_structure = json.load(f)
        logger.info("Loaded book structure for context-aware polishing")
    
    # Initialize the polish processor
    # Get use_longest_on_failure from config if not explicitly set in CLI
    validation_config = config.get('validation_strategy', {})
    # Use config value as default, CLI flag overrides if provided
    if hasattr(args, 'use_longest_on_failure') and args.use_longest_on_failure is not None:
        use_longest_on_failure = args.use_longest_on_failure
    else:
        use_longest_on_failure = validation_config.get('use_longest_on_failure', False)

    processor = PolishProcessor(
        config=config,
        book_title=book_title,
        max_workers=args.max_workers if args.max_workers is not None else config.get('max_concurrent_workers', 4),
        resume=args.resume,
        skip_truncation_check=args.skip_truncation_check,
        polish_models=config.get("polish_models"),
        content_type=args.content_type,
        use_longest_on_failure=use_longest_on_failure,
        book_structure=book_structure
    )
    
    # Process all files
    summary = processor.process_all_files()
    
    # Check for errors
    if summary.get("error"):
        logger.error(f"Processing failed: {summary['error']}")
        return 1
    
    # Return exit code based on success rate
    return 0 if summary.get("success_rate", 0) == 1.0 else 1


def breakdown_command(args):
    """Handle the breakdown subcommand."""
    from pdf2epub.breakdown import (
        load_config as load_breakdown_config,
        preprocess_pdf,
        analyze_pdf_structure,
        detect_front_and_back_matter
    )
    from pdf2epub.utils.network_utils import GeminiClient
    import json
    
    # Load configuration
    config = load_breakdown_config(args.config)
    api_key = config.get("google_api_key")
    
    if not api_key:
        logger.error("Google API key not found in config.yaml")
        return 1
    
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
    gemini_client = GeminiClient(api_key)
    
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


def epub_input_command(args):
    """Handle the EPUB input subcommand (breakdown + conversion)."""
    from pdf2epub.epub_breakdown import breakdown_epub
    from pdf2epub.epub_to_markdown import convert_epub_to_markdown
    from pathlib import Path

    epub_path = Path(args.input)

    if not epub_path.exists():
        logger.error(f"EPUB file not found: {epub_path}")
        return 1

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Auto-detect from EPUB metadata
        from pdf2epub.epub_input.epub_parser import EPUBParser
        temp_parser = EPUBParser(str(epub_path))
        book_title = temp_parser.metadata['title']
        # Clean title for directory name
        import re
        clean_title = re.sub(r'[^\w\s-]', '', book_title)
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        output_dir = Path("output") / clean_title

    logger.info(f"Processing EPUB: {epub_path}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Breakdown (analyze structure)
    logger.info("Step 1/2: Analyzing EPUB structure...")
    try:
        structure = breakdown_epub(
            str(epub_path),
            str(output_dir),
            skip_patterns=args.skip_patterns,
            front_matter_position_threshold=args.front_matter_threshold,
            back_matter_position_threshold=args.back_matter_threshold
        )
    except Exception as e:
        logger.error(f"Failed to analyze EPUB structure: {e}")
        return 1

    # Step 2: Convert to Markdown
    logger.info("Step 2/2: Converting to Markdown...")
    try:
        convert_epub_to_markdown(
            str(epub_path),
            structure_file=str(output_dir / "book_structure.json"),
            output_dir=str(output_dir),
            resume=args.resume,
            config_path=args.config,
            max_tokens_per_part=8000  # Use default for now
        )
    except Exception as e:
        logger.error(f"Failed to convert EPUB to Markdown: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.success(f"EPUB processing complete! Output: {output_dir / 'polished_markdown'}")
    logger.info("Next steps:")
    logger.info("  - For translation: pdf2epub translate --target-language <language>")
    logger.info("  - For EPUB generation: pdf2epub epub")

    return 0


def ocr_command(args):
    """Handle the OCR subcommand."""
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
        save_entities,
        GeminiClient
    )
    
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

    # Get API key
    api_key = config.get("google_api_key")
    if not api_key:
        logger.error("Google API key not found in config.yaml")
        return 1

    logger.info(f"Extracting entities from: {book_title}")
    logger.info(f"Language pair: {args.source_lang} → {args.target_lang}")

    # Initialize Gemini client
    gemini_client = GeminiClient(api_key)

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
    """Handle the epub generation subcommand."""
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


def translate_command(args):
    """Handle the translate subcommand."""
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    
    if not book_title:
        logger.error("No title found in config.yaml")
        return 1
    
    # Get language settings
    target_language = args.target_language or config.get("target_language", "Chinese")
    source_language = args.source_language or config.get("source_language", "Japanese")
    
    logger.info(f"Starting translation for: {book_title}")
    logger.info(f"Translation: {source_language} → {target_language}")
    
    # Determine use_entities value based on flags
    if args.no_entities:
        use_entities = False
    elif args.use_entities:
        use_entities = True
    else:
        use_entities = None  # Auto-detect
    
    # Initialize the translation processor
    processor = TranslateProcessor(
        config=config,
        book_title=book_title,
        source_language=source_language,
        target_language=target_language,
        max_workers=args.max_workers if args.max_workers is not None else config.get('max_concurrent_workers', 4),
        resume=args.resume,
        translation_models=config.get("translation_models"),
        use_entities=use_entities,
        use_longest_on_failure=args.use_longest_on_failure if args.use_longest_on_failure is not None else config.get('validation_strategy', {}).get('use_longest_on_failure', False)
    )
    
    # Process all files
    summary = processor.process_all_files()
    
    # Check for errors
    if summary.get("error"):
        logger.error(f"Processing failed: {summary['error']}")
        return 1
    
    # Return exit code based on success rate
    return 0 if summary.get("success_rate", 0) == 1.0 else 1


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
Examples:
  # Complete pipeline for a PDF book:
  pdf2epub breakdown -i mybook.pdf
  pdf2epub ocr
  pdf2epub polish
  pdf2epub epub

  # EPUB as input (already clean text):
  pdf2epub epub-input -i mybook.epub
  pdf2epub translate --target-language Chinese  # Optional
  pdf2epub epub

  # EPUB with custom filtering:
  pdf2epub epub-input -i mybook.epub --skip-pattern "^Cover$" --skip-pattern "Copyright"

  # Japanese book with translation (set ocr_backend: vision in config.yaml):
  pdf2epub breakdown -i manga.pdf
  pdf2epub extract-entities -i manga.pdf  # Extract for consistency
  pdf2epub ocr
  pdf2epub polish --content-type japanese
  pdf2epub translate --target-language Chinese  # Auto-uses entities
  pdf2epub epub

  # Academic book with translation:
  pdf2epub breakdown -i thesis.pdf
  pdf2epub ocr
  pdf2epub polish --content-type academic
  pdf2epub translate --target-language Chinese
  pdf2epub epub
        """
    )
    
    # Global arguments
    parser.add_argument("-c", "--config", default="config.yaml", 
                        help="Path to config file")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Breakdown subcommand
    breakdown_parser = subparsers.add_parser(
        "breakdown",
        help="Analyze PDF structure and prepare for OCR",
        description="Extract book structure, compress PDF, and prepare for processing"
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

    # EPUB Input subcommand
    epub_input_parser = subparsers.add_parser(
        "epub-input",
        help="Process EPUB file as input (extract to polished markdown)",
        description="Analyze EPUB structure and convert to polished markdown, ready for translation or EPUB generation"
    )
    epub_input_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input EPUB file"
    )
    epub_input_parser.add_argument(
        "-o", "--output",
        help="Output directory (default: auto-detect from EPUB title)"
    )
    epub_input_parser.add_argument(
        "--skip-pattern",
        action="append",
        dest="skip_patterns",
        help="Regex pattern to skip chapters (can be used multiple times)"
    )
    epub_input_parser.add_argument(
        "--front-matter-threshold",
        type=int,
        default=5,
        help="First N spine items to mark as front matter (default: 5)"
    )
    epub_input_parser.add_argument(
        "--back-matter-threshold",
        type=int,
        default=3,
        help="Last N spine items to mark as back matter (default: 3)"
    )
    epub_input_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    epub_input_parser.set_defaults(func=epub_input_command)

    # OCR subcommand
    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Extract text from PDF pages using OCR",
        description="Process PDF pages with OCR to extract text content"
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
    
    # EPUB generation subcommand
    epub_parser = subparsers.add_parser(
        "epub",
        help="Generate EPUB from polished markdown",
        description="Create EPUB file from processed markdown content"
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
