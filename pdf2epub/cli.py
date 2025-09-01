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
from pdf2epub.processors import PolishProcessor, TranslateProcessor

# Configure logger
logger = configure_logging()


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


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
    
    # Initialize the polish processor
    processor = PolishProcessor(
        config=config,
        book_title=book_title,
        max_workers=args.max_workers,
        resume=args.resume,
        skip_truncation_check=args.skip_truncation_check,
        polish_models=config.get("polish_models"),
        content_type=args.content_type
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
        add_page_number_patches,
        GeminiClient,
        compress_pdf,
        analyze_book_structure,
        save_book_structure
    )
    
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Gemini API
    gemini_client = GeminiClient(api_key)
    
    # Copy input PDF to output directory
    target_pdf = output_dir / "input_original.pdf"
    if not target_pdf.exists() or args.force:
        import shutil
        shutil.copy2(input_pdf, target_pdf)
        logger.info(f"Copied input PDF to {target_pdf}")
    
    # Add page number patches if requested
    if args.add_page_numbers:
        logger.info("Adding page number patches...")
        add_page_number_patches(target_pdf, target_pdf)
    
    # Compress PDF
    compressed_pdf = output_dir / "input.pdf"
    if not compressed_pdf.exists() or args.force:
        logger.info("Compressing PDF...")
        compress_pdf(target_pdf, compressed_pdf, dpi=args.dpi)
    
    # Analyze book structure
    logger.info("Analyzing book structure...")
    structure = analyze_book_structure(compressed_pdf, gemini_client)
    
    # Save structure
    save_book_structure(structure, book_title)
    logger.success(f"Book structure saved for {book_title}")
    
    return 0


def ocr_command(args):
    """Handle the OCR subcommand."""
    # Import based on whether Japanese OCR is requested
    if args.japanese or args.backend:
        from pdf2epub.ocr_chapters_jp import (
            load_config,
            load_book_structure,
            get_backend_module,
            process_chapters_parallel
        )
    else:
        from pdf2epub.ocr_chapters import (
            load_config,
            load_book_structure,
            process_chapter,
            GeminiClient
        )
    
    # Load configuration
    config = load_config(args.config if hasattr(args, 'config') else 'config.yaml')
    book_title = config.get("title", "book")
    
    logger.info(f"Starting OCR for: {book_title}")
    
    # For Japanese OCR
    if args.japanese or args.backend:
        # Determine backend
        backend = args.backend or config.get('jp_ocr_backend', 'azure')
        logger.info(f"Using Japanese OCR backend: {backend}")
        
        # Import backend functions
        init_client, process_page_func = get_backend_module(backend)
        
        # Load book structure
        structure = load_book_structure(book_title)
        if not structure:
            logger.error(f"Book structure not found. Run 'breakdown' first.")
            return 1
        
        chapters = structure["chapters"]
        
        # Add chapter indices
        for idx, chapter in enumerate(chapters):
            chapter["index"] = idx + 1
        
        # Get PDF path
        pdf_path = Path("output") / book_title / "input_original.pdf"
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return 1
        
        # Initialize client
        client = init_client(config)
        
        # Process chapters
        process_chapters_parallel(
            chapters=chapters,
            pdf_path=pdf_path,
            client=client,
            process_page_func=process_page_func,
            book_title=book_title,
            max_workers=args.max_workers,
            resume=args.resume
        )
    else:
        # Regular OCR
        api_key = config.get("google_api_key")
        if not api_key:
            logger.error("Google API key not found in config.yaml")
            return 1
        
        # Load book structure
        structure = load_book_structure(book_title)
        if not structure:
            logger.error(f"Book structure not found. Run 'breakdown' first.")
            return 1
        
        # Initialize Gemini client
        gemini_client = GeminiClient(api_key)
        
        # Get PDF path
        pdf_path = Path("output") / book_title / "input.pdf"
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return 1
        
        # Process each chapter
        chapters = structure.get("chapters", [])
        for chapter in chapters:
            logger.info(f"Processing {chapter['title']}...")
            process_chapter(chapter, pdf_path, gemini_client, book_title)
    
    logger.success("OCR processing completed")
    return 0


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
        
        sys.argv = new_argv
        
        # Call the main function
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
    source_language = args.source_language or config.get("source_language", "English")
    
    logger.info(f"Starting translation for: {book_title}")
    logger.info(f"Translation: {source_language} → {target_language}")
    
    # Initialize the translation processor
    processor = TranslateProcessor(
        config=config,
        book_title=book_title,
        source_language=source_language,
        target_language=target_language,
        max_workers=args.max_workers,
        resume=args.resume,
        translation_models=config.get("translation_models")
    )
    
    # Process all files
    summary = processor.process_all_files()
    
    # Check for errors
    if summary.get("error"):
        logger.error(f"Processing failed: {summary['error']}")
        return 1
    
    # Return exit code based on success rate
    return 0 if summary.get("success_rate", 0) == 1.0 else 1


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="PDF to EPUB markdown processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline for a book:
  python -m pdf2epub.cli breakdown -i mybook.pdf
  python -m pdf2epub.cli ocr
  python -m pdf2epub.cli polish
  python -m pdf2epub.cli epub
  
  # Japanese book pipeline:
  python -m pdf2epub.cli breakdown -i manga.pdf
  python -m pdf2epub.cli ocr --japanese --backend vision
  python -m pdf2epub.cli polish --content-type japanese
  python -m pdf2epub.cli epub
  
  # Academic book with translation:
  python -m pdf2epub.cli breakdown -i thesis.pdf --add-page-numbers
  python -m pdf2epub.cli ocr
  python -m pdf2epub.cli polish --content-type academic
  python -m pdf2epub.cli translate --target-language Chinese
  python -m pdf2epub.cli epub
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
        "--add-page-numbers",
        action="store_true",
        help="Add page number patches to PDF"
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
    
    # OCR subcommand
    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Extract text from PDF pages using OCR",
        description="Process PDF pages with OCR to extract text content"
    )
    ocr_parser.add_argument(
        "--japanese",
        action="store_true",
        help="Use Japanese OCR processing"
    )
    ocr_parser.add_argument(
        "--backend",
        choices=['azure', 'vision', 'vllm'],
        help="OCR backend for Japanese (overrides config)"
    )
    ocr_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    ocr_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers"
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
        default=4,
        help="Maximum number of concurrent workers"
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
        default=4,
        help="Maximum number of concurrent workers"
    )
    translate_parser.set_defaults(func=translate_command)
    
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
    epub_parser.set_defaults(func=epub_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())