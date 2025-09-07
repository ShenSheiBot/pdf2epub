#!/usr/bin/env python3
"""
Utility for tuning content splitting on large chapters.

This module provides a function to test and tune the content splitter
on chapters that exceed a certain token limit, allowing fine-tuning
of splitting behavior without running the full polish process.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import tiktoken
from loguru import logger

from pdf2epub.utils.common import load_config
from pdf2epub.utils.llm_client import LLMClient
from pdf2epub.processors.utils.content_splitter import split_content


# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def tune_content_splitter(
    book_title: Optional[str] = None,
    config_path: str = "config.yaml",
    min_tokens: int = 20000,
    content_type: str = "academic",
    output_dir: Optional[str] = None,
    chapters_to_test: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Test content splitting on large chapters without running full polish.
    
    This function mimics the logic of 'pdf2epub polish --resume --content-type academic'
    but only runs the splitter on chapters with more than the specified token count.
    
    Args:
        book_title: Title of the book (if None, uses config)
        config_path: Path to config.yaml file
        min_tokens: Minimum tokens for a chapter to be split (default: 20000)
        content_type: Content type for splitting ("academic", "japanese", "general", "auto")
        output_dir: Optional output directory for split results
        chapters_to_test: Optional list of specific chapter files to test (e.g., ["chapter_1.md"])
    
    Returns:
        Dictionary with splitting results and statistics
    """
    # Load configuration
    config = load_config(config_path)
    
    if not book_title:
        book_title = config.get("title")
        if not book_title:
            raise ValueError("No book title found in config or parameters")
    
    logger.info(f"Tuning content splitter for: {book_title}")
    logger.info(f"Content type: {content_type}")
    logger.info(f"Minimum tokens for splitting: {min_tokens:,}")
    
    # Setup directories
    base_dir = Path("output") / book_title
    ocr_dir = base_dir / "ocr_markdown"
    
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR markdown directory not found: {ocr_dir}")
    
    # Setup output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = base_dir / "split_tuning"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM client
    llm_client = LLMClient(config)
    
    # Get model configs (use polish_models from config)
    model_configs = config.get("polish_models")
    if not model_configs:
        # Default to flash model for testing
        model_configs = [
            {"provider": "gemini", "model": "gemini-1.5-flash", "max_retries": 2}
        ]
    
    # Determine max tokens based on models (same logic as PolishProcessor)
    max_tokens_per_part = 8000  # Conservative default
    limited_model_patterns = ["flash", "haiku", "-mini", "seek"]
    
    has_limited_model = any(
        any(pattern in model_config.get("model", "").lower() 
            for pattern in limited_model_patterns)
        for model_config in model_configs
    )
    
    if not has_limited_model:
        max_tokens_per_part = 20000
        logger.info(f"Using max_tokens_per_part={max_tokens_per_part:,} (no limited-context models)")
    else:
        logger.info(f"Using max_tokens_per_part={max_tokens_per_part:,} (limited-context model detected)")
    
    # Find chapters to process
    if chapters_to_test:
        # Use specified chapters
        chapter_files = [ocr_dir / ch for ch in chapters_to_test]
        chapter_files = [f for f in chapter_files if f.exists()]
    else:
        # Find all markdown files
        chapter_files = sorted(ocr_dir.glob("*.md"))
    
    results = {
        "book_title": book_title,
        "content_type": content_type,
        "min_tokens": min_tokens,
        "max_tokens_per_part": max_tokens_per_part,
        "chapters_processed": [],
        "chapters_skipped": [],
        "total_chapters": len(chapter_files),
        "split_prompts": {},
    }
    
    # Process each chapter
    for chapter_file in chapter_files:
        chapter_name = chapter_file.name
        logger.info(f"\nProcessing: {chapter_name}")
        
        # Read content
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count tokens
        token_count = len(tokenizer.encode(content))
        logger.info(f"  Token count: {token_count:,}")
        
        # Skip if below threshold
        if token_count < min_tokens:
            logger.info(f"  Skipping (below {min_tokens:,} tokens)")
            results["chapters_skipped"].append({
                "name": chapter_name,
                "tokens": token_count
            })
            continue
        
        # Run the splitter
        logger.info(f"  Splitting content...")
        try:
            parts = split_content(
                content=content,
                max_tokens=max_tokens_per_part,
                llm_client=llm_client,
                model_configs=model_configs,
                strategy=content_type,
            )
            
            # Load the saved split prompt (if it exists from our temp modification)
            split_prompt_file = Path("last_split_prompt.txt")
            split_prompt = None
            if split_prompt_file.exists():
                with open(split_prompt_file, 'r', encoding='utf-8') as f:
                    split_prompt = f.read()
                # Clean up the prompt - remove the large structural map for readability
                if "Document structure:" in split_prompt:
                    split_prompt = split_prompt[:split_prompt.index("Document structure:")]
                    split_prompt += "[Document structure omitted for brevity]"
                results["split_prompts"][chapter_name] = split_prompt
            
            # Calculate part statistics
            part_stats = []
            for i, part in enumerate(parts, 1):
                part_tokens = len(tokenizer.encode(part))
                part_stats.append({
                    "part": i,
                    "tokens": part_tokens,
                    "chars": len(part),
                    "first_100_chars": part[:100].replace('\n', '\\n'),
                })
                
                # Save part to file
                part_file = output_path / f"{chapter_file.stem}.part{i}.md"
                with open(part_file, 'w', encoding='utf-8') as f:
                    f.write(part)
            
            # Log results
            logger.success(f"  Split into {len(parts)} parts:")
            for stat in part_stats:
                logger.info(f"    Part {stat['part']}: {stat['tokens']:,} tokens")
            
            # Check if any part exceeds limit
            exceeds_limit = any(s["tokens"] > max_tokens_per_part for s in part_stats)
            if exceeds_limit:
                logger.warning(f"  WARNING: Some parts exceed {max_tokens_per_part:,} token limit!")
            
            results["chapters_processed"].append({
                "name": chapter_name,
                "original_tokens": token_count,
                "num_parts": len(parts),
                "parts": part_stats,
                "exceeds_limit": exceeds_limit,
            })
            
        except Exception as e:
            logger.error(f"  Error splitting chapter: {e}")
            results["chapters_processed"].append({
                "name": chapter_name,
                "original_tokens": token_count,
                "error": str(e),
            })
    
    # Save results summary
    summary_file = output_path / "split_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total chapters: {results['total_chapters']}")
    logger.info(f"Chapters processed: {len(results['chapters_processed'])}")
    logger.info(f"Chapters skipped: {len(results['chapters_skipped'])}")
    
    if results['chapters_processed']:
        logger.info("\nProcessed chapters:")
        for ch in results['chapters_processed']:
            if 'error' in ch:
                logger.error(f"  {ch['name']}: ERROR - {ch['error']}")
            else:
                status = "⚠️ EXCEEDS LIMIT" if ch['exceeds_limit'] else "✓"
                logger.info(f"  {ch['name']}: {ch['original_tokens']:,} tokens → {ch['num_parts']} parts {status}")
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Summary: {summary_file}")
    
    # Clean up temp file
    split_prompt_file = Path("last_split_prompt.txt")
    if split_prompt_file.exists():
        split_prompt_file.unlink()
    
    return results


def main():
    """Command-line interface for the splitter tuner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tune content splitting on large chapters"
    )
    parser.add_argument(
        "--book-title",
        help="Book title (defaults to config.yaml)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=20000,
        help="Minimum tokens for a chapter to be split (default: 20000)",
    )
    parser.add_argument(
        "--content-type",
        choices=["academic", "japanese", "general", "auto"],
        default="academic",
        help="Content type for splitting (default: academic)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for split results (default: output/BOOK_TITLE/split_tuning)",
    )
    parser.add_argument(
        "--chapters",
        nargs="+",
        help="Specific chapter files to test (e.g., chapter_1.md chapter_2.md)",
    )
    
    args = parser.parse_args()
    
    try:
        results = tune_content_splitter(
            book_title=args.book_title,
            config_path=args.config,
            min_tokens=args.min_tokens,
            content_type=args.content_type,
            output_dir=args.output_dir,
            chapters_to_test=args.chapters,
        )
        return 0 if results else 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
