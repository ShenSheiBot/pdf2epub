"""
Base processor class for markdown transformations.

This module provides the abstract base class for all markdown processors,
handling common functionality like progress tracking, retry logic, and
concurrent processing.
"""

import json
import time
import tiktoken
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from pdf2epub.utils.llm_client import LLMClient
from .utils.content_splitter import split_content_intelligently

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


class BaseMarkdownProcessor(ABC):
    """Abstract base class for markdown processors."""
    
    def __init__(
        self,
        config: Dict,
        book_title: str,
        input_dir: str,
        output_dir: str,
        max_workers: int = 4,
        resume: bool = False,
        use_longest_on_failure: bool = False
    ):
        """
        Initialize the base processor.
        
        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            input_dir: Input directory name (e.g., "polished_markdown")
            output_dir: Output directory name (e.g., "translated")
            max_workers: Maximum number of concurrent workers
            resume: Whether to resume from previous progress
            use_longest_on_failure: If True, use longest response when all attempts fail validation
        """
        self.config = config
        self.book_title = book_title
        self.max_workers = max_workers
        self.resume = resume
        self.use_longest_on_failure = use_longest_on_failure
        
        # Setup directories
        self.input_dir = Path("output") / book_title / input_dir
        self.output_dir = Path("output") / book_title / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client
        self.llm_client = LLMClient(config)
        
        # Progress tracking
        self.progress_file = self.output_dir / f"{self.get_progress_filename()}.json"
        self.progress = self.load_or_create_progress()
    
    @abstractmethod
    def get_progress_filename(self) -> str:
        """Get the name for the progress file."""
        pass
    
    @abstractmethod
    def get_progress_key(self) -> str:
        """Get the key used in progress tracking."""
        pass
    
    @abstractmethod
    def process_content(
        self,
        content: str,
        file_name: str,
        **kwargs
    ) -> str:
        """
        Process markdown content.
        
        Args:
            content: The markdown content to process
            file_name: Name of the file being processed
            **kwargs: Additional processor-specific arguments
        
        Returns:
            Processed markdown content
        """
        pass
    
    @abstractmethod
    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate the processed output.
        
        Args:
            original: Original content
            processed: Processed content
            file_name: Name of the file
        
        Returns:
            Tuple of (is_valid, reason)
        """
        pass
    
    @abstractmethod
    def get_operation_name(self, file_name: str) -> str:
        """
        Get the operation name for logging.
        
        Args:
            file_name: Name of the file being processed
        
        Returns:
            Operation name string
        """
        pass
    
    def load_or_create_progress(self) -> Dict:
        """Load existing progress or create new progress tracking."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
                # Ensure structure is correct
                key = self.get_progress_key()
                if key not in progress:
                    progress[key] = {}
                return progress
        
        # Create new progress
        progress = {
            self.get_progress_key(): {},
            "total_files": 0,
            "processor": self.__class__.__name__
        }
        return progress
    
    def save_progress(self):
        """Save progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def process_with_splitting(
        self,
        content: str,
        file_name: str,
        max_tokens: int = 30000,
        **kwargs
    ) -> Optional[str]:
        """
        Process content by splitting it into smaller parts if validation fails.
        
        Args:
            content: The content to process
            file_name: Name of the file being processed
            max_tokens: Maximum tokens per split
            **kwargs: Additional processor-specific arguments
            
        Returns:
            Processed content or None if all attempts fail
        """
        logger.info(f"Attempting to process {file_name} by splitting into parts")
        
        # Count actual tokens using tiktoken
        estimated_tokens = len(tokenizer.encode(content))
        
        # Split content into parts
        # Use a safe token limit that's 70% of the estimated current size to ensure actual splitting
        safe_token_limit = int(estimated_tokens * 0.7)
        
        # Get model configs for splitting (if available)
        model_configs = None
        if hasattr(self, 'polish_models'):
            model_configs = self.polish_models
        elif hasattr(self, 'translation_models'):
            model_configs = self.translation_models
        
        # Get content type if available
        content_type = getattr(self, 'content_type', 'auto')
        
        parts = split_content_intelligently(content, min(safe_token_limit, max_tokens), self.llm_client, model_configs, content_type)
        
        # Force at least 2 parts if we still got only 1
        if len(parts) == 1:
            # Force split into 2 parts
            logger.info(f"Content estimated at ~{estimated_tokens} tokens, forcing split into 2 parts")
            parts = split_content_intelligently(content, estimated_tokens // 2, self.llm_client, model_configs, content_type)
        
        logger.info(f"Split {file_name} into {len(parts)} parts")
        
        processed_parts = []
        for i, part in enumerate(parts, 1):
            part_name = f"{file_name} (part {i}/{len(parts)})"
            logger.info(f"Processing {part_name}")
            
            # Process each part (Tenacity handles retries)
            try:
                processed_part = self.process_content(
                    content=part,
                    file_name=part_name,
                    **kwargs
                )
                
                # Validate the part
                is_valid, reason = self.validate_output(
                    original=part,
                    processed=processed_part,
                    file_name=part_name
                )
                
                if is_valid:
                    processed_parts.append(processed_part)
                    logger.success(f"Successfully processed {part_name}")
                else:
                    # For parts, use output even if validation fails
                    logger.warning(f"Validation failed for {part_name}: {reason}, using output anyway")
                    processed_parts.append(processed_part)
                    
            except Exception as e:
                # Part processing failed after all Tenacity retries
                logger.error(f"Failed to process {part_name} after all retries: {e}")
                return None  # Abort the entire splitting strategy
        
        # Combine all processed parts
        if len(processed_parts) == len(parts):
            combined = "\n\n".join(processed_parts)
            logger.success(f"Successfully processed all {len(parts)} parts of {file_name}")
            return combined
        else:
            logger.error(f"Only processed {len(processed_parts)}/{len(parts)} parts of {file_name}")
            return None
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> bool:
        """
        Process a single markdown file.
        
        Args:
            input_path: Path to input markdown file
            output_path: Path to output markdown file
            **kwargs: Additional processor-specific arguments
        
        Returns:
            True if successful, False otherwise
        """
        file_name = input_path.name
        logger.info(f"Processing {file_name}")
        
        # Read the content
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"File {file_name} is empty, skipping")
            return False
        
        # Get max validation retries from config (default to 3)
        max_validation_attempts = 3
        if hasattr(self, 'translation_models') and self.translation_models:
            # For translator, use max_retries from model config
            max_validation_attempts = self.translation_models[0].get('max_retries', 3)
        elif hasattr(self, 'polish_models') and self.polish_models:
            # For polisher, use max_retries from model config
            max_validation_attempts = self.polish_models[0].get('max_retries', 3)
        
        last_error = None
        for attempt in range(max_validation_attempts):
            try:
                # Process the content (Tenacity handles API retries internally)
                processed_content = self.process_content(
                    content=content,
                    file_name=file_name,
                    **kwargs
                )
                
                # Validate the output
                is_valid, reason = self.validate_output(
                    original=content,
                    processed=processed_content,
                    file_name=file_name
                )
                
                if not is_valid:
                    # Validation failed - retry
                    if attempt < max_validation_attempts - 1:
                        logger.warning(f"Validation failed for {file_name}: {reason}")
                        logger.info(f"Retrying {file_name} (attempt {attempt + 2}/{max_validation_attempts})...")
                        continue
                    else:
                        # All validation attempts failed
                        last_error = ValueError(f"All {max_validation_attempts} validation attempts failed: {reason}")
                        break
                
                # Save the processed file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                
                logger.success(f"Successfully processed {file_name}")
                return True
                
            except Exception as e:
                # API or other error - also retry
                last_error = e
                if attempt < max_validation_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {file_name}: {e}")
                    logger.info(f"Retrying {file_name} (attempt {attempt + 2}/{max_validation_attempts})...")
                    continue
                else:
                    break
        
        # All attempts failed
        logger.error(f"Failed to process {file_name} after {max_validation_attempts} attempts: {last_error}")
        
        # Try fallback strategy: split into smaller parts
        logger.warning(f"Attempting to process {file_name} by splitting into smaller parts...")
        
        split_result = self.process_with_splitting(
            content=content,
            file_name=file_name,
            **kwargs
        )
        
        if split_result:
            # Save the split result
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(split_result)
            logger.success(f"Successfully processed {file_name} using splitting strategy")
            return True
        else:
            logger.error(f"Failed to process {file_name} with all strategies")
            return False
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all markdown files in the input directory.
        
        Returns:
            Summary statistics
        """
        # Find all markdown files
        all_markdown_files = sorted(self.input_dir.glob("*.md"))
        
        # Filter out combined files that have part files
        markdown_files = []
        combined_files_with_parts = set()
        
        # First pass: identify which files have parts
        for file_path in all_markdown_files:
            if '.part' in file_path.stem:
                # This is a part file, extract the base name
                base_name = file_path.stem.split('.part')[0]
                combined_files_with_parts.add(base_name)
        
        # Second pass: only include files that aren't combined files with parts
        for file_path in all_markdown_files:
            if '.part' in file_path.stem:
                # Include all part files
                markdown_files.append(file_path)
            elif file_path.stem not in combined_files_with_parts:
                # Include files that don't have parts
                markdown_files.append(file_path)
            else:
                # Skip combined files that have parts
                logger.debug(f"Skipping {file_path.name} (using part files instead)")
        
        if not markdown_files:
            logger.error(f"No markdown files found in {self.input_dir}")
            return {"error": "No files found"}
        
        logger.info(f"Found {len(markdown_files)} markdown files to process ({len(combined_files_with_parts)} combined files skipped)")
        
        # Update total files in progress
        self.progress["total_files"] = len(markdown_files)
        self.save_progress()
        
        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for markdown_path in markdown_files:
                # Check if already processed
                file_key = str(markdown_path.stem)
                progress_key = self.get_progress_key()
                
                if self.resume and file_key in self.progress[progress_key]:
                    if self.progress[progress_key][file_key].get("completed", False):
                        logger.info(f"Skipping {markdown_path.name} (already processed)")
                        continue
                
                # Submit task
                output_path = self.output_dir / markdown_path.name
                future = executor.submit(
                    self.process_file,
                    markdown_path,
                    output_path
                )
                futures.append((future, file_key))
            
            # Process completed tasks as they finish
            future_to_key = {future: file_key for future, file_key in futures}
            for future in as_completed(future_to_key):
                file_key = future_to_key[future]
                try:
                    success = future.result()
                    progress_key = self.get_progress_key()
                    
                    if success:
                        self.progress[progress_key][file_key] = {
                            "completed": True,
                            "timestamp": time.time()
                        }
                        logger.info(f"Progress saved for {file_key}")
                    else:
                        self.progress[progress_key][file_key] = {
                            "completed": False,
                            "timestamp": time.time()
                        }
                    
                    self.save_progress()
                    
                except Exception as e:
                    logger.error(f"Error processing {file_key}: {e}")
                    progress_key = self.get_progress_key()
                    self.progress[progress_key][file_key] = {
                        "completed": False,
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    self.save_progress()
        
        # Calculate summary
        progress_key = self.get_progress_key()
        completed = sum(
            1 for info in self.progress[progress_key].values() 
            if info.get("completed", False)
        )
        total = len(markdown_files)
        
        summary = {
            "completed": completed,
            "total": total,
            "success_rate": completed / total if total > 0 else 0
        }
        
        # Log summary
        logger.info(f"\n=== {self.__class__.__name__} Summary ===")
        logger.info(f"Completed: {completed}/{total} files")
        
        if completed < total:
            failed = [
                k for k, v in self.progress[progress_key].items() 
                if not v.get("completed", False)
            ]
            logger.warning(f"Failed files: {', '.join(failed)}")
        else:
            logger.success("All files processed successfully!")
        
        # Log safety block statistics
        safety_stats = self.llm_client.get_safety_stats()
        if safety_stats:
            logger.info("\n=== Safety Block Statistics ===")
            for provider, blocked_count in safety_stats.items():
                logger.info(f"{provider}: {blocked_count} operations blocked for safety")
        
        return summary
    
    def clean_markdown_response(self, content: str) -> str:
        """
        Clean up markdown response from LLM.
        
        Args:
            content: Raw response from LLM
        
        Returns:
            Cleaned markdown content
        """
        lines = content.strip().split('\n')
        
        # Look for code block markers in first 3 non-empty lines
        non_empty_count = 0
        code_block_start = -1
        
        for i, line in enumerate(lines):
            if line.strip():  # Non-empty line
                non_empty_count += 1
                # Check if this line is a code block marker
                if line.strip() in ['```markdown', '```'] or line.strip().startswith('```'):
                    code_block_start = i + 1  # Start from the line after the marker
                    break
                if non_empty_count >= 3:
                    break
        
        # If we found a code block marker, remove everything before and including it
        if code_block_start > 0:
            lines = lines[code_block_start:]
        
        # Rejoin the content
        content = '\n'.join(lines)
        
        # Also handle case where ``` appears at the end
        if content.strip().endswith('```'):
            lines = content.strip().split('\n')
            if lines[-1].strip() == '```':
                lines = lines[:-1]
                content = '\n'.join(lines)
        
        return content.strip()
