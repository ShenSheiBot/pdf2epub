"""
Base processor class for markdown transformations.

This module provides the abstract base class for all markdown processors,
handling common functionality like progress tracking, retry logic, and
concurrent processing.
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from pdf2epub.utils.llm_client import LLMClient


class BaseMarkdownProcessor(ABC):
    """Abstract base class for markdown processors."""
    
    def __init__(
        self,
        config: Dict,
        book_title: str,
        input_dir: str,
        output_dir: str,
        max_workers: int = 4,
        resume: bool = False
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
        """
        self.config = config
        self.book_title = book_title
        self.max_workers = max_workers
        self.resume = resume
        
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
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                # Process the content
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
                    logger.warning(f"Validation failed for {file_name}: {reason}")
                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying {file_name} (attempt {attempt + 2}/{max_attempts})...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # Use the output anyway on final attempt
                        logger.warning(f"Using output despite validation failure for {file_name}")
                
                # Save the processed file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                
                logger.success(f"Successfully processed {file_name}")
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed for {file_name}: {e}")
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        logger.error(f"Failed to process {file_name} after {max_attempts} attempts. Last error: {last_error}")
        return False
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all markdown files in the input directory.
        
        Returns:
            Summary statistics
        """
        # Find all markdown files
        markdown_files = sorted(self.input_dir.glob("*.md"))
        
        if not markdown_files:
            logger.error(f"No markdown files found in {self.input_dir}")
            return {"error": "No files found"}
        
        logger.info(f"Found {len(markdown_files)} markdown files to process")
        
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
            
            # Process completed tasks
            for future, file_key in futures:
                try:
                    success = future.result()
                    progress_key = self.get_progress_key()
                    
                    if success:
                        self.progress[progress_key][file_key] = {
                            "completed": True,
                            "timestamp": time.time()
                        }
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
        # Remove code block wrappers if present
        if content.startswith('```'):
            lines = content.split('\n')
            start_idx = 1 if lines[0].startswith('```') else 0
            end_idx = len(lines) - 1 if lines[-1] == '```' else len(lines)
            content = '\n'.join(lines[start_idx:end_idx])
        
        return content.strip()