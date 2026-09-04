"""
Safety utilities for preventing accidental overwrites and data loss.
"""

import fcntl
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from loguru import logger


class ProcessLockError(RuntimeError):
    """Raised when another process owns an exclusive stage lock."""


@contextmanager
def exclusive_process_lock(path: Path, operation: str) -> Iterator[None]:
    """Hold a non-blocking, process-wide lock for one output stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProcessLockError(
                f"Another {operation} process holds {path}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def extract_pdf_metadata(pdf_path: Path) -> dict:
    """
    Extract metadata from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        dict: Contains title, author, pages
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata or {}
        page_count = len(doc)
        doc.close()

        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'pages': page_count
        }
    except Exception as e:
        logger.debug(f"Could not extract PDF metadata: {e}")
        return {'title': '', 'author': '', 'pages': 0}


def check_output_directory_conflict(output_dir: Path, input_pdf: Path) -> Path:
    """
    Check if output directory exists and contains a different book.
    Returns the actual output directory to use (may be renamed).

    This prevents accidental overwrites when processing different books with
    the same title in config.yaml.

    Args:
        output_dir: Intended output directory
        input_pdf: Path to the new input PDF

    Returns:
        Path: The output directory to use (original or renamed)
    """
    if not output_dir.exists():
        return output_dir  # No conflict

    # Check if there's an existing PDF
    existing_pdf = output_dir / "input_original.pdf"
    if not existing_pdf.exists():
        logger.debug(f"Output directory exists but no input_original.pdf found, safe to proceed")
        return output_dir

    # Extract metadata from both PDFs
    new_info = extract_pdf_metadata(input_pdf)
    existing_info = extract_pdf_metadata(existing_pdf)

    # Check if they're likely the same book
    same_pages = abs(new_info['pages'] - existing_info['pages']) <= 5
    same_author = new_info['author'] and existing_info['author'] and \
                  new_info['author'].lower() == existing_info['author'].lower()

    if same_pages and (same_author or not new_info['author']):
        logger.info(f"Output directory exists with similar book ({existing_info['pages']} pages), continuing...")
        return output_dir

    # Different book detected!
    logger.warning(f"⚠️  Output directory '{output_dir.name}' already exists with different content!")
    logger.warning(f"")
    logger.warning(f"   📚 Existing: {existing_info['author'] or 'Unknown author'} ({existing_info['pages']} pages)")
    logger.warning(f"   📚 New:      {new_info['author'] or 'Unknown author'} ({new_info['pages']} pages)")
    logger.warning(f"")

    # Ask user for action
    print("   Choose action:")
    print("   1) Overwrite existing directory (will delete all existing data)")
    print("   2) Create new directory with suffix (e.g., 'Book Title (1)')")
    print("   3) Abort")

    while True:
        choice = input("   Your choice [1/2/3]: ").strip()

        if choice == '1':
            logger.warning(f"⚠️  Overwriting {output_dir}")
            return output_dir
        elif choice == '2':
            # Find next available suffix
            counter = 1
            while True:
                new_dir = output_dir.parent / f"{output_dir.name} ({counter})"
                if not new_dir.exists():
                    logger.info(f"✅ Creating new directory: {new_dir.name}")
                    return new_dir
                counter += 1
        elif choice == '3':
            logger.info("Operation aborted by user")
            sys.exit(0)
        else:
            print("   Invalid choice, please enter 1, 2, or 3")
