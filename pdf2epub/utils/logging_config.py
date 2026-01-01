import os
import sys
from pathlib import Path
from loguru import logger


def configure_logging(title=None, command=None, verbose=True):
    """
    Configure loguru logger with separate formats for file and stderr.

    - File: Detailed format with timestamp, function name, line number (DEBUG level)
    - Stderr: Concise format with just level and message (INFO level)

    Args:
        title (str, optional): The book title to use for the log folder.
        command (str, optional): Command name for separate log file (e.g., 'refine', 'polish').
        verbose (bool, optional): Whether to output logs to stderr. Defaults to True.

    Returns:
        The configured logger instance
    """
    # Remove all existing handlers
    logger.remove()

    # Add stderr handler with concise format (level from env or INFO)
    if verbose:
        stderr_level = os.environ.get("LOGURU_LEVEL", "INFO")
        logger.add(
            sink=sys.stderr,
            format="<level>{level: <8}</level> | {message}",
            level=stderr_level,
            colorize=True,
        )

    # Add file handler with detailed format (DEBUG level)
    if title:
        # Create logs directory
        log_dir = Path("output") / title / "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Determine log file name
        if command:
            log_file = log_dir / f"{command}.log"
        else:
            log_file = log_dir / "process.log"

        logger.add(
            sink=str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week",
        )

    return logger
