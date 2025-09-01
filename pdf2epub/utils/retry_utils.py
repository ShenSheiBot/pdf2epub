"""
Retry utilities using tenacity for smarter retry logic.
"""

from typing import Optional, Any, Callable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
    RetryCallState
)
from loguru import logger
import httpx


# Define transient error codes that should trigger retries
TRANSIENT_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}


def is_transient_error(exception: Exception) -> bool:
    """
    Determine if an exception is transient and should be retried.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is transient and should be retried
    """
    # Check for timeout or connection errors
    if isinstance(exception, (TimeoutError, ConnectionError, httpx.TimeoutException)):
        return True
    
    # Check for HTTP status codes
    status = getattr(exception, "status_code", None)
    if status and isinstance(status, int):
        if status in TRANSIENT_HTTP_CODES or status >= 500:
            return True
    
    # Check for specific error messages
    error_str = str(exception).lower()
    transient_keywords = [
        'rate_limit', 'rate limit', 'quota',
        'timeout', 'timed out',
        'connection', 'network',
        'unavailable', 'internal',
        'resource_exhausted', 'overloaded',
        '429', '500', '502', '503', '504'
    ]
    
    if any(keyword in error_str for keyword in transient_keywords):
        return True
    
    # Check for content safety blocks - these should NOT be retried
    safety_keywords = [
        'prohibited', 'safety', 'blocked',
        'harmful', 'inappropriate', 'violation'
    ]
    
    if any(keyword in error_str for keyword in safety_keywords):
        return False
    
    return False


def before_retry_log(retry_state: RetryCallState) -> None:
    """Log before retry attempts."""
    if retry_state.attempt_number > 1:
        operation_name = retry_state.kwargs.get('operation_name', 'API call')
        logger.warning(
            f"Retry attempt {retry_state.attempt_number} for {operation_name} "
            f"after {retry_state.outcome.exception()}"
        )


def create_retry_decorator(
    max_attempts: int = 5,
    max_wait: int = 60,
    multiplier: float = 1.0,
    operation_name: Optional[str] = None
) -> Callable:
    """
    Create a retry decorator with custom parameters.
    
    Args:
        max_attempts: Maximum number of attempts (including the first)
        max_wait: Maximum wait time between retries in seconds
        multiplier: Multiplier for exponential backoff
        operation_name: Name of the operation for logging
        
    Returns:
        A configured retry decorator
    """
    return retry(
        retry=retry_if_exception(is_transient_error),
        wait=wait_random_exponential(multiplier=multiplier, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_retry_log,
        reraise=True
    )


# Default retry decorator for most API calls
default_retry = create_retry_decorator(
    max_attempts=5,
    max_wait=30,
    multiplier=1.0
)


# Aggressive retry for critical operations
aggressive_retry = create_retry_decorator(
    max_attempts=10,
    max_wait=60,
    multiplier=1.5
)


# Quick retry for fast operations
quick_retry = create_retry_decorator(
    max_attempts=3,
    max_wait=10,
    multiplier=0.5
)