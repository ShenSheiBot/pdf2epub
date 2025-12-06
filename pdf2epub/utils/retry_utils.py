"""
Retry utilities using tenacity for smarter retry logic.

IMPORTANT: All retry logic in this project should use `retry_with_logging()`
instead of raw `tenacity.retry`. This ensures consistent exception logging.
"""

from typing import Optional, Any, Callable
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
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


def retry_with_logging(
    operation_name: str,
    retry_condition: Callable[[Exception], bool],
    wait_strategy: Any = None,
    stop_strategy: Any = None,
    max_exception_length: int = 500,
    **kwargs
) -> Callable:
    """
    Create a retry decorator that ALWAYS logs exception details on each retry.

    This is the STANDARD way to add retry logic in this project.
    Do NOT use raw `tenacity.retry` - use this function instead.

    Args:
        operation_name: Name of the operation for logging
        retry_condition: Function that takes an exception and returns True if should retry
        wait_strategy: tenacity wait strategy (default: wait_random_exponential)
        stop_strategy: tenacity stop strategy (default: stop_after_delay(300))
        max_exception_length: Max chars for exception message in log (default: 500)
        **kwargs: Additional arguments passed to tenacity.retry

    Returns:
        A decorator that wraps functions with retry logic

    Example:
        @retry_with_logging(
            operation_name="Translate chapter_1.md",
            retry_condition=is_transient_error,
        )
        def translate_file():
            ...
    """
    if wait_strategy is None:
        wait_strategy = wait_random_exponential(multiplier=2, max=120)

    if stop_strategy is None:
        stop_strategy = stop_after_delay(300)  # 5 minutes default

    def before_sleep_with_exception(retry_state: RetryCallState) -> None:
        """Log retry with exception details - this is ALWAYS included."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None

        if exc:
            exc_msg = str(exc).replace('\n', ' ').replace('\r', '')
            if len(exc_msg) > max_exception_length:
                exc_msg = exc_msg[:max_exception_length] + "..."
        else:
            exc_msg = "unknown error"

        logger.warning(
            f"Retry {retry_state.attempt_number} for {operation_name}: "
            f"waiting {retry_state.next_action.sleep:.1f}s "
            f"(elapsed {retry_state.seconds_since_start:.0f}s) | {exc_msg}"
        )

    def decorator(func: Callable) -> Callable:
        @retry(
            retry=retry_if_exception(retry_condition),
            wait=wait_strategy,
            stop=stop_strategy,
            reraise=True,
            before_sleep=before_sleep_with_exception,
            **kwargs
        )
        @wraps(func)
        def wrapper(*args, **kw):
            return func(*args, **kw)
        return wrapper

    return decorator