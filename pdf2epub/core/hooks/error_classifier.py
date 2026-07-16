"""
Error classifier - categorizes errors and determines their effects on unit state.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Protocol, runtime_checkable

from ._protocol import ErrorType, ErrorEffect


# ============================================================
# Batch Failure Actions
# ============================================================

class BatchFailureAction(Enum):
    """
    Action to take when a batch unit fails.

    This determines how the model chain should be modified for the unit.
    """
    REMOVE_PROVIDER = "remove_provider"      # Remove entire provider (all models)
    REMOVE_MODEL = "remove_model"            # Remove current model only
    RETRY_ONLINE_SAME_MODEL = "retry_online" # Try same model via online path once


def get_batch_failure_action(error_type: ErrorType) -> BatchFailureAction:
    """
    Determine the action for a batch unit failure.

    Policy:
    - SAFETY/CONTENT_FILTER: Remove provider (content issue, provider-level)
    - TRUNCATION: Remove model (model capability issue)
    - VALIDATION: Retry online same model once (might be batch-specific issue)
    - NETWORK/TIMEOUT/RATE_LIMIT: Remove model (fail fast)
    - Others: Remove model (default)

    Args:
        error_type: The classified error type

    Returns:
        BatchFailureAction indicating what to do with the model chain
    """
    if error_type in (ErrorType.SAFETY, ErrorType.CONTENT_FILTER):
        return BatchFailureAction.REMOVE_PROVIDER
    elif error_type == ErrorType.VALIDATION:
        # VALIDATION special handling: give online path one chance
        # Batch API may have subtle differences from online API
        return BatchFailureAction.RETRY_ONLINE_SAME_MODEL
    elif error_type == ErrorType.TRUNCATION:
        return BatchFailureAction.REMOVE_MODEL
    else:
        # NETWORK, TIMEOUT, RATE_LIMIT, PARSE_ERROR, UNKNOWN
        return BatchFailureAction.REMOVE_MODEL


# ============================================================
# Job Failure Attribution (for batch errors)
# ============================================================

@dataclass
class Attribution:
    """
    Attribution of a job failure to unit-level or job-level quota.

    Attributes:
        type: "unit" if attributable to a specific unit, "job" if systemic
        unit_id: The unit ID if type == "unit"
        error_type: The classified error type
    """
    type: Literal["unit", "job"]
    unit_id: Optional[str] = None
    error_type: ErrorType = ErrorType.UNKNOWN


@runtime_checkable
class HasContent(Protocol):
    """Protocol for objects with id and content attributes."""
    id: str
    content: str


def extract_unit_key_from_error(error_message: str, unit_ids: List[str]) -> Optional[str]:
    """
    Extract unit key from error message if present.

    Args:
        error_message: The error message string
        unit_ids: List of known unit IDs to search for

    Returns:
        The unit ID if found in the error message, None otherwise
    """
    for unit_id in unit_ids:
        if unit_id in error_message:
            return unit_id
    return None


def attribute_job_failure(
    job_error: str,
    units: List[HasContent],
    classifier: Optional["DefaultErrorClassifier"] = None
) -> Attribution:
    """
    Attribute a batch job failure to a specific unit or the job itself.

    Attribution strategy (by priority):
    1. Error explicitly mentions a unit key → attribute to that unit
    2. Error indicates size/shape issue → attribute to largest unit (likely poison)
    3. Default: systemic problem → attribute to job quota

    Args:
        job_error: The error message from the batch job
        units: List of work units in the batch
        classifier: Error classifier to use (defaults to DefaultErrorClassifier)

    Returns:
        Attribution indicating whether to charge unit quota or job quota
    """
    if classifier is None:
        classifier = DefaultErrorClassifier()

    unit_ids = [u.id for u in units]
    error_lower = job_error.lower()

    # 1. Check if error explicitly mentions a unit key
    if unit_key := extract_unit_key_from_error(job_error, unit_ids):
        return Attribution(
            type="unit",
            unit_id=unit_key,
            error_type=classifier.classify_from_string(job_error)
        )

    # 2. Check for size/shape issues → blame largest unit
    size_keywords = ["too large", "exceeded", "token limit", "context length", "max_tokens"]
    if any(kw in error_lower for kw in size_keywords):
        if units:
            largest = max(units, key=lambda u: len(u.content))
            return Attribution(
                type="unit",
                unit_id=largest.id,
                error_type=ErrorType.VALIDATION
            )

    # 3. Default: systemic problem
    error_type = classifier.classify_from_string(job_error)
    # For systemic errors, default to NETWORK if unknown
    if error_type == ErrorType.UNKNOWN:
        error_type = ErrorType.NETWORK

    return Attribution(type="job", error_type=error_type)


# ============================================================
# Batch Circuit Breaker (three-state model)
# ============================================================

@dataclass
class BatchCircuitBreaker:
    """
    Three-state circuit breaker for batch processing.

    States:
    - closed: Normal operation, batch allowed
    - open: Batch disabled due to failures, waiting for cooldown
    - half_open: Probing with a single batch request

    Transitions:
    - closed -> open: When failure_count >= threshold
    - open -> half_open: When cooldown expires
    - half_open -> closed: On probe success
    - half_open -> open: On probe failure
    """
    state: Literal["closed", "open", "half_open"] = "closed"
    failure_count: int = 0
    last_failure_time: float = 0.0

    # Configuration
    threshold: int = 3           # Number of failures to trigger open state
    cooldown_seconds: int = 300  # Time before probing (5 minutes default)

    def record_failure(self) -> None:
        """
        Record a batch job failure.

        Updates failure count and transitions to open state if threshold exceeded.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half_open":
            # Probe failed, go back to open
            self.state = "open"
            logger.warning("Batch circuit breaker: probe FAILED, returning to OPEN state")
        elif self.failure_count >= self.threshold:
            self.state = "open"
            logger.warning(
                f"Batch circuit breaker OPEN after {self.failure_count} failures, "
                f"cooldown: {self.cooldown_seconds}s"
            )

    def should_try_batch(self) -> bool:
        """
        Check if batch processing should be attempted.

        Returns:
            True if batch should be tried, False otherwise
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if cooldown has expired
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.state = "half_open"
                logger.info("Batch circuit breaker HALF_OPEN, probing...")
                return True  # Allow one probe attempt
            return False

        if self.state == "half_open":
            # Already probing, don't start new batches
            return True

        return False

    def record_success(self) -> None:
        """
        Record a successful batch job.

        Transitions from half_open to closed, resets failure count.
        """
        import logging
        logger = logging.getLogger(__name__)

        if self.state == "half_open":
            logger.info("Batch circuit breaker CLOSED, batch recovered")

        self.state = "closed"
        self.failure_count = 0

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0.0


class DefaultErrorClassifier:
    """
    Default error classifier with standard effects.

    Effects by error type:
    - SAFETY: Remove current model AND entire provider (batch + online)
    - NETWORK: Remove current model (fail fast, try next model), never split
    - VALIDATION: Keep current model, decrement validation quota
    - TRUNCATION: Remove current model (might produce bad results), prefer split
    - RATE_LIMIT: Remove current model (fail fast, try next model), never split
    - TIMEOUT: Remove current model (fail fast, try next model), never split
    - CONTENT_FILTER: Remove current model, remove provider (like safety)
    - PARSE_ERROR: Keep current model, decrement network quota (retry)
    - UNKNOWN: Keep current model, decrement network quota

    Network/Timeout/RateLimit policy:
    - Bottom layer (LLMClient) handles short transient retries (30s)
    - Upper layer (Executor) switches model on failure (fail fast)
    - These errors NEVER trigger split (handled in executor._handle_failure)
    """

    # Error effect definitions
    EFFECTS: Dict[ErrorType, ErrorEffect] = {
        ErrorType.SAFETY: ErrorEffect(
            remove_current_model=True,
            remove_provider=True,  # Remove all entries from same provider
            remove_all_batch=False,
            quota_type=ErrorType.SAFETY
        ),
        ErrorType.NETWORK: ErrorEffect(
            remove_current_model=True,  # Fail fast: switch to next model
            remove_provider=False,      # Provider might recover
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK
        ),
        ErrorType.VALIDATION: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.VALIDATION
        ),
        ErrorType.TRUNCATION: ErrorEffect(
            remove_current_model=True,  # Model might consistently truncate
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.TRUNCATION  # Use its own quota (2 retries by default)
        ),
        ErrorType.RATE_LIMIT: ErrorEffect(
            remove_current_model=True,  # Fail fast: switch to next model
            remove_provider=False,      # Rate limit is per-model, try another
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK  # Counts against network quota
        ),
        ErrorType.TIMEOUT: ErrorEffect(
            remove_current_model=True,  # Fail fast: switch to next model
            remove_provider=False,      # Timeout might be transient
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK
        ),
        ErrorType.CONTENT_FILTER: ErrorEffect(
            remove_current_model=True,
            remove_provider=True,  # Provider might block this content
            remove_all_batch=False,
            quota_type=ErrorType.SAFETY
        ),
        ErrorType.PARSE_ERROR: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK  # Transient, retry
        ),
        ErrorType.UNKNOWN: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK
        ),
    }

    # Keywords for error classification
    SAFETY_KEYWORDS = [
        "safety", "harmful", "content filter",
        "inappropriate", "policy violation", "cannot generate",
        "i cannot", "i can't", "against my guidelines",
        "i refuse", "refuse to generate", "refused to",
        "content blocked", "safety blocked",
        # Note: removed bare "blocked" as it matches "request blocked: rate limit" (rate limit error)
        # Note: removed bare "refused" as it matches "connection refused" (network error)
    ]

    CONTENT_FILTER_KEYWORDS = [
        "content_filter", "blocked_reason", "finish_reason.*safety",
        "recitation", "copyright", "prohibited",
    ]

    RATE_LIMIT_KEYWORDS = [
        "rate limit", "quota", "429", "too many requests",
        "resource exhausted", "capacity", "overloaded",
    ]

    NETWORK_KEYWORDS = [
        "connection error", "network error", "connection refused",
        "503", "502", "504", "404", "not found", "unavailable", "unreachable",
        "ssl", "handshake", "reset by peer", "connection reset",
        "dns", "socket error",
    ]

    TIMEOUT_KEYWORDS = [
        "timeout", "timed out", "deadline exceeded", "request timeout",
        "operation timed out", "read timed out",
    ]

    VALIDATION_KEYWORDS = [
        "validation failed", "too short", "too long",
        "missing", "empty response", "invalid format",
        "too large", "exceeded", "limit",  # Batch-specific: request shape issues
        "line count", "mismatch",  # LineCountValidator
    ]

    TRUNCATION_KEYWORDS = [
        "truncat", "incomplete", "cut off", "content lost",
        "unique content lost", "not complete",
    ]

    PARSE_KEYWORDS = [
        "parse error", "json", "decode", "malformed",
        "invalid response", "unexpected format", "serialization",
    ]

    def classify(self, error: Exception) -> ErrorType:
        """
        Classify error based on message keywords.

        Args:
            error: The exception to classify

        Returns:
            ErrorType enum value
        """
        explicit_type = getattr(error, "error_type", None)
        if isinstance(explicit_type, ErrorType):
            return explicit_type
        return self.classify_from_string(str(error))

    def classify_from_string(self, error_message: str) -> ErrorType:
        """
        Classify error from string message (for batch errors that don't have Exception objects).

        Args:
            error_message: The error message string

        Returns:
            ErrorType enum value

        Note:
            String-based classification has lower precision than exception classification,
            but is sufficient for quota decisions. Priority order is important.
        """
        msg = error_message.lower()

        # Check in priority order (most specific first)
        if any(kw in msg for kw in self.SAFETY_KEYWORDS):
            return ErrorType.SAFETY

        if any(kw in msg for kw in self.CONTENT_FILTER_KEYWORDS):
            return ErrorType.CONTENT_FILTER

        if any(kw in msg for kw in self.TRUNCATION_KEYWORDS):
            return ErrorType.TRUNCATION

        if any(kw in msg for kw in self.RATE_LIMIT_KEYWORDS):
            return ErrorType.RATE_LIMIT

        if any(kw in msg for kw in self.TIMEOUT_KEYWORDS):
            return ErrorType.TIMEOUT

        if any(kw in msg for kw in self.NETWORK_KEYWORDS):
            return ErrorType.NETWORK

        if any(kw in msg for kw in self.VALIDATION_KEYWORDS):
            return ErrorType.VALIDATION

        if any(kw in msg for kw in self.PARSE_KEYWORDS):
            return ErrorType.PARSE_ERROR

        return ErrorType.UNKNOWN

    def get_effect(self, error_type: ErrorType) -> ErrorEffect:
        """
        Get the effect for an error type.

        Args:
            error_type: The error type

        Returns:
            ErrorEffect describing how to modify unit state
        """
        return self.EFFECTS.get(error_type, self.EFFECTS[ErrorType.UNKNOWN])


class StrictErrorClassifier(DefaultErrorClassifier):
    """
    Stricter error classifier that removes model on validation failures too.
    """

    EFFECTS: Dict[ErrorType, ErrorEffect] = {
        **DefaultErrorClassifier.EFFECTS,
        ErrorType.VALIDATION: ErrorEffect(
            remove_current_model=True,  # Remove model on validation failure
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.VALIDATION
        ),
    }
