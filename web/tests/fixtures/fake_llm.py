"""
Fake LLM Client for behavioral testing.

Design: Inherit from real LLMClient and override generate() to produce
controllable, reproducible behavior using seeded randomness.

Key principles:
1. Same interface as real LLMClient - drop-in replacement
2. Produces real exception types that error_classifier can parse
3. Seeded random for reproducible tests
4. Records all calls for assertions
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum

from pdf2epub.utils.llm_client import LLMClient, SafetyBlockError


# Error messages that match what error_classifier expects
# These are based on real error patterns from each error type
ERROR_MESSAGES = {
    "network": "network error: connection refused",
    "timeout": "timeout: request timed out after 30s",
    "rate_limit": "rate limit: 429 too many requests",
    "safety": "safety: content blocked due to policy violation",
    "content_filter": "content_filter: recitation detected, blocked by safety filter",
    "parse_error": "parse error: failed to parse response as JSON",
    "truncation": "truncation: response was cut off",
    "validation": "validation: output does not match expected format",
    "unknown": "unknown error occurred",
}


class FakeErrorType(Enum):
    """Error types that can be injected. Maps to pdf2epub.core.types.ErrorType."""
    NONE = "none"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SAFETY = "safety"
    CONTENT_FILTER = "content_filter"
    PARSE_ERROR = "parse_error"
    TRUNCATION = "truncation"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class FakeResponse:
    """
    Configuration for a fake response.

    Attributes:
        content: Response content (string or callable that takes prompt)
        error: Error type to raise (NONE for success)
        succeed_after_n_calls: Fail first N calls, then succeed (0 = always fail/succeed)
        truncate_ratio: Truncate response to this ratio (1.0 = full, 0.5 = half)
    """
    content: Union[str, Callable[[str], str]] = "fake llm response"
    error: FakeErrorType = FakeErrorType.NONE
    succeed_after_n_calls: int = 0
    truncate_ratio: float = 1.0


@dataclass
class CallRecord:
    """Record of a single LLM call."""
    prompt: Union[str, List[Dict]]
    model_configs: Optional[List[Dict]]
    operation_name: str
    unit_id: str
    response: Optional[str] = None
    error: Optional[Exception] = None


class FakeLLMClient(LLMClient):
    """
    Fake LLM Client that inherits from real LLMClient.

    Only overrides generate() - everything else uses real implementation.
    This ensures interface compatibility and realistic behavior.

    Usage:
        fake = FakeLLMClient(seed=42)

        # Default: all calls succeed with default response
        fake.set_default_response("translated content")

        # Specific unit behavior
        fake.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.RATE_LIMIT,
            succeed_after_n_calls=2  # Fail twice, then succeed
        ))

        # Random errors (seeded for reproducibility)
        fake.set_random_error_rate(0.3, FakeErrorType.NETWORK)

        # Use in executor
        executor = Executor(llm_client=fake, ...)
        result = executor.execute([unit])

        # Assert on call patterns
        assert fake.call_count_for("chapter_1") == 3
    """

    def __init__(self, seed: Optional[int] = None, default_response: str = "fake llm response"):
        """
        Initialize fake client.

        Args:
            seed: Random seed for reproducible behavior (None = random)
            default_response: Default response content for successful calls
        """
        # Initialize parent with minimal config (we override generate anyway)
        super().__init__(config={})

        self._seed = seed
        self._rng = random.Random(seed)
        self._default_response = default_response

        # Per-unit response configuration
        self._responses: Dict[str, FakeResponse] = {}

        # Call tracking
        self._call_counts: Dict[str, int] = {}
        self.call_history: List[CallRecord] = []

        # Random error injection
        self._random_error_rate: float = 0.0
        self._random_error_type: FakeErrorType = FakeErrorType.NETWORK

    def set_default_response(self, content: str) -> None:
        """Set default response for units without specific configuration."""
        self._default_response = content

    def set_response(self, unit_id: str, response: FakeResponse) -> None:
        """Set response configuration for a specific unit."""
        self._responses[unit_id] = response

    def set_responses(self, responses: Dict[str, FakeResponse]) -> None:
        """Set multiple response configurations."""
        self._responses.update(responses)

    def set_random_error_rate(self, rate: float, error_type: FakeErrorType = FakeErrorType.NETWORK) -> None:
        """
        Set random error injection rate.

        Args:
            rate: Probability of random error (0.0 - 1.0)
            error_type: Type of error to inject randomly
        """
        self._random_error_rate = rate
        self._random_error_type = error_type

    def clear(self) -> None:
        """Clear all configurations and history."""
        self._responses.clear()
        self._call_counts.clear()
        self.call_history.clear()
        self._rng = random.Random(self._seed)

    def _extract_unit_id(self, operation_name: str) -> str:
        """Extract unit_id from operation_name (format: 'processor:unit_id')."""
        if ":" in operation_name:
            return operation_name.split(":", 1)[1]
        return operation_name

    def _make_error(self, error_type: FakeErrorType, provider: str = "fake") -> Exception:
        """Create an exception for the given error type."""
        message = ERROR_MESSAGES.get(error_type.value, f"error: {error_type.value}")

        # Safety errors use SafetyBlockError (real exception from LLMClient)
        if error_type == FakeErrorType.SAFETY:
            return SafetyBlockError(message, provider)

        # All other errors are generic Exceptions with recognizable messages
        return Exception(message)

    def generate(
        self,
        prompt: Union[str, List[Dict]],
        model_configs: Optional[List[Dict]] = None,
        operation_name: str = "LLM generation"
    ) -> str:
        """
        Generate content - overrides real LLMClient.generate().

        This is the only method that differs from real LLMClient.
        All behavior is controlled by set_response() and set_random_error_rate().
        """
        unit_id = self._extract_unit_id(operation_name)

        # Track call count
        self._call_counts[unit_id] = self._call_counts.get(unit_id, 0) + 1
        call_count = self._call_counts[unit_id]

        # Get response configuration
        response_config = self._responses.get(unit_id, FakeResponse(content=self._default_response))

        # Create call record
        record = CallRecord(
            prompt=prompt,
            model_configs=model_configs,
            operation_name=operation_name,
            unit_id=unit_id,
        )

        try:
            # Check for configured error
            if response_config.error != FakeErrorType.NONE:
                # succeed_after_n_calls: 0 = always error, N = error first N times
                should_error = (
                    response_config.succeed_after_n_calls == 0 or
                    call_count <= response_config.succeed_after_n_calls
                )
                if should_error:
                    error = self._make_error(response_config.error)
                    record.error = error
                    raise error

            # Check for random error
            if self._random_error_rate > 0 and self._rng.random() < self._random_error_rate:
                error = self._make_error(self._random_error_type)
                record.error = error
                raise error

            # Generate successful response
            if callable(response_config.content):
                content = response_config.content(prompt if isinstance(prompt, str) else str(prompt))
            else:
                content = response_config.content

            # Apply truncation
            if response_config.truncate_ratio < 1.0:
                truncate_point = int(len(content) * response_config.truncate_ratio)
                content = content[:truncate_point]

            record.response = content
            return content

        finally:
            self.call_history.append(record)

    # ========================================
    # Assertion helpers
    # ========================================

    def get_calls_for(self, unit_id: str) -> List[CallRecord]:
        """Get all call records for a specific unit."""
        return [c for c in self.call_history if c.unit_id == unit_id]

    def call_count_for(self, unit_id: str) -> int:
        """Get number of calls for a specific unit."""
        return len(self.get_calls_for(unit_id))

    def was_called(self, unit_id: str) -> bool:
        """Check if unit was called at least once."""
        return self.call_count_for(unit_id) > 0

    def assert_called(self, unit_id: str, times: Optional[int] = None) -> None:
        """Assert unit was called (optionally exactly N times)."""
        actual = self.call_count_for(unit_id)
        if times is not None:
            assert actual == times, f"Expected {unit_id} called {times} times, got {actual}"
        else:
            assert actual > 0, f"Expected {unit_id} to be called, but it wasn't"

    def assert_not_called(self, unit_id: str) -> None:
        """Assert unit was never called."""
        actual = self.call_count_for(unit_id)
        assert actual == 0, f"Expected {unit_id} not called, got {actual} calls"

    def get_errors_for(self, unit_id: str) -> List[Exception]:
        """Get all errors that occurred for a unit."""
        return [c.error for c in self.get_calls_for(unit_id) if c.error is not None]

    def get_successful_responses_for(self, unit_id: str) -> List[str]:
        """Get all successful responses for a unit."""
        return [c.response for c in self.get_calls_for(unit_id) if c.response is not None]


# ========================================
# Response generators for common patterns
# ========================================

def make_translation_response(target_lang: str = "Chinese") -> Callable[[str], str]:
    """Create a translation response generator."""
    def generate(prompt: str) -> str:
        # Simple mock: prefix with [翻译]
        return f"[{target_lang}翻译] " + prompt[:200]
    return generate


def make_flaky_response(
    success_content: str,
    fail_count: int,
    error_type: FakeErrorType = FakeErrorType.NETWORK
) -> FakeResponse:
    """Create a response that fails N times then succeeds."""
    return FakeResponse(
        content=success_content,
        error=error_type,
        succeed_after_n_calls=fail_count,
    )
