"""
Fake Batch Client for testing batch path behavior.

Design: Inherit from GeminiBatchClient and override methods to produce
controllable, reproducible behavior for testing batch vs online path unification.

Key principles:
1. Same interface as real GeminiBatchClient - drop-in replacement
2. Produces configurable job states (SUCCEEDED, FAILED, etc.)
3. Produces configurable per-unit responses (success, errors)
4. Records all calls for assertions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from pdf2epub.utils.batch_utils import (
    GeminiBatchClient,
    BatchRequest,
    BatchResponse,
    BatchJobInfo,
    BatchJobState,
)


class FakeBatchErrorType(Enum):
    """Error types that can be injected into batch responses."""
    NONE = "none"
    RATE_LIMIT = "rate_limit"
    SAFETY = "safety"
    TIMEOUT = "timeout"
    NETWORK = "network"
    TRUNCATION = "truncation"
    VALIDATION = "validation"
    CONTENT_FILTER = "content_filter"


# Error messages that match what error_classifier expects (per batch response)
BATCH_ERROR_MESSAGES = {
    "rate_limit": "429 rate limit: quota exceeded for this request",
    "safety": "safety: content blocked due to policy violation",
    "timeout": "timeout: request processing timed out",
    "network": "network error: service temporarily unavailable",
    "truncation": "truncation: response was incomplete, content cut off",
    "validation": "validation: response does not match expected format",
    "content_filter": "content_filter: recitation detected, blocked",
}


@dataclass
class FakeBatchUnitConfig:
    """
    Configuration for a single unit in a batch response.

    Attributes:
        text: Response text (None if error)
        error: Error type to inject
    """
    text: Optional[str] = "fake batch response"
    error: FakeBatchErrorType = FakeBatchErrorType.NONE

    def get_error_message(self) -> Optional[str]:
        """Get error message for this config."""
        if self.error == FakeBatchErrorType.NONE:
            return None
        return BATCH_ERROR_MESSAGES.get(self.error.value, f"error: {self.error.value}")


@dataclass
class FakeBatchJobConfig:
    """
    Configuration for a batch job.

    Attributes:
        state: Final job state (SUCCEEDED, FAILED, etc.)
        error: Job-level error message (for FAILED state)
        poll_count: Number of polls before completion (0 = immediate)
        unit_configs: Per-unit response configurations
    """
    state: BatchJobState = BatchJobState.SUCCEEDED
    error: Optional[str] = None
    poll_count: int = 0
    unit_configs: Dict[str, FakeBatchUnitConfig] = field(default_factory=dict)


@dataclass
class BatchCallRecord:
    """Record of a batch API call."""
    method: str  # "submit", "get_status", "get_results"
    job_name: Optional[str] = None
    requests: Optional[List[BatchRequest]] = None
    display_name: Optional[str] = None


class FakeBatchClient(GeminiBatchClient):
    """
    Fake Batch Client that inherits from GeminiBatchClient.

    Overrides key methods to provide controllable behavior for testing.

    Usage:
        fake = FakeBatchClient()

        # Configure job to succeed with specific unit responses
        fake.configure_job(FakeBatchJobConfig(
            state=BatchJobState.SUCCEEDED,
            unit_configs={
                "unit1": FakeBatchUnitConfig(text="translated unit 1"),
                "unit2": FakeBatchUnitConfig(error=FakeBatchErrorType.RATE_LIMIT),
            }
        ))

        # Configure job to fail entirely
        fake.configure_job(FakeBatchJobConfig(
            state=BatchJobState.FAILED,
            error="Batch processing failed due to internal error"
        ))

        # Use in executor
        executor = Executor(..., batch_client=fake)
        result = executor.execute(units)

        # Assert on calls
        assert fake.submit_called
        assert fake.submitted_requests == [...]
    """

    def __init__(
        self,
        default_response: str = "fake batch response",
        api_key: str = "fake-api-key",
        model: str = "fake-batch-model",
    ):
        """
        Initialize fake batch client.

        Args:
            default_response: Default response text for units without config
            api_key: Fake API key (not used)
            model: Fake model name
        """
        # Call parent init but never actually use the client
        super().__init__(api_key=api_key, model=model, poll_interval=0)

        self._default_response = default_response
        self._job_config: Optional[FakeBatchJobConfig] = None
        self._current_poll_count = 0

        # Job tracking
        self._job_counter = 0
        self._active_jobs: Dict[str, FakeBatchJobConfig] = {}

        # Call recording
        self.call_history: List[BatchCallRecord] = []
        self._submitted_requests: Dict[str, List[BatchRequest]] = {}

    def configure_job(self, config: FakeBatchJobConfig) -> None:
        """Configure the next job's behavior."""
        self._job_config = config
        self._current_poll_count = 0

    def configure_success(self, unit_responses: Optional[Dict[str, str]] = None) -> None:
        """Shorthand: configure job to succeed with given responses."""
        unit_configs = {}
        if unit_responses:
            for key, text in unit_responses.items():
                unit_configs[key] = FakeBatchUnitConfig(text=text)
        self.configure_job(FakeBatchJobConfig(
            state=BatchJobState.SUCCEEDED,
            unit_configs=unit_configs,
        ))

    def configure_failure(self, error: str = "Batch job failed") -> None:
        """Shorthand: configure job to fail with given error."""
        self.configure_job(FakeBatchJobConfig(
            state=BatchJobState.FAILED,
            error=error,
        ))

    def configure_partial_failure(
        self,
        success_keys: List[str],
        failure_configs: Dict[str, FakeBatchErrorType],
    ) -> None:
        """
        Configure job to succeed but with per-unit errors.

        Args:
            success_keys: Keys that succeed
            failure_configs: Map of key -> error type for failures
        """
        unit_configs = {}
        for key in success_keys:
            unit_configs[key] = FakeBatchUnitConfig(text=f"translated {key}")
        for key, error_type in failure_configs.items():
            unit_configs[key] = FakeBatchUnitConfig(text=None, error=error_type)
        self.configure_job(FakeBatchJobConfig(
            state=BatchJobState.SUCCEEDED,
            unit_configs=unit_configs,
        ))

    def _get_client(self):
        """Override to prevent actual API client initialization."""
        return None

    def submit(
        self,
        requests: List[BatchRequest],
        display_name: Optional[str] = None
    ) -> str:
        """
        Submit a fake batch job.

        Returns a fake job name and records the submitted requests.
        """
        self._job_counter += 1
        job_name = f"fake-job-{self._job_counter}"

        # Record call
        self.call_history.append(BatchCallRecord(
            method="submit",
            job_name=job_name,
            requests=requests,
            display_name=display_name,
        ))

        # Store requests for later retrieval
        self._submitted_requests[job_name] = list(requests)

        # Store job config
        config = self._job_config or FakeBatchJobConfig()
        self._active_jobs[job_name] = config
        self._current_poll_count = 0

        return job_name

    def get_status(self, job_name: str) -> BatchJobInfo:
        """
        Get the status of a fake batch job.

        Returns PENDING until poll_count is reached, then final state.
        """
        self.call_history.append(BatchCallRecord(
            method="get_status",
            job_name=job_name,
        ))

        config = self._active_jobs.get(job_name, FakeBatchJobConfig())

        # Simulate polling
        if self._current_poll_count < config.poll_count:
            self._current_poll_count += 1
            return BatchJobInfo(
                name=job_name,
                state=BatchJobState.RUNNING,
                model=self.model,
            )

        # Return final state
        return BatchJobInfo(
            name=job_name,
            state=config.state,
            model=self.model,
            error=config.error,
        )

    def get_results(self, job_name: str) -> List[BatchResponse]:
        """
        Get results from a fake batch job.

        Returns configured responses based on submitted request keys.
        """
        self.call_history.append(BatchCallRecord(
            method="get_results",
            job_name=job_name,
        ))

        config = self._active_jobs.get(job_name, FakeBatchJobConfig())
        requests = self._submitted_requests.get(job_name, [])

        results = []
        for req in requests:
            unit_config = config.unit_configs.get(
                req.key,
                FakeBatchUnitConfig(text=self._default_response)
            )

            results.append(BatchResponse(
                key=req.key,
                text=unit_config.text,
                error=unit_config.get_error_message(),
            ))

        return results

    # ========================================
    # Assertion helpers
    # ========================================

    @property
    def submit_called(self) -> bool:
        """Check if submit was called."""
        return any(c.method == "submit" for c in self.call_history)

    @property
    def submit_count(self) -> int:
        """Get number of submit calls."""
        return sum(1 for c in self.call_history if c.method == "submit")

    def get_submitted_keys(self, job_index: int = 0) -> List[str]:
        """Get keys from a submitted job."""
        submits = [c for c in self.call_history if c.method == "submit"]
        if job_index < len(submits):
            return [r.key for r in (submits[job_index].requests or [])]
        return []

    def get_all_submitted_keys(self) -> List[str]:
        """Get all keys from all submitted jobs."""
        keys = []
        for c in self.call_history:
            if c.method == "submit" and c.requests:
                keys.extend(r.key for r in c.requests)
        return keys

    @property
    def get_status_count(self) -> int:
        """Get number of get_status calls."""
        return sum(1 for c in self.call_history if c.method == "get_status")

    @property
    def get_results_called(self) -> bool:
        """Check if get_results was called."""
        return any(c.method == "get_results" for c in self.call_history)

    def clear(self) -> None:
        """Clear all state and history."""
        self.call_history.clear()
        self._submitted_requests.clear()
        self._active_jobs.clear()
        self._job_config = None
        self._job_counter = 0
        self._current_poll_count = 0
