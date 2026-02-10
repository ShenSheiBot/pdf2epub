"""
Gemini Batch API utilities.

This module provides a client for interacting with the Gemini Batch Prediction API,
enabling asynchronous, high-throughput processing at 50% cost reduction.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# Default batch configuration
BATCH_DEFAULTS = {
    "base_url": None,  # Must be configured via config.yaml or provider settings
    "provider": "gemini",
    "poll_interval": 60,
    "max_retries": 3,
    "polish": {
        "model": "gemini-3-flash-preview",
    },
    "translate": {
        "model": "gemini-3-pro-preview",
    },
}


class BatchJobState(Enum):
    """Batch job states."""
    # Support both JOB_STATE_* and BATCH_STATE_* formats
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

    @classmethod
    def from_api_state(cls, state_name: str) -> 'BatchJobState':
        """Convert API state string to BatchJobState."""
        # Strip prefixes like JOB_STATE_ or BATCH_STATE_
        normalized = state_name
        for prefix in ["JOB_STATE_", "BATCH_STATE_"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return cls(normalized)


@dataclass
class BatchRequest:
    """A single request in a batch job."""
    key: str  # Unique identifier for correlation
    contents: List[Dict]  # Content parts for the model
    config: Optional[Dict] = None  # Optional per-request config

    def to_dict(self) -> Dict:
        """Convert to JSONL-compatible dict."""
        request = {"contents": self.contents}
        if self.config:
            request["config"] = self.config
        return {"key": self.key, "request": request}


@dataclass
class BatchResponse:
    """A single response from a batch job."""
    key: str
    text: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class BatchJobInfo:
    """Information about a batch job."""
    name: str
    state: BatchJobState
    model: str
    created_time: Optional[float] = None
    completed_time: Optional[float] = None
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    error: Optional[str] = None


class GeminiBatchClient:
    """
    Client for Gemini Batch Prediction API.

    Provides methods to submit batch jobs, poll for completion,
    and retrieve results.
    """

    COMPLETED_STATES = {
        BatchJobState.SUCCEEDED,
        BatchJobState.FAILED,
        BatchJobState.CANCELLED,
        BatchJobState.EXPIRED,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        poll_interval: int = 60,
        base_url: Optional[str] = None
    ):
        """
        Initialize the batch client.

        Args:
            api_key: Google API key
            model: Model to use for batch processing
            poll_interval: Seconds between status polls
            base_url: Optional custom base URL
        """
        self.api_key = api_key
        self.model = model
        self.poll_interval = poll_interval
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create the genai client."""
        if self._client is None:
            from google import genai

            # Build http_options similar to GeminiClient in network_utils.py
            http_options = {}

            if self.base_url:
                # Custom endpoint (proxy)
                http_options['base_url'] = self.base_url
                logger.info(f"Using custom Batch API endpoint: {self.base_url}")

            # Set longer timeout for batch operations
            http_options['timeout'] = 300000  # 300 seconds in milliseconds

            if http_options:
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=http_options
                )
            else:
                self._client = genai.Client(api_key=self.api_key)

        return self._client

    def submit(
        self,
        requests: List[BatchRequest],
        display_name: Optional[str] = None
    ) -> str:
        """
        Submit a batch job by uploading a JSONL file.

        Using file upload instead of inline requests because:
        1. Inline requests don't support 'key' field for result correlation
        2. File upload supports larger batches

        Args:
            requests: List of BatchRequest objects
            display_name: Optional display name for the job

        Returns:
            Job name (ID) for tracking
        """
        client = self._get_client()
        from google.genai import types
        import tempfile
        import os

        # Write requests to temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for req in requests:
                f.write(json.dumps(req.to_dict(), ensure_ascii=False) + "\n")
            temp_path = f.name

        logger.info(f"Uploading batch file with {len(requests)} requests...")

        try:
            # Upload the file
            uploaded_file = client.files.upload(
                file=temp_path,
                config=types.UploadFileConfig(
                    display_name=display_name or "batch-requests",
                    mime_type="application/jsonl"
                )
            )

            logger.info(f"Uploaded file: {uploaded_file.name}")

            # Create batch job from file
            config = {}
            if display_name:
                config["display_name"] = display_name

            job = client.batches.create(
                model=f"models/{self.model}",
                src=uploaded_file.name,
                config=config if config else None
            )

            logger.info(f"Created batch job: {job.name}")
            return job.name

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def submit_from_file(
        self,
        requests: List[BatchRequest],
        display_name: Optional[str] = None
    ) -> str:
        """
        Submit a batch job by uploading a JSONL file.

        Useful for larger batches (>1000 requests).

        Args:
            requests: List of BatchRequest objects
            display_name: Optional display name for the job

        Returns:
            Job name (ID) for tracking
        """
        client = self._get_client()
        from google.genai import types
        import tempfile

        # Write to temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in requests:
                f.write(json.dumps(req.to_dict()) + "\n")
            temp_path = f.name

        logger.info(f"Uploading batch file with {len(requests)} requests...")

        # Upload the file
        uploaded_file = client.files.upload(
            file=temp_path,
            config=types.UploadFileConfig(
                display_name=display_name or "batch-requests",
                mime_type="jsonl"
            )
        )

        # Clean up temp file
        Path(temp_path).unlink()

        logger.info(f"Uploaded file: {uploaded_file.name}")

        # Create batch job from file
        config = {}
        if display_name:
            config["display_name"] = display_name

        job = client.batches.create(
            model=f"models/{self.model}",
            src=uploaded_file.name,
            config=config if config else None
        )

        logger.info(f"Created batch job: {job.name}")
        return job.name

    def get_status(self, job_name: str) -> BatchJobInfo:
        """
        Get the current status of a batch job.

        Args:
            job_name: The job name/ID

        Returns:
            BatchJobInfo with current state
        """
        client = self._get_client()
        job = client.batches.get(name=job_name)

        return BatchJobInfo(
            name=job.name,
            state=BatchJobState.from_api_state(job.state.name),
            model=self.model,
            error=str(job.error) if hasattr(job, 'error') and job.error else None
        )

    def poll(self, job_name: str) -> BatchJobState:
        """
        Poll and return the current state.

        Args:
            job_name: The job name/ID

        Returns:
            Current BatchJobState
        """
        info = self.get_status(job_name)
        return info.state

    def wait_for_completion(
        self,
        job_name: str,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> BatchJobInfo:
        """
        Wait for a batch job to complete.

        Args:
            job_name: The job name/ID
            poll_interval: Seconds between polls (uses default if None)
            timeout: Maximum seconds to wait (None = no limit)

        Returns:
            Final BatchJobInfo

        Raises:
            TimeoutError: If timeout exceeded
        """
        interval = poll_interval or self.poll_interval
        start_time = time.time()

        logger.info(f"Waiting for batch job {job_name} to complete...")

        while True:
            info = self.get_status(job_name)

            if info.state in self.COMPLETED_STATES:
                elapsed = time.time() - start_time
                logger.info(
                    f"Batch job completed with state {info.state.name} "
                    f"after {elapsed:.1f}s"
                )
                return info

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Batch job {job_name} did not complete within {timeout}s"
                )

            logger.info(f"Job state: {info.state.name}, waiting {interval}s...")
            time.sleep(interval)

    def get_results(self, job_name: str) -> List[BatchResponse]:
        """
        Get results from a completed batch job.

        Args:
            job_name: The job name/ID

        Returns:
            List of BatchResponse objects
        """
        client = self._get_client()
        job = client.batches.get(name=job_name)

        job_state = BatchJobState.from_api_state(job.state.name)
        if job_state != BatchJobState.SUCCEEDED:
            raise ValueError(
                f"Cannot get results for job in state {job.state.name}"
            )

        results = []

        # Check for inline responses
        if hasattr(job, 'dest') and job.dest:
            if hasattr(job.dest, 'inlined_responses') and job.dest.inlined_responses:
                for resp in job.dest.inlined_responses:
                    batch_resp = BatchResponse(
                        key=resp.key if hasattr(resp, 'key') else "",
                        raw_response=resp
                    )
                    if hasattr(resp, 'response') and resp.response:
                        # Extract text from response
                        if hasattr(resp.response, 'text'):
                            batch_resp.text = resp.response.text
                        elif hasattr(resp.response, 'candidates'):
                            # Handle structured response
                            candidates = resp.response.candidates
                            if candidates and len(candidates) > 0:
                                content = candidates[0].content
                                if hasattr(content, 'parts') and content.parts:
                                    batch_resp.text = content.parts[0].text
                    if hasattr(resp, 'error') and resp.error:
                        batch_resp.error = str(resp.error)
                    results.append(batch_resp)

            # Check for file-based results
            elif hasattr(job.dest, 'file_name') and job.dest.file_name:
                file_content = client.files.download(file=job.dest.file_name)
                content_str = file_content.decode('utf-8')

                for line in content_str.strip().split('\n'):
                    if line:
                        data = json.loads(line)
                        batch_resp = BatchResponse(
                            key=data.get('key', ''),
                            raw_response=data
                        )
                        if 'response' in data:
                            resp_data = data['response']
                            if isinstance(resp_data, dict):
                                if 'text' in resp_data:
                                    batch_resp.text = resp_data['text']
                                elif 'candidates' in resp_data:
                                    candidates = resp_data['candidates']
                                    if candidates:
                                        parts = candidates[0].get('content', {}).get('parts', [])
                                        if parts:
                                            batch_resp.text = parts[0].get('text', '')
                            elif isinstance(resp_data, str):
                                batch_resp.text = resp_data
                        if 'error' in data:
                            batch_resp.error = str(data['error'])
                        results.append(batch_resp)

        return results

    def cancel(self, job_name: str) -> bool:
        """
        Cancel a running batch job.

        Args:
            job_name: The job name/ID

        Returns:
            True if cancellation was successful
        """
        client = self._get_client()
        try:
            client.batches.cancel(name=job_name)
            logger.info(f"Cancelled batch job: {job_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel batch job {job_name}: {e}")
            return False

    def list_jobs(self, limit: int = 10) -> List[BatchJobInfo]:
        """
        List recent batch jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of BatchJobInfo objects
        """
        client = self._get_client()
        jobs = []

        for job in client.batches.list():
            jobs.append(BatchJobInfo(
                name=job.name,
                state=BatchJobState.from_api_state(job.state.name),
                model=self.model
            ))
            if len(jobs) >= limit:
                break

        return jobs
