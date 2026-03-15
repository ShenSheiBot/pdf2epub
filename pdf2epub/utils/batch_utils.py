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
from .retry_utils import default_retry, aggressive_retry


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
    # Support JOB_STATE_*, BATCH_STATE_* formats (Gemini + Vertex)
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    # Vertex-specific states
    QUEUED = "QUEUED"
    CANCELLING = "CANCELLING"
    PARTIALLY_SUCCEEDED = "PARTIALLY_SUCCEEDED"

    @classmethod
    def from_api_state(cls, state_name: str) -> 'BatchJobState':
        """Convert API state string to BatchJobState."""
        # Strip prefixes like JOB_STATE_ or BATCH_STATE_
        normalized = state_name
        for prefix in ["JOB_STATE_", "BATCH_STATE_"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        try:
            return cls(normalized)
        except ValueError:
            # Map unknown states to closest known state
            logger.warning(f"Unknown batch state '{state_name}', treating as RUNNING")
            return cls.RUNNING


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

    @default_retry
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
        import tempfile

        # Write to temp JSONL file (local, no retry needed)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in requests:
                f.write(json.dumps(req.to_dict()) + "\n")
            temp_path = f.name

        try:
            job_name = self._upload_and_create_batch(temp_path, display_name)
        finally:
            # Clean up temp file regardless of success/failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return job_name

    @default_retry
    def _upload_and_create_batch(self, temp_path: str, display_name: Optional[str] = None) -> str:
        """Upload file and create batch job (with retry)."""
        client = self._get_client()
        from google.genai import types

        logger.info(f"Uploading batch file...")

        uploaded_file = client.files.upload(
            file=temp_path,
            config=types.UploadFileConfig(
                display_name=display_name or "batch-requests",
                mime_type="jsonl"
            )
        )

        logger.info(f"Uploaded file: {uploaded_file.name}")

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

    @default_retry
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

    @aggressive_retry
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

    @default_retry
    def cancel(self, job_name: str) -> bool:
        """
        Cancel a running batch job.

        Args:
            job_name: The job name/ID

        Returns:
            True if cancellation was successful
        """
        client = self._get_client()
        client.batches.cancel(name=job_name)
        logger.info(f"Cancelled batch job: {job_name}")
        return True

    @default_retry
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


class VertexBatchClient:
    """
    Client for Vertex AI Batch Prediction API.

    Uses the same google-genai SDK as GeminiBatchClient, but with
    vertexai=True and GCS for input/output instead of File API.
    """

    COMPLETED_STATES = {
        BatchJobState.SUCCEEDED,
        BatchJobState.FAILED,
        BatchJobState.CANCELLED,
        BatchJobState.EXPIRED,
        BatchJobState.PARTIALLY_SUCCEEDED,
    }

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        poll_interval: int = 60,
        bucket_name: Optional[str] = None,
        proxy: Optional[str] = None,
    ):
        self.project = project
        self.location = location
        self.model = model
        self.poll_interval = poll_interval
        self._bucket_name = bucket_name
        self._proxy = proxy
        self._client = None
        self._storage_client = None
        # Track submitted key orders for line-order correlation
        self._job_keys: Dict[str, List[str]] = {}

        # Set proxy env vars for httpx (google-genai SDK uses trust_env=True)
        # and for google-cloud-storage / google-auth (ADC token refresh)
        if proxy:
            import os
            os.environ.setdefault("HTTPS_PROXY", proxy)
            os.environ.setdefault("HTTP_PROXY", proxy)
            logger.info(f"Vertex batch client using proxy: {proxy}")

    def _get_client(self):
        """Get or create the genai client with Vertex AI backend."""
        if self._client is None:
            from google import genai
            try:
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
            except Exception as e:
                if "credentials" in str(e).lower() or "default credentials" in str(e).lower():
                    raise RuntimeError(
                        f"Vertex AI authentication failed: {e}\n"
                        "Run: gcloud auth application-default login\n"
                        "Or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file."
                    ) from e
                raise
            logger.info(f"Created Vertex AI client (project={self.project}, location={self.location})")
        return self._client

    def _get_storage_client(self):
        """Get or create the GCS storage client."""
        if self._storage_client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for Vertex batch mode. "
                    "Install it with: uv add google-cloud-storage"
                )
            self._storage_client = storage.Client(project=self.project)
        return self._storage_client

    def _get_bucket_name(self) -> str:
        """Get or create the GCS bucket name."""
        if self._bucket_name:
            return self._bucket_name
        # Generate deterministic name with project prefix
        import hashlib
        project_hash = hashlib.sha256(self.project.encode()).hexdigest()[:8]
        self._bucket_name = f"pdf2epub-batch-{project_hash}"
        return self._bucket_name

    @default_retry
    def _ensure_bucket(self):
        """Ensure the GCS bucket exists, creating it if necessary."""
        storage_client = self._get_storage_client()
        bucket_name = self._get_bucket_name()

        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
            logger.debug(f"Using existing GCS bucket: {bucket_name}")
            return bucket_name

        try:
            bucket = storage_client.create_bucket(
                bucket_name,
                location=self.location,
            )
            logger.info(f"Created GCS bucket: {bucket_name} (location={self.location})")
        except Exception as e:
            error_msg = str(e)
            if "409" in error_msg:
                # Bucket name taken by another project
                raise ValueError(
                    f"GCS bucket '{bucket_name}' already exists in another project. "
                    f"Specify a custom bucket name in config: credentials.providers.vertex.bucket"
                ) from e
            elif "403" in error_msg:
                raise PermissionError(
                    f"Cannot create GCS bucket '{bucket_name}'. "
                    f"Ensure ADC credentials have storage.buckets.create permission. "
                    f"Run: gcloud auth application-default login"
                ) from e
            raise

        return bucket_name

    @default_retry
    def submit(
        self,
        requests: List[BatchRequest],
        display_name: Optional[str] = None,
    ) -> str:
        """Submit a batch job via GCS."""
        if not requests:
            raise ValueError("Cannot submit empty batch")

        client = self._get_client()
        bucket_name = self._ensure_bucket()
        storage_client = self._get_storage_client()

        import tempfile
        import os
        import uuid

        # Preserve key order for line-order correlation
        ordered_keys = [req.key for req in requests]

        # Write JSONL (Vertex format: no top-level 'key', just 'request')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for req in requests:
                line = {"request": {"contents": req.contents}}
                if req.config:
                    line["request"]["config"] = req.config
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            temp_path = f.name

        # Upload to GCS with unique path
        job_id = uuid.uuid4().hex[:12]
        gcs_input_path = f"batch-inputs/{job_id}.jsonl"

        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_input_path)
            blob.upload_from_filename(temp_path)
            gcs_uri = f"gs://{bucket_name}/{gcs_input_path}"
            logger.info(f"Uploaded {len(requests)} requests to {gcs_uri}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Submit batch job
        config = {}
        if display_name:
            config["display_name"] = display_name
        # Set output destination
        gcs_output_prefix = f"gs://{bucket_name}/batch-outputs/{job_id}/"
        config["dest"] = gcs_output_prefix

        # Vertex SDK adds "models/" prefix automatically (unlike Gemini which needs it explicit)
        job = client.batches.create(
            model=self.model,
            src=gcs_uri,
            config=config if config else None,
        )

        # Store key order for this job
        self._job_keys[job.name] = ordered_keys
        logger.info(f"Created Vertex batch job: {job.name}")
        return job.name

    @default_retry
    def get_status(self, job_name: str) -> BatchJobInfo:
        """Get the current status of a batch job."""
        client = self._get_client()
        job = client.batches.get(name=job_name)
        return BatchJobInfo(
            name=job.name,
            state=BatchJobState.from_api_state(job.state.name),
            model=self.model,
            error=str(job.error) if hasattr(job, 'error') and job.error else None,
        )

    def poll(self, job_name: str) -> BatchJobState:
        """Poll and return the current state."""
        info = self.get_status(job_name)
        return info.state

    def wait_for_completion(
        self,
        job_name: str,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> BatchJobInfo:
        """Wait for a batch job to complete."""
        interval = poll_interval or self.poll_interval
        start_time = time.time()
        logger.info(f"Waiting for Vertex batch job {job_name} to complete...")

        while True:
            info = self.get_status(job_name)
            if info.state in self.COMPLETED_STATES:
                elapsed = time.time() - start_time
                logger.info(
                    f"Vertex batch job completed with state {info.state.name} "
                    f"after {elapsed:.1f}s"
                )
                return info
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Vertex batch job {job_name} did not complete within {timeout}s"
                )
            logger.info(f"Job state: {info.state.name}, waiting {interval}s...")
            time.sleep(interval)

    @aggressive_retry
    def get_results(self, job_name: str) -> List[BatchResponse]:
        """Get results from a completed Vertex batch job via GCS."""
        client = self._get_client()
        job = client.batches.get(name=job_name)

        job_state = BatchJobState.from_api_state(job.state.name)
        if job_state not in (BatchJobState.SUCCEEDED, BatchJobState.PARTIALLY_SUCCEEDED):
            raise ValueError(
                f"Cannot get results for job in state {job.state.name}"
            )

        # Get output GCS prefix from job destination
        output_prefix = None
        if hasattr(job, 'dest') and job.dest:
            output_prefix = getattr(job.dest, 'gcs_uri', None)
            if isinstance(output_prefix, list):
                output_prefix = output_prefix[0] if output_prefix else None

        if not output_prefix:
            raise ValueError(
                f"Cannot determine output location for Vertex batch job {job_name}. "
                f"Job dest: {job.dest}"
            )

        # Download output JSONL from GCS (may be sharded into multiple files)
        storage_client = self._get_storage_client()
        # Parse gs://bucket/prefix
        if output_prefix.startswith("gs://"):
            parts = output_prefix[5:].split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid GCS URI: {output_prefix}")

        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        output_blobs = [b for b in blobs if b.name.endswith('.jsonl')]

        if not output_blobs:
            raise ValueError(
                f"No output JSONL files found under {output_prefix}"
            )

        # Read and parse all output lines
        logger.info(f"Downloading results from {output_prefix}")
        all_lines = []
        parse_errors = 0
        for blob in sorted(output_blobs, key=lambda b: b.name):
            content = blob.download_as_text()
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                if line.strip():
                    try:
                        all_lines.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        parse_errors += 1
                        logger.warning(f"JSON parse error in {blob.name} line {line_num}: {e}")
                        all_lines.append({"status": {"code": -1, "message": f"JSON parse error: {e}"}})

        if parse_errors:
            logger.warning(f"{parse_errors} lines had JSON parse errors")
        logger.info(f"Retrieved {len(all_lines)} results from {len(output_blobs)} output file(s)")

        # Map results using line-order correlation
        ordered_keys = self._job_keys.get(job_name, [])

        results = []
        for i, data in enumerate(all_lines):
            key = ordered_keys[i] if i < len(ordered_keys) else f"unknown_{i}"

            batch_resp = BatchResponse(key=key, raw_response=data)

            # Check for per-line error
            status = data.get('status', '')
            if isinstance(status, dict):
                code = status.get('code', 0)
                if code != 0:
                    batch_resp.error = status.get('message', f'Error code {code}')
                    results.append(batch_resp)
                    continue
            elif isinstance(status, str) and status and status.upper() != 'OK':
                batch_resp.error = status
                results.append(batch_resp)
                continue

            # Extract response text
            response = data.get('response', {})
            if isinstance(response, dict):
                candidates = response.get('candidates', [])
                if candidates:
                    parts = candidates[0].get('content', {}).get('parts', [])
                    if parts:
                        batch_resp.text = parts[0].get('text', '')
            elif isinstance(response, str):
                batch_resp.text = response

            results.append(batch_resp)

        # Report missing results
        if ordered_keys and len(all_lines) < len(ordered_keys):
            missing = len(ordered_keys) - len(all_lines)
            logger.warning(
                f"Vertex batch returned {len(all_lines)} results but "
                f"{len(ordered_keys)} were submitted ({missing} missing)"
            )
            for i in range(len(all_lines), len(ordered_keys)):
                results.append(BatchResponse(
                    key=ordered_keys[i],
                    error="No response from Vertex batch (missing from output)",
                ))

        # Cleanup GCS artifacts (best-effort)
        self._cleanup_gcs(bucket_name, prefix, blobs)

        return results

    @default_retry
    def _cleanup_gcs(self, bucket_name: str, prefix: str, blobs: list):
        """Clean up GCS input/output files after results are retrieved."""
        try:
            storage_client = self._get_storage_client()
            bucket = storage_client.bucket(bucket_name)
            for blob in blobs:
                blob.delete()
            # Also clean up the input file if we can find it
            # Input path: batch-inputs/{job_id}.jsonl
            # Output prefix: batch-outputs/{job_id}/
            if "batch-outputs/" in prefix:
                job_id = prefix.split("batch-outputs/")[1].rstrip("/")
                input_blob = bucket.blob(f"batch-inputs/{job_id}.jsonl")
                if input_blob.exists():
                    input_blob.delete()
            logger.debug(f"Cleaned up GCS artifacts under {prefix}")
        except Exception as e:
            logger.warning(f"Failed to clean up GCS artifacts: {e}")

    @default_retry
    def cancel(self, job_name: str) -> bool:
        """Cancel a running batch job."""
        client = self._get_client()
        client.batches.cancel(name=job_name)
        logger.info(f"Cancelled Vertex batch job: {job_name}")
        return True

    @default_retry
    def list_jobs(self, limit: int = 10) -> List[BatchJobInfo]:
        """List recent batch jobs."""
        client = self._get_client()
        jobs = []
        for job in client.batches.list():
            jobs.append(BatchJobInfo(
                name=job.name,
                state=BatchJobState.from_api_state(job.state.name),
                model=self.model,
            ))
            if len(jobs) >= limit:
                break
        return jobs
