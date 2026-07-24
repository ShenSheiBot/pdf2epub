"""Recoverable execution for one small batch request.

The main Executor manages many work units. Some post-processing operations
still need batch-only models for a single structured request. This module
provides the same core guarantees without duplicating provider setup:

- one process owns the state at a time;
- persisted jobs are resumed, never silently resubmitted;
- state is bound to provider, model, request key, and input fingerprint;
- fetched output is cached locally before validation;
- provider artifacts are removed only after the caller persists the accepted
  result and explicitly finalizes it.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

from .batch_state import BatchRunLock
from ...utils.batch_utils import BatchJobState, BatchRequest


Validator = Callable[[str], Tuple[bool, str]]


@dataclass
class PersistedBatchState:
    provider: str
    model: str
    request_key: str
    input_sha256: str
    job_name: str
    job_state: str = "RUNNING"


class PersistedBatchConflictError(RuntimeError):
    """Existing state belongs to a different request or model."""


class PersistedSingleRequestBatch:
    """Run and validate one asynchronous batch request recoverably."""

    def __init__(
        self,
        *,
        client,
        provider: str,
        model: str,
        state_path: Path,
        poll_interval: int,
    ):
        self._client = client
        self._provider = provider
        self._model = model
        self._state_path = state_path
        self._response_path = state_path.with_suffix(".response.txt")
        self._poll_interval = poll_interval

    @staticmethod
    def _fingerprint(
        provider: str,
        model: str,
        request: BatchRequest,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "request": request.to_dict(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def _save_state(self, state: PersistedBatchState) -> None:
        self._write_text_atomic(
            self._state_path,
            json.dumps(asdict(state), ensure_ascii=False, indent=2),
        )

    def _load_state(
        self,
        request: BatchRequest,
        input_sha256: str,
    ) -> PersistedBatchState | None:
        if not self._state_path.exists():
            return None
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            state = PersistedBatchState(**data)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot read persisted batch state {self._state_path}: {exc}"
            ) from exc

        expected = (
            self._provider,
            self._model,
            request.key,
            input_sha256,
        )
        actual = (
            state.provider,
            state.model,
            state.request_key,
            state.input_sha256,
        )
        if actual != expected:
            raise PersistedBatchConflictError(
                f"Persisted batch state {self._state_path} belongs to a "
                "different provider, model, or input. Inspect/cancel that job "
                "before starting a replacement."
            )
        return state

    def _restore_mapping(
        self,
        job_name: str,
        request: BatchRequest,
    ) -> None:
        if not hasattr(self._client, "restore_job_mapping"):
            return
        content_fingerprints: Dict[str, str] | None = None
        fingerprint_fn = getattr(self._client, "_content_fingerprint", None)
        if fingerprint_fn:
            content_fingerprints = {
                fingerprint_fn(request.contents): request.key
            }
        self._client.restore_job_mapping(
            job_name,
            [request.key],
            content_fingerprints,
        )

    def run(
        self,
        request: BatchRequest,
        validator: Validator,
        *,
        display_name: str,
    ) -> str:
        input_sha256 = self._fingerprint(
            self._provider,
            self._model,
            request,
        )
        lock_dir = (
            self._state_path.parent
            / ".batch_locks"
            / self._state_path.stem
        )

        with BatchRunLock(lock_dir):
            state = self._load_state(request, input_sha256)
            if state is None:
                # A crash after deleting accepted state but before deleting
                # its response cache can leave an orphan. It has no identity
                # binding anymore and must never be reused by a new request.
                self._response_path.unlink(missing_ok=True)
                state = PersistedBatchState(
                    provider=self._provider,
                    model=self._model,
                    request_key=request.key,
                    input_sha256=input_sha256,
                    job_name="",
                    job_state="SUBMITTING",
                )
                self._save_state(state)
                try:
                    state.job_name = self._client.submit(
                        [request],
                        display_name=display_name,
                    )
                except Exception:
                    state.job_state = "SUBMISSION_UNKNOWN"
                    self._save_state(state)
                    raise
                state.job_state = "RUNNING"
                self._save_state(state)
            elif not state.job_name:
                raise RuntimeError(
                    f"Batch submission outcome is unknown for "
                    f"{self._state_path}. Inspect the provider job list before "
                    "removing the state or submitting a replacement."
                )

            self._restore_mapping(state.job_name, request)

            if self._response_path.exists():
                response_text = self._response_path.read_text(encoding="utf-8")
            else:
                info = self._client.wait_for_completion(
                    state.job_name,
                    poll_interval=self._poll_interval,
                )
                state.job_state = info.state.name
                self._save_state(state)
                if info.state not in (
                    BatchJobState.SUCCEEDED,
                    BatchJobState.PARTIALLY_SUCCEEDED,
                ):
                    raise RuntimeError(
                        f"Batch job ended as {info.state.name}: {info.error}"
                    )

                responses = self._client.get_results(
                    state.job_name,
                    cleanup=False,
                )
                matching = [
                    item for item in responses
                    if item.key == request.key
                ]
                if len(matching) != 1:
                    raise RuntimeError(
                        f"Expected exactly one batch response for "
                        f"{request.key!r}, got {len(matching)}"
                    )
                response = matching[0]
                if response.error or response.text is None:
                    raise RuntimeError(
                        f"Batch response is unusable: {response.error}"
                    )
                response_text = response.text
                self._write_text_atomic(self._response_path, response_text)

            valid, reason = validator(response_text)
            if not valid:
                state.job_state = "VALIDATION_FAILED"
                self._save_state(state)
                raise ValueError(
                    f"Persisted batch response failed validation: {reason}. "
                    f"Raw response retained at {self._response_path}"
                )

            state.job_state = "VALIDATED"
            self._save_state(state)
            return response_text

    def finalize(self) -> None:
        """Release recovery material after the caller's final write succeeds."""
        lock_dir = (
            self._state_path.parent
            / ".batch_locks"
            / self._state_path.stem
        )
        with BatchRunLock(lock_dir):
            if not self._state_path.exists():
                # State is the identity binding. A response without it is an
                # orphan from a completed finalization and cannot be reused.
                self._response_path.unlink(missing_ok=True)
                return
            try:
                data = json.loads(
                    self._state_path.read_text(encoding="utf-8")
                )
                state = PersistedBatchState(**data)
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError(
                    f"Cannot finalize persisted batch state "
                    f"{self._state_path}: {exc}"
                ) from exc

            if (state.provider, state.model) != (
                self._provider,
                self._model,
            ):
                raise PersistedBatchConflictError(
                    f"Persisted batch state {self._state_path} belongs to a "
                    "different provider or model"
                )
            if state.job_state == "FINALIZED":
                self._response_path.unlink(missing_ok=True)
                self._state_path.unlink(missing_ok=True)
                return
            if state.job_state not in ("VALIDATED", "FINALIZING"):
                raise RuntimeError(
                    f"Persisted batch state {self._state_path} is "
                    f"{state.job_state!r}, not ready for finalization"
                )
            if not state.job_name or not self._response_path.exists():
                raise RuntimeError(
                    f"Persisted batch state {self._state_path} lacks a "
                    "confirmed job or validated response cache"
                )

            # Write the tombstone before remote cleanup. If the process exits
            # after deleting provider artifacts, the retained response lets a
            # retry repeat cleanup idempotently.
            state.job_state = "FINALIZING"
            self._save_state(state)
            if hasattr(self._client, "cleanup_job_artifacts"):
                self._client.cleanup_job_artifacts(state.job_name)
            state.job_state = "FINALIZED"
            self._save_state(state)
            self._response_path.unlink(missing_ok=True)
            self._state_path.unlink(missing_ok=True)
