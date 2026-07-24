import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdf2epub.core.executor import (
    BatchStateConflictError,
    ChainEntry,
    Executor,
    PersistedBatchConflictError,
    PersistedSingleRequestBatch,
    WorkUnit,
)
from pdf2epub.core.executor.state import QuotaConfig, create_unit_state
from pdf2epub.core.executor.executor import _batch_request_sha256
from pdf2epub.core.executor._protocol import ProcessResult
from pdf2epub.core.hooks import DefaultErrorClassifier
from pdf2epub.core.phase.phase import Phase
from pdf2epub.core.executor.batch_state import (
    BatchRunLock,
    BatchRunLockedError,
    MegaUnitState,
    get_mega_unit_id,
)
from pdf2epub.utils.batch_utils import (
    BatchJobState,
    BatchRequest,
    BatchResponse,
    GeminiBatchClient,
    VertexBatchClient,
    probe_explicit_vertex_credentials,
)
from pdf2epub.commands import cancel_batch
from pdf2epub.commands import translate_v2


def test_batch_run_lock_rejects_second_owner(tmp_path: Path):
    state_dir = tmp_path / "batch_states"

    with BatchRunLock(state_dir):
        with pytest.raises(BatchRunLockedError, match="already using"):
            with BatchRunLock(state_dir):
                pass


def test_executor_batch_lifecycle_lock_spans_nested_calls(tmp_path: Path):
    first = _batch_executor(tmp_path, _CompletedBatchClient())
    second = _batch_executor(tmp_path, _CompletedBatchClient())

    with first.batch_run_lock():
        with first.batch_run_lock():
            with pytest.raises(BatchRunLockedError, match="already using"):
                with second.batch_run_lock():
                    pass


def test_executor_requires_explicit_resume_for_existing_batch_state(tmp_path: Path):
    class BatchClient:
        def cancel(self, _job_name):
            raise AssertionError("execute() must not cancel an existing job")

    executor = Executor(
        llm_client=object(),
        model_chain=[
            ChainEntry(provider="vertex", model="flash", mode="batch")
        ],
        processor=object(),
        hooks=object(),
        batch_client=BatchClient(),
        batch_state_dir=tmp_path,
    )
    state_path = tmp_path / "batch_states" / "batch_existing.json"
    MegaUnitState(job_name="projects/example/jobs/1").save(state_path)

    with pytest.raises(RuntimeError, match=r"Use --resume"):
        executor.execute(
            [WorkUnit(id="chapter_1", file_key="chapter_1", content="text")]
        )

    assert state_path.exists()


def test_resume_rejects_state_for_a_different_pending_unit_set(
    tmp_path: Path,
):
    class BatchClient:
        def submit(self, _requests):
            raise AssertionError("mismatched resume must not submit")

    executor = Executor(
        llm_client=object(),
        model_chain=[
            ChainEntry(provider="vertex", model="flash", mode="batch")
        ],
        processor=object(),
        hooks=object(),
        batch_client=BatchClient(),
        batch_state_dir=tmp_path,
    )
    MegaUnitState(
        job_name="projects/example/jobs/old",
        provider="vertex",
        model="flash",
    ).save(tmp_path / "batch_states" / "batch_old.json")

    with pytest.raises(BatchStateConflictError, match="unit membership"):
        executor.execute(
            [WorkUnit(id="chapter_1", file_key="chapter_1", content="text")],
            resume_batch=True,
        )


def test_indeterminate_submission_is_fatal_and_persisted(tmp_path: Path):
    class BatchClient:
        def submit(self, _requests):
            raise TimeoutError("connection closed after request")

    class Processor:
        def build_prompt(self, content, _context):
            return content

    class Hooks:
        def pre_process(self, _uid, _content, _context):
            return SimpleNamespace(should_process=True)

    executor = Executor(
        llm_client=object(),
        model_chain=[
            ChainEntry(provider="vertex", model="flash", mode="batch")
        ],
        processor=Processor(),
        hooks=Hooks(),
        batch_client=BatchClient(),
        batch_state_dir=tmp_path,
        online_fallback_threshold=1,
    )
    unit = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="text",
    )

    with pytest.raises(BatchStateConflictError, match="no job name"):
        executor.execute([unit])

    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id([unit.id])}.json"
    )
    state = MegaUnitState.load(state_path)
    assert state is not None
    assert state.job_name == ""
    assert state.job_state == "SUBMISSION_UNKNOWN"


def test_executor_defers_remote_cleanup_until_results_are_handled(
    tmp_path: Path,
):
    class BatchClient:
        COMPLETED_STATES = {BatchJobState.SUCCEEDED}

        def __init__(self):
            self.cleanup_arguments = []
            self.cleaned_jobs = []

        def submit(self, _requests):
            return "jobs/1"

        def get_status(self, job_name):
            return SimpleNamespace(
                name=job_name,
                state=BatchJobState.SUCCEEDED,
                error=None,
            )

        def get_results(self, _job_name, cleanup=True):
            self.cleanup_arguments.append(cleanup)
            return [
                BatchResponse(
                    key="chapter_1",
                    text="translated",
                )
            ]

        def cleanup_job_artifacts(self, job_name):
            self.cleaned_jobs.append(job_name)

    class Processor:
        def build_prompt(self, content, _context):
            return content

        def clean_response(self, response):
            return response

        def post_process(self, response, _context):
            return response

    class Hooks:
        def pre_process(self, _uid, _content, _context):
            return SimpleNamespace(should_process=True)

        def post_process(
            self,
            _uid,
            _original,
            response,
            _chapter_type,
            _context,
        ):
            return response, SimpleNamespace(
                accepted=True,
                context_ready=False,
            )

    chain = [
        ChainEntry(provider="vertex", model="flash", mode="batch")
    ]
    client = BatchClient()
    executor = Executor(
        llm_client=object(),
        model_chain=chain,
        processor=Processor(),
        hooks=Hooks(),
        batch_client=client,
        batch_state_dir=tmp_path,
        online_fallback_threshold=1,
        batch_poll_interval=0,
    )
    unit = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="source",
    )
    unit_states = {
        unit.id: create_unit_state(
            chain=chain,
            quota_config=QuotaConfig(),
            content=unit.content,
        )
    }

    results = executor._process_batch_as_unit(
        [unit.id],
        unit_states,
        {unit.id: unit},
        None,
        {unit.id: unit.content},
        False,
    )

    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id([unit.id])}.json"
    )
    assert results[0][1].success
    assert client.cleanup_arguments == [False]
    assert state_path.exists()
    assert client.cleaned_jobs == []

    executor._finalize_batch_job([unit.id])

    assert client.cleaned_jobs == ["jobs/1"]
    assert not state_path.exists()


def test_batch_resume_reconstructs_units_skipped_before_submission(
    tmp_path: Path,
):
    class BatchClient:
        COMPLETED_STATES = {BatchJobState.SUCCEEDED}

        def submit(self, _requests):
            raise AssertionError("resume must not submit a replacement job")

        def restore_job_mapping(self, _job_name, _keys, _fingerprints):
            return None

        def get_status(self, job_name):
            return SimpleNamespace(
                name=job_name,
                state=BatchJobState.SUCCEEDED,
                error=None,
            )

        def get_results(self, _job_name, cleanup=True):
            assert cleanup is False
            return [
                BatchResponse(
                    key="chapter_1",
                    text="translated",
                )
            ]

    class Processor:
        def build_prompt(self, content, _context):
            return content

        def clean_response(self, response):
            return response

        def post_process(self, response, _context):
            return response

    class Hooks:
        def pre_process(self, uid, _content, _context):
            if uid == "front_cover":
                return SimpleNamespace(
                    should_process=False,
                    fallback_result="cover",
                    skip_reason="non-content",
                )
            return SimpleNamespace(should_process=True)

        def post_process(
            self,
            _uid,
            _original,
            response,
            _chapter_type,
            _context,
        ):
            return response, SimpleNamespace(
                accepted=True,
                context_ready=False,
            )

    chain = [
        ChainEntry(provider="vertex", model="flash", mode="batch")
    ]
    client = BatchClient()
    executor = Executor(
        llm_client=object(),
        model_chain=chain,
        processor=Processor(),
        hooks=Hooks(),
        batch_client=client,
        batch_state_dir=tmp_path,
        online_fallback_threshold=1,
        batch_poll_interval=0,
    )
    units = [
        WorkUnit(
            id="chapter_1",
            file_key="chapter_1",
            content="source",
        ),
        WorkUnit(
            id="front_cover",
            file_key="front_cover",
            content="image",
        ),
    ]
    unit_states = {
        unit.id: create_unit_state(
            chain=chain,
            quota_config=QuotaConfig(),
            content=unit.content,
        )
        for unit in units
    }
    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id([unit.id for unit in units])}.json"
    )
    MegaUnitState(
        job_name="jobs/1",
        job_state="SUCCEEDED",
        provider="vertex",
        model="flash",
        unit_ids=["chapter_1", "front_cover"],
        processing_keys=["chapter_1"],
        request_sha256=_batch_request_sha256(
            "vertex",
            "flash",
            ["chapter_1", "front_cover"],
            [
                BatchRequest(
                    key="chapter_1",
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": "source"}],
                        }
                    ],
                )
            ],
            [
                {
                    "key": "front_cover",
                    "fallback": "cover",
                    "reason": "non-content",
                }
            ],
        ),
    ).save(state_path)

    results = executor._process_batch_as_unit(
        [unit.id for unit in units],
        unit_states,
        {unit.id: unit for unit in units},
        None,
        {unit.id: unit.content for unit in units},
        True,
    )
    by_id = dict(results)

    assert by_id["chapter_1"].success
    assert by_id["front_cover"].success
    assert by_id["front_cover"].skipped
    assert by_id["front_cover"].content == "cover"


def test_executor_resumes_disjoint_persisted_batches_separately(
    tmp_path: Path,
):
    class Client:
        COMPLETED_STATES = {BatchJobState.SUCCEEDED}

        def submit(self, _requests):
            raise AssertionError("resume must not submit a replacement")

        def restore_job_mapping(self, *_args):
            return None

        def get_status(self, job_name):
            return SimpleNamespace(
                name=job_name,
                state=BatchJobState.SUCCEEDED,
                error=None,
            )

        def get_results(self, job_name, cleanup=True):
            assert cleanup is False
            key = job_name.rsplit("/", 1)[-1]
            return [BatchResponse(key=key, text=f"translated {key}")]

    executor = Executor(
        llm_client=object(),
        model_chain=[
            ChainEntry(provider="vertex", model="flash", mode="batch")
        ],
        processor=_AcceptingProcessor(),
        hooks=_AcceptingHooks(),
        batch_client=Client(),
        batch_state_dir=tmp_path,
        online_fallback_threshold=5,
        batch_poll_interval=0,
    )
    units = [
        WorkUnit(id=key, file_key=key, content=f"source {key}")
        for key in ("chapter_1", "chapter_2")
    ]
    for unit in units:
        request = BatchRequest(
            key=unit.id,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": unit.content}],
                }
            ],
        )
        MegaUnitState(
            job_name=f"jobs/{unit.id}",
            job_state="SUCCEEDED",
            provider="vertex",
            model="flash",
            unit_ids=[unit.id],
            processing_keys=[unit.id],
            request_sha256=_batch_request_sha256(
                "vertex",
                "flash",
                [unit.id],
                [request],
                [],
            ),
        ).save(
            tmp_path
            / "batch_states"
            / f"{get_mega_unit_id([unit.id])}.json"
        )

    result = executor.execute(units, resume_batch=True)

    assert result.completed == {"chapter_1", "chapter_2"}
    assert result.results == {
        "chapter_1": "translated chapter_1",
        "chapter_2": "translated chapter_2",
    }
    assert sorted(result.batch_jobs) == [["chapter_1"], ["chapter_2"]]


class _AcceptingProcessor:
    def build_prompt(self, content, _context):
        return content

    def clean_response(self, response):
        return response

    def post_process(self, response, _context):
        return response


class _AcceptingHooks:
    def pre_process(self, _uid, _content, _context):
        return SimpleNamespace(should_process=True)

    def post_process(
        self,
        _uid,
        _original,
        response,
        _chapter_type,
        _context,
    ):
        return response, SimpleNamespace(
            accepted=True,
            context_ready=False,
        )


class _CompletedBatchClient:
    COMPLETED_STATES = {BatchJobState.SUCCEEDED}

    def __init__(self):
        self.submissions = 0
        self.cleanups = []

    def submit(self, _requests):
        self.submissions += 1
        return "jobs/1"

    def get_status(self, job_name):
        return SimpleNamespace(
            name=job_name,
            state=BatchJobState.SUCCEEDED,
            error=None,
        )

    def get_results(self, _job_name, cleanup=True):
        assert cleanup is False
        return [BatchResponse(key="chapter_1", text="translated")]

    def cleanup_job_artifacts(self, job_name):
        self.cleanups.append(job_name)


def _batch_executor(tmp_path: Path, client, saver=None):
    return Executor(
        llm_client=object(),
        model_chain=[
            ChainEntry(provider="vertex", model="flash", mode="batch")
        ],
        processor=_AcceptingProcessor(),
        hooks=_AcceptingHooks(),
        batch_client=client,
        saver=saver,
        batch_state_dir=tmp_path,
        online_fallback_threshold=1,
        batch_poll_interval=0,
    )


def test_batch_save_failure_retains_remote_job_and_state(tmp_path: Path):
    class Attempt:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def success(self, *_args, **_kwargs):
            return False

    class Saver:
        def attempt(self, _uid, _model):
            return Attempt()

    client = _CompletedBatchClient()
    executor = _batch_executor(tmp_path, client, saver=Saver())
    unit = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="source",
    )

    with pytest.raises(BatchStateConflictError, match="failed to persist"):
        executor.execute([unit])

    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id([unit.id])}.json"
    )
    assert state_path.exists()
    assert MegaUnitState.load(state_path).job_state == "SUCCEEDED"
    assert client.cleanups == []


def test_longest_fallback_false_save_does_not_complete_unit():
    classifier = DefaultErrorClassifier()

    class Hooks:
        def classify_error(self, error):
            error_type = classifier.classify(error)
            return error_type, classifier.get_effect(error_type)

    class Attempt:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def failure(self, *_args):
            return None

        def success(self, *_args, **_kwargs):
            return False

    class Saver:
        def attempt(self, _uid, _model):
            return Attempt()

    chain = [ChainEntry(provider="vertex", model="flash", mode="batch")]
    executor = Executor(
        llm_client=object(),
        model_chain=chain,
        processor=object(),
        hooks=Hooks(),
        saver=Saver(),
    )
    state = create_unit_state(
        chain=chain,
        quota_config=QuotaConfig(),
        content="source",
    )
    state.record_attempt("fallback-output")
    results = {}
    completed = set()
    fallback_used = set()

    with pytest.raises(BatchStateConflictError, match="persist fallback"):
        executor._handle_failure(
            "chapter_1",
            ProcessResult(success=False, error=ValueError("invalid")),
            state,
            {"chapter_1": state},
            set(),
            completed,
            set(),
            set(),
            set(),
            results,
            fallback_used,
            "vertex/flash",
        )

    assert completed == set()
    assert fallback_used == set()
    assert results == {}


def test_batch_resume_rejects_changed_request_content(tmp_path: Path):
    client = _CompletedBatchClient()
    executor = _batch_executor(tmp_path, client)
    original = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="OLD INPUT",
    )
    original_states = {
        original.id: create_unit_state(
            chain=executor._model_chain,
            quota_config=QuotaConfig(),
            content=original.content,
        )
    }
    executor._process_batch_as_unit(
        [original.id],
        original_states,
        {original.id: original},
        None,
        {original.id: original.content},
        False,
    )

    class ChangedPromptProcessor(_AcceptingProcessor):
        def build_prompt(self, content, _context):
            return f"CHANGED PROMPT: {content}"

    changed_prompt_executor = _batch_executor(tmp_path, client)
    changed_prompt_executor._processor = ChangedPromptProcessor()
    prompt_states = {
        original.id: create_unit_state(
            chain=changed_prompt_executor._model_chain,
            quota_config=QuotaConfig(),
            content=original.content,
        )
    }
    with pytest.raises(
        BatchStateConflictError,
        match="exact request content",
    ):
        changed_prompt_executor._process_batch_as_unit(
            [original.id],
            prompt_states,
            {original.id: original},
            None,
            {original.id: original.content},
            True,
        )

    changed = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="NEW INPUT",
    )
    changed_states = {
        changed.id: create_unit_state(
            chain=executor._model_chain,
            quota_config=QuotaConfig(),
            content=changed.content,
        )
    }
    with pytest.raises(
        BatchStateConflictError,
        match="exact request content",
    ):
        executor._process_batch_as_unit(
            [changed.id],
            changed_states,
            {changed.id: changed},
            None,
            {changed.id: changed.content},
            True,
        )

    assert client.submissions == 1


def test_batch_resume_rejects_legacy_state_without_request_identity(
    tmp_path: Path,
):
    executor = _batch_executor(tmp_path, _CompletedBatchClient())
    unit = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="source",
    )
    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id([unit.id])}.json"
    )
    MegaUnitState(
        job_name="jobs/old",
        job_state="RUNNING",
        provider="vertex",
        model="flash",
        processing_keys=[unit.id],
    ).save(state_path)
    states = {
        unit.id: create_unit_state(
            chain=executor._model_chain,
            quota_config=QuotaConfig(),
            content=unit.content,
        )
    }

    with pytest.raises(BatchStateConflictError, match="exact request content"):
        executor._process_batch_as_unit(
            [unit.id],
            states,
            {unit.id: unit},
            None,
            {unit.id: unit.content},
            True,
        )


def test_batch_resume_rejects_any_state_without_job_name(tmp_path: Path):
    executor = _batch_executor(tmp_path, _CompletedBatchClient())
    unit = WorkUnit(
        id="chapter_1",
        file_key="chapter_1",
        content="source",
    )
    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id([unit.id])}.json"
    )
    MegaUnitState(
        job_name="",
        job_state="RUNNING",
    ).save(state_path)
    states = {
        unit.id: create_unit_state(
            chain=executor._model_chain,
            quota_config=QuotaConfig(),
            content=unit.content,
        )
    }

    with pytest.raises(BatchStateConflictError, match="no confirmed job name"):
        executor._process_batch_as_unit(
            [unit.id],
            states,
            {unit.id: unit},
            None,
            {unit.id: unit.content},
            True,
        )


def test_batch_finalization_tombstone_is_recoverable(tmp_path: Path):
    class Client:
        def __init__(self):
            self.calls = 0

        def cleanup_job_artifacts(self, _job_name):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("interrupted cleanup")

    client = Client()
    executor = _batch_executor(tmp_path, client)
    unit_ids = ["chapter_1"]
    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id(unit_ids)}.json"
    )
    MegaUnitState(
        job_name="jobs/1",
        job_state="SUCCEEDED",
        provider="vertex",
        model="flash",
        unit_ids=unit_ids,
        processing_keys=unit_ids,
        request_sha256="a" * 64,
    ).save(state_path)

    with pytest.raises(RuntimeError, match="interrupted cleanup"):
        executor.finalize_batch_jobs([unit_ids])

    assert MegaUnitState.load(state_path).job_state == "FINALIZING"
    executor.recover_finalizing_batches()
    assert not state_path.exists()
    assert client.calls == 2


def test_malformed_finalizing_state_is_retained(tmp_path: Path):
    client = _CompletedBatchClient()
    executor = _batch_executor(tmp_path, client)
    state_path = (
        tmp_path
        / "batch_states"
        / f"{get_mega_unit_id(['chapter_1'])}.json"
    )
    MegaUnitState(
        job_name="",
        job_state="FINALIZING",
    ).save(state_path)

    with pytest.raises(BatchStateConflictError, match="invalid job identity"):
        executor.recover_finalizing_batches()

    assert state_path.exists()
    assert client.cleanups == []


def test_vertex_output_without_request_fails_closed():
    client = object.__new__(VertexBatchClient)
    client.model = "model"
    client._job_keys = {}
    client._job_fingerprints = {}

    with pytest.raises(ValueError, match="without the original request"):
        client._correlate_output_key(
            "jobs/1",
            {"response": {"candidates": []}},
            {"fingerprint": "chapter_1"},
            set(),
        )


def test_vertex_mapping_requires_one_to_one_fingerprints():
    client = object.__new__(VertexBatchClient)
    client._job_keys = {}
    client._job_fingerprints = {}

    with pytest.raises(ValueError, match="one-to-one"):
        client.restore_job_mapping(
            "jobs/1",
            ["chapter_1", "chapter_2"],
            {"one-fingerprint": "chapter_2"},
        )


def test_vertex_cleanup_failure_propagates_to_keep_tombstone():
    client = object.__new__(VertexBatchClient)
    client._get_storage_client = lambda: SimpleNamespace(
        bucket=lambda _name: SimpleNamespace()
    )

    class Blob:
        name = "batch-outputs/job/result.jsonl"

        def delete(self):
            raise PermissionError("cleanup denied")

    with pytest.raises(PermissionError, match="cleanup denied"):
        client._cleanup_gcs("bucket", "batch-outputs/job/", [Blob()])


def test_vertex_cleanup_rejects_bucket_root_prefix():
    client = object.__new__(VertexBatchClient)
    client._get_client = lambda: SimpleNamespace(
        batches=SimpleNamespace(
            get=lambda **_kwargs: SimpleNamespace(
                dest=SimpleNamespace(gcs_uri="gs://shared-bucket")
            )
        )
    )
    client._get_storage_client = lambda: pytest.fail(
        "unsafe prefix must fail before listing the bucket"
    )

    with pytest.raises(ValueError, match="unsafe GCS cleanup prefix"):
        client.cleanup_job_artifacts("jobs/1")


def test_vertex_cleanup_validates_all_blobs_before_any_delete():
    client = object.__new__(VertexBatchClient)
    client._get_storage_client = lambda: SimpleNamespace(
        bucket=lambda _name: SimpleNamespace()
    )
    deleted = []

    class Blob:
        def __init__(self, name):
            self.name = name

        def delete(self):
            deleted.append(self.name)

    with pytest.raises(ValueError, match="objects outside"):
        client._cleanup_gcs(
            "shared-bucket",
            "batch-outputs/job-1/",
            [
                Blob("batch-outputs/job-1/result.jsonl"),
                Blob("unrelated/private.txt"),
            ],
        )

    assert deleted == []


def test_batch_submit_is_not_retried_after_unknown_outcome(tmp_path: Path):
    class Batches:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            raise TimeoutError("provider accepted request, response lost")

    class Files:
        def upload(self, **_kwargs):
            return SimpleNamespace(name="files/input")

    gemini_batches = Batches()
    gemini = GeminiBatchClient(api_key="test", model="model")
    gemini._client = SimpleNamespace(
        files=Files(),
        batches=gemini_batches,
    )
    request = BatchRequest(
        key="chapter_1",
        contents=[{"role": "user", "parts": [{"text": "source"}]}],
    )

    with pytest.raises(TimeoutError, match="response lost"):
        gemini.submit([request])
    assert gemini_batches.calls == 1

    vertex_batches = Batches()
    vertex = object.__new__(VertexBatchClient)
    vertex.model = "model"
    vertex._job_keys = {}
    vertex._job_fingerprints = {}
    vertex._get_client = lambda: SimpleNamespace(batches=vertex_batches)
    vertex._ensure_bucket = lambda: "bucket"

    class Blob:
        def upload_from_filename(self, _path):
            return None

    bucket = SimpleNamespace(blob=lambda _path: Blob())
    vertex._get_storage_client = lambda: SimpleNamespace(
        bucket=lambda _name: bucket
    )

    with pytest.raises(TimeoutError, match="response lost"):
        vertex.submit([request])
    assert vertex_batches.calls == 1


def test_cancel_batch_retains_state_until_remote_cancel_succeeds(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    config = {
        "title": "Book",
        "translation": {
            "models": [
                {
                    "provider": "gemini",
                    "model": "batch-model",
                    "mode": "batch",
                }
            ]
        },
    }
    monkeypatch.setattr(cancel_batch, "load_config", lambda _path: config)

    state_path = (
        tmp_path
        / "output"
        / "Book"
        / "translated"
        / "batch_states"
        / "batch_example.json"
    )
    MegaUnitState(
        job_name="batches/1",
        provider="gemini",
        model="batch-model",
    ).save(state_path)

    class Client:
        def __init__(self, should_fail):
            self.should_fail = should_fail

        def cancel(self, _job_name):
            if self.should_fail:
                raise RuntimeError("provider unavailable")
            return True

    import pdf2epub.utils.batch_utils

    monkeypatch.setattr(
        pdf2epub.utils.batch_utils,
        "create_batch_client_from_config",
        lambda *_args, **_kwargs: Client(should_fail=True),
    )
    args = SimpleNamespace(config="config.yaml", all=False)

    assert cancel_batch.run(args) == 1
    assert state_path.exists()

    monkeypatch.setattr(
        pdf2epub.utils.batch_utils,
        "create_batch_client_from_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            cancel=lambda _job_name: False
        ),
    )

    assert cancel_batch.run(args) == 1
    assert state_path.exists()

    monkeypatch.setattr(
        pdf2epub.utils.batch_utils,
        "create_batch_client_from_config",
        lambda *_args, **_kwargs: Client(should_fail=False),
    )

    assert cancel_batch.run(args) == 0
    assert not state_path.exists()


def test_vertex_batch_requires_process_level_json_credentials(
    monkeypatch,
):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(ValueError, match="No default ADC"):
        probe_explicit_vertex_credentials()


def test_vertex_batch_refreshes_the_explicit_json_only(
    monkeypatch,
    tmp_path: Path,
):
    import google.auth

    credential_file = tmp_path / "vertex.json"
    credential_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(credential_file),
    )

    refreshed = []

    class Credentials:
        def refresh(self, request):
            refreshed.append(request)

    def load_credentials(filename, scopes):
        assert filename == str(credential_file)
        assert scopes == [
            "https://www.googleapis.com/auth/cloud-platform"
        ]
        return Credentials(), "ignored-project"

    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        load_credentials,
    )

    assert probe_explicit_vertex_credentials() == credential_file
    assert len(refreshed) == 1


class FakeSingleBatchClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.submissions = 0
        self.waits = 0
        self.fetches = 0
        self.cleanups = 0
        self.mapping = None

    def submit(self, requests, display_name=None):
        self.submissions += 1
        assert len(requests) == 1
        assert display_name
        return "jobs/1"

    def restore_job_mapping(self, job_name, keys, fingerprints=None):
        self.mapping = (job_name, keys, fingerprints)

    def wait_for_completion(self, job_name, poll_interval=None):
        self.waits += 1
        return SimpleNamespace(
            state=BatchJobState.SUCCEEDED,
            error=None,
        )

    def get_results(self, job_name, cleanup=True):
        self.fetches += 1
        assert cleanup is False
        return [
            BatchResponse(
                key="toc_translation",
                text=self.response_text,
            )
        ]

    def cleanup_job_artifacts(self, job_name):
        self.cleanups += 1


def _single_request_runner(tmp_path: Path, client: FakeSingleBatchClient):
    return PersistedSingleRequestBatch(
        client=client,
        provider="vertex",
        model="model-a",
        state_path=tmp_path / "toc_batch_state.json",
        poll_interval=0,
    )


def _toc_request(prompt: str = "translate"):
    return BatchRequest(
        key="toc_translation",
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )


def test_persisted_single_batch_caches_invalid_response_for_revalidation(
    tmp_path: Path,
):
    client = FakeSingleBatchClient('{"translated":"Title"}')
    runner = _single_request_runner(tmp_path, client)

    with pytest.raises(ValueError, match="Raw response retained"):
        runner.run(
            _toc_request(),
            lambda _text: (False, "missing array"),
            display_name="toc",
        )

    state_path = tmp_path / "toc_batch_state.json"
    response_path = tmp_path / "toc_batch_state.response.txt"
    assert state_path.exists()
    assert response_path.exists()
    assert client.submissions == 1
    assert client.fetches == 1
    assert client.cleanups == 0

    result = runner.run(
        _toc_request(),
        lambda _text: (True, ""),
        display_name="toc",
    )

    assert result == '{"translated":"Title"}'
    assert client.submissions == 1
    assert client.waits == 1
    assert client.fetches == 1
    assert client.cleanups == 0
    assert state_path.exists()
    assert response_path.exists()
    assert json.loads(state_path.read_text())["job_state"] == "VALIDATED"

    runner.finalize()

    assert client.cleanups == 1
    assert not state_path.exists()
    assert not response_path.exists()


def test_persisted_single_batch_rejects_different_input(tmp_path: Path):
    client = FakeSingleBatchClient("invalid")
    runner = _single_request_runner(tmp_path, client)

    with pytest.raises(ValueError):
        runner.run(
            _toc_request("first"),
            lambda _text: (False, "invalid"),
            display_name="toc",
        )

    with pytest.raises(PersistedBatchConflictError, match="different"):
        runner.run(
            _toc_request("second"),
            lambda _text: (True, ""),
            display_name="toc",
        )

    assert client.submissions == 1


def test_persisted_single_batch_recovers_finalization_tombstone(
    tmp_path: Path,
):
    class InterruptingCleanupClient(FakeSingleBatchClient):
        def cleanup_job_artifacts(self, job_name):
            self.cleanups += 1
            if self.cleanups == 1:
                raise RuntimeError("cleanup interrupted")

    client = InterruptingCleanupClient('{"translated":"Title"}')
    runner = _single_request_runner(tmp_path, client)

    runner.run(
        _toc_request(),
        lambda _text: (True, ""),
        display_name="toc",
    )

    state_path = tmp_path / "toc_batch_state.json"
    response_path = tmp_path / "toc_batch_state.response.txt"
    assert state_path.exists()
    assert response_path.exists()
    assert json.loads(state_path.read_text())["job_state"] == "VALIDATED"

    with pytest.raises(RuntimeError, match="cleanup interrupted"):
        runner.finalize()

    assert json.loads(state_path.read_text())["job_state"] == "FINALIZING"

    runner.finalize()
    assert client.submissions == 1
    assert client.fetches == 1
    assert client.cleanups == 2
    assert not state_path.exists()
    assert not response_path.exists()


def test_toc_translation_respects_batch_first_model_chain(
    monkeypatch,
    tmp_path: Path,
):
    expected = [{"id": "book_title", "translated": "书名"}]

    def fake_batch_translate(**kwargs):
        assert kwargs["batch_models"][0]["mode"] == "batch"
        return expected

    class OnlineClient:
        def generate_with_validation(self, **_kwargs):
            raise AssertionError("batch-first TOC must not use online fallback")

    monkeypatch.setattr(
        translate_v2,
        "_translate_toc_with_batch",
        fake_batch_translate,
    )

    result = translate_v2._translate_toc_batch(
        [
            {
                "id": "book_title",
                "level": 0,
                "original": "Book",
                "reference": "",
            }
        ],
        OnlineClient(),
        [
            {
                "provider": "vertex",
                "model": "batch-model",
                "mode": "batch",
            },
            {
                "provider": "gemini",
                "model": "online-fallback",
                "mode": "online",
            },
        ],
        "English",
        "Chinese",
        config={},
        output_dir=tmp_path,
    )

    assert result == expected


def test_toc_batch_state_is_not_finalized_when_output_save_fails(
    monkeypatch,
    tmp_path: Path,
):
    finalized = []
    monkeypatch.setattr(
        translate_v2,
        "_build_toc_reference_json",
        lambda *_args: [{"id": "book_title"}],
    )
    monkeypatch.setattr(
        translate_v2,
        "_translate_toc_batch",
        lambda *_args, **_kwargs: [
            {"id": "book_title", "translated": "书名"}
        ],
    )
    monkeypatch.setattr(
        translate_v2,
        "_save_toc_tree_translated",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        translate_v2,
        "_finalize_toc_batch_state",
        lambda *_args, **_kwargs: finalized.append(True) or True,
    )

    assert not translate_v2._translate_toc_locked(
        tmp_path,
        tmp_path,
        object(),
        [{"provider": "vertex", "model": "model", "mode": "batch"}],
        "English",
        "Chinese",
        {},
    )
    assert finalized == []


def test_toc_finalized_tombstone_finishes_without_resubmission(
    monkeypatch,
    tmp_path: Path,
):
    state_path = tmp_path / "toc_translation_batch_state.json"
    response_path = state_path.with_suffix(".response.txt")
    reference = [{"id": "book_title"}]
    request = BatchRequest(
        key="toc_translation",
        contents=[{
            "role": "user",
            "parts": [{
                "text": translate_v2._build_toc_translation_prompt(
                    reference,
                    "English",
                    "Chinese",
                )
            }],
        }],
        config={"response_mime_type": "application/json"},
    )
    input_sha256 = PersistedSingleRequestBatch._fingerprint(
        "vertex",
        "model",
        request,
    )
    state_path.write_text(
        json.dumps(
            {
                "provider": "vertex",
                "model": "model",
                "request_key": "toc_translation",
                "input_sha256": input_sha256,
                "job_name": "jobs/1",
                "job_state": "FINALIZED",
            }
        ),
        encoding="utf-8",
    )
    response_path.write_text("validated", encoding="utf-8")
    (tmp_path / "toc_tree_translated.json").write_text(
        "{}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        translate_v2,
        "_build_toc_reference_json",
        lambda *_args: reference,
    )
    monkeypatch.setattr(
        translate_v2,
        "_translate_toc_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "finalized tombstone must not resubmit"
        ),
    )

    assert translate_v2._translate_toc_locked(
        tmp_path,
        tmp_path,
        object(),
        [{"provider": "vertex", "model": "model", "mode": "batch"}],
        "English",
        "Chinese",
        {},
    )
    assert not state_path.exists()
    assert not response_path.exists()


def test_malformed_toc_finalized_state_is_retained(
    monkeypatch,
    tmp_path: Path,
):
    state_path = tmp_path / "toc_translation_batch_state.json"
    response_path = state_path.with_suffix(".response.txt")
    state_path.write_text(
        json.dumps({"job_state": "FINALIZED"}),
        encoding="utf-8",
    )
    response_path.write_text("validated", encoding="utf-8")
    (tmp_path / "toc_tree_translated.json").write_text(
        "{}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        translate_v2,
        "_build_toc_reference_json",
        lambda *_args: [{"id": "book_title"}],
    )

    assert not translate_v2._translate_toc_locked(
        tmp_path,
        tmp_path,
        object(),
        [{"provider": "vertex", "model": "model", "mode": "batch"}],
        "English",
        "Chinese",
        {},
    )
    assert state_path.exists()
    assert response_path.exists()


def test_phase_resume_passes_all_units_to_pipeline(tmp_path: Path):
    units = [
        WorkUnit(
            id="chapter_1",
            file_key="chapter_1",
            content="one",
        ),
        WorkUnit(
            id="chapter_2",
            file_key="chapter_2",
            content="two",
        ),
    ]

    class Loader:
        def load_units(self, _input_dir, _pattern):
            return units

    class Pipeline:
        def __init__(self):
            self.received = None

        def process_all(self, received, resume=False):
            self.received = (received, resume)
            return SimpleNamespace(
                total=2,
                completed=2,
                failed=0,
                failed_keys=[],
            )

    pipeline = Pipeline()
    phase = Phase(
        name="translate",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        pipeline=pipeline,
        loader=Loader(),
    )
    validated = tmp_path / "output" / "validated"
    validated.mkdir(parents=True)
    (validated / "chapter_1.md").write_text(
        "already promoted",
        encoding="utf-8",
    )

    result = phase.run(resume=True)

    assert pipeline.received == (units, True)
    assert result.total == 2
