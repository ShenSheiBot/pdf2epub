from contextlib import contextmanager
from types import SimpleNamespace

from pdf2epub.core._protocol import ValidationResult
from pdf2epub.core.executor import ChainEntry, ExecutionResult, Executor, WorkUnit
from pdf2epub.core.executor._protocol import ProcessResult
from pdf2epub.core.hooks import DefaultErrorClassifier, ErrorType
from pdf2epub.core.hooks.validators import TruncationValidator
from pdf2epub.core.pipeline_v2 import BatchValidationFailure, ProcessingPipelineV2


def test_screener_failure_remains_non_blocking():
    validator = TruncationValidator(role="screener", context_ready=True)
    validator._detector = SimpleNamespace(
        detect=lambda _original, _result: (True, "content was truncated", {})
    )

    result = validator.validate("chapter", "source", "truncated")

    assert result.accepted is True
    assert result.context_ready is False


def test_batch_validation_preserves_typed_failure():
    class FakeBatchValidator:
        name = "FakeBatchValidator"

        def validate_batch(self, files):
            key = next(iter(files))
            return {
                key: ValidationResult(
                    key=key,
                    is_valid=False,
                    reason="output was truncated at the final index entry",
                    error_type=ErrorType.TRUNCATION,
                )
            }

    recorded = []
    pipeline = object.__new__(ProcessingPipelineV2)
    pipeline._batch_validators = [FakeBatchValidator()]
    pipeline._book_structure = None
    pipeline._tracker = SimpleNamespace(
        record_validation=lambda key, record: recorded.append((key, record))
    )

    failures = pipeline._run_batch_validation(
        {"chapter_9": "processed"},
        {"chapter_9": "original"},
    )

    assert failures["chapter_9"].error_type is ErrorType.TRUNCATION
    assert recorded[0][1]["is_valid"] is False

    failure = BatchValidationFailure(failures["chapter_9"])
    classifier = DefaultErrorClassifier()
    assert classifier.classify(failure) is ErrorType.TRUNCATION

    ordinary_failure = BatchValidationFailure(
        ValidationResult(
            key="chapter_9",
            is_valid=False,
            reason="output format was invalid",
            error_type=ErrorType.VALIDATION,
        )
    )
    assert classifier.classify(ordinary_failure) is ErrorType.VALIDATION


def test_batch_validation_returns_empty_failures_when_all_keys_are_skipped():
    pipeline = object.__new__(ProcessingPipelineV2)
    pipeline._batch_validators = [object()]
    pipeline._book_structure = None
    pipeline._tracker = SimpleNamespace(record_validation=lambda *_args: None)

    assert pipeline._run_batch_validation(
        {"chapter_1": "processed"},
        {"chapter_1": "original"},
        screener_passed={"chapter_1"},
    ) == {}


def test_executor_splits_typed_batch_failure_and_reuses_model_for_children():
    class Splitter:
        def split(self, content, _max_tokens):
            return ["first\n" + content, "second\n" + content]

    classifier = DefaultErrorClassifier()

    class Hooks:
        def classify_error(self, error):
            error_type = classifier.classify(error)
            return error_type, classifier.get_effect(error_type)

    executor = Executor(
        llm_client=object(),
        model_chain=[ChainEntry(provider="vertex", model="flash", mode="batch")],
        processor=object(),
        hooks=Hooks(),
        splitter=Splitter(),
        batch_client=object(),
    )

    def process_child(unit, _state, _context, _original):
        return ProcessResult(success=True, content=f"processed:{unit.id}")

    executor._process_single = process_child

    original = WorkUnit(id="chapter_9", file_key="chapter_9", content="index\n")
    failure = BatchValidationFailure(
        ValidationResult(
            key="chapter_9",
            is_valid=False,
            reason="output was truncated",
            error_type=ErrorType.TRUNCATION,
        )
    )

    result = executor.execute([original], initial_failures={"chapter_9": failure})

    assert result.failed == set()
    assert result.splits_performed == 1
    assert result.results["chapter_9"] == (
        "processed:chapter_9.sub0\n\nprocessed:chapter_9.sub1"
    )


def test_pipeline_passes_batch_failure_into_executor():
    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, units, context_base=None, resume_batch=False, initial_failures=None):
            self.calls.append(initial_failures)
            content = "retried" if initial_failures else "first result"
            return ExecutionResult(
                results={units[0].id: content},
                completed={units[0].id},
            )

    executor = FakeExecutor()
    pipeline = object.__new__(ProcessingPipelineV2)
    pipeline._split_manager = None
    pipeline._batch_validators = [object()]
    pipeline._book_structure = None
    pipeline._executor = executor
    pipeline._promoter = SimpleNamespace(
        promote=lambda _key: True,
        promote_batch=lambda _keys: None,
    )
    pipeline._tracker = SimpleNamespace(record_attempt=lambda *_args: None)
    pipeline._proactive_split = lambda units: units
    pipeline._get_pending_keys = lambda keys: set(keys)
    pipeline._run_batch_validation = lambda *_args, **_kwargs: {
        "chapter_9": ValidationResult(
            key="chapter_9",
            is_valid=False,
            reason="output was truncated",
            error_type=ErrorType.TRUNCATION,
        )
    }

    result = pipeline.process_all(
        [WorkUnit(id="chapter_9", file_key="chapter_9", content="original")]
    )

    assert result.failed == 0
    assert result.results["chapter_9"] == "retried"
    assert executor.calls[0] is None
    assert isinstance(executor.calls[1]["chapter_9"], BatchValidationFailure)
    assert executor.calls[1]["chapter_9"].error_type is ErrorType.TRUNCATION


def test_pipeline_finalizes_batch_only_after_validation_and_promotion():
    events = []

    class FakeExecutor:
        @contextmanager
        def batch_run_lock(self):
            events.append("lock-enter")
            try:
                yield
            finally:
                events.append("lock-exit")

        def recover_finalizing_batches(self):
            events.append("recover")

        def execute(self, units, _context_base=None, **_kwargs):
            events.append("execute")
            return ExecutionResult(
                results={units[0].id: "translated"},
                completed={units[0].id},
                batch_jobs=[[units[0].id]],
            )

        def finalize_batch_jobs(self, jobs):
            assert jobs == [["chapter_1"]]
            events.append("finalize")

    pipeline = object.__new__(ProcessingPipelineV2)
    pipeline._split_manager = None
    pipeline._batch_validators = [object()]
    pipeline._book_structure = None
    pipeline._executor = FakeExecutor()
    pipeline._promoter = SimpleNamespace(
        promote=lambda key: events.append(f"promote:{key}") or True,
    )
    pipeline._tracker = SimpleNamespace(record_attempt=lambda *_args: None)
    pipeline._proactive_split = lambda units: units
    pipeline._get_pending_keys = lambda keys: set(keys)
    pipeline._run_batch_validation = (
        lambda *_args, **_kwargs: events.append("validate") or {}
    )

    result = pipeline.process_all(
        [
            WorkUnit(
                id="chapter_1",
                file_key="chapter_1",
                content="source",
            )
        ]
    )

    assert result.failed == 0
    assert events == [
        "lock-enter",
        "recover",
        "execute",
        "validate",
        "promote:chapter_1",
        "finalize",
        "lock-exit",
    ]


def test_pipeline_resume_restores_full_persisted_batch_membership():
    calls = []

    class FakeExecutor:
        def recover_finalizing_batches(self):
            return None

        def get_resumable_unit_ids(self):
            return {"chapter_1", "chapter_2"}

        def execute(self, units, _context_base=None, **_kwargs):
            calls.append([unit.id for unit in units])
            return ExecutionResult(
                results={unit.id: "translated" for unit in units},
                completed={unit.id for unit in units},
            )

    pipeline = object.__new__(ProcessingPipelineV2)
    pipeline._split_manager = None
    pipeline._batch_validators = []
    pipeline._book_structure = None
    pipeline._executor = FakeExecutor()
    pipeline._promoter = SimpleNamespace(promote=lambda _key: True)
    pipeline._tracker = SimpleNamespace(record_attempt=lambda *_args: None)
    pipeline._proactive_split = lambda units: units
    # Simulate a crash after chapter_1 was promoted but before the whole
    # persisted batch could be finalized.
    pipeline._get_pending_keys = lambda _keys: {"chapter_2"}

    result = pipeline.process_all(
        [
            WorkUnit(
                id="chapter_1",
                file_key="chapter_1",
                content="source 1",
            ),
            WorkUnit(
                id="chapter_2",
                file_key="chapter_2",
                content="source 2",
            ),
        ],
        resume=True,
    )

    assert result.failed == 0
    assert calls == [["chapter_1", "chapter_2"]]
