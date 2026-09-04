from types import SimpleNamespace

import pytest

from pdf2epub.core.executor import ChainEntry, Executor, ProcessResult
from pdf2epub.core.executor.state import UnitState
from pdf2epub.core.hooks import ErrorEffect, ErrorType


class _TwoWaySplitter:
    def split(self, content: str, max_tokens: int) -> list[str]:
        return ["first half\n", "second half\n"]


@pytest.mark.parametrize(
    "error_type",
    [
        ErrorType.VALIDATION,
        ErrorType.TRUNCATION,
        ErrorType.PARSE_ERROR,
        ErrorType.UNKNOWN,
    ],
)
def test_content_failure_split_children_keep_the_exhausted_model_entry(error_type):
    entry = ChainEntry(
        provider="anthropic",
        model="k3-256k",
        mode="online",
        retries=1,
    )
    hooks = SimpleNamespace(
        classify_error=lambda error: (
            error_type,
            ErrorEffect(
                remove_current_model=True,
                quota_type=error_type,
            ),
        )
    )
    executor = Executor(
        llm_client=None,
        model_chain=[entry],
        processor=None,
        hooks=hooks,
        max_workers=1,
        splitter=_TwoWaySplitter(),
        split_max_tokens=100,
    )
    state = UnitState(
        chain=[entry],
        total_quota=2,
        quotas={error_type: 2},
        content="first half\nsecond half\n",
    )
    unit_states = {"chapter": state}
    pending: set[str] = set()

    requeued, split_depth = executor._handle_failure(
        unit_id="chapter",
        result=ProcessResult(
            success=False, error=Exception("repetition loop detected")
        ),
        state=state,
        unit_states=unit_states,
        pending=pending,
        completed=set(),
        failed=set(),
        safety_blocked=set(),
        validation_failed=set(),
        results={},
        fallback_used=set(),
    )

    assert requeued is True
    assert split_depth == 1
    assert unit_states["chapter.sub0"].chain == [entry]
    assert unit_states["chapter.sub1"].chain == [entry]
