"""
Test for .sub promotion bug - P0 issue.

Bug: Virtual .sub units created by dynamic splitting leak into:
1. ExecutionResult.results (Executor doesn't clean them after aggregation)
2. Pipeline promotes them to validated/ (using results.keys() directly)
3. Tracker records them as completed units
4. Statistics include inflated counts

Design contract (from design doc): .sub units are "virtual" and should NOT:
- Appear in ExecutionResult.results (only parent's aggregated result)
- Be promoted to validated/
- Be tracked as completed units
- Inflate completed count

.sub units MAY:
- Appear in raw/ (for debugging only)

These tests MUST FAIL to prove the bug exists.
Each test's expected value describes CORRECT behavior.
"""

import pytest
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from unittest.mock import MagicMock
from dataclasses import dataclass, field
import tempfile
import shutil

from pdf2epub.core.executor import (
    Executor,
    ChainEntry,
    ExecutionResult,
    QuotaConfig,
    UnitState,
    handle_split,
)
from pdf2epub.core.work_unit import WorkUnit, SplitType
from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier
from pdf2epub.core.types import ErrorType, is_sub_key, filter_sub_keys

from .fixtures.fake_llm import FakeLLMClient, FakeResponse, FakeErrorType


# ============================================================
# Test Utilities
# ============================================================

class SimpleProcessor:
    """Simple processor for testing."""
    name = "simple"

    def build_prompt(self, content: str, context=None) -> str:
        return f"Process: {content}"

    def clean_response(self, response: str) -> str:
        return response.strip()

    def post_process(self, result: str, context=None) -> str:
        return result


class FakeSplitter:
    """Fake splitter that returns predetermined chunks."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks

    def split(self, content: str, max_tokens: int) -> List[str]:
        return self.chunks


def make_unit_state(content: str = "") -> UnitState:
    """Create a minimal UnitState for testing."""
    return UnitState(
        chain=[ChainEntry(provider="fake", model="fake", mode="online")],
        total_quota=3,
        quotas={ErrorType.NETWORK: 2, ErrorType.VALIDATION: 2},
        content=content,
    )


# ============================================================
# Fake Persistence for testing Pipeline
# ============================================================

@dataclass
class FakePersistence:
    """Fake persistence that tracks what gets saved/promoted."""
    saved_raw: Dict[str, str] = field(default_factory=dict)
    saved_validated: Dict[str, str] = field(default_factory=dict)
    promoted_keys: Set[str] = field(default_factory=set)
    warning_keys: Set[str] = field(default_factory=set)

    def save_raw(self, key: str, content: str):
        self.saved_raw[key] = content

    def save_with_warning(self, key: str, content: str, warning: str = ""):
        self.saved_validated[key] = content
        self.warning_keys.add(key)

    def promote_batch(self, keys: List[str]):
        for key in keys:
            self.promoted_keys.add(key)
            if key in self.saved_raw:
                self.saved_validated[key] = self.saved_raw[key]


# ============================================================
# Fake Tracker for testing Pipeline
# ============================================================

@dataclass
class FakeTracker:
    """Fake tracker that records what gets marked complete."""
    completed_keys: Set[str] = field(default_factory=set)
    failed_keys: Set[str] = field(default_factory=set)
    attempts: Dict[str, List[Any]] = field(default_factory=dict)

    def is_unit_complete(self, key: str) -> bool:
        return key in self.completed_keys

    def record_attempt(self, key: str, attempt: Any):
        if key not in self.attempts:
            self.attempts[key] = []
        self.attempts[key].append(attempt)
        # Mark complete if status is "completed" or "completed_fallback"
        status = getattr(attempt, 'status', None)
        if status and 'completed' in status:
            self.completed_keys.add(key)
        elif status == 'failed':
            self.failed_keys.add(key)

    def record_validation(self, key: str, result: Dict):
        pass


# ============================================================
# Test: Executor.execute() results should NOT contain .sub keys
# ============================================================

class TestExecutorResultsMustNotContainSubKeys:
    """
    BUG: After dynamic splitting and aggregation, Executor leaves .sub
    children's results in ExecutionResult.results instead of cleaning them up.

    CORRECT BEHAVIOR (expected): Only parent's aggregated result should be
    in ExecutionResult.results. Children .sub should be removed after aggregation.

    CURRENT BEHAVIOR (bug): Children .sub results remain in results dict.

    These tests MUST FAIL to prove the bug exists.
    """

    @pytest.fixture
    def hooks_reject_first_then_accept(self):
        """
        Hooks that reject the parent once (triggering split),
        then accept all .sub children and aggregated parent.
        """
        parent_call_count = {"chapter_1": 0}

        def dynamic_validator(key, orig, processed):
            # .sub children always succeed
            if is_sub_key(key):
                return MagicMock(accepted=True, context_ready=False)
            # Parent: reject first time to trigger split, accept after
            parent_call_count[key] = parent_call_count.get(key, 0) + 1
            if parent_call_count[key] == 1:
                return MagicMock(accepted=False, context_ready=False)
            return MagicMock(accepted=True, context_ready=False)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "split_trigger"
        validator.validate = dynamic_validator
        hooks._validators = [validator]
        return hooks

    @pytest.fixture
    def hooks_always_accept(self):
        """Hooks that always accept (for .sub children to succeed)."""
        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "accepting"
        validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
        hooks._validators = [validator]
        return hooks

    def test_executor_results_must_not_contain_sub_keys_after_split(
        self, hooks_reject_first_then_accept
    ):
        """
        CRITICAL BUG TEST: After split and aggregation, .sub keys should NOT
        be in ExecutionResult.results.

        This test MUST FAIL with current buggy code.

        Expected (correct): results.keys() == {"chapter_1"}
        Actual (bug): results.keys() == {"chapter_1", "chapter_1.sub0", "chapter_1.sub1"}
        """
        fake_llm = FakeLLMClient(default_response="processed content")

        # Splitter that creates 2 children
        splitter = FakeSplitter(["Part 1 content", "Part 2 content"])

        executor = Executor(
            llm_client=fake_llm,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            processor=SimpleProcessor(),
            hooks=hooks_reject_first_then_accept,
            splitter=splitter,
            max_workers=1,
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
        )

        # Content must have newlines for split to work
        unit = WorkUnit(
            id="chapter_1",
            file_key="ch1",
            content="Line 1\nLine 2\nLine 3\nLine 4"
        )
        result = executor.execute([unit])

        # Verify split occurred (parent should be completed via aggregation)
        assert "chapter_1" in result.completed, (
            "Parent should complete via aggregation after children complete"
        )

        # THE BUG: .sub keys should NOT be in results
        sub_keys_in_results = [k for k in result.results.keys() if is_sub_key(k)]

        # This assertion MUST FAIL to prove the bug exists
        assert len(sub_keys_in_results) == 0, (
            f"BUG EXPOSED: .sub keys found in ExecutionResult.results: {sub_keys_in_results}. "
            f"Design doc says .sub units should NOT appear in results after aggregation. "
            f"Only parent's aggregated result should be present."
        )

    def test_executor_completed_must_not_contain_sub_keys(
        self, hooks_reject_first_then_accept
    ):
        """
        CRITICAL BUG TEST: .sub keys should NOT be in ExecutionResult.completed.

        This test MUST FAIL with current buggy code.
        """
        fake_llm = FakeLLMClient(default_response="processed content")
        splitter = FakeSplitter(["Part 1 content", "Part 2 content"])

        executor = Executor(
            llm_client=fake_llm,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            processor=SimpleProcessor(),
            hooks=hooks_reject_first_then_accept,
            splitter=splitter,
            max_workers=1,
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
        )

        unit = WorkUnit(
            id="chapter_1",
            file_key="ch1",
            content="Line 1\nLine 2\nLine 3\nLine 4"
        )
        result = executor.execute([unit])

        # THE BUG: .sub keys should NOT be in completed
        sub_keys_in_completed = [k for k in result.completed if is_sub_key(k)]

        # This assertion MUST FAIL to prove the bug exists
        assert len(sub_keys_in_completed) == 0, (
            f"BUG EXPOSED: .sub keys found in ExecutionResult.completed: {sub_keys_in_completed}. "
            f"Virtual .sub units should not be counted as completed units."
        )

    def test_executor_stats_must_not_count_sub_units(
        self, hooks_reject_first_then_accept
    ):
        """
        CRITICAL BUG TEST: Statistics should not count .sub units.

        If we have 1 unit that splits into 2 children:
        - completed count should be 1 (only parent)
        - NOT 3 (parent + 2 children)

        This test MUST FAIL with current buggy code.
        """
        fake_llm = FakeLLMClient(default_response="processed content")
        splitter = FakeSplitter(["Part 1", "Part 2"])

        executor = Executor(
            llm_client=fake_llm,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            processor=SimpleProcessor(),
            hooks=hooks_reject_first_then_accept,
            splitter=splitter,
            max_workers=1,
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
        )

        unit = WorkUnit(
            id="chapter_1",
            file_key="ch1",
            content="Line 1\nLine 2\nLine 3"
        )
        result = executor.execute([unit])

        # Filter out .sub to get correct count
        real_completed = filter_sub_keys(result.completed)

        # This assertion MUST FAIL if .sub keys are in completed
        assert len(result.completed) == len(real_completed), (
            f"BUG EXPOSED: completed count inflated by .sub units. "
            f"len(completed) = {len(result.completed)}, but should be {len(real_completed)}. "
            f".sub keys in completed: {[k for k in result.completed if is_sub_key(k)]}"
        )


# ============================================================
# Test: Pipeline should NOT promote .sub keys to validated/
# ============================================================

class TestPipelineMustNotPromoteSubKeys:
    """
    BUG: Pipeline uses exec_result.results.keys() directly to promote,
    which includes .sub keys that Executor left in results.

    CORRECT BEHAVIOR: Pipeline should filter out .sub keys before promoting.

    These tests MUST FAIL to prove the bug exists.
    """

    def test_pipeline_must_not_promote_sub_to_validated(self):
        """
        CRITICAL BUG TEST: .sub keys must NOT be promoted to validated/.

        This test simulates what Pipeline does:
        1. Get results from Executor
        2. Calculate successful = results.keys() - failed
        3. Promote successful to validated/

        The bug is that .sub keys in results get promoted.

        This test MUST FAIL to prove the bug exists.
        """
        from pdf2epub.core.pipeline_v2 import ProcessingPipelineV2

        # Create fake components
        fake_llm = FakeLLMClient(default_response="processed")
        fake_persistence = FakePersistence()
        fake_tracker = FakeTracker()

        # Hooks that reject parent once to trigger split
        parent_called = {"count": 0}

        def split_trigger_validator(key, orig, processed):
            if is_sub_key(key):
                return MagicMock(accepted=True, context_ready=False)
            parent_called["count"] += 1
            if parent_called["count"] == 1:
                return MagicMock(accepted=False, context_ready=False)
            return MagicMock(accepted=True, context_ready=False)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "split_trigger"
        validator.validate = split_trigger_validator
        hooks._validators = [validator]

        splitter = FakeSplitter(["Part 1", "Part 2"])

        pipeline = ProcessingPipelineV2(
            processor=SimpleProcessor(),
            llm_client=fake_llm,
            persistence=fake_persistence,
            tracker=fake_tracker,
            hooks=hooks,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
            content_splitter=splitter,
            max_workers=1,
        )

        unit = WorkUnit(
            id="chapter_1",
            file_key="ch1",
            content="Line 1\nLine 2\nLine 3"
        )

        result = pipeline.process_all([unit])

        # THE BUG: Check what got promoted
        sub_keys_promoted = [k for k in fake_persistence.promoted_keys if is_sub_key(k)]

        # This assertion MUST FAIL to prove the bug exists
        assert len(sub_keys_promoted) == 0, (
            f"BUG EXPOSED: .sub keys promoted to validated/: {sub_keys_promoted}. "
            f"Design doc says .sub units should NOT appear in output directory."
        )

    def test_pipeline_tracker_must_not_mark_sub_as_completed(self):
        """
        CRITICAL BUG TEST: Tracker should NOT record .sub as completed.

        This test MUST FAIL to prove the bug exists.
        """
        from pdf2epub.core.pipeline_v2 import ProcessingPipelineV2

        fake_llm = FakeLLMClient(default_response="processed")
        fake_persistence = FakePersistence()
        fake_tracker = FakeTracker()

        parent_called = {"count": 0}

        def split_trigger_validator(key, orig, processed):
            if is_sub_key(key):
                return MagicMock(accepted=True, context_ready=False)
            parent_called["count"] += 1
            if parent_called["count"] == 1:
                return MagicMock(accepted=False, context_ready=False)
            return MagicMock(accepted=True, context_ready=False)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "split_trigger"
        validator.validate = split_trigger_validator
        hooks._validators = [validator]

        splitter = FakeSplitter(["Part 1", "Part 2"])

        pipeline = ProcessingPipelineV2(
            processor=SimpleProcessor(),
            llm_client=fake_llm,
            persistence=fake_persistence,
            tracker=fake_tracker,
            hooks=hooks,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
            content_splitter=splitter,
            max_workers=1,
        )

        unit = WorkUnit(
            id="chapter_1",
            file_key="ch1",
            content="Line 1\nLine 2\nLine 3"
        )

        result = pipeline.process_all([unit])

        # THE BUG: Check what got marked complete in tracker
        sub_keys_tracked = [k for k in fake_tracker.completed_keys if is_sub_key(k)]

        # This assertion MUST FAIL to prove the bug exists
        assert len(sub_keys_tracked) == 0, (
            f"BUG EXPOSED: .sub keys recorded as completed in tracker: {sub_keys_tracked}. "
            f"Virtual units should not be tracked."
        )


# ============================================================
# Test: ProcessingResultV2.completed count must exclude .sub
# ============================================================

class TestProcessingResultMustExcludeSubFromStats:
    """
    BUG: ProcessingResultV2.completed includes .sub units in count.

    CORRECT: completed count should only count real units.
    """

    def test_pipeline_result_completed_count_must_exclude_sub(self):
        """
        CRITICAL BUG TEST: Pipeline result's completed count must not include .sub.

        If we process 1 unit that splits into 2, completed should be 1, not 3.

        This test MUST FAIL to prove the bug exists.
        """
        from pdf2epub.core.pipeline_v2 import ProcessingPipelineV2

        fake_llm = FakeLLMClient(default_response="processed")
        fake_persistence = FakePersistence()
        fake_tracker = FakeTracker()

        parent_called = {"count": 0}

        def split_trigger_validator(key, orig, processed):
            if is_sub_key(key):
                return MagicMock(accepted=True, context_ready=False)
            parent_called["count"] += 1
            if parent_called["count"] == 1:
                return MagicMock(accepted=False, context_ready=False)
            return MagicMock(accepted=True, context_ready=False)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "split_trigger"
        validator.validate = split_trigger_validator
        hooks._validators = [validator]

        splitter = FakeSplitter(["Part 1", "Part 2"])

        pipeline = ProcessingPipelineV2(
            processor=SimpleProcessor(),
            llm_client=fake_llm,
            persistence=fake_persistence,
            tracker=fake_tracker,
            hooks=hooks,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
            content_splitter=splitter,
            max_workers=1,
        )

        units = [
            WorkUnit(id="chapter_1", file_key="ch1", content="Line 1\nLine 2\nLine 3"),
        ]

        result = pipeline.process_all(units)

        # THE BUG: completed count should be 1, not 3
        # (1 input unit, even if it split into 2 children internally)

        # Count how many .sub keys are in the results dict
        sub_in_results = len([k for k in result.results if is_sub_key(k)])

        # This assertion MUST FAIL if .sub keys inflate the count
        assert result.completed == 1, (
            f"BUG EXPOSED: completed count = {result.completed}, should be 1. "
            f".sub keys in results.keys(): {[k for k in result.results if is_sub_key(k)]}. "
            f"Statistics are inflated by virtual .sub units."
        )


# ============================================================
# Supporting test: Verify split actually happens (sanity check)
# ============================================================

class TestSplitActuallyHappens:
    """Sanity check: verify our test setup actually triggers split."""

    def test_split_is_triggered_by_validation_failure(self):
        """Verify that our test setup triggers a split."""
        # This is just a sanity check that split happens
        parent_called = {"count": 0}

        def split_trigger_validator(key, orig, processed):
            if is_sub_key(key):
                return MagicMock(accepted=True, context_ready=False)
            parent_called["count"] += 1
            if parent_called["count"] == 1:
                return MagicMock(accepted=False, context_ready=False)
            return MagicMock(accepted=True, context_ready=False)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "split_trigger"
        validator.validate = split_trigger_validator
        hooks._validators = [validator]

        fake_llm = FakeLLMClient(default_response="processed")
        splitter = FakeSplitter(["Part 1", "Part 2"])

        executor = Executor(
            llm_client=fake_llm,
            model_chain=[ChainEntry(provider="fake", model="fake", mode="online")],
            processor=SimpleProcessor(),
            hooks=hooks,
            splitter=splitter,
            max_workers=1,
            quota_config=QuotaConfig(total=2, per_type={ErrorType.VALIDATION: 1}),
        )

        unit = WorkUnit(
            id="chapter_1",
            file_key="ch1",
            content="Line 1\nLine 2\nLine 3"
        )
        result = executor.execute([unit])

        # Verify split actually occurred
        assert result.splits_performed > 0, (
            "Test setup error: split was not triggered. "
            "Check that validation failure + splitter are configured correctly."
        )

        # Verify parent completed
        assert "chapter_1" in result.completed, (
            "Test setup error: parent did not complete. "
            "Children may have failed or aggregation did not happen."
        )


# ============================================================
# Test: is_sub_key and filter_sub_keys work correctly
# ============================================================

class TestSubKeyHelpers:
    """Verify is_sub_key and filter_sub_keys work as expected."""

    def test_is_sub_key_simple(self):
        assert is_sub_key("chapter_1.sub0") is True
        assert is_sub_key("chapter_1.sub99") is True

    def test_is_sub_key_nested(self):
        assert is_sub_key("chapter_1.sub0.sub1") is True
        assert is_sub_key("a.sub0.sub0.sub0") is True

    def test_is_sub_key_rejects_non_sub(self):
        assert is_sub_key("chapter_1") is False
        assert is_sub_key("chapter_1.part1") is False

    def test_is_sub_key_rejects_sub_without_number(self):
        assert is_sub_key("chapter_1.sub") is False
        assert is_sub_key("chapter_1.submarine") is False

    def test_filter_sub_keys(self):
        keys = {"chapter_1", "chapter_1.sub0", "chapter_1.sub1", "chapter_2"}
        filtered = filter_sub_keys(keys)
        assert filtered == {"chapter_1", "chapter_2"}
