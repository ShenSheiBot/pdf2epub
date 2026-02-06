"""
Test: Tracker/Persistence State Inconsistency Bug (P0)

This test exposes a critical bug where tracker state and persistence state
can become inconsistent, leading to permanent data loss.

Bug Description:
1. Executor._process_single() calls tracker.record_attempt(status="completed") IMMEDIATELY
   after LLM response validation passes
2. BUT persistence.save_raw() is only called in Pipeline.process_all() AFTER execute() returns
3. If interrupted between these two points:
   - tracker shows unit as "completed"
   - BUT no file exists in raw/ or validated/
4. On resume, _get_pending_keys() checks tracker.is_unit_complete() which returns True
5. Unit is SKIPPED, resulting in PERMANENT DATA LOSS

Expected Behavior:
- A unit should NOT be considered "complete" unless its output file exists
- OR tracker.record_attempt() should only be called AFTER file is saved
- OR is_unit_complete() should verify file existence, not just tracker state

These tests FAIL to prove the bug exists. Tests should PASS once bug is fixed.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch
import time

from pdf2epub.core.pipeline_v2 import ProcessingPipelineV2
from pdf2epub.core.persistence import ResultPersistence
from pdf2epub.core.tracking import ProcessingTracker, AttemptRecord
from pdf2epub.core.executor import (
    Executor,
    ChainEntry,
    ExecutionResult,
    QuotaConfig,
)
from pdf2epub.core.work_unit import WorkUnit
from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier

from .fixtures.fake_llm import FakeLLMClient, FakeResponse, FakeErrorType
from .fixtures.sample_content import SHORT_CHAPTER, MEDIUM_CHAPTER


# ============================================================
# Test Processor (minimal implementation)
# ============================================================

class TestProcessor:
    """Minimal processor for testing."""
    name = "test_processor"

    def build_prompt(self, content: str, context: Any = None) -> str:
        return f"Process: {content}"

    def clean_response(self, response: str) -> str:
        return response.strip()

    def post_process(self, result: str, context: Any = None) -> str:
        return result

    def get_model_configs(self) -> List[Dict]:
        return [{"provider": "fake", "model": "fake-model"}]


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def accepting_hooks():
    """Hooks that always accept validation."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()

    validator = MagicMock()
    validator.name = "accepting_validator"
    validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
    hooks._validators = [validator]

    return hooks


@pytest.fixture
def model_chain():
    """Single model chain for testing."""
    return [ChainEntry(provider="fake", model="fake-model", mode="online")]


def create_pipeline(
    output_dir: Path,
    hooks: CompositeHooks,
    llm_client: Optional[FakeLLMClient] = None,
    model_chain: Optional[List[ChainEntry]] = None,
) -> ProcessingPipelineV2:
    """Create a pipeline with test components."""
    persistence = ResultPersistence(output_dir)

    # Create file_checker for atomicity guarantee
    def file_checker(key: str) -> bool:
        return persistence.has_raw(key) or persistence.has_validated(key)

    tracker = ProcessingTracker(
        output_dir / "progress.json",
        "TestProcessor",
        file_checker=file_checker,
    )

    if model_chain is None:
        model_chain = [ChainEntry(provider="fake", model="fake-model", mode="online")]

    if llm_client is None:
        llm_client = FakeLLMClient(default_response="processed content")

    return ProcessingPipelineV2(
        processor=TestProcessor(),
        llm_client=llm_client,
        persistence=persistence,
        tracker=tracker,
        hooks=hooks,
        model_chain=model_chain,
        max_workers=1,
    )


def create_units(contents: Dict[str, str]) -> List[WorkUnit]:
    """Create WorkUnit list from content dict."""
    return [
        WorkUnit(id=key, file_key=key, content=content)
        for key, content in contents.items()
    ]


# ============================================================
# Test 1: Simulated Interruption Scenario
# ============================================================

class TestInterruptionCausesInconsistentState:
    """
    Test that demonstrates tracker/persistence inconsistency after interruption.

    This simulates the scenario where:
    1. Executor completes processing and marks tracker as "completed"
    2. BUT before Pipeline can save the file, interruption occurs
    3. Result: tracker says completed, but no file exists
    """

    def test_tracker_shows_completed_but_file_missing_after_simulated_interruption(
        self, temp_output_dir, accepting_hooks, model_chain
    ):
        """
        After simulated interruption, tracker shows 'completed' but file doesn't exist.

        Expected: is_unit_complete() should return False because file doesn't exist,
        even though tracker internal state shows "completed".
        """
        persistence = ResultPersistence(temp_output_dir)

        # Create file_checker for atomicity guarantee
        def file_checker(key: str) -> bool:
            return persistence.has_raw(key) or persistence.has_validated(key)

        tracker = ProcessingTracker(
            temp_output_dir / "progress.json",
            "TestProcessor",
            file_checker=file_checker,
        )

        # Simulate: Executor recorded completion to tracker, but file never saved
        # This is what happens if interrupted between tracker.record_attempt()
        # and persistence.save_raw()
        unit_key = "chapter_1"

        # Manually record "completed" in tracker (simulating what Executor does)
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed",
            model="fake/fake-model",
        )
        tracker.record_attempt(unit_key, attempt)

        # Verify tracker internal state shows completed
        assert tracker.get_unit_status(unit_key) == "completed", "Tracker internal state should show completed"

        # Verify NO file was saved (simulating interruption before save)
        assert not persistence.has_raw(unit_key), "Raw file should NOT exist"
        assert not persistence.has_validated(unit_key), "Validated file should NOT exist"

        # FIXED BEHAVIOR: is_unit_complete() checks both tracker AND file existence
        # It should return False because file doesn't exist (atomicity guarantee)
        assert not tracker.is_unit_complete(unit_key), (
            "BUG EXPOSED: tracker.is_unit_complete() returns True "
            "even though the output file does not exist! "
            "This leads to data loss on resume."
        )


# ============================================================
# Test 2: Resume Skips Unit With Missing File
# ============================================================

class TestResumeSkipsMissingFileUnit:
    """
    Test that demonstrates data loss when resuming after interruption.

    Scenario:
    1. First run: Process unit, tracker marked completed, but file not saved (interrupted)
    2. Second run: _get_pending_keys() skips the unit because tracker shows completed
    3. Result: Unit is never processed, data permanently lost
    """

    def test_pending_keys_should_include_unit_with_missing_file(
        self, temp_output_dir, accepting_hooks, model_chain
    ):
        """
        _get_pending_keys() should include units that have no output file,
        even if tracker internal state shows them as completed.

        Expected: pending_keys includes "chapter_1" because file is missing
        """
        persistence = ResultPersistence(temp_output_dir)

        # Create file_checker for atomicity guarantee
        def file_checker(key: str) -> bool:
            return persistence.has_raw(key) or persistence.has_validated(key)

        tracker = ProcessingTracker(
            temp_output_dir / "progress.json",
            "TestProcessor",
            file_checker=file_checker,
        )

        fake_llm = FakeLLMClient(default_response="processed content")

        pipeline = ProcessingPipelineV2(
            processor=TestProcessor(),
            llm_client=fake_llm,
            persistence=persistence,
            tracker=tracker,
            hooks=accepting_hooks,
            model_chain=model_chain,
            max_workers=1,
        )

        # Simulate: First run was interrupted after tracker update but before file save
        unit_key = "chapter_1"
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed",
            model="fake/fake-model",
        )
        tracker.record_attempt(unit_key, attempt)

        # Confirm state: tracker internal status is completed but file missing
        assert tracker.get_unit_status(unit_key) == "completed", "Tracker internal status should be completed"
        assert not persistence.has_raw(unit_key), "But raw file missing"
        assert not persistence.has_validated(unit_key), "And validated file missing"
        # is_unit_complete should return False due to file_checker
        assert not tracker.is_unit_complete(unit_key), "is_unit_complete should return False (file missing)"

        # Call _get_pending_keys() as resume would
        all_keys = {unit_key}
        pending_keys = pipeline._get_pending_keys(all_keys)

        # THE BUG: pending_keys is EMPTY because tracker.is_unit_complete() returns True
        # Expected: pending_keys should contain "chapter_1" because file doesn't exist

        # This assertion describes the CORRECT behavior (should pass after fix)
        # Currently FAILS because the bug exists
        assert unit_key in pending_keys, (
            f"BUG EXPOSED: _get_pending_keys() returned {pending_keys}, "
            f"missing '{unit_key}' which has no output file! "
            "This causes permanent data loss on resume."
        )


# ============================================================
# Test 3: is_unit_complete Should Check File Existence
# ============================================================

class TestIsUnitCompleteShouldCheckFile:
    """
    Test that is_unit_complete() should verify file existence,
    not just tracker state.

    Design principle: "Completion" should mean the output is actually available,
    not just that the tracker recorded it.
    """

    def test_is_unit_complete_should_return_false_when_file_missing(
        self, temp_output_dir
    ):
        """
        is_unit_complete() should return False if output file doesn't exist,
        even if tracker has a "completed" record.

        This is the root cause of the data loss bug.
        """
        persistence = ResultPersistence(temp_output_dir)

        # Create file_checker for atomicity guarantee
        def file_checker(key: str) -> bool:
            return persistence.has_raw(key) or persistence.has_validated(key)

        tracker = ProcessingTracker(
            temp_output_dir / "progress.json",
            "TestProcessor",
            file_checker=file_checker,
        )

        unit_key = "chapter_1"

        # Record as completed in tracker
        attempt = AttemptRecord(
            timestamp=time.time(),
            status="completed",
            model="fake/fake-model",
        )
        tracker.record_attempt(unit_key, attempt)

        # Verify tracker internal state shows completed
        status = tracker.get_unit_status(unit_key)
        assert status == "completed", "Internal status should be 'completed'"

        # FIXED BEHAVIOR: is_unit_complete() checks both tracker AND file existence
        # It should return False because no file exists
        current_result = tracker.is_unit_complete(unit_key)

        expected_result = False  # Because file doesn't exist
        assert current_result == expected_result, (
            f"BUG EXPOSED: is_unit_complete() returned {current_result}, "
            f"expected {expected_result}. "
            "The tracker should verify file existence before claiming completion."
        )


# ============================================================
# Test 4: E2E - Full Pipeline Interruption Simulation
# ============================================================

class TestE2EPipelineInterruption:
    """
    End-to-end test that simulates the full bug scenario:
    1. Start processing
    2. Mock persistence.save_raw to NOT actually save (simulating interruption)
    3. Verify tracker shows completed but file doesn't exist
    4. Create new pipeline (simulating restart)
    5. Verify unit is NOT reprocessed (data loss)
    """

    def test_e2e_interruption_causes_data_loss(
        self, temp_output_dir, accepting_hooks, model_chain
    ):
        """
        Full E2E test demonstrating data loss from tracker/persistence inconsistency.
        """
        persistence = ResultPersistence(temp_output_dir)
        tracker = ProcessingTracker(temp_output_dir / "progress.json", "TestProcessor")

        fake_llm = FakeLLMClient(default_response="processed content")

        pipeline = ProcessingPipelineV2(
            processor=TestProcessor(),
            llm_client=fake_llm,
            persistence=persistence,
            tracker=tracker,
            hooks=accepting_hooks,
            model_chain=model_chain,
            max_workers=1,
        )

        unit = WorkUnit(id="chapter_1", file_key="chapter_1", content=SHORT_CHAPTER)

        # Mock save_raw to fail silently (simulating interruption after tracker update)
        original_save_raw = persistence.save_raw
        save_raw_called = []

        def mock_save_raw(key, content):
            save_raw_called.append(key)
            # DON'T actually save - simulating interruption
            return temp_output_dir / "raw" / f"{key}.md"  # Return path but don't create

        with patch.object(persistence, 'save_raw', mock_save_raw):
            # Process the unit
            result = pipeline.process_all([unit])

        # The executor thinks it completed successfully
        # (because validation passed and tracker was updated)

        # Verify save_raw was called (Pipeline tried to save)
        assert "chapter_1" in save_raw_called, "Pipeline should have tried to save"

        # BUT file doesn't exist (our mock didn't save)
        assert not persistence.has_raw("chapter_1"), "File should NOT exist (mock didn't save)"

        # Check tracker state - THIS IS THE BUG
        # Tracker shows completed because Executor called record_attempt()
        # before Pipeline could call save_raw()
        tracker_shows_completed = tracker.is_unit_complete("chapter_1")

        # Now simulate restart - create new pipeline
        fake_llm_2 = FakeLLMClient(default_response="reprocessed content")

        pipeline_2 = ProcessingPipelineV2(
            processor=TestProcessor(),
            llm_client=fake_llm_2,
            persistence=persistence,
            tracker=tracker,  # Same tracker - has the "completed" record
            hooks=accepting_hooks,
            model_chain=model_chain,
            max_workers=1,
        )

        # Process again (simulating --resume)
        result_2 = pipeline_2.process_all([unit])

        # THE BUG: Unit is NOT reprocessed because tracker shows completed
        # Expected: Unit SHOULD be reprocessed because file doesn't exist

        # Check if LLM was called on second run
        llm_was_called = fake_llm_2.was_called("chapter_1")

        # Expected: LLM should be called because file doesn't exist
        # Actual (BUG): LLM is NOT called because tracker shows completed

        # This assertion describes the CORRECT behavior (should pass after fix)
        # Currently FAILS because the bug exists
        assert llm_was_called, (
            "BUG EXPOSED: LLM was NOT called on resume even though file doesn't exist! "
            f"Tracker shows completed: {tracker_shows_completed}, "
            f"File exists: {persistence.has_raw('chapter_1')}. "
            "This is permanent data loss."
        )


# ============================================================
# Test 5: Verifying the Timing Issue
# ============================================================

class TestTrackerUpdateTiming:
    """
    Test that verifies WHERE the tracker update happens.

    The bug exists because:
    - Executor updates tracker in _process_single() BEFORE returning
    - Pipeline saves file in process_all() AFTER execute() returns

    This timing gap is the root cause.
    """

    def test_tracker_updated_before_persistence_save(
        self, temp_output_dir, accepting_hooks, model_chain
    ):
        """
        Verify that tracker is updated BEFORE persistence.save_raw() is called.
        This timing gap is the root cause of the bug.

        The bug: Executor calls tracker.record_attempt(completed) BEFORE
        Pipeline calls persistence.save_raw(). If interrupted between these,
        tracker shows completed but file doesn't exist.

        Expected (safe): save_raw should happen BEFORE first tracker completed record
        Actual (buggy): first tracker completed record happens BEFORE save_raw
        """
        persistence = ResultPersistence(temp_output_dir)
        tracker = ProcessingTracker(temp_output_dir / "progress.json", "TestProcessor")

        fake_llm = FakeLLMClient(default_response="processed content")

        pipeline = ProcessingPipelineV2(
            processor=TestProcessor(),
            llm_client=fake_llm,
            persistence=persistence,
            tracker=tracker,
            hooks=accepting_hooks,
            model_chain=model_chain,
            max_workers=1,
        )

        unit = WorkUnit(id="chapter_1", file_key="chapter_1", content=SHORT_CHAPTER)

        # Track the order of operations
        operation_log = []

        original_record_attempt = tracker.record_attempt
        original_save_raw = persistence.save_raw

        def logged_record_attempt(key, attempt):
            operation_log.append(("tracker.record_attempt", key, attempt.status))
            return original_record_attempt(key, attempt)

        def logged_save_raw(key, content):
            operation_log.append(("persistence.save_raw", key))
            return original_save_raw(key, content)

        with patch.object(tracker, 'record_attempt', logged_record_attempt):
            with patch.object(persistence, 'save_raw', logged_save_raw):
                pipeline.process_all([unit])

        # Find the order of operations for chapter_1
        # IMPORTANT: Find the FIRST completed record (from Executor, not Pipeline)
        first_tracker_completed_index = None
        save_index = None

        for i, (op, key, *args) in enumerate(operation_log):
            if key == "chapter_1":
                if op == "tracker.record_attempt" and args and args[0] == "completed":
                    if first_tracker_completed_index is None:  # FIRST occurrence only
                        first_tracker_completed_index = i
                elif op == "persistence.save_raw":
                    if save_index is None:  # FIRST occurrence only (FIX: was missing break)
                        save_index = i

        # The bug: tracker.record_attempt(completed) happens BEFORE persistence.save_raw()
        # This means if we crash between them, tracker shows completed but file doesn't exist

        assert first_tracker_completed_index is not None, "tracker.record_attempt should be called"
        assert save_index is not None, "persistence.save_raw should be called"

        # Operation log analysis:
        # Index 0: tracker.record_attempt(completed) - FROM EXECUTOR (the bug!)
        # Index 1: persistence.save_raw - FROM PIPELINE (after execute returns)
        # Index 2: tracker.record_attempt(completed) - FROM PIPELINE._mark_complete

        # The BUG: first completed record (index 0) is BEFORE save_raw (index 1)
        # If we crash between index 0 and index 1:
        #   - Tracker shows completed (from Executor's record at index 0)
        #   - But file doesn't exist (save_raw at index 1 never ran)

        # Expected (safe): save_raw should happen BEFORE any completed record
        # Actual (buggy): Executor records completed BEFORE save_raw runs

        # This assertion describes the CORRECT behavior (should pass after fix)
        # Currently FAILS because the bug exists
        assert save_index < first_tracker_completed_index, (
            f"BUG EXPOSED: FIRST tracker.record_attempt(completed) at index {first_tracker_completed_index} "
            f"happens BEFORE persistence.save_raw at index {save_index}! "
            f"Operation log: {operation_log}. "
            "This timing gap allows inconsistent state on interruption: "
            "if interrupted between these two operations, tracker shows completed but file doesn't exist."
        )


# ============================================================
# Test 6: Atomic Operation Verification
# ============================================================

class TestAtomicCompletionRequirement:
    """
    Test that completion should be atomic:
    Either (file saved AND tracker updated) OR neither.

    Current implementation violates this atomicity.
    """

    def test_completion_should_be_atomic(
        self, temp_output_dir, accepting_hooks, model_chain
    ):
        """
        After process_all(), for every "completed" unit in tracker,
        the corresponding file MUST exist.

        This invariant is violated by the current implementation.
        """
        persistence = ResultPersistence(temp_output_dir)
        tracker = ProcessingTracker(temp_output_dir / "progress.json", "TestProcessor")

        fake_llm = FakeLLMClient(default_response="processed content")

        pipeline = ProcessingPipelineV2(
            processor=TestProcessor(),
            llm_client=fake_llm,
            persistence=persistence,
            tracker=tracker,
            hooks=accepting_hooks,
            model_chain=model_chain,
            max_workers=1,
        )

        units = create_units({
            "chapter_1": SHORT_CHAPTER,
            "chapter_2": MEDIUM_CHAPTER,
        })

        # Make save_raw fail for chapter_2 (simulating partial failure)
        original_save_raw = persistence.save_raw

        def selective_save_raw(key, content):
            if key == "chapter_2":
                raise IOError("Disk full (simulated)")
            return original_save_raw(key, content)

        try:
            with patch.object(persistence, 'save_raw', selective_save_raw):
                pipeline.process_all(units)
        except IOError:
            pass  # Expected

        # Check atomicity invariant:
        # For every completed unit in tracker, file must exist
        for key in ["chapter_1", "chapter_2"]:
            tracker_says_complete = tracker.is_unit_complete(key)
            file_exists = persistence.has_raw(key) or persistence.has_validated(key)

            # Atomicity: if tracker says complete, file must exist
            if tracker_says_complete:
                # This assertion describes the CORRECT behavior
                # Currently may FAIL for chapter_2 because tracker was updated
                # before save_raw was called (and save_raw failed)
                assert file_exists, (
                    f"BUG EXPOSED: Atomicity violated for {key}! "
                    f"Tracker shows completed={tracker_says_complete} "
                    f"but file exists={file_exists}. "
                    "Completion must be atomic: both tracker AND file, or neither."
                )
