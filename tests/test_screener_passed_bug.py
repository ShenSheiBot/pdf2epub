"""
Bug Exposure Tests: screener_passed not propagated from execute().

This file contains E2E tests that EXPOSE the P0 bug where screener_passed
is collected correctly in _process_online() but NOT passed through in the
final ExecutionResult returned by execute().

Bug Location: executor.py, line 381-395
- execute() collects screener_passed.update(online_result.screener_passed)
- But the final `return ExecutionResult(...)` does NOT include screener_passed
- Result: Pipeline always sees empty screener_passed, defeating the cost-saving mechanism

Expected Behavior (what tests assert):
- When a validator returns context_ready=True, the unit ID should appear in result.screener_passed
- This signals to the Pipeline that the unit passed individual screening
- Pipeline can then skip expensive batch validation for these units

Actual Behavior (why tests fail):
- screener_passed is always empty in the returned ExecutionResult
- Pipeline has no way to know which units passed screening
- All units must go through batch validation (wasted LLM calls)

Test Strategy:
1. Set up validator that returns context_ready=True
2. Execute units through Executor.execute()
3. Assert that screener_passed contains the expected unit IDs
4. Tests FAIL because bug exists - exposing the issue
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import MagicMock

from pdf2epub.core.executor import (
    Executor,
    ChainEntry,
    ExecutionResult,
    QuotaConfig,
)
from pdf2epub.core.work_unit import WorkUnit
from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier
from pdf2epub.core.hooks._protocol import HookResult

from tests.fixtures.fake_llm import FakeLLMClient
from tests.fixtures.sample_content import SHORT_CHAPTER, MEDIUM_CHAPTER, LONG_CHAPTER


# ============================================================
# Test Infrastructure
# ============================================================

class MinimalProcessor:
    """Minimal processor that satisfies the protocol."""

    name = "minimal_processor"

    def build_prompt(self, content: str, context: Any = None) -> str:
        return f"Process this:\n{content}"

    def clean_response(self, response: str) -> str:
        return response.strip()

    def post_process(self, result: str, context: Any = None) -> str:
        return result


class ContextReadyValidator:
    """
    A validator that ALWAYS returns context_ready=True.

    This simulates a screener that determines the result is high-quality
    and can be used for context injection without batch validation.
    """

    def __init__(self, context_ready: bool = True):
        self._context_ready = context_ready

    @property
    def name(self) -> str:
        return "ContextReadyValidator"

    @property
    def role(self) -> str:
        return "screener"

    def validate(self, key: str, original: str, result: str) -> HookResult:
        """Always accept, with configurable context_ready."""
        return HookResult(accepted=True, context_ready=self._context_ready)


def create_hooks_with_context_ready(context_ready: bool = True) -> CompositeHooks:
    """Create hooks that return specified context_ready value."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()
    hooks._validators = [ContextReadyValidator(context_ready=context_ready)]
    return hooks


def create_test_executor(
    hooks: CompositeHooks,
    llm_client: FakeLLMClient = None,
) -> Executor:
    """Create executor with minimal configuration for testing."""
    chain = [ChainEntry(provider="test", model="test-model", mode="online")]
    return Executor(
        llm_client=llm_client or FakeLLMClient(default_response="processed"),
        model_chain=chain,
        processor=MinimalProcessor(),
        hooks=hooks,
        max_workers=1,
    )


# ============================================================
# BUG EXPOSURE TESTS
# ============================================================

class TestScreenerPassedBugExposure:
    """
    Tests that expose the screener_passed propagation bug.

    All tests in this class assert CORRECT behavior.
    They FAIL because the bug prevents correct behavior.
    """

    def test_single_unit_screener_passed_not_empty(self):
        """
        EXPOSES BUG: Single unit with context_ready=True should be in screener_passed.

        Expected: result.screener_passed == {"unit_1"}
        Actual (bug): result.screener_passed == set()

        This is the simplest possible test case that exposes the bug.
        """
        # Setup: validator returns context_ready=True
        hooks = create_hooks_with_context_ready(context_ready=True)
        fake_llm = FakeLLMClient(default_response="translated content")
        executor = create_test_executor(hooks, fake_llm)

        # Execute single unit
        unit = WorkUnit(id="unit_1", file_key="unit_1", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # Verify unit completed successfully
        assert "unit_1" in result.completed, "Unit should complete successfully"

        # THE BUG EXPOSURE ASSERTION
        # This SHOULD pass, but FAILS due to bug
        assert "unit_1" in result.screener_passed, (
            f"unit_1 should be in screener_passed because context_ready=True.\n"
            f"Expected: {{'unit_1'}}\n"
            f"Actual: {result.screener_passed}"
        )

    def test_multiple_units_all_screener_passed(self):
        """
        EXPOSES BUG: Multiple units with context_ready=True should all be in screener_passed.

        Expected: result.screener_passed == {"unit_1", "unit_2", "unit_3"}
        Actual (bug): result.screener_passed == set()
        """
        hooks = create_hooks_with_context_ready(context_ready=True)
        fake_llm = FakeLLMClient(default_response="translated")
        executor = create_test_executor(hooks, fake_llm)

        units = [
            WorkUnit(id="unit_1", file_key="unit_1", content=SHORT_CHAPTER),
            WorkUnit(id="unit_2", file_key="unit_2", content=MEDIUM_CHAPTER),
            WorkUnit(id="unit_3", file_key="unit_3", content=LONG_CHAPTER),
        ]
        result = executor.execute(units)

        # All should complete
        assert result.completed == {"unit_1", "unit_2", "unit_3"}

        # THE BUG EXPOSURE ASSERTION
        assert result.screener_passed == {"unit_1", "unit_2", "unit_3"}, (
            f"All units should be in screener_passed.\n"
            f"Expected: {{'unit_1', 'unit_2', 'unit_3'}}\n"
            f"Actual: {result.screener_passed}"
        )

    def test_screener_passed_count_matches_completed(self):
        """
        EXPOSES BUG: When all validators return context_ready=True,
        screener_passed count should equal completed count.

        Expected: len(result.screener_passed) == len(result.completed) == 5
        Actual (bug): len(result.screener_passed) == 0
        """
        hooks = create_hooks_with_context_ready(context_ready=True)
        fake_llm = FakeLLMClient(default_response="done")
        executor = create_test_executor(hooks, fake_llm)

        units = [
            WorkUnit(id=f"chapter_{i}", file_key=f"ch{i}", content=SHORT_CHAPTER)
            for i in range(5)
        ]
        result = executor.execute(units)

        # All should complete
        assert len(result.completed) == 5

        # THE BUG EXPOSURE ASSERTION
        assert len(result.screener_passed) == 5, (
            f"screener_passed count should equal completed count.\n"
            f"Expected: 5\n"
            f"Actual: {len(result.screener_passed)}"
        )

    def test_screener_passed_is_subset_of_completed(self):
        """
        EXPOSES BUG: screener_passed should be a subset of completed.

        With context_ready=True for all, screener_passed should equal completed.
        The bug makes this always empty, so the subset relation holds vacuously,
        but the equality check fails.
        """
        hooks = create_hooks_with_context_ready(context_ready=True)
        executor = create_test_executor(hooks)

        units = [
            WorkUnit(id="a", file_key="a", content="content a"),
            WorkUnit(id="b", file_key="b", content="content b"),
        ]
        result = executor.execute(units)

        # Subset relation (always true due to bug making it empty)
        assert result.screener_passed.issubset(result.completed)

        # THE BUG EXPOSURE ASSERTION
        # With context_ready=True, screener_passed should be non-empty
        assert len(result.screener_passed) > 0, (
            "screener_passed should not be empty when validators return context_ready=True"
        )


class TestScreenerPassedDataFlowTracing:
    """
    Tests that trace the data flow to pinpoint where screener_passed gets lost.

    These tests document the exact location of the bug by checking
    intermediate state if accessible.
    """

    def test_execute_returns_empty_screener_passed_despite_context_ready(self):
        """
        Direct test that execute() return value has empty screener_passed.

        This test documents the SYMPTOM: execute() returns empty screener_passed
        even when the internal _process_online() correctly collected the IDs.
        """
        # Use a validator that definitely returns context_ready=True
        validator = MagicMock()
        validator.name = "definite_context_ready"
        validator.role = "screener"
        validator.validate.return_value = HookResult(accepted=True, context_ready=True)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        hooks._validators = [validator]

        executor = create_test_executor(hooks)

        unit = WorkUnit(id="traced_unit", file_key="traced", content="test content")
        result = executor.execute([unit])

        # Verify validator was called and returned context_ready=True
        assert validator.validate.called, "Validator should have been called"

        # Verify unit completed
        assert "traced_unit" in result.completed

        # THE BUG: Despite context_ready=True from validator, screener_passed is empty
        # This proves the bug is in execute(), not in the validator or hooks
        assert "traced_unit" in result.screener_passed, (
            "BUG EXPOSED: execute() does not include screener_passed in return value.\n"
            f"Validator returned context_ready=True, but result.screener_passed={result.screener_passed}"
        )


class TestPipelineCostSavingIntegration:
    """
    Tests that demonstrate the IMPACT of the bug on the cost-saving mechanism.

    These tests show why fixing this bug matters:
    - screener_passed tells Pipeline which units can skip batch validation
    - Empty screener_passed means ALL units go through expensive batch validation
    - This wastes LLM API calls and increases costs
    """

    def test_pipeline_cannot_skip_batch_validation_due_to_bug(self):
        """
        Demonstrates that Pipeline cannot implement cost savings due to this bug.

        Scenario:
        - 10 units processed
        - All pass individual screening (context_ready=True)
        - Pipeline should be able to skip batch validation for all 10
        - Bug: screener_passed is empty, Pipeline must validate all 10 anyway

        Cost impact: 10 extra LLM calls that could have been avoided
        """
        hooks = create_hooks_with_context_ready(context_ready=True)
        executor = create_test_executor(hooks)

        units = [
            WorkUnit(id=f"doc_{i}", file_key=f"doc_{i}", content=f"Document {i} content")
            for i in range(10)
        ]
        result = executor.execute(units)

        # All completed
        assert len(result.completed) == 10

        # Simulate Pipeline's decision logic
        # Pipeline checks: if unit_id in screener_passed, skip batch validation
        units_needing_batch_validation = result.completed - result.screener_passed

        # THE BUG IMPACT: All units "need" batch validation due to empty screener_passed
        # Expected: 0 units need batch validation (all passed screening)
        # Actual: 10 units need batch validation (bug makes screener_passed empty)

        expected_needing_validation = 0  # All passed screening
        actual_needing_validation = len(units_needing_batch_validation)

        assert actual_needing_validation == expected_needing_validation, (
            f"BUG IMPACT: Pipeline cannot skip batch validation.\n"
            f"Units that passed screening: should be 10, got {len(result.screener_passed)}\n"
            f"Units needing batch validation: should be 0, got {actual_needing_validation}\n"
            f"Extra LLM calls due to bug: {actual_needing_validation}"
        )


class TestCorrectBehaviorDocumentation:
    """
    These tests document what CORRECT behavior looks like.

    They serve as a specification for the fix.
    """

    def test_context_ready_false_should_not_add_to_screener_passed(self):
        """
        When context_ready=False, unit should NOT be in screener_passed.

        This test PASSES even with the bug, because the bug only affects
        propagation (not collection). It documents correct non-inclusion behavior.
        """
        hooks = create_hooks_with_context_ready(context_ready=False)
        executor = create_test_executor(hooks)

        unit = WorkUnit(id="no_screen", file_key="no_screen", content="test")
        result = executor.execute([unit])

        # Should complete
        assert "no_screen" in result.completed

        # Should NOT be in screener_passed (context_ready=False)
        # This passes even with bug since empty set doesn't contain anything
        assert "no_screen" not in result.screener_passed

    def test_execution_result_structure_includes_screener_passed(self):
        """
        Verify ExecutionResult has screener_passed field.

        This test PASSES - the field exists, it's just not populated correctly.
        """
        result = ExecutionResult()

        # Field exists
        assert hasattr(result, "screener_passed")

        # Default is empty set
        assert result.screener_passed == set()

        # Can be populated
        result.screener_passed.add("test_id")
        assert "test_id" in result.screener_passed


# ============================================================
# Regression Prevention Tests
# ============================================================

class TestRegressionPrevention:
    """
    Tests to prevent regression after the bug is fixed.

    These tests should PASS once the fix is applied.
    They ensure the fix doesn't break related functionality.
    """

    def test_screener_passed_survives_execute_return(self):
        """
        Core regression test: screener_passed must survive the execute() return.

        After fix, this test should pass, ensuring the bug doesn't regress.
        """
        hooks = create_hooks_with_context_ready(context_ready=True)
        executor = create_test_executor(hooks)

        unit = WorkUnit(id="regression_test", file_key="rt", content="content")
        result = executor.execute([unit])

        # This is THE regression test assertion
        assert result.screener_passed == {"regression_test"}, (
            f"REGRESSION: screener_passed should contain 'regression_test'\n"
            f"Got: {result.screener_passed}"
        )

    def test_other_result_fields_unaffected(self):
        """
        Ensure fixing screener_passed doesn't break other result fields.
        """
        hooks = create_hooks_with_context_ready(context_ready=True)
        executor = create_test_executor(hooks)

        units = [
            WorkUnit(id="u1", file_key="u1", content="c1"),
            WorkUnit(id="u2", file_key="u2", content="c2"),
        ]
        result = executor.execute(units)

        # These should all still work correctly
        assert result.completed == {"u1", "u2"}
        assert result.failed == set()
        assert result.skipped == set()
        assert len(result.results) == 2
        assert result.total_attempts >= 2
        assert result.successful_attempts >= 2
