"""
Tests to expose bugs in batch processing.

These tests are designed to FAIL when the bugs exist and PASS once the bugs are fixed.

Bug 1: Batch validation reject doesn't apply_effect/decrement quota
Bug 2: Batch attempts count includes pre_process skipped units
Bug 3: Batch chain only uses first batch entry, retry reuses same entry
"""

import pytest
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch

from pdf2epub.core.executor import (
    Executor,
    ChainEntry,
    ExecutionResult,
    QuotaConfig,
)
from pdf2epub.core.executor.state import UnitState, create_unit_state
from pdf2epub.core.work_unit import WorkUnit
from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier, PreProcessResult
from pdf2epub.core.types import ErrorType
from pdf2epub.utils.batch_utils import BatchJobState

from .fixtures.fake_llm import FakeLLMClient, FakeResponse, FakeErrorType
from .fixtures.fake_batch import (
    FakeBatchClient,
    FakeBatchJobConfig,
    FakeBatchUnitConfig,
    FakeBatchErrorType,
)
from .fixtures.sample_content import SHORT_CHAPTER, MEDIUM_CHAPTER


# ============================================================
# Test Processor
# ============================================================

class TestProcessor:
    """Test processor matching real interface."""

    name = "test_processor"

    def build_prompt(self, content: str, context: Any = None) -> str:
        return f"Process:\n{content}"

    def clean_response(self, response: str) -> str:
        return response.strip()

    def post_process(self, result: str, context: Any = None) -> str:
        return result


# ============================================================
# Helpers
# ============================================================

def create_executor(
    chain: List[ChainEntry],
    hooks: CompositeHooks,
    llm_client: Optional[FakeLLMClient] = None,
    batch_client: Optional[FakeBatchClient] = None,
    quota_config: Optional[QuotaConfig] = None,
    online_fallback_threshold: int = 0,  # Disable batch->online threshold for testing
    **kwargs,
) -> Executor:
    """Create executor with fake components."""
    return Executor(
        llm_client=llm_client or FakeLLMClient(),
        model_chain=chain,
        processor=TestProcessor(),
        hooks=hooks,
        batch_client=batch_client,
        quota_config=quota_config,
        max_workers=1,
        batch_poll_interval=0,  # No delay in tests
        online_fallback_threshold=online_fallback_threshold,
        **kwargs,
    )


def create_units(contents: Dict[str, str]) -> List[WorkUnit]:
    """Create WorkUnit list from content dict."""
    return [
        WorkUnit(id=key, file_key=key, content=content)
        for key, content in contents.items()
    ]


def create_accepting_hooks() -> CompositeHooks:
    """Create hooks that accept all validation."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()
    validator = MagicMock()
    validator.name = "accepting"
    validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
    hooks._validators = [validator]
    return hooks


def create_rejecting_hooks() -> CompositeHooks:
    """Create hooks that reject all validation."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()
    validator = MagicMock()
    validator.name = "rejecting"
    result = MagicMock()
    result.accepted = False
    result.context_ready = False
    result.rejection_reason = "Test rejection"
    validator.validate.return_value = result
    hooks._validators = [validator]
    return hooks


# ============================================================
# Bug 1: Batch validation reject doesn't apply_effect/decrement quota
# ============================================================

class TestBug1_BatchValidationNoApplyEffect:
    """
    Bug 1: When batch validation rejects a unit, it should call state.apply_effect()
    to decrement quota, but it doesn't.

    Location: executor.py around line 1245-1263

    In the batch path, when hook_result.accepted is False:
    - state.record_attempt(transformed) is called (correct)
    - But state.apply_effect() is NOT called (BUG)

    Compare to online path in _handle_failure() which ALWAYS calls:
    - state.apply_effect(effect, current_entry)

    This means batch validation failures don't decrement quota, so the unit
    could theoretically retry forever without quota protection.
    """

    def test_batch_validation_reject_should_decrement_quota(self):
        """
        Test that batch validation rejection decrements the validation quota.

        With the new Mega Unit architecture, validation failures go through
        unified _handle_failure(), which calls apply_effect() to decrement quota.

        Behavior verification:
        - If quota is properly decremented, with quota=1 validation,
          the unit should fail after first validation rejection (no retry left)
        """
        chain = [
            ChainEntry(provider="batch-provider", model="batch-model", mode="batch"),
            ChainEntry(provider="online-provider", model="online-model", mode="online"),
        ]

        # Create rejecting hooks - batch result will pass pre_process but fail validation
        hooks = create_rejecting_hooks()

        # Batch succeeds with a result, but validation will reject it
        batch_client = FakeBatchClient()
        batch_client.configure_success({"unit1": "batch result that will be rejected"})

        # Very low quota - if decremented, unit will fail (no retry left)
        quota = QuotaConfig(total=1, per_type={ErrorType.VALIDATION: 1})

        fake_llm = FakeLLMClient(default_response="online fallback")
        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            quota_config=quota,
            online_fallback_threshold=0,  # Disable threshold to force batch path
        )

        units = create_units({"unit1": SHORT_CHAPTER})
        result = executor.execute(units)

        # Verify batch was called
        assert batch_client.submit_called, "Batch submit should have been called"

        # With quota=1 and validation rejection, the quota is decremented to 0.
        # Since VALIDATION effect doesn't remove_current_model, chain still has entries.
        # But can_retry() fails because validation quota is 0.
        # Unit should end up failed (no retry allowed).
        assert "unit1" in result.failed or "unit1" in result.validation_failed, (
            f"Unit should fail after validation rejection exhausts quota. "
            f"completed={result.completed}, failed={result.failed}, "
            f"validation_failed={result.validation_failed}"
        )

    def test_batch_validation_reject_should_advance_chain_on_retry(self):
        """
        Test that batch validation rejection advances the model chain properly.

        When a unit fails validation, it should:
        1. Have apply_effect() called to potentially advance the chain
        2. When retried (if quota allows), use the next model in chain

        This test verifies that after batch validation rejection, when falling
        back to online, the chain state reflects the failure properly.
        """
        chain = [
            ChainEntry(provider="batch-provider", model="batch-model", mode="batch"),
            ChainEntry(provider="online-provider", model="online-model-1", mode="online"),
            ChainEntry(provider="online-provider", model="online-model-2", mode="online"),
        ]

        # First rejection, then accept on retry
        call_count = [0]
        def conditional_validate(key, original, processed, chapter_type, context):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (batch) - reject
                # Use "validation" keyword so error classifier identifies this as VALIDATION
                result = MagicMock()
                result.accepted = False
                result.context_ready = False
                result.rejection_reason = "validation failed: format error"
                return processed, result
            else:
                # Second call (online fallback) - accept
                result = MagicMock()
                result.accepted = True
                result.context_ready = False
                return processed, result

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        hooks.post_process = conditional_validate

        batch_client = FakeBatchClient()
        batch_client.configure_success({"unit1": "batch result"})

        # Quota=2 allows: 1 for batch validation failure + 1 for online retry
        quota = QuotaConfig(total=5, per_type={ErrorType.VALIDATION: 2})

        fake_llm = FakeLLMClient(default_response="online result")
        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            quota_config=quota,
            online_fallback_threshold=0,  # Disable threshold to force batch path
        )

        units = create_units({"unit1": SHORT_CHAPTER})
        result = executor.execute(units)

        # After fix: batch validation failure should decrement quota, then online fallback
        # should succeed. The unit should complete.
        assert "unit1" in result.completed, (
            f"Unit should complete after batch validation failure + online fallback. "
            f"completed={result.completed}, failed={result.failed}, "
            f"validation_failed={result.validation_failed}"
        )

        # Verify that apply_effect was properly called (quota was used)
        assert call_count[0] >= 2, (
            f"Validation should be called at least twice (batch + online). "
            f"Actual calls: {call_count[0]}"
        )


# ============================================================
# Bug 2: Batch attempts count includes pre_process skipped units
# ============================================================

class TestBug2_BatchAttemptsInflated:
    """
    Bug 2: The batch_attempts calculation includes units that were skipped
    by pre_process, inflating the count.

    Location: executor.py line 318 vs line 1021-1060

    At line 318: total_attempts += len(batch_units)
    But batch_units includes ALL units designated for batch, not just those
    that were actually processed. Units skipped by pre_process (lines 1054-1060)
    don't generate a batch request, but are still counted in total_attempts.

    The batch_requests list (line 1021) only contains units that passed pre_process.
    """

    def test_batch_attempts_should_not_count_skipped_units(self):
        """
        Test that total_attempts doesn't count units skipped by pre_process.

        Setup:
        - 3 units designated for batch
        - 1 unit is skipped by pre_process
        - 2 units are actually processed via batch

        Expected: total_attempts should be 2 (only actually processed units)
        Actual (BUG): total_attempts is 3 (includes skipped unit)
        """
        chain = [
            ChainEntry(provider="batch-provider", model="batch-model", mode="batch"),
        ]

        # Create hooks that skip specific units
        def conditional_pre_process(key, content, context):
            if key == "skip_me":
                # Skip this unit
                return PreProcessResult(
                    should_process=False,
                    fallback_result="skipped content",
                    skip_reason="test skip"
                )
            return PreProcessResult(should_process=True)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        hooks.pre_process = conditional_pre_process

        # Accept all validation
        validator = MagicMock()
        validator.name = "accepting"
        validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
        hooks._validators = [validator]

        batch_client = FakeBatchClient()
        batch_client.configure_success({
            "unit1": "result 1",
            "unit2": "result 2",
            # skip_me not in here since it won't be submitted
        })

        executor = create_executor(
            chain, hooks,
            batch_client=batch_client,
        )

        units = create_units({
            "unit1": SHORT_CHAPTER,
            "unit2": SHORT_CHAPTER,
            "skip_me": SHORT_CHAPTER,  # This unit will be skipped by pre_process
        })

        result = executor.execute(units)

        # Verify basic correctness
        assert "unit1" in result.completed, "unit1 should be completed"
        assert "unit2" in result.completed, "unit2 should be completed"
        assert "skip_me" in result.skipped, "skip_me should be skipped"

        # Verify batch only submitted 2 requests (not 3)
        submitted_keys = batch_client.get_submitted_keys(0)
        assert len(submitted_keys) == 2, (
            f"Batch should only submit 2 units (not skipped one). "
            f"Submitted: {submitted_keys}"
        )
        assert "skip_me" not in submitted_keys, "skip_me should not be in batch requests"

        # BUG ASSERTION: Check total_attempts count
        # When fixed, total_attempts should be 2 (only units actually in batch)
        # Currently it's 3 because it uses len(batch_units) instead of len(batch_requests)
        assert result.total_attempts == 2, (
            f"BUG: total_attempts should be 2 (units actually processed), "
            f"but got {result.total_attempts}. "
            f"The count incorrectly includes pre_process skipped units."
        )

    def test_batch_success_count_should_not_count_skipped_units(self):
        """
        Test that successful_attempts doesn't get inflated by skipped units.

        If total_attempts is inflated, the success rate calculation would be wrong.
        """
        chain = [
            ChainEntry(provider="batch-provider", model="batch-model", mode="batch"),
        ]

        # Skip 2 out of 5 units
        skip_keys = {"skip1", "skip2"}

        def conditional_pre_process(key, content, context):
            if key in skip_keys:
                return PreProcessResult(
                    should_process=False,
                    fallback_result=f"skipped {key}",
                    skip_reason="test skip"
                )
            return PreProcessResult(should_process=True)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        hooks.pre_process = conditional_pre_process

        validator = MagicMock()
        validator.name = "accepting"
        validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
        hooks._validators = [validator]

        batch_client = FakeBatchClient()
        batch_client.configure_success({
            "unit1": "result 1",
            "unit2": "result 2",
            "unit3": "result 3",
        })

        executor = create_executor(
            chain, hooks,
            batch_client=batch_client,
        )

        units = create_units({
            "unit1": SHORT_CHAPTER,
            "unit2": SHORT_CHAPTER,
            "unit3": SHORT_CHAPTER,
            "skip1": SHORT_CHAPTER,
            "skip2": SHORT_CHAPTER,
        })

        result = executor.execute(units)

        # 5 units total: 3 batch, 2 skipped
        # successful_attempts should be 3 (batch completions)
        # total_attempts should be 3 (units sent to batch)

        assert result.successful_attempts == 3, (
            f"successful_attempts should be 3, got {result.successful_attempts}"
        )

        # BUG: total_attempts is 5 instead of 3
        assert result.total_attempts == 3, (
            f"BUG: total_attempts should be 3 (units actually sent to batch), "
            f"but got {result.total_attempts}. "
            f"The count includes {len(skip_keys)} pre_process skipped units. "
            f"This inflates attempt count and skews success rate calculations."
        )


# ============================================================
# Bug 3: Batch chain only uses first batch entry, retry reuses same entry
# ============================================================

class TestBug3_BatchRetryReusesSameEntry:
    """
    Bug 3 (FIXED in Mega Unit architecture): Chain advancement on batch retry.

    With the Mega Unit architecture:
    - Batch job failure causes units to requeue with updated state
    - Unit state.chain is modified by apply_effect()
    - Requeued units pick up next model from their updated chain

    These tests verify the fix works correctly.
    """

    def test_batch_failure_allows_online_fallback(self):
        """
        Test that batch failure allows online fallback via chain advancement.

        With the new architecture:
        - Batch fails → apply_effect removes batch entry from chain
        - Units requeue with updated chain (now online-only)
        - Online fallback succeeds
        """
        chain = [
            ChainEntry(provider="provider1", model="batch-model-1", mode="batch"),
            ChainEntry(provider="fallback", model="online-model", mode="online"),
        ]

        hooks = create_accepting_hooks()

        batch_client = FakeBatchClient()
        # Use "network error" to trigger NETWORK classification (remove_current_model=True)
        batch_client.configure_job(FakeBatchJobConfig(
            state=BatchJobState.FAILED,
            error="network error: service unavailable",
        ))

        fake_llm = FakeLLMClient(default_response="online fallback result")

        # High quota to allow retries
        quota = QuotaConfig(total=5, per_type={ErrorType.NETWORK: 3})

        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            quota_config=quota,
            online_fallback_threshold=0,  # Disable threshold to force batch path
        )

        units = create_units({"unit1": SHORT_CHAPTER})
        result = executor.execute(units)

        # Verify batch was attempted
        assert batch_client.submit_called, "Batch should have been submitted"

        # Verify unit completed via online fallback
        assert "unit1" in result.completed, (
            f"Unit should complete via online fallback. "
            f"completed={result.completed}, failed={result.failed}"
        )

        # Verify LLM was called (online path)
        assert fake_llm.was_called("unit1"), (
            "LLM should be called for online fallback"
        )

    def test_batch_failure_advances_chain(self):
        """
        Test that batch failure properly advances chain via apply_effect.

        With NETWORK error effect (remove_current_model=True), the batch entry
        should be removed from chain, allowing online fallback.
        """
        # Two batch entries then online - tests chain advancement
        chain = [
            ChainEntry(provider="provider1", model="batch-model-1", mode="batch"),
            ChainEntry(provider="provider1", model="batch-model-2", mode="batch"),
            ChainEntry(provider="fallback", model="online-model", mode="online"),
        ]

        hooks = create_accepting_hooks()

        batch_client = FakeBatchClient()
        # Both batch attempts will fail - use "network error" for proper classification
        batch_client.configure_job(FakeBatchJobConfig(
            state=BatchJobState.FAILED,
            error="network error: connection refused",
        ))

        fake_llm = FakeLLMClient(default_response="online result")

        # High quota to allow multiple retries
        quota = QuotaConfig(total=10, per_type={ErrorType.NETWORK: 5})

        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            quota_config=quota,
            online_fallback_threshold=0,
        )

        units = create_units({"unit1": SHORT_CHAPTER})
        result = executor.execute(units)

        # Unit should eventually complete via online fallback
        assert "unit1" in result.completed, (
            f"Unit should complete after batch failures + online fallback. "
            f"completed={result.completed}, failed={result.failed}"
        )

        # Verify LLM was called (proves online path was used)
        assert fake_llm.was_called("unit1"), (
            "LLM should be called for online fallback after batch failures"
        )


# ============================================================
# Integration test combining all bugs
# ============================================================

class TestBatchBugsIntegration:
    """
    Integration test showing how the bugs interact in real scenarios.
    """

    def test_batch_failure_chain_should_behave_like_online(self):
        """
        Compare batch failure handling to online failure handling.

        Online path:
        1. Failure -> classify error -> apply_effect -> decrement quota -> maybe retry
        2. Each retry uses state.get_current_entry() (advances through chain)

        Batch path (buggy):
        1. Failure -> [no apply_effect] -> fallback to online
        2. Retry uses same batch_entry (doesn't advance chain)
        3. total_attempts counts skipped units

        After fix, both paths should behave consistently.
        """
        chain = [
            ChainEntry(provider="provider1", model="batch-1", mode="batch"),
            ChainEntry(provider="provider2", model="batch-2", mode="batch"),
            ChainEntry(provider="provider1", model="online-1", mode="online"),
        ]

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()

        # Reject first validation, accept subsequent
        validation_calls = [0]
        def counting_validate(key, original, processed, chapter_type, context):
            validation_calls[0] += 1
            if validation_calls[0] == 1:
                # Use "validation" keyword so error classifier identifies this as VALIDATION
                result = MagicMock()
                result.accepted = False
                result.context_ready = False
                result.rejection_reason = "validation failed: format error"
                return processed, result
            else:
                result = MagicMock()
                result.accepted = True
                result.context_ready = False
                return processed, result

        hooks.post_process = counting_validate

        batch_client = FakeBatchClient()
        batch_client.configure_success({"unit1": "batch result"})

        quota = QuotaConfig(total=5, per_type={ErrorType.VALIDATION: 2})

        fake_llm = FakeLLMClient(default_response="online result")
        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            quota_config=quota,
            online_fallback_threshold=0,  # Disable threshold to force batch path
        )

        units = create_units({"unit1": SHORT_CHAPTER})
        result = executor.execute(units)

        # After batch validation failure:
        # - Quota should be decremented (Bug 1)
        # - Chain should advance for retry (Bug 3)
        # - Only processed units should count as attempts (Bug 2)

        assert "unit1" in result.completed, (
            f"Unit should eventually complete. "
            f"completed={result.completed}, failed={result.failed}"
        )

        # The test will pass once all bugs are fixed and batch path
        # behaves consistently with online path
