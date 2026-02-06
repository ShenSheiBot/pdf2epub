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

        Setup:
        - Batch job succeeds but validation rejects
        - Check that unit's validation quota was decremented

        Expected: After batch validation rejection, state.quotas[VALIDATION] should
                  be decremented by 1 (from initial value)
        Actual (BUG): Quota is not decremented, state.apply_effect() is never called
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

        # Configure quota - we'll check if it's decremented
        initial_validation_quota = 3
        quota = QuotaConfig(total=10, per_type={ErrorType.VALIDATION: initial_validation_quota})

        # Track apply_effect calls
        apply_effect_calls = []

        # Create executor and units
        fake_llm = FakeLLMClient(default_response="online fallback")
        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            quota_config=quota,
            online_fallback_threshold=0,  # Disable threshold to force batch path
        )

        units = create_units({"unit1": SHORT_CHAPTER})

        # Access unit_states to verify quota was decremented
        # We need to patch the executor to capture the unit_states after batch processing
        original_process_batch = executor._process_batch
        captured_states = {}

        def capturing_process_batch(units, batch_entry, context_base, originals, unit_states):
            # Store reference to unit_states for later inspection
            captured_states['unit_states'] = unit_states
            captured_states['initial_quota'] = unit_states['unit1'].quotas[ErrorType.VALIDATION]
            result = original_process_batch(units, batch_entry, context_base, originals, unit_states)
            captured_states['final_quota'] = unit_states['unit1'].quotas[ErrorType.VALIDATION]
            return result

        executor._process_batch = capturing_process_batch

        result = executor.execute(units)

        # Verify batch was called and validation rejected
        assert batch_client.submit_called, "Batch submit should have been called"
        assert "unit1" in result.validation_failed, (
            f"Unit should be in validation_failed, got: completed={result.completed}, "
            f"failed={result.failed}, validation_failed={result.validation_failed}"
        )

        # BUG ASSERTION: Check that quota was decremented
        # When fixed, the validation quota should be decremented from initial value
        initial = captured_states.get('initial_quota', initial_validation_quota)
        final = captured_states.get('final_quota', initial_validation_quota)

        assert final < initial, (
            f"BUG: Batch validation rejection should decrement quota. "
            f"Initial quota: {initial}, Final quota: {final}. "
            f"apply_effect() was not called on validation failure in batch path."
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
                result = MagicMock()
                result.accepted = False
                result.context_ready = False
                result.rejection_reason = "First attempt rejected"
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

        # Low validation quota - if decremented properly, only 1 retry allowed
        quota = QuotaConfig(total=5, per_type={ErrorType.VALIDATION: 1})

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
    Bug 3: When batch fails and retries, it should try the next model in the chain,
    but it reuses the same batch_entry.

    Location: executor.py line 355-356 and line 254

    At line 254: batch_entry = self._get_batch_entry()  # Gets FIRST batch entry
    At line 355-356: retry_result = self._process_batch(
        failed_units, batch_entry, ...)  # Reuses SAME batch_entry

    The batch_entry is fetched once and reused for retries. But after a batch
    failure, the chain should advance to the next batch entry (if available).

    Compare to online path where each retry naturally uses state.get_current_entry()
    which advances through the chain.
    """

    def test_batch_retry_should_use_next_batch_model(self):
        """
        Test that batch retry uses the next batch model in chain.

        Setup:
        - Chain with two batch models from different providers
        - First batch job fails entirely
        - Configure BOTH submissions to fail (simulating that the same provider is
          being retried, which keeps failing)
        - Verify that online fallback is needed (proving the retry didn't advance)

        Expected: Second batch submission uses second batch model (would succeed)
        Actual (BUG): Second batch submission reuses first batch model (keeps failing)
        """
        chain = [
            ChainEntry(provider="provider1", model="batch-model-1", mode="batch"),
            ChainEntry(provider="provider2", model="batch-model-2", mode="batch"),
            ChainEntry(provider="fallback", model="online-model", mode="online"),
        ]

        hooks = create_accepting_hooks()

        # Track batch_entry used for each submission
        batch_entries_used = []

        class TrackingBatchClient(FakeBatchClient):
            """
            Batch client that tracks which model/provider is being used.
            Both submissions will fail to simulate that the same failing provider
            is being retried.
            """
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._submission_count = 0

            def submit(self, requests, display_name=None):
                self._submission_count += 1
                # Always configure failure - the point is that both retries use
                # the same failing provider
                self.configure_job(FakeBatchJobConfig(
                    state=BatchJobState.FAILED,
                    error=f"batch submission #{self._submission_count} failed",
                ))
                return super().submit(requests, display_name)

        batch_client = TrackingBatchClient()

        fake_llm = FakeLLMClient(default_response="online fallback")

        # LOW fallback threshold to force batch retry instead of online fallback
        # len(failed_units) > threshold triggers retry (line 352 in executor.py)
        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            online_fallback_threshold=5,  # 10 units > 5, so retry will be triggered
        )

        # Track the batch_entry parameter passed to _process_batch
        original_process_batch = executor._process_batch

        def tracking_wrapper(units, batch_entry, context_base, originals, unit_states):
            batch_entries_used.append((batch_entry.provider, batch_entry.model))
            return original_process_batch(units, batch_entry, context_base, originals, unit_states)

        executor._process_batch = tracking_wrapper

        # Need enough units to exceed online_fallback_threshold for retry
        units = create_units({f"unit{i}": SHORT_CHAPTER for i in range(10)})

        result = executor.execute(units)

        # Verify batch was submitted twice (first fail, then retry)
        assert batch_client.submit_count == 2, (
            f"Batch should be submitted twice (first fail + retry). "
            f"Actual submissions: {batch_client.submit_count}"
        )

        # For this test, we verify the units eventually complete via online fallback
        completed_count = len(result.completed)
        assert completed_count == 10, (
            f"All units should complete via online fallback. "
            f"Completed: {completed_count}, Failed: {len(result.failed)}"
        )

        # BUG ASSERTION: The second batch submission should use provider2, not provider1
        # Currently, batch_entry is captured once at execute() start and reused
        assert len(batch_entries_used) == 2, (
            f"Should have 2 _process_batch calls. Actual: {len(batch_entries_used)}"
        )

        first_entry = batch_entries_used[0]
        second_entry = batch_entries_used[1]

        assert first_entry != second_entry, (
            f"BUG: Batch retry reuses same batch_entry instead of advancing chain. "
            f"First call: {first_entry}, Second call: {second_entry}. "
            f"Expected first=('provider1', 'batch-model-1'), "
            f"second=('provider2', 'batch-model-2'). "
            f"The batch_entry is captured once at execute() start and reused for all retries."
        )

    def test_batch_entry_should_be_refreshed_from_state_on_retry(self):
        """
        Test that batch retry gets fresh batch_entry from unit state.

        The fix should involve getting the batch entry from unit state
        (which is updated via apply_effect) rather than reusing the
        original batch_entry captured at the start of execute().
        """
        chain = [
            ChainEntry(provider="provider1", model="batch-model-1", mode="batch"),
            ChainEntry(provider="provider2", model="batch-model-2", mode="batch"),
            ChainEntry(provider="fallback", model="online-fallback", mode="online"),
        ]

        hooks = create_accepting_hooks()

        # Create a batch client that fails first, succeeds second
        batch_client = FakeBatchClient()

        # Configure to fail first, succeed after
        batch_client.configure_job(FakeBatchJobConfig(
            state=BatchJobState.FAILED,
            error="First batch provider failed",
        ))

        # Track the state of unit_states during batch processing
        captured_chain_states = []

        # Patch _process_batch to capture chain state before each batch call
        fake_llm = FakeLLMClient(default_response="online result")

        # LOW threshold so retry is triggered (len(failed) > threshold)
        executor = create_executor(
            chain, hooks, fake_llm,
            batch_client=batch_client,
            online_fallback_threshold=5,  # 10 units > 5 = retry triggered
        )

        original_process_batch = executor._process_batch
        call_count = [0]

        def tracking_process_batch(units, batch_entry, context_base, originals, unit_states):
            call_count[0] += 1

            # Capture the chain state for first unit
            if units:
                first_unit = units[0]
                state = unit_states.get(first_unit.id)
                if state:
                    captured_chain_states.append({
                        'call': call_count[0],
                        'batch_entry_used': (batch_entry.provider, batch_entry.model),
                        'chain_length': len(state.chain),
                        'first_chain_entry': (state.chain[0].provider, state.chain[0].model) if state.chain else None,
                    })

            # Call original first
            result = original_process_batch(units, batch_entry, context_base, originals, unit_states)

            # After first call completes, configure success for the retry
            if call_count[0] == 1:
                batch_client.configure_job(FakeBatchJobConfig(
                    state=BatchJobState.SUCCEEDED,
                    unit_configs={u.id: FakeBatchUnitConfig(text=f"result for {u.id}") for u in units},
                ))

            return result

        executor._process_batch = tracking_process_batch

        # Create enough units to trigger retry (> online_fallback_threshold)
        units = create_units({f"unit{i}": SHORT_CHAPTER for i in range(10)})

        result = executor.execute(units)

        # Verify we got multiple batch calls (fail + retry)
        assert len(captured_chain_states) >= 2, (
            f"Should have at least 2 batch calls (fail + retry). "
            f"Actual: {len(captured_chain_states)}"
        )

        # BUG ASSERTION: On retry, batch_entry should be different
        # (reflecting that first batch failed and chain advanced)
        first_call = captured_chain_states[0]
        second_call = captured_chain_states[1]

        # The bug: both calls use same batch_entry
        # After fix: second call should use provider2's batch model
        assert first_call['batch_entry_used'] != second_call['batch_entry_used'], (
            f"BUG: Batch retry reuses same batch_entry instead of advancing chain. "
            f"First call used: {first_call['batch_entry_used']}, "
            f"Second call used: {second_call['batch_entry_used']}. "
            f"Expected second call to use provider2/batch-model-2 after first batch failed. "
            f"The batch_entry is captured once at execute() start and reused for all retries."
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
                result = MagicMock()
                result.accepted = False
                result.context_ready = False
                result.rejection_reason = "First validation rejected"
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
