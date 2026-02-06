"""
Test for Quota Semantics Bug - P0 issue.

Bug: can_retry() checks quotas[error_type], but apply_effect()
decrements quotas[effect.quota_type]. When these differ, the system
becomes inconsistent.

Example: RATE_LIMIT uses NETWORK quota pool.
- apply_effect() decrements NETWORK quota
- can_retry(RATE_LIMIT) checks RATE_LIMIT quota (wrong!)
- Result: retry loop continues even when NETWORK quota is 0

These tests verify EXPECTED CORRECT behavior using FakeLLMClient.
Tests should FAIL before the fix and PASS after.

Error types and their ACTUAL quota_type (from DefaultErrorClassifier):
- RATE_LIMIT → NETWORK
- TIMEOUT → NETWORK
- PARSE_ERROR → NETWORK
- CONTENT_FILTER → SAFETY
- UNKNOWN → NETWORK
"""

import pytest
from unittest.mock import MagicMock

from pdf2epub.core.executor import (
    Executor,
    ChainEntry,
    QuotaConfig,
)
from pdf2epub.core.work_unit import WorkUnit
from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier
from pdf2epub.core.types import ErrorType

from .fixtures.fake_llm import FakeLLMClient, FakeResponse, FakeErrorType


class SimpleProcessor:
    """Simple processor for testing."""
    name = "test"

    def build_prompt(self, content: str, context=None) -> str:
        return f"Process: {content}"

    def clean_response(self, response: str) -> str:
        return response.strip()

    def post_process(self, result: str, context=None) -> str:
        return result


def make_accepting_hooks():
    """Create hooks that always accept."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()
    validator = MagicMock()
    validator.name = "accepting"
    validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
    hooks._validators = [validator]
    return hooks


def make_large_model_chain(count: int = 20):
    """
    Many models in chain to prevent chain exhaustion.

    This ensures retry decisions are based on quota, not chain emptiness.
    With enough models, we can observe the full quota behavior.
    """
    return [
        ChainEntry(provider=f"p{i}", model=f"m{i}", mode="online")
        for i in range(count)
    ]


class TestRateLimitRespectsNetworkQuota:
    """
    Test 1: RATE_LIMIT errors should check NETWORK quota for retry decisions.

    Bug: can_retry(RATE_LIMIT) checks quotas[RATE_LIMIT],
    but apply_effect decrements quotas[NETWORK].

    This creates a situation where:
    - NETWORK quota can reach 0 (from apply_effect decrementing it)
    - But retries continue because can_retry checks RATE_LIMIT quota (which is still high)
    """

    def test_rate_limit_should_check_network_quota_not_rate_limit_quota(self):
        """
        RATE_LIMIT retries should be controlled by NETWORK quota.

        Setup:
        - NETWORK quota = 0 (should block all retries immediately)
        - RATE_LIMIT quota = 999 (decoy - should be ignored for retry decision)
        - Chain has 20 models (won't be exhausted)

        Expected (correct behavior):
        - Initial call fails with RATE_LIMIT
        - apply_effect decrements NETWORK quota (0 -> -1, clamped to 0)
        - can_retry should check NETWORK quota (0) -> no retry
        - Total calls = 1

        Bug behavior:
        - can_retry checks RATE_LIMIT quota (999) -> allows retry
        - Retries continue until chain exhaustion or circuit breaker
        - Total calls >> 1
        """
        fake_llm = FakeLLMClient()
        fake_llm.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.RATE_LIMIT,
            succeed_after_n_calls=0,  # Always fail
        ))

        quota_config = QuotaConfig(
            total=100,
            per_type={
                ErrorType.NETWORK: 0,       # Zero! Should block all retries
                ErrorType.RATE_LIMIT: 999,  # High but should be IGNORED
                # Include all error types to avoid default filling
                ErrorType.SAFETY: 999,
                ErrorType.VALIDATION: 999,
                ErrorType.TRUNCATION: 999,
                ErrorType.TIMEOUT: 999,
                ErrorType.CONTENT_FILTER: 999,
                ErrorType.PARSE_ERROR: 999,
                ErrorType.UNKNOWN: 999,
            }
        )

        executor = Executor(
            llm_client=fake_llm,
            model_chain=make_large_model_chain(20),
            processor=SimpleProcessor(),
            hooks=make_accepting_hooks(),
            quota_config=quota_config,
            max_workers=1,
            network_circuit_breaker_threshold=100,  # Disable circuit breaker
        )

        unit = WorkUnit(id="chapter_1", file_key="ch1", content="Test content")
        result = executor.execute([unit])

        assert "chapter_1" in result.failed

        # CRITICAL: With NETWORK quota = 0, should only call once (no retry)
        # Bug: checks RATE_LIMIT=999, retries many times
        call_count = fake_llm.call_count_for("chapter_1")
        assert call_count == 1, (
            f"Expected 1 call (NETWORK quota=0 blocks retry), got {call_count}. "
            f"BUG: can_retry() checks RATE_LIMIT quota (999) instead of NETWORK quota (0). "
            f"This is because RATE_LIMIT's quota_type is NETWORK, but can_retry looks at "
            f"quotas[error_type] not quotas[effect.quota_type]."
        )


class TestTimeoutRespectsNetworkQuota:
    """
    Test 2: TIMEOUT errors should check NETWORK quota for retry decisions.

    Same bug pattern as RATE_LIMIT - TIMEOUT's quota_type is NETWORK.
    """

    def test_timeout_should_check_network_quota_not_timeout_quota(self):
        """
        TIMEOUT retries should be controlled by NETWORK quota.

        Setup:
        - NETWORK quota = 0 (should block all retries immediately)
        - TIMEOUT quota = 999 (decoy - should be ignored for retry decision)

        Expected: 1 call (no retry because NETWORK=0)
        Bug: Many calls because can_retry checks TIMEOUT=999
        """
        fake_llm = FakeLLMClient()
        fake_llm.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.TIMEOUT,
            succeed_after_n_calls=0,
        ))

        quota_config = QuotaConfig(
            total=100,
            per_type={
                ErrorType.NETWORK: 0,      # Zero!
                ErrorType.TIMEOUT: 999,    # Should be IGNORED
                ErrorType.SAFETY: 999,
                ErrorType.VALIDATION: 999,
                ErrorType.TRUNCATION: 999,
                ErrorType.RATE_LIMIT: 999,
                ErrorType.CONTENT_FILTER: 999,
                ErrorType.PARSE_ERROR: 999,
                ErrorType.UNKNOWN: 999,
            }
        )

        executor = Executor(
            llm_client=fake_llm,
            model_chain=make_large_model_chain(20),
            processor=SimpleProcessor(),
            hooks=make_accepting_hooks(),
            quota_config=quota_config,
            max_workers=1,
            network_circuit_breaker_threshold=100,  # Disable circuit breaker
        )

        unit = WorkUnit(id="chapter_1", file_key="ch1", content="Test content")
        result = executor.execute([unit])

        assert "chapter_1" in result.failed

        call_count = fake_llm.call_count_for("chapter_1")
        assert call_count == 1, (
            f"Expected 1 call (NETWORK quota=0), got {call_count}. "
            f"BUG: can_retry() checks TIMEOUT quota instead of NETWORK quota."
        )


class TestParseErrorRespectsNetworkQuota:
    """
    Test 3: PARSE_ERROR errors should check NETWORK quota for retry decisions.

    PARSE_ERROR's quota_type is NETWORK (transient error, retry makes sense).
    But can_retry checks quotas[PARSE_ERROR] instead of quotas[NETWORK].
    """

    def test_parse_error_should_check_network_quota_not_parse_error_quota(self):
        """
        PARSE_ERROR retries should be controlled by NETWORK quota.

        Setup:
        - NETWORK quota = 0 (should block all retries immediately)
        - PARSE_ERROR quota = 999 (decoy - should be ignored for retry decision)

        Expected: 1 call (no retry because NETWORK=0)
        Bug: Many calls because can_retry checks PARSE_ERROR=999
        """
        fake_llm = FakeLLMClient()
        fake_llm.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.PARSE_ERROR,
            succeed_after_n_calls=0,
        ))

        quota_config = QuotaConfig(
            total=100,
            per_type={
                ErrorType.NETWORK: 0,       # Zero!
                ErrorType.PARSE_ERROR: 999, # Should be IGNORED
                ErrorType.SAFETY: 999,
                ErrorType.VALIDATION: 999,
                ErrorType.TRUNCATION: 999,
                ErrorType.RATE_LIMIT: 999,
                ErrorType.TIMEOUT: 999,
                ErrorType.CONTENT_FILTER: 999,
                ErrorType.UNKNOWN: 999,
            }
        )

        executor = Executor(
            llm_client=fake_llm,
            model_chain=make_large_model_chain(20),
            processor=SimpleProcessor(),
            hooks=make_accepting_hooks(),
            quota_config=quota_config,
            max_workers=1,
            network_circuit_breaker_threshold=100,  # Disable circuit breaker
        )

        unit = WorkUnit(id="chapter_1", file_key="ch1", content="Test content")
        result = executor.execute([unit])

        assert "chapter_1" in result.failed

        call_count = fake_llm.call_count_for("chapter_1")
        assert call_count == 1, (
            f"Expected 1 call (NETWORK quota=0), got {call_count}. "
            f"BUG: can_retry() checks PARSE_ERROR quota instead of NETWORK quota."
        )


class TestUnknownErrorRespectsNetworkQuota:
    """
    Test 4: UNKNOWN errors should check NETWORK quota for retry decisions.

    UNKNOWN's quota_type is NETWORK (default fallback).
    But can_retry checks quotas[UNKNOWN] instead of quotas[NETWORK].
    """

    def test_unknown_should_check_network_quota_not_unknown_quota(self):
        """
        UNKNOWN retries should be controlled by NETWORK quota.

        Setup:
        - NETWORK quota = 0 (should block all retries immediately)
        - UNKNOWN quota = 999 (decoy - should be ignored for retry decision)

        Expected: 1 call (no retry because NETWORK=0)
        Bug: Many calls because can_retry checks UNKNOWN=999
        """
        fake_llm = FakeLLMClient()
        fake_llm.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.UNKNOWN,
            succeed_after_n_calls=0,
        ))

        quota_config = QuotaConfig(
            total=100,
            per_type={
                ErrorType.NETWORK: 0,      # Zero!
                ErrorType.UNKNOWN: 999,    # Should be IGNORED
                ErrorType.SAFETY: 999,
                ErrorType.VALIDATION: 999,
                ErrorType.TRUNCATION: 999,
                ErrorType.RATE_LIMIT: 999,
                ErrorType.TIMEOUT: 999,
                ErrorType.CONTENT_FILTER: 999,
                ErrorType.PARSE_ERROR: 999,
            }
        )

        executor = Executor(
            llm_client=fake_llm,
            model_chain=make_large_model_chain(20),
            processor=SimpleProcessor(),
            hooks=make_accepting_hooks(),
            quota_config=quota_config,
            max_workers=1,
            network_circuit_breaker_threshold=100,  # Disable circuit breaker
        )

        unit = WorkUnit(id="chapter_1", file_key="ch1", content="Test content")
        result = executor.execute([unit])

        assert "chapter_1" in result.failed

        call_count = fake_llm.call_count_for("chapter_1")
        assert call_count == 1, (
            f"Expected 1 call (NETWORK quota=0), got {call_count}. "
            f"BUG: can_retry() checks UNKNOWN quota instead of NETWORK quota."
        )


class TestQuotaStealingFromNetworkPool:
    """
    Test 5: RATE_LIMIT errors vs NETWORK errors have asymmetric retry behavior.

    This test demonstrates the inconsistency in quota semantics:
    - RATE_LIMIT: can_retry checks RATE_LIMIT quota, apply_effect decrements NETWORK
    - NETWORK: can_retry checks NETWORK quota, apply_effect decrements NETWORK

    Result: RATE_LIMIT gets way more retries than NETWORK for the same total_quota,
    even though they both "use" the NETWORK pool.

    The asymmetry is:
    - RATE_LIMIT: checked against RATE_LIMIT quota (999) but decrements NETWORK
    - NETWORK: checked against NETWORK quota (3) and decrements NETWORK

    If the system were consistent, both should be limited by the same pool.
    """

    def test_rate_limit_gets_more_retries_than_network_despite_same_quota_pool(self):
        """
        RATE_LIMIT and NETWORK should have symmetric retry behavior.

        Setup:
        - NETWORK quota = 3 (real limit for NETWORK errors)
        - RATE_LIMIT quota = 999 (what RATE_LIMIT actually checks)
        - Same total_quota, same chain
        - Both error types should decrement NETWORK pool

        Expected (correct behavior):
        - Both RATE_LIMIT and NETWORK should get exactly 3 retries
        - Because both use NETWORK quota pool

        Bug behavior:
        - NETWORK gets 3 retries (correct - checks NETWORK=3)
        - RATE_LIMIT gets way more retries (wrong - checks RATE_LIMIT=999)

        This test verifies both should have the same retry count.
        """
        # Test NETWORK error behavior
        fake_llm_network = FakeLLMClient()
        fake_llm_network.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.NETWORK,
            succeed_after_n_calls=0,  # Always fail
        ))

        quota_config = QuotaConfig(
            total=100,
            per_type={
                ErrorType.NETWORK: 3,       # Real limit
                ErrorType.RATE_LIMIT: 999,  # Decoy for RATE_LIMIT
                ErrorType.SAFETY: 999,
                ErrorType.VALIDATION: 999,
                ErrorType.TRUNCATION: 999,
                ErrorType.TIMEOUT: 999,
                ErrorType.CONTENT_FILTER: 999,
                ErrorType.PARSE_ERROR: 999,
                ErrorType.UNKNOWN: 999,
            }
        )

        executor_network = Executor(
            llm_client=fake_llm_network,
            model_chain=make_large_model_chain(20),
            processor=SimpleProcessor(),
            hooks=make_accepting_hooks(),
            quota_config=quota_config,
            max_workers=1,
            network_circuit_breaker_threshold=100,
        )

        unit = WorkUnit(id="chapter_1", file_key="ch1", content="Test content")
        executor_network.execute([unit])
        network_call_count = fake_llm_network.call_count_for("chapter_1")

        # Test RATE_LIMIT error behavior
        fake_llm_ratelimit = FakeLLMClient()
        fake_llm_ratelimit.set_response("chapter_1", FakeResponse(
            error=FakeErrorType.RATE_LIMIT,
            succeed_after_n_calls=0,  # Always fail
        ))

        executor_ratelimit = Executor(
            llm_client=fake_llm_ratelimit,
            model_chain=make_large_model_chain(20),
            processor=SimpleProcessor(),
            hooks=make_accepting_hooks(),
            quota_config=quota_config,  # Same quota config!
            max_workers=1,
            network_circuit_breaker_threshold=100,
        )

        unit2 = WorkUnit(id="chapter_1", file_key="ch1", content="Test content")
        executor_ratelimit.execute([unit2])
        ratelimit_call_count = fake_llm_ratelimit.call_count_for("chapter_1")

        # CRITICAL: Both should have the same retry count
        # Because both use NETWORK quota pool (quota_type=NETWORK)
        assert ratelimit_call_count == network_call_count, (
            f"RATE_LIMIT got {ratelimit_call_count} calls, NETWORK got {network_call_count} calls. "
            f"BUG: Both should have the same retry count because both use NETWORK quota pool. "
            f"But RATE_LIMIT checks its own quota (999) while NETWORK checks NETWORK quota (3). "
            f"This is the quota semantic inconsistency - can_retry uses error_type, "
            f"apply_effect uses effect.quota_type."
        )
