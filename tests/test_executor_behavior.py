"""
Executor 行为测试 - 用 FakeClient 验证状态机不变量。

这些测试不花钱，但能覆盖 80% 的真实 bug：
- screener_passed 透传
- fallback_used 透传
- quota 扣减与 can_retry 一致
- chain 推进规则
- circuit breaker 状态
- 结果完整性

设计原则：
- 用 FakeLLMClient 完全匹配真实接口
- 用真实的 chapter 内容
- 验证 ExecutionResult 的状态机不变量
"""

import pytest
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock

from pdf2epub.core.executor import (
    Executor,
    ChainEntry,
    ExecutionResult,
    QuotaConfig,
)
from pdf2epub.core.work_unit import WorkUnit
from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier
from pdf2epub.core.types import ErrorType

from .fixtures.fake_llm import FakeLLMClient, FakeResponse, FakeErrorType
from .fixtures.sample_content import (
    SHORT_CHAPTER, MEDIUM_CHAPTER, LONG_CHAPTER,
    FRONT_MATTER, IMAGE_ONLY, get_chapters,
)


# ============================================================
# Processor that matches real interface
# ============================================================

class TestProcessor:
    """测试用 Processor，匹配真实接口."""

    name = "test_processor"

    def build_prompt(self, content: str, context: Any = None) -> str:
        return f"Please process:\n```\n{content}\n```"

    def clean_response(self, response: str) -> str:
        return response.strip()

    def post_process(self, result: str, context: Any = None) -> str:
        return result


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def single_model_chain():
    """单模型 chain."""
    return [ChainEntry(provider="fake", model="fake-model", mode="online")]


@pytest.fixture
def multi_model_chain():
    """多模型 chain，测试 fallback."""
    return [
        ChainEntry(provider="provider1", model="model1", mode="online"),
        ChainEntry(provider="provider2", model="model2", mode="online"),
        ChainEntry(provider="provider3", model="model3", mode="online"),
    ]


@pytest.fixture
def accepting_hooks():
    """总是接受的 hooks."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()

    validator = MagicMock()
    validator.name = "accepting_validator"
    validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
    hooks._validators = [validator]

    return hooks


@pytest.fixture
def context_ready_hooks():
    """返回 context_ready=True 的 hooks."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()

    validator = MagicMock()
    validator.name = "context_ready_validator"
    validator.validate.return_value = MagicMock(accepted=True, context_ready=True)
    hooks._validators = [validator]

    return hooks


@pytest.fixture
def rejecting_hooks():
    """总是拒绝的 hooks."""
    hooks = CompositeHooks()
    hooks._error_classifier = DefaultErrorClassifier()

    validator = MagicMock()
    validator.name = "rejecting_validator"
    validator.validate.return_value = MagicMock(accepted=False, context_ready=False)
    hooks._validators = [validator]

    return hooks


def create_executor(
    chain: List[ChainEntry],
    hooks: CompositeHooks,
    llm_client: Optional[FakeLLMClient] = None,
    quota_config: Optional[QuotaConfig] = None,
    **kwargs,
) -> Executor:
    """创建带有 fake 组件的 executor."""
    return Executor(
        llm_client=llm_client or FakeLLMClient(),
        model_chain=chain,
        processor=TestProcessor(),
        hooks=hooks,
        quota_config=quota_config,
        max_workers=1,
        **kwargs,
    )


def create_units(contents: Dict[str, str]) -> List[WorkUnit]:
    """从内容字典创建 WorkUnit 列表."""
    return [
        WorkUnit(id=key, file_key=key, content=content)
        for key, content in contents.items()
    ]


# ============================================================
# Test: screener_passed 透传
# ============================================================

class TestScreenerPassedTransparency:
    """验证 screener_passed 在整个数据流中正确透传."""

    def test_context_ready_adds_to_screener_passed(self, single_model_chain, context_ready_hooks):
        """context_ready=True 的 unit 必须出现在 screener_passed."""
        fake_llm = FakeLLMClient(default_response="processed content")
        executor = create_executor(single_model_chain, context_ready_hooks, fake_llm)

        units = create_units({
            "chapter_1": SHORT_CHAPTER,
            "chapter_2": MEDIUM_CHAPTER,
        })

        result = executor.execute(units)

        # Both should be in screener_passed since context_ready=True
        assert "chapter_1" in result.screener_passed, (
            f"chapter_1 should be in screener_passed, got: {result.screener_passed}"
        )
        assert "chapter_2" in result.screener_passed

    def test_non_context_ready_not_in_screener_passed(self, single_model_chain, accepting_hooks):
        """context_ready=False 的 unit 不应该在 screener_passed."""
        fake_llm = FakeLLMClient(default_response="processed content")
        executor = create_executor(single_model_chain, accepting_hooks, fake_llm)

        units = create_units({"chapter_1": SHORT_CHAPTER})
        result = executor.execute(units)

        # Should complete but NOT be in screener_passed
        assert "chapter_1" in result.completed
        assert "chapter_1" not in result.screener_passed

    def test_screener_passed_propagates_through_execute(self, single_model_chain, context_ready_hooks):
        """screener_passed 必须从 _process_online 传递到 execute() 返回值.

        这是之前 P0 bug 的回归测试。
        """
        fake_llm = FakeLLMClient(default_response="processed")
        executor = create_executor(single_model_chain, context_ready_hooks, fake_llm)

        unit = WorkUnit(id="test_unit", file_key="test", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # THE KEY ASSERTION
        assert result.screener_passed, "screener_passed must not be empty"
        assert "test_unit" in result.screener_passed


# ============================================================
# Test: fallback_used 透传
# ============================================================

class TestFallbackUsedTransparency:
    """验证 fallback_used 在整个数据流中正确透传."""

    def test_longest_fallback_adds_to_fallback_used(self, single_model_chain, rejecting_hooks):
        """使用 longest fallback 的 unit 必须出现在 fallback_used."""
        fake_llm = FakeLLMClient(default_response="fallback content")

        # Low quota = immediate exhaustion = fallback
        quota = QuotaConfig(total=1, per_type={ErrorType.VALIDATION: 1})
        executor = create_executor(single_model_chain, rejecting_hooks, fake_llm, quota)

        unit = WorkUnit(id="chapter_1", file_key="ch1", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # Should use fallback (not fail)
        assert "chapter_1" in result.fallback_used, (
            f"chapter_1 should be in fallback_used, got: {result.fallback_used}"
        )
        assert "chapter_1" in result.completed

    def test_fallback_used_propagates_through_execute(self, single_model_chain, rejecting_hooks):
        """fallback_used 必须从 _process_online 传递到 execute() 返回值."""
        fake_llm = FakeLLMClient(default_response="fallback content")
        quota = QuotaConfig(total=1, per_type={ErrorType.VALIDATION: 1})
        executor = create_executor(single_model_chain, rejecting_hooks, fake_llm, quota)

        unit = WorkUnit(id="fallback_test", file_key="test", content=MEDIUM_CHAPTER)
        result = executor.execute([unit])

        # THE KEY ASSERTION
        assert result.fallback_used, "fallback_used must not be empty when fallback triggered"
        assert "fallback_test" in result.fallback_used


# ============================================================
# Test: Quota 一致性
# ============================================================

class TestQuotaConsistency:
    """验证 quota 扣减与 can_retry 一致."""

    def test_retry_on_validation_failure(self, single_model_chain):
        """验证失败后应该重试（如果 quota 允许）."""
        fake_llm = FakeLLMClient(default_response="response")

        # Dynamic validator: reject first, accept second
        call_count = [0]
        def dynamic_validate(key, orig, result):
            call_count[0] += 1
            return MagicMock(accepted=call_count[0] > 1, context_ready=False)

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "dynamic"
        validator.validate = dynamic_validate
        hooks._validators = [validator]

        # Enough quota for 1 retry
        quota = QuotaConfig(total=3, per_type={ErrorType.VALIDATION: 2})
        executor = create_executor(single_model_chain, hooks, fake_llm, quota)

        unit = WorkUnit(id="chapter_1", file_key="ch1", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # Should succeed after retry
        assert "chapter_1" in result.completed
        assert call_count[0] == 2, f"Expected 2 validation calls, got {call_count[0]}"

    def test_fallback_after_quota_exhaustion(self, single_model_chain, rejecting_hooks):
        """quota 耗尽后应该使用 fallback."""
        fake_llm = FakeLLMClient(default_response="fallback content")

        # Quota of 1 = no retry = immediate fallback
        quota = QuotaConfig(total=1, per_type={ErrorType.VALIDATION: 1})
        executor = create_executor(single_model_chain, rejecting_hooks, fake_llm, quota)

        unit = WorkUnit(id="chapter_1", file_key="ch1", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # Should use fallback, not fail
        assert "chapter_1" in result.completed
        assert "chapter_1" in result.fallback_used
        assert "chapter_1" not in result.failed


# ============================================================
# Test: Chain 推进
# ============================================================

class TestChainProgression:
    """验证 model chain 推进规则."""

    def test_network_error_tries_next_model(self, multi_model_chain, accepting_hooks):
        """网络错误应该尝试下一个 model."""
        fake_llm = FakeLLMClient()

        # First model fails, second succeeds
        fake_llm.set_response("chapter_1", FakeResponse(
            content="success from model2",
            error=FakeErrorType.NETWORK,
            succeed_after_n_calls=1,  # Fail first call, succeed second
        ))

        quota = QuotaConfig(total=5, per_type={ErrorType.NETWORK: 3})
        executor = create_executor(multi_model_chain, accepting_hooks, fake_llm, quota)

        unit = WorkUnit(id="chapter_1", file_key="ch1", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # Should succeed after model switch
        assert "chapter_1" in result.completed
        assert fake_llm.call_count_for("chapter_1") >= 2

    def test_safety_error_fails_unit(self, single_model_chain, accepting_hooks):
        """Safety 错误应该导致 unit 失败."""
        fake_llm = FakeLLMClient()
        fake_llm.set_response("chapter_1", FakeResponse(error=FakeErrorType.SAFETY))

        executor = create_executor(single_model_chain, accepting_hooks, fake_llm)

        unit = WorkUnit(id="chapter_1", file_key="ch1", content=SHORT_CHAPTER)
        result = executor.execute([unit])

        # Should fail (safety blocked)
        assert "chapter_1" in result.failed or "chapter_1" in result.safety_blocked


# ============================================================
# Test: 结果完整性
# ============================================================

class TestResultCompleteness:
    """验证 ExecutionResult 的完整性."""

    def test_all_units_accounted_for(self, single_model_chain, accepting_hooks):
        """所有 unit 必须在 completed、failed 或 skipped 中."""
        fake_llm = FakeLLMClient(default_response="processed")
        executor = create_executor(single_model_chain, accepting_hooks, fake_llm)

        chapters = get_chapters(count=10)
        units = create_units(chapters)

        result = executor.execute(units)

        all_ids = {u.id for u in units}
        accounted = result.completed | result.failed | result.skipped

        assert all_ids == accounted, f"Units not accounted: {all_ids - accounted}"

    def test_results_dict_matches_completed(self, single_model_chain, accepting_hooks):
        """results dict 的 keys 应该与 completed 一致."""
        fake_llm = FakeLLMClient(default_response="processed")
        executor = create_executor(single_model_chain, accepting_hooks, fake_llm)

        units = create_units(get_chapters(count=5))
        result = executor.execute(units)

        for unit_id in result.completed:
            assert unit_id in result.results, f"{unit_id} completed but not in results"

    def test_mixed_success_and_failure(self, single_model_chain):
        """混合成功和失败的情况."""
        fake_llm = FakeLLMClient(default_response="success")
        # One unit will fail with safety error
        fake_llm.set_response("chapter_2", FakeResponse(error=FakeErrorType.SAFETY))

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()
        validator = MagicMock()
        validator.name = "accepting"
        validator.validate.return_value = MagicMock(accepted=True, context_ready=False)
        hooks._validators = [validator]

        executor = create_executor(single_model_chain, hooks, fake_llm)

        units = create_units({
            "chapter_1": SHORT_CHAPTER,
            "chapter_2": MEDIUM_CHAPTER,  # Will fail
            "chapter_3": LONG_CHAPTER,
        })

        result = executor.execute(units)

        # chapter_1 and chapter_3 should succeed
        assert "chapter_1" in result.completed
        assert "chapter_3" in result.completed
        # chapter_2 should fail
        assert "chapter_2" in result.failed or "chapter_2" in result.safety_blocked


# ============================================================
# Test: FakeLLMClient 自身的正确性
# ============================================================

class TestFakeLLMClient:
    """验证 FakeLLMClient 行为正确."""

    def test_default_response(self):
        """默认响应."""
        fake = FakeLLMClient(default_response="default")
        result = fake.generate("prompt", operation_name="test:unit1")
        assert result == "default"

    def test_custom_response_per_unit(self):
        """按 unit 自定义响应."""
        fake = FakeLLMClient()
        fake.set_response("unit1", FakeResponse(content="custom for unit1"))
        fake.set_response("unit2", FakeResponse(content="custom for unit2"))

        assert fake.generate("p", operation_name="test:unit1") == "custom for unit1"
        assert fake.generate("p", operation_name="test:unit2") == "custom for unit2"

    def test_error_injection(self):
        """错误注入."""
        fake = FakeLLMClient()
        fake.set_response("bad_unit", FakeResponse(error=FakeErrorType.NETWORK))

        with pytest.raises(Exception) as exc_info:
            fake.generate("p", operation_name="test:bad_unit")

        assert "network" in str(exc_info.value).lower()

    def test_retry_behavior(self):
        """前 N 次失败，之后成功."""
        fake = FakeLLMClient()
        fake.set_response("flaky", FakeResponse(
            content="finally success",
            error=FakeErrorType.NETWORK,
            succeed_after_n_calls=2,
        ))

        # First 2 calls fail
        with pytest.raises(Exception):
            fake.generate("p", operation_name="test:flaky")
        with pytest.raises(Exception):
            fake.generate("p", operation_name="test:flaky")

        # Third call succeeds
        result = fake.generate("p", operation_name="test:flaky")
        assert result == "finally success"

    def test_call_history(self):
        """调用历史记录."""
        fake = FakeLLMClient()
        fake.generate("prompt1", operation_name="test:unit1")
        fake.generate("prompt2", operation_name="test:unit2")
        fake.generate("prompt3", operation_name="test:unit1")

        assert fake.call_count_for("unit1") == 2
        assert fake.call_count_for("unit2") == 1
        assert fake.was_called("unit1")
        assert not fake.was_called("unit3")
