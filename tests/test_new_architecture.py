"""
Tests for the new Pipeline + Executor + Hooks architecture.

Tests cover:
- Hooks: pre-processors, transformers, validators, error classifier
- Executor: state management, re-queuing, concurrent execution
- Phase: loading, execution, no aggregation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import time

# === Hooks Tests ===

class TestHooksProtocol:
    """Test hook protocols and implementations."""

    def test_error_type_enum(self):
        """Test ErrorType enum values."""
        from pdf2epub.core.hooks import ErrorType

        assert ErrorType.SAFETY.value == "safety"
        assert ErrorType.NETWORK.value == "network"
        assert ErrorType.VALIDATION.value == "validation"

    def test_error_effect_dataclass(self):
        """Test ErrorEffect dataclass."""
        from pdf2epub.core.hooks import ErrorEffect, ErrorType

        effect = ErrorEffect(
            remove_current_model=True,
            remove_provider=True,
            quota_type=ErrorType.SAFETY
        )

        assert effect.remove_current_model is True
        assert effect.remove_provider is True
        assert effect.quota_type == ErrorType.SAFETY


class TestPreProcessors:
    """Test pre-processor implementations."""

    def test_empty_content_filter(self):
        """Test EmptyContentFilter skips empty content."""
        from pdf2epub.core.hooks import EmptyContentFilter

        filter = EmptyContentFilter()
        mock_context = Mock()

        # Empty content
        result = filter.check("key1", "", mock_context)
        assert result.should_process is False
        assert "Empty" in result.skip_reason

        # Whitespace only
        result = filter.check("key2", "   \n\t  ", mock_context)
        assert result.should_process is False

        # Normal content
        result = filter.check("key3", "Hello world", mock_context)
        assert result.should_process is True


class TestTransformers:
    """Test transformer implementations."""

    def test_remove_artifacts_transformer(self):
        """Test RemoveArtifactsTransformer removes LLM artifacts."""
        from pdf2epub.core.hooks import RemoveArtifactsTransformer

        transformer = RemoveArtifactsTransformer()

        # Test code block removal
        result = transformer.transform(
            "key1", "original",
            "```markdown\nHello world\n```"
        )
        assert "```" not in result
        assert "Hello world" in result

        # Test "Here is the..." removal
        result = transformer.transform(
            "key2", "original",
            "Here is the translation:\nHello world"
        )
        assert "Here is the" not in result


class TestValidators:
    """Test validator implementations."""

    def test_length_ratio_validator_screener(self):
        """Test LengthRatioValidator as screener."""
        from pdf2epub.core.hooks import LengthRatioValidator

        validator = LengthRatioValidator(
            min_ratio=0.5, max_ratio=2.0,
            role="screener", context_ready=True
        )

        # Within ratio - should pass with context_ready
        result = validator.validate("key1", "Hello world", "Hello!")
        assert result.accepted is True
        assert result.context_ready is True

        # Too short - screener doesn't reject, just continues
        result = validator.validate("key2", "Hello world" * 10, "Hi")
        assert result.accepted is True  # Screener doesn't reject
        assert result.context_ready is False

    def test_length_ratio_validator_final(self):
        """Test LengthRatioValidator as final."""
        from pdf2epub.core.hooks import LengthRatioValidator

        validator = LengthRatioValidator(
            min_ratio=0.5, max_ratio=2.0,
            role="final"
        )

        # Too short - final rejects
        result = validator.validate("key1", "Hello world" * 10, "Hi")
        assert result.accepted is False


class TestErrorClassifier:
    """Test error classifier."""

    def test_classify_safety_error(self):
        """Test safety error classification."""
        from pdf2epub.core.hooks import DefaultErrorClassifier, ErrorType

        classifier = DefaultErrorClassifier()

        error = Exception("Content blocked by safety filter")
        error_type = classifier.classify(error)
        assert error_type == ErrorType.SAFETY

        effect = classifier.get_effect(error_type)
        assert effect.remove_provider is True

    def test_classify_network_error(self):
        """Test network error classification."""
        from pdf2epub.core.hooks import DefaultErrorClassifier, ErrorType

        classifier = DefaultErrorClassifier()

        # Test pure network error (503 Service Unavailable)
        error = Exception("503 Service Unavailable")
        error_type = classifier.classify(error)
        assert error_type == ErrorType.NETWORK

        # Network errors use fail-fast design: remove current model and try next
        # Bottom layer handles transient retries, upper layer switches models
        effect = classifier.get_effect(error_type)
        assert effect.remove_current_model is True

    def test_classify_timeout_error(self):
        """Test timeout error classification (more specific than network)."""
        from pdf2epub.core.hooks import DefaultErrorClassifier, ErrorType

        classifier = DefaultErrorClassifier()

        error = Exception("Connection timeout after 30s")
        error_type = classifier.classify(error)
        assert error_type == ErrorType.TIMEOUT

        # Timeout errors use fail-fast design: remove current model and try next
        # Bottom layer handles transient retries, upper layer switches models
        effect = classifier.get_effect(error_type)
        assert effect.remove_current_model is True


class TestCompositeHooks:
    """Test CompositeHooks integration."""

    def test_pre_process_chain(self):
        """Test pre-processing chain stops at first skip."""
        from pdf2epub.core.hooks import (
            CompositeHooks, EmptyContentFilter, MinLengthFilter
        )

        hooks = CompositeHooks(
            pre_processors=[
                EmptyContentFilter(),
                MinLengthFilter(min_chars=10),
            ]
        )

        mock_context = Mock()

        # Empty content - first filter catches
        result = hooks.pre_process("key1", "", mock_context)
        assert result.should_process is False
        assert "Empty" in result.skip_reason

    def test_post_process_transform_and_validate(self):
        """Test post-processing transforms then validates."""
        from pdf2epub.core.hooks import (
            CompositeHooks, StripTransformer, LengthRatioValidator
        )

        hooks = CompositeHooks(
            transformers=[StripTransformer()],
            validators=[
                LengthRatioValidator(
                    min_ratio=0.1, max_ratio=10.0,
                    role="final"
                )
            ],
        )

        # Should strip and validate
        transformed, result = hooks.post_process(
            "key1", "original", "   result   "
        )
        assert transformed == "result"  # Stripped
        assert result.accepted is True


# === Executor Tests ===

class TestChainEntry:
    """Test ChainEntry."""

    def test_chain_entry_creation(self):
        """Test creating ChainEntry."""
        from pdf2epub.core.executor import ChainEntry

        entry = ChainEntry(
            provider="gemini",
            model="gemini-2.0-flash",
            mode="online"
        )

        assert entry.provider == "gemini"
        assert entry.mode == "online"

        dict_repr = entry.to_dict()
        assert dict_repr["provider"] == "gemini"
        assert dict_repr["model"] == "gemini-2.0-flash"


class TestUnitState:
    """Test UnitState."""

    def test_unit_state_can_retry(self):
        """Test can_retry logic."""
        from pdf2epub.core.executor import UnitState, ChainEntry
        from pdf2epub.core.hooks import ErrorType

        chain = [ChainEntry("gemini", "gemini-2.0-flash", "online")]
        state = UnitState(
            chain=chain,
            total_quota=3,
            quotas={ErrorType.NETWORK: 2, ErrorType.VALIDATION: 1}
        )

        assert state.can_retry(ErrorType.NETWORK) is True
        assert state.can_retry(ErrorType.VALIDATION) is True

        # Exhaust validation quota
        state.quotas[ErrorType.VALIDATION] = 0
        assert state.can_retry(ErrorType.VALIDATION) is False

    def test_unit_state_apply_effect(self):
        """Test applying error effects."""
        from pdf2epub.core.executor import UnitState, ChainEntry
        from pdf2epub.core.hooks import ErrorType, ErrorEffect

        chain = [
            ChainEntry("gemini", "gemini-2.0-flash", "batch"),
            ChainEntry("gemini", "gemini-2.0-flash", "online"),
            ChainEntry("deepseek", "deepseek-chat", "online"),
        ]
        state = UnitState(
            chain=chain,
            total_quota=5,
            quotas={ErrorType.SAFETY: 999, ErrorType.NETWORK: 3}
        )

        # Safety effect removes provider
        effect = ErrorEffect(
            remove_current_model=True,
            remove_provider=True,
            quota_type=ErrorType.SAFETY
        )
        state.apply_effect(effect, chain[0])

        # Gemini entries should be removed
        assert len(state.chain) == 1
        assert state.chain[0].provider == "deepseek"

    def test_unit_state_longest_fallback(self):
        """Test longest fallback tracking."""
        from pdf2epub.core.executor import UnitState, ChainEntry
        from pdf2epub.core.hooks import ErrorType

        state = UnitState(
            chain=[ChainEntry("gemini", "gemini-2.0-flash", "online")],
            total_quota=3,
            quotas={ErrorType.NETWORK: 2}
        )

        # Record attempts
        state.record_attempt("short")
        state.record_attempt("this is a longer result")
        state.record_attempt("medium len")

        longest = state.get_longest()
        assert longest == "this is a longer result"


class TestQuotaConfig:
    """Test QuotaConfig."""

    def test_quota_config_defaults(self):
        """Test default quota configuration."""
        from pdf2epub.core.executor import QuotaConfig
        from pdf2epub.core.hooks import ErrorType

        config = QuotaConfig()

        assert config.total == 5
        assert config.per_type[ErrorType.NETWORK] == 3
        assert config.per_type[ErrorType.VALIDATION] == 1

    def test_create_quotas_copies(self):
        """Test create_quotas returns a copy."""
        from pdf2epub.core.executor import QuotaConfig

        config = QuotaConfig()
        quotas1 = config.create_quotas()
        quotas2 = config.create_quotas()

        assert quotas1 is not quotas2


# === Phase Tests ===

class TestWorkUnit:
    """Test WorkUnit."""

    def test_work_unit_is_virtual(self):
        """Test virtual unit detection via is_sub_key function."""
        from pdf2epub.core.types import is_sub_key

        # Regular unit - not a .sub key
        assert is_sub_key("chapter_1.part0") is False

        # Virtual unit (from dynamic splitting) - contains .sub
        assert is_sub_key("chapter_1.part0.sub0") is True
        assert is_sub_key("chapter_1.sub0.sub1") is True


class TestPartBasedLoader:
    """Test PartBasedLoader."""

    def test_load_units_from_directory(self):
        """Test loading units from a directory."""
        from pdf2epub.core.phase import PartBasedLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "chapter_1.part0.md").write_text("Content 0")
            (tmppath / "chapter_1.part1.md").write_text("Content 1")
            (tmppath / "chapter_2.part0.md").write_text("Content 2")

            loader = PartBasedLoader()
            units = loader.load_units(tmppath, "*.md")

            assert len(units) == 3
            assert units[0].id == "chapter_1.part0"
            assert units[0].content == "Content 0"


class TestPhase:
    """Test Phase."""

    def test_phase_filter_completed(self):
        """Test Phase filters completed units on resume."""
        from pdf2epub.core.phase import Phase, PartBasedLoader
        from pdf2epub.core.executor import WorkUnit

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            input_dir = tmppath / "input"
            output_dir = tmppath / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create input files
            (input_dir / "unit1.md").write_text("Content 1")
            (input_dir / "unit2.md").write_text("Content 2")
            (input_dir / "unit3.md").write_text("Content 3")

            # Simulate unit1 already completed - Phase checks validated/ subdirectory
            validated_dir = output_dir / "validated"
            validated_dir.mkdir()
            (validated_dir / "unit1.md").write_text("Completed 1")

            # Create mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.process_all.return_value = Mock(
                total=2, completed=2, failed=0, results={}
            )

            phase = Phase(
                name="test",
                input_dir=input_dir,
                output_dir=output_dir,
                pipeline=mock_pipeline,
            )

            # Load and filter
            loader = PartBasedLoader()
            units = loader.load_units(input_dir)
            filtered = phase._filter_completed(units)

            # unit1 should be filtered out
            assert len(filtered) == 2
            assert all(u.id != "unit1" for u in filtered)


# === Integration Tests ===

class TestHooksExecutorIntegration:
    """Test integration between hooks and executor."""

    def test_error_effect_applied_to_state(self):
        """Test that error classification affects unit state."""
        from pdf2epub.core.hooks import DefaultErrorClassifier, ErrorType
        from pdf2epub.core.executor import UnitState, ChainEntry

        classifier = DefaultErrorClassifier()

        # Create state with multiple providers
        chain = [
            ChainEntry("gemini", "flash", "online"),
            ChainEntry("deepseek", "chat", "online"),
        ]
        state = UnitState(
            chain=chain,
            total_quota=5,
            quotas={ErrorType.SAFETY: 999, ErrorType.NETWORK: 3}
        )

        # Simulate safety error
        error = Exception("Content blocked by safety filter")
        error_type = classifier.classify(error)
        effect = classifier.get_effect(error_type)

        # Apply effect
        state.apply_effect(effect, chain[0])

        # Gemini should be removed
        assert len(state.chain) == 1
        assert state.chain[0].provider == "deepseek"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
