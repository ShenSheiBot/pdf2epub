"""
Tests for batch state persistence.

Tests the PersistentBatchState class and its integration with the Executor.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf2epub.core.executor.batch_state import (
    PersistentBatchState,
    is_safety_error,
    SAFETY_KEYWORDS,
)


class TestPersistentBatchState:
    """Tests for PersistentBatchState dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        state = PersistentBatchState(
            active_job_name="test-job-123",
            processing_keys=["key1", "key2"],
            batch_metadata={"key1": {"file_key": "chapter_1"}},
            retry_count=1,
            safety_blocked_keys=["key3"],
            batch_provider="gemini",
            batch_model="gemini-3-flash",
        )
        d = state.to_dict()

        assert d["active_job_name"] == "test-job-123"
        assert d["processing_keys"] == ["key1", "key2"]
        assert d["batch_metadata"] == {"key1": {"file_key": "chapter_1"}}
        assert d["retry_count"] == 1
        assert d["safety_blocked_keys"] == ["key3"]
        assert d["batch_provider"] == "gemini"
        assert d["batch_model"] == "gemini-3-flash"

    def test_from_dict(self):
        """Test creation from dict."""
        d = {
            "active_job_name": "test-job-456",
            "processing_keys": ["a", "b"],
            "batch_metadata": {},
            "retry_count": 2,
            "safety_blocked_keys": [],
            "batch_provider": "vertex",
            "batch_model": "gemini-pro",
        }
        state = PersistentBatchState.from_dict(d)

        assert state.active_job_name == "test-job-456"
        assert state.processing_keys == ["a", "b"]
        assert state.retry_count == 2
        assert state.batch_provider == "vertex"

    def test_from_dict_missing_fields(self):
        """Test from_dict handles missing fields gracefully."""
        d = {"active_job_name": "job-1"}
        state = PersistentBatchState.from_dict(d)

        assert state.active_job_name == "job-1"
        assert state.processing_keys == []
        assert state.batch_metadata == {}
        assert state.retry_count == 0
        assert state.safety_blocked_keys == []

    def test_save_and_load(self, tmp_path: Path):
        """Test save/load roundtrip."""
        state_file = tmp_path / "batch_state.json"

        original = PersistentBatchState(
            active_job_name="job-roundtrip",
            processing_keys=["k1", "k2", "k3"],
            batch_metadata={"k1": {"content": "hello"}},
            retry_count=0,
        )
        original.save(state_file)

        # Verify file exists and is valid JSON
        assert state_file.exists()
        with open(state_file) as f:
            data = json.load(f)
        assert data["active_job_name"] == "job-roundtrip"

        # Load and verify
        loaded = PersistentBatchState.load(state_file)
        assert loaded is not None
        assert loaded.active_job_name == "job-roundtrip"
        assert loaded.processing_keys == ["k1", "k2", "k3"]
        assert loaded.batch_metadata == {"k1": {"content": "hello"}}

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """Test save creates parent directories if needed."""
        state_file = tmp_path / "deep" / "nested" / "batch_state.json"

        state = PersistentBatchState(active_job_name="nested-job")
        state.save(state_file)

        assert state_file.exists()

    def test_load_nonexistent(self, tmp_path: Path):
        """Test load returns None for nonexistent file."""
        state_file = tmp_path / "does_not_exist.json"

        result = PersistentBatchState.load(state_file)
        assert result is None

    def test_load_corrupted(self, tmp_path: Path):
        """Test load handles corrupted JSON gracefully."""
        state_file = tmp_path / "corrupted.json"
        state_file.write_text("not valid json {{{")

        result = PersistentBatchState.load(state_file)
        assert result is None

    def test_clear(self, tmp_path: Path):
        """Test clear removes state file."""
        state_file = tmp_path / "to_clear.json"

        # Create file
        state = PersistentBatchState(active_job_name="to-clear")
        state.save(state_file)
        assert state_file.exists()

        # Clear it
        PersistentBatchState.clear(state_file)
        assert not state_file.exists()

    def test_clear_nonexistent(self, tmp_path: Path):
        """Test clear handles nonexistent file gracefully."""
        state_file = tmp_path / "never_existed.json"

        # Should not raise
        PersistentBatchState.clear(state_file)

    def test_has_active_job(self):
        """Test has_active_job method."""
        state_with_job = PersistentBatchState(active_job_name="active")
        state_without_job = PersistentBatchState(active_job_name=None)

        assert state_with_job.has_active_job() is True
        assert state_without_job.has_active_job() is False

    def test_safety_blocked_tracking(self):
        """Test safety-blocked key tracking."""
        state = PersistentBatchState()

        assert not state.is_safety_blocked("key1")

        state.add_safety_blocked("key1")
        assert state.is_safety_blocked("key1")
        assert not state.is_safety_blocked("key2")

        # Adding same key again should not duplicate
        state.add_safety_blocked("key1")
        assert state.safety_blocked_keys.count("key1") == 1


class TestSafetyErrorDetection:
    """Tests for is_safety_error function."""

    def test_safety_keywords(self):
        """Test detection of safety-related error messages."""
        # Should detect safety errors
        assert is_safety_error("PROHIBITED_CONTENT: Request blocked")
        assert is_safety_error("Error: SAFETY filter triggered")
        assert is_safety_error("Content BLOCKED by policy")
        assert is_safety_error("Request blocked due to safety concerns")
        assert is_safety_error("Harmful content detected")
        assert is_safety_error("policy violation: content not allowed")

    def test_non_safety_errors(self):
        """Test non-safety errors are not flagged."""
        assert not is_safety_error("Network timeout")
        assert not is_safety_error("Rate limit exceeded")
        assert not is_safety_error("Internal server error")
        assert not is_safety_error("Invalid request format")
        assert not is_safety_error("Model overloaded")

    def test_empty_and_none(self):
        """Test empty and None error messages."""
        assert not is_safety_error("")
        assert not is_safety_error(None)

    def test_case_insensitive(self):
        """Test detection is case-insensitive."""
        assert is_safety_error("safety error")
        assert is_safety_error("SAFETY ERROR")
        assert is_safety_error("Safety Error")
        assert is_safety_error("blocked")
        assert is_safety_error("BLOCKED")


class TestAtomicSave:
    """Tests for atomic file writing."""

    def test_atomic_save_no_partial_file(self, tmp_path: Path):
        """Test that save doesn't leave partial files on error."""
        state_file = tmp_path / "atomic_test.json"

        # Create a state with valid data
        state = PersistentBatchState(active_job_name="test")

        # Mock json.dump to raise an error after partial write
        original_dump = json.dump

        def failing_dump(*args, **kwargs):
            # Write some data then fail
            args[1].write('{"partial":')
            raise IOError("Simulated write failure")

        with patch('json.dump', failing_dump):
            with pytest.raises(IOError):
                state.save(state_file)

        # The main file should not exist (atomic write failed)
        # Note: The tmp file might exist depending on when error occurs
        assert not state_file.exists()


class TestIntegration:
    """Integration tests for batch state with executor."""

    def test_state_survives_process_restart(self, tmp_path: Path):
        """Test that state persists across simulated process restarts."""
        state_file = tmp_path / "restart_test.json"

        # First "process": create and save state
        state1 = PersistentBatchState(
            active_job_name="long-running-job",
            processing_keys=["chapter_1", "chapter_2", "chapter_3"],
            batch_provider="gemini-cf",
            batch_model="gemini-3-flash-preview",
        )
        state1.save(state_file)

        # Simulate process restart by creating new state object from file
        state2 = PersistentBatchState.load(state_file)

        assert state2 is not None
        assert state2.active_job_name == "long-running-job"
        assert state2.processing_keys == ["chapter_1", "chapter_2", "chapter_3"]
        assert state2.batch_provider == "gemini-cf"
        assert state2.batch_model == "gemini-3-flash-preview"

    def test_state_with_large_metadata(self, tmp_path: Path):
        """Test state with realistic large metadata."""
        state_file = tmp_path / "large_metadata.json"

        # Create metadata for many files
        metadata = {}
        for i in range(100):
            metadata[f"chapter_{i}.part1"] = {
                "file_key": f"chapter_{i}",
                "part_index": 1,
                "total_parts": 3,
                "original_content": f"Content for chapter {i}..." * 10,
            }

        state = PersistentBatchState(
            active_job_name="large-job",
            processing_keys=list(metadata.keys()),
            batch_metadata=metadata,
        )
        state.save(state_file)

        loaded = PersistentBatchState.load(state_file)
        assert loaded is not None
        assert len(loaded.processing_keys) == 100
        assert len(loaded.batch_metadata) == 100
