"""
Tests for batch state persistence.

Tests the MegaUnitState class for the new Mega Unit architecture.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch

from pdf2epub.core.executor.batch_state import (
    MegaUnitState,
    is_safety_error,
    SAFETY_KEYWORDS,
)


class TestMegaUnitState:
    """Tests for MegaUnitState class."""

    def test_to_dict(self):
        """Test conversion to dict."""
        state = MegaUnitState(
            job_name="batches/12345",
            job_state="RUNNING",
        )
        d = state.to_dict()

        assert d["job_name"] == "batches/12345"
        assert d["job_state"] == "RUNNING"

    def test_from_dict(self):
        """Test creation from dict."""
        d = {
            "job_name": "batches/67890",
            "job_state": "SUCCEEDED",
        }
        state = MegaUnitState.from_dict(d)

        assert state.job_name == "batches/67890"
        assert state.job_state == "SUCCEEDED"

    def test_from_dict_missing_job_state(self):
        """Test from_dict handles missing job_state."""
        d = {"job_name": "batches/abc"}
        state = MegaUnitState.from_dict(d)

        assert state.job_name == "batches/abc"
        assert state.job_state == "RUNNING"  # Default

    def test_save_and_load(self, tmp_path: Path):
        """Test save/load roundtrip."""
        state_file = tmp_path / "batch_abc123.json"

        original = MegaUnitState(
            job_name="batches/roundtrip-test",
            job_state="PENDING",
        )
        original.save(state_file)

        # Verify file exists and is valid JSON
        assert state_file.exists()
        with open(state_file) as f:
            data = json.load(f)
        assert data["job_name"] == "batches/roundtrip-test"
        assert data["job_state"] == "PENDING"

        # Load and verify
        loaded = MegaUnitState.load(state_file)
        assert loaded is not None
        assert loaded.job_name == "batches/roundtrip-test"
        assert loaded.job_state == "PENDING"

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """Test save creates parent directories if needed."""
        state_file = tmp_path / "deep" / "nested" / "batch_state.json"

        state = MegaUnitState(job_name="nested-job", job_state="RUNNING")
        state.save(state_file)

        assert state_file.exists()

    def test_load_nonexistent(self, tmp_path: Path):
        """Test load returns None for nonexistent file."""
        state_file = tmp_path / "does_not_exist.json"

        result = MegaUnitState.load(state_file)
        assert result is None

    def test_load_corrupted(self, tmp_path: Path):
        """Test load handles corrupted JSON gracefully."""
        state_file = tmp_path / "corrupted.json"
        state_file.write_text("not valid json {{{")

        result = MegaUnitState.load(state_file)
        assert result is None


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
        state = MegaUnitState(job_name="test", job_state="RUNNING")

        # Mock json.dump to raise an error after partial write
        def failing_dump(*args, **kwargs):
            # Write some data then fail
            args[1].write('{"partial":')
            raise IOError("Simulated write failure")

        with patch('json.dump', failing_dump):
            with pytest.raises(IOError):
                state.save(state_file)

        # The main file should not exist (atomic write failed)
        assert not state_file.exists()


class TestIntegration:
    """Integration tests for batch state."""

    def test_state_survives_process_restart(self, tmp_path: Path):
        """Test that state persists across simulated process restarts."""
        state_file = tmp_path / "restart_test.json"

        # First "process": create and save state
        state1 = MegaUnitState(
            job_name="batches/long-running-job",
            job_state="RUNNING",
        )
        state1.save(state_file)

        # Simulate process restart by creating new state object from file
        state2 = MegaUnitState.load(state_file)

        assert state2 is not None
        assert state2.job_name == "batches/long-running-job"
        assert state2.job_state == "RUNNING"

    def test_clear_state_file(self, tmp_path: Path):
        """Test deleting state file."""
        state_file = tmp_path / "to_clear.json"

        # Create file
        state = MegaUnitState(job_name="to-clear", job_state="RUNNING")
        state.save(state_file)
        assert state_file.exists()

        # Delete it
        state_file.unlink()
        assert not state_file.exists()
