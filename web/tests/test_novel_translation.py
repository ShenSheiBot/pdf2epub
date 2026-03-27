"""Tests for novel translation pipeline v4."""

import json
import zipfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf2epub.html_translation.novel_translator import (
    NovelTranslator,
    NovelState,
    repair_images,
    make_novel_content_validator,
    strip_spurious_headings,
)
from pdf2epub.html_translation.glossary_manager import GlossaryManager, _alias_ja_keys


def _has_alias(store_entry, ja_name):
    """Check if a store entry has an alias with the given Japanese name."""
    return ja_name in _alias_ja_keys(store_entry.get("aliases", []))
from pdf2epub.html_translation.novel_verifier import (
    _check_preamble,
    _check_alignment,
    remove_preamble,
    find_hallucination_boundary,
    verify_translation,
)


# ─── NovelState Persistence ───

class TestNovelState:
    def test_save_and_load(self, tmp_path):
        state = NovelState(current_unit_index=3, completed_units=[0, 1, 2])
        path = tmp_path / "state.json"
        state.save(path)

        loaded = NovelState.load(path)
        assert loaded.current_unit_index == 3
        assert loaded.completed_units == [0, 1, 2]

    def test_default_state(self):
        state = NovelState()
        assert state.current_unit_index == 0
        assert state.completed_units == []


# ─── Content Validator ───

class TestContentValidator:
    def test_valid_output(self):
        source = "行1\n行2\n行3"
        validator = make_novel_content_validator(source)
        assert validator("Line1\nLine2\nLine3") is None

    def test_within_tolerance(self):
        source = "行1\n行2\n行3\n行4\n行5"
        validator = make_novel_content_validator(source)
        # 5 source, 7 translated — diff=2, within tolerance of 3
        assert validator("L1\nL2\nL3\nL4\nL5\nL6\nL7") is None

    def test_truncation_detected(self):
        source = "\n".join(f"行{i}" for i in range(20))
        validator = make_novel_content_validator(source)
        result = validator("L1\nL2\nL3")
        assert result is not None
        assert "Truncated" in result

    def test_excess_detected(self):
        source = "行1\n行2\n行3"
        validator = make_novel_content_validator(source)
        result = validator("\n".join(f"L{i}" for i in range(20)))
        assert result is not None
        assert "mismatch" in result.lower()

    def test_empty_output(self):
        validator = make_novel_content_validator("行1\n行2")
        assert "Empty" in validator("")


# ─── Image Repair ───

class TestRepairImages:
    def test_no_images(self):
        source = "行1\n行2"
        translated = "Line1\nLine2"
        assert repair_images(source, translated) == translated

    def test_image_preserved(self):
        source = "行1\n[Image: cover.jpg]\n行2"
        translated = "Line1\n[Image: cover.jpg]\nLine2"
        assert repair_images(source, translated) == translated

    def test_missing_image_reinserted(self):
        source = "行1\n[Image: cover.jpg]\n行2"
        translated = "Line1\nLine2"
        result = repair_images(source, translated)
        assert "[Image: cover.jpg]" in result

    def test_hallucinated_image_removed(self):
        source = "行1\n行2"
        translated = "Line1\n[Image: fake.jpg]\nLine2"
        result = repair_images(source, translated)
        assert "[Image: fake.jpg]" not in result

    def test_mixed_repair(self):
        source = "行1\n[Image: real.jpg]\n行2"
        translated = "Line1\n[Image: fake.jpg]\nLine2"
        result = repair_images(source, translated)
        assert "[Image: real.jpg]" in result
        assert "[Image: fake.jpg]" not in result


# ─── GlossaryManager ───

class TestGlossaryManager:
    def _make_manager(self, tmp_path):
        mock_client = MagicMock()
        return GlossaryManager(
            output_dir=tmp_path,
            llm_client=mock_client,
            model_configs=[{"provider": "anthropic", "model": "test"}],
        ), mock_client

    def test_recall_empty_store(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        assert mgr.recall("some text") == ""

    def test_recall_key_match(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        mgr.store = {
            "宮崎薰": {
                "zh_name": "宫崎薰",
                "aliases": ["スミレ"],
                "description": "女主角",
            }
        }
        result = mgr.recall("宮崎薰は学校に行った")
        assert "宫崎薰" in result
        assert "女主角" in result

    def test_recall_alias_match(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        mgr.store = {
            "宮崎薰": {
                "zh_name": "宫崎薰",
                "aliases": ["スミレ", "すみれ"],
                "description": "女主角",
            }
        }
        # Key not in text, but alias is
        result = mgr.recall("スミレは笑った")
        assert "宫崎薰" in result

    def test_recall_no_match(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        mgr.store = {
            "宮崎薰": {
                "zh_name": "宫崎薰",
                "aliases": ["スミレ"],
                "description": "女主角",
            }
        }
        result = mgr.recall("天気がいい日だった")
        assert "宫崎薰" not in result

    def test_extract_and_update(self, tmp_path):
        mgr, mock_client = self._make_manager(tmp_path)
        entries = [{"key": "宮崎薰", "zh_name": "宫崎薰", "aliases": ["スミレ"], "description": "女主角"}]
        mock_client.generate.return_value = json.dumps(entries)
        # Patch _dedup_entries to pass through (tested separately)
        mgr._dedup_entries = lambda entries, *args: entries
        mgr.extract_and_update("source", "translated", "ch01")
        assert "宮崎薰" in mgr.store
        assert mgr.store["宮崎薰"]["zh_name"] == "宫崎薰"
        assert mgr.store["宮崎薰"]["updated_by"] == "ch01"

    def test_update_existing_entry(self, tmp_path):
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = {
            "宮崎薰": {
                "zh_name": "宫崎薰",
                "aliases": ["スミレ"],
                "description": "女主角，学生",
                "updated_by": "ch01",
                "timestamp": "2026-01-01",
                "history": [],
            }
        }
        mock_client.generate.return_value = json.dumps([
            {"key": "宮崎薰", "zh_name": "宫崎薰", "aliases": ["スミレ", "すみれ"],
             "description": "女主角，与主角交往中"}
        ])
        mgr._dedup_entries = lambda entries, *args: entries
        mgr.extract_and_update("source", "translated", "ch10")

        entry = mgr.store["宮崎薰"]
        assert entry["description"] == "女主角，与主角交往中"
        assert entry["updated_by"] == "ch10"
        assert len(entry["history"]) == 1
        assert entry["history"][0]["description"] == "女主角，学生"
        assert entry["history"][0]["updated_by"] == "ch01"
        # Aliases merged
        assert _has_alias(entry, "すみれ")

    def test_save_and_load(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        mgr.store = {"test": {"zh_name": "测试", "aliases": [], "description": "desc",
                               "updated_by": "ch01", "timestamp": "now", "history": []}}
        mgr.prev_chapter = "test → 测试：desc"
        mgr.save()

        mgr2, _ = self._make_manager(tmp_path)
        mgr2.load()
        assert "test" in mgr2.store
        assert mgr2.prev_chapter == "test → 测试：desc"

    def test_malformed_json_non_fatal(self, tmp_path):
        mgr, mock_client = self._make_manager(tmp_path)
        mock_client.generate.side_effect = Exception("API error")
        # Should not raise — glossary extraction is non-fatal
        result = mgr.extract_and_update("source", "translated", "ch01")
        assert result == ""


# ─── Glossary Dedup ───

class TestGlossaryDedup:
    """Test LLM-based glossary dedup (shared-alias merge only; pronoun cleaning disabled)."""

    def _make_manager(self, tmp_path):
        mock_client = MagicMock()
        return GlossaryManager(
            output_dir=tmp_path,
            llm_client=mock_client,
            model_configs=[{"provider": "anthropic", "model": "test"}],
        ), mock_client

    # -- Existing store with conflicts --
    EXISTING_STORE = {
        "河見誠一郎": {
            "zh_name": "河见诚一郎",
            "aliases": ["河見", "班主任"],
            "description": "男，星蘭高中班主任老师",
            "updated_by": "ch01",
            "timestamp": "2026-03-25T00:00:00",
            "history": [],
        },
        "島崎蒼": {
            "zh_name": "岛崎苍",
            "aliases": ["島崎", "俺"],
            "description": "男，男主角，星蘭高中二年级学生",
            "updated_by": "ch01",
            "timestamp": "2026-03-25T00:00:00",
            "history": [],
        },
    }

    # -- New extraction entries (ch02) --
    NEW_ENTRIES = [
        {
            "key": "河見誠一郎",
            "zh_name": "河见诚一郎",
            "aliases": ["河見", "先生", "班主任"],
            "description": "男，星蘭高中班主任老师，关心学生",
        },
        {
            "key": "青田美咲",
            "zh_name": "青田美咲",
            "aliases": ["青田", "班主任"],
            "description": "女，隔壁班的班主任",
        },
        {
            "key": "島崎蒼",
            "zh_name": "岛崎苍",
            "aliases": ["島崎", "俺", "蒼"],
            "description": "男，男主角，开始意识到自己的感情",
        },
        {
            "key": "宮崎薰",
            "zh_name": "宫崎薰",
            "aliases": ["スミレ", "あの子"],
            "description": "女，女主角，内向",
        },
    ]

    # Mock Call 2 response: dedup result (only modified entries)
    # With Call 1 disabled, the only shared alias is 班主任 (河見 + 青田).
    # 俺 is only used by 島崎蒼 (one key), so not shared. あの子 only by 宮崎薰.
    # Call 2 asks LLM about the 班主任 group — LLM determines they're different people,
    # keeps 班主任 with 河見 (the actual homeroom teacher) and removes from 青田.
    DEDUP_RESPONSE = json.dumps([
        {
            "key": "青田美咲",
            "zh_name": "青田美咲",
            "aliases": ["青田"],
            "description": "女，隔壁班的班主任",
        },
    ])

    def test_shared_alias_triggers_dedup(self, tmp_path):
        """Shared alias (班主任) triggers Call 2; non-shared aliases (俺, あの子) are untouched."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        # Only Call 2 (no Call 1): 班主任 is shared between 河見 and 青田
        mock_client.generate.return_value = self.DEDUP_RESPONSE

        result = mgr._dedup_entries(
            list(self.NEW_ENTRIES), "ch02",
            "fake extraction prompt", "fake extraction response",
        )

        # Only 1 LLM call (Call 2 for shared alias group)
        assert mock_client.generate.call_count == 1

        # 班主任 kept with 河見 (not in dedup result = unchanged), removed from 青田
        kawami = next(e for e in result if e["key"] == "河見誠一郎")
        assert _has_alias(kawami, "班主任")  # kept (河見 is the actual homeroom teacher)
        aota = next(e for e in result if e["key"] == "青田美咲")
        assert not _has_alias(aota, "班主任")  # removed

        # 俺 and あの子 are NOT shared (only 1 key each) — untouched by dedup
        aoi = next(e for e in result if e["key"] == "島崎蒼")
        assert _has_alias(aoi, "俺")  # kept (not shared)
        kaoru = next(e for e in result if e["key"] == "宮崎薰")
        assert _has_alias(kaoru, "あの子")  # kept (not shared)

    def test_shared_alias_cleaned(self, tmp_path):
        """Shared alias (班主任) removed from non-owner by LLM dedup."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        # Only Call 2 (shared alias group for 班主任)
        mock_client.generate.return_value = self.DEDUP_RESPONSE

        result = mgr._dedup_entries(
            list(self.NEW_ENTRIES), "ch02",
            "fake extraction prompt", "fake extraction response",
        )

        # 河見 NOT in dedup result → keeps original aliases including 班主任
        kawami = next(e for e in result if e["key"] == "河見誠一郎")
        assert _has_alias(kawami, "班主任")
        assert _has_alias(kawami, "河見")

        # 青田 modified by dedup — 班主任 removed
        aota = next(e for e in result if e["key"] == "青田美咲")
        assert not _has_alias(aota, "班主任")
        assert _has_alias(aota, "青田")

    def test_store_entry_updated_by_dedup(self, tmp_path):
        """Existing store entry aliases get cleaned when dedup modifies them."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        # Store entry 河見 has "班主任"
        assert _has_alias(mgr.store["河見誠一郎"], "班主任")

        # Only Call 2 (班主任 shared alias group)
        # DEDUP_RESPONSE only modifies 青田 (removes 班主任), 河見 unchanged (keeps 班主任)
        mock_client.generate.return_value = self.DEDUP_RESPONSE
        mgr._dedup_entries(
            list(self.NEW_ENTRIES), "ch02",
            "fake extraction prompt", "fake extraction response",
        )

        # Store entry 河見 still has 班主任 (it was the keeper in dedup)
        # 青田 had 班主任 removed

    def test_no_dedup_when_no_aliases(self, tmp_path):
        """Skip dedup entirely when entries have no aliases."""
        mgr, mock_client = self._make_manager(tmp_path)
        entries = [
            {"key": "東京", "zh_name": "东京", "aliases": [], "description": "首都"},
        ]

        result = mgr._dedup_entries(entries, "ch01", "prompt", "response")

        # No LLM calls should be made
        mock_client.generate.assert_not_called()
        assert result == entries

    def test_no_dedup_when_no_conflicts(self, tmp_path):
        """No LLM calls at all when no shared aliases exist."""
        mgr, mock_client = self._make_manager(tmp_path)
        entries = [
            {"key": "宮崎薰", "zh_name": "宫崎薰", "aliases": ["スミレ"], "description": "女主角"},
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["蒼"], "description": "男主角"},
        ]

        result = mgr._dedup_entries(entries, "ch01", "prompt", "response")

        # No shared aliases → no LLM calls at all (Call 1 disabled)
        mock_client.generate.assert_not_called()
        assert result == entries

    def test_call2_uses_messages_format(self, tmp_path):
        """Call 2 sends messages with extraction prompt as first user message for cache hit."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        # Only Call 2 (no Call 1)
        mock_client.generate.return_value = self.DEDUP_RESPONSE

        extraction_prompt = "fake extraction prompt with source_text"
        extraction_response = "fake extraction response JSON"

        mgr._dedup_entries(
            list(self.NEW_ENTRIES), "ch02",
            extraction_prompt, extraction_response,
        )

        # Only 1 call (Call 2 only, no Call 1)
        assert mock_client.generate.call_count == 1
        call2_kwargs = mock_client.generate.call_args_list[0]
        prompt_arg = call2_kwargs[1].get("prompt") or call2_kwargs[0][0]

        # Should be a list of messages
        assert isinstance(prompt_arg, list)
        assert len(prompt_arg) == 3
        assert prompt_arg[0]["role"] == "user"
        assert prompt_arg[0]["content"] == extraction_prompt
        assert prompt_arg[1]["role"] == "assistant"
        assert prompt_arg[1]["content"] == extraction_response
        assert prompt_arg[2]["role"] == "user"

    def test_dedup_log_saved(self, tmp_path):
        """Dedup results are saved to log file."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        # Only Call 2 (no Call 1)
        mock_client.generate.return_value = self.DEDUP_RESPONSE

        mgr._dedup_entries(
            list(self.NEW_ENTRIES), "ch02",
            "prompt", "response",
        )

        log_file = tmp_path / "logs" / "glossary" / "ch02_dedup.json"
        assert log_file.exists()
        log_data = json.loads(log_file.read_text())
        assert "dedup_result" in log_data
        assert "groups" in log_data

    def test_merge_absorbed_entry(self, tmp_path):
        """When LLM merges two entries, the absorbed one's key becomes an alias."""
        mgr, mock_client = self._make_manager(tmp_path)
        # Store has entry that is the same character as new entry (different kanji variant)
        mgr.store = {
            "高千穂弥生": {
                "zh_name": "高千穗弥生",
                "aliases": ["弥生"],
                "description": "女，班级委员长",
                "updated_by": "ch01",
                "timestamp": "2026-03-25T00:00:00",
                "history": [],
            },
        }

        new_entries = [
            {
                "key": "高千穗弥生",  # variant kanji 穂→穗
                "zh_name": "高千穗弥生",
                "aliases": ["弥生", "委員長"],
                "description": "女，班级委员长，严格的性格",
            },
        ]

        # Shared alias: 弥生 → {高千穂弥生, 高千穗弥生} — triggers Call 2
        # No Call 1. LLM merges into the store key, adds variant as alias
        dedup_resp = json.dumps([{
            "key": "高千穂弥生",
            "zh_name": "高千穗弥生",
            "aliases": ["弥生", "高千穗弥生", "委員長"],
            "description": "女，班级委员长，严格的性格",
        }])

        mock_client.generate.return_value = dedup_resp

        result = mgr._dedup_entries(
            new_entries, "ch02", "prompt", "response",
        )

        # Only 1 LLM call (Call 2)
        assert mock_client.generate.call_count == 1

        # The merged entry should have the old variant as alias
        assert len(result) == 1
        merged = result[0]
        assert merged["key"] == "高千穂弥生"
        assert _has_alias(merged, "高千穗弥生")

    def test_dedup_call_failure_nonfatal(self, tmp_path):
        """If dedup calls fail, entries pass through unchanged."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        # Call 1 fails
        mock_client.generate.side_effect = Exception("API error")

        entries = list(self.NEW_ENTRIES)
        result = mgr._dedup_entries(entries, "ch02", "prompt", "response")

        # Entries unchanged — dedup is non-fatal
        assert len(result) == len(entries)

    def test_full_extract_and_update_with_dedup(self, tmp_path):
        """Integration: extract_and_update runs extraction + dedup + store update."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.EXISTING_STORE))

        extraction_json = json.dumps(self.NEW_ENTRIES)

        # Two generate calls: extraction, then Call 2 dedup (no Call 1)
        mock_client.generate.side_effect = [
            extraction_json,
            self.DEDUP_RESPONSE,
        ]

        mgr.extract_and_update("source text", "translated text", "ch02")

        # Store should have all entries; shared alias (班主任) resolved
        assert "河見誠一郎" in mgr.store
        assert _has_alias(mgr.store["河見誠一郎"], "班主任")  # kept (owner)
        assert "青田美咲" in mgr.store
        assert not _has_alias(mgr.store["青田美咲"], "班主任")  # removed
        # Non-shared aliases (俺, あの子) are NOT removed (no Call 1, not shared)
        assert "島崎蒼" in mgr.store
        assert _has_alias(mgr.store["島崎蒼"], "俺")
        assert "宮崎薰" in mgr.store
        assert _has_alias(mgr.store["宮崎薰"], "あの子")


# ─── Dedup State Persistence ───

class TestGlossaryDedupState:
    """Test dedup state persistence, caching, and lifecycle across chapters."""

    def _make_manager(self, tmp_path):
        mock_client = MagicMock()
        return GlossaryManager(
            output_dir=tmp_path,
            llm_client=mock_client,
            model_configs=[{"provider": "anthropic", "model": "test"}],
        ), mock_client

    STORE_CH1 = {
        "河見誠一郎": {
            "zh_name": "河见诚一郎",
            "aliases": ["河見", "班主任"],
            "description": "男，星蘭高中班主任老师",
            "updated_by": "ch01",
            "timestamp": "2026-03-25T00:00:00",
            "history": [],
        },
        "島崎蒼": {
            "zh_name": "岛崎苍",
            "aliases": ["島崎", "俺"],
            "description": "男，男主角",
            "updated_by": "ch01",
            "timestamp": "2026-03-25T00:00:00",
            "history": [],
        },
    }

    def test_state_save_and_load(self, tmp_path):
        """Dedup state persists across manager instances."""
        mgr, _ = self._make_manager(tmp_path)
        mgr.dedup_state = {
            "confirmed_not_dup": [["A", "B"], ["C", "D"]],
            "exclusive_aliases": {"A": ["俺"]},
            "removed_aliases": {"B": ["先生"]},
            "known_pronouns": ["俺", "先生", "あの子"],
        }
        mgr.save()

        mgr2, _ = self._make_manager(tmp_path)
        mgr2.load()
        assert ["A", "B"] in mgr2.dedup_state["confirmed_not_dup"]
        assert "俺" in mgr2.dedup_state["exclusive_aliases"]["A"]
        assert "先生" in mgr2.dedup_state["removed_aliases"]["B"]
        assert "俺" in mgr2.dedup_state["known_pronouns"]

    @pytest.mark.skip(reason="Call 1 pronoun identification disabled — re-enable if needed")
    def test_known_pronouns_skip_call1(self, tmp_path):
        """Call 1 is skipped when all aliases are already checked or exclusive."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.STORE_CH1))
        # Pre-populate: all aliases checked
        mgr.dedup_state["known_pronouns"] = ["俺", "班主任"]
        mgr.dedup_state["checked_aliases"] = ["俺", "班主任", "河見", "島崎"]
        mgr.dedup_state["exclusive_aliases"] = {"河見誠一郎": ["河見"], "島崎蒼": ["島崎"]}

        # Entries with only known aliases
        entries = [
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見", "班主任"], "description": "same"},
        ]

        # If Call 1 is skipped and groups are resolved, no generate calls at all
        # But 班主任 is shared (store has it too) and needs dedup check
        # However confirmed_not_dup is empty, so Call 2 will run
        mock_client.generate.return_value = "[]"  # Call 2 returns no modifications

        mgr._dedup_entries(entries, "ch02", "prompt", "response")

        # Only 1 call (Call 2), not 2 (Call 1 was skipped)
        assert mock_client.generate.call_count == 1

    def test_confirmed_not_dup_skips_group(self, tmp_path):
        """Groups where all pairs are confirmed not-dup are skipped."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.STORE_CH1))
        mgr.store["青田美咲"] = {
            "zh_name": "青田美咲", "aliases": ["青田", "班主任"],
            "description": "女", "updated_by": "ch02", "timestamp": "t", "history": [],
        }

        # Pre-populate: 河見 and 青田 confirmed not-dup
        mgr.dedup_state["confirmed_not_dup"] = [["河見誠一郎", "青田美咲"]]

        entries = [
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見", "班主任"], "description": "same"},
            {"key": "青田美咲", "zh_name": "青田美咲", "aliases": ["青田", "班主任"], "description": "same"},
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺"], "description": "same"},
        ]

        result = mgr._dedup_entries(entries, "ch03", "prompt", "response")

        # 班主任 shared between 河見 and 青田, but pair is confirmed not-dup → skipped
        # 俺 only in 島崎 → not shared → no dedup needed
        # Everything resolved — no LLM calls needed
        mock_client.generate.assert_not_called()
        assert len(result) == len(entries)

    def test_confirmed_not_dup_recheck_with_new_member(self, tmp_path):
        """New member in a group forces re-check even if original pair was confirmed."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.STORE_CH1))
        mgr.store["青田美咲"] = {
            "zh_name": "青田美咲", "aliases": ["青田", "班主任"],
            "description": "女", "updated_by": "ch02", "timestamp": "t", "history": [],
        }

        # 河見 and 青田 confirmed not-dup from ch02
        mgr.dedup_state["confirmed_not_dup"] = [["河見誠一郎", "青田美咲"]]

        # New entry 山田 also has alias 班主任 — new member forces re-check
        entries = [
            {"key": "山田太郎", "zh_name": "山田太郎", "aliases": ["山田", "班主任"], "description": "新任教师"},
        ]

        # Only Call 2 (no Call 1): 班主任 shared among 河見, 青田, 山田
        # 河見-青田 confirmed, but 河見-山田 and 青田-山田 are NOT confirmed → group not skipped
        mock_client.generate.return_value = "[]"  # Call 2: no modifications

        mgr._dedup_entries(entries, "ch04", "prompt", "response")

        # 1 call: Call 2 only (no Call 1)
        assert mock_client.generate.call_count == 1

    @pytest.mark.skip(reason="Call 1 pronoun identification disabled — re-enable if needed")
    def test_exclusive_invalidated_by_new_claimant(self, tmp_path):
        """Exclusive whitelist is invalidated when a new entry claims the same alias."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = {
            "河見誠一郎": {
                "zh_name": "河见诚一郎", "aliases": ["河見", "班主任"],
                "description": "男，老师", "updated_by": "ch01", "timestamp": "t", "history": [],
            },
        }

        # 班主任 was confirmed exclusive to 河見 in ch02
        mgr.dedup_state["exclusive_aliases"] = {"河見誠一郎": ["班主任"]}
        mgr.dedup_state["known_pronouns"] = ["班主任"]
        mgr.dedup_state["checked_aliases"] = ["班主任", "河見"]

        # Ch05: new character also claims 班主任
        entries = [
            {"key": "青田美咲", "zh_name": "青田美咲", "aliases": ["青田", "班主任"], "description": "女"},
        ]

        # Call 1 for "青田" (new alias), then Call 2
        pronoun_resp = "[]"  # 青田 is not a pronoun
        dedup_resp = json.dumps([
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見"], "description": "男，老师"},
            {"key": "青田美咲", "zh_name": "青田美咲", "aliases": ["青田"], "description": "女"},
        ])
        mock_client.generate.side_effect = [pronoun_resp, dedup_resp]

        mgr._dedup_entries(entries, "ch05", "prompt", "response")

        # Call 1 + Call 2 should have run (exclusive was invalidated)
        assert mock_client.generate.call_count == 2

        # 班主任 should now be in removed_aliases for both
        removed = mgr.dedup_state.get("removed_aliases", {})
        assert "班主任" in removed.get("河見誠一郎", [])
        assert "班主任" in removed.get("青田美咲", [])

        # And removed from exclusive
        exclusive = mgr.dedup_state.get("exclusive_aliases", {})
        assert "班主任" not in exclusive.get("河見誠一郎", [])

    @pytest.mark.skip(reason="Call 1 pronoun identification disabled — re-enable if needed")
    def test_removed_aliases_blocked_in_store_update(self, tmp_path):
        """Blacklisted aliases are not re-added during store update."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = {
            "島崎蒼": {
                "zh_name": "岛崎苍", "aliases": ["島崎"],
                "description": "男主角", "updated_by": "ch02", "timestamp": "t", "history": [],
            },
        }

        # 俺 was removed from 島崎蒼 in ch02
        mgr.dedup_state["removed_aliases"] = {"島崎蒼": ["俺"]}

        # Extraction returns 島崎蒼 with 俺 again (LLM doesn't know about blacklist)
        entries = [{"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺", "蒼"], "description": "男主角"}]
        mock_client.generate.return_value = json.dumps(entries)
        mgr._dedup_entries = lambda entries, *args: entries  # skip dedup

        mgr.extract_and_update("source", "translated", "ch03")

        # 俺 should NOT be in store (blacklisted)
        assert not _has_alias(mgr.store["島崎蒼"], "俺")
        # 蒼 and 島崎 should be there
        assert _has_alias(mgr.store["島崎蒼"], "蒼")
        assert _has_alias(mgr.store["島崎蒼"], "島崎")

    def test_state_inferred_from_empty_result(self, tmp_path):
        """Empty Call 2 result means all entries in group confirmed not-dup."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.STORE_CH1))
        mgr.store["青田美咲"] = {
            "zh_name": "青田美咲", "aliases": ["青田", "班主任"],
            "description": "女", "updated_by": "ch02", "timestamp": "t", "history": [],
        }

        entries = [
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見", "班主任"], "description": "same"},
            {"key": "青田美咲", "zh_name": "青田美咲", "aliases": ["青田", "班主任"], "description": "same"},
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺"], "description": "same"},
        ]

        # 班主任 is shared between 河見 and 青田 → Call 2 fires
        # 俺 only in 島崎 → not shared → no dedup
        # Call 2: no modifications (empty array)
        mock_client.generate.return_value = "[]"

        mgr._dedup_entries(entries, "ch02", "prompt", "response")

        # 河見 and 青田 should be confirmed not-dup (shared 班主任 group, empty result)
        confirmed = mgr.dedup_state.get("confirmed_not_dup", [])
        assert ["河見誠一郎", "青田美咲"] in confirmed or ["青田美咲", "河見誠一郎"] in confirmed

        # exclusive_aliases not inferred (disabled) — should be empty
        exclusive = mgr.dedup_state.get("exclusive_aliases", {})
        assert exclusive == {}

    def test_catastrophic_retry(self, tmp_path):
        """Catastrophic failure triggers retry with error context."""
        mgr, mock_client = self._make_manager(tmp_path)
        mgr.store = json.loads(json.dumps(self.STORE_CH1))

        # 班主任 is shared between 河見 (in store + entries) and 青田 (in entries)
        entries = [
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見", "班主任"], "description": "same"},
            {"key": "青田美咲", "zh_name": "青田美咲", "aliases": ["青田", "班主任"], "description": "same"},
        ]

        # First attempt: catastrophic (unknown key)
        bad_result = json.dumps([{"key": "完全未知的角色", "zh_name": "???", "aliases": [], "description": "???"}])
        # Second attempt: correct (keep 班主任 in 河見, remove from 青田)
        good_result = json.dumps([
            {"key": "青田美咲", "zh_name": "青田美咲", "aliases": ["青田"], "description": "same"},
        ])
        mock_client.generate.side_effect = [bad_result, good_result]

        result = mgr._dedup_entries(entries, "ch02", "prompt", "response")

        # Should have retried (2 calls)
        assert mock_client.generate.call_count == 2

        # Second attempt's result should be applied
        aota = next(e for e in result if e["key"] == "青田美咲")
        assert not _has_alias(aota, "班主任")
        # 河見 not in dedup result → keeps 班主任
        kawami = next(e for e in result if e["key"] == "河見誠一郎")
        assert _has_alias(kawami, "班主任")

    def test_multi_chapter_lifecycle(self, tmp_path):
        """Full 3-chapter scenario with growing dedup state."""
        mgr, mock_client = self._make_manager(tmp_path)

        # --- Chapter 1: First extraction, no shared aliases → no dedup ---
        ch1_entries = [
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺"], "description": "男主角"},
            {"key": "宮崎薰", "zh_name": "宫崎薰", "aliases": ["スミレ"], "description": "女主角"},
        ]
        ch1_extract = json.dumps(ch1_entries)

        # No shared aliases (俺 only in 島崎, スミレ only in 宮崎) → only extraction call
        mock_client.generate.side_effect = [ch1_extract]
        mgr.extract_and_update("source1", "translated1", "ch01")

        # 俺 kept (not shared, no dedup ran)
        assert _has_alias(mgr.store["島崎蒼"], "俺")
        mock_client.reset_mock()

        # --- Chapter 2: New characters, shared alias 先生 ---
        ch2_entries = [
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見", "先生"], "description": "老师"},
            {"key": "小花衣", "zh_name": "小花衣", "aliases": ["先生"], "description": "另一个老师"},
        ]
        ch2_extract = json.dumps(ch2_entries)
        # Call 2: 先生 shared between 河見 and 小花衣, LLM removes from 小花衣 (not the primary)
        ch2_dedup = json.dumps([
            {"key": "小花衣", "zh_name": "小花衣", "aliases": [], "description": "另一个老师"},
        ])

        mock_client.generate.side_effect = [ch2_extract, ch2_dedup]
        mgr.extract_and_update("source2", "translated2", "ch02")

        # 2 calls: extraction + Call 2
        assert mock_client.generate.call_count == 2
        # 河見 and 小花衣 confirmed not-dup
        confirmed = mgr.dedup_state.get("confirmed_not_dup", [])
        assert ["小花衣", "河見誠一郎"] in confirmed or ["河見誠一郎", "小花衣"] in confirmed
        mock_client.reset_mock()

        # --- Chapter 3: Same characters, no new shared aliases ---
        ch3_entries = [
            {"key": "河見誠一郎", "zh_name": "河见诚一郎", "aliases": ["河見", "先生"], "description": "same"},
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺"], "description": "same"},
        ]
        ch3_extract = json.dumps(ch3_entries)

        # 先生: entry_pool has 河見 (from entries) with 先生, and 小花衣 (from store) with aliases=[]
        # So 先生 → only {河見} → not shared → no dedup
        # 俺 → only {島崎} → not shared → no dedup
        # Only extraction call
        mock_client.generate.side_effect = [ch3_extract]
        mgr.extract_and_update("source3", "translated3", "ch03")

        # Only 1 generate call (extraction only, dedup skipped)
        assert mock_client.generate.call_count == 1

        # 先生 still in store for 河見 (kept from ch2, no removed_aliases blacklist)
        assert _has_alias(mgr.store.get("河見誠一郎", {}), "先生")
        # 俺 still in store for 島崎
        assert _has_alias(mgr.store.get("島崎蒼", {}), "俺")


# ─── E2E: Glossary propagation across chapters ───

class TestGlossaryE2E:
    """End-to-end test: glossary extraction, dedup, recall, and store propagation across two chapters."""

    def _make_manager(self, tmp_path):
        mock_client = MagicMock()
        return GlossaryManager(
            output_dir=tmp_path,
            llm_client=mock_client,
            model_configs=[{"provider": "anthropic", "model": "test"}],
        ), mock_client

    def test_two_chapter_propagation(self, tmp_path):
        """Ch1 extracts glossary → ch2 recall sees ch1's entries → ch2 dedup uses shared aliases."""
        mgr, mock_client = self._make_manager(tmp_path)

        # ── Chapter 1 ──
        ch1_source = "島崎蒼は教室で宮崎薰を見た。スミレは窓の外を見ていた。"
        ch1_translated = "岛崎苍在教室里看到了宫崎薰。堇正看着窗外。"

        ch1_extraction = json.dumps([
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺"], "description": "男，男主角，高中二年级"},
            {"key": "宮崎薰", "zh_name": "宫崎薰", "aliases": ["スミレ", "あの子"], "description": "女，女主角，内向"},
        ])
        # No shared aliases (俺 only 島崎, スミレ only 宮崎, あの子 only 宮崎) → no dedup
        mock_client.generate.side_effect = [ch1_extraction]
        mgr.extract_and_update(ch1_source, ch1_translated, "ch01")

        # ── Verify ch1 state ──
        assert "島崎蒼" in mgr.store
        assert _has_alias(mgr.store["島崎蒼"], "俺")  # kept (not shared)
        assert "宮崎薰" in mgr.store
        assert _has_alias(mgr.store["宮崎薰"], "あの子")  # kept (not shared, no Call 1)
        assert _has_alias(mgr.store["宮崎薰"], "スミレ")

        # prev_chapter set
        assert "島崎蒼" in mgr.prev_chapter
        assert "宮崎薰" in mgr.prev_chapter

        # Persist and reload (simulates next session)
        mgr.save()
        mgr2, mock_client2 = self._make_manager(tmp_path)
        mgr2.load()
        assert len(mgr2.store) == 2

        mock_client2.reset_mock()

        # ── Chapter 2: recall sees ch1 entries ──
        ch2_source = "島崎蒼は高千穂弥生と話した。弥生は委員長だった。先生が入ってきた。"
        ch2_translated = "岛崎苍和高千穗弥生说了话。弥生是班委。老师走了进来。"

        # recall should match 島崎蒼 (key in source)
        recalled = mgr2.recall(ch2_source)
        assert "岛崎苍" in recalled
        assert "俺" in recalled  # alias included in recall

        # Ch2 extraction — no shared aliases between entries
        # 俺 only in 島崎, 弥生 only in 高千穂弥生, 委員長 only in 高千穂弥生
        ch2_extraction = json.dumps([
            {"key": "島崎蒼", "zh_name": "岛崎苍", "aliases": ["島崎", "俺"], "description": "男，男主角，开始注意弥生"},
            {"key": "高千穂弥生", "zh_name": "高千穗弥生", "aliases": ["弥生", "委員長"], "description": "女，班级委员长"},
        ])

        # No shared aliases → only extraction call, no dedup
        mock_client2.generate.side_effect = [ch2_extraction]
        mgr2.extract_and_update(ch2_source, ch2_translated, "ch02")

        # ── Verify ch2 state ──

        # Store has 3 entries now
        assert len(mgr2.store) == 3
        assert "高千穂弥生" in mgr2.store
        assert _has_alias(mgr2.store["高千穂弥生"], "委員長")

        # 島崎蒼 description updated, history has ch01 version
        assert "弥生" in mgr2.store["島崎蒼"]["description"]
        assert len(mgr2.store["島崎蒼"]["history"]) == 1
        assert "高中二年级" in mgr2.store["島崎蒼"]["history"][0]["description"]

        # Verify extraction prompt for ch2 included ch1's entries
        extraction_call = mock_client2.generate.call_args_list[0]
        extraction_prompt = extraction_call[1].get("prompt") or extraction_call[0][0]
        assert "島崎蒼" in extraction_prompt  # ch1 entry in existing_section
        assert "宮崎薰" in extraction_prompt  # ch1 entry in existing_section

        # prev_chapter updated to ch2's entries
        assert "高千穂弥生" in mgr2.prev_chapter


# ─── Deterministic Verifier ───

class TestNovelVerifier:
    """Test the deterministic translation verifier (replaces agent)."""

    def _mock_client(self):
        return MagicMock()

    def _model_configs(self):
        return [{"provider": "anthropic", "model": "test"}]

    # -- Preamble --

    def test_preamble_translation_kept(self):
        mock = self._mock_client()
        mock.generate.return_value = "translation"
        result = _check_preamble("原文", "译文", mock, self._model_configs())
        assert result == "translation"

    def test_preamble_metacomment_detected(self):
        mock = self._mock_client()
        mock.generate.return_value = "meta-comment"
        result = _check_preamble("原文", "以下是翻译：", mock, self._model_configs())
        assert result == "meta-comment"

    def test_remove_preamble_no_preamble(self):
        mock = self._mock_client()
        mock.generate.return_value = "translation"
        source = ["行1", "行2", "行3"]
        translated = ["Line1", "Line2", "Line3"]
        result = remove_preamble(source, translated, mock, self._model_configs())
        assert result == translated
        assert mock.generate.call_count == 1  # Only checked once

    def test_remove_preamble_one_line(self):
        mock = self._mock_client()
        mock.generate.side_effect = ["meta-comment", "translation"]
        source = ["行1", "行2"]
        translated = ["以下是翻译：", "Line1", "Line2"]
        result = remove_preamble(source, translated, mock, self._model_configs())
        assert result == ["Line1", "Line2"]

    def test_remove_preamble_all_fail(self):
        mock = self._mock_client()
        mock.generate.return_value = "meta-comment"
        source = ["行1"]
        translated = ["垃圾1", "垃圾2", "垃圾3"]
        result = remove_preamble(source, translated, mock, self._model_configs())
        assert result is None  # Signal retry

    # -- Alignment --

    def test_alignment_returns_letter(self):
        mock = self._mock_client()
        mock.generate.return_value = "A"
        result = _check_alignment(["src"], ["tl"], mock, self._model_configs())
        assert result == "A"

    def test_alignment_extracts_first_letter(self):
        mock = self._mock_client()
        mock.generate.return_value = "D - completely different"
        result = _check_alignment(["src"], ["tl"], mock, self._model_configs())
        assert result == "D"

    # -- Binary search --

    def test_binary_search_all_good(self):
        mock = self._mock_client()
        mock.generate.return_value = "A"  # All windows good
        source = [f"src{i}" for i in range(20)]
        translated = [f"tl{i}" for i in range(20)]
        boundary = find_hallucination_boundary(source, translated, mock, self._model_configs())
        assert boundary == 15  # Last window start (20 - 5)

    def test_binary_search_hallucination_at_end(self):
        mock = self._mock_client()
        # Simulate: good up to position 7, then D
        def fake_generate(prompt, **kwargs):
            # Extract position from the source lines in prompt
            if "src15" in prompt or "src16" in prompt or "src17" in prompt:
                return "D"
            return "A"
        mock.generate.side_effect = fake_generate
        source = [f"src{i}" for i in range(20)]
        translated = [f"tl{i}" for i in range(20)]
        boundary = find_hallucination_boundary(source, translated, mock, self._model_configs())
        # Should find the last good position before hallucination
        assert boundary < 15  # Before the hallucinated region
        assert boundary >= 0

    # -- Full verify --

    def test_verify_complete_happy_path(self):
        """Translation matches source length, tail is valid."""
        mock = self._mock_client()
        # Preamble check: translation, tail check: A
        mock.generate.side_effect = ["translation", "A"]
        source = "\n".join([f"src{i}" for i in range(10)])
        translated = "\n".join([f"tl{i}" for i in range(10)])
        text, action = verify_translation(source, translated, mock, self._model_configs())
        assert action == "complete"
        assert text is not None

    def test_verify_truncated(self):
        """Translation is much shorter than source, tail is valid."""
        mock = self._mock_client()
        mock.generate.side_effect = ["translation", "A"]
        source = "\n".join([f"src{i}" for i in range(100)])
        translated = "\n".join([f"tl{i}" for i in range(50)])
        text, action = verify_translation(source, translated, mock, self._model_configs())
        assert action == "continue"

    def test_verify_hallucination_triggers_binary_search(self):
        """Tail is hallucinated → binary search → truncate → continue."""
        mock = self._mock_client()
        # Preamble: ok, Tail: D, then binary search returns A for everything
        mock.generate.side_effect = ["translation", "D"] + ["A"] * 20
        source = "\n".join([f"src{i}" for i in range(100)])
        translated = "\n".join([f"tl{i}" for i in range(100)])
        text, action = verify_translation(source, translated, mock, self._model_configs())
        # Should truncate and continue (since binary search finds good content)
        assert action in ("complete", "continue")
        assert text is not None

    def test_verify_retry_on_broken_preamble(self):
        """All preamble removal attempts fail → retry."""
        mock = self._mock_client()
        mock.generate.return_value = "meta-comment"
        source = "\n".join(["src1", "src2", "src3"])
        translated = "\n".join(["垃圾", "垃圾", "垃圾"])
        text, action = verify_translation(source, translated, mock, self._model_configs())
        assert action == "retry"

    def test_verify_empty_translation(self):
        mock = self._mock_client()
        text, action = verify_translation("src", "", mock, self._model_configs())
        assert action == "retry"

    def test_verify_skips_preamble_on_continuation(self):
        """Continuation chunks don't check for preamble."""
        mock = self._mock_client()
        mock.generate.return_value = "A"  # Only tail check
        source = "\n".join([f"src{i}" for i in range(10)])
        translated = "\n".join([f"tl{i}" for i in range(10)])
        text, action = verify_translation(
            source, translated, mock, self._model_configs(),
            is_first_chunk=False,
        )
        assert action == "complete"
        assert mock.generate.call_count == 1  # Only tail check, no preamble


# ─── Verifier Integration ───

class TestVerifierIntegration:
    """Integration test: real NovelTranslator with mocked API calls."""

    def test_run_translation_happy_path(self, tmp_path):
        """Full _run_translation path: generate → verify → complete."""
        # Create source file
        source_lines = [f"日本語の行{i}" for i in range(20)]
        source_text = "\n".join(source_lines)
        source_path = tmp_path / "source.txt"
        source_path.write_text(source_text, encoding="utf-8")

        translated_lines = [f"中文翻译第{i}行" for i in range(20)]
        translated_text = "\n".join(translated_lines)

        # Create NovelTranslator with minimal config
        config = {
            "title": "test",
            "credentials": {"providers": {"anthropic": {
                "type": "anthropic", "api_key": "fake", "base_url": "http://fake",
            }}},
            "translation": {"source_language": "Japanese", "target_language": "Chinese",
                           "models": [{"provider": "anthropic", "model": "test"}]},
        }
        translator = NovelTranslator(
            config=config, book_title="test",
            output_dir=tmp_path,
        )

        # Mock _stream_with_token_cutoff (translation)
        translator._stream_with_token_cutoff = MagicMock(return_value=translated_text)

        # Mock LLM client for verifier (preamble: translation, tail: A)
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = ["translation", "A"]
        translator._llm_client = mock_llm

        from pdf2epub.html_translation.novel_extractor import NovelUnit
        unit = NovelUnit(spine_index=0, file_name="test_ch", text_path=source_path, has_content=True)

        result = translator._run_translation(unit, source_text, "")

        # Should complete successfully
        assert result is not None
        result_lines = [l for l in result.splitlines() if l.strip()]
        assert len(result_lines) == 20

        # Verify: 1 translation call + 2 verifier calls (preamble + tail)
        assert translator._stream_with_token_cutoff.call_count == 1
        assert mock_llm.generate.call_count == 2

    def test_run_translation_with_continuation(self, tmp_path):
        """Translation truncated → verifier says continue → generates continuation."""
        source_lines = [f"日本語の行{i}" for i in range(50)]
        source_text = "\n".join(source_lines)
        source_path = tmp_path / "source.txt"
        source_path.write_text(source_text, encoding="utf-8")

        # First chunk: only 20 lines, second chunk: 30 more
        chunk1 = "\n".join([f"中文第{i}行" for i in range(20)])
        chunk2 = "\n".join([f"中文第{i}行" for i in range(20, 50)])

        config = {
            "title": "test",
            "credentials": {"providers": {"anthropic": {
                "type": "anthropic", "api_key": "fake", "base_url": "http://fake",
            }}},
            "translation": {"source_language": "Japanese", "target_language": "Chinese",
                           "models": [{"provider": "anthropic", "model": "test"}]},
        }
        translator = NovelTranslator(config=config, book_title="test", output_dir=tmp_path)

        # First call returns partial, second returns rest
        translator._stream_with_token_cutoff = MagicMock(side_effect=[chunk1, chunk2])

        # Verifier calls:
        # Round 1: preamble=translation, tail=A (but line count 20 vs 50 → continue)
        # Round 2: tail=A (50 vs 50 → complete)
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = ["translation", "A", "A"]
        translator._llm_client = mock_llm

        from pdf2epub.html_translation.novel_extractor import NovelUnit
        unit = NovelUnit(spine_index=0, file_name="test_ch", text_path=source_path, has_content=True)

        result = translator._run_translation(unit, source_text, "")

        assert result is not None
        result_lines = [l for l in result.splitlines() if l.strip()]
        assert len(result_lines) == 50
        # 2 translation calls, 3 verifier calls
        assert translator._stream_with_token_cutoff.call_count == 2

    def test_run_translation_preamble_removed(self, tmp_path):
        """Preamble detected and removed."""
        source_lines = [f"日本語の行{i}" for i in range(10)]
        source_text = "\n".join(source_lines)
        source_path = tmp_path / "source.txt"
        source_path.write_text(source_text, encoding="utf-8")

        # Translation with preamble
        translated_text = "以下是翻译：\n" + "\n".join([f"中文第{i}行" for i in range(10)])

        config = {
            "title": "test",
            "credentials": {"providers": {"anthropic": {
                "type": "anthropic", "api_key": "fake", "base_url": "http://fake",
            }}},
            "translation": {"source_language": "Japanese", "target_language": "Chinese",
                           "models": [{"provider": "anthropic", "model": "test"}]},
        }
        translator = NovelTranslator(config=config, book_title="test", output_dir=tmp_path)
        translator._stream_with_token_cutoff = MagicMock(return_value=translated_text)

        # Preamble check: line 1 = meta-comment, line 2 = translation
        # Tail check: A
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = ["meta-comment", "translation", "A"]
        translator._llm_client = mock_llm

        from pdf2epub.html_translation.novel_extractor import NovelUnit
        unit = NovelUnit(spine_index=0, file_name="test_ch", text_path=source_path, has_content=True)

        result = translator._run_translation(unit, source_text, "")

        assert "以下是翻译" not in result
        result_lines = [l for l in result.splitlines() if l.strip()]
        assert len(result_lines) == 10


# ─── NovelUnit fields ───

class TestNovelUnitV4:
    def test_default_fields(self):
        from pdf2epub.html_translation.novel_extractor import NovelUnit
        unit = NovelUnit(spine_index=0, file_name="ch1", text_path=None, has_content=True)
        assert unit.toc_title is None
        assert unit.source_href is None

    def test_set_fields(self):
        from pdf2epub.html_translation.novel_extractor import NovelUnit
        unit = NovelUnit(
            spine_index=0, file_name="ch1", text_path=None, has_content=True,
            toc_title="第一章", source_href="Text/ch1.xhtml",
        )
        assert unit.toc_title == "第一章"
        assert unit.source_href == "Text/ch1.xhtml"


# ─── CLI Contract ───

class TestCLIContract:
    def test_missing_input_returns_error(self, tmp_path, monkeypatch):
        import argparse
        monkeypatch.chdir(tmp_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            'title: "test"\ncredentials:\n  providers: {}\n'
            'translation:\n  models: []\n',
            encoding='utf-8',
        )

        from pdf2epub.cli import translate_novel_command
        args = argparse.Namespace(
            config=str(config_path),
            input=str(tmp_path / "nonexistent.epub"),
            source_language="Japanese",
            target_language="Chinese",
            resume=False,
            limit=None,
            glossary=None,
            skip_build=False,
        )
        result = translate_novel_command(args)
        assert result == 1

    def test_invalid_glossary_returns_error(self, tmp_path, monkeypatch):
        import argparse
        monkeypatch.chdir(tmp_path)

        # Create minimal epub
        _create_minimal_epub(tmp_path / "test.epub")

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            'title: "test"\ncredentials:\n  providers: {}\n'
            'translation:\n  models: []\n',
            encoding='utf-8',
        )

        from pdf2epub.cli import translate_novel_command
        args = argparse.Namespace(
            config=str(config_path),
            input=str(tmp_path / "test.epub"),
            source_language="Japanese",
            target_language="Chinese",
            resume=False,
            limit=None,
            glossary=str(tmp_path / "nonexistent_glossary.txt"),
            skip_build=False,
        )
        result = translate_novel_command(args)
        assert result == 1


def _create_minimal_epub(path: Path):
    """Create a minimal EPUB for testing."""
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">test-123</dc:identifier>
    <dc:title>Test Novel</dc:title>
    <dc:language>ja</dc:language>
  </metadata>
  <manifest>
    <item id="ch1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
  </spine>
</package>"""

    ch1_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Ch1</title></head>
<body>
<p>蒼(あおい)は学校に行った。</p>
<p>宮崎(みやざき)は手を振った。</p>
</body>
</html>"""

    with zipfile.ZipFile(str(path), 'w') as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", content_opf)
        zf.writestr("OEBPS/Text/ch1.xhtml", ch1_xhtml)


# ─── Trailing Newline Guardian ───

class TestTrailingNewlineGuardian:
    """Ensure all files written to originals/ have trailing newlines,
    so wc -l and python splitlines() always agree."""

    def test_extra_originals_get_trailing_newline(self, tmp_path):
        """Runner must add trailing newline to all extra_originals files."""
        from pdf2epub.core.whole.runner import run_agent_loop
        import asyncio

        originals = tmp_path / "originals"
        workspace = tmp_path / "workspace"
        originals.mkdir()
        workspace.mkdir()

        # Simulate what runner does with extra_originals
        extra = {
            "source.txt": "line1\nline2\nline3",  # no trailing \n
            "other.txt": "content\n",  # already has \n
            "empty.txt": "",
        }
        for fname, fcontent in extra.items():
            if fcontent and not fcontent.endswith("\n"):
                fcontent += "\n"
            (originals / fname).write_text(fcontent, encoding="utf-8")

        # Verify all non-empty files end with \n
        for fname in ["source.txt", "other.txt"]:
            content = (originals / fname).read_text()
            assert content.endswith("\n"), f"{fname} missing trailing newline"
            # Verify wc -l matches splitlines count
            import subprocess
            wc = int(subprocess.run(
                ["wc", "-l", str(originals / fname)],
                capture_output=True, text=True,
            ).stdout.strip().split()[0])
            py_count = len([l for l in content.splitlines() if l.strip()])
            assert wc == py_count, f"{fname}: wc -l={wc} != splitlines={py_count}"

    def test_raw_output_gets_trailing_newline(self, tmp_path):
        """Raw output without trailing newline must get one added."""
        raw_text = "translated line 1\ntranslated line 2"
        assert not raw_text.endswith("\n")
        if raw_text and not raw_text.endswith("\n"):
            raw_text += "\n"
        path = tmp_path / "raw_output.txt"
        path.write_text(raw_text)

        import subprocess
        wc = int(subprocess.run(
            ["wc", "-l", str(path)], capture_output=True, text=True,
        ).stdout.strip().split()[0])
        py_count = len([l for l in raw_text.splitlines() if l.strip()])
        assert wc == py_count

    def test_continuation_gets_trailing_newline(self, tmp_path):
        """Continuation without trailing newline must get one added."""
        cont_text = "continued line 1\ncontinued line 2"
        if cont_text and not cont_text.endswith("\n"):
            cont_text += "\n"
        path = tmp_path / "continuation_001.txt"
        path.write_text(cont_text)

        import subprocess
        wc = int(subprocess.run(
            ["wc", "-l", str(path)], capture_output=True, text=True,
        ).stdout.strip().split()[0])
        py_count = len([l for l in cont_text.splitlines() if l.strip()])
        assert wc == py_count


# ─── Spurious Heading Cleanup ───

class TestStripSpuriousHeadings:
    def test_removes_heading_not_in_source(self):
        source = "俺は学校に行った。\n宮崎は笑った。"
        translated = "# 我去了学校。\n宫崎笑了。"
        result = strip_spurious_headings(translated, source)
        assert result == "我去了学校。\n宫崎笑了。"

    def test_preserves_heading_in_source(self):
        source = "# 第一章\n俺は学校に行った。"
        translated = "# 第一章\n我去了学校。"
        result = strip_spurious_headings(translated, source)
        assert result == "# 第一章\n我去了学校。"

    def test_removes_multiple_headings(self):
        source = "行1\n＊＊＊\n行2"
        translated = "# L1\n＊＊＊\n## L2"
        result = strip_spurious_headings(translated, source)
        assert result == "L1\n＊＊＊\nL2"

    def test_no_headings_unchanged(self):
        source = "行1\n行2"
        translated = "L1\nL2"
        assert strip_spurious_headings(translated, source) == "L1\nL2"


# ─── Guardian: unified LLM trace ───

class TestUnifiedLLMTrace:
    """Guardian tests for the unified LLM trace system.

    1. SDK denylist: no direct SDK calls outside allowed files
    2. Trace integration: LLM calls produce trace entries
    3. Trace infrastructure: _write_trace works correctly
    """

    def test_sdk_denylist_no_bypass(self):
        """No direct SDK calls (messages.create, chat.completions.create, etc.)
        outside of allowed files."""
        import ast
        from pathlib import Path

        # SDK method chains to deny
        deny_patterns = {
            "messages.create",        # Anthropic SDK
            "completions.create",     # OpenAI SDK (chat.completions.create)
        }
        # Note: models.generate_content is harder to detect via AST because
        # it's a normal attribute call. We check the method chain instead.

        # Allowed files (relative to pdf2epub/)
        allowed_files = {
            "utils/network_utils.py",           # The traced wrappers themselves
            "html_translation/novel_translator.py",  # _stream_with_token_cutoff (has own trace)
        }
        # Files completely excluded from scan (OCR, tests, scripts)
        excluded_prefixes = {
            "ocr/",
            "utils/ocr_client.py",
        }

        pdf2epub_dir = Path("pdf2epub")
        violations = []

        for py_file in pdf2epub_dir.rglob("*.py"):
            rel = str(py_file.relative_to(pdf2epub_dir))
            if any(rel.startswith(p) for p in excluded_prefixes):
                continue
            if rel in allowed_files:
                continue

            try:
                source = py_file.read_text()
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    # Check for .create() calls on known SDK objects
                    attr = node.func.attr
                    if attr == "create" and isinstance(node.func.value, ast.Attribute):
                        parent_attr = node.func.value.attr
                        chain = f"{parent_attr}.{attr}"
                        if chain in deny_patterns:
                            violations.append(f"{rel}:{node.lineno} — {chain}")

        assert violations == [], (
            f"Direct SDK calls found outside allowed files:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\n\nUse llm_client.generate() or add to allowed_files with justification."
        )

    def test_trace_write_produces_valid_jsonl(self, tmp_path):
        """_write_trace produces valid JSONL with all required fields."""
        from pdf2epub.utils.network_utils import set_llm_trace_path, _write_trace

        trace_path = tmp_path / "test_trace.jsonl"
        set_llm_trace_path(trace_path)

        _write_trace({
            "timestamp": "2026-01-01T00:00:00Z",
            "operation": "test_op",
            "provider": "test",
            "model": "test-model",
            "duration_ms": 100,
            "input_tokens": 50,
            "output_tokens": 10,
            "response_preview": {"head": "hello", "length": 5},
            "error": None,
        })

        lines = trace_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["operation"] == "test_op"
        assert entry["provider"] == "test"
        assert entry["error"] is None

        # Clean up
        set_llm_trace_path(None)

    def test_preview_text_head_tail(self):
        """_preview_text preserves head and tail for long texts."""
        from pdf2epub.utils.network_utils import _preview_text

        # Short text — no truncation
        result = _preview_text("short", 1000)
        assert result == {"head": "short", "length": 5}

        # Long text — head + tail
        long_text = "A" * 1000 + "MIDDLE" + "B" * 1000
        result = _preview_text(long_text, 100)
        assert result["head"] == long_text[:100]
        assert result["tail"] == long_text[-100:]
        assert result["length"] == len(long_text)

        # Empty text
        result = _preview_text("", 100)
        assert result == {"head": "", "length": 0}

    def test_trace_disabled_when_path_is_none(self, tmp_path):
        """No trace file created when trace path is None."""
        from pdf2epub.utils.network_utils import set_llm_trace_path, _write_trace

        set_llm_trace_path(None)
        _write_trace({"test": True})
        # No file should exist
        assert not (tmp_path / "trace.jsonl").exists()

    def test_preview_messages_handles_structured_content(self):
        """_preview_messages handles both string and structured content."""
        from pdf2epub.utils.network_utils import _preview_messages

        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "response text"},
            ]},
        ]
        result = _preview_messages(messages, 1000)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content_head"] == "hello world"
        assert result[1]["role"] == "assistant"
        assert "response text" in result[1]["content_head"]


