"""Tests for adaptive_pdf_call module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2epub.refine.adaptive_pdf_call import (
    AdaptivePdfCall,
    PdfPageLimitLearner,
    is_503_error,
    run_adaptive_batches,
    split_pages_into_batches,
    validate_chapter_structure,
)


# --- is_503_error ---

class TestIs503Error:
    def test_503_in_message(self):
        assert is_503_error(Exception("503 Service Unavailable"))

    def test_unavailable_in_message(self):
        assert is_503_error(Exception("UNAVAILABLE: server overloaded"))

    def test_unrelated_error(self):
        assert not is_503_error(Exception("404 Not Found"))

    def test_empty_error(self):
        assert not is_503_error(Exception(""))

    def test_500_not_matched(self):
        assert not is_503_error(Exception("500 Internal Server Error"))


# --- PdfPageLimitLearner ---

class TestPdfPageLimitLearner:
    def test_initial_state(self):
        learner = PdfPageLimitLearner(initial_limit=900, min_limit=50)
        assert learner.limit == 900
        assert learner.min_limit == 50
        assert learner.had_503 is False

    def test_report_503_halves_limit(self):
        learner = PdfPageLimitLearner(initial_limit=900)
        new = learner.report_503(900)
        assert new == 450
        assert learner.limit == 450
        assert learner.had_503 is True

    def test_report_503_takes_minimum_of_current_and_new(self):
        learner = PdfPageLimitLearner(initial_limit=900)
        learner.report_503(900)  # limit → 450
        # Now report 503 at 200 pages (e.g. from a different call)
        learner.report_503(200)  # 200//2=100, min(450,100)=100
        assert learner.limit == 100

    def test_report_503_does_not_increase_limit(self):
        learner = PdfPageLimitLearner(initial_limit=900)
        learner.report_503(200)  # limit → 100
        learner.report_503(800)  # 800//2=400, min(100,400)=100 (no increase)
        assert learner.limit == 100

    def test_report_503_below_minimum_raises(self):
        learner = PdfPageLimitLearner(initial_limit=900, min_limit=50)
        with pytest.raises(RuntimeError, match="below minimum"):
            learner.report_503(80)  # 80//2=40 < 50

    def test_report_503_exactly_at_minimum(self):
        learner = PdfPageLimitLearner(initial_limit=900, min_limit=50)
        new = learner.report_503(100)  # 100//2=50 == min_limit
        assert new == 50

    def test_report_success_no_op(self):
        learner = PdfPageLimitLearner(initial_limit=900)
        learner.report_503(900)  # limit → 450
        learner.report_success(450)
        assert learner.limit == 450  # unchanged

    def test_learned_limit_persists_across_calls(self):
        """The limit learned from one operation carries to the next."""
        learner = PdfPageLimitLearner(initial_limit=900)
        learner.report_503(900)  # limit → 450
        # New operation starts with the learned limit
        assert learner.limit == 450


# --- split_pages_into_batches ---

class TestSplitPagesIntoBatches:
    def test_empty(self):
        assert split_pages_into_batches([], 100) == []

    def test_fits_in_one_batch(self):
        pages = [1, 2, 3, 4, 5]
        result = split_pages_into_batches(pages, 10)
        assert result == [[1, 2, 3, 4, 5]]

    def test_exact_batch_size(self):
        pages = [1, 2, 3, 4, 5]
        result = split_pages_into_batches(pages, 5)
        assert result == [[1, 2, 3, 4, 5]]

    def test_splits_evenly(self):
        pages = list(range(1, 11))  # 10 pages
        result = split_pages_into_batches(pages, 5)
        assert result == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

    def test_splits_with_remainder(self):
        pages = list(range(1, 8))  # 7 pages
        result = split_pages_into_batches(pages, 3)
        assert result == [[1, 2, 3], [4, 5, 6], [7]]

    def test_overlap(self):
        pages = list(range(1, 11))  # 10 pages
        result = split_pages_into_batches(pages, 5, overlap=2)
        # batch1: [1,2,3,4,5], start moves to 5-2=3
        # batch2: [4,5,6,7,8], start moves to 8-2=6
        # batch3: [7,8,9,10]
        assert result == [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10]]

    def test_overlap_single_page_batch(self):
        pages = [1, 2, 3]
        result = split_pages_into_batches(pages, 2, overlap=1)
        # batch1: [1,2], start → 2-1=1
        # batch2: [2,3], start → 3 (end)
        assert result == [[1, 2], [2, 3]]

    def test_batch_size_one(self):
        pages = [10, 20, 30]
        result = split_pages_into_batches(pages, 1)
        assert result == [[10], [20], [30]]


# --- run_adaptive_batches ---

class TestRunAdaptiveBatches:
    def test_single_batch_success(self):
        """All pages fit in one batch, succeeds."""
        learner = PdfPageLimitLearner(initial_limit=100)
        calls = []

        def process(batch, idx, total, use_rasterized=False):
            calls.append((batch, idx, total))
            return f"result_{idx}"

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error, "test"
        )
        assert results == ["result_0"]
        assert len(calls) == 1
        assert calls[0] == (list(range(1, 11)), 0, 1)

    def test_multiple_batches_no_errors(self):
        """Pages split into multiple batches, all succeed."""
        learner = PdfPageLimitLearner(initial_limit=5)
        results = run_adaptive_batches(
            list(range(1, 13)),
            lambda batch, idx, total, use_rasterized=False: len(batch),
            learner, is_503_error, "test"
        )
        assert results == [5, 5, 2]

    def test_503_triggers_resplit(self):
        """503 on first attempt → halve limit → retry succeeds."""
        learner = PdfPageLimitLearner(initial_limit=10, min_limit=2)
        attempt_count = [0]

        def process(batch, idx, total, use_rasterized=False):
            attempt_count[0] += 1
            if len(batch) > 5:
                raise Exception("503 UNAVAILABLE")
            return batch

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error, "test"
        )
        # First attempt: 10 pages → 503
        # Re-split into 2 batches of 5
        assert learner.limit == 5
        assert len(results) == 2
        # All pages covered
        all_pages = []
        for r in results:
            all_pages.extend(r)
        assert sorted(set(all_pages)) == list(range(1, 11))

    def test_503_preserves_earlier_results(self):
        """503 on second batch preserves first batch's result."""
        learner = PdfPageLimitLearner(initial_limit=5, min_limit=2)
        call_count = [0]

        def process(batch, idx, total, use_rasterized=False):
            call_count[0] += 1
            # Second batch fails first time (call_count==2), succeeds after re-split
            if call_count[0] == 2 and len(batch) == 5:
                raise Exception("503")
            return sum(batch)

        pages = list(range(1, 11))  # 10 pages, limit 5 → 2 batches
        results = run_adaptive_batches(
            pages, process, learner, is_503_error, "test"
        )
        # First batch [1..5] → sum=15 (succeeds)
        # Second batch [6..10] → 503 → re-split to [6,7] [8,9,10] (limit=2)
        assert results[0] == 15  # first batch preserved
        assert len(results) >= 2

    def test_non_503_error_propagated(self):
        """Non-503 errors are re-raised, not caught."""
        learner = PdfPageLimitLearner(initial_limit=100)

        def process(batch, idx, total, use_rasterized=False):
            raise ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            run_adaptive_batches(
                [1, 2, 3], process, learner, is_503_error, "test"
            )

    def test_503_below_min_raises(self):
        """Repeated 503s until below minimum → RuntimeError."""
        learner = PdfPageLimitLearner(initial_limit=100, min_limit=50)

        def always_503(batch, idx, total, use_rasterized=False):
            raise Exception("503")

        with pytest.raises(RuntimeError, match="below minimum"):
            run_adaptive_batches(
                list(range(1, 101)), always_503, learner, is_503_error, "test"
            )

    def test_overlap_preserved_after_resplit(self):
        """After 503 re-split, overlap is maintained."""
        learner = PdfPageLimitLearner(initial_limit=10, min_limit=2)
        call_sizes = []

        def process(batch, idx, total, use_rasterized=False):
            call_sizes.append(len(batch))
            if len(batch) > 5:
                raise Exception("503 UNAVAILABLE")
            return batch

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error,
            "test", overlap=2
        )
        # After 503 at 10, limit → 5, re-split with overlap=2
        # All pages should be covered
        all_pages = set()
        for r in results:
            all_pages.update(r)
        assert all_pages == set(range(1, 11))

    def test_learner_limit_shared_across_operations(self):
        """Learner from one run_adaptive_batches carries to the next."""
        learner = PdfPageLimitLearner(initial_limit=100, min_limit=10)

        def fail_big(batch, idx, total, use_rasterized=False):
            if len(batch) > 25:
                raise Exception("503")
            return "ok"

        # First operation: learns limit=50 then 25
        run_adaptive_batches(
            list(range(1, 101)), fail_big, learner, is_503_error, "op1"
        )
        assert learner.limit == 25

        # Second operation: starts with learned limit=25
        call_batches = []

        def record(batch, idx, total, use_rasterized=False):
            call_batches.append(batch)
            return "ok"

        run_adaptive_batches(
            list(range(1, 51)), record, learner, is_503_error, "op2"
        )
        # Should start with batches of 25, not 100
        assert all(len(b) <= 25 for b in call_batches)

    def test_503_tries_rasterization_first(self):
        """503 with can_rasterize=True tries rasterized version first."""
        learner = PdfPageLimitLearner(initial_limit=100)
        calls = []

        def process(batch, idx, total, use_rasterized=False):
            calls.append((len(batch), use_rasterized))
            # First call (not rasterized) fails with 503
            # Second call (rasterized) succeeds
            if not use_rasterized:
                raise Exception("503 UNAVAILABLE")
            return "ok"

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error, "test",
            can_rasterize=True
        )
        # Should have tried normal first, then rasterized
        assert calls == [(10, False), (10, True)]
        assert results == ["ok"]
        # Limit should NOT have been reduced (rasterization succeeded)
        assert learner.limit == 100

    def test_503_rasterization_per_batch(self):
        """Each batch gets its own chance to try rasterization."""
        learner = PdfPageLimitLearner(initial_limit=5)
        calls = []

        def process(batch, idx, total, use_rasterized=False):
            calls.append((idx, use_rasterized))
            # Both batches fail without rasterization, succeed with it
            if not use_rasterized:
                raise Exception("503 UNAVAILABLE")
            return f"batch_{idx}"

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error, "test",
            can_rasterize=True
        )
        # Each batch should try normal first, then rasterized
        assert (0, False) in calls
        assert (0, True) in calls
        assert (1, False) in calls
        assert (1, True) in calls
        assert results == ["batch_0", "batch_1"]
        # Limit unchanged since rasterization succeeded
        assert learner.limit == 5

    def test_503_rasterization_fails_then_splits(self):
        """503 with rasterization also failing → fall back to split."""
        learner = PdfPageLimitLearner(initial_limit=10, min_limit=2)
        calls = []

        def process(batch, idx, total, use_rasterized=False):
            calls.append((len(batch), use_rasterized))
            # Always 503 for batches > 5 pages
            if len(batch) > 5:
                raise Exception("503 UNAVAILABLE")
            return batch

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error, "test",
            can_rasterize=True
        )
        # First call: 10 pages, not rasterized → 503
        # Second call: 10 pages, rasterized → 503
        # Then split to 5+5
        assert (10, False) in calls
        assert (10, True) in calls
        assert learner.limit == 5
        # Final results from split batches
        all_pages = []
        for r in results:
            all_pages.extend(r)
        assert sorted(set(all_pages)) == list(range(1, 11))

    def test_no_rasterization_when_disabled(self):
        """503 without can_rasterize goes straight to split."""
        learner = PdfPageLimitLearner(initial_limit=10, min_limit=2)
        calls = []

        def process(batch, idx, total, use_rasterized=False):
            calls.append((len(batch), use_rasterized))
            if len(batch) > 5:
                raise Exception("503 UNAVAILABLE")
            return batch

        results = run_adaptive_batches(
            list(range(1, 11)), process, learner, is_503_error, "test",
            can_rasterize=False
        )
        # Should never call with use_rasterized=True
        assert all(not r for _, r in calls)
        assert learner.limit == 5


# --- validate_chapter_structure ---

class TestValidateChapterStructure:
    def test_valid_chapters(self):
        chapters = [
            {'title': 'Ch1', 'start_page': 1, 'end_page': 10},
            {'title': 'Ch2', 'start_page': 11, 'end_page': 20},
        ]
        assert validate_chapter_structure(chapters) == []

    def test_missing_start_page(self):
        chapters = [{'title': 'Ch1', 'end_page': 10}]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 1
        assert 'Missing start_page' in issues[0]

    def test_missing_end_page(self):
        chapters = [{'title': 'Ch1', 'start_page': 1}]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 1
        assert 'Missing end_page' in issues[0]

    def test_end_before_start(self):
        chapters = [{'title': 'Ch1', 'start_page': 20, 'end_page': 10}]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 1
        assert 'end < start' in issues[0]

    def test_sibling_overlap(self):
        chapters = [
            {'title': 'Ch1', 'start_page': 1, 'end_page': 15},
            {'title': 'Ch2', 'start_page': 10, 'end_page': 20},
        ]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 1
        assert 'Overlap' in issues[0]

    def test_adjacent_siblings_ok(self):
        """end_page == next start_page - 1 is fine."""
        chapters = [
            {'title': 'Ch1', 'start_page': 1, 'end_page': 9},
            {'title': 'Ch2', 'start_page': 10, 'end_page': 20},
        ]
        assert validate_chapter_structure(chapters) == []

    def test_shared_page_boundary_ok(self):
        """end_page == next start_page is fine (sections share a page)."""
        chapters = [
            {'title': 'Section A', 'start_page': 13, 'end_page': 15},
            {'title': 'Section B', 'start_page': 15, 'end_page': 17},
            {'title': 'Section C', 'start_page': 17, 'end_page': 20},
        ]
        assert validate_chapter_structure(chapters) == []

    def test_real_overlap_still_caught(self):
        """end_page > next start_page is a real overlap."""
        chapters = [
            {'title': 'Romans', 'start_page': 81, 'end_page': 900},
            {'title': 'La Mort du roi Arthur', 'start_page': 849, 'end_page': 991},
        ]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 1
        assert 'Overlap' in issues[0]

    def test_children_validated_recursively(self):
        chapters = [
            {
                'title': 'Part 1', 'start_page': 1, 'end_page': 30,
                'children': [
                    {'title': 'Ch1', 'start_page': 1, 'end_page': 20},
                    {'title': 'Ch2', 'start_page': 15, 'end_page': 30},  # overlap
                ]
            },
        ]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 1
        assert 'Overlap' in issues[0]
        assert 'Ch1' in issues[0]

    def test_multiple_issues(self):
        chapters = [
            {'title': 'Bad', 'start_page': 20, 'end_page': 5},   # end < start
            {'title': 'Missing'},                                   # missing both
        ]
        issues = validate_chapter_structure(chapters)
        assert len(issues) == 3  # end<start + missing start + missing end

    def test_empty_list(self):
        assert validate_chapter_structure([]) == []

    def test_single_page_chapter(self):
        chapters = [{'title': 'Intro', 'start_page': 5, 'end_page': 5}]
        assert validate_chapter_structure(chapters) == []


# --- _filter_edge_issues ---

class TestFilterEdgeIssues:
    """Test AdaptivePdfCall._filter_edge_issues."""

    def _make_call(self):
        call = AdaptivePdfCall.__new__(AdaptivePdfCall)
        return call

    def test_single_batch_no_filtering(self):
        """Single batch (total=1): all issues kept, no edge tolerance."""
        call = self._make_call()
        chapters = [
            {'title': 'Ch1', 'start_page': 1, 'end_page': 10},
            {'title': 'Ch2', 'start_page': 5, 'end_page': 20},
        ]
        issues = ["Overlap: 'Ch1' ends at p10 but 'Ch2' starts at p5"]
        result = call._filter_edge_issues(issues, chapters, batch_idx=0, total_batches=1)
        assert result == issues

    def test_first_batch_filters_last_chapter(self):
        """First batch of many: last chapter's issues tolerated."""
        call = self._make_call()
        chapters = [
            {'title': 'Ch1', 'start_page': 1, 'end_page': 10},
            {'title': 'Ch2', 'start_page': 5, 'end_page': 20},
        ]
        issues = [
            "Overlap: 'Ch1' ends at p10 but 'Ch2' starts at p5",
            "Missing end_page: Ch1",
        ]
        result = call._filter_edge_issues(issues, chapters, batch_idx=0, total_batches=3)
        # Ch2 is last → overlap issue mentioning Ch2 is filtered
        # Ch1 issue stays (Ch1 is not an edge in first batch)
        assert len(result) == 1
        assert "Missing end_page: Ch1" in result[0]

    def test_last_batch_filters_first_chapter(self):
        """Last batch of many: first chapter's issues tolerated."""
        call = self._make_call()
        chapters = [
            {'title': 'Ch3', 'start_page': 50, 'end_page': 60},
            {'title': 'Ch4', 'start_page': 61, 'end_page': 70},
        ]
        issues = ["Missing start_page: Ch3"]
        result = call._filter_edge_issues(issues, chapters, batch_idx=2, total_batches=3)
        assert result == []

    def test_middle_batch_filters_both_edges(self):
        """Middle batch: both first and last chapter issues tolerated."""
        call = self._make_call()
        chapters = [
            {'title': 'ChA', 'start_page': 20, 'end_page': 30},
            {'title': 'ChB', 'start_page': 31, 'end_page': 40},
            {'title': 'ChC', 'start_page': 41, 'end_page': 50},
        ]
        issues = [
            "Missing start_page: ChA",
            "Invalid range (end < start): ChB (p31-p30)",
            "Missing end_page: ChC",
        ]
        result = call._filter_edge_issues(issues, chapters, batch_idx=1, total_batches=3)
        # ChA (first) and ChC (last) filtered; ChB stays
        assert len(result) == 1
        assert "ChB" in result[0]

    def test_empty_chapters_no_filtering(self):
        call = self._make_call()
        issues = ["some issue"]
        result = call._filter_edge_issues(issues, [], batch_idx=1, total_batches=3)
        assert result == issues

    def test_first_batch_keeps_first_chapter_issues(self):
        """First batch: first chapter is NOT an edge (no previous batch)."""
        call = self._make_call()
        chapters = [
            {'title': 'Ch1', 'start_page': 5, 'end_page': 3},
            {'title': 'Ch2', 'start_page': 10, 'end_page': 20},
        ]
        issues = ["Invalid range (end < start): Ch1 (p5-p3)"]
        result = call._filter_edge_issues(issues, chapters, batch_idx=0, total_batches=3)
        assert result == issues

    def test_last_batch_keeps_last_chapter_issues(self):
        """Last batch: last chapter is NOT an edge (no next batch)."""
        call = self._make_call()
        chapters = [
            {'title': 'Ch9', 'start_page': 80, 'end_page': 90},
            {'title': 'Ch10', 'start_page': 91, 'end_page': 85},
        ]
        issues = ["Invalid range (end < start): Ch10 (p91-p85)"]
        result = call._filter_edge_issues(issues, chapters, batch_idx=2, total_batches=3)
        assert result == issues


# --- Batch validation retry ---

def _make_test_call_cls(validate_fn=None):
    """Create a testable AdaptivePdfCall subclass."""

    class TestCall(AdaptivePdfCall):
        operation_name = "test"
        overlap = 0
        batch_validation_retries = 2

        def build_prompt(self, batch_pages, batch_idx, total_batches):
            return f"analyze pages {min(batch_pages)}-{max(batch_pages)}"

        def validate_batch_result(self, result, batch_idx, total_batches):
            if validate_fn:
                return validate_fn(result, batch_idx, total_batches)
            return []

        def merge_results(self, results):
            if len(results) == 1:
                return results[0]
            return {'merged': True, 'chapters': []}

    return TestCall


class TestBatchValidationRetry:
    """Test batch validation triggers retry with PDF context."""

    @patch('pdf2epub.refine.adaptive_pdf_call.Part')
    def test_no_issues_no_retry(self, mock_part):
        """Clean result → 1 LLM call, no retry."""
        TestCall = _make_test_call_cls()
        client = MagicMock()
        client.generate_content_stream.return_value = '{"chapters": []}'

        call = TestCall(
            client=client, model="m",
            prepare_pdf=lambda path, include_pages: b"pdf",
            learner=PdfPageLimitLearner(initial_limit=100),
        )
        call.run(Path("/f.pdf"), [1, 2, 3])
        assert client.generate_content_stream.call_count == 1

    @patch('pdf2epub.refine.adaptive_pdf_call.Part')
    def test_issues_trigger_retry_with_repair_prompt(self, mock_part):
        """Validation issues → retry, repair prompt contains error text."""
        call_num = [0]

        def validate(result, batch_idx, total_batches):
            call_num[0] += 1
            if call_num[0] <= 1:
                return ["Overlap: 'Ch1' ends at p10 but 'Ch2' starts at p5"]
            return []

        TestCall = _make_test_call_cls(validate_fn=validate)
        client = MagicMock()
        client.generate_content_stream.return_value = (
            '{"chapters": [{"title":"Ch1","start_page":1,"end_page":10},'
            '{"title":"Ch2","start_page":5,"end_page":20}]}'
        )

        call = TestCall(
            client=client, model="m",
            prepare_pdf=lambda path, include_pages: b"pdf",
            learner=PdfPageLimitLearner(initial_limit=100),
        )
        call.run(Path("/f.pdf"), [1, 2, 3])

        assert client.generate_content_stream.call_count == 2
        # Second call's prompt should contain repair text
        second_kwargs = client.generate_content_stream.call_args_list[1].kwargs
        second_prompt = str(second_kwargs['contents'][0])
        assert "STRUCTURAL ERRORS" in second_prompt

    @patch('pdf2epub.refine.adaptive_pdf_call.Part')
    def test_max_retries_exhausted(self, mock_part):
        """Persistent issues → exhausts retries, returns last result."""
        def always_fail(result, batch_idx, total_batches):
            return ["Persistent issue"]

        TestCall = _make_test_call_cls(validate_fn=always_fail)
        client = MagicMock()
        client.generate_content_stream.return_value = (
            '{"chapters": [{"title":"Ch1","start_page":1,"end_page":10}]}'
        )

        call = TestCall(
            client=client, model="m",
            prepare_pdf=lambda path, include_pages: b"pdf",
            learner=PdfPageLimitLearner(initial_limit=100),
        )
        result = call.run(Path("/f.pdf"), [1, 2, 3])

        # 1 initial + 2 retries = 3
        assert client.generate_content_stream.call_count == 3
        assert result is not None

    @patch('pdf2epub.refine.adaptive_pdf_call.Part')
    def test_edge_issues_tolerated_no_retry(self, mock_part):
        """Edge-only issues in multi-batch don't trigger retry."""
        def validate(result, batch_idx, total_batches):
            # First batch: issue on last chapter (edge)
            if batch_idx == 0:
                return ["Missing end_page: LastCh"]
            return []

        TestCall = _make_test_call_cls(validate_fn=validate)
        client = MagicMock()
        client.generate_content_stream.return_value = (
            '{"chapters": [{"title":"Ch1","start_page":1,"end_page":3},'
            '{"title":"LastCh","start_page":4,"end_page":5}]}'
        )

        call = TestCall(
            client=client, model="m",
            prepare_pdf=lambda path, include_pages: b"pdf",
            learner=PdfPageLimitLearner(initial_limit=5),
        )
        call.run(Path("/f.pdf"), list(range(1, 11)))

        # 2 batches, each called once (edge issue tolerated, no retry)
        assert client.generate_content_stream.call_count == 2

    @patch('pdf2epub.refine.adaptive_pdf_call.Part')
    def test_non_edge_issues_retry_in_multi_batch(self, mock_part):
        """Non-edge issues in multi-batch DO trigger retry."""
        call_num = [0]

        def validate(result, batch_idx, total_batches):
            call_num[0] += 1
            # First batch: issue on FIRST chapter (not an edge in first batch)
            if batch_idx == 0 and call_num[0] <= 1:
                return ["Invalid range (end < start): Ch1 (p5-p3)"]
            return []

        TestCall = _make_test_call_cls(validate_fn=validate)
        client = MagicMock()
        client.generate_content_stream.return_value = (
            '{"chapters": [{"title":"Ch1","start_page":5,"end_page":3},'
            '{"title":"Ch2","start_page":6,"end_page":10}]}'
        )

        call = TestCall(
            client=client, model="m",
            prepare_pdf=lambda path, include_pages: b"pdf",
            learner=PdfPageLimitLearner(initial_limit=5),
        )
        call.run(Path("/f.pdf"), list(range(1, 11)))

        # First batch: 1 initial + 1 retry = 2, second batch: 1 = total 3
        assert client.generate_content_stream.call_count == 3
