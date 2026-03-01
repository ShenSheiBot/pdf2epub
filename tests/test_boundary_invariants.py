"""Tests for boundary agent post-verification invariant enforcement."""

import pytest

from pdf2epub.refine.boundary_agent import (
    Section,
    _enforce_boundary_invariants,
    detect_boundary_issues,
    sections_to_toc_children,
)
from pdf2epub.refine.structure_analyzer import TOCNode


class TestEnforceBoundaryInvariants:
    """Tests for _enforce_boundary_invariants."""

    def test_empty_sections(self):
        """No crash on empty list."""
        _enforce_boundary_invariants([], 1, 100)

    def test_fix_end_before_start(self):
        """end_page < start_page is fixed, then gap closing extends further."""
        sections = [
            Section(title="Ch 1", start_page=10, end_page=50),
            Section(title="Ch 2", start_page=51, end_page=40),  # invalid
            Section(title="Ch 3", start_page=60, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 10, 100)
        # Fix 1 sets end_page=51, then Fix 4 closes gap to next section (59)
        assert sections[1].end_page == 59
        assert sections[1].end_line is None

    def test_first_child_extended_to_parent_start(self):
        """First child's start_page extended backward to parent's start."""
        sections = [
            Section(title="Ch 1", start_page=15, end_page=50),
            Section(title="Ch 2", start_page=51, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 10, 100)
        assert sections[0].start_page == 10
        assert sections[0].start_line is None

    def test_first_child_not_shrunk_past_parent(self):
        """First child already at or before parent start is not moved."""
        sections = [
            Section(title="Ch 1", start_page=10, end_page=50),
            Section(title="Ch 2", start_page=51, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 10, 100)
        assert sections[0].start_page == 10

    def test_last_child_extended_to_parent_end(self):
        """Last child's end_page extended to parent's end_page."""
        sections = [
            Section(title="Ch 1", start_page=10, end_page=50),
            Section(title="Ch 2", start_page=51, end_page=95),  # 5 pages short
        ]
        _enforce_boundary_invariants(sections, 10, 100)
        assert sections[-1].end_page == 100
        assert sections[-1].end_line is None

    def test_last_child_not_shrunk(self):
        """Last child already at parent end is not changed."""
        sections = [
            Section(title="Ch 1", start_page=10, end_page=50),
            Section(title="Ch 2", start_page=51, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 10, 100)
        assert sections[-1].end_page == 100

    def test_gap_closed_by_extending_previous(self):
        """Gaps between consecutive sections are closed."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=40),
            # Gap: pages 41-49
            Section(title="Ch 2", start_page=50, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 1, 100)
        assert sections[0].end_page == 49  # extended to close gap
        assert sections[0].end_line is None

    def test_no_gap_no_change(self):
        """Consecutive sections without gaps are not modified."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=49),
            Section(title="Ch 2", start_page=50, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 1, 100)
        assert sections[0].end_page == 49

    def test_shared_page_not_treated_as_gap(self):
        """Shared page (end == start) is not a gap, not modified."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=50),
            Section(title="Ch 2", start_page=50, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 1, 100)
        assert sections[0].end_page == 50  # unchanged

    def test_single_section_extends_both_ends(self):
        """Single section is extended to match parent bounds."""
        sections = [
            Section(title="Ch 1", start_page=5, end_page=95),
        ]
        _enforce_boundary_invariants(sections, 1, 100)
        assert sections[0].start_page == 1
        assert sections[0].end_page == 100

    def test_split_preserved_when_no_fix_needed(self):
        """Existing start_line/end_line preserved if no invariant fix needed."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=50, end_line=10),
            Section(title="Ch 2", start_page=50, end_page=100, start_line=10),
        ]
        _enforce_boundary_invariants(sections, 1, 100)
        assert sections[0].end_line == 10
        assert sections[1].start_line == 10

    def test_child_before_parent_clamped(self):
        """Child with start_page before parent is clamped to parent start."""
        sections = [
            Section(title="Why Civilization", start_page=13, end_page=26),
            Section(title="Postmodernity", start_page=27, end_page=50),
        ]
        _enforce_boundary_invariants(sections, 27, 50)
        assert sections[0].start_page == 27
        assert sections[0].start_line is None

    def test_child_after_parent_clamped(self):
        """Child with end_page after parent is clamped to parent end."""
        sections = [
            Section(title="Return to Sender", start_page=204, end_page=211),
            Section(title="End Matter", start_page=212, end_page=212),
        ]
        _enforce_boundary_invariants(sections, 165, 208)
        assert sections[-1].end_page == 208
        assert sections[0].end_page <= 208

    def test_clamp_then_invalid_range_fixed(self):
        """Clamping can create end < start, which is then fixed."""
        # Child at p50-p60 in parent p55-p100: start clamped to 55, end stays 60
        sections = [
            Section(title="Ch 1", start_page=50, end_page=60),
            Section(title="Ch 2", start_page=61, end_page=100),
        ]
        _enforce_boundary_invariants(sections, 55, 100)
        assert sections[0].start_page == 55
        assert sections[0].end_page >= sections[0].start_page

    def test_freudian_robot_pattern(self):
        """Reproduce the Freudian Robot systemic issue: last subsection truncated."""
        # Parent chapter is pages 10-49, but agent set last child to end at 45
        sections = [
            Section(title="Intro", start_page=10, end_page=20),
            Section(title="Body", start_page=21, end_page=35),
            Section(title="Techne of the Unconscious", start_page=36, end_page=45),
        ]
        _enforce_boundary_invariants(sections, 10, 49)
        # Last child should now extend to 49
        assert sections[-1].end_page == 49


class TestDetectBoundaryIssues:
    """Tests for detect_boundary_issues."""

    def test_invalid_range_detected(self):
        """end_page < start_page is detected as INVALID RANGE."""
        sections = [
            Section(title="Ch 1", start_page=10, end_page=50),
            Section(title="Return to Sender", start_page=208, end_page=207),
        ]
        issues = detect_boundary_issues(sections)
        assert any("INVALID RANGE" in i for i in issues)
        assert any("Return to Sender" in i for i in issues)

    def test_clean_sections_no_issues(self):
        """No issues for well-formed sections."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=49),
            Section(title="Ch 2", start_page=50, end_page=100),
        ]
        issues = detect_boundary_issues(sections)
        assert issues == []

    def test_gap_detected(self):
        """Gaps between sections are detected."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=40),
            Section(title="Ch 2", start_page=50, end_page=100),
        ]
        issues = detect_boundary_issues(sections)
        assert any("GAP" in i for i in issues)

    def test_overlap_detected(self):
        """Overlaps between sections are detected."""
        sections = [
            Section(title="Ch 1", start_page=1, end_page=55),
            Section(title="Ch 2", start_page=50, end_page=100),
        ]
        issues = detect_boundary_issues(sections)
        assert any("OVERLAP" in i for i in issues)


class TestSectionsToTocChildren:
    """Tests for sections_to_toc_children."""

    def test_inserted_section_preserved(self):
        """Inserted sections (original_index=None) create new TOCNodes."""
        original_children = [
            TOCNode(title="Ch 1", level=1, start_page=1, end_page=49),
            TOCNode(title="Ch 3", level=1, start_page=70, end_page=100),
        ]
        sections = [
            Section(title="Ch 1", start_page=1, end_page=49, original_index=0),
            Section(title="Ch 2 (inserted)", start_page=50, end_page=69, original_index=None),
            Section(title="Ch 3", start_page=70, end_page=100, original_index=1),
        ]
        result = sections_to_toc_children(sections, original_children)
        assert len(result) == 3
        assert result[1].title == "Ch 2 (inserted)"
        assert result[1].start_page == 50
        assert result[1].end_page == 69
        assert result[1].gap_filled is True

    def test_inserted_section_not_lost_via_reference(self):
        """Simulate the VirtualRoot pattern: assignment replaces reference."""
        original_children = [
            TOCNode(title="Ch 1", level=1, start_page=1, end_page=49),
            TOCNode(title="Ch 3", level=1, start_page=70, end_page=100),
        ]
        toc_tree = list(original_children)  # copy

        # Simulate what verify_node_boundaries does
        sections = [
            Section(title="Ch 1", start_page=1, end_page=49, original_index=0),
            Section(title="Ch 2 (inserted)", start_page=50, end_page=69, original_index=None),
            Section(title="Ch 3", start_page=70, end_page=100, original_index=1),
        ]
        new_children = sections_to_toc_children(sections, original_children)

        # This is the fix: capture updated reference
        toc_tree = new_children
        assert len(toc_tree) == 3
        assert toc_tree[1].title == "Ch 2 (inserted)"
