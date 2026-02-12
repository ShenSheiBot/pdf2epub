"""
Unit tests for _fix_containment_overlaps in StructureAnalyzer.
"""

import copy
import pytest
from pdf2epub.refine.structure_analyzer import StructureAnalyzer


class TestFixContainmentOverlaps:
    """Test the containment overlap fix logic."""

    def test_basic_containment(self):
        """A (p100-300) contains B (p100-200) → B becomes child of A."""
        chapters = [
            {"title": "Erster Abschnitt", "level": 2, "start_page": 100, "end_page": 300},
            {"title": "Verwandlung", "level": 2, "start_page": 100, "end_page": 200},
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 1
        assert chapters[0]["title"] == "Erster Abschnitt"
        assert len(chapters[0]["children"]) == 1
        assert chapters[0]["children"][0]["title"] == "Verwandlung"
        assert chapters[0]["children"][0]["level"] == 3

    def test_nested_containment(self):
        """
        Marx case: A contains B, C, D; C contains D.
        Result: A → [B, C → [D]]
        """
        chapters = [
            {"title": "Erster Abschnitt", "level": 2, "start_page": 168, "end_page": 313},
            {"title": "Verwandlung von Geld", "level": 2, "start_page": 168, "end_page": 202},
            {"title": "Arbeitsprozess", "level": 2, "start_page": 228, "end_page": 281},
            {"title": "Absoluter Mehrwert", "level": 2, "start_page": 247, "end_page": 281},
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        # Top level: only Erster Abschnitt remains
        assert len(chapters) == 1
        container = chapters[0]
        assert container["title"] == "Erster Abschnitt"

        # Erster Abschnitt has 2 direct children (Verwandlung, Arbeitsprozess)
        children = container["children"]
        assert len(children) == 2
        assert children[0]["title"] == "Verwandlung von Geld"
        assert children[0]["level"] == 3
        assert children[1]["title"] == "Arbeitsprozess"
        assert children[1]["level"] == 3

        # Arbeitsprozess has 1 child (Absoluter Mehrwert)
        assert len(children[1]["children"]) == 1
        assert children[1]["children"][0]["title"] == "Absoluter Mehrwert"
        assert children[1]["children"][0]["level"] == 4

    def test_reverse_order_containment(self):
        """B comes before A in list but A's range contains B → B becomes child of A."""
        chapters = [
            {"title": "Small Section", "level": 1, "start_page": 50, "end_page": 80},
            {"title": "Big Section", "level": 1, "start_page": 50, "end_page": 200},
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 1
        assert chapters[0]["title"] == "Big Section"
        assert len(chapters[0]["children"]) == 1
        assert chapters[0]["children"][0]["title"] == "Small Section"
        assert chapters[0]["children"][0]["level"] == 2

    def test_no_containment(self):
        """Non-overlapping siblings → no change."""
        chapters = [
            {"title": "Chapter 1", "level": 1, "start_page": 1, "end_page": 50},
            {"title": "Chapter 2", "level": 1, "start_page": 51, "end_page": 100},
            {"title": "Chapter 3", "level": 1, "start_page": 101, "end_page": 150},
        ]
        original = copy.deepcopy(chapters)
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 3
        for i in range(3):
            assert chapters[i]["title"] == original[i]["title"]
            assert chapters[i]["start_page"] == original[i]["start_page"]

    def test_partial_overlap_no_change(self):
        """Partial overlap (neither contains the other) → no reparenting."""
        chapters = [
            {"title": "Section A", "level": 1, "start_page": 10, "end_page": 60},
            {"title": "Section B", "level": 1, "start_page": 40, "end_page": 90},
        ]
        original = copy.deepcopy(chapters)
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 2
        assert chapters[0]["title"] == original[0]["title"]
        assert chapters[1]["title"] == original[1]["title"]

    def test_existing_children_preserved(self):
        """Container already has children; new child is merged in and sorted."""
        chapters = [
            {
                "title": "Part I",
                "level": 1,
                "start_page": 1,
                "end_page": 200,
                "children": [
                    {"title": "Existing Child", "level": 2, "start_page": 1, "end_page": 50},
                ],
            },
            {"title": "New Child", "level": 1, "start_page": 60, "end_page": 100},
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 1
        children = chapters[0]["children"]
        assert len(children) == 2
        # Sorted by start_page
        assert children[0]["title"] == "Existing Child"
        assert children[1]["title"] == "New Child"
        assert children[1]["level"] == 2

    def test_level_update_deep(self):
        """Reparented node's descendants also get level updated."""
        chapters = [
            {"title": "Container", "level": 1, "start_page": 1, "end_page": 300},
            {
                "title": "Inner",
                "level": 1,
                "start_page": 10,
                "end_page": 100,
                "children": [
                    {"title": "Deep Child", "level": 2, "start_page": 20, "end_page": 50},
                ],
            },
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 1
        inner = chapters[0]["children"][0]
        assert inner["title"] == "Inner"
        assert inner["level"] == 2
        deep = inner["children"][0]
        assert deep["title"] == "Deep Child"
        assert deep["level"] == 3

    def test_empty_list(self):
        """Empty chapters list → no crash."""
        chapters = []
        StructureAnalyzer._fix_containment_overlaps(chapters)
        assert chapters == []

    def test_single_chapter(self):
        """Single chapter → no change."""
        chapters = [{"title": "Only", "level": 1, "start_page": 1, "end_page": 100}]
        StructureAnalyzer._fix_containment_overlaps(chapters)
        assert len(chapters) == 1

    def test_recurse_into_existing_children(self):
        """Containment overlaps within existing children are also fixed."""
        chapters = [
            {
                "title": "Part I",
                "level": 1,
                "start_page": 1,
                "end_page": 200,
                "children": [
                    {"title": "Big Sub", "level": 2, "start_page": 10, "end_page": 150},
                    {"title": "Small Sub", "level": 2, "start_page": 20, "end_page": 80},
                ],
            },
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        children = chapters[0]["children"]
        assert len(children) == 1
        assert children[0]["title"] == "Big Sub"
        assert len(children[0]["children"]) == 1
        assert children[0]["children"][0]["title"] == "Small Sub"
        assert children[0]["children"][0]["level"] == 3

    def test_multiple_containers_at_same_level(self):
        """Two separate containers each absorb their own contained siblings."""
        chapters = [
            {"title": "Part I", "level": 1, "start_page": 1, "end_page": 100},
            {"title": "Ch 1", "level": 1, "start_page": 10, "end_page": 50},
            {"title": "Part II", "level": 1, "start_page": 200, "end_page": 400},
            {"title": "Ch 2", "level": 1, "start_page": 210, "end_page": 300},
        ]
        StructureAnalyzer._fix_containment_overlaps(chapters)

        assert len(chapters) == 2
        assert chapters[0]["title"] == "Part I"
        assert chapters[0]["children"][0]["title"] == "Ch 1"
        assert chapters[1]["title"] == "Part II"
        assert chapters[1]["children"][0]["title"] == "Ch 2"
