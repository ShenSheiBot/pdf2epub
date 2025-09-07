"""
Comprehensive tests for the structure_validator module with complex real-world scenarios.
"""

import json
import pytest
from pathlib import Path
from pdf2epub.utils.structure_validator import (
    resolve_overlaps,
    find_missing_pages,
    add_missing_pages_as_chapters,
    validate_structure
)


class TestComplexOverlaps:
    """Test cases for complex overlapping scenarios."""
    
    def test_nested_subsections_with_overlaps(self):
        """Test the exact scenario from Kant's book - nested subsections within subchapters."""
        structure = {
            "chapters": [
                {
                    "title": "The Transcendental Refutation",
                    "start_page": 73,
                    "end_page": 128,
                    "subchapters": [
                        {"title": "The Refutation a Mere Word Play", "start_page": 75, "end_page": 78},
                        {"title": "Ontological Distinctness", "start_page": 79, "end_page": 90},
                        {"title": "The Argument in the Refutation", "start_page": 91, "end_page": 110},  # Parent
                        {"title": "Structure of the Proof", "start_page": 91, "end_page": 99},  # Child of above
                        {"title": "Instantiation", "start_page": 100, "end_page": 105},  # Child of above
                        {"title": "Argument from Sensory", "start_page": 106, "end_page": 110},  # Child of above
                        {"title": "What is an Object?", "start_page": 111, "end_page": 128},  # Parent
                        {"title": "The A Priori", "start_page": 111, "end_page": 123},  # Child of above
                        {"title": "Circularity", "start_page": 124, "end_page": 128},  # Child of above
                    ]
                }
            ]
        }
        
        result = resolve_overlaps(structure)
        subchapters = result["chapters"][0]["subchapters"]
        
        # Main subchapters should not overlap
        main_subs = [s for s in subchapters if s["title"] in [
            "The Refutation a Mere Word Play",
            "Ontological Distinctness", 
            "The Argument in the Refutation",
            "What is an Object?"
        ]]
        
        for i in range(len(main_subs) - 1):
            assert main_subs[i]["end_page"] < main_subs[i+1]["start_page"], \
                f"Main subchapters overlap: {main_subs[i]['title']} and {main_subs[i+1]['title']}"
        
        # Nested subsections should remain within their parent's range
        argument_section = next(s for s in subchapters if s["title"] == "The Argument in the Refutation")
        for sub in ["Structure of the Proof", "Instantiation", "Argument from Sensory"]:
            subsection = next(s for s in subchapters if s["title"] == sub)
            assert subsection["start_page"] >= argument_section["start_page"]
            assert subsection["end_page"] <= argument_section["end_page"]
    
    def test_complete_chaos_overlaps(self):
        """Test with completely chaotic overlapping structure."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 1, "end_page": 50,
                 "subchapters": [
                     {"title": "1.1", "start_page": 5, "end_page": 40},  # Overlaps with 1.2
                     {"title": "1.2", "start_page": 10, "end_page": 45},  # Overlaps with 1.1 and 1.3
                     {"title": "1.3", "start_page": 35, "end_page": 60},  # Exceeds parent
                 ]},
                {"title": "Ch2", "start_page": 40, "end_page": 80,  # Overlaps with Ch1
                 "subchapters": [
                     {"title": "2.1", "start_page": 30, "end_page": 50},  # Starts before parent
                     {"title": "2.2", "start_page": 45, "end_page": 55},  # Overlaps with 2.1
                     {"title": "2.3", "start_page": 50, "end_page": 90},  # Exceeds parent
                 ]},
                {"title": "Ch3", "start_page": 70, "end_page": 100,  # Overlaps with Ch2
                 "subchapters": []}
            ]
        }
        
        result = resolve_overlaps(structure)
        
        # Verify no subchapter exceeds its parent
        for chapter in result["chapters"]:
            for sub in chapter.get("subchapters", []):
                assert sub["end_page"] <= chapter["end_page"], \
                    f"Subchapter {sub['title']} exceeds parent {chapter['title']}"
        
        # Verify main subchapters within each chapter don't overlap
        for chapter in result["chapters"]:
            subs = chapter.get("subchapters", [])
            # Get only main subchapters (not nested ones)
            main_subs = []
            for sub in subs:
                is_nested = any(
                    other["start_page"] <= sub["start_page"] and 
                    other["end_page"] >= sub["end_page"] and 
                    other != sub
                    for other in subs
                )
                if not is_nested:
                    main_subs.append(sub)
            
            main_subs.sort(key=lambda x: x["start_page"])
            for i in range(len(main_subs) - 1):
                assert main_subs[i]["end_page"] < main_subs[i+1]["start_page"], \
                    f"Overlapping subchapters in {chapter['title']}"
    
    def test_single_page_chapters_and_subchapters(self):
        """Test handling of single-page chapters and subchapters."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 1, "end_page": 1},  # Single page
                {"title": "Ch2", "start_page": 2, "end_page": 10,
                 "subchapters": [
                     {"title": "2.1", "start_page": 2, "end_page": 2},  # Single page
                     {"title": "2.2", "start_page": 3, "end_page": 3},  # Single page
                     {"title": "2.3", "start_page": 3, "end_page": 5},  # Overlaps with 2.2
                 ]},
                {"title": "Ch3", "start_page": 11, "end_page": 11},  # Single page
            ]
        }
        
        result = resolve_overlaps(structure)
        
        # Single page chapters should remain unchanged
        ch1 = next(c for c in result["chapters"] if c["title"] == "Ch1")
        assert ch1["start_page"] == 1 and ch1["end_page"] == 1
        
        # Check that single-page subchapter overlap is handled
        ch2 = next(c for c in result["chapters"] if c["title"] == "Ch2")
        sub_2_2 = next(s for s in ch2["subchapters"] if s["title"] == "2.2")
        sub_2_3 = next(s for s in ch2["subchapters"] if s["title"] == "2.3")
        
        # 2.2 should be adjusted to not overlap with 2.3 (or recognized as nested)
        assert sub_2_2["end_page"] < sub_2_3["start_page"] or \
               (sub_2_2["start_page"] >= sub_2_3["start_page"] and 
                sub_2_2["end_page"] <= sub_2_3["end_page"])


class TestComplexMissingPages:
    """Test cases for complex missing page scenarios."""
    
    def test_multiple_gaps_with_subchapters(self):
        """Test finding gaps in a complex structure with subchapters."""
        structure = {
            "front_matter": {"start_page": 1, "end_page": 5},
            "chapters": [
                {"title": "Ch1", "start_page": 10, "end_page": 20,  # Gap: 6-9
                 "has_intro": True, "intro_end_page": 12,
                 "subchapters": [
                     {"title": "1.1", "start_page": 13, "end_page": 15},
                     {"title": "1.2", "start_page": 18, "end_page": 20},  # Gap: 16-17
                 ]},
                {"title": "Ch2", "start_page": 25, "end_page": 30},  # Gap: 21-24
                {"title": "Ch3", "start_page": 35, "end_page": 40,  # Gap: 31-34
                 "subchapters": [
                     {"title": "3.1", "start_page": 35, "end_page": 37},
                     {"title": "3.2", "start_page": 39, "end_page": 40},  # Gap: 38
                 ]},
            ],
            "back_matter": {"start_page": 45, "end_page": 50}  # Gap: 41-44
        }
        
        missing = find_missing_pages(structure, total_pages=55)
        
        expected_gaps = [
            (6, 9),    # Between front matter and Ch1
            (16, 17),  # Within Ch1 subchapters
            (21, 24),  # Between Ch1 and Ch2
            (31, 34),  # Between Ch2 and Ch3
            (38, 38),  # Within Ch3 subchapters
            (41, 44),  # Between Ch3 and back matter
            (51, 55),  # After back matter
        ]
        
        assert missing == expected_gaps
    
    def test_completely_fragmented_book(self):
        """Test a book where every other page is missing."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 2, "end_page": 2},
                {"title": "Ch2", "start_page": 4, "end_page": 4},
                {"title": "Ch3", "start_page": 6, "end_page": 6},
                {"title": "Ch4", "start_page": 8, "end_page": 8},
                {"title": "Ch5", "start_page": 10, "end_page": 10},
            ]
        }
        
        missing = find_missing_pages(structure, total_pages=12)
        
        expected_gaps = [
            (1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 12)
        ]
        
        assert missing == expected_gaps
    
    def test_book_with_only_middle_pages(self):
        """Test a book missing beginning and end."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 50, "end_page": 100},
            ]
        }
        
        missing = find_missing_pages(structure, total_pages=150)
        
        assert missing == [(1, 49), (101, 150)]


class TestAddMissingPagesComplex:
    """Test adding missing pages in complex scenarios."""
    
    def test_add_pages_preserving_hierarchy(self):
        """Test that adding missing pages preserves chapter hierarchy."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 5, "end_page": 10,
                 "has_intro": True, "intro_end_page": 6,
                 "subchapters": [
                     {"title": "1.1", "start_page": 7, "end_page": 8},
                     {"title": "1.2", "start_page": 10, "end_page": 10},  # Gap at 9
                 ]},
                {"title": "Ch2", "start_page": 15, "end_page": 20},  # Gap 11-14
            ]
        }
        
        result = add_missing_pages_as_chapters(structure, total_pages=25)
        
        # Should add chapters for gaps 1-4, 9, 11-14, and 21-25
        # The gap at page 9 within subchapters will also be filled
        added_chapters = [c for c in result["chapters"] if "Additional Content" in c["title"]]
        
        assert len(added_chapters) == 4
        assert any(c["start_page"] == 1 and c["end_page"] == 4 for c in added_chapters)
        assert any(c["start_page"] == 9 and c["end_page"] == 9 for c in added_chapters)
        assert any(c["start_page"] == 11 and c["end_page"] == 14 for c in added_chapters)
        assert any(c["start_page"] == 21 and c["end_page"] == 25 for c in added_chapters)
        
        # Original chapter structure should be preserved
        ch1 = next(c for c in result["chapters"] if c["title"] == "Ch1")
        assert ch1.get("has_intro") == True
        assert len(ch1["subchapters"]) == 2
    
    def test_massive_gaps(self):
        """Test adding chapters for very large gaps."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 1, "end_page": 10},
                {"title": "Ch2", "start_page": 500, "end_page": 510},  # Huge gap
                {"title": "Ch3", "start_page": 1000, "end_page": 1010},  # Another huge gap
            ]
        }
        
        result = add_missing_pages_as_chapters(structure, total_pages=2000)
        
        added_chapters = [c for c in result["chapters"] if "Additional Content" in c["title"]]
        
        # Should add 3 large gap chapters
        assert len(added_chapters) == 3
        assert any(c["start_page"] == 11 and c["end_page"] == 499 for c in added_chapters)
        assert any(c["start_page"] == 511 and c["end_page"] == 999 for c in added_chapters)
        assert any(c["start_page"] == 1011 and c["end_page"] == 2000 for c in added_chapters)


class TestRealWorldScenarios:
    """Test complete real-world book structures."""
    
    def test_academic_book_structure(self):
        """Test a typical academic book with complex hierarchy."""
        structure = {
            "book_title": "Complex Academic Book",
            "front_matter": {"start_page": 1, "end_page": 15},
            "chapters": [
                # Part I
                {"title": "Part I: Introduction", "start_page": 16, "end_page": 20},
                {"title": "Chapter 1: Background", "start_page": 21, "end_page": 50,
                 "subchapters": [
                     {"title": "1.1 Historical Context", "start_page": 25, "end_page": 35},
                     {"title": "1.1.1 Ancient Period", "start_page": 25, "end_page": 28},
                     {"title": "1.1.2 Modern Period", "start_page": 29, "end_page": 35},
                     {"title": "1.2 Current State", "start_page": 36, "end_page": 50},
                     {"title": "1.2.1 Technology", "start_page": 36, "end_page": 42},
                     {"title": "1.2.2 Society", "start_page": 43, "end_page": 50},
                 ]},
                # Gap here: pages 51-55
                {"title": "Chapter 2: Methodology", "start_page": 56, "end_page": 80,
                 "subchapters": [
                     {"title": "2.1 Research Design", "start_page": 60, "end_page": 70},
                     {"title": "2.2 Data Collection", "start_page": 71, "end_page": 80},
                 ]},
                # Part II with overlap
                {"title": "Part II: Analysis", "start_page": 81, "end_page": 85},
                {"title": "Chapter 3: Results", "start_page": 82, "end_page": 120,  # Overlaps with Part II
                 "subchapters": [
                     {"title": "3.1 Quantitative", "start_page": 85, "end_page": 100},
                     {"title": "3.2 Qualitative", "start_page": 95, "end_page": 120},  # Overlaps with 3.1
                 ]},
            ],
            "back_matter": {"start_page": 121, "end_page": 130}
        }
        
        result = validate_structure(structure, fix=True, total_pages=135)
        
        # Check that overlaps are fixed
        ch3 = next(c for c in result["chapters"] if "Chapter 3" in c["title"])
        subs = ch3["subchapters"]
        quant = next(s for s in subs if "Quantitative" in s["title"])
        qual = next(s for s in subs if "Qualitative" in s["title"])
        assert quant["end_page"] < qual["start_page"] or \
               (quant["start_page"] >= qual["start_page"] and quant["end_page"] <= qual["end_page"])
        
        # Check that gaps are filled
        all_titles = [c["title"] for c in result["chapters"]]
        assert any("Additional Content" in title and "51-55" in title for title in all_titles)
        assert any("Additional Content" in title and "131-135" in title for title in all_titles)
    
    def test_edge_case_empty_book(self):
        """Test handling of an empty or minimal book structure."""
        # Empty structure
        empty_structure = {}
        result = validate_structure(empty_structure, fix=True)
        assert result == empty_structure
        
        # Structure with no chapters
        no_chapters = {
            "book_title": "Empty Book",
            "front_matter": {"start_page": 1, "end_page": 5}
        }
        result = validate_structure(no_chapters, fix=True, total_pages=10)
        # Should add a chapter for pages 6-10
        assert "chapters" in result
        assert len(result["chapters"]) == 1
        assert result["chapters"][0]["start_page"] == 6
        assert result["chapters"][0]["end_page"] == 10
    
    def test_book_with_appendices_and_indices(self):
        """Test book with complex back matter structure."""
        structure = {
            "chapters": [
                {"title": "Ch1", "start_page": 10, "end_page": 50},
                {"title": "Ch2", "start_page": 51, "end_page": 100},
            ],
            "back_matter": {"start_page": 101, "end_page": 200},
            "appendices": [  # This might be in the structure
                {"title": "Appendix A", "start_page": 101, "end_page": 120},
                {"title": "Appendix B", "start_page": 121, "end_page": 140},
            ],
            "index": {"start_page": 141, "end_page": 180},
            "bibliography": {"start_page": 181, "end_page": 200}
        }
        
        # The validator should handle extra fields gracefully
        result = validate_structure(structure, fix=True, total_pages=205)
        
        # Should add missing pages 1-9 and 201-205
        missing = find_missing_pages(result, total_pages=205)
        assert (1, 9) in missing or any(c["start_page"] == 1 for c in result.get("chapters", []))
        
    def test_malformed_structure_recovery(self):
        """Test recovery from malformed structures."""
        # Chapters with invalid page numbers
        malformed = {
            "chapters": [
                {"title": "Ch1", "start_page": 10, "end_page": 5},  # End before start
                {"title": "Ch2", "start_page": -5, "end_page": 20},  # Negative start
                {"title": "Ch3", "start_page": 25, "end_page": 25},  # Single page (valid)
                {"title": "Ch4", "start_page": 30, "end_page": 1000000},  # Huge end page
            ]
        }
        
        # The validator should handle these gracefully
        # This might need additional error handling in the actual implementation
        try:
            result = validate_structure(malformed, fix=True, total_pages=50)
            # Should not crash and should produce some reasonable output
            assert "chapters" in result
        except Exception as e:
            # Should provide meaningful error message
            assert "page" in str(e).lower() or "invalid" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])