"""
Unit tests for the structure_validator module.
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


class TestResolveOverlaps:
    """Test cases for resolve_overlaps function."""
    
    def test_no_overlaps(self):
        """Test structure with no overlapping pages."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10,
                    "subchapters": []
                },
                {
                    "title": "Chapter 2",
                    "start_page": 11,
                    "end_page": 20,
                    "subchapters": []
                }
            ]
        }
        result = resolve_overlaps(structure)
        assert result == structure
    
    def test_overlapping_subchapters(self):
        """Test fixing overlapping subchapters."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 30,
                    "subchapters": [
                        {
                            "title": "Section 1.1",
                            "start_page": 5,
                            "end_page": 15
                        },
                        {
                            "title": "Section 1.2",
                            "start_page": 10,  # Overlaps with 1.1
                            "end_page": 20
                        },
                        {
                            "title": "Section 1.3",
                            "start_page": 21,
                            "end_page": 30
                        }
                    ]
                }
            ]
        }
        
        result = resolve_overlaps(structure)
        
        # Check that overlaps are fixed
        subchapters = result["chapters"][0]["subchapters"]
        assert subchapters[0]["end_page"] == 9  # Adjusted to not overlap with next
        assert subchapters[1]["start_page"] == 10
        assert subchapters[1]["end_page"] == 20  # No change needed
        assert subchapters[2]["start_page"] == 21
        assert subchapters[2]["end_page"] == 30
    
    def test_subchapter_exceeding_parent(self):
        """Test fixing subchapter that exceeds parent chapter range."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 20,
                    "subchapters": [
                        {
                            "title": "Section 1.1",
                            "start_page": 5,
                            "end_page": 25  # Exceeds parent
                        }
                    ]
                }
            ]
        }
        
        result = resolve_overlaps(structure)
        
        # Check that subchapter is adjusted
        subchapter = result["chapters"][0]["subchapters"][0]
        assert subchapter["end_page"] == 20  # Adjusted to parent's end_page
    
    def test_chapter_with_intro(self):
        """Test detecting intro pages before first subchapter."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 30,
                    "subchapters": [
                        {
                            "title": "Section 1.1",
                            "start_page": 10,  # Intro pages 1-9
                            "end_page": 20
                        },
                        {
                            "title": "Section 1.2",
                            "start_page": 21,
                            "end_page": 30
                        }
                    ]
                }
            ]
        }
        
        result = resolve_overlaps(structure)
        
        chapter = result["chapters"][0]
        assert chapter["has_intro"] == True
        assert chapter["intro_end_page"] == 9


class TestFindMissingPages:
    """Test cases for find_missing_pages function."""
    
    def test_no_missing_pages(self):
        """Test structure with complete coverage."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 11,
                    "end_page": 20
                }
            ]
        }
        
        missing = find_missing_pages(structure, total_pages=20)
        assert missing == []
    
    def test_gap_between_chapters(self):
        """Test finding gap between chapters."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 15,  # Gap: pages 11-14
                    "end_page": 20
                }
            ]
        }
        
        missing = find_missing_pages(structure)
        assert missing == [(11, 14)]
    
    def test_missing_beginning_pages(self):
        """Test finding missing pages at the beginning."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 5,  # Missing pages 1-4
                    "end_page": 10
                }
            ]
        }
        
        missing = find_missing_pages(structure)
        assert missing == [(1, 4)]
    
    def test_missing_ending_pages(self):
        """Test finding missing pages at the end."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10
                }
            ]
        }
        
        missing = find_missing_pages(structure, total_pages=15)
        assert missing == [(11, 15)]
    
    def test_with_front_and_back_matter(self):
        """Test with front and back matter."""
        structure = {
            "front_matter": {
                "start_page": 1,
                "end_page": 3
            },
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 4,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 15,  # Gap: pages 11-14
                    "end_page": 20
                }
            ],
            "back_matter": {
                "start_page": 21,
                "end_page": 25
            }
        }
        
        missing = find_missing_pages(structure)
        assert missing == [(11, 14)]
    
    def test_with_subchapters(self):
        """Test with chapters containing subchapters."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 20,
                    "has_intro": True,
                    "intro_end_page": 4,
                    "subchapters": [
                        {
                            "title": "Section 1.1",
                            "start_page": 5,
                            "end_page": 10
                        },
                        {
                            "title": "Section 1.2",
                            "start_page": 15,  # Gap: pages 11-14
                            "end_page": 20
                        }
                    ]
                }
            ]
        }
        
        missing = find_missing_pages(structure)
        assert missing == [(11, 14)]


class TestAddMissingPagesAsChapters:
    """Test cases for add_missing_pages_as_chapters function."""
    
    def test_add_single_gap(self):
        """Test adding chapter for a single gap."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 15,
                    "end_page": 20
                }
            ]
        }
        
        result = add_missing_pages_as_chapters(structure)
        
        # Should have 3 chapters now
        assert len(result["chapters"]) == 3
        
        # Check the new chapter
        new_chapter = result["chapters"][1]
        assert new_chapter["title"] == "Additional Content (Pages 11-14)"
        assert new_chapter["start_page"] == 11
        assert new_chapter["end_page"] == 14
    
    def test_add_multiple_gaps(self):
        """Test adding chapters for multiple gaps."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 5,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 15,
                    "end_page": 20
                }
            ]
        }
        
        result = add_missing_pages_as_chapters(structure, total_pages=25)
        
        # Should have 5 chapters now (2 original + 3 gaps)
        assert len(result["chapters"]) == 5
        
        # Check ordering
        chapters = result["chapters"]
        assert chapters[0]["title"] == "Additional Content (Pages 1-4)"
        assert chapters[1]["title"] == "Chapter 1"
        assert chapters[2]["title"] == "Additional Content (Pages 11-14)"
        assert chapters[3]["title"] == "Chapter 2"
        assert chapters[4]["title"] == "Additional Content (Pages 21-25)"
    
    def test_no_gaps(self):
        """Test when there are no gaps."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 11,
                    "end_page": 20
                }
            ]
        }
        
        result = add_missing_pages_as_chapters(structure, total_pages=20)
        
        # Should remain unchanged
        assert len(result["chapters"]) == 2
        assert result == structure


class TestValidateStructure:
    """Test cases for validate_structure function."""
    
    def test_validate_and_fix(self):
        """Test validation with fixing enabled."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10,
                    "subchapters": [
                        {
                            "title": "Section 1.1",
                            "start_page": 5,
                            "end_page": 15  # Overlaps and exceeds parent
                        }
                    ]
                },
                {
                    "title": "Chapter 2",
                    "start_page": 20,  # Gap: pages 11-19
                    "end_page": 30
                }
            ]
        }
        
        result = validate_structure(structure, fix=True, total_pages=30)
        
        # Check that overlap is fixed
        subchapter = result["chapters"][0]["subchapters"][0]
        assert subchapter["end_page"] == 10
        
        # Check that gap is filled
        assert len(result["chapters"]) == 3
        gap_chapter = result["chapters"][1]
        assert gap_chapter["title"] == "Additional Content (Pages 11-19)"
    
    def test_validate_without_fix(self):
        """Test validation without fixing."""
        structure = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "start_page": 1,
                    "end_page": 10
                },
                {
                    "title": "Chapter 2",
                    "start_page": 15,  # Gap
                    "end_page": 20
                }
            ]
        }
        
        result = validate_structure(structure, fix=False)
        
        # Structure should remain unchanged
        assert result == structure
        assert len(result["chapters"]) == 2
    
    def test_empty_structure(self):
        """Test validation of empty structure."""
        structure = {}
        
        result = validate_structure(structure, fix=True)
        
        # Should handle gracefully
        assert result == structure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])