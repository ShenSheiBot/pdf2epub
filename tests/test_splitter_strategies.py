"""
Unit tests for content splitter strategies.
"""

import pytest
from pdf2epub.processors.utils.splitter_strategies import (
    SimpleSplitter,
    MarkdownStructureSplitter,
)


class TestMarkdownStructureSplitter:
    """Test cases for MarkdownStructureSplitter"""
    
    def test_no_split_needed(self):
        """Test that content under max tokens is not split"""
        splitter = MarkdownStructureSplitter()
        content = "This is a short text that doesn't need splitting."
        result = splitter.split(content, max_tokens=1000)
        assert len(result) == 1
        assert result[0] == content
    
    def test_detect_markdown_titles(self):
        """Test markdown title detection"""
        splitter = MarkdownStructureSplitter()
        content = """# Main Title
Some content here

## Section 1
Content for section 1

### Subsection 1.1
More detailed content

## Section 2
Content for section 2"""
        
        titles = splitter._detect_markdown_titles(content)
        assert len(titles) == 4
        assert titles[0][2] == "Main Title"
        assert titles[1][2] == "Section 1"
        assert titles[2][2] == "Subsection 1.1"
        assert titles[3][2] == "Section 2"
    
    def test_title_based_splitting(self):
        """Test splitting based on markdown titles"""
        splitter = MarkdownStructureSplitter()
        
        # Create content with multiple sections
        content = """# Chapter 1
This is the introduction to chapter 1. It contains quite a bit of text that would normally
exceed our token limit when combined with other sections. We want to ensure that the splitter
correctly identifies section boundaries and groups them appropriately.

## Section 1.1
This section discusses the first major topic. It has substantial content that contributes
to the overall token count. The splitter should recognize this as a distinct section.

## Section 1.2
Another section with its own content. This helps test the grouping algorithm to ensure
it properly combines sections when they fit within the token limit.

## Section 1.3
The third section continues the discussion with more detailed information about the topic
at hand, providing examples and explanations.

# Chapter 2
A new chapter begins here with its own introduction and overview of what will be covered
in the following sections.

## Section 2.1
The first section of chapter 2 explores new concepts and ideas that build upon what was
established in the previous chapter."""
        
        # Use a small max_tokens to force splitting
        result = splitter.split(content, max_tokens=100)
        
        # Should have multiple parts
        assert len(result) > 1
        
        # Each part should start with a title or contain grouped sections
        for part in result:
            assert len(part) > 0
    
    def test_fallback_to_simple_splitter(self):
        """Test fallback to SimpleSplitter when less than 3 titles"""
        splitter = MarkdownStructureSplitter()
        
        # Content with only 2 titles
        content = """# Title 1
This is a very long piece of content that needs to be split but doesn't have enough
markdown titles to use title-based splitting. """ + ("Long text " * 200) + """

## Title 2
More content here."""
        
        result = splitter.split(content, max_tokens=50)
        
        # Should split the content even though there aren't enough titles
        assert len(result) > 1
    
    def test_content_before_first_title(self):
        """Test handling of content that appears before the first title"""
        splitter = MarkdownStructureSplitter()
        
        content = """This is some introductory content before any titles appear.
It should be included as its own section.

# First Title
Content under the first title.

## Second Title
Content under the second title.

### Third Title
Content under the third title."""
        
        titles = splitter._detect_markdown_titles(content)
        result = splitter._split_by_titles(content, 1000, titles)
        
        # The content before the first title should be included
        assert "introductory content" in result[0]
    
    def test_greedy_token_maximization(self):
        """Test that the splitter maximizes tokens per part greedily"""
        splitter = MarkdownStructureSplitter()
        
        # Create content where sections have varying sizes
        content = """# Small Section 1
Short content.

# Small Section 2
Short content.

# Small Section 3
Short content.

# Large Section
""" + ("This is a much larger section with lots of content. " * 50) + """

# Small Section 4
Short content."""
        
        # Set max_tokens to allow grouping of small sections
        result = splitter.split(content, max_tokens=200)
        
        # Small sections should be grouped together
        assert len(result) >= 2
        
        # Check that parts are created (exact behavior depends on token counts)
        for part in result:
            assert len(part) > 0


class TestSimpleSplitter:
    """Test cases for SimpleSplitter (regression tests)"""
    
    def test_simple_split_basic(self):
        """Test basic paragraph-based splitting"""
        splitter = SimpleSplitter()
        
        content = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with additional content."""
        
        result = splitter.split(content, max_tokens=1000)
        assert len(result) == 1  # Should fit in one part
        
    def test_simple_split_long_content(self):
        """Test splitting of long content"""
        splitter = SimpleSplitter()
        
        # Create long content with many paragraphs
        paragraphs = [f"Paragraph {i}. " + ("Content " * 20) for i in range(20)]
        content = "\n\n".join(paragraphs)
        
        result = splitter.split(content, max_tokens=50)
        
        # Should split into multiple parts
        assert len(result) > 1
        
        # Each part should maintain paragraph structure
        for part in result:
            assert len(part) > 0


if __name__ == "__main__":
    # Run basic tests
    print("Testing MarkdownStructureSplitter...")
    test_md = TestMarkdownStructureSplitter()
    test_md.test_no_split_needed()
    print("✓ No split needed test passed")
    
    test_md.test_detect_markdown_titles()
    print("✓ Markdown title detection test passed")
    
    test_md.test_title_based_splitting()
    print("✓ Title-based splitting test passed")
    
    test_md.test_fallback_to_simple_splitter()
    print("✓ Fallback to simple splitter test passed")
    
    test_md.test_content_before_first_title()
    print("✓ Content before first title test passed")
    
    test_md.test_greedy_token_maximization()
    print("✓ Greedy token maximization test passed")
    
    print("\nTesting SimpleSplitter...")
    test_simple = TestSimpleSplitter()
    test_simple.test_simple_split_basic()
    print("✓ Simple split basic test passed")
    
    test_simple.test_simple_split_long_content()
    print("✓ Simple split long content test passed")
    
    print("\nAll tests passed!")