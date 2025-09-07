"""
Unit tests for document parser module.
"""

import pytest
from pdf2epub.processors.utils.document_parser import (
    extract_first_sentence,
    find_citations_in_text,
    find_footnote_definitions,
    identify_sections,
    analyze_document_structure,
    find_split_positions,
)


class TestExtractFirstSentence:
    """Test cases for extract_first_sentence function"""
    
    def test_simple_sentence(self):
        """Test extracting a simple sentence"""
        text = "This is the first sentence. This is the second sentence."
        result = extract_first_sentence(text)
        assert result == "This is the first sentence."
    
    def test_multiple_punctuation(self):
        """Test with different punctuation marks"""
        text = "Is this a question? Yes, it is! And here's more."
        result = extract_first_sentence(text)
        assert result == "Is this a question?"
    
    def test_japanese_punctuation(self):
        """Test with Japanese punctuation"""
        text = "これは日本語の文章です。次の文章はこちら。"
        result = extract_first_sentence(text)
        assert result == "これは日本語の文章です。"
    
    def test_no_punctuation(self):
        """Test text without sentence ending"""
        text = "This text has no sentence ending punctuation"
        result = extract_first_sentence(text, max_length=20)
        assert result == "This text has no sen..."
    
    def test_empty_text(self):
        """Test with empty text"""
        result = extract_first_sentence("")
        assert result == ""


class TestFindCitations:
    """Test cases for find_citations_in_text function"""
    
    def test_markdown_citations(self):
        """Test standard markdown citations"""
        text = "This is a citation[^1] and another[^note] in the text."
        citations = find_citations_in_text(text)
        assert "1" in citations
        assert "note" in citations
    
    def test_numbered_citations(self):
        """Test numbered bracket citations"""
        text = "According to research[1], the findings[2] show that[3]..."
        citations = find_citations_in_text(text)
        assert "1" in citations
        assert "2" in citations
        assert "3" in citations
    
    def test_latex_citations(self):
        """Test LaTeX style citations"""
        text = "This result$^1$ contradicts earlier work$^{2}$."
        citations = find_citations_in_text(text)
        assert "1" in citations
        assert "2" in citations
    
    def test_superscript_citations(self):
        """Test Unicode superscript numbers"""
        text = "The study¹ found that mice² respond to stimuli³."
        citations = find_citations_in_text(text)
        assert "1" in citations
        assert "2" in citations
        assert "3" in citations
    
    def test_mixed_citations(self):
        """Test mixed citation styles"""
        text = "Various formats[^1] are used[2] in papers$^3$ today¹."
        citations = find_citations_in_text(text)
        assert len(citations) >= 3
        assert "1" in citations
        assert "2" in citations
        assert "3" in citations
    
    def test_ocr_style_citations(self):
        """Test OCR-style citations with spaces"""
        text = "This interpretation is disputed{ }^{41} and another view{ }^42 exists."
        citations = find_citations_in_text(text)
        assert "41" in citations
        assert "42" in citations


class TestFindFootnoteDefinitions:
    """Test cases for find_footnote_definitions function"""
    
    def test_markdown_definitions(self):
        """Test standard markdown footnote definitions"""
        text = """[^1]: This is the first footnote.
[^note]: This is a named footnote."""
        definitions = find_footnote_definitions(text)
        assert "1" in definitions
        assert "note" in definitions
    
    def test_numbered_definitions(self):
        """Test numbered bracket definitions"""
        text = """[1] Smith, J. (2020). Title of Paper.
[2] Jones, K. (2021). Another Paper."""
        definitions = find_footnote_definitions(text)
        assert "1" in definitions
        assert "2" in definitions
    
    def test_numbered_list_definitions(self):
        """Test numbered list style definitions"""
        text = """1. First footnote content here.
2. Second footnote content here."""
        definitions = find_footnote_definitions(text)
        assert "1" in definitions
        assert "2" in definitions
    
    def test_inline_brackets_not_definitions(self):
        """Test that inline brackets are not treated as definitions"""
        text = "This sentence contains [1] in the middle and should not match."
        definitions = find_footnote_definitions(text)
        assert "1" not in definitions
    
    def test_mixed_definitions(self):
        """Test mixed definition styles"""
        text = """[^1]: Markdown style footnote.
[2] Academic style footnote.
3. Numbered list style footnote."""
        definitions = find_footnote_definitions(text)
        assert "1" in definitions
        assert "2" in definitions
        assert "3" in definitions


class TestIdentifySections:
    """Test cases for identify_sections function"""
    
    def test_basic_sections(self):
        """Test identifying basic markdown sections"""
        content = """# Title
Introduction text.

## Section 1
Section 1 content.

### Subsection 1.1
Subsection content.

## Section 2
Section 2 content."""
        
        sections = identify_sections(content)
        assert len(sections) == 4
        assert sections[0][3] == "Title"
        assert sections[1][3] == "Section 1"
        assert sections[2][3] == "Subsection 1.1"
        assert sections[3][3] == "Section 2"
    
    def test_content_before_first_heading(self):
        """Test handling content before first heading"""
        content = """This is content before any heading.

# First Heading
Content under first heading."""
        
        sections = identify_sections(content)
        assert len(sections) == 2
        assert sections[0][3] == ""  # No heading for first section
        assert sections[1][3] == "First Heading"
    
    def test_no_headings(self):
        """Test content with no headings"""
        content = """This is just plain text.
No headings at all.
Multiple paragraphs."""
        
        sections = identify_sections(content)
        assert len(sections) == 1
        assert sections[0][3] == ""


class TestAnalyzeDocumentStructure:
    """Test cases for analyze_document_structure function"""
    
    def test_general_content_analysis(self):
        """Test analyzing general content"""
        content = """# Introduction
This is the introduction to our document.

## Background
Some background information here.

## Methods
Our methodology is described here."""
        
        result = analyze_document_structure(content, "general")
        assert len(result) == 3
        assert result[0]["section"] == "# Introduction"
        assert "tokens" in result[0]
        assert "starts_with" in result[0]
        assert "citations_made" not in result[0]  # General content shouldn't have citations
    
    def test_academic_content_analysis(self):
        """Test analyzing academic content with citations"""
        content = """# Introduction
Recent studies[^1] have shown interesting results[^2].

## Methods
We followed the protocol described by Smith[^3].

## References
[^1]: First reference here.
[^2]: Second reference here.
[^3]: Third reference here."""
        
        result = analyze_document_structure(content, "academic")
        assert len(result) == 3
        
        # Check Introduction section
        intro = result[0]
        assert "citations_made" in intro
        assert "1" in intro["citations_made"]
        assert "2" in intro["citations_made"]
        assert len(intro["footnotes_defined"]) == 0
        
        # Check References section
        refs = result[2]
        assert "footnotes_defined" in refs
        assert "1" in refs["footnotes_defined"]
        assert "2" in refs["footnotes_defined"]
        assert "3" in refs["footnotes_defined"]
    
    def test_japanese_content_analysis(self):
        """Test analyzing Japanese content"""
        content = """# 第一章
これは日本語のコンテンツです。

## セクション１
詳細な内容がここにあります。"""
        
        result = analyze_document_structure(content, "japanese")
        assert len(result) == 2
        assert result[0]["section"] == "# 第一章"
        assert "tokens" in result[0]
        assert "starts_with" in result[0]


class TestFindSplitPositions:
    """Test cases for find_split_positions function"""
    
    def test_basic_split_positions(self):
        """Test finding basic split positions"""
        content = """First section content.
        
Second section starts here.

Third section starts here."""
        
        markers = ["Second section starts here.", "Third section starts here."]
        positions = find_split_positions(content, markers)
        
        assert len(positions) == 4  # Start, two markers, end
        assert positions[0] == 0
        assert positions[-1] == len(content)
        assert content[positions[1]:].startswith("Second section")
        assert content[positions[2]:].startswith("Third section")
    
    def test_markers_with_ellipsis(self):
        """Test handling markers with ellipsis"""
        content = "This is the full sentence without ellipsis."
        markers = ["This is the full sentence..."]
        
        positions = find_split_positions(content, markers)
        assert len(positions) >= 2
        assert positions[0] == 0
        assert positions[-1] == len(content)
    
    def test_missing_markers(self):
        """Test handling missing markers"""
        content = "Some content here."
        markers = ["This marker doesn't exist", "Neither does this"]
        
        positions = find_split_positions(content, markers)
        # Should still have start and end positions
        assert positions[0] == 0
        assert positions[-1] == len(content)


if __name__ == "__main__":
    # Run basic smoke tests
    print("Testing document parser functions...")
    
    # Test extract_first_sentence
    test_extract = TestExtractFirstSentence()
    test_extract.test_simple_sentence()
    print("✓ Extract first sentence test passed")
    
    # Test citation finding
    test_citations = TestFindCitations()
    test_citations.test_markdown_citations()
    test_citations.test_numbered_citations()
    print("✓ Citation finding tests passed")
    
    # Test footnote definition finding
    test_definitions = TestFindFootnoteDefinitions()
    test_definitions.test_markdown_definitions()
    test_definitions.test_numbered_definitions()
    print("✓ Footnote definition tests passed")
    
    # Test section identification
    test_sections = TestIdentifySections()
    test_sections.test_basic_sections()
    print("✓ Section identification test passed")
    
    # Test full document analysis
    test_analysis = TestAnalyzeDocumentStructure()
    test_analysis.test_general_content_analysis()
    test_analysis.test_academic_content_analysis()
    print("✓ Document analysis tests passed")
    
    # Test split position finding
    test_splits = TestFindSplitPositions()
    test_splits.test_basic_split_positions()
    print("✓ Split position finding test passed")
    
    print("\nAll document parser tests passed!")
