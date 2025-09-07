from abc import ABC, abstractmethod
from typing import List, Tuple
import re
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


class ContentSplitter(ABC):
    """Abstract base class for content splitters."""

    @abstractmethod
    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Splits the content into parts.

        Args:
            content: The content to split.
            max_tokens: The maximum number of tokens per part.

        Returns:
            A list of content parts.
        """
        pass


class SimpleSplitter(ContentSplitter):
    """A simple content splitter."""

    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Simple content splitter that divides content into roughly equal parts.

        Args:
            content: The content to split
            max_tokens: Maximum tokens per part

        Returns:
            List of content parts
        """
        actual_tokens = len(tokenizer.encode(content))

        if actual_tokens <= max_tokens:
            return [content]

        num_parts = (actual_tokens // max_tokens) + 1
        paragraphs = content.split("\n\n")

        if len(paragraphs) < num_parts * 2:
            paragraphs = content.split("\n")

        parts = []
        target_tokens_per_part = actual_tokens // num_parts
        current_part = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(tokenizer.encode(para))
            if (
                current_tokens + para_tokens > target_tokens_per_part * 1.2
                and current_part
            ):
                parts.append(
                    "\n\n".join(current_part)
                    if "\n\n" in content
                    else "\n".join(current_part)
                )
                current_part = [para]
                current_tokens = para_tokens
            else:
                current_part.append(para)
                current_tokens += para_tokens

        if current_part:
            parts.append(
                "\n\n".join(current_part)
                if "\n\n" in content
                else "\n".join(current_part)
            )

        return parts if parts else [content]


class MarkdownStructureSplitter(ContentSplitter):
    """
    A content splitter that intelligently chooses between title-based 
    and paragraph-based splitting strategies.
    """

    def _detect_markdown_titles(self, content: str) -> List[Tuple[int, str, str]]:
        """
        Detect markdown titles in the content.
        
        Args:
            content: The content to analyze
            
        Returns:
            List of tuples: (line_index, title_level, title_text)
        """
        lines = content.split('\n')
        titles = []
        
        for i, line in enumerate(lines):
            # Match markdown headers (# Title, ## Title, ### Title, etc.)
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                title_level = match.group(1)
                title_text = match.group(2)
                titles.append((i, title_level, title_text))
        
        return titles
    
    def _split_by_titles(self, content: str, max_tokens: int, titles: List[Tuple[int, str, str]]) -> List[str]:
        """
        Split content based on markdown titles using greedy token maximization.
        
        Args:
            content: The content to split
            max_tokens: Maximum tokens per part
            titles: List of detected titles
            
        Returns:
            List of content parts
        """
        lines = content.split('\n')
        
        # Create sections: each title with its content
        sections = []
        for i, (line_idx, level, text) in enumerate(titles):
            # Find the end of this section (start of next title or end of document)
            if i < len(titles) - 1:
                next_line_idx = titles[i + 1][0]
            else:
                next_line_idx = len(lines)
            
            # Extract section content
            section_lines = lines[line_idx:next_line_idx]
            section_content = '\n'.join(section_lines).strip()
            
            if section_content:
                sections.append(section_content)
        
        # If we have content before the first title, include it
        if titles and titles[0][0] > 0:
            pre_title_lines = lines[:titles[0][0]]
            pre_title_content = '\n'.join(pre_title_lines).strip()
            if pre_title_content:
                sections.insert(0, pre_title_content)
        
        # If no titles were found or no sections created, fallback
        if not sections:
            return SimpleSplitter().split(content, max_tokens)
        
        # Greedy grouping of sections
        parts = []
        current_part = []
        current_tokens = 0
        
        for section in sections:
            section_tokens = len(tokenizer.encode(section))
            
            # If adding this section would exceed max_tokens and we have content
            if current_tokens + section_tokens > max_tokens and current_part:
                # Save current part
                parts.append('\n\n'.join(current_part))
                # Start new part with current section
                current_part = [section]
                current_tokens = section_tokens
            else:
                # Add section to current part
                current_part.append(section)
                current_tokens += section_tokens
        
        # Add remaining content
        if current_part:
            parts.append('\n\n'.join(current_part))
        
        # If we ended up with just one part and it's too large, fallback
        if len(parts) == 1 and current_tokens > max_tokens:
            return SimpleSplitter().split(content, max_tokens)
        
        return parts if parts else [content]
    
    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Split content intelligently based on structure.
        
        Args:
            content: The content to split
            max_tokens: Maximum tokens per part
            
        Returns:
            List of content parts
        """
        # Quick check if content fits in one part
        actual_tokens = len(tokenizer.encode(content))
        if actual_tokens <= max_tokens:
            return [content]
        
        # Detect markdown titles
        titles = self._detect_markdown_titles(content)
        
        # If we have 3 or more titles, use title-based splitting
        if len(titles) >= 3:
            return self._split_by_titles(content, max_tokens, titles)
        
        # Otherwise, fallback to paragraph-based splitting
        return SimpleSplitter().split(content, max_tokens)
