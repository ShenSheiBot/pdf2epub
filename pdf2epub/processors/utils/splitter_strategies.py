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
    """A simple content splitter with intelligent paragraph detection."""

    def split(self, content: str, max_tokens: int) -> List[str]:
        """
        Simple content splitter that divides content into roughly equal parts.
        Uses intelligent paragraph detection to avoid splitting at OCR line-wrap artifacts.

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

        # Multi-level fallback for finding real paragraph boundaries
        paragraphs = self._find_paragraphs(content, num_parts)

        parts = []
        target_tokens_per_part = actual_tokens // num_parts
        current_part = []
        current_tokens = 0

        # Determine separator based on what we found
        separator = "\n\n"
        if "\n\n\n" in content:
            separator = "\n\n\n"
        elif "\n\n" not in content and "\n" in content:
            separator = "\n"
        elif "\n" not in content:
            separator = " "

        for para in paragraphs:
            para_tokens = len(tokenizer.encode(para))
            if (
                current_tokens + para_tokens > target_tokens_per_part * 1.2
                and current_part
            ):
                parts.append(separator.join(current_part))
                current_part = [para]
                current_tokens = para_tokens
            else:
                current_part.append(para)
                current_tokens += para_tokens

        if current_part:
            parts.append(separator.join(current_part))

        return parts if parts else [content]

    def _find_paragraphs(self, content: str, min_paragraphs_needed: int) -> List[str]:
        """
        Find real paragraphs using multi-level fallback.

        Fallback order:
        1. Triple newlines (\n\n\n) - clear section breaks
        2. Double newlines (\n\n) - standard paragraph breaks
        3. Single newlines ending with punctuation - filtered for real breaks
        4. Sentence boundaries (periods, !, ?) - if no newlines
        5. Raw string split - extreme fallback

        Args:
            content: The content to analyze
            min_paragraphs_needed: Minimum number of paragraphs we need

        Returns:
            List of paragraphs
        """
        import re

        # Try triple newlines first (clear section breaks)
        if "\n\n\n" in content:
            paragraphs = content.split("\n\n\n")
            if len(paragraphs) >= min_paragraphs_needed:
                return [p.strip() for p in paragraphs if p.strip()]

        # Try double newlines (standard paragraph breaks)
        if "\n\n" in content:
            paragraphs = content.split("\n\n")
            if len(paragraphs) >= min_paragraphs_needed * 2:
                return [p.strip() for p in paragraphs if p.strip()]

        # Try single newlines, but filter for real paragraph endings
        if "\n" in content:
            lines = content.split("\n")
            paragraphs = []
            current_para = []

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    # Empty line - might be a paragraph break
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
                    continue

                current_para.append(line)

                # Check if this line ends a paragraph
                # A line ends a paragraph if:
                # 1. It ends with sentence-ending punctuation (including CJK)
                # 2. The next line starts with capital letter, number, or CJK character (new sentence)

                # Western and CJK sentence-ending punctuation
                sentence_endings = '.!?。！？」』】）)」』'  # Including quotation closers that often end sentences

                if line and line[-1] in sentence_endings:
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if next_line:
                        # Check if next line starts a new sentence
                        # For Western text: capital letter or number
                        # For CJK text: any CJK character (they don't have case)
                        first_char = next_line[0]
                        is_cjk = '\u4e00' <= first_char <= '\u9fff' or \
                                '\u3040' <= first_char <= '\u309f' or \
                                '\u30a0' <= first_char <= '\u30ff' or \
                                '\uac00' <= first_char <= '\ud7af'  # Korean

                        if first_char.isupper() or first_char.isdigit() or is_cjk:
                            # This looks like a real paragraph ending
                            paragraphs.append(" ".join(current_para))
                            current_para = []

            if current_para:
                paragraphs.append(" ".join(current_para))

            if len(paragraphs) >= min_paragraphs_needed * 2:
                return [p for p in paragraphs if p]

        # Fallback to sentence boundaries if no good newline breaks
        if "\n" not in content or len(paragraphs) < min_paragraphs_needed * 2:
            # Split by sentence-ending punctuation (Western and CJK)
            # Western: period/exclamation/question followed by space and capital
            # CJK: CJK sentence endings (no space needed in CJK)
            sentences = re.split(
                r'(?<=[.!?])\s+(?=[A-Z0-9])|'  # Western sentences
                r'(?<=[。！？」』】）])',  # CJK sentence endings
                content
            )

            if len(sentences) > min_paragraphs_needed * 3:
                # Group sentences into pseudo-paragraphs
                paragraphs = []
                sentences_per_para = max(3, len(sentences) // (min_paragraphs_needed * 2))

                for i in range(0, len(sentences), sentences_per_para):
                    para = " ".join(sentences[i:i + sentences_per_para])
                    if para.strip():
                        paragraphs.append(para.strip())

                return paragraphs

        # Ultimate fallback: split by character count if nothing else works
        if len(paragraphs) < min_paragraphs_needed:
            # This is the extreme case - no structure at all
            chars_per_part = len(content) // (min_paragraphs_needed * 2)
            if chars_per_part > 100:
                paragraphs = []
                for i in range(0, len(content), chars_per_part):
                    # Try to break at a space at least
                    end = min(i + chars_per_part, len(content))
                    if end < len(content):
                        # Look for nearest space
                        space_idx = content.rfind(' ', i, end)
                        if space_idx > i:
                            end = space_idx

                    part = content[i:end].strip()
                    if part:
                        paragraphs.append(part)

                return paragraphs

        # Return what we have
        return [p for p in paragraphs if p] if paragraphs else [content]


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
