"""
N-gram based truncation detection for polished content.

This module provides intelligent detection of content truncation vs. deduplication
using n-gram multiset analysis. It's particularly suited for detecting truncation
in content where deduplication is expected and acceptable.
"""

import re
import unicodedata
from collections import Counter
from typing import Tuple, Dict
import tiktoken
from .base import BaseTruncationDetector

# Initialize tokenizer (cl100k_base is a good approximation for Gemini models)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define punctuation for CJK and ASCII
CJK_PUNCT = "、。！？：「」『』（）【】《》—…・，；「」『』〔〕〈〉—〜"
ASCII_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~"""


class NGramTruncationDetector(BaseTruncationDetector):
    """N-gram based truncation detector for distinguishing truncation from deduplication."""
    
    def __init__(
        self,
        min_unique_preserved_ratio: float = 0.60,
        allow_deduplication: bool = True
    ):
        """
        Initialize the detector.
        
        Args:
            min_unique_preserved_ratio: Minimum ratio of unique content to preserve
            allow_deduplication: Whether deduplication is acceptable
        """
        self.min_unique_preserved_ratio = min_unique_preserved_ratio
        self.allow_deduplication = allow_deduplication
    
    def detect(
        self,
        original: str,
        processed: str,
        **kwargs
    ) -> Tuple[bool, str, Dict]:
        """
        Detect if output is truncated vs. properly deduplicated.
        
        Uses n-gram multiset analysis to distinguish between:
        - Deduplication: Removing duplicate content (acceptable)
        - Truncation: Losing unique content (problematic)
        
        Args:
            original: Original text
            processed: Processed text
            
        Returns:
            Tuple of (is_truncated, reason, details)
        """
        # Quick check for empty output
        if not processed.strip():
            return True, "Output is empty", {'token_ratio': 0}
        
        # Calculate basic token metrics
        input_tokens = self._count_tokens(original)
        output_tokens = self._count_tokens(processed)
        token_ratio = output_tokens / input_tokens if input_tokens > 0 else 0
        
        # Check tail integrity
        tail_complete, tail_reason = self._check_tail_integrity(processed)
        
        # Calculate character n-gram metrics (primary)
        char_metrics = self._calculate_duplicate_aware_metrics(
            original, processed, "char", k=5
        )
        
        # Calculate word n-gram metrics (secondary, for verification)
        word_metrics = self._calculate_duplicate_aware_metrics(
            original, processed, "word", k=3
        )
        
        # Use average of char and word unique recall for robustness
        avg_unique_recall = (
            char_metrics['unique_recall'] + word_metrics['unique_recall']
        ) / 2
        avg_dup_ratio = (
            char_metrics['duplicate_deletion_ratio'] + 
            word_metrics['duplicate_deletion_ratio']
        ) / 2
        
        # Build detailed results
        details = {
            'token_ratio': token_ratio,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'tail_complete': tail_complete,
            'tail_reason': tail_reason,
            'char_ngram_unique_recall': char_metrics['unique_recall'],
            'char_ngram_dup_ratio': char_metrics['duplicate_deletion_ratio'],
            'word_ngram_unique_recall': word_metrics['unique_recall'],
            'word_ngram_dup_ratio': word_metrics['duplicate_deletion_ratio'],
            'avg_unique_recall': avg_unique_recall,
            'avg_dup_ratio': avg_dup_ratio,
            'char_metrics': char_metrics,
            'word_metrics': word_metrics
        }
        
        # Decision logic based on n-gram analysis
        
        # 1. If token ratio is very high (≥80%), definitely not truncated
        if token_ratio >= 0.80:
            return False, f"High token retention ({token_ratio:.1%})", details
        
        # 2. If tail is incomplete (mid-sentence), likely truncated
        if not tail_complete and "No proper ending" in tail_reason:
            return True, f"Text ends mid-sentence: {tail_reason}", details
        
        # 3. Excellent unique recall - definitely not truncated
        if avg_unique_recall >= 0.9:
            return False, f"Excellent unique content preservation ({avg_unique_recall:.1%})", details
        
        # 4. Good unique recall with high deduplication - acceptable
        if avg_unique_recall >= 0.8 and avg_dup_ratio >= 0.7 and self.allow_deduplication:
            return False, f"Good preservation ({avg_unique_recall:.1%}) with deduplication ({avg_dup_ratio:.1%})", details
        
        # 5. Moderate unique recall but very high deduplication - probably OK
        if avg_unique_recall >= 0.75 and avg_dup_ratio >= 0.85 and self.allow_deduplication:
            return False, f"Moderate preservation ({avg_unique_recall:.1%}) but mostly deduplication ({avg_dup_ratio:.1%})", details
        
        # 6. Poor unique recall - likely truncated
        if avg_unique_recall < 0.75:
            return True, f"Too much unique content lost (only {avg_unique_recall:.1%} preserved)", details
        
        # 7. Edge case: decent recall but low dedup ratio and low token ratio
        if token_ratio < 0.7 and avg_dup_ratio < 0.5:
            return True, f"Low token ratio ({token_ratio:.1%}) without sufficient deduplication", details
        
        # Default: acceptable
        return False, f"Acceptable preservation (recall: {avg_unique_recall:.1%}, dedup: {avg_dup_ratio:.1%})", details
    
    def get_summary(
        self,
        is_truncated: bool,
        reason: str,
        details: Dict
    ) -> str:
        """Generate a human-readable summary of truncation analysis."""
        summary_parts = []
        
        if is_truncated:
            summary_parts.append(f"⚠️ TRUNCATION DETECTED: {reason}")
        else:
            summary_parts.append(f"✓ Content complete: {reason}")
        
        # Token ratio
        summary_parts.append(
            f"Token ratio: {details['token_ratio']:.1%} "
            f"({details['output_tokens']:,}/{details['input_tokens']:,})"
        )
        
        # N-gram analysis results
        if 'avg_unique_recall' in details:
            summary_parts.append(f"Unique content recall: {details['avg_unique_recall']:.1%}")
            summary_parts.append(f"Deduplication ratio: {details['avg_dup_ratio']:.1%}")
        
        # More detailed n-gram stats if available
        if 'char_metrics' in details:
            cm = details['char_metrics']
            summary_parts.append(
                f"Character n-grams: {cm['unique_removed']:,} unique lost, "
                f"{cm['duplicate_removed']:,} duplicates removed"
            )
        
        # Tail integrity warning
        if 'tail_complete' in details and not details['tail_complete']:
            summary_parts.append(f"⚠️ Tail issue: {details['tail_reason']}")
        
        return "\n".join(summary_parts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(tokenizer.encode(text))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for n-gram analysis."""
        # Unicode normalization (NFKC handles CJK compatibility)
        text = unicodedata.normalize("NFKC", text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace ideographic space with regular space
        text = text.replace("\u3000", " ")
        
        # Collapse multiple whitespaces
        text = re.sub(r"\s+", " ", text)
        
        # Remove ASCII and CJK punctuation
        translator = {ord(ch): None for ch in (ASCII_PUNCT + CJK_PUNCT)}
        text = text.translate(translator)
        
        return text.strip()
    
    def _char_ngrams(self, text: str, k: int = 5) -> list:
        """Generate character n-grams from text."""
        text = self._normalize_text(text)
        
        # Handle edge case of very short text
        if len(text) < k:
            return [text] if text else []
        
        return [text[i:i+k] for i in range(len(text) - k + 1)]
    
    def _word_ngrams(self, text: str, k: int = 3) -> list:
        """Generate word n-grams from text."""
        text = self._normalize_text(text)
        words = text.split()
        
        # Handle edge case of very few words
        if len(words) < k:
            return [' '.join(words)] if words else []
        
        return [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]
    
    def _get_ngram_counts(self, text: str, ngram_type: str = "char", k: int = 5) -> Counter:
        """Get n-gram counts for text."""
        if ngram_type == "word":
            ngrams = self._word_ngrams(text, k)
        else:
            ngrams = self._char_ngrams(text, k)
        
        return Counter(ngrams)
    
    def _calculate_duplicate_aware_metrics(
        self,
        original: str,
        polished: str,
        ngram_type: str = "char",
        k: int = 5
    ) -> Dict:
        """Calculate duplicate-aware metrics using n-gram multisets."""
        # Get n-gram counts for both texts
        count_in = self._get_ngram_counts(original, ngram_type, k)
        count_out = self._get_ngram_counts(polished, ngram_type, k)
        
        # Calculate duplicate vs unique removals
        duplicate_removed = 0
        unique_removed = 0
        
        for ngram, c_in in count_in.items():
            c_out = count_out.get(ngram, 0)
            
            if c_out >= 1:
                # Some copies remain - count excess as duplicate removals
                duplicate_removed += max(0, c_in - c_out)
            else:
                # All copies gone - count as unique removal
                unique_removed += c_in
        
        # Calculate unique recall (what fraction of unique n-grams survived)
        unique_ngrams_in = len(count_in)
        unique_ngrams_preserved = sum(1 for ngram in count_in if count_out.get(ngram, 0) > 0)
        unique_recall = unique_ngrams_preserved / max(1, unique_ngrams_in)
        
        # Calculate duplicate deletion ratio
        total_removed = duplicate_removed + unique_removed
        dup_deletion_ratio = duplicate_removed / max(1, total_removed)
        
        # Calculate token-based metrics for reference
        total_ngrams_in = sum(count_in.values())
        total_ngrams_out = sum(count_out.values())
        ngram_retention_ratio = total_ngrams_out / max(1, total_ngrams_in)
        
        return {
            'unique_recall': unique_recall,
            'duplicate_deletion_ratio': dup_deletion_ratio,
            'duplicate_removed': duplicate_removed,
            'unique_removed': unique_removed,
            'unique_ngrams_preserved': unique_ngrams_preserved,
            'unique_ngrams_total': unique_ngrams_in,
            'ngram_retention_ratio': ngram_retention_ratio,
            'total_ngrams_in': total_ngrams_in,
            'total_ngrams_out': total_ngrams_out
        }
    
    def _check_tail_integrity(self, text: str) -> Tuple[bool, str]:
        """Check if text ends properly (not truncated mid-sentence)."""
        if not text:
            return False, "Empty text"
        
        # Get last non-empty line
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if not lines:
            return False, "No content lines"
        
        last_line = lines[-1]
        
        # Check for sentence-ending punctuation (including CJK)
        sentence_endings = {'.', '!', '?', ':', '。', '！', '？', '：', '；'}
        
        # Check for footnote endings like [^1] or [^1]:
        footnote_pattern = r'\[\^\d+\](:)?$'
        
        # Check for code block ending
        code_block_ending = '```'
        
        # Check for proper ending
        if (any(last_line.endswith(ending) for ending in sentence_endings) or 
            re.search(footnote_pattern, last_line) or
            last_line.endswith(code_block_ending)):
            return True, "Ends properly"
        
        # Check if it's a heading (which might not have punctuation)
        if last_line.startswith('#') or len(last_line) < 20:
            return True, "Ends with heading or short line"
        
        # Check for unbalanced brackets/quotes
        open_brackets = last_line.count('[') + last_line.count('{') + last_line.count('(')
        close_brackets = last_line.count(']') + last_line.count('}') + last_line.count(')')
        if open_brackets != close_brackets:
            return False, "Unbalanced brackets"
        
        # If doesn't end with proper punctuation and is long, likely truncated
        return False, "No proper ending punctuation"