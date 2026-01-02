"""
LLM-based truncation detection for various tasks.

This module provides lightweight truncation detection using an LLM to verify
if content has been properly processed by comparing the endings of original
and processed text.
"""

from typing import Tuple, Dict, Optional
from .base import BaseTruncationDetector
from ....utils.model_utils import get_cheapest_model_configs


class LLMTruncationDetector(BaseTruncationDetector):
    """LLM-based truncation detector with task-specific prompts."""

    def __init__(self, llm_client, num_lines: int = 3, task_type: str = "translate"):
        """
        Initialize the detector.

        Args:
            llm_client: LLM client for verification
            num_lines: Number of lines from the end to check (default: 3)
            task_type: Type of task - "translate" or "polish" (default: "translate")
        """
        self.llm_client = llm_client
        self.num_lines = num_lines
        self.task_type = task_type
    
    def detect(
        self,
        original: str,
        processed: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, str, Dict]:
        """
        Detect if translation is truncated using LLM verification.
        
        Args:
            original: Original text
            processed: Translated text
            source_language: Source language (optional)
            target_language: Target language (optional)
            
        Returns:
            Tuple of (is_truncated, reason, details)
        """
        # Quick check for empty output
        if not processed.strip():
            return True, "Translation is empty", {'last_lines_original': '', 'last_lines_translated': ''}
        
        # Extract last N lines from both texts
        # Count all lines for statistics
        all_original_lines = original.strip().split('\n')
        all_translated_lines = processed.strip().split('\n')
        
        # Get non-empty lines for content comparison
        original_lines = [l for l in all_original_lines if l.strip()]
        translated_lines = [l for l in all_translated_lines if l.strip()]
        
        # Quick check: if line counts match exactly, skip LLM check
        if len(translated_lines) < len(original_lines) + 3 and len(translated_lines) > len(original_lines) - 3:
            return False, "Line counts match exactly", {
                'original_line_count': len(original_lines),
                'translated_line_count': len(translated_lines),
                'line_ratio': len(translated_lines) / len(original_lines),
                'skipped_llm_check': True
            }
        
        # Get last N non-empty lines for comparison
        last_original = '\n'.join(original_lines[-self.num_lines:]) if original_lines else ""
        last_processed = '\n'.join(translated_lines[-self.num_lines:]) if translated_lines else ""

        # Generate task-specific prompt
        prompt = self._generate_prompt(
            last_original, last_processed,
            source_language, target_language
        )
        
        try:
            # Use cheapest models for this check
            cheapest_models = get_cheapest_model_configs(
                self.llm_client.config,
                max_models=3
            )
            
            # If no cheap models configured, fall back to default
            if not cheapest_models:
                cheapest_models = [
                    {"provider": "gemini", "model": "gemini-2.5-flash", "max_retries": 1}
                ]
            
            response = self.llm_client.generate(
                prompt=prompt,
                model_configs=cheapest_models,
                operation_name="Translation truncation check"
            )
            
            # Parse response
            answer = response.strip().lower()
            
            details = {
                'last_lines_original': last_original,
                'last_lines_processed': last_processed,
                'llm_response': answer,
                'original_line_count': len(original_lines),
                'processed_line_count': len(translated_lines)
            }
            
            if "truncated" in answer:
                return True, "LLM detected incomplete translation", details
            elif "complete" in answer:
                return False, "Translation appears complete", details
            else:
                # Unclear response, fall back to line count comparison
                line_ratio = len(translated_lines) / len(original_lines) if original_lines else 0
                details['line_ratio'] = line_ratio
                
                if line_ratio < 0.7:
                    return True, f"Significant line count difference (ratio: {line_ratio:.1%})", details
                else:
                    return False, "Unable to determine, assuming complete", details
                    
        except Exception as e:
            # If LLM check fails, fall back to simple heuristics
            details = {
                'last_lines_original': last_original,
                'last_lines_processed': last_processed,
                'error': str(e)
            }
            
            # Simple heuristic: check line count ratio
            line_ratio = len(translated_lines) / len(original_lines) if original_lines else 0
            details['line_ratio'] = line_ratio
            
            if line_ratio < 0.5:
                return True, f"Very low line count ratio ({line_ratio:.1%})", details
            
            # Check if translation ends with punctuation
            if translated_lines:
                last_char = translated_lines[-1][-1] if translated_lines[-1] else ''
                if last_char not in '.!?。！？':
                    return True, "Translation doesn't end with punctuation", details
            
            return False, "LLM check failed, using heuristics (appears complete)", details
    
    def get_summary(
        self,
        is_truncated: bool,
        reason: str,
        details: Dict
    ) -> str:
        """Generate a human-readable summary of truncation analysis."""
        summary_parts = []
        
        if is_truncated:
            summary_parts.append(f"⚠️ TRANSLATION TRUNCATED: {reason}")
        else:
            summary_parts.append(f"✓ Translation complete: {reason}")
        
        # Add line count info if available
        if 'line_ratio' in details:
            summary_parts.append(f"Line count ratio: {details['line_ratio']:.1%}")
        elif 'original_line_count' in details and 'translated_line_count' in details:
            summary_parts.append(
                f"Lines: {details['translated_line_count']} translated / "
                f"{details['original_line_count']} original"
            )
        
        # Add LLM response if available
        if 'llm_response' in details and details['llm_response'] in ['complete', 'truncated']:
            summary_parts.append(f"LLM verdict: {details['llm_response']}")
        
        # Add error info if LLM failed
        if 'error' in details:
            summary_parts.append(f"Note: LLM check failed, used fallback heuristics")

        return "\n".join(summary_parts)

    def _generate_prompt(
        self,
        last_original: str,
        last_processed: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> str:
        """
        Generate task-specific prompt for truncation detection.

        Args:
            last_original: Last N lines of original content
            last_processed: Last N lines of processed content
            source_language: Source language (for translation)
            target_language: Target language (for translation)

        Returns:
            Prompt string
        """
        if self.task_type == "polish":
            return self._generate_polish_prompt(last_original, last_processed)
        else:
            return self._generate_translate_prompt(
                last_original, last_processed,
                source_language, target_language
            )

    def _generate_polish_prompt(self, last_original: str, last_processed: str) -> str:
        """Generate prompt for polish truncation detection."""
        return f"""You are a markdown polishing completeness checker. Analyze if the polished output covers all the meaningful content from the original OCR text.

Compare these endings:

ORIGINAL OCR (last {self.num_lines} non-empty lines):
{last_original}

POLISHED OUTPUT (last {self.num_lines} non-empty lines):
{last_processed}

Answer with ONLY "complete" or "truncated" based on:
1. Does the polished output include the MEANINGFUL content from the original's ending?
2. Was actual content cut off or stopped mid-way?

IMPORTANT - These are NOT truncation (answer "complete"):
- Removing duplicate text (OCR often duplicates headers/titles)
- Removing PDF watermarks like "Powered by TCPDF" or similar
- Removing page numbers or other metadata
- Consolidating repeated content into a single clean version
- Formatting changes (spacing, headers, etc.)
- Converting HTML tables (<table>, <tr>, <td>) to Markdown tables (| column |)
- Converting HTML formatting to Markdown (removing HTML tags while preserving data)

These ARE truncation (answer "truncated"):
- Missing sentences or paragraphs that contain unique information
- Content that stops mid-sentence or mid-paragraph
- Entire sections of unique content removed

Your answer (one word only):"""

    def _generate_translate_prompt(
        self,
        last_original: str,
        last_processed: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> str:
        """Generate prompt for translation truncation detection."""
        lang_info = ""
        if source_language and target_language:
            lang_info = f" from {source_language} to {target_language}"

        return f"""You are a translation completeness checker. Analyze if a translation{lang_info} has been completed or was cut off mid-way.

Compare these endings:

ORIGINAL (last {self.num_lines} non-empty lines):
{last_original}

TRANSLATION (last {self.num_lines} non-empty lines):
{last_processed}

Answer with ONLY "complete" or "truncated" based on:
1. Does the translation include all the content from the original's ending?
2. Are there significant content elements from the original's ending that are completely missing in the translation?

Note:
- Minor formatting differences or slight variations in the final lines are acceptable
- A translation can be truncated even if it ends at a complete sentence
- Focus on whether ALL original content was translated, not how it ends

Your answer (one word only):"""
