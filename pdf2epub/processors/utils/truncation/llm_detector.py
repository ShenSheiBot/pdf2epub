"""
LLM-based truncation detection for translations.

This module provides lightweight truncation detection using an LLM to verify
if a translation has been completed by comparing the endings of original
and translated text.
"""

from typing import Tuple, Dict, Optional
from .base import BaseTruncationDetector
from ....utils.model_utils import get_cheapest_model_configs


class LLMTruncationDetector(BaseTruncationDetector):
    """LLM-based truncation detector for translations."""
    
    def __init__(self, llm_client, num_lines: int = 3):
        """
        Initialize the detector.
        
        Args:
            llm_client: LLM client for verification
            num_lines: Number of lines from the end to check (default: 3)
        """
        self.llm_client = llm_client
        self.num_lines = num_lines
    
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
        original_lines = [l for l in original.strip().split('\n') if l.strip()]
        translated_lines = [l for l in processed.strip().split('\n') if l.strip()]
        
        # Get last N lines
        last_original = '\n'.join(original_lines[-self.num_lines:]) if original_lines else ""
        last_translated = '\n'.join(translated_lines[-self.num_lines:]) if translated_lines else ""
        
        # Prepare the verification prompt
        lang_info = ""
        if source_language and target_language:
            lang_info = f" from {source_language} to {target_language}"
        
        prompt = f"""You are a translation completeness checker. Analyze if a translation{lang_info} has been completed or was cut off mid-way.

Compare these endings:

ORIGINAL (last {self.num_lines} lines):
{last_original}

TRANSLATION (last {self.num_lines} lines):
{last_translated}

Answer with ONLY "complete" or "truncated" based on:
1. Does the translation end at a natural stopping point like the original?
2. Are there missing elements that appear in the original's ending?
3. Does the translation stop mid-sentence or mid-paragraph?

Your answer (one word only):"""
        
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
                'last_lines_translated': last_translated,
                'llm_response': answer,
                'original_line_count': len(original_lines),
                'translated_line_count': len(translated_lines)
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
                'last_lines_translated': last_translated,
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
