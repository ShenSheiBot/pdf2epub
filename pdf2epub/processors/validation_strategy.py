"""
Validation and retry strategy management for processors.

This module provides a clean separation of validation logic from retry logic,
allowing for flexible and predictable handling of LLM output validation.

Error Handling Behavior Matrix
==============================

| Error Type           | Retry Same Model? | Try Next Model? | Use Fallback? | Notes |
|---------------------|------------------|----------------|--------------|--------|
| Truncation          | Yes*             | Yes**          | Yes***       | Partial content, may succeed on retry |
| Wrong Language      | Yes*             | Yes**          | No           | Randomness may fix it on retry |
| Empty Response      | Yes*             | Yes**          | No           | Likely transient error |
| Safety Block        | No               | Yes            | No           | Content flagged by provider |
| Rate Limit          | Yes (api level)  | Yes            | N/A          | Handled by api_retries |
| Network Error       | Yes (api level)  | Yes            | N/A          | Handled by api_retries |

* If validation_retries > 0 for the current model
** If fallback_between_models = True
*** If use_longest_on_failure = True (only applies to truncation errors)

Configuration Options
====================

Model-level (per model in model list):
- api_retries: Number of retries for transient API errors
- validation_retries: Number of retries when validation fails

Strategy-level (global):
- max_attempts: Maximum validation attempts per file across all models
- use_longest_on_failure: Use longest response as fallback for truncation errors
- fallback_between_models: Try next model when current model validation fails

Fallback Behavior
=================

When use_longest_on_failure = True:
- Only applies to truncation-type errors (partial content is better than none)
- Does NOT apply to wrong language or empty responses (invalid content is worse than failure)

When use_longest_on_failure = False:
- All invalid responses are rejected
- Process fails with exception rather than using invalid content
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class AttemptResult:
    """Represents a single attempt at generating content."""
    response: str
    model_config: Dict
    is_valid: bool
    validation_reason: str
    attempt_number: int
    content_length: int
    error_output_path: Optional[str] = None


class ValidationStrategy:
    """
    Manages validation and retry strategies across processors.

    This class centralizes the decision-making logic for:
    - When to retry with the same model
    - When to fallback to the next model
    - How to select the best response from multiple attempts
    """

    def __init__(self, config: Optional[Dict] = None, error_output_dir: Optional[Path] = None):
        """
        Initialize the validation strategy.

        Args:
            config: Configuration dict containing:
                - max_attempts: Max validation attempts per file (default: 2)
                - use_longest_on_failure: Use longest response on failure (default: False)
                - fallback_between_models: Try next model on validation failure (default: True)
            error_output_dir: Directory to save error outputs for debugging
        """
        config = config or {}

        # Global validation settings
        self.max_attempts = config.get('max_attempts', 2)
        self.use_longest_on_failure = config.get('use_longest_on_failure', False)
        self.fallback_between_models = config.get('fallback_between_models', True)

        # Error output directory
        self.error_output_dir = error_output_dir

        # Track attempts for decision making
        self.current_attempts: List[AttemptResult] = []

    def save_error_response(
        self,
        unit_key: str,
        attempt_number: int,
        response: str,
        validation_reason: str
    ) -> Optional[str]:
        """
        Save error response to file for debugging.

        Args:
            unit_key: Unit identifier (e.g., "chapter_5.part1")
            attempt_number: Attempt number
            response: The full LLM response
            validation_reason: Why validation failed

        Returns:
            Relative path to saved file, or None if error_output_dir not set
        """
        if not self.error_output_dir:
            return None

        self.error_output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = int(time.time() * 1000)
        # Sanitize unit_key for filename
        safe_unit_key = unit_key.replace("/", "_").replace(" ", "_")
        filename = f"{safe_unit_key}_attempt{attempt_number}_{timestamp}.txt"
        filepath = self.error_output_dir / filename

        # Write error details and full response
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Unit: {unit_key}\n")
            f.write(f"Attempt: {attempt_number}\n")
            f.write(f"Timestamp: {time.time()}\n")
            f.write(f"Validation Reason: {validation_reason}\n")
            f.write(f"Response Length: {len(response)} chars\n")
            f.write("=" * 60 + "\n")
            f.write("FULL RESPONSE:\n")
            f.write("=" * 60 + "\n")
            f.write(response)

        # Return relative path from parent of error_output_dir
        try:
            relative_path = str(filepath.relative_to(self.error_output_dir.parent))
        except ValueError:
            relative_path = str(filepath)

        logger.debug(f"Saved error response to {relative_path}")
        return relative_path

    def parse_model_config(self, model_config: Dict) -> Tuple[int, int]:
        """
        Parse model configuration to extract retry settings.

        Provides backward compatibility by interpreting 'max_retries' as both
        API and validation retries if specific settings aren't provided.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Tuple of (api_retries, validation_retries)
        """
        # Check for new explicit configuration
        if 'api_retries' in model_config or 'validation_retries' in model_config:
            api_retries = model_config.get('api_retries', 1)
            validation_retries = model_config.get('validation_retries', 0)
        else:
            # Backward compatibility: use max_retries for API retries
            # and set validation_retries to 0 by default
            max_retries = model_config.get('max_retries', 1)
            api_retries = max_retries
            validation_retries = 0  # Don't retry validation by default for backward compat

            logger.debug(
                f"Using backward compatibility for {model_config.get('model')}: "
                f"api_retries={api_retries}, validation_retries={validation_retries}"
            )

        return api_retries, validation_retries

    def should_retry_validation(
        self,
        model_idx: int,
        attempt: int,
        validation_retries: int,
        is_valid: bool,
        reason: str
    ) -> bool:
        """
        Decide whether to retry validation with the same model.

        Args:
            model_idx: Index of the current model in the model list
            attempt: Current attempt number (0-based)
            validation_retries: Number of validation retries configured for this model
            is_valid: Whether the validation passed
            reason: Validation failure reason

        Returns:
            True if should retry with same model, False otherwise
        """
        if is_valid:
            return False  # No need to retry if valid

        # Check if we have validation retries left for this model
        if attempt < validation_retries:
            logger.info(
                f"Validation failed (attempt {attempt + 1}/{validation_retries + 1}): {reason}. "
                f"Retrying with same model..."
            )
            return True

        return False

    def should_try_next_model(
        self,
        model_idx: int,
        total_models: int,
        all_attempts_failed: bool
    ) -> bool:
        """
        Decide whether to try the next model in the fallback chain.

        Args:
            model_idx: Index of the current model (0-based)
            total_models: Total number of available models
            all_attempts_failed: Whether all validation attempts for current model failed

        Returns:
            True if should try next model, False otherwise
        """
        # Can't try next if we're at the last model
        if model_idx >= total_models - 1:
            return False

        # Only try next model if fallback is enabled and current model failed
        if self.fallback_between_models and all_attempts_failed:
            logger.info(f"Model {model_idx + 1} exhausted all attempts, trying next model...")
            return True

        return False

    def record_attempt(
        self,
        response: str,
        model_config: Dict,
        is_valid: bool,
        validation_reason: str,
        attempt_number: int,
        error_output_path: Optional[str] = None
    ) -> None:
        """
        Record an attempt result for later selection.

        Args:
            response: The generated response
            model_config: Configuration of the model used
            is_valid: Whether validation passed
            validation_reason: Reason for validation result
            attempt_number: The attempt number
            error_output_path: Path to saved error output file (if validation failed)
        """
        self.current_attempts.append(AttemptResult(
            response=response,
            model_config=model_config,
            is_valid=is_valid,
            validation_reason=validation_reason,
            attempt_number=attempt_number,
            content_length=len(response),
            error_output_path=error_output_path
        ))

    def select_best_response(self, attempts: Optional[List[AttemptResult]] = None) -> Optional[str]:
        """
        Select the best response from multiple attempts.

        Args:
            attempts: List of attempt results (uses self.current_attempts if None)

        Returns:
            The best response according to the strategy, or None if no attempts
        """
        attempts = attempts or self.current_attempts

        if not attempts:
            return None

        # First preference: any valid response
        valid_attempts = [a for a in attempts if a.is_valid]
        if valid_attempts:
            # Return the first valid response (could be enhanced to pick best valid)
            return valid_attempts[0].response

        # No valid responses - apply fallback strategy if configured
        if self.use_longest_on_failure:
            # Check if any attempts have truncation-type errors (partial content)
            # Only use fallback for truncation, not for wrong language or empty responses
            truncation_attempts = [
                a for a in attempts
                if "truncat" in a.validation_reason.lower() or
                   "incomplete" in a.validation_reason.lower() or
                   "mid-sentence" in a.validation_reason.lower()
            ]

            if truncation_attempts:
                # Select the longest response from truncated attempts
                longest = max(truncation_attempts, key=lambda a: a.content_length)
                logger.info(
                    f"Using longest response ({longest.content_length} chars) from "
                    f"{longest.model_config.get('model')} after all validations failed (truncation fallback)"
                )
                return longest.response
            else:
                # No truncation errors, just other validation failures (wrong language, empty)
                # These should not use fallback even if configured
                logger.warning(
                    f"All {len(attempts)} validation attempts failed with non-truncation errors. "
                    f"Cannot use longest fallback for: {attempts[-1].validation_reason}"
                )
                return None
        else:
            # No fallback strategy configured - reject all invalid responses
            logger.warning(
                f"All {len(attempts)} validation attempts failed and use_longest_on_failure=False. "
                f"Rejecting all responses. Last failure: {attempts[-1].validation_reason}"
            )
            return None

    def clear_attempts(self) -> None:
        """Clear the recorded attempts for a new operation."""
        self.current_attempts = []

    def get_summary(self) -> str:
        """
        Get a summary of all attempts for logging.

        Returns:
            A formatted summary string
        """
        if not self.current_attempts:
            return "No attempts recorded"

        lines = ["Validation Summary:"]
        for attempt in self.current_attempts:
            status = "✓" if attempt.is_valid else "✗"
            model = attempt.model_config.get('model', 'unknown')
            line = (
                f"  {status} Attempt {attempt.attempt_number} with {model}: "
                f"{attempt.validation_reason} ({attempt.content_length} chars)"
            )
            if attempt.error_output_path:
                line += f" [saved: {attempt.error_output_path}]"
            lines.append(line)

        return "\n".join(lines)