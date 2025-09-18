"""
Validation and retry strategy management for processors.

This module provides a clean separation of validation logic from retry logic,
allowing for flexible and predictable handling of LLM output validation.
"""

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


class ValidationStrategy:
    """
    Manages validation and retry strategies across processors.

    This class centralizes the decision-making logic for:
    - When to retry with the same model
    - When to fallback to the next model
    - How to select the best response from multiple attempts
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validation strategy.

        Args:
            config: Configuration dict containing:
                - max_attempts: Max validation attempts per file (default: 2)
                - use_longest_on_failure: Use longest response on failure (default: False)
                - fallback_between_models: Try next model on validation failure (default: True)
        """
        config = config or {}

        # Global validation settings
        self.max_attempts = config.get('max_attempts', 2)
        self.use_longest_on_failure = config.get('use_longest_on_failure', False)
        self.fallback_between_models = config.get('fallback_between_models', True)

        # Track attempts for decision making
        self.current_attempts: List[AttemptResult] = []

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
            logger.info(f"Model {model_idx + 1} failed validation, trying next model...")
            return True

        return False

    def record_attempt(
        self,
        response: str,
        model_config: Dict,
        is_valid: bool,
        validation_reason: str,
        attempt_number: int
    ) -> None:
        """
        Record an attempt result for later selection.

        Args:
            response: The generated response
            model_config: Configuration of the model used
            is_valid: Whether validation passed
            validation_reason: Reason for validation result
            attempt_number: The attempt number
        """
        self.current_attempts.append(AttemptResult(
            response=response,
            model_config=model_config,
            is_valid=is_valid,
            validation_reason=validation_reason,
            attempt_number=attempt_number,
            content_length=len(response)
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

        # No valid responses - apply fallback strategy
        if self.use_longest_on_failure:
            # Select the longest response
            longest = max(attempts, key=lambda a: a.content_length)
            logger.info(
                f"Using longest response ({longest.content_length} chars) from "
                f"{longest.model_config.get('model')} after all validations failed"
            )
            return longest.response
        else:
            # Default: use the last attempt
            last = attempts[-1]
            logger.info(
                f"Using last attempt from {last.model_config.get('model')} "
                f"after all validations failed"
            )
            return last.response

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
            lines.append(
                f"  {status} Attempt {attempt.attempt_number} with {model}: "
                f"{attempt.validation_reason} ({attempt.content_length} chars)"
            )

        return "\n".join(lines)