"""
Validators - judge processing results.

Validators determine if a result is acceptable.
They can be configured as screeners (OR logic) or finals (AND logic).
"""

from typing import Literal, Optional, Any, TYPE_CHECKING
from loguru import logger
from ._protocol import HookResult

if TYPE_CHECKING:
    from .._protocol import IndividualValidator


class IndividualValidatorAdapter:
    """
    Adapter to use IndividualValidator as a hooks.Validator.

    Handles screener vs final logic:
    - Screener: pass = accepted + context_ready, fail = continue (not rejected)
    - Final: pass = accepted, fail = rejected
    """

    def __init__(
        self,
        validator: "IndividualValidator",
        role: Literal["screener", "final"],
        context_ready: bool = False
    ):
        """
        Initialize adapter.

        Args:
            validator: The IndividualValidator to wrap
            role: "screener" (OR logic) or "final" (AND logic)
            context_ready: If True, passing means result can be used for context injection
        """
        self._validator = validator
        self._role = role
        self._context_ready = context_ready

    @property
    def name(self) -> str:
        return self._validator.name

    @property
    def role(self) -> str:
        return self._role

    def validate(self, key: str, original: str, result: str) -> HookResult:
        """
        Validate result using the wrapped validator.

        Args:
            key: Unit identifier
            original: Original content
            result: Processed result

        Returns:
            HookResult based on role:
            - Screener pass: accepted=True, context_ready=self._context_ready
            - Screener fail: accepted=True (continue to next), context_ready=False
            - Final pass: accepted=True, context_ready=self._context_ready
            - Final fail: accepted=False
        """
        validation_result = self._validator.validate(original, result, key)

        if self._role == "screener":
            if validation_result.is_valid:
                # Screener passed - short circuit with success
                return HookResult(accepted=True, context_ready=self._context_ready)
            else:
                # Screener failed - not conclusive, continue to next
                return HookResult(accepted=True, context_ready=False)
        else:  # final
            if validation_result.is_valid:
                return HookResult(accepted=True, context_ready=self._context_ready)
            else:
                # Final failed - reject with reason
                reason = getattr(validation_result, 'reason', None) or f"{self.name} rejected"
                return HookResult(accepted=False, context_ready=False, rejection_reason=reason)


class LengthRatioValidator:
    """
    Simple length ratio validator.

    Checks that result length is within acceptable ratio of original.
    """

    def __init__(
        self,
        min_ratio: float = 0.3,
        max_ratio: float = 3.0,
        role: Literal["screener", "final"] = "screener",
        context_ready: bool = True
    ):
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._role = role
        self._context_ready = context_ready

    @property
    def name(self) -> str:
        return "LengthRatioValidator"

    @property
    def role(self) -> str:
        return self._role

    def validate(self, key: str, original: str, result: str) -> HookResult:
        """Check length ratio."""
        orig_len = len(original.strip()) if original else 0
        result_len = len(result.strip()) if result else 0

        if orig_len == 0:
            # Can't compute ratio, accept
            return HookResult(accepted=True, context_ready=self._context_ready)

        ratio = result_len / orig_len

        is_valid = self._min_ratio <= ratio <= self._max_ratio

        if self._role == "screener":
            if is_valid:
                return HookResult(accepted=True, context_ready=self._context_ready)
            else:
                return HookResult(accepted=True, context_ready=False)  # Continue
        else:
            if is_valid:
                return HookResult(accepted=True, context_ready=self._context_ready)
            else:
                reason = f"Length ratio {ratio:.2f} outside range [{self._min_ratio}, {self._max_ratio}]"
                return HookResult(accepted=False, context_ready=False, rejection_reason=reason)


class NonEmptyValidator:
    """Validate that result is not empty."""

    def __init__(self, role: Literal["screener", "final"] = "final"):
        self._role = role

    @property
    def name(self) -> str:
        return "NonEmptyValidator"

    @property
    def role(self) -> str:
        return self._role

    def validate(self, key: str, original: str, result: str) -> HookResult:
        """Check that result is not empty."""
        is_valid = bool(result and result.strip())

        if self._role == "screener":
            if is_valid:
                return HookResult(accepted=True, context_ready=True)
            else:
                return HookResult(accepted=True, context_ready=False)
        else:
            if is_valid:
                return HookResult(accepted=True, context_ready=True)
            else:
                return HookResult(accepted=False, context_ready=False, rejection_reason="Result is empty")


class TruncationValidator:
    """
    Truncation detection validator using N-gram analysis.

    Uses the sophisticated NGramTruncationDetector to detect if output
    is truncated vs properly deduplicated.
    """

    def __init__(
        self,
        min_unique_preserved_ratio: float = 0.60,
        allow_deduplication: bool = True,
        role: Literal["screener", "final"] = "final",
        context_ready: bool = True
    ):
        """
        Initialize truncation validator.

        Args:
            min_unique_preserved_ratio: Minimum ratio of unique content to preserve
            allow_deduplication: Whether deduplication is acceptable
            role: "screener" (OR logic) or "final" (AND logic)
            context_ready: If True, passing means result can be used for context injection
        """
        self._min_ratio = min_unique_preserved_ratio
        self._allow_dedup = allow_deduplication
        self._role = role
        self._context_ready = context_ready
        self._detector = None

    @property
    def name(self) -> str:
        return "TruncationValidator"

    @property
    def role(self) -> str:
        return self._role

    def _get_detector(self):
        """Lazy load detector to avoid import issues."""
        if self._detector is None:
            from ..validators.truncation import NGramTruncationDetector
            self._detector = NGramTruncationDetector(
                min_unique_preserved_ratio=self._min_ratio,
                allow_deduplication=self._allow_dedup
            )
        return self._detector

    def validate(self, key: str, original: str, result: str) -> HookResult:
        """
        Check for truncation using N-gram analysis.

        Args:
            key: Unit identifier
            original: Original content
            result: Processed result

        Returns:
            HookResult based on truncation detection
        """
        try:
            detector = self._get_detector()
            is_truncated, reason, details = detector.detect(original, result)

            is_valid = not is_truncated

            if is_truncated:
                logger.debug(f"{key}: Truncation detected - {reason}")

            if self._role == "screener":
                if is_valid:
                    return HookResult(accepted=True, context_ready=self._context_ready)
                else:
                    # Screener: truncation detected but continue to next validator
                    return HookResult(accepted=True, context_ready=False)
            else:  # final
                if is_valid:
                    return HookResult(accepted=True, context_ready=self._context_ready)
                else:
                    return HookResult(
                        accepted=False,
                        context_ready=False,
                        rejection_reason=f"Truncation detected: {reason}"
                    )

        except Exception as e:
            logger.warning(f"{key}: Truncation check error: {e}")
            # On error, accept (don't block processing)
            return HookResult(accepted=True, context_ready=self._context_ready)


class CompositeTruncationValidator:
    """
    Composite truncation validator using N-gram + LLM fallback.

    Uses NGram detection first, then LLM verification if unique content loss suspected.
    """

    def __init__(
        self,
        llm_client: Any,
        min_unique_preserved_ratio: float = 0.60,
        allow_deduplication: bool = True,
        truncation_check_lines: int = 5,
        task_type: str = "translate",
        role: Literal["screener", "final"] = "final",
        context_ready: bool = True
    ):
        """
        Initialize composite truncation validator.

        Args:
            llm_client: LLM client for LLM-based verification
            min_unique_preserved_ratio: Minimum ratio of unique content to preserve
            allow_deduplication: Whether deduplication is acceptable
            truncation_check_lines: Number of lines to check with LLM
            task_type: "translate" or "polish"
            role: "screener" or "final"
            context_ready: If True, passing means result can be used for context injection
        """
        self._llm_client = llm_client
        self._min_ratio = min_unique_preserved_ratio
        self._allow_dedup = allow_deduplication
        self._check_lines = truncation_check_lines
        self._task_type = task_type
        self._role = role
        self._context_ready = context_ready
        self._detector = None

    @property
    def name(self) -> str:
        return "CompositeTruncationValidator"

    @property
    def role(self) -> str:
        return self._role

    def _get_detector(self):
        """Lazy load detector."""
        if self._detector is None:
            from ..validators.truncation import CompositeTruncationDetector
            self._detector = CompositeTruncationDetector(
                llm_client=self._llm_client,
                min_unique_preserved_ratio=self._min_ratio,
                allow_deduplication=self._allow_dedup,
                truncation_check_lines=self._check_lines,
                task_type=self._task_type
            )
        return self._detector

    def validate(self, key: str, original: str, result: str) -> HookResult:
        """
        Check for truncation using composite N-gram + LLM detection.
        """
        try:
            detector = self._get_detector()
            is_truncated, reason, details = detector.detect(original, result)

            is_valid = not is_truncated

            if is_truncated:
                logger.debug(f"{key}: Truncation detected (composite) - {reason}")

            if self._role == "screener":
                if is_valid:
                    return HookResult(accepted=True, context_ready=self._context_ready)
                else:
                    return HookResult(accepted=True, context_ready=False)
            else:
                if is_valid:
                    return HookResult(accepted=True, context_ready=self._context_ready)
                else:
                    return HookResult(
                        accepted=False,
                        context_ready=False,
                        rejection_reason=f"Truncation detected (composite): {reason}"
                    )

        except Exception as e:
            logger.warning(f"{key}: Composite truncation check error: {e}")
            return HookResult(accepted=True, context_ready=self._context_ready)
