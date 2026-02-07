"""
CompositeHooks - combines all hook types into a single interface.

This is the main entry point for hooks in the new architecture.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

from ._protocol import (
    PreProcessor, PreProcessResult,
    Transformer,
    Validator, HookResult,
    SkipValidator,
    ErrorClassifier, ErrorType, ErrorEffect,
)
from .error_classifier import DefaultErrorClassifier

if TYPE_CHECKING:
    from .._protocol import ProcessContext
    from ..tracking import ProcessingTracker


class CompositeHooks:
    """
    Combines all hook types into a single interface.

    Usage:
        hooks = CompositeHooks(
            pre_processors=[EmptyContentFilter(), ImageOnlyFilter(book_structure)],
            transformers=[RestoreImagesTransformer()],
            validators=[LengthRatioValidator(), NGramValidator()],
            skip_validators=[ChapterTypeSkipper()],
            error_classifier=DefaultErrorClassifier(),
        )

        # Pre-processing
        pre_result = hooks.pre_process(key, content, context)
        if not pre_result.should_process:
            return pre_result.fallback_result

        # After LLM call
        transformed, hook_result = hooks.post_process(key, original, result, chapter_type, context)
        if not hook_result.accepted:
            # Retry...

        # Error handling
        error_type, effect = hooks.classify_error(error)
    """

    def __init__(
        self,
        pre_processors: Optional[List[PreProcessor]] = None,
        transformers: Optional[List[Transformer]] = None,
        validators: Optional[List[Validator]] = None,
        skip_validators: Optional[List[SkipValidator]] = None,
        error_classifier: Optional[ErrorClassifier] = None,
        tracker: Optional["ProcessingTracker"] = None,
    ):
        """
        Initialize CompositeHooks.

        Args:
            pre_processors: List of pre-processors (any skip = skip)
            transformers: List of transformers (chained)
            validators: List of validators (screener/final logic)
            skip_validators: List of skip validators (any skip = skip validation)
            error_classifier: Error classifier for error handling
            tracker: Optional tracker for recording validation results
        """
        self._pre_processors = pre_processors or []
        self._transformers = transformers or []
        self._validators = validators or []
        self._skip_validators = skip_validators or []
        self._error_classifier = error_classifier or DefaultErrorClassifier()
        self._tracker = tracker

    def pre_process(
        self,
        key: str,
        content: str,
        context: "ProcessContext"
    ) -> PreProcessResult:
        """
        Run pre-processing hooks.

        Any pre-processor that says skip = skip.

        Args:
            key: Unit identifier
            content: Content to check
            context: Processing context

        Returns:
            PreProcessResult with should_process flag
        """
        for pp in self._pre_processors:
            try:
                result = pp.check(key, content, context)
                if not result.should_process:
                    logger.debug(f"{key}: Pre-processor {pp.name} says skip: {result.skip_reason}")
                    return result
            except Exception as e:
                logger.warning(f"{key}: Pre-processor {pp.name} error: {e}")
                # Continue on error

        return PreProcessResult(should_process=True)

    def post_process(
        self,
        key: str,
        original: str,
        result: str,
        chapter_type: str = "",
        context: Optional["ProcessContext"] = None
    ) -> Tuple[str, HookResult]:
        """
        Run post-processing hooks: transform + validate.

        1. Transform (chained): RestoreImages -> RemoveArtifacts -> ...
        2. Skip validation check: ChapterTypeSkipper, etc.
        3. Validate (screener/final logic)

        Args:
            key: Unit identifier
            original: Original content
            result: LLM result to process
            chapter_type: Chapter type for skip validation
            context: Optional processing context

        Returns:
            Tuple of (transformed_result, HookResult)
        """
        # Step 1: Transform (chained)
        transformed = result
        for t in self._transformers:
            try:
                transformed = t.transform(key, original, transformed)
                logger.debug(f"{key}: Transformer {t.name} applied")
            except Exception as e:
                logger.warning(f"{key}: Transformer {t.name} error: {e}")
                # Continue with untransformed result

        # Step 2: Skip validation check
        for sv in self._skip_validators:
            try:
                if sv.should_skip(key, chapter_type, context):
                    logger.debug(f"{key}: SkipValidator {sv.name} says skip validation")
                    return (transformed, HookResult(accepted=True, context_ready=True))
            except Exception as e:
                logger.warning(f"{key}: SkipValidator {sv.name} error: {e}")

        # Step 3: Validate (screener/final logic)
        # Screeners: any pass = pass (short circuit)
        # Finals: all must pass, any fail = fail (short circuit)

        accepted = True
        context_ready = False
        screeners_passed = False

        for v in self._validators:
            try:
                hook_result = v.validate(key, original, transformed)

                # Record validation
                if self._tracker:
                    self._tracker.record_validation(key, {
                        "validator": v.name,
                        "accepted": hook_result.accepted,
                        "context_ready": hook_result.context_ready,
                        "role": getattr(v, 'role', 'unknown'),
                    })

                # Update context_ready
                if hook_result.context_ready:
                    context_ready = True

                # Get role (for adapters and direct validators)
                role = getattr(v, 'role', 'final')

                if role == "screener":
                    # Screener: if it returned context_ready, it passed (short circuit)
                    if hook_result.context_ready:
                        screeners_passed = True
                        logger.debug(f"{key}: Screener {v.name} passed, short-circuiting")
                        break
                    # Otherwise continue to next validator
                else:  # final
                    # Final: if not accepted, reject immediately
                    if not hook_result.accepted:
                        accepted = False
                        logger.debug(f"{key}: Final validator {v.name} rejected")
                        break

            except Exception as e:
                logger.warning(f"{key}: Validator {v.name} error: {e}")
                # Continue on error

        # If any screener passed, we're good
        if screeners_passed:
            accepted = True

        # Debug: log only first few and when context_ready is False (unexpected)
        if not context_ready and len(self._validators) > 0:
            logger.debug(f"{key}: post_process finished with context_ready=False (validators={[v.name for v in self._validators]})")

        return (transformed, HookResult(accepted=accepted, context_ready=context_ready))

    def classify_error(self, error: Exception) -> Tuple[ErrorType, ErrorEffect]:
        """
        Classify error and get its effect.

        Args:
            error: The exception to classify

        Returns:
            Tuple of (ErrorType, ErrorEffect)
        """
        error_type = self._error_classifier.classify(error)
        effect = self._error_classifier.get_effect(error_type)
        return (error_type, effect)

    def get_error_effect(self, error_type: ErrorType) -> ErrorEffect:
        """
        Get the effect for a known error type (without re-classifying).

        Args:
            error_type: The error type

        Returns:
            ErrorEffect describing how to modify unit state
        """
        return self._error_classifier.get_effect(error_type)

    def should_skip_validation(
        self,
        key: str,
        chapter_type: str,
        context: Optional["ProcessContext"]
    ) -> bool:
        """
        Check if validation should be skipped (convenience method).

        Args:
            key: Unit identifier
            chapter_type: Chapter type
            context: Processing context

        Returns:
            True if any skip validator says skip
        """
        for sv in self._skip_validators:
            try:
                if sv.should_skip(key, chapter_type, context):
                    return True
            except Exception as e:
                logger.warning(f"{key}: SkipValidator {sv.name} error: {e}")
        return False
