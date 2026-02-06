"""
Validators subsystem - Individual and Batch validators.

Two types of validators:
1. Individual validators: Run after each file is processed
   - Interface: validate(original, processed, key) -> ValidationResult
   - Examples: LengthValidator, TruncationValidatorAdapter

2. Batch validators: Run after all files are processed
   - Interface: validate_batch(files) -> Dict[str, ValidationResult]
   - Examples: PolishBatchValidator, TranslationBatchValidator

Components:
- Truncation detectors: NGram, LLM, Composite
- Batch validators: PolishBatchValidator, TranslationBatchValidator
- VerificationTools: Tools for agent verification
- Individual validators: LengthValidator, TruncationValidatorAdapter
"""

from .truncation import (
    BaseTruncationDetector,
    NGramTruncationDetector,
    LLMTruncationDetector,
    CompositeTruncationDetector,
)
from .agent import (
    # Internal agent classes (used by batch validators)
    AgentVerifier,
    PolishVerificationAgent,
    TranslationVerificationAgent,
    verify_batch,
    # BatchValidator implementations
    PolishBatchValidator,
    TranslationBatchValidator,
)
from .verification_tools import (
    VerificationFile,
    VerificationTools,
)
from .adapters import (
    TruncationValidatorAdapter,
    LengthValidator,
    create_individual_validators,
)

__all__ = [
    # Truncation detectors
    'BaseTruncationDetector',
    'NGramTruncationDetector',
    'LLMTruncationDetector',
    'CompositeTruncationDetector',
    # Batch validators (implement BatchValidator protocol)
    'PolishBatchValidator',
    'TranslationBatchValidator',
    # Internal agent classes (for advanced use)
    'AgentVerifier',
    'PolishVerificationAgent',
    'TranslationVerificationAgent',
    'verify_batch',
    # Verification tools
    'VerificationFile',
    'VerificationTools',
    # Individual validators (implement IndividualValidator protocol)
    'TruncationValidatorAdapter',
    'LengthValidator',
    'create_individual_validators',
]
