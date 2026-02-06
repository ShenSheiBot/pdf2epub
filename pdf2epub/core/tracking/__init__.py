"""
Tracking subsystem - handles state tracking and validation decisions.

Components:
- ProcessingTracker: Complete state tracking with audit log
- ValidationStrategy: Validation decision matrix
"""

from .tracker import (
    ProcessingTracker,
    AttemptRecord,
    SplitRecord,
    ErrorType,
)
from .validation_strategy import (
    ValidationStrategy,
    AttemptResult,
)

__all__ = [
    # ProcessingTracker
    'ProcessingTracker',
    'AttemptRecord',
    'SplitRecord',
    'ErrorType',
    # ValidationStrategy
    'ValidationStrategy',
    'AttemptResult',
]
