"""
Commands module - new CLI commands using the new architecture.

This module provides CLI commands that use the new:
- Phase (composable stages)
- Pipeline V2 (with Executor + Hooks)
- Unified model chain (batch + online)
"""

from .polish_v2 import polish_v2_command
from .translate_v2 import translate_v2_command

__all__ = [
    'polish_v2_command',
    'translate_v2_command',
]
