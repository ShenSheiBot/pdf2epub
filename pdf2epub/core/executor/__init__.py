"""
Executor module - handles LLM execution with state management.

Core design:
- No retry loop: failed units update state and re-enter pool
- Per-unit state (chain, quotas, dependencies)
- Batch + Online run simultaneously (not if-else)
- Unified dependency tree (context injection + aggregation)

Components:
- WorkUnit: A unit of work to process
- ChainEntry: Model entry with mode (batch/online)
- UnitState: Per-unit mutable state
- QuotaConfig: Quota configuration
- Executor: Unified executor (batch + online simultaneously)
"""

from ._protocol import (
    WorkUnit,
    ChainEntry,
    ExecutionResult,
    ProcessResult,
    Executor as ExecutorProtocol,
)

from .state import (
    UnitState,
    QuotaConfig,
    create_unit_state,
    remove_batch_entries,
    remove_provider,
    chain_from_model_configs,
)

from .executor import Executor, handle_split

__all__ = [
    # Protocol and types
    'WorkUnit',
    'ChainEntry',
    'ExecutionResult',
    'ProcessResult',
    'ExecutorProtocol',
    # State management
    'UnitState',
    'QuotaConfig',
    'create_unit_state',
    'remove_batch_entries',
    'remove_provider',
    'chain_from_model_configs',
    # Executor
    'Executor',
    'handle_split',
]
