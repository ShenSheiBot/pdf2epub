"""
Cancel active batch jobs.

Usage:
    pdf2epub cancel-batch [-c CONFIG]
"""

import argparse
from pathlib import Path
from loguru import logger

from ..utils.common import load_config


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the cancel-batch command."""
    parser = subparsers.add_parser(
        "cancel-batch",
        help="Cancel active batch jobs",
        description="Cancel all active batch jobs for the current project.",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Cancel ALL batch jobs (not just from state files)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Run the cancel-batch command."""
    config = load_config(args.config)

    title = config.get("title", "unknown")
    output_dir = Path(config.get("output_dir", "output")) / title

    # Find all batch_states directories (in subdirectories like polished_markdown/, translated/)
    batch_states_dirs = list(output_dir.glob("*/batch_states"))

    # Get batch client from config
    from ..utils.batch_utils import GeminiBatchClient, VertexBatchClient

    # Find batch provider in config
    batch_provider = None
    for section in ['polish', 'translation']:
        models = config.get(section, {}).get('models', [])
        for m in models:
            if m.get('mode') == 'batch':
                batch_provider = m.get('provider')
                break
        if batch_provider:
            break

    if not batch_provider:
        logger.error("No batch provider found in config (mode: batch).")
        return 1

    # Get credentials for provider
    creds = config.get('credentials', {}).get('providers', {}).get(batch_provider)
    if not creds:
        logger.error(f"No credentials found for provider: {batch_provider}")
        return 1

    if batch_provider == "vertex":
        project = creds.get('project')
        if not project:
            logger.error("No 'project' found for vertex provider")
            return 1
        batch_client = VertexBatchClient(
            project=project,
            location=creds.get('location', 'us-central1'),
            model='any',
        )
    else:
        batch_client = GeminiBatchClient(
            api_key=creds.get('api_key', ''),
            base_url=creds.get('base_url'),
            model='any',  # Not used for cancel/list
        )

    cancelled_count = 0

    if args.all:
        # Cancel all jobs from API
        try:
            from ..utils.batch_utils import BatchJobState
            jobs = batch_client.list_jobs(limit=50)
            for job in jobs:
                if job.state in (BatchJobState.PENDING, BatchJobState.RUNNING,
                                BatchJobState.QUEUED, BatchJobState.CANCELLING):
                    try:
                        batch_client.cancel(job.name)
                        logger.info(f"Cancelled: {job.name}")
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel {job.name}: {e}")
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return 1
    else:
        # Cancel jobs from state files
        if not batch_states_dirs:
            logger.info("No batch state files found. Nothing to cancel.")
            return 0

        from ..core.executor.batch_state import MegaUnitState

        for batch_states_dir in batch_states_dirs:
            for state_file in batch_states_dir.glob("batch_*.json"):
                try:
                    state = MegaUnitState.load(state_file)
                    if state and state.job_name:
                        batch_client.cancel(state.job_name)
                        logger.info(f"Cancelled: {state.job_name}")
                        cancelled_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cancel job from {state_file}: {e}")
                finally:
                    # Always remove state file
                    state_file.unlink(missing_ok=True)

    if cancelled_count == 0:
        logger.info("No active batch jobs found.")
    else:
        logger.info(f"Cancelled {cancelled_count} batch job(s).")

    return 0
