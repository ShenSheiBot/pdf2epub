"""
Cancel active batch jobs.

Usage:
    pdf2epub cancel-batch [-c CONFIG]
"""

import argparse
import json
from pathlib import Path
from typing import Optional
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

    from ..core.factory_v2 import get_task_model_configs
    from ..utils.batch_utils import create_batch_client_from_config

    def batch_model_for_task(task_type: str) -> Optional[dict]:
        return next(
            (
                model
                for model in get_task_model_configs(config, task_type)
                if model.get("mode", "online") == "batch"
            ),
            None,
        )

    configured_models = [
        model
        for task_type in ("polish", "translate")
        if (model := batch_model_for_task(task_type)) is not None
    ]

    def make_client(provider: str, model: str):
        return create_batch_client_from_config(
            config,
            provider=provider,
            model=model,
        )

    cancelled_count = 0

    if args.all:
        if not configured_models:
            logger.error("No batch model found in polish/translation config.")
            return 1

        # This intentionally has provider-wide impact and is only reached by
        # the explicit --all flag.
        try:
            from ..utils.batch_utils import BatchJobState
            active_states = {
                BatchJobState.PENDING,
                BatchJobState.RUNNING,
                BatchJobState.QUEUED,
                BatchJobState.CANCELLING,
            }
            seen_providers = set()
            for model_config in configured_models:
                provider = model_config.get("provider", "gemini")
                if provider in seen_providers:
                    continue
                seen_providers.add(provider)
                client = make_client(provider, model_config.get("model", "any"))
                for job in client.list_jobs(limit=50):
                    if job.state not in active_states:
                        continue
                    try:
                        client.cancel(job.name)
                        logger.info(f"Cancelled: {job.name}")
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel {job.name}: {e}")
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return 1
    else:
        state_files = sorted(output_dir.glob("*/batch_states/batch_*.json"))
        toc_state_path = output_dir / "toc_translation_batch_state.json"
        if toc_state_path.exists():
            state_files.append(toc_state_path)

        if not state_files:
            logger.info("No batch state files found. Nothing to cancel.")
            return 0

        from ..core.executor.batch_state import MegaUnitState

        had_error = False
        for state_file in state_files:
            provider = None
            model = None
            job_name = None
            is_toc_state = state_file == toc_state_path

            if is_toc_state:
                try:
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    provider = data.get("provider")
                    model = data.get("model")
                    job_name = data.get("job_name")
                except Exception as e:
                    logger.error(f"Cannot read TOC batch state {state_file}: {e}")
                    had_error = True
                    continue
            else:
                state = MegaUnitState.load(state_file)
                if state:
                    job_name = state.job_name
                    provider = state.provider
                    model = state.model
                stage_name = state_file.parent.parent.name
                task_type = (
                    "polish"
                    if stage_name == "polished_markdown"
                    else "translate"
                    if stage_name == "translated"
                    else None
                )
                model_config = (
                    batch_model_for_task(task_type)
                    if task_type
                    else None
                )
                if model_config and (not provider or not model):
                    provider = model_config.get("provider", "gemini")
                    model = model_config.get("model")

            if not job_name:
                logger.error(
                    f"State {state_file} has no job name; submission may be "
                    "indeterminate. State was retained for manual inspection."
                )
                had_error = True
                continue
            if not provider or not model:
                logger.error(
                    f"Cannot determine provider/model for {state_file}; "
                    "state was retained."
                )
                had_error = True
                continue

            try:
                client = make_client(provider, model)
                cancelled = client.cancel(job_name)
                if cancelled is False:
                    raise RuntimeError(
                        "provider did not confirm cancellation"
                    )
                logger.info(f"Cancelled: {job_name}")
                cancelled_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to cancel job from {state_file}: {e}. "
                    "State was retained."
                )
                had_error = True
                continue

            state_file.unlink(missing_ok=True)
            if is_toc_state:
                state_file.with_suffix(".response.txt").unlink(missing_ok=True)

        if had_error:
            return 1

    if cancelled_count == 0:
        logger.info("No active batch jobs found.")
    else:
        logger.info(f"Cancelled {cancelled_count} batch job(s).")

    return 0
