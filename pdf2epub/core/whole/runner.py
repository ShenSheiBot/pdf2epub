"""
Agent-assisted generation loop with continuation support.

The core primitive: generate → agent inspects → Decision(continue/complete).
"""

import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from .sandbox import Sandbox
from .tools import register_tools

# Strip markdown fences (```json ... ```) and BOM from LLM output.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def _strip_fences_and_bom(text: str) -> str:
    """Strip BOM, markdown code fences, and leading/trailing whitespace."""
    text = text.lstrip("\ufeff")
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    text = text.strip()
    # Handle unclosed fences
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:]).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


class Decision(BaseModel):
    """Agent's decision after inspecting the work directory."""

    action: Literal["continue", "complete"]
    file_path: str


class AgentLoopExhausted(Exception):
    """Raised when max_continuations is exceeded without completion."""

    pass


def _create_agent(system_prompt: str, model: Model, sandbox: Sandbox) -> Agent:
    """Create a pydantic-ai agent with the Decision output type and validators."""
    agent = Agent(
        model,
        output_type=Decision,
        system_prompt=system_prompt,
        retries=3,
        output_retries=3,
    )

    @agent.output_validator
    def _validate_decision(ctx, decision: Decision) -> Decision:
        """Enforce workspace path and JSON validity before accepting a decision."""
        work_dir = sandbox.work_dir
        # Resolve path
        p = Path(decision.file_path)
        if not p.is_absolute():
            p = work_dir / p
        resolved = p.resolve()

        # Path must be within work_dir (both actions)
        if not sandbox.is_within_work_dir(resolved):
            raise ModelRetry(
                f"file_path must be inside the work directory. "
                f"You returned: {decision.file_path}. "
                f"Use relative paths like workspace/output.json."
            )

        # Must exist and be a file (not a directory)
        if not resolved.exists():
            raise ModelRetry(
                f"File not found: {decision.file_path}. "
                f"Create the file in workspace/ first."
            )
        if not resolved.is_file():
            raise ModelRetry(
                f"Path is a directory, not a file: {decision.file_path}. "
                f"Specify the actual file path (e.g., workspace/output.json)."
            )

        # For 'complete', must be in workspace/ and valid JSON
        if decision.action == "complete":
            if not sandbox.is_writable_path(resolved):
                raise ModelRetry(
                    f"file_path must point inside workspace/ for 'complete'. "
                    f"You returned: {decision.file_path}. "
                    f"Copy/repair into workspace/ first."
                )
            content = resolved.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                raise ModelRetry(
                    "Selected file is empty. Write repaired JSON to workspace/ first."
                )
            try:
                json.loads(content, strict=False)
            except json.JSONDecodeError as e:
                hint = ""
                if "```" in content:
                    hint = " Remove markdown fences (```...```) and any preamble."
                raise ModelRetry(
                    f"Cannot complete: file is not valid JSON: {e}.{hint} "
                    f"Fix the file and try again."
                )

        return decision

    return agent


async def run_agent_loop(
    generate_fn: Callable[..., str],
    system_prompt: str,
    agent_model: Model,
    max_continuations: int = 5,
    request_limit: int = 100,
    work_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> str:
    """
    Universal agent-assisted generation loop.

    1. Call generate_fn() to get initial output
    2. Save to originals/raw_output.txt in a temporary work directory
    3. Run a pydantic-ai agent with standard tools (bash, read, edit, write, glob, grep)
    4. Agent returns Decision:
       - complete(file_path) → read file, return content
       - continue(file_path) → read file as prefix, call generate_fn(prefix=...),
         save as continuation_NNN.txt, run agent again (fresh run)
    5. If max_continuations exceeded → raise AgentLoopExhausted

    Args:
        generate_fn: Generation function. Signature: generate_fn(prefix=None) -> str.
                     Caller is responsible for constructing multi-turn messages from prefix.
        system_prompt: Agent system prompt.
        agent_model: pydantic-ai Model instance.
        max_continuations: Maximum continuation rounds before giving up.
        request_limit: Max tool calls per agent round.
        work_dir: Work directory (default: auto-create temp directory).
        artifacts_dir: If provided, copy originals/ and workspace/ here before cleanup.

    Returns:
        Final content (from the file the agent marked as complete).

    Raises:
        AgentLoopExhausted: max_continuations exceeded without completion.
    """
    own_work_dir = work_dir is None
    if own_work_dir:
        work_dir = Path(tempfile.mkdtemp(prefix="agent_work_"))

    originals_dir = work_dir / "originals"
    workspace_dir = work_dir / "workspace"

    try:
        originals_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate initial output
        logger.info("[agent-loop] Calling generate_fn for initial output...")
        raw_output = generate_fn()
        if not isinstance(raw_output, str):
            raw_output = str(raw_output) if raw_output is not None else ""
        # Strip markdown fences and BOM before saving
        raw_output = _strip_fences_and_bom(raw_output)
        raw_output_path = originals_dir / "raw_output.txt"
        raw_output_path.write_text(raw_output, encoding="utf-8")
        logger.info(
            f"[agent-loop] Initial output: {len(raw_output)} chars → {raw_output_path.name}"
        )

        # Step 2-5: Agent inspection + continuation loop
        sandbox = Sandbox(work_dir)
        continuation_count = 0
        empty_continuation_streak = 0

        while True:
            # Create a fresh agent each round (with output validator)
            agent = _create_agent(system_prompt, agent_model, sandbox)
            register_tools(agent, sandbox)

            # Build user prompt describing current state
            originals_files = sorted(originals_dir.iterdir())
            file_list = "\n".join(f"  - {f.name}" for f in originals_files)
            user_prompt = (
                f"Work directory contents:\n"
                f"originals/:\n{file_list}\n"
                f"workspace/: (your writable area)\n\n"
                f"Inspect the files and produce your decision."
            )

            logger.info(
                f"[agent-loop] Running agent (round {continuation_count + 1}, "
                f"request_limit={request_limit})..."
            )

            result = await agent.run(
                user_prompt,
                usage_limits=UsageLimits(
                    request_limit=request_limit,
                ),
            )
            decision = result.output

            logger.info(
                f"[agent-loop] Agent decision: action={decision.action}, "
                f"file_path={decision.file_path}"
            )

            try:
                resolved_path = _resolve_decision_path(decision.file_path, work_dir)
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"[agent-loop] Invalid decision path: {e}")
                raise

            if decision.action == "complete":
                # Read the completed file and return
                content = resolved_path.read_text(encoding="utf-8")
                logger.info(
                    f"[agent-loop] Complete. Final output: {len(content)} chars"
                )
                return content

            elif decision.action == "continue":
                continuation_count += 1
                if continuation_count > max_continuations:
                    raise AgentLoopExhausted(
                        f"Agent requested {continuation_count} continuations "
                        f"(max: {max_continuations}). Giving up."
                    )

                # Read the prefix file
                prefix = resolved_path.read_text(encoding="utf-8")
                logger.info(
                    f"[agent-loop] Continuation {continuation_count}/{max_continuations}: "
                    f"prefix={len(prefix)} chars"
                )

                # Call generate_fn with prefix for continuation
                continuation_output = generate_fn(prefix=prefix)
                if not isinstance(continuation_output, str):
                    continuation_output = str(continuation_output) if continuation_output is not None else ""
                # Strip markdown fences and BOM
                continuation_output = _strip_fences_and_bom(continuation_output)

                # Always write the artifact (even if empty) for observability
                continuation_path = originals_dir / f"continuation_{continuation_count:03d}.txt"
                continuation_path.write_text(continuation_output, encoding="utf-8")

                if not continuation_output.strip():
                    empty_continuation_streak += 1
                    logger.warning(
                        f"[agent-loop] Empty continuation output "
                        f"(streak: {empty_continuation_streak}/2)"
                    )
                    if empty_continuation_streak >= 2:
                        raise AgentLoopExhausted(
                            "Continuation model returned empty output twice in a row. "
                            "Aborting to prevent infinite loop."
                        )
                    continue
                else:
                    empty_continuation_streak = 0

                logger.info(
                    f"[agent-loop] Continuation output: {len(continuation_output)} chars "
                    f"→ {continuation_path.name}"
                )
                # Loop back to run agent again on updated workspace

    finally:
        # Preserve artifacts for observability
        if artifacts_dir and work_dir.exists():
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                for subdir in ("originals", "workspace"):
                    src = work_dir / subdir
                    if src.exists():
                        dst = artifacts_dir / subdir
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                logger.debug(f"[agent-loop] Saved artifacts to {artifacts_dir}")
            except OSError as e:
                logger.warning(f"[agent-loop] Failed to save artifacts: {e}")

        if own_work_dir and work_dir.exists():
            try:
                shutil.rmtree(work_dir)
                logger.debug(f"[agent-loop] Cleaned up work dir: {work_dir}")
            except OSError as e:
                logger.warning(f"[agent-loop] Failed to clean up work dir: {e}")


def _resolve_decision_path(file_path: str, work_dir: Path) -> Path:
    """Resolve a file path from a Decision, relative to work_dir.

    Validates that the path is non-empty, within work_dir, exists, and is a file.
    """
    if not file_path or not file_path.strip():
        raise ValueError("Agent returned empty file_path in Decision.")
    p = Path(file_path)
    if not p.is_absolute():
        p = work_dir / p
    # Prevent path traversal — resolved path must be inside work_dir
    resolved = p.resolve()
    work_resolved = work_dir.resolve()
    if not (resolved == work_resolved or str(resolved).startswith(str(work_resolved) + "/")):
        raise ValueError(
            f"Agent referenced path outside work directory: {file_path} "
            f"(resolved to {resolved})"
        )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Agent referenced non-existent file: {file_path} "
            f"(resolved to {resolved})"
        )
    if not resolved.is_file():
        raise ValueError(
            f"Agent referenced a directory, not a file: {file_path} "
            f"(resolved to {resolved})"
        )
    return resolved


def run_agent_loop_sync(
    generate_fn: Callable[..., str],
    system_prompt: str,
    agent_model: Model,
    max_continuations: int = 5,
    request_limit: int = 100,
    work_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> str:
    """Synchronous wrapper for run_agent_loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                run_agent_loop(
                    generate_fn=generate_fn,
                    system_prompt=system_prompt,
                    agent_model=agent_model,
                    max_continuations=max_continuations,
                    request_limit=request_limit,
                    work_dir=work_dir,
                    artifacts_dir=artifacts_dir,
                ),
            )
            return future.result()
    else:
        return asyncio.run(
            run_agent_loop(
                generate_fn=generate_fn,
                system_prompt=system_prompt,
                agent_model=agent_model,
                max_continuations=max_continuations,
                request_limit=request_limit,
                work_dir=work_dir,
                artifacts_dir=artifacts_dir,
            )
        )
