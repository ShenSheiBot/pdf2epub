"""
Phase implementation - a composable processing stage.

Key design:
- Any phase can follow any other
- No aggregation between phases
- Each phase reads parts and outputs parts
- Aggregation only happens at build-epub
"""

from typing import Optional, List, TYPE_CHECKING
from pathlib import Path
from loguru import logger
import time

from ._protocol import PhaseResult, WorkUnitLoader
from .loader import PartBasedLoader
from ..executor import WorkUnit

if TYPE_CHECKING:
    from .._protocol import ProcessorProtocol
    from ..pipeline import ProcessingPipeline


class Phase:
    """
    A composable processing phase.

    Phase is an orchestrator - it does NOT save files directly.
    Pipeline's ResultPersistence handles all file I/O to output_dir/validated/.

    Usage:
        polish_phase = Phase(
            name="polish",
            input_dir=output_dir / "pages_merged",
            output_dir=output_dir / "polished",
            pipeline=polish_pipeline,
        )
        result = polish_phase.run(resume=True)

        # Chain to next phase - read from previous phase's validated/ dir
        translate_phase = Phase(
            name="translate",
            input_dir=output_dir / "polished" / "validated",
            output_dir=output_dir / "translated",
            pipeline=translate_pipeline,
        )
        result = translate_phase.run(resume=True)
    """

    def __init__(
        self,
        name: str,
        input_dir: Path,
        output_dir: Path,
        pipeline: "ProcessingPipeline",
        loader: Optional[WorkUnitLoader] = None,
        file_pattern: str = "*.md",
    ):
        """
        Initialize phase.

        Args:
            name: Phase name (polish, translate, etc.)
            input_dir: Directory to read input from
            output_dir: Directory to write output to
            pipeline: ProcessingPipeline to use
            loader: WorkUnitLoader (default: PartBasedLoader)
            file_pattern: Glob pattern for input files
        """
        self._name = name
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._pipeline = pipeline
        self._loader = loader or PartBasedLoader()
        self._file_pattern = file_pattern

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def run(self, resume: bool = False) -> PhaseResult:
        """
        Run the phase.

        Args:
            resume: If True, skip already completed units

        Returns:
            PhaseResult with statistics
        """
        start_time = time.time()

        logger.info(f"Starting phase '{self._name}'")
        logger.info(f"  Input: {self._input_dir}")
        logger.info(f"  Output: {self._output_dir}")

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Load work units
        units = self._loader.load_units(self._input_dir, self._file_pattern)

        if not units:
            logger.warning(f"No units found in {self._input_dir}")
            return PhaseResult(
                phase=self._name,
                total=0,
                completed=0,
                failed=0,
            )

        logger.info(f"Loaded {len(units)} units")

        # Process via pipeline (Pipeline's persistence handles saving)
        # The pipeline owns resume filtering because an active persisted batch
        # may need units whose validated files were written before a crash.
        # Pre-filtering here would destroy exact mega-unit membership.
        result = self._pipeline.process_all(units, resume=resume)

        duration = time.time() - start_time

        logger.info(
            f"Phase '{self._name}' completed: "
            f"{result.completed}/{result.total} succeeded, "
            f"{result.failed} failed in {duration:.1f}s"
        )

        return PhaseResult(
            phase=self._name,
            total=result.total,
            completed=result.completed,
            failed=result.failed,
            failed_keys=result.failed_keys if hasattr(result, 'failed_keys') else [],
            duration_seconds=duration,
        )

    def _filter_completed(self, units: List[WorkUnit]) -> List[WorkUnit]:
        """Filter out units that already exist in validated directory."""
        pending = []
        validated_dir = self._output_dir / "validated"
        for unit in units:
            # Check validated directory (where Pipeline's persistence saves)
            validated_path = validated_dir / f"{unit.id}.md"
            if not validated_path.exists():
                pending.append(unit)
            else:
                logger.debug(f"Skipping completed unit: {unit.id}")
        return pending
