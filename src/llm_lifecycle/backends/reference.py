"""Compiler for the tiny local reference engine."""

from __future__ import annotations

import sys

from llm_lifecycle.plan import ExecutionPlan, StagePlan
from llm_lifecycle.serialization import manifest_to_dict
from llm_lifecycle.specs import LifecycleManifest, StageKind


def compile_reference(manifest: LifecycleManifest) -> ExecutionPlan:
    """Compile the whole lifecycle into one stateful CPU process."""

    return ExecutionPlan(
        manifest=manifest,
        stages=(
            StagePlan(
                stage_id="stage-00-reference-pipeline",
                kind=manifest.stages[0].kind if manifest.stages else StageKind.PRETRAIN,
                executor="reference",
                command=(
                    sys.executable,
                    "-m",
                    "llm_lifecycle.reference_runner",
                    "--config",
                    "{config}",
                    "--output-dir",
                    "{run_output}",
                ),
                config={"manifest": manifest_to_dict(manifest)},
                config_suffix=".json",
                required_packages=("torch",),
                notes=("Runs all stages in one process so weights flow through the complete lifecycle.",),
            ),
        ),
        metadata={"compiler": "reference", "purpose": "CPU correctness, not quality or throughput"},
    )
