"""Materialized backend execution plans."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_lifecycle.serialization import manifest_to_dict
from llm_lifecycle.specs import LifecycleManifest, StageKind


@dataclass(frozen=True)
class StagePlan:
    """One executable backend-native stage."""

    stage_id: str
    kind: StageKind
    executor: str
    command: tuple[str, ...]
    config: dict[str, Any]
    config_suffix: str = ".yaml"
    required_packages: tuple[str, ...] = ()
    working_directory_env: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def config_filename(self) -> str:
        """Deterministic filename used by placeholders in ``command``."""

        return f"{self.stage_id}-{self.executor}{self.config_suffix}"


@dataclass(frozen=True)
class ExecutionPlan:
    """Portable manifest plus materialized backend commands."""

    manifest: LifecycleManifest
    stages: tuple[StagePlan, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _document(self) -> dict[str, Any]:
        return {
            "manifest": manifest_to_dict(self.manifest),
            "metadata": self.metadata,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "kind": stage.kind.value,
                    "executor": stage.executor,
                    "command": list(stage.command),
                    "config_file": stage.config_filename,
                    "required_packages": list(stage.required_packages),
                    "working_directory_env": stage.working_directory_env,
                    "notes": list(stage.notes),
                }
                for stage in self.stages
            ],
        }

    def write(self, output_dir: str | Path) -> Path:
        """Write deterministic JSON/YAML-compatible artifacts and return their directory."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").write_text(
            json.dumps(self._document(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for stage in self.stages:
            (destination / stage.config_filename).write_text(
                json.dumps(stage.config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return destination

    def run(
        self,
        output_dir: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> tuple[subprocess.CompletedProcess[str], ...]:
        """Execute stages in order using argument vectors, never a shell."""

        run_environment = dict(os.environ) if environment is None else dict(environment)
        missing = sorted(
            {
                stage.working_directory_env
                for stage in self.stages
                if stage.working_directory_env and not run_environment.get(stage.working_directory_env)
            }
        )
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"missing required backend environment variable(s): {joined}")

        destination = self.write(output_dir)
        completed: list[subprocess.CompletedProcess[str]] = []
        for stage in self.stages:
            stage_output = destination / stage.stage_id
            stage_output.mkdir(parents=True, exist_ok=True)
            substitutions = {
                "{config}": str((destination / stage.config_filename).resolve()),
                "{run_output}": str(destination.resolve()),
                "{stage_output}": str(stage_output.resolve()),
            }
            materialized_config = _replace_value(stage.config, substitutions)
            (destination / stage.config_filename).write_text(
                json.dumps(materialized_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            command = tuple(_replace_tokens(token, substitutions) for token in stage.command)
            working_directory = None
            if stage.working_directory_env is not None:
                working_directory = run_environment[stage.working_directory_env]
            completed.append(
                subprocess.run(
                    command,
                    cwd=working_directory,
                    env=run_environment,
                    check=check,
                    text=True,
                    capture_output=True,
                )
            )
        return tuple(completed)


def _replace_tokens(token: str, substitutions: Mapping[str, str]) -> str:
    result = token
    for placeholder, value in substitutions.items():
        result = result.replace(placeholder, value)
    return result


def _replace_value(value: Any, substitutions: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_tokens(value, substitutions)
    if isinstance(value, list):
        return [_replace_value(item, substitutions) for item in value]
    if isinstance(value, dict):
        return {key: _replace_value(item, substitutions) for key, item in value.items()}
    return value
