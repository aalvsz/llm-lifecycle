from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from llm_lifecycle import LLM, Backend, Chat


def test_plan_write_is_reproducible_and_json_is_valid_yaml(tmp_path: Path) -> None:
    plan = LLM("Qwen/Qwen3-0.6B", backend=Backend.AUTOMODEL).sft(Chat("data/chat.jsonl")).plan()

    first = plan.write(tmp_path / "first")
    second = plan.write(tmp_path / "second")

    assert (first / "manifest.json").read_text() == (second / "manifest.json").read_text()
    config_path = next(first.glob("*.yaml"))
    assert json.loads(config_path.read_text())["recipe"] == "TrainFinetuneRecipeForNextTokenPrediction"


def test_run_reports_missing_backend_environment_before_mutating(tmp_path: Path) -> None:
    plan = (
        LLM(
            "Qwen/Qwen3-0.6B",
            backend=Backend.MEGATRON_BRIDGE,
            bridge_model="qwen3_600m",
            bridge_checkpoint="/checkpoints/qwen3",
        )
        .sft(Chat("data/chat.jsonl", bridge_preset="mock"))
        .plan()
    )

    try:
        plan.run(tmp_path / "run", environment={})
    except RuntimeError as error:
        assert "MEGATRON_BRIDGE_HOME" in str(error)
    else:
        raise AssertionError("missing backend environment was accepted")


def test_run_materializes_output_placeholders(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("llm_lifecycle.plan.subprocess.run", fake_run)
    plan = LLM("Qwen/Qwen3-0.6B", backend=Backend.AUTOMODEL).sft(Chat("data/chat.jsonl")).plan()
    plan.run(tmp_path / "run", environment={"PATH": "/usr/bin"})

    config_path = next((tmp_path / "run").glob("*.yaml"))
    config = json.loads(config_path.read_text())
    assert "{run_output}" not in config["checkpoint"]["checkpoint_dir"]
    assert calls[0][1] == str(config_path.resolve())
