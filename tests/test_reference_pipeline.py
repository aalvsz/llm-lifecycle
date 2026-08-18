from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from llm_lifecycle import LLM, Backend, Chat, Preferences, Text, Train

torch = pytest.importorskip("torch")


def test_tiny_cpu_model_runs_pretrain_sft_dpo_and_inference(tmp_path: Path) -> None:
    model = (
        LLM("tiny-random-transformer", backend=Backend.REFERENCE)
        .pretrain(
            Text("builtin://tiny-text"),
            train=Train(max_steps=2, learning_rate=0.02, sequence_length=12),
        )
        .sft(Chat("builtin://tiny-chat"), train=Train(max_steps=2, learning_rate=0.02, sequence_length=12))
        .align(
            Preferences("builtin://tiny-preferences"),
            train=Train(max_steps=2, learning_rate=0.01, sequence_length=12),
        )
        .generate("cpu test: ", max_new_tokens=5)
    )

    completed = model.run(tmp_path / "tiny-run")
    results = json.loads((tmp_path / "tiny-run" / "reference-results.json").read_text())

    assert completed[0].returncode == 0
    assert [item["kind"] for item in results["stages"]] == ["pretrain", "sft", "align", "generate"]
    assert all(math.isfinite(item["final_loss"]) for item in results["stages"][:3])
    assert isinstance(results["generated_text"], str) and results["generated_text"]
    assert (tmp_path / "tiny-run" / "tiny-model.pt").exists()
