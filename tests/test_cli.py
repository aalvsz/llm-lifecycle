from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_lifecycle.cli import main


@pytest.mark.parametrize(
    ("backend", "expected_executors"),
    [
        ("reference", ["reference"]),
        ("automodel", ["nemo-automodel", "nemo-automodel", "nemo-rl"]),
        ("megatron_bridge", ["megatron-bridge", "megatron-bridge", "nemo-rl"]),
    ],
)
def test_demo_materializes_each_backend_without_optional_packages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    backend: str,
    expected_executors: list[str],
) -> None:
    output = tmp_path / backend

    assert main(["demo", "--backend", backend, "--output", str(output)]) == 0

    document = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert document["manifest"]["model"]["backend"] == backend
    assert [stage["executor"] for stage in document["stages"]] == expected_executors
    assert len(list(output.glob("stage-*-*.*"))) == len(expected_executors)
    assert capsys.readouterr().out.strip() == str(output.resolve())
