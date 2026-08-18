"""Command-line utilities for generating or running a minimal lifecycle."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lifecycle import LLM, Backend, Chat, Preferences, Text, Train


def _demo(backend: Backend) -> LLM:
    if backend is Backend.REFERENCE:
        return (
            LLM("tiny-random-transformer", backend=backend)
            .pretrain(Text("builtin://tiny-text"), train=Train(max_steps=2, sequence_length=12))
            .sft(Chat("builtin://tiny-chat"), train=Train(max_steps=2, sequence_length=12))
            .align(Preferences("builtin://tiny-preferences"), train=Train(max_steps=2, sequence_length=12))
        )
    if backend is Backend.AUTOMODEL:
        return (
            LLM("Qwen/Qwen3-0.6B-Base", backend=backend)
            .pretrain(Text("data/fineweb_train_*.bin"), train=Train(max_steps=10))
            .sft(Chat("data/chat.jsonl"), train=Train(max_steps=10))
            .align(Preferences("data/preferences.jsonl"), train=Train(max_steps=10))
        )
    return (
        LLM("Qwen/Qwen3-0.6B-Base", backend=backend)
        .pretrain(Text("data/fineweb", bridge_preset="mock"), train=Train(max_steps=10))
        .sft(Chat("data/chat.jsonl", bridge_preset="mock"), train=Train(max_steps=10))
        .align(Preferences("data/preferences.jsonl"), train=Train(max_steps=10))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-lifecycle", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="materialize a three-stage example")
    demo.add_argument("--backend", choices=[backend.value for backend in Backend], default="reference")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--run", action="store_true", help="execute after materializing")
    arguments = parser.parse_args(argv)

    model = _demo(Backend(arguments.backend))
    if arguments.run:
        model.run(arguments.output)
    else:
        model.plan().write(arguments.output)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
