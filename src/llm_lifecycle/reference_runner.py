"""A deliberately tiny causal Transformer used to test lifecycle semantics on CPU."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_TEXT = (
    "Small models make orchestration bugs cheap to find. "
    "A lifecycle should be reproducible before it is distributed. "
)
_CHATS = (
    "user: say hello assistant: hello",
    "user: add one and two assistant: three",
    "user: name a color assistant: blue",
)
_PREFERENCES = (
    ("Answer briefly: ", "clear and correct", "purple elephants calculate clouds"),
    ("Be helpful: ", "use a small reproducible test", "skip every test"),
)


def _load_torch() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        from torch import nn
        from torch.nn import functional as functional  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "the reference backend requires PyTorch; install with `pip install 'llm-lifecycle[reference]'`"
        ) from error
    return torch, nn, functional


def _build_model(nn: Any, *, max_length: int, vocab_size: int = 96) -> Any:
    class TinyCausalTransformer(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            width = 24
            self.token_embedding = nn.Embedding(vocab_size, width)
            self.position_embedding = nn.Embedding(max_length, width)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=2,
                dim_feedforward=48,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.lm_head = nn.Linear(width, vocab_size, bias=False)

        def forward(self, tokens: Any) -> Any:
            length = tokens.shape[1]
            positions = _torch.arange(length, device=tokens.device).unsqueeze(0)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)
            causal_mask = _torch.triu(
                _torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1
            )
            return self.lm_head(self.norm(self.transformer(hidden, mask=causal_mask)))

    _torch, _, _ = _load_torch()
    return TinyCausalTransformer()


def _encode(text: str, *, vocab_size: int = 96) -> list[int]:
    return [(ord(character) % (vocab_size - 1)) + 1 for character in text]


def _decode(tokens: Iterable[int], *, vocab_size: int = 96) -> str:
    return "".join(chr(((token - 1) % (vocab_size - 1)) + 32) for token in tokens if token)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else [value]


def _text_corpus(data: dict[str, Any]) -> list[str]:
    source = str(data["source"])
    if source == "builtin://tiny-text":
        return [_TEXT]
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"reference text source does not exist: {source}")
    return [path.read_text(encoding="utf-8")]


def _chat_corpus(data: dict[str, Any]) -> list[str]:
    source = str(data["source"])
    if source == "builtin://tiny-chat":
        return list(_CHATS)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"reference chat source does not exist: {source}")
    messages_column = str(data.get("messages_column", "messages"))
    examples: list[str] = []
    for row in _read_json_rows(path):
        messages = row[messages_column]
        examples.append(" ".join(f"{item['role']}: {item['content']}" for item in messages))
    return examples


def _preference_corpus(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    source = str(data["source"])
    if source == "builtin://tiny-preferences":
        return list(_PREFERENCES)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"reference preference source does not exist: {source}")
    prompt = str(data.get("prompt_column", "prompt"))
    chosen = str(data.get("chosen_column", "chosen"))
    rejected = str(data.get("rejected_column", "rejected"))
    return [(str(row[prompt]), str(row[chosen]), str(row[rejected])) for row in _read_json_rows(path)]


def _batch(torch: Any, texts: list[str], sequence_length: int, batch_size: int, step: int) -> tuple[Any, Any]:
    sequences: list[list[int]] = []
    for offset in range(batch_size):
        text = texts[(step + offset) % len(texts)]
        tokens = _encode((text + " ") * (math.ceil((sequence_length + 1) / max(len(text), 1)) + 1))
        start = (step * 3 + offset * 5) % max(1, len(tokens) - sequence_length)
        window = tokens[start : start + sequence_length + 1]
        window.extend([0] * (sequence_length + 1 - len(window)))
        sequences.append(window)
    tensor = torch.tensor(sequences, dtype=torch.long)
    return tensor[:, :-1], tensor[:, 1:]


def _train_language_stage(model: Any, stage: dict[str, Any], texts: list[str]) -> tuple[float, float]:
    torch, _, functional = _load_torch()
    train = stage["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]))
    losses: list[float] = []
    model.train()
    for step in range(int(train["max_steps"])):
        inputs, labels = _batch(
            torch,
            texts,
            int(train["sequence_length"]),
            int(train["micro_batch_size"]),
            step,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses[0], losses[-1]


def _sequence_log_probability(torch: Any, functional: Any, model: Any, text: str, length: int) -> Any:
    tokens = _encode(text)
    tokens = (tokens + [0] * (length + 1))[: length + 1]
    tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    logits = model(tensor[:, :-1])
    log_probs = functional.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, tensor[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def _train_dpo_stage(
    model: Any,
    stage: dict[str, Any],
    examples: list[tuple[str, str, str]],
) -> tuple[float, float]:
    torch, _, functional = _load_torch()
    train = stage["train"]
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]))
    beta = float(stage["beta"])
    losses: list[float] = []
    model.train()
    for step in range(int(train["max_steps"])):
        prompt, chosen, rejected = examples[step % len(examples)]
        length = int(train["sequence_length"])
        policy_margin = _sequence_log_probability(torch, functional, model, prompt + chosen, length) - (
            _sequence_log_probability(torch, functional, model, prompt + rejected, length)
        )
        with torch.no_grad():
            chosen_reference = _sequence_log_probability(
                torch, functional, reference, prompt + chosen, length
            )
            rejected_reference = _sequence_log_probability(
                torch, functional, reference, prompt + rejected, length
            )
            reference_margin = chosen_reference - rejected_reference
        loss = -functional.logsigmoid(beta * (policy_margin - reference_margin)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses[0], losses[-1]


def _generate(model: Any, prompt: str, max_new_tokens: int, max_length: int) -> str:
    torch, _, _ = _load_torch()
    tokens = _encode(prompt)[-max_length:]
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor(tokens[-max_length:], dtype=torch.long).unsqueeze(0)
            next_token = int(model(context)[0, -1].argmax())
            tokens.append(next_token)
    return prompt + _decode(tokens[-max_new_tokens:])


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute a serialized reference lifecycle and return its metrics."""

    torch, nn, _ = _load_torch()
    manifest = json.loads(config_path.read_text(encoding="utf-8"))["manifest"]
    stages = manifest["stages"]
    training_stages = [stage for stage in stages if "train" in stage]
    max_length = max((int(stage["train"]["sequence_length"]) for stage in training_stages), default=64)
    seed = int(training_stages[0]["train"]["seed"]) if training_stages else 42
    torch.manual_seed(seed)
    model = _build_model(nn, max_length=max_length)

    metrics: list[dict[str, Any]] = []
    generated_text: str | None = None
    for stage in stages:
        kind = stage["kind"]
        if kind == "pretrain":
            initial, final = _train_language_stage(model, stage, _text_corpus(stage["data"]))
        elif kind == "sft":
            initial, final = _train_language_stage(model, stage, _chat_corpus(stage["data"]))
        elif kind == "align":
            initial, final = _train_dpo_stage(model, stage, _preference_corpus(stage["data"]))
        elif kind == "generate":
            generated_text = _generate(
                model,
                str(stage["prompt"]),
                max_new_tokens=int(stage["max_new_tokens"]),
                max_length=max_length,
            )
            metrics.append({"kind": kind, "generated_text": generated_text})
            continue
        else:
            raise ValueError(f"unsupported reference stage kind: {kind}")
        metrics.append({"kind": kind, "initial_loss": initial, "final_loss": final})

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "tiny-model.pt"
    torch.save({"model": model.state_dict(), "manifest": manifest}, checkpoint)
    result = {
        "backend": "reference",
        "device": "cpu",
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stages": metrics,
        "generated_text": generated_text
        or _generate(model, "test: ", max_new_tokens=8, max_length=max_length),
        "checkpoint": checkpoint.name,
        "evidence_scope": "orchestration and mathematical smoke test; not model-quality evidence",
    }
    (output_dir / "reference-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = execute(arguments.config, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
