"""Small Hugging Face inference entry point for consolidated lifecycle checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face inference requires torch and transformers; "
            "install with `pip install 'llm-lifecycle[inference]'`"
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def generate(model_source: str, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    """Load a model id/checkpoint, greedily generate text, and return a JSON-safe result."""

    torch, auto_model, auto_tokenizer = _dependencies()
    tokenizer = auto_tokenizer.from_pretrained(model_source)
    model = auto_model.from_pretrained(model_source)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "model_source": model_source,
        "device": str(device),
        "prompt": prompt,
        "generated_text": text,
        "max_new_tokens": max_new_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = generate(arguments.model, arguments.prompt, arguments.max_new_tokens)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
