# LLM Lifecycle

[![CI](https://github.com/aalvsz/llm-lifecycle/actions/workflows/ci.yml/badge.svg)](https://github.com/aalvsz/llm-lifecycle/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

> **Status:** experimental alpha. The CPU reference lifecycle and backend compilers are tested; NVIDIA GPU and
> distributed execution remain explicit validation gates.

`llm-lifecycle` is an early, executable answer to a simple question: why can a CV model expose
`model.train()` and `model.predict()` while an LLM lifecycle still needs several framework-specific entrypoints?

The repository gives one typed Python API to three deliberately separate execution planes:

- `reference`: a real ~10K-parameter causal Transformer that performs pretraining, SFT, DPO, greedy inference,
  and checkpoint writing on CPU;
- `automodel`: native NeMo AutoModel recipe configuration for PyTorch DTensor/FSDP2 training;
- `megatron_bridge`: native Megatron Bridge library recipe commands for Megatron-Core training;
- DPO on either NVIDIA backend is compiled to NeMo RL, which already supports both DTensor/AutoModel and
  Megatron policies.

It does not install or import the NVIDIA stacks just to build a plan. Production execution stays in NVIDIA's
supported Linux/CUDA environments; the package compiles explicit native configs and commands that can be
reviewed before launch.

## Fifteen-line lifecycle

```python
from llm_lifecycle import Backend, Chat, LLM, Preferences, Text, Train

model = LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.MEGATRON_BRIDGE)
model = model.pretrain(
    Text("data/pretrain", bridge_preset="mock"), train=Train(max_steps=10)
)
model = model.sft(
    Chat("data/chat.jsonl", bridge_preset="tulu3"),
    train=Train(max_steps=10),
)
model = model.align(
    Preferences("data/preferences.jsonl"), train=Train(max_steps=10)
)
model.plan().write("runs/qwen-lifecycle")
```

Replace the final line with `model.run(...)` inside the configured backend environments.

The API is immutable: every method returns a new `LLM`, so a base manifest can be reused safely across
experiments. `plan().write(...)` emits the portable manifest, native stage configs, exact argument vectors,
package requirements, and backend-root requirements.

For an even closer Ultralytics shape, `model.train(Text(...), config=Train(...))` dispatches to pretraining and
`model.train(Chat(...), config=Train(...))` dispatches to SFT. The explicit `.pretrain()` and `.sft()` forms are
preferred in multi-stage manifests because they make intent immediately visible.

Inference is also a stage. `model.generate("Hello", max_new_tokens=32)` (or the `predict` alias) uses
Transformers for AutoModel/Hugging Face checkpoints and Bridge's maintained
`scripts/inference/text_generation.py` for Megatron checkpoints. After NeMo RL, pass an explicitly converted
checkpoint; the compiler refuses to guess whether an RL checkpoint is DCP, Megatron, merged PEFT, or HF.

## CPU-first verification

No GPU is needed to validate control flow, serialization, training math, checkpoint creation, or inference:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[reference,dev]'
python examples/tiny_cpu.py
pytest -q
```

The reference result explicitly labels itself a mathematical/orchestration smoke test. It is not evidence for
quality, distributed correctness, throughput, mixed precision, or CUDA-kernel compatibility.

Plans for every backend can also be generated and inspected without installing an NVIDIA package:

```bash
llm-lifecycle demo --backend automodel --output runs/automodel-plan
llm-lifecycle demo --backend megatron_bridge --output runs/bridge-plan
```

## NeMo AutoModel

```python
model = LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.AUTOMODEL)
model = model.pretrain(Text("data/fineweb_train_*.bin"))
model = model.sft(Chat("data/chat.jsonl"))
model = model.align(Preferences("data/preferences.jsonl"))
model.plan().write("runs/automodel")
```

The generated training files use `TrainFinetuneRecipeForNextTokenPrediction`,
`NeMoAutoModelForCausalLM.from_pretrained`, FSDP2, `ChatDatasetConfig`, AutoModel's LoRA `PeftConfig`, and
safetensors checkpoints. In this first release, pretraining input is an already tokenized NanoGPT `.bin` glob,
matching AutoModel's maintained pretraining example.

Generated AutoModel stages share a checkpoint root, explicitly restore `LATEST`, and use cumulative step limits
so each stage contributes the requested number of updates. The plan wires full-parameter training to NeMo RL's
final consolidated Hugging Face checkpoint path. AutoModel LoRA/DoRA must be merged first, so the compiler
requires `align(..., checkpoint="path/to/merged-checkpoint")` instead of silently aligning the unmodified base;
the production handoff still needs the GPU gate below.

Run AutoModel stages wherever the `automodel` CLI is installed. DPO stages require `NEMO_RL_HOME` to point to
a NeMo RL checkout because they execute `examples/run_dpo.py` with a maintained base config and explicit
overrides.

## Megatron Bridge

```bash
export MEGATRON_BRIDGE_HOME=/workspace/Megatron-Bridge
export NEMO_RL_HOME=/workspace/NeMo-RL
```

Bridge plans call `scripts/training/run_recipe.py` with a library model recipe, mode, dataset preset,
parallelism, optimizer, and checkpoint arguments. Known aliases currently include Qwen3 0.6B, Qwen2.5 0.5B,
and Llama 3.2 1B; `bridge_model=` makes any new recipe stem explicit. A Bridge dataset preset is mandatory
because that is the runner's current contract.

The compiler intentionally does not use performance recipes as convergence recipes. DPO switches NeMo RL to
its Megatron policy by emitting `policy.megatron_cfg.enabled=true` and
`policy.dtensor_cfg.enabled=false`.

The generated Bridge plan feeds pretraining checkpoints to subsequent SFT and wires a full-SFT checkpoint to
NeMo RL through its native `checkpointing.pretrained_checkpoint` contract. As with AutoModel, a PEFT adapter
must be merged before DPO; pass the merged artifact through `align(..., checkpoint=...)`.

## What is verified now

- The core package has no runtime dependencies and does not import AutoModel, Bridge, or NeMo RL to compile.
- The tiny CPU backend executes a stateful pretrain -> SFT -> DPO -> inference -> checkpoint pipeline.
- Contract tests pin current AutoModel config targets, Bridge runner arguments, and NeMo RL backend switches.
- Missing backend-root environment variables fail before run artifacts are written.

Still pending GPU verification: launching the generated commands in official containers, multi-rank behavior,
cross-stage production checkpoint handoff, backend parity, convergence, throughput, and fault recovery. Those
are separate evidence gates, not prerequisites for testing the repository architecture.

## Upstream contracts

- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
- [AutoModel end-to-end recipes](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/overview)
- [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron Bridge training entry points](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/entry-points.html)
- [Megatron Bridge recipe usage](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL)

The broader landscape, impact analysis, Reddit/X evidence, and staged product roadmap live in
[`LLM_LIFECYCLE_REPO_PLAN.md`](LLM_LIFECYCLE_REPO_PLAN.md).
