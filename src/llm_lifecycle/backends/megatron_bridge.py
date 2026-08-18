"""Megatron Bridge recipe-runner compiler."""

from __future__ import annotations

from llm_lifecycle.backends.common import compile_nemo_rl, stage_id
from llm_lifecycle.plan import ExecutionPlan, StagePlan
from llm_lifecycle.specs import (
    AlignStage,
    FinetuneMethod,
    GenerateStage,
    LifecycleManifest,
    PretrainStage,
    SFTStage,
)

_MODEL_ALIASES = {
    "Qwen/Qwen3-0.6B": "qwen3_600m",
    "Qwen/Qwen3-0.6B-Base": "qwen3_600m",
    "Qwen/Qwen2.5-0.5B": "qwen25_500m",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen25_500m",
    "meta-llama/Llama-3.2-1B": "llama32_1b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama32_1b",
}


def compile_megatron_bridge(manifest: LifecycleManifest) -> ExecutionPlan:
    """Compile pretraining/SFT to Bridge and DPO to NeMo RL's Megatron policy."""

    bridge_model = manifest.model.bridge_model or _MODEL_ALIASES.get(manifest.model.model_id)
    if bridge_model is None:
        raise ValueError(
            f"no Megatron Bridge recipe alias for {manifest.model.model_id!r}; pass bridge_model explicitly"
        )

    plans: list[StagePlan] = []
    previous_checkpoint = manifest.model.bridge_checkpoint
    previous_training_stage: PretrainStage | SFTStage | None = None
    previous_was_alignment = False
    for index, stage in enumerate(manifest.stages):
        if isinstance(stage, GenerateStage):
            inference_checkpoint = stage.checkpoint
            if inference_checkpoint is None and previous_was_alignment:
                raise ValueError(
                    "NeMo RL output needs an explicit converted or native inference checkpoint; "
                    "pass generate(..., checkpoint='path/to/checkpoint')"
                )
            if (
                inference_checkpoint is None
                and isinstance(previous_training_stage, SFTStage)
                and previous_training_stage.method is not FinetuneMethod.FULL
            ):
                raise ValueError(
                    "Megatron Bridge LoRA/DoRA must be merged before inference; "
                    "pass generate(..., checkpoint='path/to/merged-bridge-checkpoint')"
                )
            plans.append(
                _compile_bridge_generate(
                    index,
                    manifest,
                    stage,
                    checkpoint=inference_checkpoint or previous_checkpoint,
                )
            )
            continue
        if isinstance(stage, AlignStage):
            if (
                stage.checkpoint is None
                and isinstance(previous_training_stage, SFTStage)
                and previous_training_stage.method is not FinetuneMethod.FULL
            ):
                raise ValueError(
                    "Megatron Bridge LoRA/DoRA must be merged before NeMo RL DPO; "
                    "pass align(..., checkpoint='path/to/merged-bridge-checkpoint')"
                )
            plans.append(
                compile_nemo_rl(
                    index,
                    manifest.model,
                    stage,
                    model_checkpoint=stage.checkpoint or previous_checkpoint,
                )
            )
            previous_was_alignment = True
            continue
        if isinstance(stage, PretrainStage):
            mode = "pretrain"
            dataset = stage.data.bridge_preset
        elif isinstance(stage, SFTStage):
            mode = stage.method.value if stage.method is not FinetuneMethod.FULL else "sft"
            dataset = stage.data.bridge_preset
        else:
            raise TypeError(f"unsupported Megatron Bridge stage: {type(stage).__name__}")
        if dataset is None:
            raise ValueError(
                f"{stage.kind.value} requires data.bridge_preset for Megatron Bridge's run_recipe.py"
            )
        if isinstance(stage, SFTStage) and previous_checkpoint is None:
            raise ValueError(
                "Megatron Bridge SFT/PEFT needs model weights; add a pretrain stage or pass bridge_checkpoint"
            )

        train = stage.train
        checkpoint_every = train.checkpoint_every or train.max_steps
        command_parts = [
            "python",
            "scripts/training/run_recipe.py",
            "--model",
            bridge_model,
            "--mode",
            mode,
            "--dataset",
            dataset,
            "--max_steps",
            str(train.max_steps),
            "--global_batch_size",
            str(train.global_batch_size),
            "--micro_batch_size",
            str(train.micro_batch_size),
            "--seq_length",
            str(train.sequence_length),
            "--lr",
            str(train.learning_rate),
            "--tensor_model_parallel_size",
            str(manifest.model.parallel.tensor),
            "--pipeline_model_parallel_size",
            str(manifest.model.parallel.pipeline),
            "--context_parallel_size",
            str(manifest.model.parallel.context),
            "--expert_model_parallel_size",
            str(manifest.model.parallel.expert),
            "--save_dir",
            "{stage_output}/checkpoints",
            "--save_interval",
            str(checkpoint_every),
        ]
        if isinstance(stage, SFTStage) and previous_checkpoint is not None:
            command_parts.extend(("--pretrained_checkpoint", previous_checkpoint))
        command = tuple(command_parts)
        config = {
            "runner": "scripts/training/run_recipe.py",
            "model": bridge_model,
            "mode": mode,
            "dataset_preset": dataset,
            "overrides": list(command[8:]),
        }
        plans.append(
            StagePlan(
                stage_id=stage_id(index, stage.kind.value),
                kind=stage.kind,
                executor="megatron-bridge",
                command=command,
                config=config,
                config_suffix=".json",
                required_packages=("megatron-bridge",),
                working_directory_env="MEGATRON_BRIDGE_HOME",
                notes=(
                    "Uses the maintained library recipe runner, not performance-only recipes.",
                    "Set MEGATRON_BRIDGE_HOME to an official Bridge checkout or container mount.",
                ),
            )
        )
        previous_checkpoint = f"{{run_output}}/{stage_id(index, stage.kind.value)}/checkpoints"
        previous_training_stage = stage
        previous_was_alignment = False
    return ExecutionPlan(
        manifest=manifest,
        stages=tuple(plans),
        metadata={"compiler": "megatron-bridge", "recipe_model": bridge_model},
    )


def _compile_bridge_generate(
    index: int,
    manifest: LifecycleManifest,
    stage: GenerateStage,
    *,
    checkpoint: str | None,
) -> StagePlan:
    command_parts = [
        "torchrun",
        "--nproc-per-node",
        str(manifest.model.parallel.devices),
        "scripts/inference/text_generation.py",
        "--hf-model-path",
        manifest.model.model_id,
        "--prompt",
        stage.prompt,
        "--max_new_tokens",
        str(stage.max_new_tokens),
        "--tp",
        str(manifest.model.parallel.tensor),
        "--pp",
        str(manifest.model.parallel.pipeline),
        "--ep",
        str(manifest.model.parallel.expert),
    ]
    if checkpoint is not None:
        command_parts.extend(("--megatron-model-path", checkpoint))
    return StagePlan(
        stage_id=stage_id(index, stage.kind.value),
        kind=stage.kind,
        executor="megatron-bridge-inference",
        command=tuple(command_parts),
        config={
            "hf_model_path": manifest.model.model_id,
            "megatron_model_path": checkpoint,
            "prompt": stage.prompt,
            "max_new_tokens": stage.max_new_tokens,
        },
        config_suffix=".json",
        required_packages=("megatron-bridge", "torch"),
        working_directory_env="MEGATRON_BRIDGE_HOME",
        notes=("Uses Bridge's MCore high-level offline text generation entry point.",),
    )
