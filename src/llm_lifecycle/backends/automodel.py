"""NeMo AutoModel compiler."""

from __future__ import annotations

from typing import Any

from llm_lifecycle.backends.common import compile_hf_generate, compile_nemo_rl, stage_id
from llm_lifecycle.plan import ExecutionPlan, StagePlan
from llm_lifecycle.specs import (
    AlignStage,
    FinetuneMethod,
    GenerateStage,
    LifecycleManifest,
    PretrainStage,
    SFTStage,
    Train,
)


def compile_automodel(manifest: LifecycleManifest) -> ExecutionPlan:
    """Compile trainable stages to current AutoModel recipe YAML."""

    plans: list[StagePlan] = []
    cumulative_steps = 0
    previous_training_stage: PretrainStage | SFTStage | None = None
    previous_was_alignment = False
    for index, stage in enumerate(manifest.stages):
        if isinstance(stage, GenerateStage):
            model_source = stage.checkpoint
            if model_source is None and previous_was_alignment:
                raise ValueError(
                    "NeMo RL output must be converted to Hugging Face format before inference; "
                    "pass generate(..., checkpoint='path/to/hf-checkpoint')"
                )
            if model_source is None and previous_training_stage is not None:
                if (
                    isinstance(previous_training_stage, SFTStage)
                    and previous_training_stage.method is not FinetuneMethod.FULL
                ):
                    raise ValueError(
                        "AutoModel LoRA/DoRA must be merged before inference; "
                        "pass generate(..., checkpoint='path/to/merged-checkpoint')"
                    )
                model_source = "{run_output}/automodel-checkpoints/LATEST/model/consolidated"
            plans.append(
                compile_hf_generate(
                    index,
                    stage,
                    model_source=model_source or manifest.model.model_id,
                )
            )
            continue
        if isinstance(stage, AlignStage):
            model_checkpoint = stage.checkpoint
            if model_checkpoint is None and previous_training_stage is not None:
                if (
                    isinstance(previous_training_stage, SFTStage)
                    and previous_training_stage.method is not FinetuneMethod.FULL
                ):
                    raise ValueError(
                        "AutoModel LoRA/DoRA must be merged to a Hugging Face checkpoint before DPO; "
                        "pass align(..., checkpoint='path/to/merged-checkpoint')"
                    )
                model_checkpoint = "{run_output}/automodel-checkpoints/LATEST/model/consolidated"
            plans.append(
                compile_nemo_rl(
                    index,
                    manifest.model,
                    stage,
                    model_checkpoint=model_checkpoint,
                )
            )
            previous_was_alignment = True
            continue
        cumulative_steps += stage.train.max_steps
        if isinstance(stage, PretrainStage):
            config = _base_config(
                manifest,
                stage.train,
                cumulative_steps=cumulative_steps,
                restore=previous_training_stage is not None,
            )
            config["dataset"] = {
                "_target_": "nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDatasetConfig",
                "file_pattern": stage.data.source,
                "seq_len": stage.train.sequence_length,
                "shuffle_files": True,
                "align_to_bos": False,
            }
        elif isinstance(stage, SFTStage):
            config = _base_config(
                manifest,
                stage.train,
                cumulative_steps=cumulative_steps,
                restore=previous_training_stage is not None,
            )
            config["dataset"] = {
                "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDatasetConfig",
                "path_or_dataset_id": stage.data.source,
                "split": stage.data.split,
                "seq_length": stage.train.sequence_length,
                "padding": "do_not_pad",
                "truncation": True,
            }
            if stage.method is not FinetuneMethod.FULL:
                config["peft"] = {
                    "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
                    "target_modules": "*_proj",
                    "dim": stage.lora_rank,
                    "alpha": stage.lora_alpha,
                    "use_triton": True,
                    **({"use_dora": True} if stage.method is FinetuneMethod.DORA else {}),
                }
        else:
            raise TypeError(f"unsupported AutoModel stage: {type(stage).__name__}")

        plans.append(
            StagePlan(
                stage_id=stage_id(index, stage.kind.value),
                kind=stage.kind,
                executor="nemo-automodel",
                command=("automodel", "{config}", "--nproc-per-node", str(manifest.model.parallel.devices)),
                config=config,
                required_packages=("nemo-automodel",),
                notes=(
                    "JSON is emitted with a .yaml suffix because JSON is valid YAML 1.2.",
                    "Pretraining source is a pre-tokenized NanoGPT .bin glob in this release.",
                ),
            )
        )
        previous_training_stage = stage
        previous_was_alignment = False
    return ExecutionPlan(
        manifest=manifest,
        stages=tuple(plans),
        metadata={
            "compiler": "nemo-automodel",
            "config_contract": "TrainFinetuneRecipeForNextTokenPrediction",
        },
    )


def _base_config(
    manifest: LifecycleManifest,
    train: Train,
    *,
    cumulative_steps: int,
    restore: bool,
) -> dict[str, Any]:
    checkpoint_every = train.checkpoint_every or train.max_steps
    return {
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
        "step_scheduler": {
            "global_batch_size": train.global_batch_size,
            "local_batch_size": train.micro_batch_size,
            "ckpt_every_steps": checkpoint_every,
            "num_epochs": 1,
            "max_steps": cumulative_steps,
        },
        "dist_env": {"backend": "nccl", "timeout_minutes": 30},
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": train.seed,
            "ranked": True,
        },
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": manifest.model.model_id,
        },
        "compile": {"enabled": False, "mode": "default", "fullgraph": False, "dynamic": True},
        "checkpoint": {
            "enabled": True,
            "model_save_format": "safetensors",
            "checkpoint_dir": "{run_output}/automodel-checkpoints",
            "save_consolidated": True,
            **({"restore_from": "LATEST"} if restore else {}),
        },
        "distributed": {
            "strategy": "fsdp2",
            "dp_size": None,
            "tp_size": manifest.model.parallel.tensor,
            "cp_size": manifest.model.parallel.context,
            "sequence_parallel": manifest.model.parallel.tensor > 1,
        },
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "shuffle": False,
        },
        "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": train.learning_rate,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
        "lr_scheduler": {"lr_decay_style": "cosine", "min_lr": train.learning_rate * 0.1},
    }
