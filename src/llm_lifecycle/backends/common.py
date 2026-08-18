"""Shared compiler helpers."""

from __future__ import annotations

from llm_lifecycle.plan import StagePlan
from llm_lifecycle.specs import AlignStage, Backend, GenerateStage, ModelSpec


def stage_id(index: int, kind: str) -> str:
    """Return a stable, sortable stage identifier."""

    return f"stage-{index + 1:02d}-{kind}"


def compile_nemo_rl(
    index: int,
    model: ModelSpec,
    stage: AlignStage,
    *,
    model_checkpoint: str | None = None,
) -> StagePlan:
    """Compile DPO to NeMo RL and select its AutoModel or Megatron policy backend."""

    use_dtensor = model.backend is Backend.AUTOMODEL
    validation_source = stage.data.validation_source or stage.data.source
    policy_model = model_checkpoint if use_dtensor and model_checkpoint else model.model_id
    command_parts = [
        "python",
        "examples/run_dpo.py",
        "--config",
        "examples/configs/dpo.yaml",
        f"dpo.max_num_steps={stage.train.max_steps}",
        f"dpo.seed={stage.train.seed}",
        f"dpo.reference_policy_kl_penalty={stage.beta}",
        f"policy.model_name={policy_model}",
        f"policy.tokenizer.name={model.model_id}",
        f"policy.train_global_batch_size={stage.train.global_batch_size}",
        f"policy.train_micro_batch_size={stage.train.micro_batch_size}",
        f"policy.max_total_sequence_length={stage.train.sequence_length}",
        f"policy.optimizer.kwargs.lr={stage.train.learning_rate}",
        f"policy.dtensor_cfg.enabled={str(use_dtensor).lower()}",
        f"policy.megatron_cfg.enabled={str(not use_dtensor).lower()}",
        f"policy.dtensor_cfg.tensor_parallel_size={model.parallel.tensor}",
        f"policy.dtensor_cfg.context_parallel_size={model.parallel.context}",
        f"policy.megatron_cfg.tensor_model_parallel_size={model.parallel.tensor}",
        f"policy.megatron_cfg.pipeline_model_parallel_size={model.parallel.pipeline}",
        "data.train.dataset_name=BinaryPreferenceDataset",
        f"+data.train.data_path={stage.data.source}",
        f"data.train.split={stage.data.split}",
        "data.validation.dataset_name=BinaryPreferenceDataset",
        f"+data.validation.data_path={validation_source}",
        f"data.validation.split={stage.data.validation_split}",
        "+data.default={dataset_name:BinaryPreferenceDataset,"
        f"prompt_key:{stage.data.prompt_column},chosen_key:{stage.data.chosen_column},"
        f"rejected_key:{stage.data.rejected_column},prompt_file:null,system_prompt_file:null}}",
        "checkpointing.checkpoint_dir={stage_output}/checkpoints",
        f"checkpointing.save_period={stage.train.max_steps}",
        "logger.log_dir={stage_output}/logs",
    ]
    if not use_dtensor and model_checkpoint:
        command_parts.append(
            f"+checkpointing.pretrained_checkpoint={{path:{model_checkpoint},format:megatron_bridge}}"
        )
    command = tuple(command_parts)
    return StagePlan(
        stage_id=stage_id(index, stage.kind.value),
        kind=stage.kind,
        executor="nemo-rl",
        command=command,
        config={
            "base_config": "examples/configs/dpo.yaml",
            "overrides": list(command[4:]),
            "policy_backend": "automodel-dtensor" if use_dtensor else "megatron-bridge",
        },
        config_suffix=".json",
        required_packages=("nemo-rl",),
        working_directory_env="NEMO_RL_HOME",
        notes=("DPO uses NeMo RL's maintained base config plus explicit OmegaConf overrides.",),
    )


def compile_hf_generate(
    index: int,
    stage: GenerateStage,
    *,
    model_source: str,
) -> StagePlan:
    """Compile portable Hugging Face inference for a model id or consolidated checkpoint."""

    command = (
        "python",
        "-m",
        "llm_lifecycle.hf_generate",
        "--model",
        model_source,
        "--prompt",
        stage.prompt,
        "--max-new-tokens",
        str(stage.max_new_tokens),
        "--output",
        "{stage_output}/generation.json",
    )
    return StagePlan(
        stage_id=stage_id(index, stage.kind.value),
        kind=stage.kind,
        executor="hf-transformers",
        command=command,
        config={
            "model_source": model_source,
            "prompt": stage.prompt,
            "max_new_tokens": stage.max_new_tokens,
        },
        config_suffix=".json",
        required_packages=("torch", "transformers"),
        notes=("Inference consumes a Hugging Face model id or consolidated checkpoint.",),
    )
