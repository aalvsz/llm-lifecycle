from __future__ import annotations

from llm_lifecycle import LLM, Backend, Chat, Preferences, Text, Train


def test_automodel_compiler_emits_native_recipe_config() -> None:
    model = LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.AUTOMODEL).sft(
        Chat("data/chat.jsonl", split="train"),
        train=Train(max_steps=3, global_batch_size=4, micro_batch_size=1, sequence_length=128),
        method="lora",
    )

    plan = model.plan()
    stage = plan.stages[0]

    assert stage.executor == "nemo-automodel"
    assert stage.command[:2] == ("automodel", "{config}")
    assert stage.config["recipe"] == "TrainFinetuneRecipeForNextTokenPrediction"
    assert stage.config["model"]["_target_"] == "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
    assert stage.config["dataset"]["_target_"].endswith("ChatDatasetConfig")
    assert stage.config["peft"]["_target_"].endswith("PeftConfig")
    assert stage.config["distributed"]["strategy"] == "fsdp2"


def test_megatron_bridge_compiler_uses_library_recipe_runner_and_checkpoints() -> None:
    model = LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.MEGATRON_BRIDGE).pretrain(
        Text("builtin://tiny-text", bridge_preset="mock"),
        train=Train(max_steps=3, global_batch_size=4, micro_batch_size=1, sequence_length=128),
    )

    stage = model.plan().stages[0]

    assert stage.executor == "megatron-bridge"
    assert stage.working_directory_env == "MEGATRON_BRIDGE_HOME"
    assert stage.command[:4] == ("python", "scripts/training/run_recipe.py", "--model", "qwen3_600m")
    assert "--mode" in stage.command and "pretrain" in stage.command
    assert "--dataset" in stage.command and "mock" in stage.command
    assert "--save_dir" in stage.command and "{stage_output}/checkpoints" in stage.command
    assert "--tensor_model_parallel_size" in stage.command
    assert "--pipeline_model_parallel_size" in stage.command


def test_megatron_bridge_hands_pretraining_checkpoint_to_sft() -> None:
    plan = (
        LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.MEGATRON_BRIDGE)
        .pretrain(Text("data/pretrain", bridge_preset="mock"))
        .sft(Chat("data/chat.jsonl", bridge_preset="tulu3"), method="lora")
        .plan()
    )

    assert "--pretrained_checkpoint" in plan.stages[1].command
    checkpoint_index = plan.stages[1].command.index("--pretrained_checkpoint") + 1
    assert plan.stages[1].command[checkpoint_index].endswith("stage-01-pretrain/checkpoints")


def test_automodel_resumes_one_shared_checkpoint_with_cumulative_steps() -> None:
    plan = (
        LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.AUTOMODEL)
        .pretrain(Text("data/pretrain_*.bin"), train=Train(max_steps=2))
        .sft(Chat("data/chat.jsonl"), train=Train(max_steps=3))
        .plan()
    )

    assert plan.stages[0].config["step_scheduler"]["max_steps"] == 2
    assert plan.stages[1].config["step_scheduler"]["max_steps"] == 5
    assert plan.stages[1].config["checkpoint"]["restore_from"] == "LATEST"
    assert (
        plan.stages[0].config["checkpoint"]["checkpoint_dir"]
        == plan.stages[1].config["checkpoint"]["checkpoint_dir"]
    )


def test_peft_to_dpo_requires_an_explicit_merged_checkpoint() -> None:
    model = (
        LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.MEGATRON_BRIDGE)
        .pretrain(Text("data/pretrain", bridge_preset="mock"))
        .sft(Chat("data/chat.jsonl", bridge_preset="tulu3"), method="lora")
        .align(Preferences("data/preferences.jsonl"))
    )

    try:
        model.plan()
    except ValueError as error:
        assert "merged" in str(error)
    else:
        raise AssertionError("PEFT adapter was treated as a full DPO checkpoint")


def test_nemo_rl_compiler_selects_dtensor_or_megatron_from_same_stage() -> None:
    data = Preferences("data/preferences.jsonl")
    train = Train(max_steps=3)

    automodel = LLM("Qwen/Qwen3-0.6B", backend=Backend.AUTOMODEL).align(data, train=train).plan().stages[0]
    megatron = (
        LLM("Qwen/Qwen3-0.6B", backend=Backend.MEGATRON_BRIDGE, bridge_model="qwen3_600m")
        .align(data, train=train)
        .plan()
        .stages[0]
    )

    assert automodel.executor == megatron.executor == "nemo-rl"
    assert "policy.dtensor_cfg.enabled=true" in automodel.command
    assert "policy.megatron_cfg.enabled=false" in automodel.command
    assert "policy.dtensor_cfg.enabled=false" in megatron.command
    assert "policy.megatron_cfg.enabled=true" in megatron.command
    assert "+data.train.data_path=data/preferences.jsonl" in automodel.command
    assert "dpo.reference_policy_kl_penalty=0.1" in automodel.command


def test_compiling_nvidia_backends_does_not_import_them() -> None:
    model = LLM("Qwen/Qwen3-0.6B", backend=Backend.AUTOMODEL).sft(Chat("data/chat.jsonl"))
    assert model.plan().stages[0].required_packages == ("nemo-automodel",)


def test_inference_compiles_to_hf_for_automodel_and_native_bridge_for_megatron() -> None:
    automodel = (
        LLM("Qwen/Qwen3-0.6B", backend=Backend.AUTOMODEL).generate("Hello", max_new_tokens=4).plan().stages[0]
    )
    bridge = (
        LLM("Qwen/Qwen3-0.6B", backend=Backend.MEGATRON_BRIDGE, bridge_model="qwen3_600m")
        .generate("Hello", max_new_tokens=4)
        .plan()
        .stages[0]
    )

    assert automodel.executor == "hf-transformers"
    assert automodel.command[:3] == ("python", "-m", "llm_lifecycle.hf_generate")
    assert bridge.executor == "megatron-bridge-inference"
    assert "scripts/inference/text_generation.py" in bridge.command
    assert "--hf-model-path" in bridge.command
    assert bridge.working_directory_env == "MEGATRON_BRIDGE_HOME"
