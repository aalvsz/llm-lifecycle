from __future__ import annotations

from llm_lifecycle import LLM, Backend, Chat, Preferences, Text, Train


def test_builder_is_immutable_and_keeps_a_portable_manifest() -> None:
    base = LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.AUTOMODEL)
    trained = base.pretrain(Text("HuggingFaceFW/fineweb", text_column="text"), train=Train(max_steps=2))
    aligned = trained.sft(Chat("data/chat.jsonl"), train=Train(max_steps=2)).align(
        Preferences("data/preferences.jsonl"), train=Train(max_steps=2)
    )

    assert base.manifest.stages == ()
    assert len(trained.manifest.stages) == 1
    assert [stage.kind.value for stage in aligned.manifest.stages] == ["pretrain", "sft", "align"]
    assert aligned.manifest.model.model_id == "Qwen/Qwen3-0.6B-Base"


def test_training_parameters_reject_invalid_batching() -> None:
    try:
        Train(global_batch_size=3, micro_batch_size=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid batching was accepted")


def test_generate_and_predict_are_immutable_inference_stages() -> None:
    base = LLM("tiny-random-transformer", backend=Backend.REFERENCE)
    generated = base.generate("hello", max_new_tokens=4)
    predicted = base.predict("world", max_new_tokens=3)

    assert base.manifest.stages == ()
    assert generated.manifest.stages[0].kind.value == "generate"
    assert generated.manifest.stages[0].prompt == "hello"
    assert predicted.manifest.stages[0].max_new_tokens == 3


def test_train_dispatches_text_to_pretraining_and_chat_to_sft() -> None:
    model = LLM("tiny-random-transformer", backend=Backend.REFERENCE)

    assert model.train(Text("builtin://tiny-text")).manifest.stages[0].kind.value == "pretrain"
    assert model.train(Chat("builtin://tiny-chat")).manifest.stages[0].kind.value == "sft"
