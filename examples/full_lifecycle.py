"""A complete lifecycle in 14 executable lines."""

from llm_lifecycle import LLM, Backend, Chat, Preferences, Text, Train

model = LLM("Qwen/Qwen3-0.6B-Base", backend=Backend.MEGATRON_BRIDGE)
model = model.pretrain(Text("data/pretrain", bridge_preset="mock"), train=Train(max_steps=10))
model = model.sft(
    Chat("data/chat.jsonl", bridge_preset="tulu3"),
    train=Train(max_steps=10),
)
model = model.align(Preferences("data/preferences.jsonl"), train=Train(max_steps=10))
model.plan().write("runs/qwen-lifecycle")  # use model.run(...) in NVIDIA containers
