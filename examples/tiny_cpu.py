"""Run the same lifecycle semantics on a tiny random Transformer without a GPU."""

from llm_lifecycle import LLM, Backend, Chat, Preferences, Text, Train

model = LLM("tiny-random-transformer", backend=Backend.REFERENCE)
model = model.pretrain(Text("builtin://tiny-text"), train=Train(max_steps=4, sequence_length=16))
model = model.sft(Chat("builtin://tiny-chat"), train=Train(max_steps=4, sequence_length=16))
model = model.align(Preferences("builtin://tiny-preferences"), train=Train(max_steps=4, sequence_length=16))
model = model.generate("cpu test: ", max_new_tokens=8)
model.run("runs/tiny-cpu")
