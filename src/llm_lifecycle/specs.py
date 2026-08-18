"""Typed, backend-neutral lifecycle specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Backend(StrEnum):
    """Execution engine selected for trainable stages."""

    REFERENCE = "reference"
    AUTOMODEL = "automodel"
    MEGATRON_BRIDGE = "megatron_bridge"


class StageKind(StrEnum):
    """Kinds of stages supported by the first vertical slice."""

    PRETRAIN = "pretrain"
    SFT = "sft"
    ALIGN = "align"
    GENERATE = "generate"


class FinetuneMethod(StrEnum):
    """Supervised fine-tuning strategy."""

    FULL = "full"
    LORA = "lora"
    DORA = "dora"


class AlignmentMethod(StrEnum):
    """Post-training objective."""

    DPO = "dpo"


@dataclass(frozen=True)
class Parallel:
    """Portable parallelism settings compiled into backend-native names."""

    devices: int = 1
    tensor: int = 1
    pipeline: int = 1
    context: int = 1
    expert: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("devices", self.devices),
            ("tensor", self.tensor),
            ("pipeline", self.pipeline),
            ("context", self.context),
            ("expert", self.expert),
        ):
            if value < 1:
                raise ValueError(f"parallel.{name} must be at least 1")


@dataclass(frozen=True)
class Train:
    """Parameters shared by pretraining, SFT, and preference optimization."""

    max_steps: int = 10
    global_batch_size: int = 4
    micro_batch_size: int = 1
    learning_rate: float = 1e-4
    sequence_length: int = 128
    seed: int = 42
    checkpoint_every: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.micro_batch_size < 1 or self.global_batch_size < 1:
            raise ValueError("batch sizes must be at least 1")
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError("global_batch_size must be divisible by micro_batch_size")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.checkpoint_every is not None and self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")


@dataclass(frozen=True)
class Text:
    """Next-token data.

    AutoModel treats ``source`` as a NanoGPT ``.bin`` file pattern. Megatron
    Bridge uses ``bridge_preset`` with its recipe runner. The reference engine
    accepts ``builtin://tiny-text``.
    """

    source: str
    split: str = "train"
    text_column: str = "text"
    bridge_preset: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("text source must not be empty")


@dataclass(frozen=True)
class Chat:
    """OpenAI-format chat data or a Hugging Face dataset identifier."""

    source: str
    split: str = "train"
    messages_column: str = "messages"
    bridge_preset: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("chat source must not be empty")


@dataclass(frozen=True)
class Preferences:
    """Chosen/rejected preference pairs for DPO."""

    source: str
    validation_source: str | None = None
    split: str = "train"
    validation_split: str = "validation"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("preference source must not be empty")


@dataclass(frozen=True)
class ModelSpec:
    """Model and backend identity."""

    model_id: str
    backend: Backend
    bridge_model: str | None = None
    bridge_checkpoint: str | None = None
    parallel: Parallel = field(default_factory=Parallel)


@dataclass(frozen=True)
class PretrainStage:
    """Causal next-token training stage."""

    data: Text
    train: Train
    kind: StageKind = field(default=StageKind.PRETRAIN, init=False)


@dataclass(frozen=True)
class SFTStage:
    """Supervised instruction-tuning stage."""

    data: Chat
    train: Train
    method: FinetuneMethod = FinetuneMethod.FULL
    lora_rank: int = 8
    lora_alpha: int = 32
    kind: StageKind = field(default=StageKind.SFT, init=False)

    def __post_init__(self) -> None:
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be at least 1")


@dataclass(frozen=True)
class AlignStage:
    """Preference-optimization stage delegated to NeMo RL."""

    data: Preferences
    train: Train
    method: AlignmentMethod = AlignmentMethod.DPO
    beta: float = 0.1
    checkpoint: str | None = None
    kind: StageKind = field(default=StageKind.ALIGN, init=False)

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("DPO beta must be positive")


@dataclass(frozen=True)
class GenerateStage:
    """One offline text-generation request."""

    prompt: str
    max_new_tokens: int = 32
    checkpoint: str | None = None
    kind: StageKind = field(default=StageKind.GENERATE, init=False)

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("generation prompt must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")


LifecycleStage = PretrainStage | SFTStage | AlignStage | GenerateStage


@dataclass(frozen=True)
class LifecycleManifest:
    """Versioned, portable source of truth for a lifecycle."""

    model: ModelSpec
    stages: tuple[LifecycleStage, ...] = ()
    schema_version: int = 1
