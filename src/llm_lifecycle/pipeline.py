"""Ultralytics-style immutable public API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from subprocess import CompletedProcess

from llm_lifecycle.specs import (
    AlignmentMethod,
    AlignStage,
    Backend,
    Chat,
    FinetuneMethod,
    GenerateStage,
    LifecycleManifest,
    ModelSpec,
    Parallel,
    Preferences,
    PretrainStage,
    SFTStage,
    Text,
    Train,
)


@dataclass(frozen=True, init=False)
class LLM:
    """A model plus an immutable sequence of lifecycle operations."""

    manifest: LifecycleManifest

    def __init__(
        self,
        model_id: str,
        *,
        backend: Backend | str = Backend.REFERENCE,
        bridge_model: str | None = None,
        bridge_checkpoint: str | None = None,
        parallel: Parallel | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        selected_backend = backend if isinstance(backend, Backend) else Backend(backend)
        object.__setattr__(
            self,
            "manifest",
            LifecycleManifest(
                model=ModelSpec(
                    model_id=model_id,
                    backend=selected_backend,
                    bridge_model=bridge_model,
                    bridge_checkpoint=bridge_checkpoint,
                    parallel=parallel or Parallel(),
                )
            ),
        )

    @classmethod
    def _from_manifest(cls, manifest: LifecycleManifest) -> LLM:
        instance = object.__new__(cls)
        object.__setattr__(instance, "manifest", manifest)
        return instance

    def _append(self, stage: object) -> LLM:
        if not isinstance(stage, PretrainStage | SFTStage | AlignStage | GenerateStage):
            raise TypeError(f"unsupported lifecycle stage: {type(stage).__name__}")
        return self._from_manifest(
            LifecycleManifest(
                model=self.manifest.model,
                stages=(*self.manifest.stages, stage),
                schema_version=self.manifest.schema_version,
            )
        )

    def pretrain(self, data: Text, *, train: Train | None = None) -> LLM:
        """Add causal next-token training."""

        return self._append(PretrainStage(data=data, train=train or Train()))

    def train(
        self,
        data: Text | Chat,
        *,
        config: Train | None = None,
        method: FinetuneMethod | str = FinetuneMethod.FULL,
    ) -> LLM:
        """Ultralytics-style dispatcher: text means pretraining, chat means SFT."""

        if isinstance(data, Text):
            return self.pretrain(data, train=config)
        if isinstance(data, Chat):
            return self.sft(data, train=config, method=method)
        raise TypeError(f"train data must be Text or Chat, got {type(data).__name__}")

    def sft(
        self,
        data: Chat,
        *,
        train: Train | None = None,
        method: FinetuneMethod | str = FinetuneMethod.FULL,
        lora_rank: int = 8,
        lora_alpha: int = 32,
    ) -> LLM:
        """Add full-parameter, LoRA, or DoRA supervised tuning."""

        selected_method = method if isinstance(method, FinetuneMethod) else FinetuneMethod(method)
        return self._append(
            SFTStage(
                data=data,
                train=train or Train(),
                method=selected_method,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        )

    def align(
        self,
        data: Preferences,
        *,
        train: Train | None = None,
        method: AlignmentMethod | str = AlignmentMethod.DPO,
        beta: float = 0.1,
        checkpoint: str | None = None,
    ) -> LLM:
        """Add DPO post-training through NeMo RL."""

        selected_method = method if isinstance(method, AlignmentMethod) else AlignmentMethod(method)
        return self._append(
            AlignStage(
                data=data,
                train=train or Train(),
                method=selected_method,
                beta=beta,
                checkpoint=checkpoint,
            )
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        checkpoint: str | None = None,
    ) -> LLM:
        """Add an offline inference request."""

        return self._append(
            GenerateStage(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                checkpoint=checkpoint,
            )
        )

    def predict(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        checkpoint: str | None = None,
    ) -> LLM:
        """Alias for :meth:`generate`, matching familiar model APIs."""

        return self.generate(prompt, max_new_tokens=max_new_tokens, checkpoint=checkpoint)

    def plan(self) -> ExecutionPlan:
        """Compile the portable lifecycle into backend-native stages."""

        from llm_lifecycle.backends import compile_manifest

        return compile_manifest(self.manifest)

    def run(
        self,
        output_dir: str | PathLike[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> tuple[CompletedProcess[str], ...]:
        """Compile and execute the lifecycle, failing before launch on missing backend roots."""

        return self.plan().run(Path(output_dir), environment=environment)


from llm_lifecycle.plan import ExecutionPlan  # noqa: E402  (typing plus public discoverability)
