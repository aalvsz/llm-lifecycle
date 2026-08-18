"""Backend compiler registry."""

from __future__ import annotations

from llm_lifecycle.backends.automodel import compile_automodel
from llm_lifecycle.backends.megatron_bridge import compile_megatron_bridge
from llm_lifecycle.backends.reference import compile_reference
from llm_lifecycle.plan import ExecutionPlan
from llm_lifecycle.specs import Backend, LifecycleManifest


def compile_manifest(manifest: LifecycleManifest) -> ExecutionPlan:
    """Compile without importing any optional backend package."""

    if not manifest.stages:
        raise ValueError("the lifecycle has no stages")
    if manifest.model.backend is Backend.REFERENCE:
        return compile_reference(manifest)
    if manifest.model.backend is Backend.AUTOMODEL:
        return compile_automodel(manifest)
    if manifest.model.backend is Backend.MEGATRON_BRIDGE:
        return compile_megatron_bridge(manifest)
    raise ValueError(f"unsupported backend: {manifest.model.backend}")


__all__ = ["compile_manifest"]
