"""Stable JSON-boundary serialization for typed lifecycle specifications."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from llm_lifecycle.specs import LifecycleManifest


def to_primitive(value: object) -> Any:
    """Convert a lifecycle value into JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple | list):
        return [to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    return value


def manifest_to_dict(manifest: LifecycleManifest) -> dict[str, Any]:
    """Serialize a manifest without importing a backend."""

    result = to_primitive(manifest)
    if not isinstance(result, dict):
        raise TypeError("manifest serialization did not produce an object")
    return result
