"""The socket every base-model engine plugs into: one registry, a descriptor per engine read as data, and the slots
(provision, train, render, measure) the rooms call. Engine packages live beside this module; their heavy imports are
lazy, so reading a descriptor never loads a model."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from khaos_attribution.engines.descriptor import Descriptor, DescriptorError, load_descriptor, packaged_descriptor

# Engine id -> the package that implements it. Adding an engine is one line here and one package.
KNOWN_ENGINES = {
    "acestep-1.5": "khaos_attribution.engines.acestep",
    "mock": "khaos_attribution.engines.mock",
}


class EngineUnavailable(RuntimeError):
    """The engine is not installed here, or its package cannot import; the sentence says which."""


def descriptor(engine_id: str) -> Descriptor:
    """An engine's descriptor by id, without importing its code."""
    package = KNOWN_ENGINES.get(engine_id)
    if package is None:
        raise EngineUnavailable(f"no engine is known as {engine_id!r}; known: {', '.join(sorted(KNOWN_ENGINES))}")
    return packaged_descriptor(package)


def descriptors() -> dict:
    return {engine_id: descriptor(engine_id) for engine_id in KNOWN_ENGINES}


def engine(engine_id: str):
    """The engine object (its slots), importing its package; EngineUnavailable with the reason when it cannot."""
    package = KNOWN_ENGINES.get(engine_id)
    if package is None:
        raise EngineUnavailable(f"no engine is known as {engine_id!r}")
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise EngineUnavailable(f"engine {engine_id!r} cannot import: {exc}") from exc
    return module.Engine()


@runtime_checkable
class Engine(Protocol):
    """What every engine object carries: its descriptor, and the three slots as modules (render: a renderer class
    and request; train: the training door, configs and open_module; measure: memory plan and checkpointing)."""
    descriptor: Descriptor

    @property
    def render(self): ...

    @property
    def train(self): ...

    @property
    def measure(self): ...


__all__ = ["Descriptor", "DescriptorError", "Engine", "EngineUnavailable", "KNOWN_ENGINES", "descriptor",
           "descriptors", "engine", "load_descriptor", "packaged_descriptor"]
