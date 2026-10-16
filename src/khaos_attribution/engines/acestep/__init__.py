"""ACE-Step 1.5 as a Khaos engine. The descriptor is data (engine.json); the slots import ACE-Step only when called."""

from __future__ import annotations

from importlib import import_module

from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)


class Engine:
    """The slots: each a module, imported on first use so reading the descriptor never loads torch."""
    descriptor = DESCRIPTOR

    @property
    def render(self):
        return import_module(__name__ + ".render")

    @property
    def train(self):
        return import_module(__name__ + ".train")

    @property
    def measure(self):
        return import_module(__name__ + ".measure")
