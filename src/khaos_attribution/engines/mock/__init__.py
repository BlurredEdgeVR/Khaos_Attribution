"""The stand-in engine every test and the checklist itself run against: it makes audio from nothing and takes
seconds, and it passes the same checklist a real engine must."""

from __future__ import annotations

from importlib import import_module

from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)


class Engine:
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
