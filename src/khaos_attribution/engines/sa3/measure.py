"""Stable Audio 3's measure slot, not built yet: the descriptor declares no platform measures on this engine, and
every call here says so rather than guessing a number."""

from __future__ import annotations

from typing import Any

from khaos_attribution.engines.sa3 import DESCRIPTOR

SYSTEM_RESERVE_GB = 6.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"
NOT_BUILT = f"{DESCRIPTOR.name} has no influence measure yet; outputs made on it carry no measured split"

__all__ = ["LOW_MEMORY_OVERRIDE_ENV", "NOT_BUILT", "SYSTEM_RESERVE_GB", "enable_checkpointing", "flush_accelerator",
           "memory_plan", "scorer_need_gb"]


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False) -> float:
    raise NotImplementedError(NOT_BUILT)


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    return {"fits": None, "checkpointed": False, "need_gb": None, "total_gb": total_gb, "refusal": NOT_BUILT}


def enable_checkpointing(module: Any) -> int:
    return 0


def flush_accelerator() -> None:
    return None
