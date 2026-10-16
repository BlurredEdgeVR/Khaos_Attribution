"""The mock engine's measure slot: everything fits, nothing is checkpointed, no accelerator to flush."""

from __future__ import annotations

from typing import Any

SCORER_BASE_GB = 0.1
SYSTEM_RESERVE_GB = 0.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"

__all__ = ["LOW_MEMORY_OVERRIDE_ENV", "SCORER_BASE_GB", "SYSTEM_RESERVE_GB", "enable_checkpointing",
           "flush_accelerator", "memory_plan", "scorer_need_gb"]


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False) -> float:
    return SCORER_BASE_GB


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    return {"fits": "resident", "checkpointed": False, "need_gb": SCORER_BASE_GB, "total_gb": total_gb, "refusal": None}


def enable_checkpointing(module: Any) -> int:
    return 0


def flush_accelerator() -> None:
    return None
