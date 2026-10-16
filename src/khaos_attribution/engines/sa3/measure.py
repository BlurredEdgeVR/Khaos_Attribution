"""Stable Audio 3's measure slot, not built yet: the descriptor declares no platform measures on this engine, and
every call here says so rather than guessing a number."""

from __future__ import annotations

from typing import Any

from khaos_attribution.engines.sa3 import DESCRIPTOR

SYSTEM_RESERVE_GB = 6.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"
NOT_BUILT = f"{DESCRIPTOR.name} has no influence measure yet; outputs made on it carry no measured split"

__all__ = ["Handle", "LOW_MEMORY_OVERRIDE_ENV", "NOT_BUILT", "SYSTEM_RESERVE_GB", "enable_checkpointing", "flush_accelerator",
           "index_rows", "memory_plan", "output_feature", "output_loss", "paired_loss", "parameter_count", "scorer_need_gb"]


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False) -> float:
    raise NotImplementedError(NOT_BUILT)


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    return {"fits": None, "checkpointed": False, "need_gb": None, "total_gb": total_gb, "refusal": NOT_BUILT}


def enable_checkpointing(module: Any) -> int:
    return 0


def flush_accelerator() -> None:
    return None


# ── the features, by a worker in the engine's environment ─────────────────────────────────────────────────────────

import json  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

MEASURE_WORKER = Path(__file__).with_name("measure_worker.py")


@dataclass
class Handle:
    """What open_module gives for Stable Audio 3: not a module, but where the engine and the adapter are. The
    worker loads them for each call; device_type and dtype are what the contract's readings record."""
    root: Path
    python: Path
    adapter: Path
    device_type: str = "cuda"
    dtype: str = "torch.float32"
    model: str = DESCRIPTOR.variant_config("base") or "medium-base"


def _env(handle: Handle) -> dict:
    import os  # noqa: PLC0415
    return {**os.environ, "PYTHONPATH": str(handle.root), "SA3_ROOT": str(handle.root), "WANDB_MODE": "disabled",
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}


def _ask(handle: Handle, request: dict) -> dict:
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    done = subprocess.run([str(handle.python), str(MEASURE_WORKER)], input=json.dumps(request), cwd=str(handle.root),
                          capture_output=True, text=True, env=_env(handle))
    last = (done.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        reply = json.loads(last[0])
    except ValueError:
        reply = None
    if not isinstance(reply, dict) or reply.get("error") or done.returncode != 0:
        why = (reply or {}).get("error") if isinstance(reply, dict) else "\n".join((done.stderr or "").splitlines()[-6:])
        raise GradientUnavailable(f"the Stable Audio 3 measure worker failed: {why}")
    return reply


def _base(handle: Handle, index_meta: dict, files, command: str) -> dict:
    return {"command": command, "model": handle.model, "adapter": str(Path(handle.adapter).resolve()),
            "files": [str(Path(f).resolve()) for f in files], "timesteps": [float(t) for t in index_meta["timesteps"]],
            "noise_seed": int(index_meta["noise_seed"]), "target": str(index_meta.get("target") or "vnorm"),
            "index_meta": {k: index_meta[k] for k in ("projection",) if k in index_meta}}


def parameter_count(handle: Handle) -> tuple:
    r = _ask(handle, {**_base(handle, {"timesteps": [0.5], "noise_seed": 0}, [], "parameters")})
    return int(r["n_params"]), int(r["n_tensors"]), str(r["signature"])


def index_rows(handle: Handle, files, *, projection, timesteps, noise_seed: int, target: str = "vnorm",
               timestep_norm: bool = False, on_row=None) -> tuple:
    """(matrix, rows) of an index through the worker: the matrix lands beside the files as a temp .npy."""
    import numpy as np  # noqa: PLC0415
    if timestep_norm:
        raise NotImplementedError("per-timestep normalisation is not built on the Stable Audio 3 worker")
    out = Path(files[0]).resolve().parent / ".khaos_index_rows.tmp.npy" if files else Path(".khaos_index_rows.tmp.npy")
    r = _ask(handle, {**_base(handle, {"timesteps": timesteps, "noise_seed": noise_seed, "target": target}, files, "index"),
                      "projection": {"dim": int(projection.dim), "seed": int(projection.seed)}, "out": str(out)})
    matrix = np.load(str(r["matrix"]))
    Path(r["matrix"]).unlink(missing_ok=True)
    if on_row is not None:
        for row in r["rows"]:
            on_row(row["row"], Path(files[row["row"]]), row["gradient_norm"])
    return matrix.astype(np.float32), r["rows"]


def output_feature(handle: Handle, index_meta: dict, files):
    import numpy as np  # noqa: PLC0415
    r = _ask(handle, _base(handle, index_meta, files, "feature"))
    return np.asarray(r["feature"], dtype=np.float32)


def paired_loss(handle: Handle, index_meta: dict, files) -> dict:
    r = _ask(handle, _base(handle, index_meta, files, "loss"))
    return {"with": float(r["with"]), "without": float(r["without"])}


def output_loss(handle: Handle, index_meta: dict, files) -> float:
    return paired_loss(handle, index_meta, files)["with"]
