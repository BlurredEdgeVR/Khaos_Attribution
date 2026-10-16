"""The mock engine's train slot: the same names as a real engine's, with pure-Python stand-ins, so the checklist and
the rooms' tests can drive the socket without a model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from khaos_attribution.engines.mock import DESCRIPTOR
from khaos_attribution.engines.train import TrainingUnavailable

__all__ = ["TrainingApi", "TrainingUnavailable", "adapter_file_in", "adapter_kwargs", "open_module",
           "preprocess_pairs", "training_api"]


@dataclass(frozen=True)
class TrainingApi:
    preprocess: Any
    Module: Any


def _preprocess(*, output_dir, dataset_json, **_ignored) -> dict:
    """One ``<name>.pt``-named file per sample, holding its caption: a pair in shape, not in content."""
    samples = json.loads(Path(dataset_json).read_text(encoding="utf-8")) if dataset_json else []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        (out / (Path(s["filename"]).stem + ".pt")).write_text(json.dumps({"caption": s.get("caption", "")}))
    return {"processed": len(samples), "failed": 0}


class _Module(SimpleNamespace):
    pass


def training_api(repo_dir=None) -> TrainingApi:
    return TrainingApi(preprocess=_preprocess, Module=_Module)


def adapter_kwargs(cfg: dict) -> tuple:
    return str(cfg.get("adapter_type") or DESCRIPTOR.adapter["type"]), {"rank": int(cfg.get("adapter_rank") or 1)}


def adapter_file_in(directory) -> Path | None:
    directory = Path(directory)
    for name in DESCRIPTOR.adapter["filenames"]:
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("*.safetensors"))
    return files[0] if files else None


def preprocess_pairs(api: TrainingApi, *, out_dir, checkpoint_root=None, variant=None, dataset_json=None, device="cpu",
                     precision=None, audio_dir=None) -> dict:
    return api.preprocess(output_dir=out_dir, dataset_json=dataset_json)


def open_module(api: TrainingApi, cfg: dict, adapter_file, *, checkpoint_root=None, device="cpu", dataset_dir=None,
                output_dir=None, precision=None) -> Any:
    if not Path(adapter_file).is_file():
        raise TrainingUnavailable(f"no adapter file at {adapter_file}")
    return api.Module(dtype="mock", device="cpu", device_type="cpu", adapter=str(adapter_file),
                      model=SimpleNamespace(decoder=SimpleNamespace(config=None)))
