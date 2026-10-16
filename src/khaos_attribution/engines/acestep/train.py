"""ACE-Step's training side behind the socket: the one door to its training_v2 modules, the precision rule on
Apple silicon, the adapter and training configs built from a recipe, the preprocessing call, and a live module
with an adapter's weights loaded, which is the object the gradient feature is taken on. Both rooms call these;
neither imports ACE-Step's training code by itself."""

from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from khaos_attribution.engines.acestep import DESCRIPTOR

log = logging.getLogger(__name__)

PRECISION = "bf16"


from khaos_attribution.engines.train import TrainingUnavailable  # noqa: E402,F401


@dataclass(frozen=True)
class TrainingApi:
    """The engine's training classes, by Khaos's names, so a rename upstream is absorbed in one place."""
    preprocess: Any
    Trainer: Any
    TrainingConfig: Any
    LoRAConfig: Any
    LoKRConfig: Any
    load_decoder: Any
    save_final: Any
    Module: Any
    select_dtype: Any

    def as_tuple(self) -> tuple:
        """The seven the Workshop's door hands out, in its order."""
        return (self.preprocess, self.Trainer, self.TrainingConfig, self.LoRAConfig, self.LoKRConfig,
                self.load_decoder, self.save_final)


def preload_npp() -> None:
    """torch's CUDA wheels do not preload NVIDIA's NPP, which torchcodec's CUDA build links; load it when the venv
    carries it."""
    import ctypes  # noqa: PLC0415
    try:
        import nvidia.npp as npp  # noqa: PLC0415
    except ImportError:
        return
    lib = Path(next(iter(npp.__path__))) / "lib"
    for name in ("libnppc.so.12", "libnppicc.so.12"):
        try:
            ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return


def training_api(repo_dir: Path | str) -> TrainingApi:
    """ACE-Step's training_v2 door: the checkout on the path, its classes imported, the MPS precision rule applied."""
    root = str(Path(repo_dir).resolve())
    if not Path(root).is_dir():
        raise TrainingUnavailable(f"ACE-Step not found at {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    preload_npp()
    try:
        from acestep.training_v2.configs import LoKRConfigV2, LoRAConfigV2, TrainingConfigV2  # noqa: PLC0415
        from acestep.training_v2.fixed_lora_module import FixedLoRAModule, _select_compute_dtype  # noqa: PLC0415
        from acestep.training_v2.model_loader import load_decoder_for_training  # noqa: PLC0415
        from acestep.training_v2.preprocess import preprocess_audio_files  # noqa: PLC0415
        from acestep.training_v2.trainer_fixed import FixedLoRATrainer  # noqa: PLC0415
        from acestep.training_v2.trainer_helpers import save_final  # noqa: PLC0415
    except ImportError as exc:
        raise TrainingUnavailable(f"ACE-Step's training modules are not importable from {root}: {exc}") from exc
    patch_mps_precision()
    return TrainingApi(preprocess=preprocess_audio_files, Trainer=FixedLoRATrainer, TrainingConfig=TrainingConfigV2,
                       LoRAConfig=LoRAConfigV2, LoKRConfig=LoKRConfigV2, load_decoder=load_decoder_for_training,
                       save_final=save_final, Module=FixedLoRAModule, select_dtype=_select_compute_dtype)


def patch_mps_precision() -> None:
    """MPS trains in bf16 as CUDA does: the engine's fp16 trips the GradScaler there and fp32 gives non-finite
    gradients. Both modules import the selectors by value, so both are patched; a revision whose selectors take
    requested_precision honours the config already and is left alone."""
    import torch  # noqa: PLC0415
    from acestep.training_v2 import fixed_lora_module as _flm  # noqa: PLC0415
    from acestep.training_v2 import trainer_fixed as _tf  # noqa: PLC0415

    if getattr(_flm, "_khaos_mps_precision_patched", False):
        return
    orig_dtype, orig_precision = _flm._select_compute_dtype, _flm._select_fabric_precision
    if "requested_precision" in inspect.signature(orig_dtype).parameters:
        _flm._khaos_mps_precision_patched = True
        log.info("ACE-Step's dtype selector honours requested_precision; MPS precision patch not needed")
        return

    def _compute_dtype(device_type: str):
        return torch.bfloat16 if device_type == "mps" else orig_dtype(device_type)

    def _fabric_precision(device_type: str):
        return "bf16-mixed" if device_type == "mps" else orig_precision(device_type)

    for mod in (_flm, _tf):
        if hasattr(mod, "_select_compute_dtype"):
            mod._select_compute_dtype = _compute_dtype
        if hasattr(mod, "_select_fabric_precision"):
            mod._select_fabric_precision = _fabric_precision
    _flm._khaos_mps_precision_patched = True
    log.info("Patched ACE-Step MPS precision to bf16 / bf16-mixed")


def accepted_kwargs(cls: Any, kwargs: dict, *, keep_none: bool = True, what: str | None = None) -> dict:
    """Only the kwargs this revision's constructor takes, the rest logged and left out; a callable whose signature
    cannot be read takes them all. None values are dropped unless kept, so a default applies in their place."""
    try:
        accepted = set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        accepted = set(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in accepted and (keep_none or v is not None)}
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        log.warning("%s on this ACE-Step revision does not accept %s; omitting", what or getattr(cls, "__name__", cls),
                    dropped)
    return kept


def accepted_config(cls: Any, kwargs: dict, *, keep_none: bool = True) -> Any:
    """``cls`` built from the kwargs it takes."""
    return cls(**accepted_kwargs(cls, kwargs, keep_none=keep_none))


def adapter_kwargs(cfg: dict) -> tuple:
    """(adapter type, the adapter config's candidate kwargs) a recipe describes."""
    adapter_type = str(cfg.get("adapter_type") or DESCRIPTOR.adapter["type"])
    rank, alpha = int(cfg["adapter_rank"]), int(cfg["adapter_alpha"])
    if adapter_type == "lokr":
        return adapter_type, {"linear_dim": rank, "linear_alpha": alpha, "factor": int(cfg["factor"]),
                              "weight_decompose": bool(cfg["weight_decompose"]),
                              "attention_type": str(cfg["attention_type"]), "target_modules": cfg.get("target_modules")}
    return adapter_type, {"r": rank, "alpha": alpha, "dropout": float(cfg.get("dropout", 0.0)),
                          "attention_type": str(cfg["attention_type"]), "target_modules": cfg.get("target_modules")}


def adapter_config(api: TrainingApi, cfg: dict, *, keep_none: bool = True) -> tuple:
    """(adapter type, the engine's adapter config) for a recipe."""
    adapter_type, kwargs = adapter_kwargs(cfg)
    cls = api.LoKRConfig if adapter_type == "lokr" else api.LoRAConfig
    return adapter_type, accepted_config(cls, kwargs, keep_none=keep_none)


def resolve_device(requested: str = "auto") -> str:
    """The device TYPE ("cuda", "mps", "cpu"), never an index like cuda:0: what an index records."""
    import torch  # noqa: PLC0415
    if requested and requested != "auto":
        return torch.device(requested).type
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure_precision(requested: str, device_type: str) -> str:
    """The trainer's rule for a feature: bf16 on MPS whatever was asked (fp16 overflows there), else as asked."""
    return PRECISION if device_type == "mps" else requested


def preprocess_precision(device_type: str) -> str:
    """What preprocessing runs in: bf16 on MPS, the engine's own choice elsewhere."""
    return PRECISION if device_type == "mps" else "auto"


def compute_dtype(selector: Any, device_type: str, precision: str) -> Any:
    """The compute dtype the trainer uses for ``precision``: the selector is asked for it when it takes
    requested_precision; older revisions get bf16 on MPS, as the patch gives the trainer."""
    import torch  # noqa: PLC0415
    try:
        takes = "requested_precision" in inspect.signature(selector).parameters
    except (TypeError, ValueError):
        takes = False
    if takes:
        return selector(device_type, requested_precision=precision)
    return torch.bfloat16 if device_type == "mps" else selector(device_type)


def dtype_tag(dtype: Any) -> str:
    """``torch.bfloat16`` -> ``"bf16"``: the short name results are filed under."""
    name = str(dtype).replace("torch.", "")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(name, name)


def adapter_file_in(directory: Path | str) -> Path | None:
    """The adapter weights inside a checkpoint directory: the descriptor's names first, then any safetensors."""
    directory = Path(directory)
    for name in DESCRIPTOR.adapter["filenames"]:
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("*.safetensors"))
    return files[0] if files else None


def preprocess_pairs(api: TrainingApi, *, out_dir: Path | str, checkpoint_root: Path | str, variant: str,
                     dataset_json: Path | str | None, device: str, precision: str | None = None,
                     audio_dir: Path | str | None = None) -> dict:
    """The engine's preprocessing of a dataset into pairs under ``out_dir``, with only the kwargs this revision
    takes; its own counts come back."""
    device_type = str(device).split(":")[0]
    kwargs = {"audio_dir": None if audio_dir is None else str(audio_dir), "output_dir": str(out_dir),
              "checkpoint_dir": str(checkpoint_root), "variant": str(variant),
              "dataset_json": None if dataset_json is None else str(dataset_json), "device": str(device),
              "precision": precision or preprocess_precision(device_type)}
    return api.preprocess(**accepted_kwargs(api.preprocess, kwargs, what="preprocess_audio_files")) or {}


def open_module(api: TrainingApi, cfg: dict, adapter_file: Path | str, *, checkpoint_root: Path | str, device: str,
                dataset_dir: Path | str, output_dir: Path | str, precision: str | None = None) -> Any:
    """A live training-side module for this recipe with the adapter's weights loaded: the object the gradient
    feature is taken on. Device and precision are this machine's, never the run's recorded ones."""
    import torch  # noqa: PLC0415
    torch_device = torch.device(device)
    precision = measure_precision(str(precision or cfg.get("mixed_precision") or PRECISION), torch_device.type)
    variant = str(cfg.get("model_variant") or DESCRIPTOR.default_variant)
    model = api.load_decoder(str(checkpoint_root), variant, device, precision)
    adapter_type, adapter_cfg = adapter_config(api, cfg)
    train_kwargs = {"adapter_type": adapter_type, "model_variant": variant, "checkpoint_dir": str(checkpoint_root),
                    "dataset_dir": str(dataset_dir), "output_dir": str(output_dir), "device": device,
                    "precision": precision, "seed": int(cfg.get("seed", 0)), "batch_size": 1}
    train_cfg = accepted_config(api.TrainingConfig, train_kwargs)
    dtype = compute_dtype(api.select_dtype, torch_device.type, precision)
    module = api.Module(model=model, adapter_config=adapter_cfg, training_config=train_cfg, device=torch_device,
                        dtype=dtype)
    net = getattr(module, "lycoris_net", None)
    if net is None or not hasattr(net, "load_weights"):
        raise TrainingUnavailable(f"adapter type {adapter_type!r} exposes no load_weights on this ACE-Step revision; "
                                  "the feature would be taken on an untrained adapter")
    net.load_weights(str(adapter_file))
    log.info("Opened %s with %s in %s", variant, Path(adapter_file).name, dtype_tag(module.dtype))
    return module
