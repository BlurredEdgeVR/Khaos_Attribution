"""The held-out loss on Stable Audio 3, inside the engine's environment: the request as JSON on stdin (clips or
latents, the adapter, the timesteps, the seed), the reply as one JSON line of per-sample bucket losses. The loss is
the trainer's own rectified-flow velocity MSE with the randomness fixed: t a bucket centre, noise seeded per clip,
no CFG dropout, no inpainting, the padding mask as the trainer masks it."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path


def pair_noise_seed(seed: int, name: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(name.encode("utf-8"))) % (2 ** 31 - 1)


def heldout_torch(req: dict) -> dict:
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from stable_audio_3 import StableAudioModel  # noqa: PLC0415

    wrapped = StableAudioModel.from_pretrained(req["model"])
    if req.get("adapter"):
        wrapped.load_lora([str(req["adapter"])])
        wrapped.set_lora_strength(1.0)
    model = wrapped.model
    model.eval()
    device = wrapped.device
    sr = int(model.sample_rate)
    ratio = int(model.pretransform.downsampling_ratio)
    timesteps = [float(t) for t in req["timesteps"]]
    per_sample = {}
    clips = sorted(Path(req["clips_dir"]).glob("*.wav"))
    with torch.no_grad():
        for wav in clips:
            audio, file_sr = sf.read(str(wav), dtype="float32", always_2d=True)
            if file_sr != sr:
                raise RuntimeError(f"{wav.name} is at {file_sr} Hz, the model wants {sr}")
            x = torch.from_numpy(audio.T).unsqueeze(0).to(device)
            n = x.shape[-1]
            padded = ((n + ratio - 1) // ratio) * ratio
            if padded != n:
                x = torch.nn.functional.pad(x, (0, padded - n))
            caption = wav.with_suffix(".txt").read_text(encoding="utf-8").strip() if wav.with_suffix(".txt").exists() else ""
            seconds = n / sr
            with torch.autocast(device_type=device if isinstance(device, str) else device.type, enabled=str(device).startswith("cuda")):
                latents = model.pretransform.encode(x)
            latents = latents.float()
            cond = model.conditioner([{"prompt": caption, "seconds_total": seconds}], device)
            # The trainer always conditions on an inpaint mask (0 = generate, 1 = keep); a held-out clip is
            # generated whole, so the mask is all zeros and the masked input empty.
            cond["inpaint_mask"] = [torch.zeros(1, 1, latents.shape[-1], dtype=torch.bool, device=latents.device)]
            cond["inpaint_masked_input"] = [torch.zeros_like(latents)]
            valid = int(round(n / ratio))
            mask = torch.zeros(1, latents.shape[-1], dtype=torch.bool, device=latents.device)
            mask[:, :max(1, min(valid, latents.shape[-1]))] = True
            gen = torch.Generator(device="cpu").manual_seed(pair_noise_seed(int(req["seed"]), wav.name))
            noise = torch.randn(latents.shape, generator=gen, dtype=torch.float32).to(latents.device)
            target = noise - latents
            row = {}
            for t in timesteps:
                tt = torch.full((1,), t, device=latents.device, dtype=torch.float32)
                noised = latents * (1.0 - t) + noise * t
                out = model(noised.to(model.dtype if hasattr(model, "dtype") else noised.dtype), tt, cond=cond,
                            cfg_dropout_prob=0.0, padding_mask=mask).float()
                mse = ((out - target) ** 2).mean(dim=1)
                row[f"{t:.2f}"] = float(mse[mask].mean())
            per_sample[wav.name] = row
    return {"per_sample": per_sample, "runtime": "torch"}


def heldout_mlx(req: dict) -> dict:
    """Not built yet: the MLX trainer keeps its model, conditioning and adapter injection inside its main(), so a
    held-out loss on Apple silicon needs a loader written against those internals, on a Mac with the MLX runtime."""
    raise NotImplementedError("the held-out loss on the MLX runtime is not built yet; a run on Apple silicon records "
                              "no held-out loss and its best epoch is the last")


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = heldout_mlx(req) if req.get("runtime") == "mlx" else heldout_torch(req)
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
