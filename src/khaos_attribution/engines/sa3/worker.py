"""One Stable Audio 3 render, run inside the engine's own environment: the request as JSON on stdin, the reply as
one JSON line last on stdout. "torch" drives Stability's package; "mlx" drives the MLX port's script in the same
checkout. Audio is written as 16-bit PCM WAV at the model's rate, so the same seed gives the same bytes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def render_torch(req: dict) -> dict:
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from stable_audio_3 import StableAudioModel  # noqa: PLC0415

    model = StableAudioModel.from_pretrained(req["model"])
    if req.get("adapter"):
        model.load_lora([str(req["adapter"])])
        model.set_lora_strength(float(req["adapter_scale"]))
    seed = int(req["seed"]) if req.get("seed") is not None else int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    kwargs = {"prompt": req["prompt"], "duration": float(req["seconds"]), "steps": int(req["steps"]),
              "cfg_scale": float(req["cfg_scale"]), "apg_scale": float(req.get("apg_scale") or 1.0), "seed": seed}
    if req.get("negative_prompt"):
        kwargs["negative_prompt"] = req["negative_prompt"]
    dropped = []
    source = req.get("source_audio")
    if source and req.get("mode") in ("audio2audio", "repaint"):
        audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(audio.T)
        if req["mode"] == "audio2audio":
            kwargs["init_audio"] = (sr, tensor)
            kwargs["init_noise_level"] = float(req.get("cover_strength") or 0.5)
        else:
            kwargs["inpaint_audio"] = (sr, tensor)
            kwargs["inpaint_mask_start_seconds"] = float(req.get("repaint_start") or 0.0)
            end = req.get("repaint_end")
            kwargs["inpaint_mask_end_seconds"] = float(end) if end is not None and float(end) >= 0 else float(req["seconds"])
    elif req.get("mode") not in (None, "text2music", "retake"):
        dropped.append(f"mode {req['mode']} without source audio")
    audio = model.generate(**kwargs)
    rate = int(model.model.sample_rate)
    out = Path(req["out"])
    tmp = out.with_name(out.name + ".tmp.wav")
    sf.write(str(tmp), audio[0].T.float().cpu().numpy(), rate, subtype="PCM_16")
    tmp.replace(out)
    return {"seed": seed, "sample_rate": rate, "dropped": dropped, "runtime": "torch"}


def render_mlx(req: dict) -> dict:
    """Stability's MLX script, as the trial drove it; the DiT is the one the variant names."""
    here = Path.cwd() / "optimized" / "mlx"
    seed = int(req["seed"]) if req.get("seed") is not None else int.from_bytes(os.urandom(4), "big") % (2 ** 31 - 1)
    dit = "medium"
    cmd = [sys.executable, str(here / "scripts" / "sa3_mlx.py"), "--prompt", req["prompt"], "--dit", dit,
           "--decoder", "same-l", "--seconds", str(float(req["seconds"])), "--seed", str(seed),
           "--steps", str(int(req["steps"])), "--cfg", str(float(req["cfg_scale"])),
           "--apg", str(float(req.get("apg_scale") or 1.0)), "--out", req["out"]]
    if req.get("negative_prompt"):
        cmd += ["--negative-prompt", req["negative_prompt"]]
    if req.get("adapter"):
        cmd += ["--lora", str(req["adapter"]), "--lora-strength", str(float(req["adapter_scale"]))]
    dropped = []
    if req.get("source_audio") and req.get("mode") == "audio2audio":
        cmd += ["--init-audio", str(req["source_audio"]), "--init-noise-level", str(float(req.get("cover_strength") or 0.5))]
    elif req.get("source_audio") and req.get("mode") == "repaint":
        end = req.get("repaint_end")
        cmd += ["--init-audio", str(req["source_audio"]), "--inpaint-range",
                f"{float(req.get('repaint_start') or 0.0)}:{float(end) if end is not None and float(end) >= 0 else float(req['seconds'])}"]
    elif req.get("mode") not in (None, "text2music", "retake"):
        dropped.append(f"mode {req['mode']} without source audio")
    done = _run_forwarding_stop(cmd, cwd=str(here))
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-8:])
        return {"error": f"sa3_mlx.py failed ({done.returncode}): {tail}"}
    rate = None
    try:
        import wave  # noqa: PLC0415
        with wave.open(req["out"], "rb") as fh:
            rate = fh.getframerate()
    except Exception:  # noqa: BLE001
        pass
    return {"seed": seed, "sample_rate": rate, "dropped": dropped, "runtime": "mlx"}


def _run_forwarding_stop(cmd: list, *, cwd: str):
    """The render as a child that dies with this worker: a stop asked of the worker is passed on, so no render
    outlives a cancel."""
    import signal  # noqa: PLC0415
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    def _forward(signum, _frame):
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _forward)
    out, err = proc.communicate()
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = render_mlx(req) if req.get("runtime") == "mlx" else render_torch(req)
    except Exception as exc:  # noqa: BLE001 — the reply carries it; the caller raises RenderFailed
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
