"""Stable Audio 3's provision slot: Stability's code at the descriptor's pin, its own environment for this
machine's runtime (torch with Flash Attention on CUDA; the MLX port on Apple silicon), the weights the descriptor
lists, and the MLX base weights exported once. --check reports without changing anything. Nothing here handles a
Hugging Face token: the gated text encoder downloads only after the user's own `hf auth login`."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3.render import MLX_SUBDIR, find_root, runtime_for
from khaos_attribution.engines.sa3.train import BASE_NPZ

# Stability's CUDA 12.6 build of torch 2.7.1 cannot drive a Blackwell card; its CUDA 12.8 build can.
TORCH = "torch==2.7.1"
TORCHAUDIO = "torchaudio==2.7.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN = "flash-attn==2.8.3"
PYTHON_VERSION = "3.11"
COMPLETE_MARK = ".khaos-complete"


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def ok(self, what: str) -> None:
        self.checks.append((True, what))

    def bad(self, what: str) -> None:
        self.checks.append((False, what))

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks)

    def text(self) -> str:
        return "\n".join(f"[{'ok' if ok else 'FAIL'}] {what}" for ok, what in self.checks) + \
            ("\nCHECK OK" if self.passed else "\nCHECK FAILED")


def uv_path() -> str:
    found = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not Path(found).exists():
        raise RuntimeError("uv is not installed; the engine's environment is made with it")
    return found


def _run(cmd: list, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True)


def checkout(root: Path, *, check_only: bool = False) -> tuple:
    """Stability's code at the descriptor's commit; (ok, note)."""
    repo, commit = DESCRIPTOR.doc["code"]["repo"], DESCRIPTOR.commit
    if not (root / ".git").is_dir():
        if check_only:
            return False, f"Stable Audio 3 code not cloned at {root}"
        root.parent.mkdir(parents=True, exist_ok=True)
        done = _run(["git", "clone", "-q", repo, root])
        if done.returncode != 0:
            return False, f"clone failed: {done.stderr.strip()[-300:]}"
    head = _run(["git", "-C", root, "rev-parse", "HEAD"]).stdout.strip()
    if head == commit:
        return True, f"Stable Audio 3 code at the pin {commit[:12]}"
    if check_only:
        return False, f"Stable Audio 3 code at {head[:12]}, pin is {commit[:12]}"
    _run(["git", "-C", root, "fetch", "-q", "origin", commit])
    done = _run(["git", "-C", root, "checkout", "-q", "--detach", commit])
    return (done.returncode == 0), (f"checked out {commit[:12]}" if done.returncode == 0 else f"checkout failed: {done.stderr.strip()[-300:]}")


def environment(root: Path, runtime: str, *, check_only: bool = False) -> tuple:
    """The engine's own interpreter: (ok, note)."""
    if runtime == "mlx":
        mlx = root / MLX_SUBDIR
        python = mlx / ".venv" / "bin" / "python"
        if python.is_file():
            probe = _run([python, "-c", "import mlx.core, safetensors; print('ok')"], cwd=mlx)
            if probe.returncode == 0:
                return True, "the MLX runtime imports"
        if check_only:
            return False, f"no MLX environment at {python}"
        done = _run(["bash", "install.sh", "-y"], cwd=mlx, env={**os.environ, "UV": uv_path()})
        if done.returncode != 0:
            return False, f"the MLX installer failed: {done.stderr.strip()[-300:]}"
        _run([uv_path(), "pip", "install", "-q", "--python", python, "safetensors"])
        return python.is_file(), "the MLX runtime installed"
    python = root / ".venv" / "bin" / "python"
    if python.is_file():
        probe = _run([python, "-c", "import torch, stable_audio_3, flash_attn; assert torch.cuda.is_available(); print('ok')"])
        if probe.returncode == 0:
            return True, "torch sees the GPU, Flash Attention and stable_audio_3 import"
        if check_only:
            return False, f"the environment at {python} cannot run the engine: {probe.stderr.strip()[-200:]}"
    elif check_only:
        return False, f"no environment at {python}"
    uv = uv_path()
    work = root / "work"
    work.mkdir(exist_ok=True)
    if not python.is_file():
        _run([uv, "venv", "-q", "--python", PYTHON_VERSION, root / ".venv"])
    export = _run([uv, "export", "--frozen", "--no-hashes", "--extra", "lora", "--no-emit-project", "--no-dev"], cwd=root)
    if export.returncode != 0 or not export.stdout.strip():
        return False, f"Stability's lock could not be exported: {export.stderr.strip()[-300:]}"
    locked = "\n".join(l for l in export.stdout.splitlines()
                       if not (l.startswith(("torch==", "torchaudio==", "triton==", "nvidia-", "#", "-e")) or not l.strip()))
    (work / "stability-locked.txt").write_text(locked + "\n")
    steps = [
        [uv, "pip", "install", "-q", "--python", python, "-r", work / "stability-locked.txt", f"{TORCH}+cu128",
         f"{TORCHAUDIO}+cu128", "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"],
        [uv, "pip", "install", "-q", "--python", python, "--no-sources", "--no-deps", "-e", root],
        [uv, "pip", "install", "-q", "--python", python, "setuptools", "wheel", "ninja", "psutil", "matplotlib", "wandb"],
        [uv, "pip", "install", "--python", python, "--no-build-isolation", FLASH_ATTN],
    ]
    for cmd in steps:
        done = _run(cmd, cwd=root)
        if done.returncode != 0:
            return False, f"environment step failed ({' '.join(str(c) for c in cmd[:4])}…): {done.stderr.strip()[-300:]}"
    ok, note = contract_into(python)
    if not ok:
        return False, f"the engine's environment installed; {note}"
    probe = _run([python, "-c", "import torch, stable_audio_3, flash_attn; assert torch.cuda.is_available(); print('ok')"])
    if probe.returncode != 0:
        return False, f"the environment installed but cannot run the engine: {probe.stderr.strip()[-200:]}"
    return True, "the engine's environment installed; torch sees the GPU, Flash Attention and stable_audio_3 import"


CONTRACT_SRC_ENV = "KHAOS_CONTRACT_SRC"
CONTRACT_REPO = "https://github.com/BlurredEdgeVR/Khaos_Attribution.git"


def contract_into(python: Path) -> tuple:
    """The contract's gradient maths into the engine's environment, the version this room runs: the measure worker
    projects its features with it. From KHAOS_CONTRACT_SRC (a checkout) when set, else the tagged release."""
    from khaos_attribution import __version__  # noqa: PLC0415
    src = os.environ.get(CONTRACT_SRC_ENV, "").strip()
    spec = src if src else f"khaos-attribution @ git+{CONTRACT_REPO}@v{__version__}"
    # With its base dependencies (jsonschema, numpy): the engine's own pins are Stability's, the contract's are light.
    done = _run([uv_path(), "pip", "install", "-q", "--python", python, spec])
    if done.returncode != 0:
        return False, f"the contract could not be installed into the engine's environment: {done.stderr.strip()[-200:]}"
    return True, f"khaos-attribution {__version__} in the engine's environment"


def weights_complete(target: Path) -> bool:
    """A weight folder counts only when whole: its sentinel, or real files outside the downloader's cache with
    no partial download left behind."""
    if not target.is_dir():
        return False
    if (target / COMPLETE_MARK).is_file():
        return True
    cache = target / ".cache"
    if any(cache.rglob("*.incomplete")):
        return False
    return any(f.is_file() and cache not in f.parents for f in target.rglob("*"))


def weights(root: Path, runtime: str, weights_dir: Path, *, check_only: bool = False) -> list:
    """Every weight set the descriptor lists, under weights_dir, by the engine's own downloader; the gated text
    encoder downloads only once the user's Hugging Face login has been approved."""
    python = (root / MLX_SUBDIR if runtime == "mlx" else root) / ".venv" / "bin" / "python"
    out = []
    for w in DESCRIPTOR.weights:
        target = weights_dir / w["name"]
        have = weights_complete(target)
        if have:
            out.append((True, f"{w['repo']} present at {target}"))
            continue
        if check_only or not w.get("required", True):
            out.append((not w.get("required", True), f"{w['repo']} missing at {target}" + ("" if w.get("required", True) else " (optional)")))
            continue
        code = (f"import sys; from huggingface_hub import snapshot_download; "
                f"snapshot_download({w['repo']!r}, local_dir=sys.argv[1])")
        done = _run([python, "-c", code, target], env={**os.environ, "HF_HUB_OFFLINE": "0"})
        if done.returncode != 0:
            why = done.stderr.strip().splitlines()[-1:] or ["download failed"]
            note = (f"{w['repo']} could not be downloaded: {why[0][:200]}"
                    + ("; request access on its page, wait for approval, then `hf auth login`" if w.get("gated") else ""))
            out.append((False, note))
        else:
            (target / COMPLETE_MARK).write_text(f"{w['repo']}\n", encoding="utf-8")
            out.append((True, f"{w['repo']} downloaded to {target}"))
    if runtime == "mlx":
        npz = weights_dir / BASE_NPZ
        base = weights_dir / "stable-audio-3-medium-base"
        if npz.is_file():
            out.append((True, f"MLX base weights at {npz}"))
        elif check_only or not base.is_dir():
            out.append((False, f"MLX base weights missing at {npz}"))
        else:
            done = _run([python, root / MLX_SUBDIR / "scripts" / "export_base_npz.py", "--base-dir", base, "--output", npz],
                        cwd=root / MLX_SUBDIR)
            out.append((done.returncode == 0 and npz.is_file(),
                        f"MLX base weights exported to {npz}" if done.returncode == 0 else f"export failed: {done.stderr.strip()[-200:]}"))
    return out


def provision(repo_dir=None, weights_dir=None, *, device: str = "auto", check_only: bool = False) -> Report:
    root = find_root(repo_dir)
    runtime = runtime_for(device)
    weights_dir = Path(weights_dir) if weights_dir else root / "checkpoints"
    report = Report()
    if os.name == "nt":
        report.bad(f"{DESCRIPTOR.name} is not provisioned on Windows yet: its environment and workers are POSIX")
        return report
    ok, note = checkout(root, check_only=check_only)
    (report.ok if ok else report.bad)(note)
    if ok:
        ok, note = environment(root, runtime, check_only=check_only)
        (report.ok if ok else report.bad)(note)
    if ok:
        for good, line in weights(root, runtime, weights_dir, check_only=check_only):
            (report.ok if good else report.bad)(line)
    return report


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="Provision Stable Audio 3 for Khaos on this machine.")
    ap.add_argument("--repo-dir", type=Path, default=None)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check", action="store_true", help="report without changing anything")
    a = ap.parse_args(argv)
    report = provision(a.repo_dir, a.weights_dir, device=a.device, check_only=a.check)
    print(report.text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
