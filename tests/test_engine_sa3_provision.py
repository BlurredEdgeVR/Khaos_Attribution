"""Stable Audio 3's provision slot behind fakes: --check reports what is missing without changing anything, the
pin is compared not assumed, every listed weight is accounted for, and the report says CHECK OK only when all is."""
import subprocess
from pathlib import Path

import pytest

from khaos_attribution.engines.sa3 import provision as P


def _done(code=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr=err)


def test_check_reports_a_missing_checkout_and_changes_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(P, "_run", lambda cmd, cwd=None, env=None: calls.append(cmd) or _done())
    report = P.provision(tmp_path / "sa3", tmp_path / "w", device="cuda", check_only=True)
    assert not report.passed and "not cloned" in report.text() and "CHECK FAILED" in report.text()
    assert calls == [] and not (tmp_path / "sa3").exists()


def test_check_compares_the_pin_and_the_environment_and_each_weight(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    pin = P.DESCRIPTOR.commit

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        if "rev-parse" in cmd:
            return _done(out=pin + "\n")
        if "-c" in cmd and "import torch" in cmd[-1]:
            return _done(0, "ok")
        return _done()
    monkeypatch.setattr(P, "_run", fake_run)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    weights = tmp_path / "w"
    (weights / "stable-audio-3-medium-base").mkdir(parents=True)
    (weights / "stable-audio-3-medium-base" / "model.safetensors").write_bytes(b"w")
    report = P.provision(root, weights, device="cuda", check_only=True)
    text = report.text()
    assert "at the pin" in text and "Flash Attention" in text
    assert "google/t5gemma-b-b-ul2 missing" in text and "stable-audio-3-medium missing" in text and "(optional)" in text
    assert not report.passed, "the gated text encoder is required and absent"
    (weights / "t5gemma-b-b-ul2").mkdir()
    (weights / "t5gemma-b-b-ul2" / "config.json").write_text("{}")
    assert P.provision(root, weights, device="cuda", check_only=True).passed


def test_a_moved_pin_is_named_not_moved_under_check(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(P, "_run", lambda cmd, cwd=None, env=None: _done(out="deadbeef" * 5 + "\n") if "rev-parse" in [str(c) for c in cmd] else _done())
    ok, note = P.checkout(root, check_only=True)
    assert not ok and "pin is" in note


def test_the_mlx_runtime_wants_the_exported_base_weights(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    mlx_python = root / "optimized" / "mlx" / ".venv" / "bin" / "python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.write_text("")
    pin = P.DESCRIPTOR.commit
    monkeypatch.setattr(P, "_run", lambda cmd, cwd=None, env=None: _done(out=pin + "\n") if "rev-parse" in [str(c) for c in cmd] else _done(0, "ok"))
    weights = tmp_path / "w"
    for name in ("stable-audio-3-medium-base", "stable-audio-3-medium", "t5gemma-b-b-ul2"):
        (weights / name).mkdir(parents=True)
        (weights / name / "x").write_bytes(b"w")
    report = P.provision(root, weights, device="mps", check_only=True)
    assert "MLX base weights missing" in report.text() and not report.passed
    (weights / P.BASE_NPZ).write_bytes(b"npz")
    assert P.provision(root, weights, device="mps", check_only=True).passed
    with pytest.raises(SystemExit):
        P.main(["--help"])
