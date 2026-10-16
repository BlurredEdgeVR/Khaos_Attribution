"""Stable Audio 3's train slot behind a fake trainer process: epochs become steps at the pair count, the losses are
read from the trainer's own logs, each checkpoint lands in the Workshop's layout, and the session's updates are the
same kinds ACE-Step's are."""
import json
import struct
import sys
import wave

import pytest

from khaos_attribution.engines.sa3 import train as T
from khaos_attribution.engines.train import TrainingUnavailable

FAKE_TRAINER = '''
import csv, os, sys, time
from pathlib import Path
args = sys.argv[1:]
def arg(name, default=None):
    return args[args.index(name) + 1] if name in args else default
save_dir = Path(arg("--save_dir") or arg("--save-dir"))
steps = int(arg("--steps") or arg("--max-steps"))
every = int(arg("--checkpoint_every") or arg("--checkpoint-every"))
runtime = "mlx" if "--max-steps" in args else "torch"
if runtime == "torch":
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "khaos_losses.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "loss"]); w.writeheader()
        for s in range(steps):
            w.writerow({"step": s + 1, "loss": f"{1.0 / (s + 1):.4f}"}); fh.flush()
            if (s + 1) % every == 0:
                (save_dir / f"epoch=0-step={s + 1}.ckpt").write_bytes(b"ckpt")
else:
    import struct
    with open("loss_by_timestep.bin", "ab") as fh:
        for s in range(1, steps + 1):
            fh.write(struct.pack("Iff", s, 0.5, 1.0 / s)); fh.flush()
            if s % every == 0:
                d = save_dir / "khaos" / "s1" / "checkpoints"; d.mkdir(parents=True, exist_ok=True)
                (d / f"step={s}-epoch=0.safetensors").write_bytes(b"st")
'''

FAKE_CONVERT = '''
import sys
from pathlib import Path
Path(sys.argv[2]).write_bytes(b"converted:" + Path(sys.argv[1]).read_bytes())
'''

RECIPE = {"adapter_type": "lokr", "adapter_rank": 16, "adapter_alpha": 16, "max_epochs": 2, "batch_size": 1,
          "checkpoint_every_n_epochs": 1, "learning_rate": 1e-4, "window_sec": 10.0}


def _clips(d, n):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        with wave.open(str(d / f"trk{i % 2}_seg_{i:03d}.wav"), "wb") as fh:
            fh.setnchannels(2); fh.setsampwidth(2); fh.setframerate(44100); fh.writeframes(b"\0\0\0\0" * 4410)
        (d / f"trk{i % 2}_seg_{i:03d}.txt").write_text("a caption\n")


@pytest.fixture
def api(tmp_path, monkeypatch):
    root = tmp_path / "stable-audio-3"
    (root / "scripts").mkdir(parents=True)
    driver = tmp_path / "driver.py"
    driver.write_text(FAKE_TRAINER)
    monkeypatch.setattr(T, "TRAIN_DRIVER", driver)
    (root / "optimized" / "mlx" / "scripts").mkdir(parents=True)
    (root / "optimized" / "mlx" / "scripts" / "lora_train_mlx.py").write_text(FAKE_TRAINER)
    convert = tmp_path / "convert.py"
    convert.write_text(FAKE_CONVERT)
    monkeypatch.setattr(T, "CONVERT_WORKER", convert)
    return T.TrainingApi(root=root, runtime="torch", python=sys.executable)


def test_epochs_become_steps_at_the_pair_count():
    s = T.steps_for({"max_epochs": 8, "batch_size": 1, "checkpoint_every_n_epochs": 2}, 316)
    assert s == {"per_epoch": 316, "total": 2528, "checkpoint_every": 632, "epochs": 8}
    assert T.steps_for({"max_epochs": 1, "batch_size": 4}, 3)["per_epoch"] == 1, "never zero steps an epoch"
    kind, a = T.adapter_kwargs(RECIPE)
    assert kind == "lora" and a == {"rank": 16, "alpha": 16, "dropout": 0.0}, "a LoKR recipe maps to the engine's own adapter type"
    assert T.adapter_kwargs({**RECIPE, "adapter_type": "dora-rows"})[0] == "dora-rows", "Stability's other types pass through"


def test_the_trainer_is_driven_as_a_process_and_its_run_becomes_khaos_updates(api, tmp_path):
    tensors = tmp_path / "run" / "tensors"
    _clips(tensors, 3)
    session = T.open_session(api, RECIPE, checkpoint_root=tmp_path / "ck", tensor_dir=tensors,
                             output_dir=tmp_path / "run", device="cuda", precision="bf16", seed=4)
    assert session.steps == {"per_epoch": 3, "total": 6, "checkpoint_every": 3, "epochs": 2}
    cmd = session.command()
    assert "--data_dir" in cmd and cmd[cmd.index("--adapter_type") + 1] == "lora" and "--duration" in cmd
    assert cmd[1].endswith("driver.py")
    updates = list(session.train())
    kinds = [u.kind for u in updates]
    assert kinds[0] == "info" and kinds[-1] == "complete"
    epochs = [u for u in updates if u.kind == "epoch"]
    assert [u.epoch for u in epochs] == [1, 2] and epochs[0].loss == pytest.approx((1 + 0.5 + 1 / 3) / 3, abs=1e-3)
    checkpoints = [u for u in updates if u.kind == "checkpoint"]
    assert [u.epoch for u in checkpoints] == [1, 2]
    laid = sorted((tmp_path / "run" / "checkpoints").glob("epoch_*/adapter_model.safetensors"))
    assert len(laid) == 2 and laid[0].read_bytes().startswith(b"converted:")
    assert session.save_final(tmp_path / "run" / "final").name == "adapter_model.safetensors"
    assert (tmp_path / "run" / "sa3" / "train_command.json").is_file()
    with session.install_hooks(record_into=(rec := {})):
        pass
    assert rec == {} and session.module is None


def test_the_mlx_runtime_reads_the_telemetry_and_needs_the_base_npz(api, tmp_path):
    mlx = T.TrainingApi(root=api.root, runtime="mlx", python=sys.executable)
    tensors = tmp_path / "run" / "tensors"
    _clips(tensors, 2)
    with pytest.raises(TrainingUnavailable, match="medium-base.npz"):
        T.open_session(mlx, RECIPE, checkpoint_root=tmp_path / "ck", tensor_dir=tensors, output_dir=tmp_path / "run",
                       device="mps", precision="bf16", seed=1)
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "medium-base.npz").write_bytes(b"npz")
    session = T.open_session(mlx, RECIPE, checkpoint_root=tmp_path / "ck", tensor_dir=tensors, output_dir=tmp_path / "run",
                             device="mps", precision="bf16", seed=1)
    cmd = session.command()
    assert "--latents-dir" in cmd and "--dit-weights" in cmd and "--grad-checkpoint" in cmd
    config = json.loads((tmp_path / "run" / "sa3" / "prompt_config.json").read_text())
    assert config["tag_keys"] == ["prompt"] and config["hide_tag_names"] is True and config["shuffle"] is False, \
        "the caption reaches the MLX trainer verbatim"
    assert cmd[cmd.index("--prompt-config") + 1].endswith("prompt_config.json")
    updates = list(session.train())
    assert [u.kind for u in updates if u.kind in ("epoch", "checkpoint")] == ["epoch", "checkpoint", "epoch", "checkpoint"]
    laid = sorted((tmp_path / "run" / "checkpoints").glob("epoch_*/adapter_model.safetensors"))
    assert len(laid) == 2 and laid[0].read_bytes() == b"st", "a safetensors checkpoint is copied, not converted"


def test_a_trainer_that_dies_is_a_fail_update_with_its_last_lines(api, tmp_path):
    T.TRAIN_DRIVER.write_text("import sys; print('boom'); sys.exit(3)")
    tensors = tmp_path / "run" / "tensors"
    _clips(tensors, 1)
    session = T.open_session(api, RECIPE, checkpoint_root=tmp_path, tensor_dir=tensors, output_dir=tmp_path / "run",
                             device="cuda", precision="bf16", seed=1)
    updates = list(session.train())
    assert updates[-1].kind == "fail" and "exited 3" in updates[-1].msg and "boom" in updates[-1].msg
    with pytest.raises(RuntimeError, match="no checkpoint"):
        session.save_final(tmp_path / "final")


def test_losses_are_read_from_either_trainers_log(tmp_path):
    (tmp_path / "khaos_losses.csv").write_text("step,loss\n1,0.5\n2,\n3,0.25\n")
    assert T.losses_from_metrics_csv(tmp_path) == [(1, 0.5), (3, 0.25)]
    assert T.losses_from_metrics_csv(tmp_path / "absent") == []
    tele = tmp_path / "loss_by_timestep.bin"
    tele.write_bytes(struct.pack("Iff", 1, 0.5, 0.75) + struct.pack("Iff", 2, 0.4, 0.5) + b"\x00")
    assert T.losses_from_telemetry(tele) == [(1, 0.75), (2, 0.5)]
    assert T.losses_from_telemetry(tmp_path / "absent.bin") == []
    assert T.checkpoint_step(tmp_path / "epoch=0-step=316.ckpt") == 316 and T.checkpoint_step(tmp_path / "x.ckpt") is None


def test_the_heldout_record_is_summarised_as_acestep_summarises_it():
    ts = [0.25, 0.75, 0.95]
    per = {"a_seg_000.wav": {"0.25": 1.0, "0.75": 2.0, "0.95": 4.0}, "a_seg_001.wav": {"0.25": 1.0, "0.75": 2.0, "0.95": 2.0},
           "b_seg_000.wav": {"0.25": 3.0, "0.75": 3.0, "0.95": 3.0}}
    out = T.summarise(per, ts)
    assert out["n_pairs"] == 3 and out["buckets"]["0.95"] == pytest.approx(3.0) and out["high_noise"] == pytest.approx(3.0)
    assert out["by_track"]["a"]["n_pairs"] == 2 and out["by_track"]["b"]["mean"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        T.summarise({}, ts)


def test_clips_are_written_from_khaos_dataset_json_and_lyrics_never_travel(tmp_path):
    src = tmp_path / "src.wav"
    with wave.open(str(src), "wb") as fh:
        fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(48000); fh.writeframes(b"\0\0" * 4800)
    ds = tmp_path / "dataset.json"
    ds.write_text(json.dumps([{"audio_path": str(src), "filename": "t_seg_000.wav", "caption": "bright", "lyrics": "la"},
                              {"audio_path": str(tmp_path / "missing.wav"), "filename": "t_seg_001.wav", "caption": "x"}]))
    pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    out = T.write_clips(ds, tmp_path / "out")
    assert out == {"processed": 1, "failed": 1}
    with wave.open(str(tmp_path / "out" / "t_seg_000.wav"), "rb") as fh:
        assert fh.getframerate() == 44100 and fh.getnchannels() == 2
    assert (tmp_path / "out" / "t_seg_000.txt").read_text() == "bright\n"
    assert T.pair_files(tmp_path / "out") == [tmp_path / "out" / "t_seg_000.wav"]
