"""The fingerprint index beside the register: outputs released under their model's bucket by the owning Workshop or a
serving Space, and verify in three grades on real audio through the transforms the bar passed."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
sf = pytest.importorskip("soundfile")
from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.fingerprint import fingerprint_array, query_phases  # noqa: E402
from khaos_attribution.watermark import derive_payload, encode_payload  # noqa: E402
from register_service.app import create_app  # noqa: E402
from tests.test_register_service import _Transport, _machine, _signed  # noqa: E402


@pytest.fixture
def register(tmp_path, monkeypatch):
    cfg = {"data": tmp_path / "data", "key_file": tmp_path / "data" / "k.pem", "dev_key": True, "public_url": "https://register.example"}
    app = create_app(cfg)
    app.state.store.create_account("artist-one")
    app.state.store.create_account("artist-two")
    http = TestClient(app)

    def public(self, path, body):
        r = http.post(path, content=R.canonical_bytes(body), headers={"Content-Type": "application/json"})
        if r.status_code >= 400:
            raise R.RegisterError(r.status_code, r.text)
        return r.json()
    monkeypatch.setattr(R.RegisterClient, "_post_public", public)
    return app, app.state.store, http


def _music(seed: int, seconds: float = 12.0, sr: int = 22050):
    """A deterministic piece with real spectral structure: a few notes with harmonics and a beat, not a sine."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    out = np.zeros_like(t)
    notes = rng.uniform(110, 880, size=8)
    for i, f in enumerate(notes):
        start, end = i * seconds / 8, (i + 1) * seconds / 8
        env = ((t >= start) & (t < end)).astype(float)
        for k, a in ((1, 1.0), (2, 0.5), (3, 0.25), (5, 0.1)):
            out += a * env * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28))
    beat = (np.mod(t, 0.5) < 0.02).astype(float) * rng.normal(0, 1, size=t.shape)
    out = out / np.max(np.abs(out)) * 0.8 + 0.1 * beat
    return out.astype(np.float32), sr


def _clip(audio, sr, start_s, seconds):
    a, b = int(start_s * sr), int((start_s + seconds) * sr)
    return audio[a:b]


def test_release_is_by_the_owning_workshop_or_a_serving_space_and_verify_names_the_output(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, stranger = _machine(register, "artist-two")
    _, space = _machine(register, "artist-two", role="space")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    audio, sr = _music(1)
    fps = fingerprint_array(audio, sr)
    with pytest.raises(R.RegisterError, match="own account"):
        stranger.release("gen-1", serial, fps)
    with pytest.raises(R.RegisterError, match="reports serving"):
        space.release("gen-1", serial, fps)
    out = client.release("gen-1", serial, fps)
    assert out["serial"] == serial and out["landmarks"] == len(fps) and out["already"] is False
    assert client.release("gen-1", serial, fps)["already"] is True
    space.served(serial, "https://space.example", "serving")
    assert space.release("gen-2", serial, fingerprint_array(*_music(2)))["already"] is False
    assert http.get("/health").json()["outputs"] == 2
    # verify: a clip off the grid, re-encoded, with the watermark read → the exact output
    codeword = encode_payload(signed["record"]["model"]["watermark_payload"])
    clip = _clip(audio, sr, 3.0, 5.0)
    answer = client.verify(query_phases(clip, sr), codeword)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-1" and answer["searched"] == "bucket"
    assert answer["model"]["serial"] == serial and answer["model"]["artist_name"] == "Artist X"
    # no watermark: the whole index, same answer
    answer = client.verify(query_phases(clip, sr), None)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-1" and answer["searched"] == "all"
    # the watermark read but the audio is not in the index: the bucket's models
    answer = client.verify(query_phases(_music(9)[0], sr), codeword)
    assert answer["grade"] == "bucket" and [m["serial"] for m in answer["models"]] == [serial]
    # nothing read, nothing matched
    assert client.verify(query_phases(_music(9)[0], sr), None)["grade"] == "none"
    with pytest.raises(R.RegisterError, match="codeword"):
        client.verify(query_phases(clip, sr), "not-a-codeword")


def test_verify_survives_mp3_when_ffmpeg_is_here(register, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    audio, sr = _music(3, seconds=20)
    client.release("gen-3", serial, fingerprint_array(audio, sr))
    wav, mp3, back = tmp_path / "a.wav", tmp_path / "a.mp3", tmp_path / "b.wav"
    sf.write(str(wav), audio, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "4.37", "-t", "8", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), str(back)], check=True)
    q, qsr = sf.read(str(back), dtype="float32", always_2d=True)
    answer = client.verify(query_phases(q[:, 0], qsr), None)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-3"


def test_a_withdrawn_models_outputs_are_not_released(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "x")
    with pytest.raises(R.RegisterError, match="withdrawn"):
        client.release("gen-w", serial, fingerprint_array(*_music(4)))
