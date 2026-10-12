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
    from khaos_attribution.fingerprint import query_phases_file
    assert client.verify(query_phases_file(back), None)["output"]["generation_id"] == "gen-3"


def test_a_withdrawn_models_outputs_are_not_released(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "x")
    with pytest.raises(R.RegisterError, match="withdrawn") as e:
        client.release("gen-w", serial, fingerprint_array(*_music(4)))
    assert e.value.status == 410, "a refusal of the entry for what it is: set aside by sync, never a jam"
    box = R.Outbox(store.path.parent / "box")
    box.append("release", {"generation_id": "gen-w2", "serial": serial, "fingerprint": fingerprint_array(*_music(5)), "nonce": "a1b2c3d4e5f60718"})
    assert client.sync(box) == [] and len(box.refused()) == 1 and box.pending() == []


def _tracks(*ids):
    return {"rights_sha256": "f" * 64, "tracks": [{"track_id": t, "title": t, "duration_sec": 20.0, "writers": []} for t in ids]}


def _fp(audio, sr):
    return [list(l) for l in fingerprint_array(audio, sr)]


def _ph(audio, sr):
    return [[list(l) for l in p] for p in query_phases(audio, sr)]


def test_a_training_track_is_claimed_once_and_another_accounts_claim_on_it_is_refused(register, tmp_path):
    """The first account to register a model on a recording holds it; another account's claim is refused, counted and
    set aside by the outbox; the same account may train on it again; a claim names one of the record's own tracks;
    withdrawing the model frees its recordings; only a Workshop claims or checks."""
    app, store, http = register
    key, one = _machine(register, "artist-one")
    key2, two = _machine(register, "artist-two")
    _, space = _machine(register, "artist-two", role="space")
    first = _signed(key)
    serial = one.publish(first)["serial"]
    audio, sr = _music(1, seconds=20.0)
    fps = _fp(audio, sr)
    assert one.check_claim(_ph(audio, sr))["grade"] == "none"
    with pytest.raises(R.RegisterError, match="training tracks the model's own record lists"):
        one.claim(first["record_sha256"], "not-in-the-record", fps, "khaos.declaration/1")
    out = one.claim(first["record_sha256"], "t1", fps, "khaos.declaration/1", phases=_ph(audio, sr))
    assert out["claimed"] is True and out["already"] is False and out["serial"] == serial
    assert one.claim(first["record_sha256"], "t1", fps, "khaos.declaration/1")["already"] is True
    mine = one.check_claim(_ph(audio, sr))
    assert mine["grade"] == "mine" and mine["serial"] == serial and mine["track_id"] == "t1" and mine["released_output"] is None
    theirs = two.check_claim(_ph(_clip(audio, sr, 2.0, 15.0), sr))
    assert theirs["grade"] == "other" and "serial" not in theirs and "track_id" not in theirs
    with pytest.raises(R.RegisterError, match="only a Workshop"):
        space.check_claim(_ph(audio, sr))
    second = _signed(key2, training_set=_tracks("x1", "x2", "x3", "x4"))
    serial2 = two.publish(second)["serial"]
    with pytest.raises(R.RegisterError) as refused:
        two.claim(second["record_sha256"], "x1", fps, "khaos.declaration/1")
    assert refused.value.status == 403 and "another account" in str(refused.value)
    assert two.claim(second["record_sha256"], "x2", _fp(*_music(5, 20.0)), "khaos.declaration/1")["claimed"]
    assert http.get(f"/api/models/{serial2}/claims").json() == {"serial": serial2, "tracks": 4, "claimed": 1, "refused": 1}
    # A long piece holding the claimed track is caught either way round.
    medley = np.concatenate([_music(7, 60.0)[0], audio, _music(8, 60.0)[0]])
    assert two.check_claim(_ph(medley, sr))["grade"] == "other"
    box = R.Outbox(tmp_path / "box")
    box.append("claim", {"record_sha256": second["record_sha256"], "track_id": "x3", "fingerprint": fps,
                         "declaration": "khaos.declaration/1", "nonce": "claim-nonce-1"})
    box.append("claim", {"record_sha256": second["record_sha256"], "track_id": "x4",
                         "fingerprint": _fp(*_music(6, 20.0)), "declaration": "khaos.declaration/1", "nonce": "claim-nonce-2"})
    done = two.sync(box)
    assert len(done) == 1 and len(box.refused()) == 1 and "another account" in json.dumps(box.refused())
    with pytest.raises(R.RegisterError, match="own account"):
        two.claim(first["record_sha256"], "t1", fps, "khaos.declaration/1")
    # Withdrawn, the first model's claims end with it, and the recording is free to claim.
    one.withdraw(serial, "consent withdrawn")
    assert two.check_claim(_ph(audio, sr))["grade"] == "none"
    assert two.claim(second["record_sha256"], "x1", fps, "khaos.declaration/1")["claimed"] is True
