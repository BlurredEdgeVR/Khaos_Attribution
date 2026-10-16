"""Content Credentials: one identity per installation, a signed manifest that says AI-generated and carries Khaos's
record (never the prompt), and a reader that sees tampering."""

import math
import os
import stat
import struct
import wave

import pytest

from khaos_attribution import credentials as C

pytestmark = pytest.mark.skipif(not C.available(), reason="the credentials extra is not installed")
RECORD = {"artist_id": "art", "artist_name": "Art", "adapter_version": "run_x", "adapter_hash": "abc",
          "base_model": "ACE-Step 1.5 (turbo)", "base_model_licence": "MIT", "generation_id": "job/a.wav",
          "watermark_id": 59368, "prompt": "never inside the file", "timestamp": "2026-10-15T10:00:00"}


def _wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / 16000)))
                               for i in range(int(16000 * seconds))))
    return path


def test_one_identity_per_installation_kept_private(tmp_path):
    a = C.load_identity(tmp_path / "id", product="Khaos Workshop", subject_id="m1")
    b = C.load_identity(tmp_path / "id", product="Khaos Workshop", subject_id="m1")
    assert a.signer_id == b.signer_id and a.chain_pem == b.chain_pem
    for name in ("issuer.key", "signer.key"):
        assert stat.S_IMODE(os.stat(tmp_path / "id" / name).st_mode) == 0o600
    assert "BEGIN CERTIFICATE" in a.chain_pem and a.chain_pem.count("BEGIN CERTIFICATE") == 2


def test_a_signed_output_says_ai_generated_and_carries_the_record_but_not_the_prompt(tmp_path):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", subject_id="m1")
    wav = _wav(tmp_path / "a.wav")
    out = C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    assert out == {"signed": True, "timestamped": False, "signer": ident.signer_id}
    got = C.read_file(wav)
    assert got["present"] and got["state"] in ("Valid", "Trusted") and got["ai_generated"]
    assert got["khaos"]["ai_generated"] is True and got["khaos"]["watermark_id"] == 59368
    assert got["khaos"]["base_model_licence"] == "MIT" and "prompt" not in got["khaos"]
    assert b"never inside the file" not in wav.read_bytes()


def test_tampered_audio_breaks_the_signature(tmp_path):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", subject_id="m1")
    wav = _wav(tmp_path / "a.wav")
    C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF   # inside the audio samples, after the RIFF header
    wav.write_bytes(bytes(data))
    assert C.read_file(wav)["state"] not in ("Valid", "Trusted")


def test_a_file_without_credentials_says_so(tmp_path):
    got = C.read_file(_wav(tmp_path / "plain.wav"))
    assert got["present"] is False and got["ai_generated"] is False


def test_the_manifest_names_the_trained_model_source_type():
    m = C.manifest(RECORD, product="Khaos Workshop", version="0.1.0", title="a.wav", mime="audio/wav")
    action = m["assertions"][0]["data"]["actions"][0]
    assert action["action"] == "c2pa.created" and action["digitalSourceType"] == C.DIGITAL_SOURCE_TYPE
    assert set(m["assertions"][1]["data"]) <= set(C.RECORD_FIELDS) | {"ai_generated", "assertion_version"}


def test_concurrent_first_use_makes_one_coherent_identity(tmp_path):
    """Two makers at once must leave one issuer whose signer it issued, never a crossed chain."""
    from concurrent.futures import ThreadPoolExecutor
    from cryptography import x509
    for trial in range(5):
        d = tmp_path / f"id{trial}"
        with ThreadPoolExecutor(4) as pool:
            ids = list(pool.map(lambda _: C.load_identity(d, product="Khaos Workshop", subject_id="m1"), range(4)))
        assert len({i.signer_id for i in ids}) == 1
        x509.load_pem_x509_certificate((d / "signer.pem").read_bytes()).verify_directly_issued_by(
            x509.load_pem_x509_certificate((d / "issuer.pem").read_bytes()))


def test_a_signer_that_is_not_the_issuers_is_remade(tmp_path):
    a = C.load_identity(tmp_path / "a", product="Khaos Workshop", subject_id="m1")
    C.load_identity(tmp_path / "b", product="Khaos Workshop", subject_id="m1")
    for name in ("signer.pem", "signer.key"):
        (tmp_path / "a" / name).write_bytes((tmp_path / "b" / name).read_bytes())
    again = C.load_identity(tmp_path / "a", product="Khaos Workshop", subject_id="m1")
    assert again.issuer_pem == a.issuer_pem and again.signer_id != a.signer_id
    from cryptography import x509
    signer, issuer = x509.load_pem_x509_certificates(again.chain_pem.encode())
    signer.verify_directly_issued_by(issuer)
    wav = _wav(tmp_path / "x.wav")
    C.sign_file(wav, RECORD, again, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    assert C.is_signed(wav)


def test_an_unreachable_timestamp_service_does_not_hold_the_signature(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(C, "TSA_PROBE_S", 0.5)
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", subject_id="m1")
    t = time.monotonic()
    for n in range(3):
        out = C.sign_file(_wav(tmp_path / f"{n}.wav"), RECORD, ident, product="Khaos Workshop", version="0.1.0",
                          timestamp_url="http://10.255.255.1:9")
        assert out["signed"] and not out["timestamped"]
    assert time.monotonic() - t < 5


def test_is_signed_is_false_for_a_broken_signature(tmp_path):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", subject_id="m1")
    wav = _wav(tmp_path / "a.wav")
    C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    assert C.is_signed(wav)
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF
    wav.write_bytes(bytes(data))
    assert not C.is_signed(wav) and not C.is_signed(_wav(tmp_path / "plain.wav"))


def test_a_signed_file_keeps_its_permissions_and_off_is_read_in_any_case(tmp_path, monkeypatch):
    import os
    import stat
    import wave
    from khaos_attribution import credentials as C
    wav = tmp_path / "p.wav"
    with wave.open(str(wav), "wb") as fh:
        fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(8000); fh.writeframes(b"\x00\x00" * 8000)
    os.chmod(wav, 0o644)
    ident = C.load_identity(tmp_path / "id", product="Khaos Test", subject_id="t")
    C.sign_file(wav, {"generation_id": "g", "watermark_id": 59368}, ident, product="Khaos Test", version="0", timestamp_url=None)
    assert stat.S_IMODE(wav.stat().st_mode) == 0o644
    for value in ("off", "OFF", "none", ""):
        monkeypatch.setenv("KHAOS_TSA_URL", value)
        assert C.tsa_url() is None, value
