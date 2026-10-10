"""The release bundle: written once per content, read only whole and only where every hash matches."""
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from khaos_attribution import release as B  # noqa: E402


def _files(tmp_path):
    src = tmp_path / "src"
    (src / "loras" / "run_1").mkdir(parents=True)
    (src / "loras" / "run_1" / "model_card.json").write_text('{"a": 1}')
    (src / "loras" / "run_1" / "adapter.safetensors").write_bytes(b"\x00" * 5000)
    (src / "rights.json").write_text("{}")
    return {"artist/loras/run_1/model_card.json": src / "loras" / "run_1" / "model_card.json",
            "artist/loras/run_1/adapter.safetensors": src / "loras" / "run_1" / "adapter.safetensors",
            "artist/rights.json": src / "rights.json"}


META = {"artist_id": "ava", "run_id": "run_1", "checkpoint": "best", "serial": 65536, "record_sha256": "r" * 64}


def test_a_bundle_is_content_addressed_and_reads_back_whole(tmp_path):
    files = _files(tmp_path)
    m1 = B.write_bundle(tmp_path / "a.tar", files, META)
    m2 = B.write_bundle(tmp_path / "b.tar", files, {**META, "made_at_utc": "later"})
    assert m1["bundle_id"] == m2["bundle_id"], "when it was made is not part of the address"
    assert B.read_manifest(tmp_path / "a.tar")["bundle_id"] == m1["bundle_id"]
    out = tmp_path / "out"
    manifest = B.read_bundle(tmp_path / "a.tar", out)
    assert manifest["serial"] == 65536 and (out / "artist" / "loras" / "run_1" / "adapter.safetensors").stat().st_size == 5000
    assert json.loads((out / B.MANIFEST_NAME).read_text())["bundle_id"] == m1["bundle_id"]
    with pytest.raises(B.BundleError, match="exists"):
        B.read_bundle(tmp_path / "a.tar", out)
    other = B.write_bundle(tmp_path / "c.tar", files, {**META, "checkpoint": "final"})
    assert other["bundle_id"] != m1["bundle_id"]


def test_a_tampered_or_hostile_bundle_leaves_nothing_behind(tmp_path):
    files = _files(tmp_path)
    B.write_bundle(tmp_path / "a.tar", files, META)

    def rewrite(name, mutate):
        src, dst = tmp_path / "a.tar", tmp_path / name
        with tarfile.open(src) as tin, tarfile.open(dst, "w") as tout:
            for member in tin:
                data = tin.extractfile(member).read() if member.isfile() else None
                member, data = mutate(member, data)
                if member is not None:
                    tout.addfile(member, __import__("io").BytesIO(data) if data is not None else None)
        return dst

    def flip(member, data):
        if member.name.endswith("adapter.safetensors"):
            data = b"\x01" + data[1:]
        return member, data
    bad = rewrite("flip.tar", flip)
    with pytest.raises(B.BundleError, match="does not match its hash"):
        B.read_bundle(bad, tmp_path / "o1")
    assert not (tmp_path / "o1").exists()

    def extra(member, data):
        return member, data
    extra_tar = rewrite("extra.tar", extra)
    with tarfile.open(extra_tar, "a") as t:
        info = tarfile.TarInfo("artist/../evil.txt")
        info.size = 2
        t.addfile(info, __import__("io").BytesIO(b"hi"))
    with pytest.raises(B.BundleError, match="not a bundle path"):
        B.read_bundle(extra_tar, tmp_path / "o2")
    assert not (tmp_path / "o2").exists()

    def drop(member, data):
        return (None, None) if member.name.endswith("rights.json") else (member, data)
    short = rewrite("short.tar", drop)
    with pytest.raises(B.BundleError, match="lacks 1 file"):
        B.read_bundle(short, tmp_path / "o3")

    def unlisted(member, data):
        return member, data
    unl = rewrite("unlisted.tar", unlisted)
    with tarfile.open(unl, "a") as t:
        info = tarfile.TarInfo("artist/extra.bin")
        info.size = 1
        t.addfile(info, __import__("io").BytesIO(b"x"))
    with pytest.raises(B.BundleError, match="does not name"):
        B.read_bundle(unl, tmp_path / "o4")

    def forged_id(member, data):
        if member.name == B.MANIFEST_NAME:
            doc = json.loads(data)
            doc["serial"] = 70000
            data = json.dumps(doc).encode()
            member.size = len(data)
        return member, data
    forged = rewrite("forged.tar", forged_id)
    with pytest.raises(B.BundleError, match="bundle id"):
        B.read_manifest(forged)
    plain = tmp_path / "plain.tar"
    with tarfile.open(plain, "w") as t:
        t.add(str(tmp_path / "src" / "rights.json"), arcname="rights.json")
    with pytest.raises(B.BundleError, match="begin with"):
        B.read_manifest(plain)
