"""The Guild Register's contract: machine keys, the record a Workshop signs at
Publish, the register's countersignature, the outbox that survives being
offline, and the client. The service itself lives in register_service/.
Ed25519 through `cryptography` (the `registry` extra); nothing else here
needs it, so the module imports it lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

RECORD_VERSION = "guild.register.record/1"
SIGNED_VERSION = "guild.register.signed/1"
FIRST_SERIAL = 65536   # above every 16-bit codeword a seal struck before the register carried
PUBLIC_VIEWS = ("model", "tracks", "writers")
OUTBOX_DIR = ".register_outbox"
MACHINE_KEY_FILE = ".register_machine_key.pem"
MACHINE_KEY_PUBLIC_FILE = ".register_machine_key.pub"


def canonical_bytes(doc: dict) -> bytes:
    """One byte form for one document, so a signature over it means one thing everywhere."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ed25519():
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415
    return ed25519, serialization


# ---- keys ----

@dataclass
class SigningKey:
    """An Ed25519 key with its id: the first 16 hex characters of the sha256 of the raw public key."""
    _private: object
    key_id: str

    @classmethod
    def generate(cls) -> "SigningKey":
        ed25519, _ = _ed25519()
        priv = ed25519.Ed25519PrivateKey.generate()
        return cls(priv, key_id_of(public_raw(priv.public_key())))

    @classmethod
    def load(cls, path: Path) -> "SigningKey":
        _, serialization = _ed25519()
        priv = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
        return cls(priv, key_id_of(public_raw(priv.public_key())))

    @classmethod
    def load_or_create(cls, path: Path) -> "SigningKey":
        """A machine makes its key once; the private half never leaves the file."""
        path = Path(path)
        if path.is_file():
            return cls.load(path)
        key = cls.generate()
        _, serialization = _ed25519()
        pem = key._private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption())
        tmp = path.with_suffix(".new")
        tmp.write_bytes(pem)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        path.with_name(MACHINE_KEY_PUBLIC_FILE).write_text(key.public_pem(), encoding="utf-8")
        return key

    def public_pem(self) -> str:
        _, serialization = _ed25519()
        return self._private.public_key().public_bytes(serialization.Encoding.PEM,
                                                       serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")

    def sign(self, data: bytes) -> str:
        return self._private.sign(data).hex()


def public_raw(public_key) -> bytes:
    _, serialization = _ed25519()
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id_of(raw_public: bytes) -> str:
    return sha256_hex(raw_public)[:16]


def load_public_pem(pem: str):
    _, serialization = _ed25519()
    return serialization.load_pem_public_key(pem.encode("ascii"))


def verify(pem: str, data: bytes, signature_hex: str) -> bool:
    """True when the signature is the key's over the bytes; a bad signature is False, never an exception."""
    try:
        load_public_pem(pem).verify(bytes.fromhex(signature_hex), data)
        return True
    except Exception:  # noqa: BLE001 — every failure means the same thing here
        return False


# ---- the record ----

REQUIRED_RECORD_KEYS = (
    "record_version", "artist_id", "artist_name", "artist_mark", "consent", "model", "provenance",
    "training_set", "measured", "public_view", "published_at_utc",
)


def record_problems(record: dict) -> list:
    """What keeps a record out of the register, in words; empty means it may be signed."""
    out = []
    for k in REQUIRED_RECORD_KEYS:
        if k not in record:
            out.append(f"missing {k}")
    if out:
        return out
    if record["record_version"] != RECORD_VERSION:
        out.append(f"record_version is {record['record_version']!r}, not {RECORD_VERSION!r}")
    if record["public_view"] not in PUBLIC_VIEWS:
        out.append(f"public_view must be one of {PUBLIC_VIEWS}")
    model = record["model"]
    for k in ("run_id", "adapter_sha256", "card_sha256", "watermark_payload", "base_model", "base_model_licence", "machine_key_id"):
        if not model.get(k) and model.get(k) != 0:
            out.append(f"model.{k} is empty")
    payload = model.get("watermark_payload")
    if not isinstance(payload, int) or not 32 <= payload < 2048:
        out.append("model.watermark_payload must be a derived payload, 32 to 2047")
    for k in ("adapter_sha256", "card_sha256"):
        v = model.get(k)
        if not (isinstance(v, str) and len(v) == 64):
            out.append(f"model.{k} must be a sha256 hex digest")
    consent = record["consent"]
    if not isinstance(consent.get("statement_sha256"), str) or len(consent["statement_sha256"]) != 64:
        out.append("consent.statement_sha256 must be a sha256 hex digest")
    tracks = record["training_set"].get("tracks")
    if not isinstance(tracks, list) or not tracks:
        out.append("training_set.tracks must name at least one track")
    if record["public_view"] == "writers":
        agreed = record["training_set"].get("writers_agreed")
        if agreed is not True:
            out.append("public_view 'writers' needs training_set.writers_agreed: true, every named writer having agreed")
    return out


def sign_record(record: dict, key: SigningKey) -> dict:
    """The Workshop's signature over the canonical record; refused with the problems when the record is not whole."""
    problems = record_problems(record)
    if problems:
        raise ValueError("; ".join(problems))
    body = canonical_bytes(record)
    return {"signed_version": SIGNED_VERSION, "record": record, "record_sha256": sha256_hex(body),
            "machine_key_id": key.key_id, "machine_signature": key.sign(body)}


def signed_problems(signed: dict, machine_public_pem: str) -> list:
    """What keeps a signed record out: the hash not matching the record, or the signature not the key's."""
    out = []
    if signed.get("signed_version") != SIGNED_VERSION:
        out.append("not a signed record")
        return out
    body = canonical_bytes(signed.get("record") or {})
    if signed.get("record_sha256") != sha256_hex(body):
        out.append("record_sha256 does not match the record")
    if key_id_of(public_raw(load_public_pem(machine_public_pem))) != signed.get("machine_key_id"):
        out.append("machine_key_id is not this key's")
    if not verify(machine_public_pem, body, signed.get("machine_signature", "")):
        out.append("machine_signature is not this key's over this record")
    return out


def countersign(signed: dict, serial: int, register_key: SigningKey, at_utc: str) -> dict:
    """The register's answer: the serial and its signature over the record hash, the serial and the time."""
    if serial < FIRST_SERIAL:
        raise ValueError(f"a serial is {FIRST_SERIAL} or more")
    payload = {"record_sha256": signed["record_sha256"], "serial": serial, "countersigned_at_utc": at_utc}
    return {**payload, "register_key_id": register_key.key_id, "register_signature": register_key.sign(canonical_bytes(payload))}


def countersignature_valid(counter: dict, record_sha256: str, register_public_pem: str) -> bool:
    """True when the register's signature covers this record's hash, this serial and this time."""
    try:
        payload = {"record_sha256": counter["record_sha256"], "serial": int(counter["serial"]),
                   "countersigned_at_utc": counter["countersigned_at_utc"]}
    except (KeyError, TypeError, ValueError):
        return False
    if payload["record_sha256"] != record_sha256 or payload["serial"] < FIRST_SERIAL:
        return False
    return verify(register_public_pem, canonical_bytes(payload), counter.get("register_signature", ""))


# ---- the outbox ----

class Outbox:
    """Records a machine made while offline, kept in order until the register has taken each one."""

    def __init__(self, root: Path):
        self.dir = Path(root) / OUTBOX_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, body: dict) -> str:
        entry = {"kind": kind, "body": body, "created_at": time.time()}
        entry_id = sha256_hex(canonical_bytes({"kind": kind, "body": body}))
        existing = sorted(self.dir.glob("*.json")) + sorted((self.dir / "sent").glob("*.json"))
        if any(p.name.endswith(f"-{entry_id[:12]}.json") for p in existing):
            return entry_id   # the same record twice is one entry
        # Order is the sequence number, never the clock: two appends in one tick still keep their order.
        seq = max((int(p.name.split("-", 1)[0]) for p in existing if p.name.split("-", 1)[0].isdigit()), default=0) + 1
        path = self.dir / f"{seq:08d}-{entry_id[:12]}.json"
        tmp = path.with_suffix(".new")
        tmp.write_text(json.dumps({"id": entry_id, **entry}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return entry_id

    def pending(self) -> list:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append((p, json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return out

    def mark_sent(self, path: Path, answer: dict) -> None:
        sent = self.dir / "sent"
        sent.mkdir(exist_ok=True)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        doc["answer"] = answer
        (sent / Path(path).name).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        Path(path).unlink()


# ---- the client ----

class RegisterError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


class RegisterClient:
    """The five calls a machine makes; every one carries the machine's signature over its body."""

    def __init__(self, base_url: str, key: SigningKey, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        data = canonical_bytes(body)
        req = urllib.request.Request(self.base_url + path, data=data, method="POST", headers={
            "Content-Type": "application/json", "X-Machine-Key-Id": self.key.key_id,
            "X-Machine-Signature": self.key.sign(data)})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise RegisterError(e.code, (e.read().decode("utf-8", "replace") or e.reason)[:400]) from None

    def enrol(self, invite: str, machine_name: str) -> dict:
        return self._post("/api/enrol", {"invite": invite, "machine_name": machine_name,
                                         "public_key_pem": self.key.public_pem(), "nonce": secrets.token_hex(8)})

    def publish(self, signed: dict) -> dict:
        return self._post("/api/records", signed)

    def withdraw(self, serial: int, reason: str) -> dict:
        return self._post("/api/withdraw", {"serial": serial, "reason": reason, "nonce": secrets.token_hex(8)})

    def served(self, serial: int, space_url: str, state: str) -> dict:
        return self._post("/api/served", {"serial": serial, "space_url": space_url, "state": state, "nonce": secrets.token_hex(8)})

    def sync(self, outbox: Outbox) -> list:
        """Every pending entry, in order; stops at the first the register refuses so order is kept."""
        done = []
        for path, entry in outbox.pending():
            kind, body = entry["kind"], entry["body"]
            answer = {"publish": self.publish, "withdraw": lambda b: self.withdraw(**b),
                      "served": lambda b: self.served(**b)}[kind](body)
            outbox.mark_sent(path, answer)
            done.append((entry["id"], answer))
        return done
