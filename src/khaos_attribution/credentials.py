"""Content Credentials (C2PA) for Khaos outputs: a signed, time-stamped record inside the audio file saying it is
AI-generated, by which model, under which licence and with which watermark. Each installation signs with its own
identity: a local issuer certificate (certified by Khaos when the machine enrols) and a signer certificate under it.
Needs the `credentials` extra (c2pa-python, cryptography).
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# The IPTC term C2PA uses for media made by a trained model.
DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
KHAOS_ASSERTION = "org.khaos.provenance"
ASSERTION_VERSION = 1
# A public RFC 3161 time-stamping service; KHAOS_TSA_URL overrides it, "off" signs without a timestamp.
DEFAULT_TSA_URL = "http://timestamp.digicert.com"
MIME_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4"}
IDENTITY_FILES = ("issuer.pem", "issuer.key", "signer.pem", "signer.key")
SIGNER_DAYS = 825
ISSUER_DAYS = 3650
RENEW_DAYS = 30
SIGNED_STATES = ("Valid", "Trusted")
# A time-stamping service that does not answer within this is skipped for TSA_RETRY_S, not waited on per file.
TSA_PROBE_S = 2.0
TSA_RETRY_S = 600.0
# Record fields that travel inside the file; the prompt and anything about a listener never do.
RECORD_FIELDS = ("artist_id", "artist_name", "adapter_version", "adapter_hash", "base_model", "base_model_licence",
                 "generation_id", "watermark_id", "register_serial", "timestamp")


class CredentialsUnavailable(RuntimeError):
    """The credentials extra is not installed, or the identity cannot be made."""


def available() -> bool:
    try:
        import c2pa  # noqa: F401, PLC0415
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


@dataclass
class Identity:
    """One installation's signing identity: the chain (signer then issuer, PEM) and the signer's key."""
    chain_pem: str
    key: object
    signer_id: str
    issuer_pem: str


def _name(product: str, role: str, subject_id: str):
    from cryptography import x509  # noqa: PLC0415
    from cryptography.x509.oid import NameOID  # noqa: PLC0415
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, product),
                      x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, role),
                      x509.NameAttribute(NameOID.COMMON_NAME, f"{product} {role} {subject_id}")])


def _certificate(subject, subject_key, issuer, issuer_key, *, ca: bool, days: int):
    from cryptography import x509  # noqa: PLC0415
    from cryptography.x509.oid import ExtendedKeyUsageOID  # noqa: PLC0415
    now = datetime.datetime.now(datetime.timezone.utc)
    usage = dict(content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False,
                 encipher_only=False, decipher_only=False)
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(subject_key.public_key())
         .serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=days))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=0 if ca else None), critical=True)
         .add_extension(x509.KeyUsage(digital_signature=not ca, key_cert_sign=ca, crl_sign=ca, **usage), critical=True)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(subject_key.public_key()), critical=False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False))
    if not ca:
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
    return b.sign(issuer_key, None)


def _write(path: Path, data: bytes, *, private: bool) -> None:
    """Write through a temporary file in the same folder, so a reader never sees half a key or certificate."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o600 if private else 0o644)
        _replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace(src, dst) -> None:
    """os.replace, retried briefly on Windows where a file being read cannot be replaced."""
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or time.monotonic() > deadline:
                raise
            time.sleep(0.05)


_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _identity_lock(directory: Path):
    """One maker of an identity at a time, across threads and processes."""
    with _THREAD_LOCK, open(directory / ".lock", "a+b") as fh:
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415
            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _loaded(paths: dict):
    """The four files as (issuer, issuer_key, signer, signer_key), or None when any is missing or unreadable."""
    from cryptography import x509  # noqa: PLC0415
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    try:
        return (x509.load_pem_x509_certificate(paths["issuer.pem"].read_bytes()),
                serialization.load_pem_private_key(paths["issuer.key"].read_bytes(), None),
                x509.load_pem_x509_certificate(paths["signer.pem"].read_bytes()),
                serialization.load_pem_private_key(paths["signer.key"].read_bytes(), None))
    except Exception:  # noqa: BLE001 — a missing, partial or foreign file means the identity is remade
        return None


def _public(key) -> bytes:
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _sound(loaded) -> tuple[bool, bool]:
    """(issuer usable, signer usable): each key matches its certificate, the signer is the issuer's and in date."""
    if loaded is None:
        return False, False
    issuer, issuer_key, signer, signer_key = loaded
    soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=RENEW_DAYS)
    issuer_ok = (issuer.public_key().public_bytes_raw() == _public(issuer_key) and issuer.not_valid_after_utc > soon)
    if not issuer_ok:
        return False, False
    try:
        signer.verify_directly_issued_by(issuer)
    except Exception:  # noqa: BLE001 — any failure means this signer is not the issuer's
        return True, False
    return True, signer.public_key().public_bytes_raw() == _public(signer_key) and signer.not_valid_after_utc > soon


def load_identity(directory: Path | str, *, product: str, subject_id: str) -> Identity:
    """This installation's identity, made once under `directory` (keys readable by the owner only) and reused;
    a signer that is near expiry, or not the issuer's, is remade under a lock."""
    if not available():
        raise CredentialsUnavailable("Content Credentials need the credentials extra (c2pa-python)")
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415
    d = Path(directory)
    paths = {name: d / name for name in IDENTITY_FILES}
    loaded = _loaded(paths)
    if _sound(loaded) != (True, True):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        with _identity_lock(d):
            loaded = _loaded(paths)
            issuer_ok, signer_ok = _sound(loaded)
            pem = serialization.Encoding.PEM
            pkcs8 = (pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
            if issuer_ok:
                issuer, issuer_key = loaded[0], loaded[1]
            else:
                issuer_key = ed25519.Ed25519PrivateKey.generate()
                name = _name(product, "issuer", subject_id)
                issuer = _certificate(name, issuer_key, name, issuer_key, ca=True, days=ISSUER_DAYS)
                _write(paths["issuer.key"], issuer_key.private_bytes(*pkcs8), private=True)
                _write(paths["issuer.pem"], issuer.public_bytes(pem), private=False)
            if not (issuer_ok and signer_ok):
                signer_key = ed25519.Ed25519PrivateKey.generate()
                left = (issuer.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)).days
                signer = _certificate(_name(product, "signer", subject_id), signer_key, issuer.subject, issuer_key,
                                      ca=False, days=min(SIGNER_DAYS, left))
                _write(paths["signer.key"], signer_key.private_bytes(*pkcs8), private=True)
                _write(paths["signer.pem"], signer.public_bytes(pem), private=False)
            loaded = _loaded(paths)
    issuer, _, signer, key = loaded
    issuer_pem = issuer.public_bytes(serialization.Encoding.PEM).decode()
    return Identity(chain_pem=signer.public_bytes(serialization.Encoding.PEM).decode() + issuer_pem, key=key,
                    signer_id=hashlib.sha256(_public(key)).hexdigest()[:16], issuer_pem=issuer_pem)


def manifest(record: dict, *, product: str, version: str, title: str, mime: str, note: str | None = None) -> dict:
    """The C2PA manifest for one output: created by a trained model, and Khaos's own record of which one."""
    khaos = {"ai_generated": True, "assertion_version": ASSERTION_VERSION,
             **{k: record[k] for k in RECORD_FIELDS if record.get(k) is not None}}
    if note:
        khaos["note"] = note
    return {
        "claim_generator_info": [{"name": product, "version": version}],
        "title": title,
        "format": mime,
        "assertions": [
            {"label": "c2pa.actions", "data": {"actions": [{
                "action": "c2pa.created", "digitalSourceType": DIGITAL_SOURCE_TYPE,
                "softwareAgent": {"name": product, "version": version}}]}},
            {"label": KHAOS_ASSERTION, "data": khaos},
        ],
    }


def tsa_url() -> str | None:
    url = os.environ.get("KHAOS_TSA_URL", DEFAULT_TSA_URL).strip()
    return None if url in ("", "off") else url


_TSA_DOWN: dict[str, float] = {}


def _tsa_reachable(url: str) -> bool:
    """A quick connect to the service; a refusal is remembered for TSA_RETRY_S so blocked machines sign at once."""
    if time.monotonic() < _TSA_DOWN.get(url, 0.0):
        return False
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=TSA_PROBE_S).close()
    except OSError:
        _TSA_DOWN[url] = time.monotonic() + TSA_RETRY_S
        return False
    return True


def sign_file(path: Path | str, record: dict, identity: Identity, *, product: str, version: str,
              note: str | None = None, timestamp_url: str | None = "default") -> dict:
    """Embed a signed manifest in the audio file in place; time-stamped when the service answers, else signed
    without one. Returns {signed, timestamped, signer}; raises when the file cannot be signed."""
    import c2pa  # noqa: PLC0415
    path = Path(path)
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"{path.suffix} files cannot carry Content Credentials here")
    doc = json.dumps(manifest(record, product=product, version=version, title=path.name, mime=mime, note=note))
    url = tsa_url() if timestamp_url == "default" else timestamp_url
    if url and not _tsa_reachable(url):
        url = None
    last = None
    for attempt_url in ([url, None] if url else [None]):
        signer = c2pa.Signer.from_callback(lambda data: identity.key.sign(data), c2pa.C2paSigningAlg.ED25519,
                                           identity.chain_pem, attempt_url)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".c2pa")
        os.close(fd)
        try:
            with open(path, "rb") as src, open(tmp, "wb") as dst:
                c2pa.Builder.from_json(doc).sign(signer, mime, src, dst)
            _replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 — a time-stamping service that fails leaves the file unstamped
            last = exc
            continue
        finally:
            signer.close()
            Path(tmp).unlink(missing_ok=True)
        if attempt_url:
            _TSA_DOWN.pop(attempt_url, None)
        return {"signed": True, "timestamped": attempt_url is not None, "signer": identity.signer_id}
    raise RuntimeError(f"the Content Credentials could not be signed: {last}")


def is_signed(path: Path | str) -> bool:
    """True when the file carries Content Credentials whose signature still holds."""
    return read_file(path).get("state") in SIGNED_STATES


def read_file(path: Path | str) -> dict:
    """What a file's Content Credentials say: {present, state, ai_generated, khaos, signer, timestamped, trusted}.
    `state` is the SDK's verdict (Valid, Invalid, Trusted); absent credentials read as present=False."""
    out = {"present": False, "state": None, "ai_generated": False, "khaos": None, "signer": None,
           "timestamped": False, "trusted": False}
    if not available():
        out["reason"] = "the credentials extra is not installed"
        return out
    import c2pa  # noqa: PLC0415
    try:
        reader = c2pa.Reader(str(path))
    except Exception as exc:  # noqa: BLE001 — no manifest, or a format the SDK cannot open
        out["reason"] = str(exc)[:200]
        return out
    try:
        active = reader.get_active_manifest() or {}
        results = (reader.get_validation_results() or {}).get("activeManifest") or {}
        codes = {r.get("code") for kind in ("success", "informational", "failure") for r in results.get(kind, [])}
        out.update(present=bool(active), state=reader.get_validation_state())
        for a in active.get("assertions", []):
            data = a.get("data") or {}
            if a.get("label", "").startswith("c2pa.actions"):
                out["ai_generated"] = out["ai_generated"] or any(
                    act.get("digitalSourceType") == DIGITAL_SOURCE_TYPE for act in data.get("actions", []))
            elif a.get("label") == KHAOS_ASSERTION:
                out["khaos"] = data
        info = active.get("signature_info") or {}
        out["signer"] = info.get("common_name") or info.get("issuer")
        out["timestamped"] = "timeStamp.validated" in codes
        out["trusted"] = out["state"] == "Trusted"
    finally:
        reader.close()
    return out
