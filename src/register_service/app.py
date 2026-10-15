"""The register's web face: five signed calls for machines, public reads for everyone.
Settings by environment: REGISTER_DATA_DIR (store, mirror), REGISTER_KEY_FILE (the
register's private key; REGISTER_DEV_KEY=1 lets a missing one be made, for development
only), REGISTER_PUBLIC_URL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from khaos_attribution import registry as R
from khaos_attribution.watermark import decode_codeword
from register_service.index import Index
from register_service.store import Store, now_utc

MAX_BODY = 8_000_000   # a long output's four-phase fingerprint runs to a few megabytes
CLAIM_MAX_BODY = 24_000_000   # a twelve-minute track's four phases, with room
DEFAULT_PORT = 8912   # 8900 is the Engine Room's
log = logging.getLogger("register_service")


def settings() -> dict:
    data = Path(os.environ.get("REGISTER_DATA_DIR", "./register-data")).resolve()
    key_file = Path(os.environ.get("REGISTER_KEY_FILE", str(data / "register-key.pem")))
    return {"data": data, "key_file": key_file, "dev_key": os.environ.get("REGISTER_DEV_KEY") == "1",
            "public_url": os.environ.get("REGISTER_PUBLIC_URL", f"http://127.0.0.1:{DEFAULT_PORT}").rstrip("/")}


def dev_marker(cfg: dict) -> Path:
    return cfg["key_file"].with_suffix(".dev")


def load_register_key(cfg: dict) -> R.SigningKey:
    """The countersigning key; a key ever served as the development key is refused once that setting is gone."""
    marker = dev_marker(cfg)
    if cfg["key_file"].is_file():
        key = R.SigningKey.load(cfg["key_file"])
        if cfg["dev_key"]:
            marker.write_text(key.key_id + "\n", encoding="utf-8")
        elif marker.is_file() and marker.read_text(encoding="utf-8").strip() == key.key_id:
            raise SystemExit(f"the key at {cfg['key_file']} is the development key ({key.key_id}); "
                             "the Guild's key is placed by the ceremony (deploy/key-ceremony.md)")
        return key
    if not cfg["dev_key"]:
        raise SystemExit(f"no register key at {cfg['key_file']}: the Guardians' key is made in the ceremony, never here "
                         "(REGISTER_DEV_KEY=1 makes a development key)")
    cfg["key_file"].parent.mkdir(parents=True, exist_ok=True)
    key = R.SigningKey.load_or_create(cfg["key_file"])
    marker.write_text(key.key_id + "\n", encoding="utf-8")
    return key


def public_view(record: dict) -> dict:
    """The record as the artist chose to show it: model alone, with tracks, or with writers' shares."""
    view = record.get("public_view", "model")
    out = json.loads(json.dumps(record))
    ts = out.get("training_set", {})
    tracks = ts.get("tracks", [])
    if view == "model":
        ts["tracks"] = []
        ts["track_count"] = len(tracks)
    elif view == "tracks":
        for t in tracks:
            t.pop("writers", None)
    return out


class RateLimit:
    """A token bucket per client address: the public doors take a bounded number of calls a minute each."""

    def __init__(self, per_minute: int):
        self.per_minute, self._buckets, self._lock = per_minute, {}, threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(client, (float(self.per_minute), now))
            tokens = min(float(self.per_minute), tokens + (now - last) * self.per_minute / 60.0)
            if tokens < 1.0:
                self._buckets[client] = (tokens, now)
                return False
            self._buckets[client] = (tokens - 1.0, now)
            if len(self._buckets) > 10000:
                stale = [k for k, (_, t) in self._buckets.items() if now - t > 600]
                for k in stale:
                    del self._buckets[k]
            return True


PUBLIC_LIMITS = {"/api/verify": 20, "/api/enrol": 6}
SIGNED_LIMIT = 120
# A verify key's own lane: far above the public bound, still a bound, per key.
KEYED_VERIFY_LIMIT = 600
# A Publish checks and claims every training track at once; a large catalogue must fit in a minute.
CLAIM_LIMITS = {"/api/claims": 600, "/api/claims/check": 600}


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or settings()
    store = Store(cfg["data"] / "register.sqlite3")
    index = Index(cfg["data"] / "index.sqlite3")
    key = load_register_key(cfg)
    mirror = cfg["data"] / "mirror"
    (mirror / "models").mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Guild Register", docs_url=None, redoc_url=None)
    app.state.store, app.state.index, app.state.key, app.state.cfg = store, index, key, cfg

    async def signed_body(request: Request, enrolling: bool = False, limit: int = MAX_BODY) -> tuple:
        raw = await request.body()
        if len(raw) > limit:
            raise HTTPException(413, "too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "not JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "the body must be a JSON object")
        if R.canonical_bytes(body) != raw:
            raise HTTPException(400, "the body must be canonical JSON (sorted keys, no spaces)")
        key_id = request.headers.get("X-Machine-Key-Id", "")
        sig = request.headers.get("X-Machine-Signature", "")
        if enrolling:
            pem = body.get("public_key_pem", "")
            try:
                if R.key_id_of(R.public_raw(R.load_public_pem(pem))) != key_id:
                    raise HTTPException(401, "the key id is not the public key's")
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001 — any malformed key is the same refusal
                raise HTTPException(400, "public_key_pem is not an Ed25519 public key") from None
            machine = None
        else:
            machine = store.machine(key_id)
            if not machine:
                raise HTTPException(401, "this machine is not enrolled")
            pem = machine["public_pem"]
        if not R.verify(pem, raw, sig):
            raise HTTPException(401, "the signature is not this machine's over this body")
        return body, machine, key_id, pem

    def write_mirror(serial: int) -> None:
        rec = store.record(serial)
        doc = {"serial": serial, "state": rec["state"], "record": public_view(rec["signed"]["record"]),
               "record_sha256": rec["record_sha256"], "machine_key_id": rec["machine_key_id"],
               "machine_public_pem": rec.get("machine_public_pem"),
               "countersignature": rec["countersignature"], "served": rec["served"],
               "withdrawn_at": rec["withdrawn_at"], "withdraw_reason": rec["withdraw_reason"]}
        tmp = mirror / "models" / f"{serial}.json.new"
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, mirror / "models" / f"{serial}.json")
        with open(mirror / "ledger.jsonl", "w", encoding="utf-8") as fh:
            for e in store.ledger():
                fh.write(json.dumps(e, sort_keys=True) + "\n")

    def mirror_quietly(serial: int) -> None:
        # The record is in the ledger; a mirror that cannot be written is logged and written on the next change.
        try:
            write_mirror(serial)
        except Exception as exc:  # noqa: BLE001
            log.error("mirror not written for %s: %s", serial, exc)

    limits = {path: RateLimit(n) for path, n in {**PUBLIC_LIMITS, **CLAIM_LIMITS}.items()}
    signed_limit = RateLimit(SIGNED_LIMIT)
    from register_service.verify_keys import Allowed  # noqa: PLC0415
    verify_key_allowed = Allowed(cfg)
    keyed_verify_limit = RateLimit(KEYED_VERIFY_LIMIT)

    @app.middleware("http")
    async def rate_limited(request: Request, call_next):
        # Caddy appends the real address to whatever the client sent, so the LAST entry is the one to trust.
        client = (request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or (request.client.host if request.client else "?"))
        path = request.url.path
        if request.method == "POST":
            bucket = limits.get(path) or (signed_limit if path.startswith("/api/") else None)
            if path == "/api/verify" and verify_key_allowed(request.headers.get("x-khaos-verify-key")):
                client, bucket = "key:" + request.headers["x-khaos-verify-key"].strip(), keyed_verify_limit
            if bucket is not None and not bucket.allow(client):
                return JSONResponse({"detail": "too many requests from this address; try again in a minute"}, status_code=429,
                                    headers={"Retry-After": "60"})
        return await call_next(request)

    @app.post("/api/enrol")
    async def enrol(request: Request):
        body, _, key_id, pem = await signed_body(request, enrolling=True)
        try:
            return store.enrol(str(body.get("invite", "")), key_id, str(body.get("machine_name", ""))[:80], pem)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None

    @app.post("/api/records")
    async def records(request: Request):
        signed, machine, key_id, pem = await signed_body(request)
        problems = R.signed_problems(signed, pem)
        if problems:
            raise HTTPException(422, "; ".join(problems))
        record = signed["record"]
        problems = R.record_problems(record)
        if problems:
            raise HTTPException(422, "; ".join(problems))
        if record["model"]["machine_key_id"] != key_id:
            raise HTTPException(422, "the record names another machine as its maker")
        serial, counter, already = store.register(signed, key_id, machine["account_id"],
                                                  lambda n: R.countersign(signed, n, key, now_utc()))
        if not already:
            mirror_quietly(serial)
        return {"serial": serial, "countersignature": counter, "already": already,
                "page": f"{cfg['public_url']}/models/{serial}"}

    def serial_of(body: dict) -> int:
        try:
            return int(body.get("serial"))
        except (TypeError, ValueError):
            raise HTTPException(422, "serial must be an integer") from None

    def once(body: dict, key_id: str, work):
        """The nonce is an idempotency key: the same signed body again gets its first answer, never a second effect."""
        digest = R.sha256_hex(R.canonical_bytes(body))
        try:
            earlier = store.nonce_answer(key_id, str(body.get("nonce", "")), digest)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        if earlier is not None:
            return earlier
        out = work()
        store.record_nonce(key_id, str(body.get("nonce")), digest, out)
        return out

    @app.post("/api/withdraw")
    async def withdraw(request: Request):
        body, _, key_id, _ = await signed_body(request)

        def work():
            try:
                out = store.withdraw(serial_of(body), key_id, str(body.get("reason", "")))
            except KeyError as e:
                raise HTTPException(404, str(e)) from None
            except PermissionError as e:
                raise HTTPException(403, str(e)) from None
            index.release_claims(out["serial"])
            mirror_quietly(out["serial"])
            return out
        return once(body, key_id, work)

    @app.post("/api/served")
    async def served(request: Request):
        body, _, key_id, _ = await signed_body(request)
        state = str(body.get("state", ""))
        if state not in ("serving", "removed"):
            raise HTTPException(422, "state is serving or removed")

        def work():
            try:
                out = store.served(serial_of(body), key_id, str(body.get("space_url", ""))[:200], state)
            except KeyError as e:
                raise HTTPException(404, str(e)) from None
            except PermissionError as e:
                raise HTTPException(403, str(e)) from None
            mirror_quietly(out["serial"])
            return out
        return once(body, key_id, work)


    @app.post("/api/outputs")
    async def outputs(request: Request):
        """A released output into the index, under its model's bucket: by the model's own account, or by a Space serving it."""
        body, machine, key_id, _ = await signed_body(request)

        def work():
            rec = store.record(serial_of(body))
            if not rec:
                raise HTTPException(404, "no such model")
            if rec["state"] == "withdrawn":
                raise HTTPException(410, "the model is withdrawn; its outputs are not released into the index")
            if machine["role"] == "workshop" and rec["account_id"] != machine["account_id"]:
                raise HTTPException(403, "a Workshop releases outputs of its own account's models only")
            if machine["role"] == "space" and not any(s["key_id"] == key_id and s["state"] == "serving" for s in rec["served"]):
                raise HTTPException(403, "a Space releases outputs of the models it reports serving")
            gid = str(body.get("generation_id") or "").strip()
            if not gid or len(gid) > 120:
                raise HTTPException(422, "generation_id is a short string")
            try:
                out = index.release(gid, rec["serial"], int(rec["signed"]["record"]["model"]["watermark_payload"]),
                                    rec["signed"]["record"]["artist_id"], body.get("fingerprint"), key_id, machine["role"])
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            return {k: out[k] for k in ("generation_id", "serial", "payload", "released_at", "landmarks", "already")}
        return once(body, key_id, work)

    def workshop_only(machine: dict) -> None:
        if machine["role"] != "workshop":
            raise HTTPException(403, "only a Workshop claims or checks training tracks")

    def phases_of(body: dict) -> list:
        phases = body.get("phases")
        if not isinstance(phases, list) or not phases or not all(isinstance(p, list) for p in phases):
            raise HTTPException(422, "phases is a list of fingerprint lists")
        try:
            return [[(int(h), int(o)) for h, o in p] for p in phases]
        except (TypeError, ValueError):
            raise HTTPException(422, "a landmark is [hash, offset_ms]") from None

    @app.post("/api/claims")
    async def claims(request: Request):
        """A registered model's training track, claimed on its account's declaration by the record it was registered under."""
        body, machine, key_id, _ = await signed_body(request, limit=CLAIM_MAX_BODY)
        workshop_only(machine)

        def work():
            serial = store.serial_of_record(str(body.get("record_sha256") or ""))
            rec = store.record(serial) if serial is not None else None
            if not rec:
                raise HTTPException(404, "no model is registered under that record")
            if rec["account_id"] != machine["account_id"]:
                raise HTTPException(403, "a Workshop claims tracks for its own account's models only")
            if rec["state"] == "withdrawn":
                raise HTTPException(410, "the model is withdrawn")
            track_id = str(body.get("track_id") or "").strip()
            declaration = str(body.get("declaration") or "").strip()
            trained = {str(t.get("track_id")) for t in (rec["signed"]["record"].get("training_set") or {}).get("tracks") or []}
            if track_id not in trained:
                raise HTTPException(422, "a claim names one of the training tracks the model's own record lists")
            if not declaration or len(declaration) > 64:
                raise HTTPException(422, "declaration names the declaration the artist made")
            phases = body.get("phases")
            if phases is not None and not (isinstance(phases, list) and all(isinstance(p, list) for p in phases) and len(phases) <= 8):
                raise HTTPException(422, "phases is a short list of fingerprint lists")
            fingerprint = body.get("fingerprint") or (phases[0] if phases else None)
            try:
                out = index.claim(rec["account_id"], rec["serial"], track_id, fingerprint, declaration, key_id, phases)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            return {"serial": rec["serial"], "track_id": track_id, **out}
        # The fingerprint matching is heavy: off the event loop, so verify and enrol stay answered meanwhile.
        return await run_in_threadpool(once, body, key_id, work)

    @app.post("/api/claims/check")
    async def claims_check(request: Request):
        """Whether a recording already trains a registered model (nobody's, this account's, or also another account's)
        and whether it is a released output: the Workshop's one question at import."""
        body, machine, _, _ = await signed_body(request)
        workshop_only(machine)
        return await run_in_threadpool(index.check_claim, phases_of(body), machine["account_id"])

    @app.get("/api/models/{serial}/claims")
    def model_claims(serial: int):
        rec = store.record(serial)
        if not rec:
            raise HTTPException(404, "no such model")
        tracks = len((rec["signed"]["record"].get("training_set") or {}).get("tracks") or [])
        return {"serial": serial, "tracks": tracks, **index.claims_of(serial)}

    @app.post("/api/verify")
    async def verify(request: Request):
        """Anyone's question, no key: a four-phase fingerprint and, when it read, the watermark codeword."""
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise HTTPException(413, "too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "not JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "the body must be a JSON object")
        phases = body.get("phases")
        if not isinstance(phases, list) or not phases or not all(isinstance(p, list) for p in phases):
            raise HTTPException(422, "phases is a list of fingerprint lists")
        try:
            phases = [[(int(h), int(o)) for h, o in p] for p in phases]
        except (TypeError, ValueError):
            raise HTTPException(422, "a landmark is [hash, offset_ms]") from None
        payload = None
        codeword = body.get("watermark_id")
        if codeword is not None:
            try:
                decoded = decode_codeword(int(codeword))
            except (TypeError, ValueError):
                decoded = None
            if decoded is None:
                raise HTTPException(422, "watermark_id is not a codeword")
            payload = decoded[0]
        answer = await run_in_threadpool(index.verify, phases, payload)
        if answer["grade"] == "output":
            rec = store.record(int(answer["output"]["serial"]))
            answer["model"] = _model_summary(rec) if rec else None
        elif answer["grade"] == "bucket":
            answer["models"] = [_model_summary(store.record(s)) for s in store.serials_in_bucket(payload)]
        return answer

    def _claims_line(c: dict, total: int) -> str:
        from register_service import pages as P  # noqa: PLC0415
        line = f"{c['claimed']} of {total} claimed on the artist's declaration"
        if c["shared"]:
            line += f" · {c['shared']} also train{'s' if c['shared'] == 1 else ''} another account's registered model"
        return P.esc(line)

    def _model_summary(rec: dict) -> dict:
        r = rec["signed"]["record"]
        return {"serial": rec["serial"], "artist_id": r["artist_id"], "artist_name": r["artist_name"], "run_id": r["model"]["run_id"],
                "state": rec["state"], "page": f"{cfg['public_url']}/models/{rec['serial']}"}

    @app.get("/api/public-key")
    def public_key():
        return {"register_key_id": key.key_id, "public_key_pem": key.public_pem(), "development": bool(cfg["dev_key"])}

    @app.get("/api/models/{serial}")
    def model_json(serial: int):
        path = mirror / "models" / f"{serial}.json"
        if not path.is_file():
            raise HTTPException(404, "no such model")
        return Response(path.read_bytes(), media_type="application/json")

    @app.get("/api/withdrawn")
    def withdrawn():
        return {"withdrawn": store.withdrawn_serials()}

    @app.get("/api/ledger")
    def ledger(since: int = 0):
        return PlainTextResponse("\n".join(json.dumps(e, sort_keys=True) for e in store.ledger(since)) + "\n",
                                 media_type="application/x-ndjson")

    @app.get("/models/{serial}", response_class=HTMLResponse)
    def model_page(serial: int):
        from register_service import pages as P  # noqa: PLC0415
        rec = store.record(serial)
        if not rec:
            raise HTTPException(404, "no such model")
        r = public_view(rec["signed"]["record"])
        m, ts = r["model"], r["training_set"]
        tracks = ts.get("tracks") or []
        facts = [P.state_chip(rec["state"]), f"registered {P.esc(P.day(rec['published_at']))}", P.esc(m["base_model"]),
                 P.count(len(tracks) or ts.get("track_count", 0), "track")]
        note = ("What the artist's Workshop attested, bound to hashes the artist's own machine can reproduce. The register cannot "
                "see the audio and does not verify the claim; it makes it checkable.")
        body = P.hero(f"Register No. {serial}", r["artist_name"], note, facts, P.seal_for(serial, rec["signed"]["record"]))
        kv = [("Base model", f"{P.esc(m['base_model'])} <span class=\"muted\">· {P.esc(m['base_model_licence'])} licence</span>"),
              ("Adapter", f'<span class="mono">{P.esc(m["adapter_sha256"])}</span>'),
              ("Model card", f'<span class="mono">{P.esc(m["card_sha256"])}</span>'),
              ("Dataset hash", f'<span class="mono">{P.esc(r["provenance"]["dataset_hash"])}</span>'),
              ("Watermark bucket", P.esc(m["watermark_payload"])),
              ("Countersigned by", f'<span class="mono">{P.esc(rec["countersignature"]["register_key_id"])}</span>'),
              ("Training tracks", _claims_line(index.claims_of(serial), len((rec["signed"]["record"].get("training_set") or {}).get("tracks") or []))),
              ("Registered", P.esc(rec["published_at"]))]
        body += P.section("The record", "every value as the artist's Workshop attested it")
        body += P.table("200px minmax(0, 1fr)", ["", ""], [[(f'<span class="stage-kv-label">{P.esc(k)}</span>', ""), (v, "wrap")] for k, v in kv])
        if tracks:
            rows = []
            for i, t in enumerate(tracks, 1):
                writers = ", ".join(f"{P.esc(w['name'])} {w['share']:.0%}" for w in t.get("writers", [])) or '<span class="muted">—</span>'
                rows.append([(str(i), "num"), P.esc(t.get("title") or t["track_id"]), (writers, "muted")])
            body += P.section("Trained on", P.count(len(tracks), "track"))
            body += P.table("44px minmax(0, 1fr) minmax(0, 1fr)", ["#", "Track", "Writers"], rows)
        else:
            body += P.section("Trained on", "the artist shows the model alone")
            body += f'<div class="empty">{P.count(ts.get("track_count", 0), "track")}; the artist chose not to list them.</div>'
        if rec["served"]:
            body += P.section("Served at")
            body += P.table("minmax(0, 1fr) 120px 160px", ["Listening Space", "State", "Since"],
                            [[(f'<a href="{P.esc(s_["space_url"])}">{P.esc(s_["space_url"])}</a>', "wrap"), (P.esc(s_["state"]), "muted"), (P.esc(P.day(s_["at"])), "muted")]
                             for s_ in rec["served"]])
        body += f'<div class="stage-pill-row"><a class="k-btn k-btn--pill" href="/api/models/{serial}">The record as JSON</a></div>'
        return P.page(f"Register No. {serial} · {r['artist_name']}", body, dev=bool(cfg["dev_key"]))

    @app.get("/", response_class=HTMLResponse)
    def home():
        from register_service import pages as P  # noqa: PLC0415
        serials = store.serials()
        rows = []
        for serial in serials[-100:][::-1]:
            rec = store.record(serial)
            if not rec:
                continue
            r = rec["signed"]["record"]
            rows.append((f"/models/{serial}", [(str(serial), "num"), P.esc(r["artist_name"]), (P.esc(P.day(rec["published_at"])), "muted"),
                                                (P.state_chip(rec["state"]), "")]))
        para = ("The public ledger of registered fine-tuned models: what each was trained on, as its artist attested, "
                "countersigned by the register and never edited.")
        facts = [P.count(len(serials), "model") + " registered", "ledger unbroken" if store.ledger_ok() else "ledger BROKEN",
                 "serials from 65,536"]
        body = P.hero("The Guild of Fine Tuners", "The Guild Register.", para, facts)
        body += P.section("Registered models", "newest first")
        body += (P.table("80px minmax(0, 1fr) 120px 120px", ["No.", "Artist", "Registered", "State"], rows) if rows
                 else '<div class="empty">No model is registered yet.</div>')
        return P.page("Guild Register", body, dev=bool(cfg["dev_key"]))

    @app.get("/health")
    def health():
        return JSONResponse({"ok": True, "ledger_ok": store.ledger_ok(), "models": len(store.serials()), "outputs": index.count(),
                             "development": bool(cfg["dev_key"])})

    return app
