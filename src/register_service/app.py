"""The register's web face: five signed calls for machines, public reads for everyone.
Settings by environment: REGISTER_DATA_DIR (store, mirror), REGISTER_KEY_FILE (the
register's private key; REGISTER_DEV_KEY=1 lets a missing one be made, for development
only), REGISTER_PUBLIC_URL.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from khaos_attribution import registry as R
from register_service.store import Store, now_utc

MAX_BODY = 2_000_000


def settings() -> dict:
    data = Path(os.environ.get("REGISTER_DATA_DIR", "./register-data")).resolve()
    key_file = Path(os.environ.get("REGISTER_KEY_FILE", str(data / "register-key.pem")))
    return {"data": data, "key_file": key_file, "dev_key": os.environ.get("REGISTER_DEV_KEY") == "1",
            "public_url": os.environ.get("REGISTER_PUBLIC_URL", "http://127.0.0.1:8900").rstrip("/")}


def load_register_key(cfg: dict) -> R.SigningKey:
    if cfg["key_file"].is_file():
        return R.SigningKey.load(cfg["key_file"])
    if not cfg["dev_key"]:
        raise SystemExit(f"no register key at {cfg['key_file']}: the Guardians' key is made in the ceremony, never here "
                         "(REGISTER_DEV_KEY=1 makes a development key)")
    cfg["key_file"].parent.mkdir(parents=True, exist_ok=True)
    return R.SigningKey.load_or_create(cfg["key_file"])


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


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or settings()
    store = Store(cfg["data"] / "register.sqlite3")
    key = load_register_key(cfg)
    mirror = cfg["data"] / "mirror"
    (mirror / "models").mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Guild Register", docs_url=None, redoc_url=None)
    app.state.store, app.state.key, app.state.cfg = store, key, cfg

    async def signed_body(request: Request, enrolling: bool = False) -> tuple:
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise HTTPException(413, "too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "not JSON") from None
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
               "countersignature": rec["countersignature"], "served": rec["served"],
               "withdrawn_at": rec["withdrawn_at"], "withdraw_reason": rec["withdraw_reason"]}
        tmp = mirror / "models" / f"{serial}.json.new"
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, mirror / "models" / f"{serial}.json")
        with open(mirror / "ledger.jsonl", "w", encoding="utf-8") as fh:
            for e in store.ledger():
                fh.write(json.dumps(e, sort_keys=True) + "\n")

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
        have = store.existing(signed["record_sha256"])
        if have:
            return {"serial": have["serial"], "countersignature": json.loads(have["counter_json"]), "already": True}
        serial = store.next_serial()
        counter = R.countersign(signed, serial, key, now_utc())
        store.insert_record(serial, signed, counter, key_id, machine["account_id"])
        write_mirror(serial)
        return {"serial": serial, "countersignature": counter, "already": False,
                "page": f"{cfg['public_url']}/models/{serial}"}

    @app.post("/api/withdraw")
    async def withdraw(request: Request):
        body, _, key_id, _ = await signed_body(request)
        try:
            out = store.withdraw(int(body.get("serial", 0)), key_id, str(body.get("reason", "")))
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None
        write_mirror(out["serial"])
        return out

    @app.post("/api/served")
    async def served(request: Request):
        body, _, key_id, _ = await signed_body(request)
        state = str(body.get("state", ""))
        if state not in ("serving", "removed"):
            raise HTTPException(422, "state is serving or removed")
        try:
            out = store.served(int(body.get("serial", 0)), key_id, str(body.get("space_url", ""))[:200], state)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        write_mirror(out["serial"])
        return out

    @app.get("/api/public-key")
    def public_key():
        return {"register_key_id": key.key_id, "public_key_pem": key.public_pem()}

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
        rec = store.record(serial)
        if not rec:
            raise HTTPException(404, "no such model")
        r = public_view(rec["signed"]["record"])
        m, ts = r["model"], r["training_set"]
        rows = [("Register number", str(serial)), ("Artist", r["artist_name"]), ("Registered", rec["published_at"]),
                ("State", rec["state"]), ("Base model", f"{m['base_model']} ({m['base_model_licence']})"),
                ("Adapter", m["adapter_sha256"]), ("Model card", m["card_sha256"]),
                ("Dataset hash", r["provenance"]["dataset_hash"]), ("Watermark bucket", str(m["watermark_payload"])),
                ("Countersigned by", rec["countersignature"]["register_key_id"])]
        tracks = "".join(f"<li>{html.escape(t.get('title') or t['track_id'])}"
                         + (" · " + ", ".join(f"{html.escape(w['name'])} {w['share']:.0%}" for w in t.get("writers", [])) if t.get("writers") else "")
                         + "</li>" for t in ts.get("tracks", []))
        served = "".join(f"<li>{html.escape(s['space_url'])} · {s['state']} · {s['at']}</li>" for s in rec["served"])
        body = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows)
        note = ("This page records what the artist's Workshop attested, bound to hashes the artist's own machine can "
                "reproduce. The register cannot see the audio and does not verify the claim; it makes it checkable.")
        return f"""<!doctype html><meta charset="utf-8"><title>Guild Register · {serial}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#141a1b}}
th{{text-align:left;padding:4px 12px 4px 0;vertical-align:top;white-space:nowrap}}td{{word-break:break-all}}h1{{font-size:1.6rem}}
p.note{{color:#5a6668;font-size:0.95rem}}</style>
<h1>Register No. {serial}</h1><table>{body}</table>
{'<h2>Trained on</h2><ul>' + tracks + '</ul>' if tracks else f'<p>Trained on {ts.get("track_count", 0)} tracks (the artist shows the model alone).</p>'}
{'<h2>Served at</h2><ul>' + served + '</ul>' if served else ''}
<p class="note">{note}</p>
<p><a href="/api/models/{serial}">The record as JSON</a> · <a href="/api/ledger">The ledger</a></p>"""

    @app.get("/", response_class=HTMLResponse)
    def home():
        serials = store.serials()
        items = "".join(f'<li><a href="/models/{s}">Register No. {s}</a></li>' for s in serials[-50:][::-1])
        return f"""<!doctype html><meta charset="utf-8"><title>Guild Register</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px}}</style>
<h1>The Guild Register</h1><p>The public ledger of registered fine-tuned models: what each was trained on, as its
artist attested, countersigned by the register and never edited.</p><p>{len(serials)} models registered.</p>
<ul>{items}</ul><p><a href="/api/ledger">The ledger</a> · <a href="/api/public-key">The register's public key</a></p>"""

    @app.get("/health")
    def health():
        return JSONResponse({"ok": True, "ledger_ok": store.ledger_ok(), "models": len(store.serials())})

    return app
