"""The register's store: accounts, invites, enrolled machines, records, where
they are served, and the ledger. SQLite, one writer, every change a ledger row.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from khaos_attribution.registry import FIRST_SERIAL, canonical_bytes, sha256_hex

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invites (code TEXT PRIMARY KEY, account_id INTEGER NOT NULL, created_at TEXT NOT NULL,
  used_by TEXT, used_at TEXT);
CREATE TABLE IF NOT EXISTS machines (key_id TEXT PRIMARY KEY, account_id INTEGER NOT NULL, name TEXT NOT NULL,
  public_pem TEXT NOT NULL, enrolled_at TEXT NOT NULL, revoked_at TEXT);
CREATE TABLE IF NOT EXISTS records (serial INTEGER PRIMARY KEY, record_sha256 TEXT UNIQUE NOT NULL,
  machine_key_id TEXT NOT NULL, account_id INTEGER NOT NULL, signed_json TEXT NOT NULL, counter_json TEXT NOT NULL,
  state TEXT NOT NULL, published_at TEXT NOT NULL, withdrawn_at TEXT, withdraw_reason TEXT);
CREATE TABLE IF NOT EXISTS served (serial INTEGER NOT NULL, space_url TEXT NOT NULL, state TEXT NOT NULL,
  at TEXT NOT NULL, key_id TEXT NOT NULL, PRIMARY KEY (serial, space_url));
CREATE TABLE IF NOT EXISTS ledger (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, serial INTEGER, record_sha256 TEXT,
  key_id TEXT, at TEXT NOT NULL, detail_json TEXT NOT NULL, entry_sha256 TEXT NOT NULL, prev_sha256 TEXT);
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # ---- accounts and invites (the operator's, by command line) ----

    def create_account(self, name: str) -> int:
        with self._lock, self.db:
            cur = self.db.execute("INSERT INTO accounts (name, created_at) VALUES (?, ?)", (name, now_utc()))
            self._ledger("account", None, None, None, {"account": name})
            return cur.lastrowid

    def create_invite(self, account_name: str) -> str:
        with self._lock, self.db:
            row = self.db.execute("SELECT id FROM accounts WHERE name = ?", (account_name,)).fetchone()
            if not row:
                raise KeyError(f"no account {account_name!r}")
            code = secrets.token_urlsafe(18)
            self.db.execute("INSERT INTO invites (code, account_id, created_at) VALUES (?, ?, ?)", (code, row["id"], now_utc()))
            return code

    # ---- machines ----

    def enrol(self, invite: str, key_id: str, name: str, public_pem: str) -> dict:
        with self._lock, self.db:
            inv = self.db.execute("SELECT * FROM invites WHERE code = ?", (invite,)).fetchone()
            if not inv or inv["used_by"]:
                raise PermissionError("the invite is unknown or already used")
            if self.db.execute("SELECT 1 FROM machines WHERE key_id = ?", (key_id,)).fetchone():
                raise PermissionError("this machine key is already enrolled")
            at = now_utc()
            self.db.execute("INSERT INTO machines (key_id, account_id, name, public_pem, enrolled_at) VALUES (?, ?, ?, ?, ?)",
                            (key_id, inv["account_id"], name, public_pem, at))
            self.db.execute("UPDATE invites SET used_by = ?, used_at = ? WHERE code = ?", (key_id, at, invite))
            account = self.db.execute("SELECT name FROM accounts WHERE id = ?", (inv["account_id"],)).fetchone()["name"]
            self._ledger("enrol", None, None, key_id, {"machine": name, "account": account})
            return {"machine_key_id": key_id, "account": account, "enrolled_at": at}

    def machine(self, key_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM machines WHERE key_id = ? AND revoked_at IS NULL", (key_id,)).fetchone()
        return dict(row) if row else None

    def revoke_machine(self, key_id: str, reason: str) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE machines SET revoked_at = ? WHERE key_id = ?", (now_utc(), key_id))
            self._ledger("revoke", None, None, key_id, {"reason": reason})

    # ---- records ----

    def next_serial(self) -> int:
        row = self.db.execute("SELECT MAX(serial) AS m FROM records").fetchone()
        return max(FIRST_SERIAL, (row["m"] or 0) + 1)

    def existing(self, record_sha256: str) -> dict | None:
        row = self.db.execute("SELECT * FROM records WHERE record_sha256 = ?", (record_sha256,)).fetchone()
        return dict(row) if row else None

    def insert_record(self, serial: int, signed: dict, counter: dict, machine_key_id: str, account_id: int) -> None:
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO records (serial, record_sha256, machine_key_id, account_id, signed_json, counter_json, state, published_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'registered', ?)",
                (serial, signed["record_sha256"], machine_key_id, account_id, json.dumps(signed, sort_keys=True),
                 json.dumps(counter, sort_keys=True), counter["countersigned_at_utc"]))
            self._ledger("publish", serial, signed["record_sha256"], machine_key_id,
                         {"artist_id": signed["record"]["artist_id"], "public_view": signed["record"]["public_view"]})

    def record(self, serial: int) -> dict | None:
        row = self.db.execute("SELECT * FROM records WHERE serial = ?", (serial,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["signed"] = json.loads(out.pop("signed_json"))
        out["countersignature"] = json.loads(out.pop("counter_json"))
        out["served"] = [dict(r) for r in self.db.execute("SELECT * FROM served WHERE serial = ? ORDER BY at", (serial,))]
        return out

    def withdraw(self, serial: int, key_id: str, reason: str) -> dict:
        with self._lock, self.db:
            row = self.db.execute("SELECT * FROM records WHERE serial = ?", (serial,)).fetchone()
            if not row:
                raise KeyError(f"no record {serial}")
            mach = self.machine(key_id)
            if not mach or mach["account_id"] != row["account_id"]:
                raise PermissionError("only a machine of the account that published a model may withdraw it")
            if row["state"] == "withdrawn":
                return {"serial": serial, "state": "withdrawn", "withdrawn_at": row["withdrawn_at"]}
            at = now_utc()
            self.db.execute("UPDATE records SET state = 'withdrawn', withdrawn_at = ?, withdraw_reason = ? WHERE serial = ?",
                            (at, reason[:400], serial))
            self._ledger("withdraw", serial, row["record_sha256"], key_id, {"reason": reason[:400]})
            return {"serial": serial, "state": "withdrawn", "withdrawn_at": at}

    def served(self, serial: int, key_id: str, space_url: str, state: str) -> dict:
        with self._lock, self.db:
            row = self.db.execute("SELECT record_sha256 FROM records WHERE serial = ?", (serial,)).fetchone()
            if not row:
                raise KeyError(f"no record {serial}")
            at = now_utc()
            self.db.execute("INSERT INTO served (serial, space_url, state, at, key_id) VALUES (?, ?, ?, ?, ?)"
                            " ON CONFLICT(serial, space_url) DO UPDATE SET state = excluded.state, at = excluded.at, key_id = excluded.key_id",
                            (serial, space_url, state, at, key_id))
            self._ledger("served", serial, row["record_sha256"], key_id, {"space_url": space_url, "state": state})
            return {"serial": serial, "space_url": space_url, "state": state, "at": at}

    def withdrawn_serials(self) -> list:
        return [r["serial"] for r in self.db.execute("SELECT serial FROM records WHERE state = 'withdrawn' ORDER BY serial")]

    def serials(self) -> list:
        return [r["serial"] for r in self.db.execute("SELECT serial FROM records ORDER BY serial")]

    # ---- the ledger ----

    def _ledger(self, kind: str, serial, record_sha256, key_id, detail: dict) -> None:
        prev = self.db.execute("SELECT entry_sha256 FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        prev_hash = prev["entry_sha256"] if prev else None
        at = now_utc()
        body = {"kind": kind, "serial": serial, "record_sha256": record_sha256, "key_id": key_id, "at": at,
                "detail": detail, "prev_sha256": prev_hash}
        entry_hash = sha256_hex(canonical_bytes(body))
        self.db.execute("INSERT INTO ledger (kind, serial, record_sha256, key_id, at, detail_json, entry_sha256, prev_sha256)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (kind, serial, record_sha256, key_id, at, json.dumps(detail, sort_keys=True), entry_hash, prev_hash))

    def ledger(self, since_seq: int = 0) -> list:
        rows = self.db.execute("SELECT * FROM ledger WHERE seq > ? ORDER BY seq", (since_seq,))
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json"))
            out.append(d)
        return out

    def ledger_ok(self) -> bool:
        """Every entry's hash is over its own body and names the entry before it: the chain is unbroken."""
        prev = None
        for e in self.ledger():
            body = {"kind": e["kind"], "serial": e["serial"], "record_sha256": e["record_sha256"], "key_id": e["key_id"],
                    "at": e["at"], "detail": e["detail"], "prev_sha256": prev}
            if e["prev_sha256"] != prev or e["entry_sha256"] != sha256_hex(canonical_bytes(body)):
                return False
            prev = e["entry_sha256"]
        return True
