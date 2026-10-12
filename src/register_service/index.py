"""The fingerprint index beside the register: released outputs under their model's watermark bucket, the
verify that names one of them, and the training tracks each registered model claims. Fingerprints are
landmark hashes, never audio.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from khaos_attribution.fingerprint import best_phase_stats, dominant, is_confident
from register_service.store import now_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS outputs (generation_id TEXT PRIMARY KEY, serial INTEGER NOT NULL, payload INTEGER NOT NULL,
  artist_id TEXT NOT NULL, released_at TEXT NOT NULL, released_by TEXT NOT NULL, kind TEXT NOT NULL, landmarks INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fingerprints (hash INTEGER NOT NULL, generation_id TEXT NOT NULL, offset_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS fingerprints_hash ON fingerprints (hash);
CREATE INDEX IF NOT EXISTS fingerprints_gen ON fingerprints (generation_id);
CREATE INDEX IF NOT EXISTS outputs_payload ON outputs (payload);
CREATE TABLE IF NOT EXISTS claims (claim_id INTEGER PRIMARY KEY, account_id INTEGER NOT NULL, serial INTEGER NOT NULL,
  track_id TEXT NOT NULL, declaration TEXT NOT NULL, claimed_at TEXT NOT NULL, claimed_by TEXT NOT NULL,
  landmarks INTEGER NOT NULL, UNIQUE (serial, track_id));
CREATE TABLE IF NOT EXISTS claim_fingerprints (hash INTEGER NOT NULL, claim_id INTEGER NOT NULL, offset_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS claim_fingerprints_hash ON claim_fingerprints (hash);
CREATE INDEX IF NOT EXISTS claim_fingerprints_claim ON claim_fingerprints (claim_id);
CREATE TABLE IF NOT EXISTS claim_conflicts (at TEXT NOT NULL, account_id INTEGER NOT NULL, serial INTEGER NOT NULL,
  track_id TEXT NOT NULL, held_by_claim INTEGER NOT NULL, votes INTEGER NOT NULL);
"""
MAX_LANDMARKS = 200_000
CANDIDATES = 50


class Index:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def release(self, generation_id: str, serial: int, payload: int, artist_id: str, fingerprint: list,
                released_by: str, kind: str) -> dict:
        """One released output under its model's bucket; the same id again is the first release, unchanged."""
        rows = [(h, generation_id, o) for h, o in _landmarks(fingerprint)]
        with self._lock:
            have = self.db.execute("SELECT * FROM outputs WHERE generation_id = ?", (generation_id,)).fetchone()
            if have:
                return {**dict(have), "already": True}
            self.db.execute("BEGIN IMMEDIATE")
            try:
                at = now_utc()
                self.db.execute("INSERT INTO outputs (generation_id, serial, payload, artist_id, released_at, released_by, kind, landmarks)"
                                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (generation_id, serial, payload, artist_id, at, released_by, kind, len(rows)))
                self.db.executemany("INSERT INTO fingerprints (hash, generation_id, offset_ms) VALUES (?, ?, ?)", rows)
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return {"generation_id": generation_id, "serial": serial, "payload": payload, "artist_id": artist_id,
                "released_at": at, "released_by": released_by, "kind": kind, "landmarks": len(rows), "already": False}

    def output(self, generation_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM outputs WHERE generation_id = ?", (generation_id,)).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM outputs").fetchone()[0])

    def _candidates(self, phases: list, payload: int | None) -> dict:
        """The outputs sharing hashes with the query, by generation id, with their stored landmarks; the bucket first."""
        hashes = sorted({h for q in phases for h, _ in q})
        if not hashes:
            return {}
        counts: dict = {}
        for i in range(0, len(hashes), 900):
            chunk = hashes[i:i + 900]
            q = "SELECT f.generation_id AS g, COUNT(*) AS n FROM fingerprints f"
            params: list = list(chunk)
            if payload is not None:
                q += " JOIN outputs o ON o.generation_id = f.generation_id WHERE o.payload = ? AND f.hash IN (%s)" % ",".join("?" * len(chunk))
                params = [payload] + params
            else:
                q += " WHERE f.hash IN (%s)" % ",".join("?" * len(chunk))
            q += " GROUP BY f.generation_id"
            for row in self.db.execute(q, params):
                counts[row["g"]] = counts.get(row["g"], 0) + row["n"]
        top = sorted(counts, key=counts.get, reverse=True)[:CANDIDATES]
        out = {}
        for g in top:
            out[g] = [(r["hash"], r["offset_ms"]) for r in self.db.execute("SELECT hash, offset_ms FROM fingerprints WHERE generation_id = ?", (g,))]
        return out

    def verify(self, phases: list, payload: int | None) -> dict:
        """The three grades: a named output; a watermark alone narrowing to a bucket; nothing."""
        query_len = len(phases[0]) if phases else 0
        scored = []
        for g, fps in self._candidates(phases, payload).items():
            votes, distinct = best_phase_stats(phases, fps)
            scored.append((votes, distinct, g))
        scored.sort(reverse=True)
        if scored:
            votes, distinct, g = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0
            if is_confident(votes, query_len, distinct) and dominant(votes, runner):
                out = self.output(g) or {}
                return {"grade": "output", "output": out, "votes": votes, "runner_up": runner, "searched": "bucket" if payload is not None else "all"}
        if payload is not None:
            return {"grade": "bucket", "payload": payload, "searched": "bucket"}
        return {"grade": "none", "searched": "all"}

    def _claim_matches(self, phases: list) -> list:
        """Every claim the query confidently matches, strongest first: (votes, claim row). Confidence is read against
        the shorter of the two, so a short claimed track inside a long medley matches either way round."""
        hashes = sorted({h for q in phases for h, _ in q})
        counts: dict = {}
        for i in range(0, len(hashes), 900):
            chunk = hashes[i:i + 900]
            q = "SELECT claim_id AS c, COUNT(*) AS n FROM claim_fingerprints WHERE hash IN (%s) GROUP BY claim_id" % ",".join("?" * len(chunk))
            for row in self.db.execute(q, chunk):
                counts[row["c"]] = counts.get(row["c"], 0) + row["n"]
        query_len = len(phases[0]) if phases else 0
        out = []
        for c in sorted(counts, key=counts.get, reverse=True)[:CANDIDATES]:
            fps = [(r["hash"], r["offset_ms"]) for r in self.db.execute("SELECT hash, offset_ms FROM claim_fingerprints WHERE claim_id = ?", (c,))]
            votes, distinct = best_phase_stats(phases, fps)
            if is_confident(votes, min(query_len, len(fps)), distinct):
                out.append((votes, dict(self.db.execute("SELECT * FROM claims WHERE claim_id = ?", (c,)).fetchone())))
        out.sort(key=lambda m: m[0], reverse=True)
        return out

    def check_claim(self, phases: list, account_id: int) -> dict:
        """Whether a recording is already a registered model's training track (none, this account's, or another's),
        and the serial of the released output it is, when it is one."""
        matches = self._claim_matches(phases)
        output = self.verify(phases, None)
        released = int(output["output"]["serial"]) if output["grade"] == "output" else None
        others = [(v, c) for v, c in matches if c["account_id"] != account_id]
        if others:
            return {"grade": "other", "claim_ref": others[0][1]["claim_id"], "votes": others[0][0], "released_output": released}
        if matches:
            return {"grade": "mine", "serial": matches[0][1]["serial"], "track_id": matches[0][1]["track_id"], "released_output": released}
        return {"grade": "none", "released_output": released}

    def claim(self, account_id: int, serial: int, track_id: str, fingerprint: list, declaration: str, claimed_by: str,
              phases: list | None = None) -> dict:
        """A registered model's training track, claimed for its account; refused when another account holds it first.
        `phases`, the four-phase query of the same audio, is what the conflict test reads when given."""
        rows = _landmarks(fingerprint)
        query = [_landmarks(p) for p in phases] if phases else [rows]
        with self._lock:
            have = self.db.execute("SELECT * FROM claims WHERE serial = ? AND track_id = ?", (serial, track_id)).fetchone()
            if have:
                return {"claimed": True, "claim_ref": have["claim_id"], "already": True}
            others = [(v, c) for v, c in self._claim_matches(query) if c["account_id"] != account_id]
            if others:
                votes, held = others[0]
                self.db.execute("INSERT INTO claim_conflicts (at, account_id, serial, track_id, held_by_claim, votes) VALUES (?, ?, ?, ?, ?, ?)",
                                (now_utc(), account_id, serial, track_id, held["claim_id"], votes))
                return {"claimed": False, "conflict": {"claim_ref": held["claim_id"], "votes": votes}}
            self.db.execute("BEGIN IMMEDIATE")
            try:
                cur = self.db.execute("INSERT INTO claims (account_id, serial, track_id, declaration, claimed_at, claimed_by, landmarks)"
                                      " VALUES (?, ?, ?, ?, ?, ?, ?)", (account_id, serial, track_id, declaration, now_utc(), claimed_by, len(rows)))
                cid = cur.lastrowid
                self.db.executemany("INSERT INTO claim_fingerprints (hash, claim_id, offset_ms) VALUES (?, ?, ?)",
                                    [(h, cid, o) for h, o in rows])
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return {"claimed": True, "claim_ref": cid, "already": False}

    def release_claims(self, serial: int) -> int:
        """A withdrawn model's claims end with it: the recordings are free for the account that owns them."""
        with self._lock:
            ids = [r["claim_id"] for r in self.db.execute("SELECT claim_id FROM claims WHERE serial = ?", (serial,))]
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.executemany("DELETE FROM claim_fingerprints WHERE claim_id = ?", [(i,) for i in ids])
                self.db.execute("DELETE FROM claims WHERE serial = ?", (serial,))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return len(ids)

    def claims_of(self, serial: int) -> dict:
        """What a model's page says: how many of its training tracks are claimed, and how many were refused."""
        claimed = int(self.db.execute("SELECT COUNT(*) FROM claims WHERE serial = ?", (serial,)).fetchone()[0])
        refused = int(self.db.execute("SELECT COUNT(DISTINCT track_id) FROM claim_conflicts WHERE serial = ?", (serial,)).fetchone()[0])
        return {"claimed": claimed, "refused": refused}


def _landmarks(fingerprint) -> list:
    if not isinstance(fingerprint, list) or not fingerprint or len(fingerprint) > MAX_LANDMARKS:
        raise ValueError(f"a fingerprint is 1 to {MAX_LANDMARKS} landmarks")
    rows = []
    for item in fingerprint:
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in item)):
            raise ValueError("a landmark is [hash, offset_ms], both integers")
        rows.append((int(item[0]), int(item[1])))
    return rows
