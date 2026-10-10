"""The fingerprint index beside the register: released outputs under their model's watermark bucket, and the
verify that names one of them. Fingerprints are landmark hashes, never audio.
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
        if not isinstance(fingerprint, list) or not fingerprint or len(fingerprint) > MAX_LANDMARKS:
            raise ValueError(f"a fingerprint is 1 to {MAX_LANDMARKS} landmarks")
        rows = []
        for item in fingerprint:
            if not (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in item)):
                raise ValueError("a landmark is [hash, offset_ms], both integers")
            rows.append((int(item[0]), generation_id, int(item[1])))
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
