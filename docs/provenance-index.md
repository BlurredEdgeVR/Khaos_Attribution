# The provenance index — schema and rebuild rules (watermarking v2 §4)

A **derived, rebuildable SQLite index**. Canonical truth stays in plain
files (sidecars, model cards, rights.json, tombstone records); the index
is destroyed and rebuilt without loss, and no writer ever treats it as a
place to record something that exists nowhere else.

Location: the Platform's data root, `provenance_index.sqlite3` (it serves
/verify). The Engine Room may open it read-only. Never synced between
machines — each machine rebuilds its own from the files that do sync.

## DDL (index_schema_version 1)

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- keys: index_schema_version, built_at_utc, cursor_mtime

CREATE TABLE models (
  watermark_id INTEGER PRIMARY KEY,     -- the run codeword
  run_id TEXT NOT NULL,
  artist_id TEXT NOT NULL,
  adapter_hash TEXT,
  artist_name TEXT,
  consent_statement TEXT,               -- snapshot for display; card is truth
  card_path TEXT NOT NULL
);

CREATE TABLE outputs (
  generation_id TEXT PRIMARY KEY,
  watermark_id INTEGER,                 -- run codeword (v2) or legacy codeword
  artist_id TEXT NOT NULL,
  run_id TEXT,
  audio_path TEXT NOT NULL,
  sidecar_path TEXT NOT NULL,
  created_at_utc TEXT,
  home TEXT NOT NULL                    -- 'platform' | 'workshop'
);

CREATE TABLE legacy_ids (
  watermark_id INTEGER PRIMARY KEY,     -- pre-v2 per-output codeword
  generation_id TEXT                    -- NULL when the frozen list knows the
                                        -- ID but no sidecar survives locally
);

CREATE TABLE fingerprints (
  hash INTEGER NOT NULL,                -- landmark hash (fingerprint_version v1)
  generation_id TEXT NOT NULL,
  kind TEXT NOT NULL,                   -- 'mix' | stem name
  offset_ms INTEGER NOT NULL
);
CREATE INDEX fingerprints_hash ON fingerprints (hash);

CREATE TABLE tombstones (
  generation_id TEXT PRIMARY KEY,
  artist_id TEXT NOT NULL,
  run_id TEXT,
  watermark_id INTEGER,
  content_sha256 TEXT NOT NULL,
  deleted_at_utc TEXT NOT NULL
);
-- tombstone fingerprints go into `fingerprints` with kind 'tombstone'.
```

## Rebuild and staleness

- On open: `index_schema_version` mismatch → rebuild from scratch.
- Incremental passes scan for files newer than `cursor_mtime`; the
  builder reads only atomically-written files and skips torn ones (they
  surface next pass).
- Fingerprinting runs as a queued post-generation step, never a bulk scan
  on the serving hot path.
- /verify never blocks on a rebuild: it reads the current index and,
  when the cursor lags the newest sidecar, says the exact-output layer is
  catching up instead of silently answering UNKNOWN.

## Fingerprint format

`fingerprint_version` "v1" is defined at phase 5 (docs/watermarking-v2.md
§10) and documented here when it lands; the `fingerprints` table shape
above is stable regardless (hash + location + offset).
