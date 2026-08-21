# The registry service — leased IDs, central fingerprints, one consent root

Plan, not commitment (2026-08-22). Code is the truth; this is the shape
to build towards once the Workshop release work starts (it is the same
"move trust out of the client" move as that plan).

## 1. Why

Today's model identity is an 11-bit watermark payload — **2048 codewords
in total** — allocated by each training machine *scanning the model cards
it can see* and taking a number from its own **band** (§3 of
`watermarking-v2.md`: laptop 32–703, studio 704–1375, threadripper
1376–2047). The bands exist only because the machines never talk. Three
consequences followed this week:

- the scan *is* the ledger, so removing an artist had to write a retired
  record beside the artists directory or its IDs would have been re-issued;
- a Listening Space can only `/verify` what it made itself — its
  provenance index is local;
- revocation is a file deletion on one machine; other Spaces learn nothing.

A central **registry service** owns the ledger, the fingerprint index and
the consent record. It does **not** replace filesystem handoff between
the apps, and it must never make an offline machine unable to finish a
training run — hence leases (§3).

## 2. What it holds

| table | rows | written by |
|---|---|---|
| `leases` | machine key, payload block (start, size), issued/expires | service |
| `ids` | codeword → state (leased / stamped / retired), run id, artist id, card hash, machine | Workshop (stamp), either app (retire) |
| `cards` | run id, artist id, adapter hash, card JSON, signature | Workshop at training completion |
| `consent` | artist id, statement hash, granted/withdrawn timestamps | Workshop (grant), Platform or Workshop (withdraw) |
| `fingerprints` | output id, artist id, run id, fingerprint rows (the contract's `fingerprint_array`), stems flag | Platform at store time; Workshop for auditions |
| `audit` | append-only log of every call | service |

Fingerprints are spectral-peak hashes, not audio. Nothing in the service
can reproduce a catalogue.

## 3. Leases, not live calls

Training completes offline. The allocator therefore never calls the
service at completion time; it takes from a **leased block**:

1. When online, a machine asks `POST /leases {machine, size}` and receives
   a contiguous block of payloads (default 32) it alone may stamp from.
   Blocks are accounted, expire (90 days) unless renewed, and are never
   re-leased while any ID in them is stamped.
2. `allocate_model_watermark_id` takes the next unused payload **from the
   lease file** (`<data root>/.watermark_lease.json`), falling back to the
   static band only when the machine has never held a lease — and says so
   in the run manifest (`watermark_source: lease | band`).
3. Everything the machine did while offline — cards, stamps, retirements,
   audition fingerprints — sits in an **outbox**
   (`<data root>/.registry_outbox/`) and syncs in order when online.
   Sync is idempotent: every record carries a content hash.

The static bands thus become the fallback of last resort; once every
machine has leased, the band table is retired from the contract.

## 4. The API (five calls the apps make)

| call | who | when |
|---|---|---|
| `POST /leases` | Workshop | online, when the local lease has < 8 free payloads or is near expiry |
| `POST /cards` | Workshop | training completion (via outbox); the service signs the card (Ed25519) and returns the signature the Platform's gate can check |
| `POST /retire` | both apps | artist removal; the service marks the IDs retired and records withdrawal in `consent` |
| `POST /fingerprints` | Platform (outputs), Workshop (auditions) | at store time (via outbox) |
| `POST /verify` | anyone, no key | a fingerprint array (and optional decoded codeword) → the same content-first answer `/verify` gives today, for any output any machine made |

Machines authenticate with a per-machine key issued at install; the
verify call is public and read-only. The service is small: FastAPI,
SQLite to start, Postgres when a second site exists.

## 5. What changes in the apps

- **Contract** (`khaos_attribution`): `registry.py` — lease file
  read/write, outbox append/sync, the request/response shapes, card
  signature verification. The static `MACHINE_BANDS` stay until §3 step 3.
- **Workshop**: the allocator prefers the lease; the outbox syncs on the
  System tab ("registry: 3 records waiting") and at startup when online;
  removal posts `retire`.
- **Platform**: the model-card gate also checks the service signature
  when one is present (unsigned cards stay servable until the cut-over
  date in the release plan); `/verify` consults the service when local
  resolution fails; withdrawal posts `retire` and a Space that learns of
  a withdrawal at startup refuses the bundle even if the files are still
  on disk.
- **Engine Room**: a registry-sync audit (outbox depth, lease headroom,
  unsigned cards) — UNKNOWN is not OK.

## 6. What it does not fix

The 2048 ceiling is the payload, not the partition. Central allocation
reclaims the slack of the bands; it does not make the space bigger. The
escape is the direction v2 already took — **content-first** resolution:
with a central fingerprint index, the fingerprint identifies the exact
output and the watermark only corroborates. When the space runs short,
the watermark can carry something coarse ("Khaos-made, issued in epoch
N") and identity lives in the index. That is a later decision; the
service makes it possible without re-embedding anything.

## 7. Phases

1. **Service + leases** (offline-safe): leases, `ids`, `retire`, outbox
   sync; the allocator prefers the lease. Bands remain the fallback.
2. **Cards and consent**: `POST /cards` with signatures; the Platform
   gate verifies when present; withdrawal propagates at startup.
3. **Fingerprints + public verify**: outputs and auditions register;
   `/verify` answers for any machine's output; the Public Listening
   Space's verify page points here.
4. **Retire the bands** once every machine has leased.

Each phase ships with the sign-off ritual; phase 1 is a few days of work
and is the one that removes this week's retired-IDs file from the critical
path.
