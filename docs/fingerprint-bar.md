# The fingerprint bar

Pre-registered 10 October 2026, before the measurement was run. The register
plan ("One register, many rooms", decision 3) names outputs by fingerprint,
and the index is built only if the fingerprint as it exists passes this bar.

## The question

Does the contract's fingerprint (`khaos_attribution.fingerprint`, format v1,
a landmark constellation) still identify a released output after what
released audio goes through, and does it refuse audio it has never seen?

## The index

Every rendered output on the measuring machine, fingerprinted from its
original WAV. Nothing is excluded for being short, quiet or repetitive; the
index is what the register would hold.

## The transforms (the bar set)

Re-encoding and clipping, as decided. Each output is pushed through each of:

| name | what |
|---|---|
| `pcm_baseline` | the WAV re-written as 16-bit PCM |
| `mp3_128k`, `mp3_320k` | MP3 at 128 and 320 kbit/s |
| `aac_128k` | AAC at 128 kbit/s |
| `opus_96k` | Opus at 96 kbit/s |
| `vorbis_q4` | Vorbis quality 4 (the game-audio middleware's music codec) |
| `resample_44k1` | resampled to 44.1 kHz |
| `loudnorm` | loudness-normalised to −14 LUFS |
| `clip_20s_mid`, `clip_10s_mid`, `clip_5s_mid` | an excerpt from the middle |
| `mp3_128k_clip_10s` | MP3 128 kbit/s, then a 10 s excerpt from the middle |

Pitch and tempo changes are not in the set: outputs altered that way get
the weaker verify answer ("made by a registered model, one of these").

## The decoys

The catalogue's own tracks (ingested originals that no output is), pushed
through the same transforms. None may match any output.

## What is counted

For each (output, transform): the query is matched against every entry in
the index with `match_stats`; a hit is the top candidate being the right
output AND `is_confident` AND dominance (top votes at least twice the
runner-up's), the closed-set rule the module itself states.

## The bar

1. **Identification: at least 95% of (output, transform) pairs are hits**,
   over the whole bar set, and **at least 90% for every transform on its
   own**. Clips of 5 seconds are the one allowed exception: they are
   reported but do not count against the bar, because a 5 s excerpt of
   repetitive material is the module's documented limitation.
2. **Refusal: no decoy, under any transform, is a hit against any output.**
   A single false match fails the bar.
3. Both numbers are reported with their counts, not rounded away.

## If it fails

The plan stops at phase 2. The two ways on are a decision, not a quiet
change: narrow the bar set (the failing transforms go to the weaker
answer), or replace the scheme. Either is recorded as an amendment to this
file with the numbers that forced it.

## Where

`scripts/fingerprint_survival.py` runs it and writes the record beside this
file as `fingerprint-bar.<machine>.json`. The first run is on the laptop
(M2 Max, 198 outputs, 35 catalogue tracks on 10 October 2026).

## Amendment 1 (10 October 2026, after the first run)

The first run indexed every WAV by path and 128 of 204 entries were copies:
the scorecard's base-model control renders the same audio for every
checkpoint, so 32 renders appeared four times each. Against itself a copy
ties its twin exactly and the dominance rule cannot choose, which is the
module's rule working as stated, not a fingerprint failure. The index is
one entry per distinct content (by file hash), which is what a register
would hold. The bar, the transforms and the decoys are unchanged. The
first run's numbers (792 of 2,244 before deduplication, no false matches)
are kept in git history; the record file is the rerun's.

## Amendment 2 (10 October 2026, after the second run)

Second run, one entry per content: 1,136 of 1,188 pairs (95.6%), no false
matches, every transform at or above 90% except the 20 s clip at 74 of 108.
The cause is not clip length but where a clip starts: the fingerprint lays
its analysis frames on a fixed 46.5 ms grid, and a clip that starts half a
hop off the grid keeps about a twentieth of its votes (147 against 2,322
for a start 20 ms later, on one output). The 20 s transform happened to
start exactly half a hop off; the 10 s transform happened to land near the
grid. A released clip starts anywhere, so the bar must hold at any phase.

The remedy is on the query side and leaves the stored format and every
index untouched: a query is fingerprinted at four sub-hop offsets and each
candidate is scored by its best phase (worst case on that output: 147 to
2,395 votes). The bar, the transforms and the decoys are unchanged; the
third run measures the four-phase matcher. If adopted, it is the matcher
the index service uses.
