"""Catalogue metadata — the tempo, key and time signature an adapter was
trained with, and the defaults a generation should start from.

Shared by the Workshop (which holds the per-track metadata the pipeline
extracted) and the Listening Space (which receives a snapshot of it in
the artist bundle at import). Both compute the same defaults from the
same document shape, so a blank tempo or key field means "the artist's
usual" in both apps — the adapter was conditioned on these values at
training; "model decides" lets the base model pick from its whole
distribution instead.

Document (``catalogue_metadata.json``)::

    {"schema_version": "1.0.0", "artist_id": "…",
     "tracks": [{"track_id", "title", "duration", "bpm", "bpm_confidence",
                 "octave_resolved", "keyscale", "key_confidence", "timesignature"}]}

``keyscale`` is in the apps' picker form (``"A minor"``, sharps only);
``timesignature`` is beats per bar as a string (``"4"``).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from khaos_attribution.prompt_expansion import normalise_keyscale, normalise_timesignature

CATALOGUE_METADATA_VERSION = "1.0.0"
CATALOGUE_METADATA_FILENAME = "catalogue_metadata.json"

# Tracks whose tempo or key the extractor was unsure about — or never
# rated — are left out of the defaults: a wrong octave or a guessed key
# would steer every blank prompt. 0.5 is the extractor's "coin flip" level.
MIN_CONFIDENCE = 0.5
BPM_RANGE = (40, 240)
# Perceived-tempo band. Beat trackers report octave errors freely (the
# catalogue this was built on had 41 of 77 tracks under 80 bpm, many of
# them marked 2/4 — half-tempo readings of 4/4 songs). Conditioning wants
# the tempo a listener would tap, so tempos are folded into this band
# before the median: doubled below it, halved above it.
PERCEIVED_BPM = (70, 160)


def track_row(track_id: str, title: str | None, metadata: dict | None,
              duration: float | None = None) -> dict:
    """One document row from a Workshop ``metadata/<track>.json`` payload
    (tempo/key/time_signature dicts as the metadata stage writes them).
    Missing pieces become None; nothing is guessed."""
    md = metadata or {}
    tempo = md.get("tempo") or {}
    key = md.get("key") or {}
    ts = md.get("time_signature") or {}
    keyscale = None
    if key.get("tonic") and key.get("mode"):
        keyscale = normalise_keyscale(f"{key['tonic']} {key['mode']}")
    return {
        "track_id": track_id,
        "title": title,
        "duration": float(duration) if duration is not None else None,
        "bpm": _number(tempo.get("bpm")),
        "bpm_confidence": _number(tempo.get("confidence")),
        # The metadata stage (0.2.0+) records whether it resolved an octave
        # disagreement against the downbeat grid; such a tempo is final.
        "octave_resolved": tempo.get("octave_resolved") is True,
        "keyscale": keyscale,
        "key_confidence": _number(key.get("confidence")),
        "timesignature": normalise_timesignature(ts.get("numerator")),
    }


def build_document(artist_id: str, rows: list[dict]) -> dict:
    return {"schema_version": CATALOGUE_METADATA_VERSION, "artist_id": artist_id,
            "tracks": [r for r in rows if r.get("track_id")]}


def catalogue_defaults(document: dict | None) -> dict:
    """``{"bpm", "keyscale", "timesignature", "tracks_used"}`` — each None when
    the catalogue gives no confident answer. Tempo is the duration-weighted
    median (a long track counts for more of what the adapter heard); key
    and time signature are the most common confident values."""
    out = {"bpm": None, "keyscale": None, "timesignature": None, "tracks_used": 0}
    raw = (document or {}).get("tracks") if isinstance(document, dict) else None
    # A hand-edited or torn file must degrade to "no defaults", never
    # break the server that reads it: rows are coerced, not trusted.
    tracks = [_clean_row(t) for t in (raw or []) if isinstance(t, dict) and t.get("track_id")]
    if not tracks:
        return out
    lo, hi = BPM_RANGE
    tempo_rows = [t for t in tracks
                  if t["bpm"] is not None and lo <= t["bpm"] <= hi
                  and t["bpm_confidence"] is not None and t["bpm_confidence"] >= MIN_CONFIDENCE]
    if tempo_rows:
        weighted: list[float] = []
        for t in tempo_rows:
            weight = max(1, int(round((t["duration"] or 60.0) / 30.0)))
            # A tempo the extractor already resolved against the downbeat
            # grid is trusted as read; only unresolved readings are folded.
            bpm = t["bpm"] if t["octave_resolved"] else fold_tempo(t["bpm"])
            weighted.extend([bpm] * weight)
        out["bpm"] = int(round(statistics.median(weighted)))
    key_rows = [t["keyscale"] for t in tracks if t["keyscale"]
                and t["key_confidence"] is not None and t["key_confidence"] >= MIN_CONFIDENCE]
    if key_rows:
        out["keyscale"] = Counter(key_rows).most_common(1)[0][0]
    ts_rows = [t["timesignature"] for t in tracks if t["timesignature"]]
    if ts_rows:
        out["timesignature"] = Counter(ts_rows).most_common(1)[0][0]
    out["tracks_used"] = len({t["track_id"] for t in tempo_rows}
                             | {t["track_id"] for t in tracks if t["keyscale"]
                                and t["key_confidence"] is not None
                                and t["key_confidence"] >= MIN_CONFIDENCE})
    return out


def _clean_row(t: dict) -> dict:
    """A row with every field in the type the maths expects (strings and
    junk become None; the key and time signature re-normalised)."""
    return {
        "track_id": str(t.get("track_id")),
        "duration": _number(t.get("duration")),
        "bpm": _number(t.get("bpm")),
        "bpm_confidence": _number(t.get("bpm_confidence")),
        "keyscale": normalise_keyscale(t.get("keyscale")),
        "key_confidence": _number(t.get("key_confidence")),
        "timesignature": normalise_timesignature(t.get("timesignature")),
        "octave_resolved": t.get("octave_resolved") is True,
    }


def fold_tempo(bpm: float) -> float:
    """``bpm`` folded by octaves into the perceived band (doubled while
    below it, halved while above it). 60 → 120; 180 → 90; 100 → 100."""
    import math  # noqa: PLC0415
    lo, hi = PERCEIVED_BPM
    if not math.isfinite(bpm) or bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_TITLE_EXT = re.compile(r"\.[A-Za-z0-9]{2,5}$")


def title_from_filename(filename: str | None) -> str | None:
    """A display title from an original filename (``"Band - Song.wav"`` →
    ``"Band - Song"``); None for nothing."""
    if not filename:
        return None
    return _TITLE_EXT.sub("", str(filename)).strip() or None
