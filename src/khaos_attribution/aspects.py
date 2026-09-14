"""Per-aspect influence — one share per musical channel, not one per track.

Three channels (after Aria, arXiv:2605.16181): harmony is key agreement
(same or related key), rhythm is tempo proximity in log2 octaves plus metre,
timbre is the acoustic similarity the estimator already computes. This is
descriptor agreement, not a causal decomposition, and must never be
presented as one. `aspect_shares` returns shares that sum to one across
tracks within a channel, never across channels; the agreement functions
return agreements in [0, 1]. Pure: the caller loads the documents.
"""

from __future__ import annotations

import math

# Guards, not calibrations: nothing here has been measured against a
# listening test or a musicologist.
RELATED_KEY_WEIGHT = 0.5      # a relative or dominant key: related, not the same
TEMPO_OCTAVE_TOLERANCE = 1.0  # log2 distance at which tempo similarity reaches zero
# One octave, so a half-time or double-time reading lands exactly at zero.
METRE_WEIGHT = 0.25           # a shared time signature is a small part of rhythm

_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
               "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


def _mode_is_minor(mode: str | None) -> bool | None:
    """True minor, False major, None when the record does not say."""
    text = str(mode or "").strip().lower()
    if not text:
        return None
    if text.startswith("min") or text in {"m", "aeolian", "dorian", "phrygian"}:
        return True
    if text.startswith("maj") or text in {"ionian", "lydian", "mixolydian"}:
        return False
    return None


def _pitch_class(tonic: str | None) -> int | None:
    if not tonic:
        return None
    name = str(tonic).strip().capitalize().replace("♯", "#").replace("♭", "b")
    name = _ENHARMONIC.get(name, name)
    return _PITCHES.index(name) if name in _PITCHES else None


def key_agreement(a: dict | None, b: dict | None) -> float | None:
    """1.0 same key, RELATED_KEY_WEIGHT for a relative or dominant, else 0.

    None when either key is unknown — an unmeasured track must not read as
    a disagreement, which would quietly push its share to zero.
    """
    if not a or not b:
        return None
    pa, pb = _pitch_class(a.get("tonic")), _pitch_class(b.get("tonic"))
    if pa is None or pb is None:
        return None
    ma, mb = _mode_is_minor(a.get("mode")), _mode_is_minor(b.get("mode"))
    if ma is None or mb is None:
        return None  # an absent mode must not read as major
    if pa == pb and ma == mb:
        return 1.0
    # Relative major/minor; the direction matters (the relative major sits
    # three semitones above the minor tonic, and a symmetric test would
    # accept A minor against F# major).
    if ma and not mb and (pa + 3) % 12 == pb:
        return RELATED_KEY_WEIGHT
    if mb and not ma and (pb + 3) % 12 == pa:
        return RELATED_KEY_WEIGHT
    # Dominant / subdominant, same mode: a fifth either way.
    if ma == mb and (pa - pb) % 12 in (5, 7):
        return RELATED_KEY_WEIGHT
    return 0.0


def tempo_agreement(bpm_a: float | None, bpm_b: float | None) -> float | None:
    """1.0 at the same tempo, falling to 0 at TEMPO_OCTAVE_TOLERANCE octaves.

    Distance is measured in log2, so a half-time or double-time reading is
    exactly one octave away rather than an enormous linear gap.
    """
    if not bpm_a or not bpm_b or bpm_a <= 0 or bpm_b <= 0:
        return None
    octaves = abs(math.log2(float(bpm_a) / float(bpm_b)))
    return max(0.0, 1.0 - octaves / TEMPO_OCTAVE_TOLERANCE)


def metre_agreement(ts_a: dict | None, ts_b: dict | None) -> float | None:
    if not ts_a or not ts_b:
        return None
    na, nb = ts_a.get("numerator"), ts_b.get("numerator")
    if not na or not nb:
        return None
    return 1.0 if int(na) == int(nb) else 0.0


def rhythm_agreement(a: dict | None, b: dict | None) -> float | None:
    """Tempo, mostly, with metre as a small correction."""
    tempo = tempo_agreement((a or {}).get("bpm"), (b or {}).get("bpm"))
    metre = metre_agreement((a or {}).get("time_signature"), (b or {}).get("time_signature"))
    if tempo is None and metre is None:
        return None
    if tempo is None:
        # Metre alone must not read as all of rhythm: 4/4 is near-universal.
        return METRE_WEIGHT * metre
    if metre is None:
        return tempo
    return (1 - METRE_WEIGHT) * tempo + METRE_WEIGHT * metre


def _shares(raw: dict[str, float | None]) -> dict[str, float] | None:
    """Agreements to shares that sum to 1. None when nothing is measurable.

    A track whose descriptor is unknown is dropped from the channel rather
    than scored zero; the caller reports the coverage.
    """
    live = {t: v for t, v in raw.items() if v is not None}
    if not live:
        return None
    # Clipped at zero before normalising: raw cosines are signed, and a signed
    # sum produces shares over 100% and negative shares.
    clipped = {t: max(0.0, float(v)) for t, v in live.items()}
    total = sum(clipped.values())
    if total <= 0:
        return None
    return {t: v / total for t, v in clipped.items()}


def aspect_shares(output: dict, tracks: dict[str, dict],
                  timbre_scores: dict[str, float] | None = None,
                  timbre_weights: dict[str, float] | None = None) -> dict:
    """Per-channel influence shares over the training tracks.

    `output` and each entry of `tracks` are metadata-shaped: `{"bpm", "key":
    {"tonic","mode"}, "time_signature": {"numerator"}}`. `timbre_weights` is
    the estimator's own similarity share, passed in so the timbre channel is
    the same number the blend uses; `timbre_scores` (raw cosines) is a
    fallback, and the result says which was used. A channel with no
    measurable tracks is present and null, never absent.
    """
    harmony = {t: key_agreement(output.get("key"), m.get("key")) for t, m in tracks.items()}
    rhythm = {t: rhythm_agreement(output, m) for t, m in tracks.items()}

    if timbre_weights:
        timbre, timbre_source = dict(timbre_weights), "the estimator's own similarity share (its softmax weight), unchanged"
    else:
        timbre = {t: (timbre_scores or {}).get(t) for t in tracks}
        timbre_source = ("raw CLAP cosines, clipped at zero and normalised linearly — NOT the "
                         "blend's softmax share, so this channel and similarity_share_pct differ")

    out = {}
    for name, raw, source in (
        ("harmony", harmony, "key agreement (tonic and mode, relative and dominant related)"),
        ("rhythm", rhythm, "tempo proximity in log2 octaves, with metre"),
        ("timbre", timbre, timbre_source),
    ):
        measured = [t for t, v in raw.items() if v is not None]
        shares = _shares(raw)
        out[name] = {
            "shares_pct": ({t: round(v * 100, 4) for t, v in shares.items()} if shares else None),
            "measured_tracks": len(measured),
            # Nothing measurable is distinct from measured-and-all-disagreeing.
            "unmeasurable": len(measured) == 0,
            "of_tracks": len(tracks),
            "source": source,
        }
    out["caveat"] = ("Per-aspect shares are descriptor agreement, not causal influence. "
                     "They say which musical channel a resemblance ran through, not what "
                     "the model learned from whom.")
    return out
