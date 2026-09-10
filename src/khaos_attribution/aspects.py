"""Per-aspect influence — one share per musical channel, not one per track.

Aria (arXiv:2605.16181) argues that a single scalar per training work cannot
support copyright analysis, because infringement is assessed aspect by
aspect: the idea-and-expression distinction asks what was taken, not how
much. A lawyer reasons about a melody, a chord movement, a groove, a sound.
One number answers none of those questions.

Khaos is unusually well placed to answer them cheaply, because the analysis
pipeline has already produced the material. Every track carries a tempo, a
key, a time signature and a section structure, and every track has been
separated into four stems. Nothing here needs a new model or a new download.

THREE CHANNELS, and what each is actually made of:

  harmony   key agreement — same tonic and mode, or a related key (the
            relative major/minor and the dominant, which share most of
            their notes). Coarse: two tracks in A minor are not thereby
            harmonically alike, they are merely not disqualified.
  rhythm    tempo proximity on a log scale, so 90 and 180 BPM are one
            octave apart rather than 90 units apart, plus metre agreement.
  timbre    the acoustic similarity the estimator already computes (CLAP).
            This is the channel CLAP is actually good at; using it for the
            other two is what the single-scalar critique objects to.

WHAT THIS IS NOT. It is not a causal decomposition and must never be
presented as one. Aria's own limitation applies here twice over: it measures
whether retrieved tracks resemble EACH OTHER on a channel, has no per-aspect
ground truth, and says establishing per-track causal influence "remains
future work". Ours is coarser still, being descriptor agreement rather than
a learned representation. Its value is EXPLANATORY: it lets an estimate say
which musical channel drove a resemblance, which is the question a rights
determination actually turns on. Every function here returns a share that
sums to one across tracks WITHIN a channel, never across channels.

Pure: numbers and dicts in, numbers out. The caller loads the documents.
"""

from __future__ import annotations

import math

# Guards, not calibrations. Nothing here has been measured against a
# listening test or a musicologist; they are the coarsest defensible
# readings of the descriptors already on disk.
RELATED_KEY_WEIGHT = 0.5      # a relative or dominant key: related, not the same
TEMPO_OCTAVE_TOLERANCE = 0.5  # log2 distance at which tempo similarity reaches zero
METRE_WEIGHT = 0.25           # a shared time signature is a small part of rhythm

_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
               "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


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
    ma = str(a.get("mode") or "").lower().startswith("min")
    mb = str(b.get("mode") or "").lower().startswith("min")
    if pa == pb and ma == mb:
        return 1.0
    # Relative major/minor, and the direction matters: the relative MAJOR
    # sits three semitones above the minor tonic (A minor -> C major). The
    # symmetric test (either three or nine semitones apart, opposite mode)
    # also accepts A minor against F# major, which are not relatives at all
    # — they share four notes, not seven.
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
    exactly one octave away rather than an enormous linear gap — the octave
    ambiguity is real in this pipeline and the metadata stage resolves it
    from downbeat spacing, but a resolved value can still be an octave out.
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
        return metre
    if metre is None:
        return tempo
    return (1 - METRE_WEIGHT) * tempo + METRE_WEIGHT * metre


def _shares(raw: dict[str, float | None]) -> dict[str, float] | None:
    """Agreements to shares that sum to 1. None when nothing is measurable.

    A track whose descriptor is unknown is DROPPED from the channel rather
    than scored zero, and the caller reports the coverage — a channel
    computed over three of forty tracks is not a channel, and saying so is
    the point.
    """
    live = {t: v for t, v in raw.items() if v is not None}
    total = sum(live.values())
    if not live or total <= 0:
        return None
    return {t: v / total for t, v in live.items()}


def aspect_shares(output: dict, tracks: dict[str, dict],
                  timbre_scores: dict[str, float] | None = None) -> dict:
    """Per-channel influence shares over the training tracks.

    `output` and each entry of `tracks` are metadata-shaped: `{"bpm", "key":
    {"tonic","mode"}, "time_signature": {"numerator"}}`. `timbre_scores` is
    the estimator's own per-track acoustic similarity, passed in rather than
    recomputed so the timbre channel is the SAME number the blend uses and
    the two can never quietly diverge.

    Every channel reports its own coverage. A channel with no measurable
    tracks is present and null, never absent — a reader must be able to tell
    "we looked and could not say" from "we did not look".
    """
    harmony = {t: key_agreement(output.get("key"), m.get("key")) for t, m in tracks.items()}
    rhythm = {t: rhythm_agreement(output, m) for t, m in tracks.items()}
    timbre = {t: (timbre_scores or {}).get(t) for t in tracks}

    out = {}
    for name, raw, source in (
        ("harmony", harmony, "key agreement (tonic and mode, relative and dominant related)"),
        ("rhythm", rhythm, "tempo proximity in log2 octaves, with metre"),
        ("timbre", timbre, "the estimator's own CLAP similarity, unchanged"),
    ):
        measured = [t for t, v in raw.items() if v is not None]
        out[name] = {
            "shares_pct": ({t: round(v * 100, 4) for t, v in _shares(raw).items()}
                           if _shares(raw) else None),
            "measured_tracks": len(measured),
            "of_tracks": len(tracks),
            "source": source,
        }
    out["caveat"] = ("Per-aspect shares are descriptor agreement, not causal influence. "
                     "They say which musical channel a resemblance ran through, not what "
                     "the model learned from whom.")
    return out
