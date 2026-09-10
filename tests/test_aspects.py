"""Per-aspect shares: the music theory has to be right, and the coverage honest.

From the 2026-09-10 prior-art review: Aria argues a single scalar per track
cannot support copyright analysis, because infringement is assessed aspect by
aspect. These channels are built from descriptors Khaos already stores.
"""
from __future__ import annotations

import pytest

from khaos_attribution.aspects import (METRE_WEIGHT, RELATED_KEY_WEIGHT, aspect_shares,
                                       key_agreement, metre_agreement,
                                       rhythm_agreement, tempo_agreement)


def _k(t, m):
    return {"tonic": t, "mode": m}


def test_the_same_key_is_one_and_the_tritone_is_nothing():
    assert key_agreement(_k("A", "minor"), _k("A", "minor")) == 1.0
    assert key_agreement(_k("C", "major"), _k("F#", "major")) == 0.0


def test_relative_major_and_minor_are_related_in_both_directions():
    assert key_agreement(_k("A", "minor"), _k("C", "major")) == RELATED_KEY_WEIGHT
    assert key_agreement(_k("C", "major"), _k("A", "minor")) == RELATED_KEY_WEIGHT


def test_a_key_three_semitones_away_the_WRONG_way_is_not_a_relative():
    """The first cut tested 'three or nine semitones apart, opposite mode',
    which accepts A minor against F# major. They share four notes, not
    seven, and calling them related would inflate a harmony share."""
    assert key_agreement(_k("A", "minor"), _k("F#", "major")) == 0.0
    assert key_agreement(_k("C", "major"), _k("D#", "minor")) == 0.0


def test_the_dominant_is_related_and_the_enharmonic_spelling_does_not_matter():
    assert key_agreement(_k("C", "major"), _k("G", "major")) == RELATED_KEY_WEIGHT
    assert key_agreement(_k("Db", "major"), _k("C#", "major")) == 1.0


def test_an_unknown_key_is_none_not_a_disagreement():
    """Scoring it zero would quietly push an unmeasured track's share to
    nothing, which reads as a finding about the track."""
    assert key_agreement(None, _k("A", "minor")) is None
    assert key_agreement(_k("", ""), _k("A", "minor")) is None
    assert key_agreement(_k("H", "major"), _k("A", "minor")) is None


def test_tempo_is_measured_in_octaves_so_double_time_is_one_octave_away():
    assert tempo_agreement(120, 120) == 1.0
    assert tempo_agreement(120, 240) == 0.0            # exactly one octave, past the tolerance
    assert tempo_agreement(120, 240) == tempo_agreement(120, 60)
    assert 0 < tempo_agreement(120, 132) < 1
    assert tempo_agreement(0, 120) is None and tempo_agreement(None, 120) is None


def test_metre_and_the_rhythm_blend():
    assert metre_agreement({"numerator": 4}, {"numerator": 4}) == 1.0
    assert metre_agreement({"numerator": 3}, {"numerator": 4}) == 0.0
    # Tempo dominates; metre is a correction.
    same_tempo_other_metre = rhythm_agreement(
        {"bpm": 120, "time_signature": {"numerator": 4}},
        {"bpm": 120, "time_signature": {"numerator": 3}})
    assert 0.7 < same_tempo_other_metre < 1.0
    # Tempo alone still answers in full; METRE alone is capped at its own
    # weight. 4/4 is near-universal, so an unmeasured tempo beside a shared
    # metre used to score a perfect 1.0 and outrank a track 3 BPM out.
    assert rhythm_agreement({"bpm": 120}, {"bpm": 120}) == 1.0
    assert rhythm_agreement({"time_signature": {"numerator": 4}},
                            {"time_signature": {"numerator": 4}}) == METRE_WEIGHT
    assert rhythm_agreement({"bpm": 118}, {"bpm": 120}) > METRE_WEIGHT
    assert rhythm_agreement({}, {}) is None


OUT = {"bpm": 120, "key": _k("A", "minor"), "time_signature": {"numerator": 4}}
TRACKS = {
    "t1": {"bpm": 120, "key": _k("A", "minor"), "time_signature": {"numerator": 4}},
    "t2": {"bpm": 90, "key": _k("C", "major"), "time_signature": {"numerator": 3}},
    "t3": {"bpm": None, "key": None, "time_signature": None},
}


def test_each_channel_sums_to_a_hundred_and_reports_its_own_coverage():
    r = aspect_shares(OUT, TRACKS, timbre_scores={"t1": 0.9, "t2": 0.3, "t3": 0.1})
    for channel in ("harmony", "rhythm", "timbre"):
        shares = r[channel]["shares_pct"]
        assert sum(shares.values()) == pytest.approx(100, abs=0.01), channel
        assert r[channel]["of_tracks"] == 3
    # The unmeasured track is dropped from the descriptor channels, and the
    # coverage says so — it is not scored zero.
    assert r["harmony"]["measured_tracks"] == 2 and "t3" not in r["harmony"]["shares_pct"]
    assert r["timbre"]["measured_tracks"] == 3          # CLAP had a score for all three


def test_a_channel_with_nothing_measurable_is_present_and_null():
    """'We looked and could not say' must be distinguishable from 'we did
    not look'."""
    r = aspect_shares({"bpm": None, "key": None}, {"t1": {"bpm": None, "key": None}})
    assert r["harmony"]["shares_pct"] is None and r["harmony"]["measured_tracks"] == 0
    assert "harmony" in r and "rhythm" in r and "timbre" in r


def test_the_timbre_channel_is_the_blend_s_own_share_when_it_is_given_one():
    """It used to normalise the raw cosines linearly while the estimate's
    similarity_share_pct came from the blend's softmax — one document, two
    numbers for 'the estimator's own CLAP similarity', one of them labelled
    unchanged. On a real spread that was 74.9/18.6/6.5 against 35.4/33.1/31.4."""
    tracks = {"t1": TRACKS["t1"], "t2": TRACKS["t2"]}
    r = aspect_shares(OUT, tracks, timbre_weights={"t1": 0.749, "t2": 0.251})
    assert r["timbre"]["shares_pct"] == {"t1": 74.9, "t2": 25.1}
    assert "unchanged" in r["timbre"]["source"]


def test_without_the_blend_s_share_the_channel_says_it_is_not_the_same_number():
    r = aspect_shares(OUT, {"t1": TRACKS["t1"], "t2": TRACKS["t2"]},
                      timbre_scores={"t1": 0.6, "t2": 0.2})
    assert r["timbre"]["shares_pct"] == {"t1": 75.0, "t2": 25.0}
    assert "NOT the blend" in r["timbre"]["source"]


def test_a_negative_cosine_can_never_produce_a_negative_or_over_100_share():
    """CLAP cosines are signed. A signed sum gave 0.9/-0.85/0.1 the shares
    600% / -566.7% / 66.7%, which summed to 100 and were not shares."""
    r = aspect_shares(OUT, {"t1": TRACKS["t1"], "t2": TRACKS["t2"], "t3": TRACKS["t3"]},
                      timbre_scores={"t1": 0.9, "t2": -0.85, "t3": 0.1})
    shares = r["timbre"]["shares_pct"]
    assert all(0 <= v <= 100 for v in shares.values()), shares
    assert sum(shares.values()) == pytest.approx(100, abs=0.01)


def test_all_measured_and_all_disagreeing_is_not_the_same_as_nothing_measurable():
    """Both used to report shares_pct: null. One is a finding."""
    nothing = aspect_shares({"bpm": None, "key": None}, {"t1": {"bpm": None, "key": None}})
    assert nothing["harmony"]["unmeasurable"] is True
    disagree = aspect_shares({"key": _k("C", "major")},
                             {"t1": {"key": _k("F#", "major")}, "t2": {"key": _k("B", "major")}})
    assert disagree["harmony"]["unmeasurable"] is False
    assert disagree["harmony"]["measured_tracks"] == 2 and disagree["harmony"]["shares_pct"] is None


def test_the_caveat_refuses_the_causal_reading():
    r = aspect_shares(OUT, TRACKS)
    assert "not causal influence" in r["caveat"]
