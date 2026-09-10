"""Per-aspect shares: the music theory has to be right, and the coverage honest.

From the 2026-09-10 prior-art review: Aria argues a single scalar per track
cannot support copyright analysis, because infringement is assessed aspect by
aspect. These channels are built from descriptors Khaos already stores.
"""
from __future__ import annotations

import pytest

from khaos_attribution.aspects import (RELATED_KEY_WEIGHT, aspect_shares,
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
    # Either descriptor alone still answers.
    assert rhythm_agreement({"bpm": 120}, {"bpm": 120}) == 1.0
    assert rhythm_agreement({"time_signature": {"numerator": 4}},
                            {"time_signature": {"numerator": 4}}) == 1.0
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


def test_the_timbre_channel_is_the_estimator_s_own_number_not_a_second_one():
    """If this recomputed similarity it could drift from the blend."""
    r = aspect_shares(OUT, {"t1": TRACKS["t1"], "t2": TRACKS["t2"]},
                      timbre_scores={"t1": 3.0, "t2": 1.0})
    assert r["timbre"]["shares_pct"] == {"t1": 75.0, "t2": 25.0}
    assert "unchanged" in r["timbre"]["source"]


def test_the_caveat_refuses_the_causal_reading():
    r = aspect_shares(OUT, TRACKS)
    assert "not causal influence" in r["caveat"]
