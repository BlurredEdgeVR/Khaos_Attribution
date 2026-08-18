"""The blend estimator's arithmetic — the canonical suite.

blend.py lives in this contract package because both the Workshop and the
Listening Space compute estimates from it. No CLAP, no files: every
function is pure, so every property worth money is checkable with floats.
"""

from __future__ import annotations

import numpy as np
import pytest

from khaos_attribution import blend


def test_exposure_follows_training_segments():
    shares = blend.exposure_shares({"a": 30, "b": 10})
    assert shares == {"a": 0.75, "b": 0.25}


def test_exposure_ignores_trackless_counts():
    assert blend.exposure_shares({"a": 0, "b": 10}) == {"b": 1.0}
    assert blend.exposure_shares({}) == {}


def test_similarity_scores_are_topk_not_max_or_mean():
    """One lucky segment must not speak for a track (max would), and a long
    track's quiet passages must not dilute it (mean would)."""
    output = np.array([1.0, 0.0], dtype=np.float32)
    embeddings = np.array([
        [1.0, 0.0],    # a: sims 1.0, 0.8, 0.6, 0.0 -> top3 mean 0.8
        [0.8, 0.6],
        [0.6, 0.8],
        [0.0, 1.0],
        [0.5, 0.866],  # b: single segment 0.5
    ], dtype=np.float32)
    rows = ["a", "a", "a", "a", "b"]
    scores = blend.similarity_scores(output, embeddings, rows, {"a", "b"})
    assert scores["a"] == pytest.approx(0.8, abs=1e-6)
    assert scores["b"] == pytest.approx(0.5, abs=1e-3)


def test_similarity_ignores_tracks_outside_the_run():
    """The store holds every track ever embedded; only the adapter's
    training tracks may influence its outputs."""
    output = np.array([1.0, 0.0], dtype=np.float32)
    embeddings = np.eye(2, dtype=np.float32)
    scores = blend.similarity_scores(output, embeddings, ["in", "out"], {"in"})
    assert set(scores) == {"in"}


def test_tight_clusters_are_declared_uninformative():
    """Same-genre catalogues cluster in embedding space. A 0.001 spread is
    noise, and pretending to read track-level signal from it would put
    astrology on a royalty statement."""
    weights, temperature = blend.similarity_weights(
        {"a": 0.5001, "b": 0.5000, "c": 0.4999})
    assert weights is None and temperature is None


def test_single_track_has_no_similarity_signal():
    assert blend.similarity_weights({"only": 0.9}) == (None, None)


def test_informative_similarity_sharpens_toward_the_closer_track():
    weights, temperature = blend.similarity_weights({"near": 0.6, "far": 0.3})
    assert weights["near"] > weights["far"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert temperature == pytest.approx(0.15)


def test_blend_is_prior_times_likelihood():
    exposure = {"a": 0.8, "b": 0.2}
    sim = {"a": 0.3, "b": 0.7}
    blended = blend.blend_shares(exposure, sim)
    assert blended["a"] == pytest.approx(0.24 / 0.38)
    assert sum(blended.values()) == pytest.approx(1.0)


def test_blend_without_signal_is_the_prior():
    exposure = {"a": 0.8, "b": 0.2}
    assert blend.blend_shares(exposure, None) == exposure


def test_ranges_span_every_method_including_the_blend():
    """The product of prior and likelihood can sharpen past both — the range
    must cover the blend too, or the point lands outside its own interval."""
    exposure = {"a": 0.8, "b": 0.2}
    sim = {"a": 0.7, "b": 0.3}
    blended = blend.blend_shares(exposure, sim)   # a ≈ 0.903 — outside [0.7, 0.8]
    ranges = blend.disagreement_ranges([exposure, sim, blended])
    lo, hi = ranges["a"]
    assert lo <= blended["a"] <= hi
    assert lo == 0.7 and hi == pytest.approx(blended["a"])


def test_money_follows_influence_times_track_shares():
    blended = {"t1": 0.6, "t2": 0.4}
    ranges = {"t1": (0.5, 0.7), "t2": (0.3, 0.5)}
    rights = {
        "t1": {"writers": [{"name": "Ava", "share_pct": 50},
                           {"name": "Ben", "share_pct": 50}],
               "publishers": [{"name": "Pub Ltd", "share_pct": 100}]},
        "t2": {"writers": [{"name": "Ava", "share_pct": 100}],
               "publishers": []},
    }
    splits = blend.money_splits(blended, ranges, rights)
    by_name = {w["name"]: w for w in splits["writers"]}
    # Ava: 50% of t1 (0.6) + 100% of t2 (0.4) = 70%
    assert by_name["Ava"]["share_pct"] == pytest.approx(70.0)
    assert by_name["Ben"]["share_pct"] == pytest.approx(30.0)
    assert by_name["Ava"]["share_range_pct"][0] <= 70.0 <= by_name["Ava"]["share_range_pct"][1]
    assert splits["publishers"][0]["share_pct"] == pytest.approx(60.0)
    assert splits["unattributed_pct"] == 0.0


def test_missing_rights_become_unattributed_not_redistributed():
    """A splits sheet that quietly absorbed unknown ownership into the known
    writers would be lying with clean margins."""
    blended = {"known": 0.7, "unknown": 0.3}
    ranges = {t: (v, v) for t, v in blended.items()}
    rights = {"known": {"writers": [{"name": "Ava", "share_pct": 100}],
                        "publishers": []}}
    splits = blend.money_splits(blended, ranges, rights)
    assert splits["unattributed_pct"] == pytest.approx(30.0)
    assert splits["writers"][0]["share_pct"] == pytest.approx(70.0), (
        "Ava's share must stay 70 — the unknown 30 is not hers to inherit")


def test_party_ranges_never_exceed_the_whole_output():
    """Per-track maxima are independent envelopes; summed for a writer with
    shares in every track they can pass 100% — found live at 147.99%, and
    the schema validator rightly refused the document. Nobody owns more
    than the whole."""
    blended = {"a": 0.6, "b": 0.4}
    ranges = {"a": (0.3, 0.9), "b": (0.2, 0.8)}     # hi sum = 1.7
    rights = {t: {"writers": [{"name": "Solo", "share_pct": 100}],
                  "publishers": []} for t in blended}
    splits = blend.money_splits(blended, ranges, rights)
    (writer,) = splits["writers"]
    assert writer["share_range_pct"][1] == 100.0
    assert writer["share_range_pct"][0] <= writer["share_pct"] <= writer["share_range_pct"][1]


def test_largest_remainder_survives_a_251_track_catalogue():
    """Naive 4dp rounding drifts past the validator's 0.01 sum tolerance at
    201+ tracks (found in review at 251). Largest-remainder sums to exactly
    100 at any catalogue size."""
    shares = {f"t{i}": 20 / 5022 for i in range(250)}
    shares["big"] = 22 / 5022
    total = sum(shares.values())
    shares = {t: v / total for t, v in shares.items()}
    pcts = blend.largest_remainder_pcts(shares)
    assert round(sum(pcts.values()), 4) == 100.0
    assert all(abs(pcts[t] - shares[t] * 100) <= 0.0002 for t in shares)


def test_a_torn_bundle_is_refused_not_misattributed():
    """Index rows and the embedding array are two separately copied files;
    when their counts disagree the store is torn, and zip-truncation would
    silently drop tracks from similarity (found in review: a 3-track index
    over a 2-row array crashed on one shape and mislabeled itself as
    'uninformative' on another)."""
    from khaos_attribution.estimator import build_estimate
    with pytest.raises(ValueError, match="torn"):
        build_estimate(
            generation_id="g", artist_id="a", adapter_version="r",
            run_tracks={"a", "b", "c"},
            segment_counts={"a": 1, "b": 1, "c": 1},
            n_training_pairs=3,
            embeddings=np.eye(2, dtype=np.float32),
            row_track_ids=["a", "b", "c"],
            output_embedding=np.array([1.0, 0.0], dtype=np.float32),
            rights={}, fallback_titles={},
            embedding_model=None, embedding_version=None,
            exposure_basis="test")


def test_ranges_are_temperature_sensitivity_not_ingredient_spread():
    """The range answers "how much would this share move if calibration says
    we sharpen twice too much, or half enough" — it must contain the point
    (factor 1.0 is in the sweep) and be narrower than the old raw
    exposure-to-similarity interval that read as wider uncertainty than the
    method actually has."""
    exposure = {"a": 0.8, "b": 0.2}
    scores = {"a": 0.30, "b": 0.60}
    weights, temperature = blend.similarity_weights(scores)
    blended = blend.blend_shares(exposure, weights)
    ranges = blend.temperature_sweep_ranges(exposure, scores, temperature)
    for track in exposure:
        lo, hi = ranges[track]
        assert lo <= blended[track] <= hi
    old_lo = min(exposure["a"], weights["a"], blended["a"])
    old_hi = max(exposure["a"], weights["a"], blended["a"])
    lo, hi = ranges["a"]
    assert (hi - lo) < (old_hi - old_lo), (
        "the sweep interval should be tighter than the ingredient spread")

