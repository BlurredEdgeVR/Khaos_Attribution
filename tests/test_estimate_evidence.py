"""The estimate carries its own reliability, and can carry its aspects.

From the 2026-09-10 prior-art review. Two published findings land on this
estimator: similarity correlates near zero with causal influence (Deng et
al.), and a single scalar per track cannot support copyright analysis
(Aria). Neither is fixed by arithmetic. Both are answered by the document
saying more about itself — how far its signal carried, and which musical
channel a resemblance ran through.
"""
from __future__ import annotations

import numpy as np
import pytest

from khaos_attribution.estimator import build_estimate

BASE = dict(
    generation_id="g1", artist_id="art", adapter_version="run_1",
    run_tracks={"t1", "t2", "t3"},
    segment_counts={"t1": 4, "t2": 4, "t3": 4},
    n_training_pairs=12,
    row_track_ids=["t1", "t2", "t3"],
    rights={}, fallback_titles={"t1": "One", "t2": "Two", "t3": "Three"},
    embedding_model="clap", embedding_version="1", exposure_basis="test",
)
EMB = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
OUT = np.array([1.0, 0.05], dtype=np.float32)


def _estimate(**over):
    return build_estimate(embeddings=EMB, output_embedding=OUT, **{**BASE, **over})


def test_without_earlier_outputs_the_reliability_is_unknown_and_says_why():
    """A first generation has no evidence about the signal across outputs.
    Reporting 'informative' there would be inventing a finding."""
    rel = _estimate()["method"]["reliability"]
    assert rel["verdict"] == "unknown" and "no earlier outputs" in rel["why"]


def test_a_signal_that_ignores_the_output_is_reported_as_collapsed():
    """Ten earlier generations that all scored the catalogue identically."""
    same = {"t1": 0.81, "t2": 0.42, "t3": 0.63}
    recent = {f"g{i}": dict(same) for i in range(10)}
    rel = _estimate(recent_similarity=recent)["method"]["reliability"]
    assert rel["verdict"] == "collapsed"
    assert rel["n_queries"] == 11 and rel["method"] == "aria_collapse_diagnostics"


def test_a_signal_that_moves_with_the_output_is_reported_as_informative():
    rng = np.random.default_rng(7)
    recent = {f"g{i}": {t: float(v) for t, v in zip(("t1", "t2", "t3"), rng.normal(size=3))}
              for i in range(12)}
    rel = _estimate(recent_similarity=recent)["method"]["reliability"]
    assert rel["verdict"] == "informative" and rel["kappa"] < 0.9


def test_earlier_outputs_over_a_different_catalogue_are_unknown_not_blended():
    """Columns from another artist's tracks would be arithmetic over
    unrelated numbers."""
    rel = _estimate(recent_similarity={"g9": {"other": 0.5}})["method"]["reliability"]
    assert rel["verdict"] == "unknown" and "no track this estimate also scores" in rel["why"]


def test_the_aspects_block_appears_only_when_the_caller_supplies_descriptors():
    assert "aspects" not in _estimate()["method"]
    doc = _estimate(
        output_metadata={"bpm": 120, "key": {"tonic": "A", "mode": "minor"},
                         "time_signature": {"numerator": 4}},
        track_metadata={
            "t1": {"bpm": 120, "key": {"tonic": "A", "mode": "minor"},
                   "time_signature": {"numerator": 4}},
            "t2": {"bpm": 175, "key": {"tonic": "F", "mode": "major"},
                   "time_signature": {"numerator": 3}},
            "t3": {"bpm": None, "key": None, "time_signature": None}})
    asp = doc["method"]["aspects"]
    assert set(asp) >= {"harmony", "rhythm", "timbre", "caveat"}
    assert asp["harmony"]["of_tracks"] == 3 and asp["harmony"]["measured_tracks"] == 2
    assert "not causal influence" in asp["caveat"]


def test_the_document_still_validates_with_both_blocks_present():
    """They ride under `method`, which the schema leaves open — the top
    level is closed and its version is a const, so a new key there would
    invalidate every estimate ever written."""
    doc = _estimate(
        recent_similarity={f"g{i}": {"t1": 0.8, "t2": 0.4, "t3": 0.6} for i in range(9)},
        output_metadata={"bpm": 120, "key": {"tonic": "A", "mode": "minor"}},
        track_metadata={"t1": {"bpm": 120, "key": {"tonic": "A", "mode": "minor"}}})
    # build_estimate validates before returning; reaching here is the pin.
    assert doc["schema_version"] == "1.0.0"
    assert "reliability" in doc["method"] and "aspects" in doc["method"]


def test_the_existing_estimate_is_unchanged_by_the_new_blocks():
    """Adding evidence must not move a single share."""
    plain = _estimate()
    with_evidence = _estimate(recent_similarity={f"g{i}": {"t1": 0.8, "t2": 0.4, "t3": 0.6}
                                                 for i in range(9)})
    assert plain["influence"] == with_evidence["influence"]
    assert plain["splits"] == with_evidence["splits"]
