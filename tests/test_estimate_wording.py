"""What the estimate tells its reader — pinned after the 2026-09-11 release
review found the contract calling a resemblance measure "influence" seven
times and "resemblance" never, with the honest sentence living in a source
comment. The JSON key `influence` stays (a wire name; the top level is
closed); every sentence a person reads must say resemblance.
"""
from __future__ import annotations

import json
from importlib import resources

import numpy as np
import pytest

from khaos_attribution import estimator
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
    return build_estimate(**{"embeddings": EMB, "output_embedding": OUT, **BASE, **over})


def test_the_first_caveat_names_the_method_as_resemblance_not_influence():
    doc = _estimate()
    first = doc["caveats"][0]
    assert first == estimator.BASE_CAVEAT
    assert "resemblance estimate" in first and "not a measurement of influence" in first
    assert "never a legal statement of ownership" in first


def test_no_caveat_the_estimator_writes_claims_influence():
    """Every sentence a reader sees, across the degenerate paths too."""
    docs = [
        _estimate(),
        _estimate(run_tracks={"t1"}, segment_counts={"t1": 4}, n_training_pairs=4),
        _estimate(segment_counts={"t1": 4, "t2": 0}, run_tracks={"t1", "t2", "t3"}),
        _estimate(recent_similarity={f"g{i}": {"t1": 0.8, "t2": 0.4, "t3": 0.6} for i in range(10)}),
    ]
    for doc in docs:
        for caveat in doc["caveats"]:
            if "not a measurement of influence" in caveat:
                continue                      # the one place the word belongs
            assert "influenc" not in caveat.lower(), caveat


def test_the_schema_descriptions_say_resemblance_and_explain_the_key():
    schema = json.loads(resources.files("khaos_attribution")
                        .joinpath("schemas", "attribution_estimate.schema.json")
                        .read_text(encoding="utf-8"))
    assert "RESEMBLES" in schema["description"]
    assert "not a measurement of what shaped the model" in schema["description"]
    key = schema["properties"]["influence"]["description"]
    assert "RESEMBLANCE" in key and "compatibility" in key
    assert "influenced" not in schema["description"]


def test_a_single_track_adapter_says_the_share_is_exposure_not_resemblance():
    doc = _estimate(run_tracks={"t1"}, segment_counts={"t1": 4}, n_training_pairs=4)
    assert doc["influence"][0]["blended_share_pct"] == 100
    assert doc["method"]["similarity_informative"] is False
    assert any("resemblance was not measured" in c for c in doc["caveats"])


def test_a_zero_count_track_is_named_as_missing_not_dropped_in_silence():
    doc = _estimate(segment_counts={"t1": 4, "t2": 4, "t3": 0})
    assert {e["track_id"] for e in doc["influence"]} == {"t1", "t2"}
    assert any("1 of 3 training tracks have no stored embeddings" in c and "t3" in c
               for c in doc["caveats"])


def test_a_nan_embedding_fails_with_a_sentence_not_inside_the_validator():
    with pytest.raises(ValueError, match="output's embedding for g1 contains NaN"):
        _estimate(output_embedding=np.array([np.nan, 0.0], dtype=np.float32))
    bad = EMB.copy(); bad[1, 0] = np.inf
    with pytest.raises(ValueError, match="stored embeddings for run_1 contain NaN"):
        _estimate(embeddings=bad)
