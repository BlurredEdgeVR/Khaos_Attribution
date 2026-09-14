"""The degenerate paths of the estimate say what happened, in words, and
the guards around it fail with a sentence."""
from __future__ import annotations

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


def test_the_first_caveat_is_the_base_caveat():
    doc = _estimate()
    assert doc["caveats"][0] == estimator.BASE_CAVEAT
    assert "not a legal statement of ownership" in doc["caveats"][0]


def test_a_single_track_adapter_says_the_share_is_by_construction():
    doc = _estimate(run_tracks={"t1"}, segment_counts={"t1": 4}, n_training_pairs=4)
    assert doc["influence"][0]["blended_share_pct"] == 100
    assert doc["method"]["similarity_informative"] is False
    assert any("Single-track adapter" in c for c in doc["caveats"])


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
