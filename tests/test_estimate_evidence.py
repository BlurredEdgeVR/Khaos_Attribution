"""The estimate carries its own reliability, and can carry its aspects."""
from __future__ import annotations

import numpy as np

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
    """A first generation has no evidence about the signal across outputs."""
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
    assert rel["verdict"] == "unknown" and "without inventing values" in rel["why"]


def test_partial_columns_are_intersected_never_zero_filled():
    """Zero-filling a missing track would manufacture the very variation the
    readings look for."""
    same = {"t1": 0.81, "t2": 0.42, "t3": 0.63}
    partial = {}
    for i in range(10):
        col = dict(same)
        col.pop("t3" if i % 2 else "t2")      # each column drops a different track
        partial[f"g{i}"] = col
    rel = _estimate(recent_similarity=partial)["method"]["reliability"]
    # t1 is the only track every column carries, so one row is all there is.
    assert rel["n_tracks"] == 1
    assert rel["verdict"] == "collapsed"


def test_a_collapsed_verdict_reaches_the_caveats_not_just_the_method_block():
    """Every surface renders `caveats`; a verdict left in `method` alone
    would be invisible."""
    doc = _estimate(recent_similarity={f"g{i}": {"t1": 0.81, "t2": 0.42, "t3": 0.63}
                                       for i in range(10)})
    assert doc["method"]["reliability"]["verdict"] == "collapsed"
    assert any("did not vary with the output" in c for c in doc["caveats"])
    assert any("exposure prior" in c for c in doc["caveats"])


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
    """Both ride under `method`, which the schema leaves open."""
    doc = _estimate(
        recent_similarity={f"g{i}": {"t1": 0.8, "t2": 0.4, "t3": 0.6} for i in range(9)},
        output_metadata={"bpm": 120, "key": {"tonic": "A", "mode": "minor"}},
        track_metadata={"t1": {"bpm": 120, "key": {"tonic": "A", "mode": "minor"}}})
    # build_estimate validates before returning; reaching here is the pin.
    assert doc["schema_version"] == "1.0.0"
    assert "reliability" in doc["method"] and "aspects" in doc["method"]


def test_informative_evidence_does_not_move_a_share():
    """An informative reading is not an ingredient; every share stays put."""
    rng = np.random.default_rng(3)
    recent = {f"g{i}": {t: float(v) for t, v in zip(("t1", "t2", "t3"), rng.normal(size=3))}
              for i in range(12)}
    plain = _estimate()
    with_evidence = _estimate(recent_similarity=recent)
    assert with_evidence["method"]["reliability"]["verdict"] == "informative"
    assert plain["influence"] == with_evidence["influence"]
    assert plain["splits"] == with_evidence["splits"]


def test_a_collapsed_signal_changes_the_number_not_just_the_footnote():
    """A collapsed signal makes the share the exposure prior, and the
    document must say resemblance was not used."""
    plain = _estimate()
    collapsed = _estimate(recent_similarity={f"g{i}": {"t1": 0.8, "t2": 0.4, "t3": 0.6}
                                             for i in range(10)})
    assert collapsed["method"]["reliability"]["verdict"] == "collapsed"
    assert plain["influence"] != collapsed["influence"], "the share did not move"
    for entry in collapsed["influence"]:
        # One 4-dp step is largest-remainder rounding; anything more is a moved share.
        assert abs(entry["blended_share_pct"] - entry["exposure_share_pct"]) <= 0.0001
        assert entry["similarity_share_pct"] is None
        lo, hi = entry["share_range_pct"]
        assert lo == hi and abs(lo - entry["blended_share_pct"]) <= 0.0001
    assert collapsed["method"]["similarity_informative"] is False
    assert collapsed["method"]["temperature"] is None
    assert any("exposure prior" in c for c in collapsed["caveats"])


def test_each_document_stores_its_own_similarity_column():
    """The one input the collapse readings need across outputs."""
    doc = _estimate()
    col = doc["method"]["similarity_scores"]
    assert set(col) == {"t1", "t2", "t3"}
    assert col["t1"] > col["t3"] > col["t2"]     # OUT is nearest t1, then t3


def test_recent_columns_are_read_back_from_earlier_documents():
    """The contract picks the right adapter, skips the output being estimated
    and documents without the block, and keeps the newest `limit`."""
    from khaos_attribution.estimator import recent_similarity_from_documents
    docs = []
    for i in range(12):
        d = _estimate(generation_id=f"g{i}")
        d["created_at"] = f"2026-09-11T00:00:{i:02d}+00:00"
        docs.append(d)
    docs.append({**docs[0], "adapter_version": "other_run", "generation_id": "x"})
    docs.append({"generation_id": "old", "adapter_version": "run_1",
                 "method": {"estimator_version": "0.4.0"}})      # pre-0.5.0: no column
    docs.append("not a document")
    recent = recent_similarity_from_documents(docs, adapter_version="run_1",
                                              exclude_generation_id="g11", limit=8)
    assert "g11" not in recent and "x" not in recent and "old" not in recent
    assert list(recent) == [f"g{i}" for i in range(3, 11)], "newest `limit` by created_at"
    assert recent["g3"] == docs[3]["method"]["similarity_scores"]
    # And the round trip reaches a verdict: identical outputs are collapsed.
    rel = _estimate(recent_similarity=recent)["method"]["reliability"]
    assert rel["verdict"] == "collapsed"
