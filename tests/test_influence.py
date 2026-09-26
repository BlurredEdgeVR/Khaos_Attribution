"""Measured influence: the kernel matches its primal form, shares and writer
shares follow the rules, an index loads to raw rows, and the document
validates with influence as the share the money follows."""
from __future__ import annotations

import json

import numpy as np
import pytest

from khaos_attribution import influence as I
from khaos_attribution import validate_attribution_estimate

TRACKS = ["ta", "tb", "tc"]


def _index_dir(tmp_path, *, rows=6, dim=16, seed=0, dead=()):
    rng = np.random.default_rng(seed)
    unit = rng.standard_normal((rows, dim)).astype(np.float32)
    norms = rng.uniform(0.5, 2.0, size=rows)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    records = []
    for i in range(rows):
        norm = float(norms[i]) if i not in dead else 0.0
        if i in dead:
            unit[i] = 0.0
        records.append({"row": i, "pair_id": f"{TRACKS[i % 3]}_seg_{i:03d}", "track_id": TRACKS[i % 3],
                        "gradient_norm": norm})
    d = tmp_path / "influence_index"
    d.mkdir()
    np.save(str(d / I.INDEX_MATRIX), unit)
    (d / I.INDEX_RECORD).write_text(json.dumps({
        "schema": I.INDEX_SCHEMA, "target": "vnorm", "timesteps": [0.25, 0.75], "noise_seed": 7,
        "projection": {"n_params": 100, "dim": dim, "seed": 1, "fingerprint": None}, "rows": records}))
    return d, unit, norms


def test_load_index_rebuilds_raw_rows_and_marks_dead_rows(tmp_path):
    d, unit, norms = _index_dir(tmp_path, dead=(4,))
    idx = I.load_index(d)
    assert idx.dim == 16 and idx.track_ids == [TRACKS[i % 3] for i in range(6)]
    assert np.allclose(idx.phi[0], unit[0].astype(np.float64) * norms[0])
    assert idx.invalid.tolist() == [False, False, False, False, True, False]
    assert np.all(idx.phi[4] == 0.0)
    assert len(I.index_sha256(d)) == 64


def test_load_index_refuses_a_foreign_schema(tmp_path):
    d, *_ = _index_dir(tmp_path)
    rec = json.loads((d / I.INDEX_RECORD).read_text())
    rec["schema"] = "something.else/1"
    (d / I.INDEX_RECORD).write_text(json.dumps(rec))
    with pytest.raises(I.InfluenceRefused, match="schema"):
        I.load_index(d)


def test_kernel_dual_form_equals_the_primal_form():
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((5, 8))
    q = rng.standard_normal(8)
    lam = 0.7
    primal = phi @ np.linalg.solve(phi.T @ phi + lam * np.eye(8), q)
    assert np.allclose(I.kernel_scores(phi, q, lam), primal)
    with pytest.raises(I.InfluenceRefused):
        I.kernel_scores(phi, q, 0.0)
    with pytest.raises(I.InfluenceRefused):
        I.kernel_scores(phi, q[:4], lam)


def test_shares_and_writer_shares_follow_the_rules():
    assert I.shares({"a": 2.0, "b": -1.0, "c": 2.0}) == {"a": 0.5, "b": 0.0, "c": 0.5}
    assert I.shares({"a": -1.0}) == {"a": 0.0}
    rights = {"a": {"writers": [{"name": "W1", "share_pct": 50}, {"name": "W2", "share_pct": 50}]},
              "b": {"writers": [{"name": "W1", "share_pct": 100}]}}
    ws = I.writer_shares({"a": 0.4, "b": 0.35, "c": 0.25}, rights)
    assert ws == pytest.approx({"W1": 0.55, "W2": 0.2, I.UNASSIGNED: 0.25})
    assert I.per_track([1.0, 2.0, 3.0], ["x", "y", "x"]) == {"x": 4.0, "y": 2.0}


def test_attribute_scores_every_valid_row_and_zeroes_dead_ones(tmp_path):
    d, *_ = _index_dir(tmp_path, dead=(1,))
    idx = I.load_index(d)
    out = I.attribute(idx, idx.phi[0])
    assert out["scores"][1] == 0.0 and out["lam"] == pytest.approx(I.lambda_for(idx))
    assert set(out["track_totals"]) == set(TRACKS) and sum(out["shares"].values()) == pytest.approx(1.0)
    # The row's own gradient is the best explanation of itself.
    assert out["track_totals"]["ta"] == max(out["track_totals"].values())
    with pytest.raises(I.InfluenceRefused):
        I.attribute(idx, np.array([np.nan] * 16))


def _doc(tmp_path, **over):
    d, *_ = _index_dir(tmp_path)
    idx = I.load_index(d)
    rights = {"ta": {"title": "A", "writers": [{"name": "W1", "share_pct": 100}], "publishers": [], "masters": []},
              "tb": {"title": "B", "writers": [{"name": "W2", "share_pct": 100}], "publishers": [], "masters": []}}
    kw = dict(generation_id="g", artist_id="art", adapter_version="run_1",
              track_totals={"ta": 3.0, "tb": 1.0, "tc": -0.5}, rights=rights,
              fallback_titles={"tc": "C"}, index=idx, index_sha256="x" * 64, adapter_sha256="y" * 64, lam=0.5)
    kw.update(over)
    return I.build_influence_estimate(**kw)


def test_document_validates_with_influence_as_the_share_the_money_follows(tmp_path):
    doc = _doc(tmp_path)
    assert validate_attribution_estimate(doc) is doc
    assert doc["schema_version"] == "1.1.0" and doc["method"]["kind"] == "dtrak"
    by = {t["track_id"]: t for t in doc["influence"]}
    assert by["ta"]["influence_share_pct"] == by["ta"]["blended_share_pct"] == 75.0
    assert by["tc"]["blended_share_pct"] == 0.0 and by["tc"]["title"] == "C"
    assert by["ta"]["share_range_pct"] == [71.7, 78.3] and by["tc"]["share_range_pct"][0] == 0.0
    assert doc["splits"]["writers"][0] == {"name": "W1", "share_pct": 75.0, "share_range_pct": [71.7, 78.3]}
    assert doc["splits"]["unattributed_pct"] == 0.0
    assert doc["method"]["validation"]["money_on_the_right_tracks"] == 0.802
    assert doc["method"]["similarity_informative"] is False and "resemblance" not in doc
    assert any("no rights record" in c for c in doc["caveats"])


def test_document_carries_resemblance_beside_influence_never_inside_it(tmp_path):
    clap = {"method": {"estimator_version": "0.5.0", "similarity_informative": True},
            "influence": [{"track_id": "ta", "blended_share_pct": 50.0, "similarity_share_pct": 40.0},
                          {"track_id": "tb", "blended_share_pct": 30.0, "similarity_share_pct": 35.0},
                          {"track_id": "tc", "blended_share_pct": 20.0, "similarity_share_pct": 25.0}]}
    res = I.resemblance_block(clap, {"ta": 0.75, "tb": 0.25, "tc": 0.0})
    assert res["shares_pct"] == {"ta": 40.0, "tb": 35.0, "tc": 25.0}
    assert res["money_agreement_with_influence"] == pytest.approx(1 - 0.5 * (0.35 + 0.10 + 0.25))
    # Uninformative similarity: the blend is only the exposure prior, so no resemblance shares are carried.
    dull = {"method": {"estimator_version": "0.5.0", "similarity_informative": False},
            "influence": [{"track_id": "ta", "blended_share_pct": 60.0, "similarity_share_pct": None}]}
    quiet = I.resemblance_block(dull, {"ta": 1.0})
    assert quiet["shares_pct"] == {} and quiet["money_agreement_with_influence"] is None and quiet["similarity_informative"] is False
    doc = _doc(tmp_path, resemblance=res)
    assert doc["resemblance"]["method"] == "clap_blend" and doc["method"]["similarity_informative"] is True
    by = {t["track_id"]: t for t in doc["influence"]}
    assert by["ta"]["similarity_share_pct"] == 40.0 and by["ta"]["blended_share_pct"] == 75.0


def test_a_dataset_hash_note_becomes_a_caveat(tmp_path):
    doc = _doc(tmp_path, dataset_hash=None, dataset_hash_note="Attributed against today's data; the run recorded no dataset hash.")
    assert doc["method"]["dataset_hash"] is None
    assert any("today's data" in c for c in doc["caveats"])


def test_an_output_no_track_helped_is_refused_not_split(tmp_path):
    with pytest.raises(I.InfluenceRefused, match="no training track has positive influence"):
        _doc(tmp_path, track_totals={"ta": -1.0, "tb": -0.5, "tc": 0.0})
