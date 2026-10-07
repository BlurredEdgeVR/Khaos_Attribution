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
    d.mkdir(exist_ok=True)
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


def test_the_base_caveat_says_what_the_number_is_and_nothing_else():
    """One sentence on the card: what was measured and how. The validation figures
    travel in method.validation, where a surface can choose to show them."""
    assert I.BASE_CAVEAT.startswith("Measured influence: how much of this output's adapter gradient")
    assert I.BASE_CAVEAT.endswith("gradient index.") and I.BASE_CAVEAT.count(".") == 1
    assert I.VALIDATION["money_on_the_right_tracks"] == 0.802


def _own(n=80, seed=0):
    rng = np.random.default_rng(seed)
    with_adapter = rng.normal(0.80, 0.03, size=n)
    return [{"with": float(w), "without": float(w + g)} for w, g in zip(with_adapter, rng.normal(0.70, 0.05, size=n))]


def test_the_two_numbers_are_percentiles_of_the_adapters_own_outputs():
    """About one in twenty of its own outputs falls outside them, and too few outputs give none."""
    own = _own()
    t = I.abstention_thresholds(own, device="mps", compute_dtype="bf16", harness_id="h")
    gains = [r["without"] - r["with"] for r in own]
    assert t["gain_min"] == round(float(np.percentile(gains, 2.5)), 4)
    assert t["loss_max"] == round(float(np.percentile([r["with"] for r in own], 97.5)), 4)
    assert (t["n_outputs"], t["device"], t["compute_dtype"], t["harness_id"]) == (80, "mps", "bf16", "h")
    assert 72 <= t["own_accepted"] <= 78 and t["validation"]["record_content_hash"] == "a47e34ce8f7aebe2"
    with pytest.raises(I.InfluenceRefused, match="needs 40 measured outputs"):
        I.abstention_thresholds(own[:39], device="mps", compute_dtype="bf16")
    holed = own[:39] + [{"with": float("nan"), "without": 1.0}, {"with": 0.8, "without": None}]
    with pytest.raises(I.InfluenceRefused, match="not 39"):
        I.abstention_thresholds(holed, device="mps", compute_dtype="bf16")


def test_the_rule_needs_both_a_gain_and_a_good_enough_loss():
    assert I.accepts(0.6, 0.9, 0.55, 0.99)
    assert not I.accepts(0.55, 0.9, 0.55, 0.99) and not I.accepts(0.6, 0.99, 0.55, 0.99)
    assert not I.accepts(None, 0.9, 0.55, 0.99) and not I.accepts(0.6, None, 0.55, 0.99)


def test_a_reading_says_accepted_refused_or_why_it_was_not_read():
    t = {"gain_min": 0.55, "loss_max": 0.99, "device": "mps", "compute_dtype": "bf16"}
    here = dict(device_type="mps", compute_dtype="bf16")
    ok = I.abstention_reading(t, {"with": 0.80, "without": 1.50}, **here)
    assert ok["checked"] and ok["accepted"] and ok["gain"] == pytest.approx(0.70) and ok["gain_min"] == 0.55
    base = I.abstention_reading(t, {"with": 0.95, "without": 1.00}, **here)
    assert base["checked"] and base["accepted"] is False
    poor = I.abstention_reading(t, {"with": 1.20, "without": 2.00}, **here)
    assert poor["checked"] and poor["accepted"] is False
    for reading, word in ((I.abstention_reading(None, {"with": 0.8, "without": 1.5}, **here), "published without"),
                          (I.abstention_reading({**t, "loss_max": None}, {"with": 0.8, "without": 1.5}, **here), "published without"),
                          (I.abstention_reading(t, {"with": 0.8, "without": 1.5}, device_type="cuda", compute_dtype="bf16"), "taken on mps"),
                          (I.abstention_reading({**t, "device": "cuda:0"}, {"with": 0.8, "without": 1.5}, device_type="cuda", compute_dtype="fp32"), "bf16 precision"),
                          (I.abstention_reading(t, {"with": float("nan"), "without": 1.5}, **here), "could not be measured"),
                          (I.abstention_reading(t, None, **here), "could not be measured")):
        assert reading["checked"] is False and reading["accepted"] is None and word in reading["reason"], word
    assert I.abstention_reading({**t, "device": "cuda:0"}, {"with": 0.8, "without": 1.5}, device_type="cuda", compute_dtype="bf16")["accepted"]
    assert I.abstention_unreadable(t, **here) is None
    assert "published without" in I.abstention_unreadable({}, **here)
    assert "taken on mps" in I.abstention_unreadable(t, device_type="cuda", compute_dtype="bf16")


def test_a_refused_output_gets_no_document_and_an_unread_one_says_so(tmp_path):
    """Never a percentage for an output the rule refuses; a document that was not checked carries the reason."""
    refused = {"checked": True, "accepted": False, "gain": 0.05, "loss_with": 0.95, "gain_min": 0.55, "loss_max": 0.99}
    with pytest.raises(I.NotAttributable) as caught:
        _doc(tmp_path, abstention=refused)
    assert str(caught.value) == I.NOT_ATTRIBUTABLE and caught.value.reading["gain"] == 0.05
    assert isinstance(caught.value, I.InfluenceRefused) and I.NOT_ATTRIBUTABLE.count(".") == 1
    accepted = _doc(tmp_path, abstention={**refused, "accepted": True, "gain": 0.7})
    assert accepted["method"]["abstention"]["accepted"] is True and len(accepted["caveats"]) == len(_doc(tmp_path)["caveats"])
    unread = _doc(tmp_path, abstention={"checked": False, "accepted": None, "reason": "its two numbers were taken on mps"})
    assert unread["method"]["abstention"]["checked"] is False
    assert "Not checked for outputs this adapter did not shape: its two numbers were taken on mps." in unread["caveats"]
    assert "abstention" not in _doc(tmp_path)["method"]


def test_a_kept_refusal_stands_until_its_own_adapter_can_be_read_afresh():
    """Another adapter served, no bundle, or no numbers: the refusal still answers. The same adapter
    under another index or other numbers: the output is read again."""
    reading = {"checked": True, "accepted": False, "gain": 0.05, "gain_min": 0.55, "loss_max": 0.99}
    meta = {"index_sha256": "abc", "adapter_sha256": "w", "abstention": {"gain_min": 0.55, "loss_max": 0.99}}
    kept = I.refusal_record("g1", reading, meta)
    assert kept["not_attributable"] is True and kept["reason"] == I.NOT_ATTRIBUTABLE and kept["reading"]["gain"] == 0.05
    assert (kept["generation_id"], kept["index_sha256"], kept["adapter_sha256"]) == ("g1", "abc", "w") and kept["refused_at_utc"]
    assert I.refusal_stands(kept, meta)
    assert I.refusal_stands(kept, None) and I.refusal_stands(kept, {})
    assert I.refusal_stands(kept, {**meta, "adapter_sha256": "other weights", "index_sha256": "xyz"})
    assert I.refusal_stands(kept, {**meta, "abstention": None}) and I.refusal_stands(kept, {**meta, "abstention": {"gain_min": None}})
    assert not I.refusal_stands(kept, {**meta, "index_sha256": "rebuilt"})
    assert not I.refusal_stands(kept, {**meta, "abstention": {"gain_min": 0.50, "loss_max": 0.99}})
    assert not I.refusal_stands(kept, {**meta, "abstention": {"gain_min": 0.55, "loss_max": 1.10}})
    assert not I.refusal_stands(None, meta) and not I.refusal_stands({**kept, "not_attributable": False}, meta)


def test_numbers_taken_beside_another_index_are_not_read():
    t = {"gain_min": 0.55, "loss_max": 0.99, "device": "mps", "compute_dtype": "bf16", "index_sha256": "abc"}
    here = dict(device_type="mps", compute_dtype="bf16")
    assert I.abstention_unreadable(t, index_sha256="abc", **here) is None and I.abstention_unreadable(t, **here) is None
    assert "another influence index" in I.abstention_unreadable(t, index_sha256="rebuilt", **here)
    assert I.abstention_reading(t, {"with": 0.95, "without": 1.0}, index_sha256="rebuilt", **here)["checked"] is False


def test_the_numbers_are_read_only_near_the_length_they_were_taken_at():
    t = {"gain_min": 0.55, "loss_max": 0.99, "device": "mps", "compute_dtype": "bf16", "render_seconds": 30}
    here = dict(device_type="mps", compute_dtype="bf16")
    for seconds in (15, 30, 60, 90):
        assert I.abstention_unreadable(t, duration_sec=seconds, **here) is None
    for seconds, word in ((14.9, "15 seconds"), (91, "91 seconds"), (240, "240 seconds")):
        why = I.abstention_unreadable(t, duration_sec=seconds, **here)
        assert "taken on 30-second outputs" in why and word in why
    assert I.abstention_unreadable(t, **here) is None, "a length nobody gave is not a reason"
    assert I.abstention_unreadable({**t, "render_seconds": None}, duration_sec=240, **here) is None
    long = I.abstention_reading(t, {"with": 0.95, "without": 1.0}, duration_sec=120, **here)
    assert long["checked"] is False and "120 seconds" in long["reason"], "an output outside the band is never refused by the rule"
    assert I.abstention_reading(t, {"with": 0.95, "without": 1.0}, duration_sec=30, **here)["accepted"] is False


def test_an_adapters_own_control_says_how_many_base_outputs_its_numbers_refuse():
    own = _own()
    plain = I.abstention_thresholds(own, device="mps", compute_dtype="bf16")
    assert plain["control"] is None
    base = [{"with": 0.95, "without": 1.00}] * 46 + [{"with": 0.80, "without": 1.60}] * 2 + [{"with": float("nan"), "without": 1.0}]
    t = I.abstention_thresholds(own, device="mps", compute_dtype="bf16", control=base)
    assert t["control"] == {"n": 48, "refused": 46, "rate": 0.9583, "bar": 0.9, "met": True}
    weak = I.control_reading([{"with": 0.80, "without": 1.60}] * 8 + [{"with": 0.95, "without": 1.0}] * 40, t["gain_min"], t["loss_max"])
    assert (weak["refused"], weak["rate"], weak["met"]) == (40, 0.8333, False)
    assert I.control_reading([], 0.5, 0.9) is None and I.control_reading(None, 0.5, 0.9) is None
    exact = I.control_reading([{"with": 0.95, "without": 1.0}] * 9 + [{"with": 0.80, "without": 1.60}], t["gain_min"], t["loss_max"])
    assert exact["rate"] == 0.9 and exact["met"] is True


def _calibration(**over):
    block = {"schema": I.CALIBRATION_SCHEMA, "artist_id": "art", "factors": {"ta": 0.5, "tb": 2.0}, "fit_outputs": 80,
             "fitted_on": {"record": "payment_test_v2.calibrated.x.json", "content_hash": "abc"},
             "validation": {"test": "T2 v2 calibrated", "money_on_the_right_tracks": 0.879, "kernel_alone": 0.811},
             "measured_error_pp": 2.6}
    block.update(over)
    return block


def test_calibrated_shares_scale_each_track_by_its_factor_and_renormalise():
    """A factor of 0.5 on the kernel's 0.75 share and 2.0 on its 0.25 share meet in the middle."""
    assert I.calibrated_shares({"ta": 3.0, "tb": 1.0, "tc": -0.5}, {"ta": 0.5, "tb": 2.0}) == \
        {"ta": pytest.approx(0.375 / 0.875), "tb": pytest.approx(0.5 / 0.875), "tc": 0.0}
    # A track without a factor keeps the kernel's own weight; nothing named is nothing split.
    assert I.calibrated_shares({"ta": 1.0, "tb": 1.0}, {"ta": 1.0}) == {"ta": 0.5, "tb": 0.5}
    assert I.calibrated_shares({"ta": 0.0, "tb": -1.0}, {"ta": 2.0}) == {}
    assert I.calibrated_shares({"ta": 1.0}, {"ta": 0.0}) == {}


def test_a_calibration_block_is_read_only_when_it_is_whole():
    """Each defect names itself; the kernel alone is the answer to every one."""
    ok = I.read_calibration(_calibration(), index_tracks=["ta", "tb", "tc"])
    assert ok["applied"] and ok["factors"] == {"ta": 0.5, "tb": 2.0} and ok["reason"] is None
    cases = [
        (None, "no calibration"),
        (_calibration(schema="khaos.other/1.0"), "schema"),
        (_calibration(factors={}), "no factors"),
        (_calibration(factors={"ta": -0.1}), "negative or not finite"),
        (_calibration(factors={"ta": float("nan")}), "negative or not finite"),
        (_calibration(factors={"ta": "big"}), "not a number"),
        (_calibration(factors={"ta": True}), "not a number"),
        (_calibration(fit_outputs=59), "under 60"),
        (_calibration(fit_outputs=None), "under 60"),
        (_calibration(factors={"zz": 1.0}), "names a track this index holds"),
    ]
    for block, words in cases:
        r = I.read_calibration(block, index_tracks=["ta", "tb", "tc"])
        assert not r["applied"] and words in r["reason"], (block, r)


def test_a_calibrated_document_splits_by_the_factors_and_cites_its_own_record(tmp_path):
    """Calibrated shares are the money's shares; the validation, the error and the caveat are the block's."""
    doc = _doc(tmp_path, calibration=_calibration())
    by = {t["track_id"]: t for t in doc["influence"]}
    assert by["ta"]["influence_share_pct"] == by["ta"]["blended_share_pct"] == pytest.approx(42.86, abs=0.01)
    assert by["tb"]["blended_share_pct"] == pytest.approx(57.14, abs=0.01)
    assert doc["splits"]["writers"][0]["name"] == "W2"
    m = doc["method"]
    assert m["kind"] == "dtrak" and m["calibrated"] is True and m["measured_error_pp"] == 2.6
    assert m["validation"]["money_on_the_right_tracks"] == 0.879
    assert m["calibration"]["factors"] == {"ta": 0.5, "tb": 2.0}
    assert m["calibration"]["fitted_on"]["content_hash"] == "abc" and m["calibration"]["fit_outputs"] == 80
    assert m["calibration"]["kernel_shares"]["ta"] == 0.75
    assert I.CALIBRATED_CAVEAT in doc["caveats"]
    assert by["ta"]["share_range_pct"] == [pytest.approx(40.26, abs=0.01), pytest.approx(45.46, abs=0.01)]


def test_an_unreadable_calibration_leaves_the_kernel_alone_and_says_why(tmp_path):
    """A block that cannot be read changes no share, keeps the kernel's validation and is named in a caveat."""
    doc = _doc(tmp_path, calibration=_calibration(fit_outputs=12))
    by = {t["track_id"]: t for t in doc["influence"]}
    assert by["ta"]["blended_share_pct"] == 75.0
    assert doc["method"]["calibrated"] is False and doc["method"]["measured_error_pp"] == I.MEASURED_ERROR_PP
    assert doc["method"]["validation"]["money_on_the_right_tracks"] == 0.802
    assert {k: v for k, v in doc["method"]["calibration"].items() if k != "block_sha256"} == \
        {"applied": False, "reason": "calibration fitted on 12 outputs, under 60"}
    assert any(c.startswith("Calibration not applied: ") for c in doc["caveats"])
    assert I.CALIBRATED_CAVEAT not in doc["caveats"]
    plain = _doc(tmp_path)
    assert "calibration" not in plain["method"] and plain["method"]["calibrated"] is False


def test_a_calibration_needs_its_own_validation_a_sane_error_and_the_right_artist(tmp_path):
    """Without a validation number, with a bad error, or from another artist, the kernel alone answers."""
    for block, words in [
        (_calibration(validation=None), "no validation"),
        (_calibration(validation={"test": "x"}), "no validation"),
        (_calibration(measured_error_pp="2.6"), "not a number of points"),
        (_calibration(measured_error_pp=-1), "not a number of points"),
        (_calibration(measured_error_pp=0), "not a number of points"),
        (_calibration(measured_error_pp=float("nan")), "not a number of points"),
        (_calibration(measured_error_pp=True), "not a number of points"),
        (_calibration(artist_id="other"), "is other's, not art's"),
        (["not", "a", "record"], "not a record"),
    ]:
        r = I.read_calibration(block, index_tracks=["ta"], artist_id="art")
        assert not r["applied"] and words in r["reason"], (block, r)
    assert I.read_calibration(_calibration(artist_id="art"), index_tracks=["ta"], artist_id="art")["applied"]
    assert "None's" in I.read_calibration(_calibration(artist_id=None), index_tracks=["ta"], artist_id="art")["reason"]
    assert I.read_calibration(_calibration(measured_error_pp=None), index_tracks=["ta"])["applied"]
    doc = _doc(tmp_path, calibration=_calibration(measured_error_pp=None))
    assert doc["method"]["calibrated"] and doc["method"]["measured_error_pp"] == I.MEASURED_ERROR_PP
    other = _doc(tmp_path, calibration=_calibration(artist_id="other"))
    assert other["method"]["calibrated"] is False and "is other's" in other["method"]["calibration"]["reason"]


def test_factors_of_zero_on_every_named_track_refuse_with_the_true_reason(tmp_path):
    with pytest.raises(I.InfluenceRefused, match="factor of zero"):
        _doc(tmp_path, calibration=_calibration(factors={"ta": 0.0, "tb": 0.0}))
    doc = _doc(tmp_path, calibration=_calibration(factors={"ta": 0.0, "tb": 1.0}))
    assert {t["track_id"]: t["blended_share_pct"] for t in doc["influence"]}["tb"] == 100.0


def test_bad_loss_rows_are_skipped_or_unread_never_a_crash():
    """A string or a NaN loss is not a loss: the thresholds skip the row, the reading says unmeasured."""
    good = [{"with": 0.5 + i * 0.001, "without": 1.0} for i in range(40)]
    bad = [{"with": "0.1", "without": 0.2}, {"with": float("nan"), "without": 1.0}, {"with": True, "without": 1.0}]
    th = I.abstention_thresholds(good + bad, device="cpu", compute_dtype="fp32", control=bad + [{"with": 0.9, "without": 1.0}])
    assert th["n_outputs"] == 40 and th["control"]["n"] == 1
    r = I.abstention_reading(th, {"with": "x", "without": 1.0}, device_type="cpu", compute_dtype="fp32")
    assert r["checked"] is False and "could not be measured" in r["reason"]
    with pytest.raises(TypeError, match="measured_on may not carry"):
        I.abstention_thresholds(good, device="cpu", compute_dtype="fp32", gain_min=99.0)


def test_a_calibrated_document_reports_coverage_and_reads_resemblance_against_its_own_shares(tmp_path):
    """Tracks the fit never covered are named and keep the kernel's weight; the resemblance block's
    agreement is with the calibrated shares, not the kernel's."""
    clap = {"method": {"estimator_version": "0.5.0", "similarity_informative": True},
            "influence": [{"track_id": "ta", "blended_share_pct": 50.0, "similarity_share_pct": 43.0},
                          {"track_id": "tb", "blended_share_pct": 50.0, "similarity_share_pct": 57.0}]}
    kernel = I.shares({"ta": 3.0, "tb": 1.0, "tc": -0.5})
    doc = _doc(tmp_path, calibration=_calibration(factors={"ta": 0.5}), resemblance=I.resemblance_block(clap, kernel))
    cov = doc["method"]["calibration"]["coverage"]
    assert cov == {"fitted": 1, "of": 3, "unfitted": ["tb", "tc"], "extra": []}
    assert any("2 of this index's 3 tracks had no factor" in c for c in doc["caveats"])
    shown = {t["track_id"]: t["blended_share_pct"] / 100 for t in doc["influence"]}
    expect = 1 - 0.5 * (abs(shown["ta"] - 0.43) + abs(shown["tb"] - 0.57) + shown["tc"])
    assert doc["resemblance"]["money_agreement_with_influence"] == pytest.approx(round(expect, 4))
    plain = _doc(tmp_path, resemblance=I.resemblance_block(clap, kernel))
    assert plain["resemblance"]["money_agreement_with_influence"] == pytest.approx(round(1 - 0.5 * (0.32 + 0.32), 4))


def test_a_document_names_the_calibration_block_it_was_read_with_applied_or_not(tmp_path):
    block = _calibration()
    sha = I.calibration_sha256(block)
    assert sha and sha == I.calibration_sha256(dict(block)) and I.calibration_sha256(None) is None
    assert _doc(tmp_path, calibration=block)["method"]["calibration"]["block_sha256"] == sha
    refused = _doc(tmp_path, calibration=_calibration(fit_outputs=3))
    assert refused["method"]["calibration"]["applied"] is False and refused["method"]["calibration"]["block_sha256"] == I.calibration_sha256(_calibration(fit_outputs=3))
