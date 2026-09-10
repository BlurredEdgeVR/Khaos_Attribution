"""The collapse diagnostics must be able to say 'this signal means nothing'.

From the 2026-09-10 prior-art review: Deng et al. measured CLAP-style
similarity against causal influence at 0.014-0.068, and Aria showed that
several published attribution results were artefacts of a signal that
returned the same tracks whatever was generated. Khaos uses CLAP similarity.
These readings are how an estimate reports whether its own signal moved.
"""
from __future__ import annotations

import numpy as np
import pytest

from khaos_attribution.diagnostics import (MIN_QUERIES, collapse_readings,
                                           reliability, verdict)


def _rng():
    return np.random.default_rng(20260910)


def test_a_signal_that_ignores_the_output_is_called_collapsed():
    """One column shape repeated: every output ranks the catalogue alike."""
    rng = _rng()
    one_axis = rng.normal(size=(24, 1)) @ np.ones((1, 16))
    one_axis = one_axis + 1e-9 * rng.normal(size=one_axis.shape)   # not exactly singular
    r = reliability(one_axis)
    assert r["verdict"] == "collapsed"
    assert r["kappa"] > 0.99 and r["rank1_energy"] > 0.99
    assert "rank the catalogue almost identically" in r["why"] or "one axis holds" in r["why"]


def test_a_signal_that_varies_is_called_informative():
    r = reliability(_rng().normal(size=(24, 16)))
    assert r["verdict"] == "informative" and r["kappa"] < 0.5


def test_too_few_outputs_is_unknown_and_never_either_answer():
    """A new adapter has no evidence. That is not 'informative' and it is
    not 'collapsed' — reporting either would be inventing a finding."""
    for n in range(1, MIN_QUERIES):
        r = reliability(_rng().normal(size=(24, n)))
        assert r["verdict"] == "unknown", n
        assert str(n) in r["why"]


def test_constant_columns_are_counted_and_reported():
    s = _rng().normal(size=(10, 12))
    s[:, :3] = 0.7                       # three outputs scored every track the same
    r = reliability(s)
    assert r["constant_columns"] == 3 and "3 of 12" in r["why"]


def test_the_readings_are_taken_on_the_centred_matrix():
    """A large shared offset is the catalogue's mean similarity, not a
    finding. Adding one must not turn a varied signal into a collapsed one —
    it did, before the matrix was centred."""
    varied = _rng().normal(size=(24, 16))
    assert reliability(varied)["verdict"] == "informative"
    assert reliability(varied + 50.0)["verdict"] == "informative"


def test_concentration_is_reported_against_what_flat_would_be():
    flat = np.ones((25, 12))
    r = collapse_readings(flat)
    assert r["flat_concentration"] == pytest.approx(1 / 25)
    assert r["concentration"] == pytest.approx(1 / 25)


def test_an_empty_or_wrong_shaped_matrix_raises_rather_than_guessing():
    for bad in (np.zeros((0, 0)), np.zeros(5)):
        with pytest.raises(ValueError):
            collapse_readings(bad)


def test_the_verdict_names_its_method_and_that_no_code_exists_to_copy():
    r = reliability(_rng().normal(size=(24, 16)))
    assert r["method"] == "aria_collapse_diagnostics"
    assert "2605.16181" in r["method_source"] and "no code exists" in r["method_source"]


def test_each_band_alone_is_enough_to_call_collapse(monkeypatch):
    """The bands are what call it, proved by moving them rather than by
    editing the readings. The first cut's docstring said 'with the band
    moved to 1.1' and then mutated the readings dict instead, which proves
    only that verdict() reads a key — and left RANK1_COLLAPSED untested."""
    import khaos_attribution.diagnostics as d

    rng = _rng()
    one_axis = rng.normal(size=(24, 1)) @ np.ones((1, 16)) + 1e-9 * rng.normal(size=(24, 16))
    healthy = collapse_readings(rng.normal(size=(24, 16)) + 0.5)
    collapsed = collapse_readings(one_axis)
    collapsed["median_column_spread"] = 1.0      # keep the absolute floor out of it
    healthy["median_column_spread"] = 1.0

    assert verdict(collapsed)["verdict"] == "collapsed"
    assert verdict(healthy)["verdict"] == "informative"

    # Kappa alone: put rank1 out of reach, and the kappa band still calls it.
    monkeypatch.setattr(d, "RANK1_COLLAPSED", 1.1)
    assert verdict(collapsed)["verdict"] == "collapsed"
    monkeypatch.setattr(d, "KAPPA_COLLAPSED", 1.1)
    assert verdict(collapsed)["verdict"] == "informative"

    # Rank1 alone: the same matrix, with only that band reachable.
    monkeypatch.setattr(d, "RANK1_COLLAPSED", 0.95)
    assert verdict(collapsed)["verdict"] == "collapsed"


def test_the_absolute_floor_catches_what_every_ratio_misses(monkeypatch):
    """Kappa, the energy ratios and the concentration are all scale-free, so
    a dead-flat signal plus 1e-9 of noise read 'informative' — the exact
    failure this module exists to catch."""
    import khaos_attribution.diagnostics as d

    flat = np.full((40, 12), 0.42) + 1e-9 * _rng().normal(size=(40, 12))
    r = reliability(flat)
    assert r["verdict"] == "collapsed" and "noise floor" in r["why"]
    assert r["median_column_spread"] < r["spread_floor"]
    # And it is the FLOOR doing it: drop the floor and the ratios are happy.
    monkeypatch.setattr(d, "SPREAD_FLOOR", 0.0)
    assert d.verdict(d.collapse_readings(flat))["verdict"] == "informative"


def test_the_block_carries_its_version_and_says_the_bands_are_conventions():
    r = reliability(_rng().normal(size=(24, 16)) + 0.5)
    assert r["method_version"] == d_version()
    assert "conventions, not" in r["caveat"] and "exposure prior" in r["caveat"]


def d_version():
    from khaos_attribution.diagnostics import DIAGNOSTICS_VERSION
    return DIAGNOSTICS_VERSION
