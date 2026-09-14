"""Collapse diagnostics: does the similarity signal vary with the output?

Independent implementation of the Aria readings (arXiv:2605.16181) over a
score matrix with one row per training track and one column per generated
output: kappa (mean |correlation| between columns), rank1_energy (share of
energy in the first singular value), next4_energy (the following four) and
concentration (mean of max score / sum per column). They say whether the
attribution varies, not whether it is correct. Pure: numpy in, numbers out.
"""

from __future__ import annotations

import numpy as np

# Bands, not calibrations: nothing has been measured against a Khaos catalogue.
KAPPA_COLLAPSED = 0.90        # columns this alike are one ranking wearing many hats
RANK1_COLLAPSED = 0.95        # one axis holding this much is the whole matrix
DIAGNOSTICS_VERSION = "0.2.0"  # 0.1.0 had no absolute spread floor
MIN_QUERIES = 8               # fewer columns than this and the readings are noise
# The readings are ratios and cannot see magnitude; the estimator's own noise
# floor is named here so the two guards cannot drift. blend.SPREAD_FLOOR owns it.
from khaos_attribution.blend import SPREAD_FLOOR as _SPREAD_FLOOR  # noqa: E402
SPREAD_FLOOR = _SPREAD_FLOOR


def _column_correlations(scores: np.ndarray) -> float | None:
    """Mean absolute Pearson correlation between every pair of columns."""
    if scores.shape[1] < 2:
        return None
    centred = scores - scores.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0)
    live = norms > 0
    if live.sum() < 2:
        return 1.0  # every column constant: perfectly collapsed, not uncorrelated
    unit = centred[:, live] / norms[live]
    corr = unit.T @ unit
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.abs(off).mean())


def collapse_readings(scores: np.ndarray) -> dict:
    """The four readings over a tracks x outputs score matrix.

    `scores` must hold the raw similarity scores (cosine), one column per
    generated output — never the blended share, whose constant exposure
    prior would manufacture the collapse this looks for.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.size == 0:
        raise ValueError("collapse_readings needs a 2-D tracks x outputs matrix")
    n_tracks, n_queries = s.shape

    # Range, not std: std of an identical column is a tiny non-zero float.
    constant = int((s.max(axis=0) - s.min(axis=0) == 0).sum())
    kappa = _column_correlations(s)

    # Column-centred, or the first component is the catalogue's mean similarity.
    centred = s - s.mean(axis=0, keepdims=True)
    if min(centred.shape) >= 1 and np.any(centred):
        sv = np.linalg.svd(centred, compute_uv=False)
        energy = float((sv ** 2).sum())
        rank1 = float(sv[0] ** 2 / energy) if energy > 0 else None
        next4 = float((sv[1:5] ** 2).sum() / energy) if energy > 0 else None
    else:
        rank1, next4 = None, None

    # Concentration: how much of a column sits in its single best track.
    positive = np.clip(s, 0, None)
    totals = positive.sum(axis=0)
    live = totals > 0
    concentration = float((positive.max(axis=0)[live] / totals[live]).mean()) if live.any() else None

    # The absolute spread the ratios cannot see, in cosine units.
    spread = float(np.median(s.std(axis=0)))

    return {
        "n_tracks": int(n_tracks),
        "n_queries": int(n_queries),
        "median_column_spread": spread,
        "spread_floor": float(SPREAD_FLOOR),
        "constant_columns": constant,
        "kappa": kappa,
        "rank1_energy": rank1,
        "next4_energy": next4,
        "concentration": concentration,
        "flat_concentration": float(1.0 / n_tracks),
    }


def verdict(readings: dict) -> dict:
    """One word for the readings, with the reason that produced it.

    `informative`: the signal moved with the output. `collapsed`: it did not,
    and the estimate falls back to the exposure prior. `unknown`: too few
    outputs to tell; never reported as either of the others.
    """
    n = readings.get("n_queries") or 0
    if n < MIN_QUERIES:
        return {"verdict": "unknown",
                "why": f"{n} generated outputs on file; {MIN_QUERIES} are needed "
                       f"before the similarity signal can be judged across outputs"}

    reasons = []
    kappa, rank1 = readings.get("kappa"), readings.get("rank1_energy")
    spread = readings.get("median_column_spread")
    floor = readings.get("spread_floor", SPREAD_FLOOR)
    if spread is not None and spread < floor:
        # Below the noise floor the ratios only describe the shape of noise.
        return {"verdict": "collapsed",
                "why": (f"the similarity scores vary by {spread:.2g} across tracks, below the "
                        f"estimator's noise floor of {floor:.2g} — there is no signal to read")}
    if readings.get("constant_columns"):
        reasons.append(f"{readings['constant_columns']} of {n} outputs scored every track identically")
    live = n - (readings.get("constant_columns") or 0)
    if kappa is not None and kappa >= KAPPA_COLLAPSED:
        reasons.append(f"outputs rank the catalogue almost identically "
                       f"(kappa {kappa:.2f} over {live} varying outputs)")
    if rank1 is not None and rank1 >= RANK1_COLLAPSED:
        reasons.append(f"one axis holds {rank1:.0%} of the signal")
    if reasons:
        return {"verdict": "collapsed", "why": "; ".join(reasons)}
    bits = [f"the similarity signal varies across {n} outputs"]
    if kappa is not None:
        bits.append(f"kappa {kappa:.2f}")
    if rank1 is not None:
        bits.append(f"leading axis {rank1:.0%}")
    return {"verdict": "informative",
            "why": bits[0] + (" (" + ", ".join(bits[1:]) + ")" if len(bits) > 1 else "")}


def reliability(scores: np.ndarray) -> dict:
    """The readings and the verdict together, stamped with the method's name
    and version — what an estimate carries."""
    readings = collapse_readings(scores)
    return {
        "method": "aria_collapse_diagnostics",
        "method_version": DIAGNOSTICS_VERSION,
        "method_source": "arXiv:2605.16181 (Han, Panahi & Tatar, 2026) — equations reimplemented, no code exists",
        "caveat": ("The bands that turn these readings into a word are conventions, not "
                   "calibrations: nothing has been measured against a Khaos catalogue. "
                   "A collapsed verdict means the estimate is its exposure prior."),
        **readings,
        **verdict(readings),
    }
