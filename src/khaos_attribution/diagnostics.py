"""Is the similarity signal saying anything? — the collapse diagnostics.

An attribution estimate over a catalogue is only worth its similarity term
if that term RESPONDS to the output being attributed. A signal that returns
the same tracks whatever you generated is not attribution; it is a portrait
of the catalogue, and reading a per-track share out of it is astrology with
four decimal places.

Khaos already half-suspected this: ``blend.SPREAD_FLOOR`` refuses to read
track-level signal out of a cosine spread below 0.005. But that is a guess
made per generation, from one column of scores. It cannot see the failure
that matters, which is visible only ACROSS generations: every output
retrieving the same handful of tracks.

These four readings are from Aria (arXiv:2605.16181, Han, Panahi & Tatar,
2026), which showed that they rank attribution methods in the same order as
ground truth that costs hundreds of GPU-hours to obtain — while needing no
ground truth, no retraining, and no gradients. Aria ships no code; this is
an independent implementation from the paper's equations, which are a
correlation, a singular value decomposition and a column mean.

Given a score matrix S with one ROW per training track and one COLUMN per
generated output:

  kappa            mean |correlation| between columns. High: every output
                   ranks the catalogue the same way.
  rank1_energy     share of the matrix's energy in its first singular
                   value. Near 1: one global axis is the whole signal.
  next4_energy     the following four, for context — a signal spread over
                   several axes is a different thing from one that is not.
  concentration    mean over columns of (max score / sum of scores). Near
                   the reciprocal of the track count: the column is flat.

None of these says the attribution is CORRECT. They say whether it varies.
A method can be responsive and wrong. It cannot be useful and flat.

Everything here is pure: numpy in, plain numbers out, no files and no model.
"""

from __future__ import annotations

import numpy as np

# Bands, not calibrations. Aria reports diagnostics for collapsed and
# healthy settings but proposes no threshold, and no one has calibrated
# these against a Khaos catalogue. They exist so a document can carry a
# word as well as a number, and the word is deliberately cautious.
KAPPA_COLLAPSED = 0.90        # columns this alike are one ranking wearing many hats
RANK1_COLLAPSED = 0.95        # one axis holding this much is the whole matrix
DIAGNOSTICS_VERSION = "0.2.0"  # 0.1.0 had no absolute spread floor
MIN_QUERIES = 8               # fewer columns than this and the readings are noise
# Every reading above is a RATIO and so cannot see magnitude: a matrix of
# 0.42 plus noise of 1e-9 produced kappa 0.13 and a verdict of informative,
# which is the exact failure this module exists to catch (found in review,
# 2026-09-10). This is the estimator's own noise floor, named here so the two
# guards cannot drift; blend.SPREAD_FLOOR is the owner.
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
        # Every column constant: perfectly collapsed, and correlation is
        # undefined rather than zero. Say so as 1.0 with the caller's own
        # `constant_columns` count beside it.
        return 1.0
    unit = centred[:, live] / norms[live]
    corr = unit.T @ unit
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.abs(off).mean())


def collapse_readings(scores: np.ndarray) -> dict:
    """The four readings over a tracks x outputs score matrix.

    `scores` holds the SIMILARITY scores the estimator computed, one column
    per generated output, in the estimator's own units (cosine). It must not
    be the blended share: the blend contains the exposure prior, which is
    constant across outputs by construction and would manufacture exactly
    the collapse this is looking for.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.size == 0:
        raise ValueError("collapse_readings needs a 2-D tracks x outputs matrix")
    n_tracks, n_queries = s.shape

    # Range, not standard deviation: std of an identical column is a tiny
    # non-zero float (the mean and the deviations are each rounded), so
    # `std == 0` counted none of them. Max minus min is exact, and it is
    # the property meant — every track scored the same.
    constant = int((s.max(axis=0) - s.min(axis=0) == 0).sum())
    kappa = _column_correlations(s)

    # Energy in the leading singular values, on the column-centred matrix:
    # an uncentred matrix's first component is dominated by the catalogue's
    # mean similarity, which is not a finding, it is an offset.
    centred = s - s.mean(axis=0, keepdims=True)
    if min(centred.shape) >= 1 and np.any(centred):
        sv = np.linalg.svd(centred, compute_uv=False)
        energy = float((sv ** 2).sum())
        rank1 = float(sv[0] ** 2 / energy) if energy > 0 else None
        next4 = float((sv[1:5] ** 2).sum() / energy) if energy > 0 else None
    else:
        rank1, next4 = None, None

    # Concentration: how much of a column sits in its single best track.
    # Compared against 1/n_tracks, the value a perfectly flat column gives.
    positive = np.clip(s, 0, None)
    totals = positive.sum(axis=0)
    live = totals > 0
    concentration = float((positive.max(axis=0)[live] / totals[live]).mean()) if live.any() else None

    # The absolute spread the ratios cannot see: the median per-column
    # standard deviation, in the estimator's own cosine units.
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

    Three answers only. `informative` means the signal moved with the
    output. `collapsed` means it did not, and the estimate should be read as
    the exposure prior it will fall back to. `unknown` means there were not
    enough outputs to tell, which is the honest answer for a new adapter and
    must never be reported as either of the others.
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
        # Below the estimator's own noise floor nothing else is worth
        # reading: the ratios will happily describe the shape of noise.
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
    """The readings and the verdict together — what an estimate carries.

    The method's name and version travel with it so a stored reliability
    block can never be read as having been computed some other way.
    """
    readings = collapse_readings(scores)
    return {
        "method": "aria_collapse_diagnostics",
        "method_version": DIAGNOSTICS_VERSION,
        "method_source": "arXiv:2605.16181 (Han, Panahi & Tatar, 2026) — equations reimplemented, no code exists",
        # The bands are not calibrated against any Khaos catalogue, and a
        # document read by a rights body must carry that, not leave it in a
        # source comment.
        "caveat": ("The bands that turn these readings into a word are conventions, not "
                   "calibrations: nothing has been measured against a Khaos catalogue. "
                   "A collapsed verdict means the estimate is its exposure prior."),
        **readings,
        **verdict(readings),
    }
