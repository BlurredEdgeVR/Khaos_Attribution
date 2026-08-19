"""The blend: exposure prior × calibrated acoustic similarity, with ranges.

Pure functions over plain data — no file I/O, no model. This lives in the
CONTRACT package because both the Workshop and the Listening Space compute
estimates: two apps deriving different numbers from the same data would be
a scandal, so there is exactly one implementation, versioned with the
schemas it feeds.

The method, in full (this docstring is the reference the estimate's
`method` block points at):

1. **Exposure prior.** Each training track's share of the adapter's
   training segments. Deterministic, auditable, and the defensible floor:
   absent evidence of differential influence, influence follows exposure.

2. **Similarity likelihood.** Cosine similarity of the output's CLAP
   embedding against each track's segment embeddings (mean of the top-k
   segments, so a track's one lucky segment does not speak for it and a
   long track is not diluted by its quiet passages). Scores become weights
   through a softmax whose temperature is the score spread — and when that
   spread is below SPREAD_FLOOR the whole signal is declared uninformative
   and the estimator falls back to the exposure prior alone. Same-genre
   catalogues cluster tightly in embedding space; pretending to read
   track-level signal out of a 0.003 spread would be astrology.

3. **Blend.** Posterior ∝ exposure × similarity weight, renormalised.

4. **Ranges.** Per track: [min, max] of the blend recomputed with the
   softmax temperature halved and doubled. The temperature is the one
   uncalibrated parameter (the leave-one-track-out study is its planned
   calibration), so the interval answers "how much would this share move
   if the calibration says we are sharpening twice too much, or half
   enough?" — parameter sensitivity, not ingredient spread. The earlier
   interval spanned raw exposure to raw similarity, two things that were
   never candidate answers, and read as wider uncertainty than the
   method actually has.

5. **Money.** A party's share of the output = Σ over tracks
   (track influence × party's share of that track). Tracks without rights
   records contribute to `unattributed_pct`, reported and never silently
   renormalised away — a splits sheet that quietly absorbed unknown
   ownership would be lying with clean margins.
"""

from __future__ import annotations

import math

import numpy as np

ESTIMATOR_VERSION = "0.3.0"
# exposure_basis moved to a per-document string naming the data source
# ("live Workshop store" / "bundled at artist import") — see estimator.py.
TOP_K = 3            # segments per track that speak for it
SPREAD_FLOOR = 0.005  # cosine std-dev below which similarity is noise


def largest_remainder_pcts(shares: dict[str, float]) -> dict[str, float]:
    """Fractions → percentages at 4dp that sum to EXACTLY 100.

    Naive per-entry rounding drifts by up to N×5e-5, which crosses the
    validator's 0.01 sum tolerance at 201 tracks — a hard catalogue-size
    ceiling found in review. Largest-remainder assigns the rounding residue
    to the entries that lost the most to truncation, so the sum is exact at
    any N and no entry moves by more than one 4dp step.
    """
    scaled = {t: v * 100 for t, v in shares.items()}
    floored = {t: math.floor(v * 10_000) / 10_000 for t, v in scaled.items()}
    residue = round((100 - sum(floored.values())) * 10_000)
    by_remainder = sorted(shares, key=lambda t: -(scaled[t] - floored[t]))
    out = dict(floored)
    for t in by_remainder[:residue]:
        out[t] = round(out[t] + 0.0001, 4)
    return {t: round(v, 4) for t, v in out.items()}


def exposure_shares(segments_per_track: dict[str, int]) -> dict[str, float]:
    """Fraction of training exposure per track (sums to 1)."""
    total = sum(segments_per_track.values())
    if total <= 0:
        return {}
    return {t: n / total for t, n in segments_per_track.items() if n > 0}


def similarity_scores(output_embedding: np.ndarray,
                      embeddings: np.ndarray,
                      row_track_ids: list[str],
                      tracks: set[str]) -> dict[str, float]:
    """Top-k mean cosine similarity per track.

    Embeddings are unit vectors on both sides, so cosine is a dot product.
    """
    sims = embeddings @ output_embedding.astype(np.float32)
    by_track: dict[str, list[float]] = {}
    for sim, track_id in zip(sims, row_track_ids):
        if track_id in tracks:
            by_track.setdefault(track_id, []).append(float(sim))
    scores = {}
    for track_id, values in by_track.items():
        values.sort(reverse=True)
        k = min(TOP_K, len(values))
        scores[track_id] = sum(values[:k]) / k
    return scores


def softmax_weights(scores: dict[str, float],
                    temperature: float) -> dict[str, float]:
    """Scores → weights at an explicit temperature (the sweep uses this)."""
    values = np.array(list(scores.values()), dtype=np.float64)
    logits = (values - values.max()) / temperature
    weights = np.exp(logits)
    weights /= weights.sum()
    return dict(zip(scores.keys(), (float(w) for w in weights)))


def similarity_weights(scores: dict[str, float]
                       ) -> tuple[dict[str, float] | None, float | None]:
    """Softmax the scores into weights, or (None, None) when uninformative.

    Temperature = the score spread itself: a catalogue whose tracks genuinely
    differ gets sharp weights, a tight cluster gets flat ones, and below the
    floor there is no signal to sharpen.
    """
    if len(scores) < 2:
        return None, None
    values = np.array(list(scores.values()), dtype=np.float64)
    spread = float(values.std())
    if spread < SPREAD_FLOOR:
        return None, None
    temperature = spread
    return softmax_weights(scores, temperature), temperature


# The DEFAULT sweep, used when no catalogue calibration exists: a factor-2
# robustness band by convention. A leave-one-track-out study replaces it
# with a measured band per adapter.
TEMPERATURE_SWEEP = (0.5, 1.0, 2.0)


def temperature_sweep_ranges(exposure: dict[str, float],
                             scores: dict[str, float],
                             temperature: float,
                             factors: tuple[float, ...] = TEMPERATURE_SWEEP
                             ) -> dict[str, tuple[float, float]]:
    """Per-track [min, max] of the blend across the temperature sweep.

    `factors` must include 1.0 so the interval contains the point estimate
    by construction and validator containment can never fail.
    """
    if 1.0 not in factors:
        factors = tuple(sorted({*factors, 1.0}))
    members = [blend_shares(exposure, softmax_weights(scores, temperature * f))
               for f in factors]
    return disagreement_ranges(members)


def blend_shares(exposure: dict[str, float],
                 sim_weights: dict[str, float] | None) -> dict[str, float]:
    """Posterior ∝ exposure × similarity; the prior alone when no signal."""
    if not sim_weights:
        return dict(exposure)
    product = {t: exposure[t] * sim_weights.get(t, 0.0) for t in exposure}
    total = sum(product.values())
    if total <= 0:
        return dict(exposure)
    return {t: v / total for t, v in product.items()}


def disagreement_ranges(per_method: list[dict[str, float]]
                        ) -> dict[str, tuple[float, float]]:
    """Per-track [min, max] across every method that produced a share."""
    tracks = set().union(*(m.keys() for m in per_method if m))
    out = {}
    for t in tracks:
        seen = [m[t] for m in per_method if m and t in m]
        out[t] = (min(seen), max(seen))
    return out


def money_splits(blended: dict[str, float],
                 ranges: dict[str, tuple[float, float]],
                 rights_by_track: dict[str, dict]) -> dict:
    """Writer and publisher shares of the output, with ranges, in percent."""
    writers: dict[str, dict] = {}
    publishers: dict[str, dict] = {}
    masters: dict[str, dict] = {}
    unattributed = 0.0

    def _accumulate(pool: dict, name: str, fraction: float,
                    share: float, lo: float, hi: float) -> None:
        entry = pool.setdefault(name, {"pct": 0.0, "lo": 0.0, "hi": 0.0})
        entry["pct"] += share * fraction / 100.0
        entry["lo"] += lo * fraction / 100.0
        entry["hi"] += hi * fraction / 100.0

    for track_id, share in blended.items():
        rights = rights_by_track.get(track_id)
        if rights is None:
            unattributed += share
            continue
        lo, hi = ranges.get(track_id, (share, share))
        for w in rights["writers"]:
            _accumulate(writers, w["name"], w["share_pct"], share, lo, hi)
        for p in rights["publishers"]:
            _accumulate(publishers, p["name"], p["share_pct"], share, lo, hi)
        for m in rights.get("masters") or []:
            _accumulate(masters, m["name"], m["share_pct"], share, lo, hi)

    def _pool(pool: dict) -> list[dict]:
        # The hi bound sums per-track maxima, which are INDEPENDENT method
        # envelopes — for a party with shares in every track they can sum
        # past 100%, a claim nobody can make about one output (found live:
        # a three-track writer hit 147.99% and the schema validator rightly
        # refused the document). Nobody owns more than the whole; cap it.
        return [{"name": name,
                 "share_pct": round(e["pct"] * 100, 4),
                 "share_range_pct": [round(min(e["lo"], 1.0) * 100, 4),
                                     round(min(e["hi"], 1.0) * 100, 4)]}
                for name, e in sorted(pool.items(),
                                      key=lambda kv: -kv[1]["pct"])]

    return {"writers": _pool(writers), "publishers": _pool(publishers),
            "masters": _pool(masters),
            "unattributed_pct": round(unattributed * 100, 4)}
