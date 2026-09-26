"""Measured influence: the D-TRAK kernel over an adapter's gradient index,
per-track shares, and the estimate document both rooms write from them.

The index is built once per adapter where the training pairs are; scoring
needs only the index and the output's projected gradient. Influence here is
what a retraining ground truth measured it to be (``VALIDATION``), on the
catalogue it was measured on; resemblance (CLAP) is a different quantity
and travels beside it under its own name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from khaos_attribution import ATTRIBUTION_ESTIMATE_SCHEMA_VERSION, blend
from khaos_attribution.validation import validate_attribution_estimate

INFLUENCE_METHOD = "dtrak"
INFLUENCE_VERSION = "0.1.0"
FEATURE_TARGET = "vnorm"
LAMBDA_MULTIPLE = 10.0
PROJECTION_DIM = 8192
TIMESTEPS_PER_PAIR = 10
# The per-track error the validation measured; every share's range is ± this.
MEASURED_ERROR_PP = 3.3
INDEX_MATRIX = "gradients.npy"
INDEX_RECORD = "index.json"
INDEX_SCHEMA = "khaos.gradient_index/0.1.0"
NORM_EPS = 1e-12
UNASSIGNED = "unassigned"

VALIDATION = {
    "test": "T2 v2: per-output shares against a half-subset retraining regression, pre-registered",
    "catalogue": "chris_green_dataset_theleap",
    "money_on_the_right_tracks": 0.802,
    "interval": [0.797, 0.807],
    "training_share_rule": 0.695,
    "slope": 1.00,
    "record_content_hash": "6caa44b1bf9a79e1",
    "record_commit": "9b28e8e",
    "standing": "confirmatory",
    "scope": "one catalogue of twelve tracks, one composer; a 20-epoch calibration adapter's final "
             "weights, not yet a served recipe",
}

BASE_CAVEAT = (
    "Measured influence: how much of this output's adapter gradient each training track "
    "explains, read by the D-TRAK kernel over the adapter's gradient index. In a pre-registered "
    "test on one catalogue it put 80% of the money on the right tracks, against 70% for a flat "
    "split by training share. It is not resemblance, and it has not been validated on every "
    "adapter recipe.")


class InfluenceRefused(ValueError):
    """The index or the query cannot be scored, with the reason."""


@dataclass
class InfluenceIndex:
    """The index's raw projected gradients, one row per training pair."""
    phi: np.ndarray
    pair_ids: list
    track_ids: list
    invalid: np.ndarray
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.phi.shape[1]) if self.phi.ndim == 2 else 0


def load_index(index_dir) -> InfluenceIndex:
    """Read ``gradients.npy`` + ``index.json``; rows are rebuilt to their raw
    scale from the stored norm, and a zero or non-finite row is marked invalid."""
    d = Path(index_dir)
    record_path, matrix_path = d / INDEX_RECORD, d / INDEX_MATRIX
    if not record_path.is_file() or not matrix_path.is_file():
        raise InfluenceRefused(f"no gradient index under {d}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != INDEX_SCHEMA:
        raise InfluenceRefused(f"index at {d} is schema {record.get('schema')!r}, not {INDEX_SCHEMA!r}")
    rows = list(record.pop("rows", []))
    matrix = np.load(str(matrix_path))
    if matrix.ndim != 2 or matrix.shape[0] != len(rows):
        raise InfluenceRefused(f"index at {d} has a matrix that does not match its rows")
    norms = np.asarray([float(r.get("gradient_norm", float("nan"))) for r in rows], dtype=np.float64)
    invalid = ~np.isfinite(norms) | (norms <= NORM_EPS)
    phi = np.asarray(matrix, dtype=np.float64) * np.where(invalid, 0.0, norms)[:, None]
    return InfluenceIndex(phi=phi, pair_ids=[r["pair_id"] for r in rows],
                          track_ids=[r["track_id"] for r in rows], invalid=invalid, meta=record)


def index_sha256(index_dir) -> str:
    """One hash over the matrix bytes and the record, so a bundle can name the exact index."""
    d = Path(index_dir)
    h = hashlib.sha256()
    for name in (INDEX_MATRIX, INDEX_RECORD):
        h.update((d / name).read_bytes())
    return h.hexdigest()


def kernel_scores(phi: np.ndarray, q: np.ndarray, lam: float) -> np.ndarray:
    """ϕ(x)ᵀ (ΦᵀΦ + λI)⁻¹ Φᵀ in the dual (n × n) form Φᵀ(ΦΦᵀ + λI)⁻¹, one
    score per training pair."""
    phi = np.asarray(phi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if phi.ndim != 2 or phi.shape[1] != q.size:
        raise InfluenceRefused(f"features are {phi.shape}, the query has {q.size} entries")
    if lam <= 0:
        raise InfluenceRefused("lambda must be positive")
    gram = phi @ phi.T + float(lam) * np.eye(phi.shape[0])
    return np.linalg.solve(gram, phi @ q)


def lambda_for(index: InfluenceIndex, multiple: float = LAMBDA_MULTIPLE) -> float:
    """λ as a multiple of the mean squared row norm of the valid rows."""
    valid = index.phi[~index.invalid]
    scale = float(np.mean(np.sum(valid ** 2, axis=1))) if len(valid) else 1.0
    return float(multiple) * scale


def per_track(scores: Sequence[float], track_ids: Sequence[str]) -> dict:
    """Each track's total over its pairs."""
    out: dict = {}
    for s, t in zip(scores, track_ids):
        out[t] = out.get(t, 0.0) + float(s)
    return out


def shares(totals: Mapping[str, float]) -> dict:
    """Positive totals normalised to one; a track that did not help gets zero."""
    pos = {t: max(0.0, float(v)) for t, v in totals.items()}
    s = sum(pos.values())
    return {t: (v / s if s > 0 else 0.0) for t, v in pos.items()}


def writer_shares(track_shares: Mapping[str, float], rights_by_track: Mapping[str, dict] | None) -> dict:
    """Track shares through each track's writer split to writer shares; a
    track without writers counts as unassigned."""
    out: dict = {}
    for track, share in track_shares.items():
        writers = ((rights_by_track or {}).get(track) or {}).get("writers") or []
        total = sum(float(w.get("share_pct") or 0.0) for w in writers)
        if not writers or total <= 0:
            out[UNASSIGNED] = out.get(UNASSIGNED, 0.0) + float(share)
            continue
        for w in writers:
            out[w["name"]] = out.get(w["name"], 0.0) + float(share) * float(w.get("share_pct") or 0.0) / total
    return out


def attribute(index: InfluenceIndex, feature: Sequence[float], *, multiple: float = LAMBDA_MULTIPLE) -> dict:
    """One output's per-pair scores, per-track totals and shares under the kernel."""
    f = np.asarray(feature, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(f)):
        raise InfluenceRefused("the output's feature is non-finite")
    lam = lambda_for(index, multiple)
    scores = np.where(index.invalid, 0.0, kernel_scores(index.phi, f, lam))
    totals = per_track(scores, index.track_ids)
    return {"scores": scores, "track_totals": totals, "shares": shares(totals), "lam": lam,
            "lambda_multiple": float(multiple)}


def exposure_from_index(index: InfluenceIndex) -> dict:
    """Each track's share of the index's valid rows: the exposure prior the document shows beside influence."""
    counts: dict = {}
    for t, bad in zip(index.track_ids, index.invalid):
        if not bad:
            counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values())
    return {t: n / total for t, n in counts.items()} if total else {}


def resemblance_block(clap_document: Mapping, influence_shares: Mapping[str, float]) -> dict:
    """The CLAP estimate read beside influence: its similarity shares (none
    when the signal was uninformative: the blend is then only the exposure
    prior), and how far the two splits agree."""
    informative = bool(clap_document.get("method", {}).get("similarity_informative"))
    clap = ({t["track_id"]: float(t["similarity_share_pct"]) / 100.0
             for t in clap_document.get("influence", []) if t.get("similarity_share_pct") is not None}
            if informative else {})
    tracks = sorted(set(influence_shares) | set(clap))
    a = [float(influence_shares.get(t, 0.0)) for t in tracks]
    b = [clap.get(t, 0.0) for t in tracks]
    money = (1.0 - 0.5 * sum(abs(x - y) for x, y in zip(a, b))) if clap else None
    return {"method": "clap_blend", "estimator_version": str(clap_document.get("method", {}).get("estimator_version")),
            "similarity_informative": informative,
            "shares_pct": {t: round(v * 100, 4) for t, v in clap.items()},
            "money_agreement_with_influence": (round(money, 4) if money is not None else None),
            "rank_agreement_with_influence": (_spearman(a, b) if clap else None),
            "note": "resemblance to the training audio, not influence on this output"}


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = _ranks(x), _ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def build_influence_estimate(*, generation_id: str, artist_id: str, adapter_version: str,
                             track_totals: Mapping[str, float], rights: Mapping[str, dict],
                             fallback_titles: Mapping[str, str | None],
                             index: InfluenceIndex, index_sha256: str, adapter_sha256: str | None,
                             lam: float, lambda_multiple: float = LAMBDA_MULTIPLE,
                             dataset_hash: str | None = None, dataset_hash_note: str | None = None,
                             resemblance: Mapping | None = None,
                             extra_caveats: Sequence[str] = ()) -> dict:
    """Shares → ranges → money → a validated document with ``method.kind`` ``dtrak``."""
    share = shares(track_totals)
    if not share:
        raise InfluenceRefused("no track totals to build an estimate from")
    if sum(share.values()) <= 0:
        raise InfluenceRefused("no training track has positive influence on this output; nothing to split")
    caveats = [BASE_CAVEAT, *extra_caveats]
    if dataset_hash_note:
        caveats.append(dataset_hash_note)
    without_rights = [t for t in share if t not in rights]
    if without_rights:
        caveats.append(f"{len(without_rights)} of {len(share)} influencing tracks have no rights record; "
                       "their influence is reported as unattributed, not redistributed.")
    err = MEASURED_ERROR_PP / 100.0
    ranges = {t: (max(0.0, v - err), min(1.0, v + err)) for t, v in share.items()}
    splits = blend.money_splits(share, ranges, rights)
    pct = blend.largest_remainder_pcts(share)
    exposure = exposure_from_index(index)
    resemblance_shares = {t: v / 100.0 for t, v in (resemblance or {}).get("shares_pct", {}).items()}
    influence = []
    for track_id, v in sorted(share.items(), key=lambda kv: -kv[1]):
        lo, hi = ranges[track_id]
        influence.append({
            "track_id": track_id,
            "title": rights[track_id]["title"] if track_id in rights else fallback_titles.get(track_id),
            "exposure_share_pct": round(exposure.get(track_id, 0.0) * 100, 4),
            "similarity_share_pct": (round(resemblance_shares[track_id] * 100, 4)
                                     if track_id in resemblance_shares else None),
            "influence_share_pct": pct[track_id],
            "blended_share_pct": pct[track_id],
            "share_range_pct": [round(lo * 100, 4), round(hi * 100, 4)],
        })
    meta = index.meta
    document = {
        "schema_version": ATTRIBUTION_ESTIMATE_SCHEMA_VERSION,
        "generation_id": generation_id,
        "artist_id": artist_id,
        "adapter_version": adapter_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "kind": INFLUENCE_METHOD,
            "estimator_version": INFLUENCE_VERSION,
            "exposure_basis": "training pairs per track, from the index rows",
            "embedding_model": None,
            "embedding_version": None,
            "similarity_informative": bool((resemblance or {}).get("similarity_informative", False)),
            "temperature": None,
            "feature_target": str(meta.get("target", FEATURE_TARGET)),
            "lambda_multiple": float(lambda_multiple),
            "lambda": float(lam),
            "projection_dim": int(index.dim),
            "timesteps": list(meta.get("timesteps") or []),
            "noise_seed": meta.get("noise_seed"),
            "n_pairs": int(len(index.pair_ids)),
            "n_tracks": int(len(set(index.track_ids))),
            "index_sha256": index_sha256,
            "adapter_sha256": adapter_sha256,
            "dataset_hash": dataset_hash,
            "measured_error_pp": MEASURED_ERROR_PP,
            "validation": dict(VALIDATION),
        },
        "influence": influence,
        "splits": splits,
        "caveats": caveats,
    }
    if resemblance is not None:
        document["resemblance"] = dict(resemblance)
    return validate_attribution_estimate(document)
