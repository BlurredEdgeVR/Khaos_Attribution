"""Assemble one attribution estimate document — the shared glue.

Both the Workshop (from its live training store) and the Listening Space
(from the attribution bundle an artist import carries) compute estimates.
The math lives in blend.py; THIS module owns everything around it — the
caveat wording, the rounding discipline, the method stamp, the validation
gate — because two apps producing documents that differ in anything but
their data source would be drift wearing a contract's name.

The caller supplies data it alone knows how to load (embeddings, rights,
run metadata, the output's embedding); this function does the rest and
returns a document that has already passed validate_attribution_estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from khaos_attribution import blend
from khaos_attribution.validation import validate_attribution_estimate

BASE_CAVEAT = ("This is an estimate with the stated method, "
               "not a legal statement of ownership.")


def build_estimate(*, generation_id: str, artist_id: str,
                   adapter_version: str,
                   run_tracks: set[str],
                   segment_counts: dict[str, int],
                   n_training_pairs: int | None,
                   embeddings: np.ndarray,
                   row_track_ids: list[str],
                   output_embedding: np.ndarray,
                   rights: dict[str, dict],
                   fallback_titles: dict[str, str | None],
                   embedding_model: str | None,
                   embedding_version: str | None,
                   exposure_basis: str,
                   extra_caveats: tuple[str, ...] = ()) -> dict:
    """The full pipeline after embedding: exposure → similarity → blend →
    ranges → money → validated document.

    Preconditions the caller owns: every value in `rights` must be a
    schema-valid track_rights record (validate_track_rights) — money math
    reads writers/publishers/title directly; and `exposure_basis` names the
    data source, the ONE field that legitimately differs between apps.

    Raises ValueError when no training track has embeddings, or when the
    embedding index and array disagree — a torn bundle must fail loudly,
    not misattribute quietly.
    """
    if len(row_track_ids) != len(embeddings):
        raise ValueError(
            f"Embedding index lists {len(row_track_ids)} rows but the array "
            f"holds {len(embeddings)} — the store/bundle is torn (files "
            f"copied at different times?). Refusing to estimate from it.")
    caveats = [BASE_CAVEAT, *extra_caveats]

    missing = sorted(run_tracks - set(segment_counts))
    if missing:
        caveats.append(
            f"{len(missing)} of {len(run_tracks)} training tracks have no "
            f"stored embeddings and are excluded from the estimate: "
            f"{', '.join(missing[:5])}" + ("…" if len(missing) > 5 else ""))
    if n_training_pairs and sum(segment_counts.values()) != n_training_pairs:
        caveats.append(
            f"The embedding store holds {sum(segment_counts.values())} "
            f"segments for these tracks but the run trained on "
            f"{n_training_pairs} pairs — the store has changed since "
            f"training, and exposure shares reflect the store as it is now.")
    exposure = blend.exposure_shares(segment_counts)
    if not exposure:
        raise ValueError(f"No embeddings for any training track of "
                         f"{adapter_version} — nothing to estimate from")

    scores = blend.similarity_scores(output_embedding, embeddings,
                                     row_track_ids, set(exposure))
    sim_weights, temperature = blend.similarity_weights(scores)
    if sim_weights is None:
        caveats.append(
            "Single-track adapter: influence is the whole output by "
            "construction; similarity adds nothing."
            if len(run_tracks) == 1 else
            "Acoustic similarity was uninformative (score spread below the "
            "noise floor); blended shares equal the exposure prior.")

    blended = blend.blend_shares(exposure, sim_weights)
    ranges = blend.disagreement_ranges([exposure, sim_weights or {}, blended])

    without_rights = [t for t in blended if t not in rights]
    if without_rights:
        caveats.append(
            f"{len(without_rights)} of {len(blended)} influencing tracks "
            f"have no rights record; their influence is reported as "
            f"unattributed, not redistributed.")
    splits = blend.money_splits(blended, ranges, rights)

    titles = {t: (rights[t]["title"] if t in rights
                  else fallback_titles.get(t))
              for t in blended}

    blended_pct = blend.largest_remainder_pcts(blended)
    influence = []
    for track_id, share in sorted(blended.items(), key=lambda kv: -kv[1]):
        lo, hi = ranges[track_id]
        influence.append({
            "track_id": track_id,
            "title": titles[track_id],
            "exposure_share_pct": round(exposure[track_id] * 100, 4),
            "similarity_share_pct": (round(sim_weights[track_id] * 100, 4)
                                     if sim_weights else None),
            "blended_share_pct": blended_pct[track_id],
            "share_range_pct": [round(lo * 100, 4), round(hi * 100, 4)],
        })

    document = {
        "schema_version": "1.0.0",
        "generation_id": generation_id,
        "artist_id": artist_id,
        "adapter_version": adapter_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "estimator_version": blend.ESTIMATOR_VERSION,
            "exposure_basis": exposure_basis,
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "similarity_informative": sim_weights is not None,
            "temperature": temperature,
        },
        "influence": influence,
        "splits": splits,
        "caveats": caveats,
    }
    return validate_attribution_estimate(document)
