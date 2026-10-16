"""The base model identifier every record carries, `<engine>@<variant>`, and the one honest reading of records
written before it existed: ACE-Step 1.5 was the only engine Khaos ever ran, so its free-text name and the manifest's
variant are read as its id once, by a migration, and never inferred after that.
"""

from __future__ import annotations

import re

BASE_MODEL_ID_PATTERN = r"^[a-z0-9][a-z0-9.-]*@[a-z0-9][a-z0-9._-]*$"
_ID = re.compile(BASE_MODEL_ID_PATTERN)
ACESTEP_ENGINE = "acestep-1.5"
ACESTEP_VARIANTS = ("turbo", "base")


def is_base_model_id(value) -> bool:
    return isinstance(value, str) and bool(_ID.fullmatch(value))


def make_base_model_id(engine: str, variant: str) -> str:
    out = f"{engine}@{variant}"
    if not is_base_model_id(out):
        raise ValueError(f"not a base model id: {out!r}")
    return out


def split_base_model_id(value: str) -> tuple[str, str]:
    """(engine, variant); raises ValueError for anything that is not an id."""
    if not is_base_model_id(value):
        raise ValueError(f"not a base model id: {value!r}")
    engine, _, variant = value.partition("@")
    return engine, variant


def engine_of(value: str) -> str:
    return split_base_model_id(value)[0]


def legacy_base_model_id(base_model, model_variant=None) -> str | None:
    """The id a record from before the field means. Only ACE-Step 1.5 ever ran: its name in the free text, with the
    variant from the manifest or from the text ("ace-step-1.5 (turbo)", "ACE-Step 1.5 (acestep-v15-base)"); None
    when neither says, which a migration reports rather than guesses."""
    text = str(base_model or "").lower()
    if "ace-step" not in text and "acestep" not in text:
        return None
    variant = str(model_variant or "").lower() or None
    if variant not in ACESTEP_VARIANTS:
        variant = next((v for v in ACESTEP_VARIANTS if v in text), None)
    return make_base_model_id(ACESTEP_ENGINE, variant) if variant else None


def base_model_id_of(record: dict, manifest: dict | None = None) -> str | None:
    """A record's own `base_model_id`, else the legacy reading of its `base_model` with the manifest's variant."""
    own = (record or {}).get("base_model_id")
    if is_base_model_id(own):
        return own
    variant = (manifest or {}).get("model_variant") or (record or {}).get("model_variant")
    return legacy_base_model_id((record or {}).get("base_model") or (manifest or {}).get("base_model"), variant)
