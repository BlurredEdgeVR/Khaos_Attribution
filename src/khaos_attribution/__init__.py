"""khaos_attribution — JSON schemas and validators for artist attribution records."""

from khaos_attribution.validation import (
    AttributionValidationError,
    load_schema,
    validate_model_card,
    validate_provenance_record,
)

# THE schema_version values writers must stamp into records/cards.
# Each matches the `const` on `schema_version` in the corresponding bundled schema;
# any other value is rejected by validation.
PROVENANCE_SCHEMA_VERSION = "1.0.0"
MODEL_CARD_SCHEMA_VERSION = "1.0.0"

__all__ = [
    "AttributionValidationError",
    "MODEL_CARD_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "load_schema",
    "validate_model_card",
    "validate_provenance_record",
]

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("khaos_attribution")
except Exception:  # pragma: no cover - package not installed (e.g. run from a checkout)
    __version__ = "0.4.1"
