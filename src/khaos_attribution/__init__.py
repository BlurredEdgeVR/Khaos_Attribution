"""khaos_attribution — JSON schemas and validators for artist attribution records."""

from khaos_attribution.validation import (
    AttributionValidationError,
    load_schema,
    validate_model_card,
    validate_provenance_record,
)

__all__ = [
    "AttributionValidationError",
    "load_schema",
    "validate_model_card",
    "validate_provenance_record",
]

__version__ = "0.2.0"
