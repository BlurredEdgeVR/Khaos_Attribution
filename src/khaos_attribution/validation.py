"""Validators for the provenance record and model card schemas."""

import json
from importlib import resources

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import best_match


class AttributionValidationError(ValueError):
    """Raised when a record does not match its schema."""


def load_schema(name):
    """Load a bundled schema by filename, e.g. 'provenance_record.schema.json'."""
    schema_file = resources.files("khaos_attribution").joinpath("schemas", name)
    return json.loads(schema_file.read_text(encoding="utf-8"))


def _validate(record, schema_name, record_kind):
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    if not errors:
        return record

    lines = []
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path) or "(top level)"
        lines.append(f"  - at {location}: {error.message}")
    primary = best_match(errors)
    raise AttributionValidationError(
        f"Invalid {record_kind}: {primary.message}\n"
        f"All problems:\n" + "\n".join(lines)
    )


def validate_provenance_record(record):
    """Validate a provenance record dict; raise AttributionValidationError if invalid."""
    return _validate(record, "provenance_record.schema.json", "provenance record")


def validate_model_card(record):
    """Validate a model card dict; raise AttributionValidationError if invalid."""
    return _validate(record, "model_card.schema.json", "model card")
