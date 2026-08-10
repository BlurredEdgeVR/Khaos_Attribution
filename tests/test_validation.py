import copy

import pytest

from khaos_attribution import (
    AttributionValidationError,
    validate_model_card,
    validate_provenance_record,
)

GOOD_PROVENANCE_RECORD = {
    "schema_version": "1.0.0",
    "artist_name": "Ava Example",
    "artist_id": "artist-0001",
    "adapter_version": "0.3.1",
    "adapter_hash": "a3f5c9d871be4402913377aa5cc0e1b2",
    "prompt": "A slow ambient piece with tape hiss and distant piano.",
    "timestamp": "2026-08-10T12:00:00Z",
    "base_model": "khaos-audio-base-1",
    "base_model_licence": "CC-BY-NC-4.0",
    "watermark_id": None,
}

GOOD_MODEL_CARD = {
    "schema_version": "1.0.0",
    "artist_name": "Ava Example",
    "artist_id": "artist-0001",
    "consent_statement": "I, Ava Example, consent to an adapter being trained on the listed tracks.",
    "training_catalogue": [
        {"title": "Glasshouse", "duration": 214.5},
        {"title": "Northern Line", "duration": 187.0},
    ],
    "training_date": "2026-07-01",
    "adapter_version": "0.3.1",
    "base_model": "khaos-audio-base-1",
    "base_model_licence": "CC-BY-NC-4.0",
}


def test_good_provenance_record_passes():
    validate_provenance_record(GOOD_PROVENANCE_RECORD)


def test_provenance_record_with_integer_watermark_passes():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["watermark_id"] = 65535
    validate_provenance_record(record)


def test_bad_provenance_record_rejected():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    del record["adapter_hash"]
    record["watermark_id"] = 70000
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_provenance_record(record)
    message = str(excinfo.value)
    assert "adapter_hash" in message
    assert "watermark_id" in message


def test_good_model_card_passes():
    validate_model_card(GOOD_MODEL_CARD)


def test_bad_model_card_rejected():
    card = copy.deepcopy(GOOD_MODEL_CARD)
    del card["consent_statement"]
    card["training_catalogue"] = [{"title": "Glasshouse"}]
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_model_card(card)
    message = str(excinfo.value)
    assert "consent_statement" in message
    assert "duration" in message
