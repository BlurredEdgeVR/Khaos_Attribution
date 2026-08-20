import copy

import pytest

from khaos_attribution import (
    MODEL_CARD_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    AttributionValidationError,
    load_schema,
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


def test_derived_output_lineage_fields_pass():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["generation_mode"] = "retake"
    record["source_generation_id"] = "a1b2c3d4e5f60718"
    validate_provenance_record(record)


def test_fresh_generation_lineage_defaults_pass():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["generation_mode"] = "text2music"
    record["source_generation_id"] = None
    validate_provenance_record(record)


# A literal record exactly as writers produced it under schema package v0.1.0:
# no generation_mode, no source_generation_id, no watermark_id keys.
# Frozen deliberately — do not "modernise" it; it guards backwards compatibility.
V010_PROVENANCE_RECORD = {
    "schema_version": "1.0.0",
    "artist_name": "Ava Example",
    "artist_id": "artist-0001",
    "adapter_version": "0.1.0",
    "adapter_hash": "a3f5c9d871be4402913377aa5cc0e1b2",
    "prompt": "A slow ambient piece with tape hiss and distant piano.",
    "timestamp": "2026-01-15T12:00:00Z",
    "base_model": "khaos-audio-base-1",
    "base_model_licence": "CC-BY-NC-4.0",
}


def test_v010_record_without_lineage_fields_still_valid():
    # Records written before v0.2.0 must keep validating unchanged.
    validate_provenance_record(copy.deepcopy(V010_PROVENANCE_RECORD))


def test_unknown_generation_mode_rejected():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["generation_mode"] = "remix"
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_provenance_record(record)
    assert "generation_mode" in str(excinfo.value) or "remix" in str(excinfo.value)


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


def test_wrong_provenance_schema_version_rejected():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["schema_version"] = "9.9.9"
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_provenance_record(record)
    assert "schema_version" in str(excinfo.value)


def test_wrong_model_card_schema_version_rejected():
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["schema_version"] = "9.9.9"
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_model_card(card)
    assert "schema_version" in str(excinfo.value)


def test_exported_constants_match_schema_const():
    provenance_schema = load_schema("provenance_record.schema.json")
    card_schema = load_schema("model_card.schema.json")
    assert (
        PROVENANCE_SCHEMA_VERSION
        == provenance_schema["properties"]["schema_version"]["const"]
    )
    # The card schema accepts BOTH versions mid-migration (watermarking
    # v2); the exported constant is what writers stamp — the newest.
    accepted = card_schema["properties"]["schema_version"]["enum"]
    assert MODEL_CARD_SCHEMA_VERSION == max(accepted)
    assert "1.0.0" in accepted


def test_catalogue_vocal_flag_optional_and_boolean():
    # 0.4.1: 'vocal' is optional (older cards predate it) but must be a
    # boolean when present — no third state sneaks into the contract.
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["training_catalogue"] = [
        {"title": "Sung one", "duration": 201.5, "vocal": True},
        {"title": "Instrumental one", "duration": 187.0, "vocal": False},
        {"title": "Pre-0.4.1 entry", "duration": 90.0},
    ]
    validate_model_card(card)

    card["training_catalogue"][0]["vocal"] = "yes"
    with pytest.raises(AttributionValidationError):
        validate_model_card(card)


def test_tombstone_validates_and_refuses_garbage():
    from khaos_attribution import AttributionValidationError, validate_tombstone
    good = {"schema_version": "1.0.0", "generation_id": "g1",
            "artist_id": "art", "content_sha256": "0" * 64,
            "deleted_at_utc": "2026-08-20T12:00:00Z",
            "watermark_id": 1782, "run_id": "run_x",
            "fingerprint_version": "v1", "fingerprints": [1, 2, 3]}
    validate_tombstone(good)
    import pytest
    with pytest.raises(AttributionValidationError):
        validate_tombstone({**good, "content_sha256": "short"})
    with pytest.raises(AttributionValidationError):
        validate_tombstone({k: v for k, v in good.items()
                            if k != "deleted_at_utc"})


def test_model_card_accepts_watermark_id_and_both_versions():
    # 1.1.0 adds the optional field; 1.0.0 cards (no field) must keep
    # validating so cards sync between machines mid-migration.
    base = copy.deepcopy(GOOD_MODEL_CARD)
    validate_model_card(base)                       # 1.0.0, no watermark_id
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["schema_version"] = "1.1.0"
    card["watermark_id"] = 52464
    validate_model_card(card)
    bad = copy.deepcopy(GOOD_MODEL_CARD)
    bad["watermark_id"] = 99999999
    with pytest.raises(AttributionValidationError):
        validate_model_card(bad)
