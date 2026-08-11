# khaos_attribution

JSON schemas and validators for artist attribution. The package bundles two
schemas — a **provenance record** attached to every generated output, and a
**model card** describing an artist adapter — plus Python functions that
validate records against them using [jsonschema](https://python-jsonschema.readthedocs.io/).

This repo is the *data* contract of the Khaos ecosystem; the *visual*
contract (shared design language for the toolkit and listening-space UIs)
lives in the `Khaos_Platform` repo's `DESIGN.md`.

## Installation

Directly from GitHub:

```bash
pip install "git+https://github.com/BlurredEdgeVR/Khaos_Attribution.git"
```

## Usage

```python
from khaos_attribution import validate_provenance_record, validate_model_card

validate_provenance_record(record)   # returns the record if valid
validate_model_card(card)
```

Both functions raise `khaos_attribution.AttributionValidationError` with a
message listing every problem found (missing fields, wrong types, out-of-range
values) when a record does not match its schema.

The raw schema files are bundled with the package and can be loaded with
`khaos_attribution.load_schema("provenance_record.schema.json")`.

## Provenance record fields

A provenance record travels with a generated output and says exactly how it
was made.

| Field | Meaning |
| --- | --- |
| `schema_version` | Which version of the provenance record schema this record follows, written as three numbers like `1.0.0`. Lets readers know what fields to expect. |
| `artist_name` | The human-readable name of the artist whose adapter was used to generate the output. |
| `artist_id` | A stable, unique identifier for that artist. Names can change or clash; this identifier never does. |
| `adapter_version` | The version of the artist's adapter (the fine-tuned add-on to the base model) that produced this output. |
| `adapter_hash` | A cryptographic fingerprint (hex string) of the adapter's weights. Proves exactly which adapter file was used — if the weights change, the hash changes. |
| `prompt` | The text prompt that was given to the model to produce this output. |
| `timestamp` | When the output was generated, in ISO 8601 date-time form, e.g. `2026-08-10T12:00:00Z`. |
| `base_model` | The name and version of the underlying base model the adapter was applied to. |
| `base_model_licence` | The licence the base model is distributed under, so downstream users know what terms apply. |
| `watermark_id` | *(optional)* A small whole number between 0 and 65535 that will be embedded in the audio watermark, tying the file back to this record. Until watermarking is implemented, this is `null` (or omitted entirely). |
| `generation_mode` | *(optional, since v0.2.0)* How the output was produced: `text2music` for a fresh generation (the default when omitted), or `retake`, `repaint`, `extend`, `audio2audio` for outputs derived from an earlier one. |
| `source_generation_id` | *(optional, since v0.2.0)* When the output was derived from an earlier generation, the generation ID of that source output, so lineage can be traced through the registry. `null` (or omitted) for fresh generations. |

## Model card fields

A model card describes one artist adapter: who consented, what it was trained
on, and what it extends.

| Field | Meaning |
| --- | --- |
| `schema_version` | Which version of the model card schema this card follows, written as three numbers like `1.0.0`. |
| `artist_name` | The human-readable name of the artist the adapter represents. |
| `artist_id` | The stable, unique identifier for that artist — the same one used in provenance records. |
| `consent_statement` | A plain-language statement recording that the artist consented to an adapter being trained on their work. |
| `training_catalogue` | The list of tracks the adapter was trained on. Each entry has a `title` (the track's name) and a `duration` (its length in seconds). At least one track is required. |
| `training_date` | The date the adapter was trained, as `YYYY-MM-DD`. |
| `adapter_version` | The version of the adapter this card describes — match it against the `adapter_version` in provenance records. |
| `base_model` | The name and version of the base model the adapter was trained against. |
| `base_model_licence` | The licence the base model is distributed under. |

Both schemas reject unknown fields, so typos in field names fail validation
rather than passing silently.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```
