# khaos_attribution

JSON schemas and validators for artist attribution. The package bundles five
schemas — a **provenance record** attached to every generated output, a
**model card** describing an artist adapter, **track rights**, an
**attribution estimate**, and an output **tombstone** — plus Python
functions that validate records against them using
[jsonschema](https://python-jsonschema.readthedocs.io/), and the shared
implementation modules the two apps pin: `watermark` (AudioSeal v2 —
run-level SECDED codewords, retirement, the recorded reset history),
`fingerprint`, `estimator` and `blend` (the attribution estimate and
its blend of signals), `diagnostics` (whether that estimate's similarity
signal varied with the output at all, or returns the same tracks whatever
was generated) and `aspects` (per-aspect shares — harmony, rhythm, timbre —
which are descriptor agreement, never causal influence), `embedding`,
`lyric_align`, `prompt_expansion` (Simple mode's rules and exemplars),
and `catalogue` (the artist's usual tempo/key/time signature).

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
| `watermark_id` | *(optional)* A small whole number between 0 and 65535 embedded in the audio watermark (AudioSeal, v2: the adapter RUN's ID from its model card), tying the file back to this record. `null` when the serving machine could not mark the output — the `/verify` surfaces say so loudly. |
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

The model card and provenance record reject unknown fields; the attribution
estimate's `method` block deliberately does NOT (`additionalProperties: true`),
which is where `reliability` and `aspects` ride — the document's top level is
closed and its `schema_version` is a const, so a key there would invalidate
every estimate ever written. Both schemas otherwise reject unknown fields, so typos in field names fail validation
rather than passing silently.

## What the attribution estimate carries about its own signal

Since estimator 0.5.0 `method.similarity_scores` stores the raw per-track
cosines, so a producer can hand the earlier documents for an adapter to
`estimator.recent_similarity_from_documents` and the next estimate can judge
whether the similarity signal varies across outputs at all
(`method.reliability`); when that verdict is `collapsed` the influence
shares fall back to the exposure prior — the number changes, not just the
footnote.

## The Guild seal

`khaos_attribution.seal` draws the circular Guild stamp — a registration
number set on an arc around the Khaos star — as dependency-free SVG:
`render_seal("123456")` returns the source, `python -m khaos_attribution.seal
123456 out.svg` writes it, and `seal_for_provenance(document)` draws it for a
verified provenance document and **refuses** unless
`embedded_agrees_with_sidecar` is `True`. Type is vector paths from a
committed glyph table (Red Hat Display Bold, OFL — licence beside it); there
is no `<text>`, no font and nothing fetched at runtime, and the module's
`SPEC.md` says why that must stay so. Rebuild the table with
`tools/build_glyphs.py` only when the charset or the font changes.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```
