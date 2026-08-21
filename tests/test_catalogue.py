"""Catalogue defaults: the rows both apps build from the metadata stage and
the defaults they draw from them — confident values only, tempo weighted by
duration, nothing guessed."""

from khaos_attribution.catalogue import (
    CATALOGUE_METADATA_VERSION,
    build_document,
    catalogue_defaults,
    fold_tempo,
    title_from_filename,
    track_row,
)


def _md(bpm, conf, tonic, mode, kconf=1.0, num=4):
    return {"tempo": {"bpm": bpm, "confidence": conf},
            "key": {"tonic": tonic, "mode": mode, "confidence": kconf},
            "time_signature": {"numerator": num, "denominator": 4}}


def test_row_normalises_to_the_pickers_forms():
    row = track_row("t1", "Band - Song", _md(122.3, 0.9, "Ab", "minor", num=4), duration=200)
    assert row["keyscale"] == "G# minor" and row["timesignature"] == "4"
    assert row["octave_resolved"] is False
    assert row["bpm"] == 122.3 and row["duration"] == 200.0
    empty = track_row("t2", None, {})
    assert empty["bpm"] is None and empty["keyscale"] is None and empty["timesignature"] is None
    assert build_document("a", [row, {"no": "id"}])["tracks"] == [row]
    assert build_document("a", [])["schema_version"] == CATALOGUE_METADATA_VERSION


def test_defaults_are_confident_and_duration_weighted():
    rows = [
        track_row("t1", "a", _md(120, 0.9, "A", "minor"), duration=60),
        track_row("t2", "b", _md(124, 0.9, "A", "minor"), duration=60),
        track_row("t3", "c", _md(60, 0.9, "C", "major"), duration=600),   # long track dominates
        track_row("t4", "d", _md(200, 0.4, "F#", "major"), duration=60),  # unsure tempo: out
    ]
    d = catalogue_defaults(build_document("a", rows))
    assert d["bpm"] == 120                      # 20 votes at 60 → folded to 120 — vs 2+2 at 120/124
    assert d["keyscale"] == "A minor"           # F# major counted (key confident), A minor wins 2:1:1
    assert d["timesignature"] == "4"
    assert d["tracks_used"] == 4
    assert catalogue_defaults({"tracks": []}) == {"bpm": None, "keyscale": None,
                                                  "timesignature": None, "tracks_used": 0}
    assert catalogue_defaults(None)["bpm"] is None
    # out-of-range tempos never become a default
    d = catalogue_defaults(build_document("a", [track_row("t", "x", _md(300, 0.9, "A", "minor"))]))
    assert d["bpm"] is None and d["keyscale"] == "A minor"


def test_title_from_filename():
    assert title_from_filename("Port Sulphur Band - An Acquired Taste.wav") == "Port Sulphur Band - An Acquired Taste"
    assert title_from_filename("") is None and title_from_filename(None) is None


def test_tempo_is_folded_into_the_perceived_band():
    assert fold_tempo(60) == 120 and fold_tempo(30) == 120 and fold_tempo(180) == 90
    assert fold_tempo(100) == 100 and fold_tempo(0) == 0
    d = catalogue_defaults(build_document("a", [
        track_row("t1", "a", _md(65, 0.9, "A", "minor"), duration=60),   # half-tempo reading → 130
        track_row("t2", "b", _md(128, 0.9, "A", "minor"), duration=60),
    ]))
    assert d["bpm"] == 129


def test_a_torn_catalogue_degrades_to_no_defaults_not_an_error():
    doc = {"tracks": [
        "not a row", {"no": "id"},
        {"track_id": "t1", "duration": "200", "bpm": "120", "bpm_confidence": "0.9",
         "keyscale": "E# minor", "key_confidence": 1.0, "timesignature": "6/8"},
        {"track_id": "t2", "bpm": 124, "bpm_confidence": None, "keyscale": "A minor", "key_confidence": None},
        {"track_id": "t3", "bpm": float("inf"), "bpm_confidence": 0.9},
    ]}
    d = catalogue_defaults(doc)
    assert d["bpm"] == 120 and d["keyscale"] == "F minor" and d["timesignature"] == "6"
    assert d["tracks_used"] == 1            # t2's unrated tempo/key count for nothing
    assert catalogue_defaults("junk")["bpm"] is None
    assert catalogue_defaults({"tracks": "junk"})["bpm"] is None
    assert fold_tempo(float("inf")) == float("inf") and fold_tempo(float("nan")) != fold_tempo(float("nan"))


def test_a_resolved_tempo_is_trusted_as_read():
    md = _md(168, 0.6, "A", "minor"); md["tempo"]["octave_resolved"] = True
    rows = [track_row("t1", "a", md, duration=60)]
    assert catalogue_defaults(build_document("a", rows))["bpm"] == 168   # not folded to 84
    md2 = _md(168, 0.9, "A", "minor")
    assert catalogue_defaults(build_document("a", [track_row("t2", "b", md2)]))["bpm"] == 84
