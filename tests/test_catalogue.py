"""Catalogue defaults: the rows both apps build from the metadata stage and
the defaults they draw from them — confident values only, tempo weighted by
duration, nothing guessed."""

from khaos_attribution.catalogue import (
    CATALOGUE_METADATA_VERSION,
    build_document,
    catalogue_defaults,
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
    assert d["bpm"] == 60                       # 20 votes at 60 vs 2+2 at 120/124
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
