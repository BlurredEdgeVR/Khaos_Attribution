"""The prompt-expansion contract: exemplar file shapes, relevance ranking,
retry rotation, the query, the tidy-up, the trigger rule and the
normalisation of the LM's whole answer. No torch."""

import pytest

from khaos_attribution.prompt_expansion import (
    CAPTION_WORDS,
    FEW_SHOT_COUNT,
    MAX_CAPTION_CHARS,
    build_query,
    clip_caption,
    harvest,
    lexical_score,
    normalise_exemplars,
    normalise_keyscale,
    normalise_timesignature,
    rank_exemplars,
    select_exemplars,
    strip_timestamps,
    with_trigger,
)

INTRO = "The piece opens with a soft ambient synth pad that creates a dreamy atmosphere."
GROOVE = "Driving percussion under a hypnotic, dark bass groove gives the track a restless energy."
VOCAL = "A haunting female vocal repeats a short melody over deep resonant synthesizers."


def test_normalise_reads_both_file_versions():
    old = normalise_exemplars({"exemplars": [INTRO, "  ", GROOVE]})
    assert [e["text"] for e in old] == [INTRO, GROOVE]
    assert all(e["vector"] is None for e in old)
    new = normalise_exemplars({"exemplars": [
        {"text": f"  {VOCAL}  ", "section": "chorus", "track": "t1", "vector": [1.0, 0.0]},
        {"text": ""}, 7]})
    assert new == [{"text": VOCAL, "section": "chorus", "track": "t1", "vector": [1.0, 0.0]}]
    assert normalise_exemplars(None) == []


def test_lexical_ranking_prefers_the_matching_caption():
    ex = normalise_exemplars({"exemplars": [INTRO, GROOVE, VOCAL]})
    order, method = rank_exemplars("dark hypnotic groove", ex)
    assert method == "lexical"
    assert order[0] == 1
    assert lexical_score("dark hypnotic groove", GROOVE) > lexical_score("dark hypnotic groove", INTRO)
    # stems meet: "drums" ~ "drum", "driving" ~ "drive"
    assert lexical_score("driving drums", "a drum that drives") > 0


def test_embedding_ranking_when_vectors_exist():
    ex = normalise_exemplars({"exemplars": [
        {"text": INTRO, "vector": [1.0, 0.0]},
        {"text": GROOVE, "vector": [0.0, 1.0]},
    ]})
    order, method = rank_exemplars("anything", ex, request_vector=[0.1, 0.9])
    assert method == "embedding" and order == [1, 0]
    # a request vector of another dimension (another model) → lexical, never
    # file order wearing the "embedding" label
    _, method = rank_exemplars("anything", ex, request_vector=[0.1, 0.9, 0.0])
    assert method == "lexical"
    # one exemplar without a vector → the whole ranking falls back
    ex[0]["vector"] = None
    _, method = rank_exemplars("anything", ex, request_vector=[0.1, 0.9])
    assert method == "lexical"


def test_select_rotates_per_attempt_and_wraps():
    ex = normalise_exemplars({"exemplars": [f"caption number {i} about {w}" for i, w in
                                            enumerate(["rain", "fire", "rain", "sea", "rain"])]})
    first, _ = select_exemplars("rain", ex, attempt=0)
    second, _ = select_exemplars("rain", ex, attempt=1)
    assert len(first) == FEW_SHOT_COUNT and set(first).isdisjoint(second[:2])
    assert set(first) == {0, 2, 4}            # the three rain captions lead
    # wraps around the ranking instead of running out
    third, _ = select_exemplars("rain", ex, attempt=2)
    assert len(third) == FEW_SHOT_COUNT
    assert select_exemplars("x", [], attempt=3) == ([], "lexical")
    # fewer exemplars than a window: every attempt sees them all, the lead rotates
    two = ex[:2]
    assert select_exemplars("x", two, attempt=0)[0] == [0, 1]
    assert select_exemplars("x", two, attempt=1)[0] == [1, 0]
    assert select_exemplars("x", two, attempt=2)[0] == [0, 1]


def test_query_carries_shots_length_and_vocal_stance():
    q = build_query("  rainy morning ", [INTRO, GROOVE], instrumental=True)
    assert f"- {INTRO}" in q and f"- {GROOVE}" in q
    assert "Request: rainy morning" in q
    assert f"{CAPTION_WORDS[0]}–{CAPTION_WORDS[1]} words" in q
    assert "no vocals" in q
    assert "sung vocals" in build_query("x", [INTRO], instrumental=False)


def test_timestamps_are_stripped_and_captions_clipped():
    assert strip_timestamps("Soft pads open. At 1:26, the drums enter, and by 2:10 it peaks.") == \
        "Soft pads open. the drums enter, and it peaks."
    assert strip_timestamps("A build (0:45) rises") == "A build rises"
    long = ("A sentence that is long enough to matter. " * 15).strip()
    clipped = clip_caption(long)
    assert len(clipped) <= MAX_CAPTION_CHARS and clipped.endswith(".")
    assert clip_caption("short") == "short"


def test_trigger_is_added_once_and_respects_track_triggers():
    assert with_trigger("warm dusk", "sng_a") == "sng_a warm dusk"
    assert with_trigger("sng_a warm dusk", "sng_a") == "sng_a warm dusk"
    assert with_trigger("sng_a trk_x warm dusk", "sng_a") == "sng_a trk_x warm dusk"
    assert with_trigger("warm dusk", "sng_a trk_x") == "sng_a trk_x warm dusk"
    assert with_trigger("sng_ab warm", "sng_a") == "sng_a sng_ab warm"  # a different artist's trigger
    assert with_trigger("warm", None) == "warm"


@pytest.mark.parametrize("raw, expected", [
    ("A minor", "A minor"), ("a min", "A minor"), ("F#m", "F# minor"), ("Bb Major", "A# major"),
    ("C", "C major"), ("Db", "C# major"), ("E♭ minor", "D# minor"), ("H major", None), ("", None), (7, None),
])
def test_keyscale_normalisation(raw, expected):
    assert normalise_keyscale(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("4/4", "4"), ("3/4", "3"), ("6/8", "6"), ("4", "4"), (4, "4"), ("13/8", None), ("x", None), (None, None),
])
def test_timesignature_normalisation(raw, expected):
    assert normalise_timesignature(raw) == expected


def test_harvest_normalises_the_whole_answer():
    got = harvest({"caption": "Warm pads. At 0:30 a beat.", "bpm": "124", "keyscale": "a minor",
                   "timesignature": "4/4", "duration": 180.0, "lyrics": "[Instrumental]",
                   "language": "unknown"}, instrumental=True)
    assert got == {"caption": "Warm pads. a beat.", "bpm": 124, "keyscale": "A minor",
                   "timesignature": "4", "duration": 180, "lyrics": None, "language": None}
    sung = harvest({"caption": "x", "bpm": 999, "lyrics": "[Verse]\nhold on\n", "language": "en"},
                   instrumental=False)
    assert sung["bpm"] is None and sung["lyrics"] == "[Verse]\nhold on" and sung["language"] == "en"
    # a sung request whose LM answer carried no words gives no draft
    assert harvest({"caption": "x", "lyrics": "  "}, instrumental=False)["lyrics"] is None
    assert harvest(None)["caption"] == ""
