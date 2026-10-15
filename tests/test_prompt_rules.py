"""A prompt asks for a sound, never for another artist: names to imitate are refused, the model's own artist is not,
and a description that merely uses the same words is left alone."""

import pytest

from khaos_attribution import prompt_rules as R


@pytest.mark.parametrize("prompt", [
    "a ballad in the style of Adele",
    "synth pop that sounds like Daft Punk",
    "sung like Freddie Mercury over piano",
    "the voice of Amy Winehouse on a slow blues",
    "a dance track featuring Dua Lipa",
    "ft. Drake, trap beat",
    "a cover of Bohemian Rhapsody",
    "imitate Taylor Swift",
    "reminiscent of Radiohead",
    "by the band Coldplay",
    "inspired by Hans Zimmer",
])
def test_naming_someone_to_imitate_is_refused(prompt):
    assert R.imitation_hits(prompt) and R.refusal(prompt).startswith("Describe the sound you want")


@pytest.mark.parametrize("prompt", [
    "warm analogue pads that sound like rain on glass",
    "a slow piano piece, sparse and hushed",
    "drums played like a heartbeat",
    "in the style of a lullaby",
    "a voice of quiet longing over strings",
    "Berlin techno at 128 BPM",
])
def test_a_description_is_left_alone(prompt):
    assert R.imitation_hits(prompt) == [] and R.refusal(prompt) is None


def test_the_models_own_artist_may_be_named():
    assert R.refusal("a new song in the style of Chris Green", allowed=("Chris Green", "chris_green")) is None
    assert R.refusal("sounds like Chris Green meets Adele", allowed=("Chris Green",)) is None
    assert R.refusal("in the style of Adele", allowed=("Chris Green",))
    assert R.refusal("in the style of Artist A", allowed=("Artist A", "artist-a")) is None
