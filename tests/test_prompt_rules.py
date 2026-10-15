"""A prompt asks for a sound, never for another artist: names to imitate are refused however they are capitalised, the
model's own artist is not, and genres, places, eras, instruments and occasions are never taken for names."""

import pytest

from khaos_attribution import prompt_rules as R


@pytest.mark.parametrize("prompt", [
    "a ballad in the style of Adele", "synth pop that sounds like Daft Punk", "sung like Freddie Mercury over piano",
    "the voice of Amy Winehouse on a slow blues", "a dance track featuring Dua Lipa", "ft. Drake, trap beat",
    "a cover of Bohemian Rhapsody", "imitate Taylor Swift", "reminiscent of Radiohead", "by the band Coldplay",
    "inspired by Hans Zimmer", "Sounds like Adele", "SOUNDS LIKE Adele", "Voice of Adele", "Similar to Daft Punk",
    "Inspired By Adele", "Adele-style ballad", "an Adele style ballad", "Adele's voice over strings", "with Billie Eilish vocals",
    "vocals like Adele", "Drake type beat", "Kanye West type beat", "as if sung by Adele", "influenced by Prince",
    "styled after Bjork", "emulate Sade", "channelling Bowie", "in the style of 2Pac", "a copy of Halo",
])
def test_naming_someone_to_imitate_is_refused(prompt):
    assert R.imitation_hits(prompt) and R.refusal(prompt).startswith("Describe the sound you want")


@pytest.mark.parametrize("prompt", [
    "warm analogue pads that sound like rain on glass", "a slow piano piece, sparse and hushed", "drums played like a heartbeat",
    "in the style of a lullaby", "a voice of quiet longing over strings", "Berlin techno at 128 BPM",
    "in the style of Detroit techno", "in the style of Motown", "in the style of Baroque music", "in the style of Bollywood",
    "in the vein of Afrobeat", "the style of New Orleans jazz", "in the style of UK garage", "style of K-pop",
    "inspired by J-Pop", "inspired by Japanese city pop", "inspired by Irish folk", "inspired by Berlin nights",
    "reminiscent of Italo disco", "similar to Eurodance", "a remix of Latin house", "featuring Rhodes piano",
    "feat. Moog bass", "ft. Strings", "sounds like TR-808 drums", "in the spirit of Christmas", "the voice of God",
    "the singing of Gregorian monks", "A Ballad In The Style Of A Lullaby", "played by the band live in a small club",
    "vocals by the singer herself", "sounds like the producer had fun", "a groove like the DJ would drop at 3am",
    "by the artist himself", "in the style of 80s synthpop", "sounds like the 90s",
])
def test_a_description_is_left_alone(prompt):
    assert R.imitation_hits(prompt) == [] and R.refusal(prompt) is None


def test_the_models_own_artist_may_be_named_and_nothing_wider():
    assert R.refusal("a new song in the style of Chris Green", allowed=("Chris Green", "chris_green")) is None
    assert R.refusal("in the style of Artist A", allowed=("Artist A", "artist-a")) is None
    assert R.refusal("in the style of Adele", allowed=("Chris Green",))
    assert R.refusal("in the style of the Beatles", allowed=("style",)), "an own name that is a trigger word exempts nothing"
    assert R.refusal("the voice of Adele", allowed=("of",))
    assert R.refusal("in the style of Chris Brown", allowed=("Chris",)), "a one-word own name never exempts a longer name"


def test_one_phrase_is_said_once():
    assert R.imitation_hits("in the style of Adele") == ["in the style of Adele"]
