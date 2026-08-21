"""The lyric-alignment contract: word tokens with exact character offsets,
a splice that touches only the selected span, and the sung/instrumental
predicate both apps share. The aligner itself (torchaudio) is exercised
only when installed."""

import pytest

from khaos_attribution.lyric_align import (
    LYRICS_VERSION,
    available,
    is_sung,
    splice_words,
    tokenize_lyrics,
)


def test_tokens_carry_exact_offsets_and_skip_tags():
    text = "[Verse]\nthe night is  falling down\n\n[Chorus]\nhold on"
    words = tokenize_lyrics(text)
    assert [w["word"] for w in words] == ["the", "night", "is", "falling", "down", "hold", "on"]
    for w in words:
        assert text[w["char_start"]:w["char_end"]] == w["word"]


def test_splice_replaces_only_the_span():
    text = "[Verse]\nthe night is falling down\n[Chorus]\nhold on"
    words = tokenize_lyrics(text)
    assert splice_words(text, words, 1, 3, "morning comes") == \
        "[Verse]\nthe morning comes down\n[Chorus]\nhold on"
    with pytest.raises(ValueError):
        splice_words(text, words, 3, 1, "x")
    with pytest.raises(ValueError):
        splice_words(text, words, 0, 99, "x")


def test_is_sung_treats_tag_only_text_as_instrumental():
    assert is_sung("la la") is True
    assert is_sung("[Instrumental]") is False
    assert is_sung("") is False
    assert is_sung(None) is False
    assert is_sung("[Verse]\n[Chorus]") is False


def test_version_and_availability_are_plain_values():
    assert LYRICS_VERSION == "1.0.0"
    assert isinstance(available(), bool)


@pytest.mark.skipif(not available(), reason="torchaudio aligner not installed")
def test_align_file_places_words_monotonically(tmp_path):
    import numpy as np
    import soundfile as sf
    from khaos_attribution.lyric_align import align_file
    # Two seconds of shaped noise: no real words, but the mechanics must hold —
    # every word placed or interpolated, monotonic, inside the clip.
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 2) * 0.05).astype("float32")
    sf.write(str(tmp_path / "x.wav"), audio, 16000)
    doc = align_file(tmp_path / "x.wav", "hello there friend")
    starts = [w["start"] for w in doc["words"]]
    assert len(doc["words"]) == 3
    assert all(a <= b for a, b in zip(starts, starts[1:]))
    assert all(0 <= w["start"] <= w["end"] <= doc["duration"] + 1e-6 for w in doc["words"])
