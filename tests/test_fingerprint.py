"""Fingerprint v1: same audio matches (incl. excerpts + mp3), different
audio does not. Synthetic but structured audio — tones with onsets — so
the constellation has real peaks."""

import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from khaos_attribution.fingerprint import (  # noqa: E402
    fingerprint_array,
    is_confident,
    match_votes,
)


def _music_like(seed: int, seconds: float = 12.0, sr: int = 44100):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    audio = np.zeros_like(t)
    for _ in range(24):   # note events with attack/decay envelopes
        f = rng.uniform(80, 2000)
        start = rng.uniform(0, seconds - 1)
        dur = rng.uniform(0.2, 1.0)
        mask = (t >= start) & (t < start + dur)
        env = np.exp(-4 * (t[mask] - start) / dur)
        audio[mask] += env * np.sin(2 * np.pi * f * t[mask])
    return (0.5 * audio / np.max(np.abs(audio))).astype("float32")


def test_same_audio_matches_and_different_does_not():
    a = _music_like(1)
    b = _music_like(2)
    fa = fingerprint_array(a, 44100)
    fb = fingerprint_array(b, 44100)
    assert len(fa) > 50 and len(fb) > 50
    self_votes = match_votes(fa, fa)
    cross_votes = match_votes(fa, fb)
    assert is_confident(self_votes, len(fa))
    assert not is_confident(cross_votes, len(fa))
    assert self_votes > cross_votes * 3


def test_excerpt_matches_with_offset():
    a = _music_like(3, seconds=20.0)
    full = fingerprint_array(a, 44100)
    excerpt = fingerprint_array(a[int(7.3 * 44100):int(13.3 * 44100)], 44100)
    votes = match_votes(excerpt, full)
    assert is_confident(votes, len(excerpt)), (votes, len(excerpt))


def test_survives_mp3(tmp_path):
    import soundfile as sf
    a = _music_like(4)
    wav = tmp_path / "a.wav"
    sf.write(str(wav), a, 44100)
    mp3 = tmp_path / "a.mp3"
    back = tmp_path / "b.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-b:a", "128k", str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        str(back)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg unavailable")
    b, sr = sf.read(str(back), dtype="float32", always_2d=True)
    votes = match_votes(fingerprint_array(b, sr), fingerprint_array(a, 44100))
    assert is_confident(votes, len(fingerprint_array(b, sr)))


def test_repetitive_audio_limitation_is_known_and_dominance_separates():
    """DOCUMENTED LIMITATION: same-tempo click tracks are near-identical
    to the whole landmark-algorithm class (the constellation is the
    transient's, and the tempo locks the offsets) — pairwise confidence
    alone cannot separate them. What DOES separate the true source is
    DOMINANCE: the true candidate out-votes the impostor decisively, so
    the Platform's closed-set matcher requires best >= 2x runner-up.
    This test pins those facts so the limitation stays visible."""
    sr = 44100
    t = np.arange(int(10 * sr)) / sr

    def click_loop(freq, period):
        audio = np.zeros_like(t)
        for k in range(int(10 / period)):
            mask = (t >= k * period) & (t < k * period + 0.05)
            audio[mask] += np.sin(2 * np.pi * freq * t[mask])
        return (0.5 * audio).astype("float32")

    from khaos_attribution.fingerprint import match_stats
    a = fingerprint_array(click_loop(440.0, 0.5), sr)
    true_votes, _ = match_stats(a, fingerprint_array(click_loop(440.0, 0.5), sr))
    cross_votes, cross_distinct = match_stats(
        a, fingerprint_array(click_loop(661.0, 0.5), sr))
    # the limitation: the cross-match IS pairwise-confident...
    assert is_confident(cross_votes, len(a), distinct=cross_distinct)
    # ...but the true source dominates it by well over 2x
    assert true_votes >= 2 * cross_votes, (true_votes, cross_votes)
    # different-tempo loops do not even get that far
    other_votes, other_distinct = match_stats(
        a, fingerprint_array(click_loop(440.0, 0.73), sr))
    assert not is_confident(other_votes, len(a), distinct=other_distinct)
