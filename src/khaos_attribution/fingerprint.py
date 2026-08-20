"""Landmark audio fingerprints, format v1 (watermarking v2 §5).

Closed-set identification: the corpus is Khaos's own stored outputs, held
in pristine form, and the watermark usually narrows a query to one
model's outputs first — so this is deliberately the SIMPLE, classic
constellation scheme (Wang 2003 family), numpy-only, no scipy/librosa.

Format v1 (stable; the index stores (hash, offset_ms) pairs):

- mono mix, magnitude STFT with a Hann window of ~93 ms and 50% hop,
  computed at the audio's native rate; peaks are mapped to Hz and
  seconds so fingerprints are sample-rate independent.
- spectral peaks: local maxima over a (time, freq) neighbourhood, kept
  above an adaptive floor, densest ~30 per second, 40 Hz–5 kHz.
- landmarks: each peak pairs with up to 5 later peaks 0.1–1.6 s ahead
  within ±1.5 kHz; hash packs (f1, f2, dt) quantised to (9, 9, 7) bits
  → 25-bit integers.
- matching: shared hashes vote on the time offset between query and
  candidate; the winning offset's vote count is the score. A confident
  match needs >= 12 aligned votes AND >= 5% of the query's landmarks.

Heavy deps stay lazy (numpy, soundfile) so the schema package remains
light — the pattern watermark.py set.
"""

FINGERPRINT_VERSION = "v1"

_FREQ_MIN_HZ = 40.0
_FREQ_MAX_HZ = 5000.0
_WINDOW_S = 0.093
_HOP_S = 0.0465
_PEAKS_PER_SEC = 30
_FAN_OUT = 5
_PAIR_DT_S = (0.1, 1.6)
_PAIR_DF_HZ = 1500.0
_FREQ_BITS, _DT_BITS = 9, 7
_MATCH_MIN_VOTES = 12
_MATCH_MIN_FRACTION = 0.05


def _spectral_peaks(audio, sample_rate: int):
    """[(t_seconds, f_hz)] — the constellation."""
    import numpy as np

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    window = int(sample_rate * _WINDOW_S)
    hop = max(1, int(sample_rate * _HOP_S))
    if len(audio) < window:
        return []
    hann = np.hanning(window)
    n_frames = 1 + (len(audio) - window) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, window),
        strides=(audio.strides[0] * hop, audio.strides[0])).copy()
    mags = np.abs(np.fft.rfft(frames * hann, axis=1))
    freqs = np.fft.rfftfreq(window, 1.0 / sample_rate)
    band = (freqs >= _FREQ_MIN_HZ) & (freqs <= _FREQ_MAX_HZ)
    mags = mags[:, band]
    freqs = freqs[band]

    # local maxima over a 3x3 (time x freq-neighbourhood) grid
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(mags, 1, mode="constant")
    neigh = sliding_window_view(padded, (3, 3)).max(axis=(2, 3))
    is_peak = (mags >= neigh) & (mags > 0)

    # adaptive floor + density cap: strongest peaks per one-second block
    duration = n_frames * hop / sample_rate
    budget = max(8, int(duration * _PEAKS_PER_SEC))
    t_idx, f_idx = np.nonzero(is_peak)
    if len(t_idx) == 0:
        return []
    order = np.argsort(mags[t_idx, f_idx])[::-1][:budget]
    peaks = [((t_idx[i] * hop) / sample_rate, float(freqs[f_idx[i]]))
             for i in order]
    peaks.sort()
    return peaks


def _quantise(f_hz: float, bits: int) -> int:
    span = _FREQ_MAX_HZ - _FREQ_MIN_HZ
    q = int((f_hz - _FREQ_MIN_HZ) / span * ((1 << bits) - 1))
    return max(0, min((1 << bits) - 1, q))


def fingerprint_array(audio, sample_rate: int) -> list:
    """[(hash, offset_ms)] landmarks for one piece of audio."""
    peaks = _spectral_peaks(audio, sample_rate)
    out = []
    for i, (t1, f1) in enumerate(peaks):
        fanned = 0
        for t2, f2 in peaks[i + 1:]:
            dt = t2 - t1
            if dt < _PAIR_DT_S[0]:
                continue
            if dt > _PAIR_DT_S[1] or fanned >= _FAN_OUT:
                break
            if abs(f2 - f1) > _PAIR_DF_HZ:
                continue
            dt_q = int(dt / _PAIR_DT_S[1] * ((1 << _DT_BITS) - 1))
            landmark = ((_quantise(f1, _FREQ_BITS) << (_FREQ_BITS + _DT_BITS))
                        | (_quantise(f2, _FREQ_BITS) << _DT_BITS) | dt_q)
            out.append((landmark, int(t1 * 1000)))
            fanned += 1
    return out


def fingerprint_file(path) -> list:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return fingerprint_array(audio, sample_rate)


def match_votes(query_fps: list, candidate_fps: list) -> int:
    """Aligned-offset votes between two fingerprint lists (both
    [(hash, offset_ms)]). Higher = better; see module docstring for the
    confidence thresholds."""
    by_hash: dict = {}
    for landmark, offset in candidate_fps:
        by_hash.setdefault(landmark, []).append(offset)
    votes: dict = {}
    for landmark, q_offset in query_fps:
        for c_offset in by_hash.get(landmark, ()):
            # 100 ms offset buckets absorb codec/trim jitter
            bucket = (c_offset - q_offset) // 100
            votes[bucket] = votes.get(bucket, 0) + 1
    return max(votes.values()) if votes else 0


def is_confident(votes: int, query_landmarks: int) -> bool:
    return (votes >= _MATCH_MIN_VOTES
            and query_landmarks > 0
            and votes / query_landmarks >= _MATCH_MIN_FRACTION)
