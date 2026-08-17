"""The embedding contract: how ANY Khaos surface turns audio into the one
CLAP vector space where catalogue and outputs are comparable.

The Workshop embeds training segments; the Listening Space embeds generated
outputs. An estimate compares the two, so both sides MUST embed identically
— same model, same pinned checkpoint, same deterministic windowing, same
pooling. That is why this file lives in the contract package: the constants
and the pure math are single-sourced here, and each app supplies only its
model forward pass.

Why deterministic windowing exists at all (hard-won in the Workshop):
laion_clap crops anything longer than 10 s to a RANDOM 10 s window with the
unseeded global RNG, so the same audio embedded twice differed by more than
real musical difference. Fixed 10 s windows at a 5 s hop, each unit-
normalised BEFORE mean-pooling, make the same audio produce the same vector
forever.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "laion/larger_clap_music"
# HF URL for the music-specific CLAP checkpoint (HTSAT-base), pinned to a
# commit SHA so the downloaded binary cannot change silently.
CLAP_CKPT_REVISION = "b3708341862f581175dba5c356a4ebf74a9b6651"
CLAP_CKPT_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_URL = (
    f"https://huggingface.co/lukewys/laion_clap/resolve/{CLAP_CKPT_REVISION}/"
    f"{CLAP_CKPT_FILENAME}"
)

TARGET_SR = 48000     # CLAP requires 48 kHz
WINDOW_SEC = 10.0     # exactly what the encoder takes
HOP_SEC = 5.0
MAX_WINDOW_BATCH = 16  # ~31 MB of float32 per forward pass


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise a 1D float32 vector; near-zero vectors pass through."""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return vector
    return (vector / norm).astype(np.float32)


def window_starts(n_samples: int, win: int, hop: int) -> list[int]:
    """Deterministic window offsets covering the whole signal.

    The final window is flush with the END of the audio rather than dropped,
    so the last few seconds are never silently unseen.
    """
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)
    return starts


def embed_windows(audio: np.ndarray, embed_batch) -> np.ndarray:
    """Mean-pool the embeddings of deterministic windows over `audio`.

    Each window is unit-normalised BEFORE pooling so one loud window cannot
    dominate the mean; the caller normalises the pooled result. Short audio
    yields exactly one window through the same path — no special case.
    `embed_batch` is the app's model forward: (batch of windows) -> vectors.
    """
    win = int(WINDOW_SEC * TARGET_SR)
    hop = int(HOP_SEC * TARGET_SR)
    starts = window_starts(len(audio), win, hop)
    parts = []
    for k in range(0, len(starts), MAX_WINDOW_BATCH):
        chunk = np.stack([audio[i:i + win]
                          for i in starts[k:k + MAX_WINDOW_BATCH]])
        parts.append(np.asarray(embed_batch(chunk), dtype=np.float32))
    embs = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if embs.ndim == 1:                      # a stub returned a single vector
        return embs
    unit = np.stack([l2_normalise(e) for e in embs])
    return unit.mean(axis=0).astype(np.float32)
