"""Word timings for a sung piece — forced alignment of KNOWN lyrics.

Shared by the Workshop (auditions) and the Listening Space (outputs): both
show a piece's words as the selection surface for a repaint and both must
place them the same way. The contract here is the document shape:

    {"schema_version": "1.0.0", "aligner": {...}, "sample_rate", "duration",
     "lyrics": <text>, "words": [{"word", "char_start", "char_end",
                                  "start", "end", "score", "placed"}]}

``char_start``/``char_end`` index the ORIGINAL lyrics text; a client splices
an edit by character span (``splice_words``) so line breaks and structure
tags outside the span are untouched, and the engine is asked to sing the
full edited text.

Mechanics: torchaudio's CTC ``forced_align`` against a wav2vec2 emission
(English characters A–Z and apostrophe; other scripts fold to their base
letters or are interpolated). Audio is decoded with soundfile — torchaudio's
own loader needs FFmpeg via torchcodec, which the Macs lack. Emissions are
computed in 30 s chunks (attention is quadratic in time). One model per
process. Words the aligner cannot place are interpolated between their
neighbours with score 0 and ``placed: False`` — shown approximate, never
precise.

Heavy imports stay inside functions: importing this module costs nothing,
and ``available()`` is memoised.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

LYRICS_VERSION = "1.0.0"
ALIGNER_NAME = "torchaudio.WAV2VEC2_ASR_BASE_960H+forced_align"
CHUNK_SECONDS = 30.0

# Structure tags ([Verse], [Chorus], [Instrumental]…) are not sung.
_TAG_LINE = re.compile(r"^\s*\[[^\]]*\]\s*$")
_WORD = re.compile(r"\S+")

_AVAILABLE: bool | None = None
_MODEL = None
_MODEL_LOCK = None


def available() -> bool:
    """torchaudio's forced aligner plus soundfile for decoding. Memoised:
    the check imports torch."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import soundfile  # noqa: F401, PLC0415
            import torchaudio  # noqa: F401, PLC0415
            import torchaudio.functional as F  # noqa: PLC0415
            _AVAILABLE = hasattr(F, "forced_align")
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def tokenize_lyrics(text: str) -> list[dict]:
    """Words with character offsets, skipping structure-tag lines. Pure and
    deterministic: every client and the aligner agree on word indices
    because all derive them from this one function."""
    words: list[dict] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        if not _TAG_LINE.match(line):
            for m in _WORD.finditer(line):
                words.append({"word": m.group(0),
                              "char_start": pos + m.start(),
                              "char_end": pos + m.end()})
        pos += len(line)
    return words


def splice_words(text: str, words: list[dict], first: int, last: int,
                 replacement: str) -> str:
    """``text`` with words[first..last] (inclusive) replaced by
    ``replacement`` — by character span."""
    if not words or first < 0 or last >= len(words) or first > last:
        raise ValueError("word range out of bounds")
    a = words[first]["char_start"]
    b = words[last]["char_end"]
    return text[:a] + replacement + text[b:]


def is_sung(lyrics: str | None) -> bool:
    """A lyrics text that carries sung words (not empty, not only tags such
    as the Workshop's ``[Instrumental]`` placeholder)."""
    return bool(lyrics) and bool(tokenize_lyrics(lyrics))


def _alignable(word: str) -> str:
    """The characters the English CTC model can emit: A–Z and apostrophe.
    Diacritics are folded (é → E); anything else is dropped."""
    folded = unicodedata.normalize("NFKD", word)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z']", "", folded.upper())


def _model_and_labels():
    """One model per process: 360 MB read and constructed once."""
    global _MODEL, _MODEL_LOCK
    import threading  # noqa: PLC0415
    if _MODEL_LOCK is None:
        _MODEL_LOCK = threading.Lock()
    with _MODEL_LOCK:
        if _MODEL is None:
            import torchaudio  # noqa: PLC0415
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            _MODEL = (bundle.get_model().eval(), bundle.get_labels(), bundle.sample_rate)
        return _MODEL


def align_file(audio_path: Path | str, lyrics: str) -> dict:
    """Forced-align ``lyrics`` to ``audio_path``. Heavy (loads the model on
    first use; downloads it on first use ever) — call from a thread."""
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    import torchaudio.functional as F  # noqa: PLC0415

    words = tokenize_lyrics(lyrics)
    if not words:
        raise ValueError("No sung words in the lyrics (only structure tags)")

    model, labels, model_sr = _model_and_labels()
    dictionary = {c: i for i, c in enumerate(labels)}

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())        # (channels, frames)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != model_sr:
        waveform = F.resample(waveform, sr, model_sr)
        sr = model_sr
    total_seconds = waveform.shape[1] / sr

    # Targets: each word's alignable characters, words separated by the
    # model's boundary token "|". Unalignable words contribute no characters
    # and are interpolated afterwards. ``spans`` indexes into ``targets``
    # and merge_tokens returns one span per target token ("|" included),
    # so the two stay aligned.
    targets: list[int] = []
    spans: list[tuple[int, int] | None] = []
    for w in words:
        chars = _alignable(w["word"])
        idxs = [dictionary[c] for c in chars if c in dictionary]
        if not idxs:
            spans.append(None)
            continue
        if targets:
            targets.append(dictionary["|"])
        start = len(targets)
        targets.extend(idxs)
        spans.append((start, len(targets) - 1))
    if not targets:
        raise ValueError("None of the lyrics' characters are alignable")

    with torch.inference_mode():
        chunk = int(CHUNK_SECONDS * sr)
        pieces = []
        for i in range(0, waveform.shape[1], chunk):
            em, _ = model(waveform[:, i:i + chunk])
            pieces.append(em)
        emissions = torch.cat(pieces, dim=1)
        log_probs = torch.log_softmax(emissions, dim=-1)
    tgt = torch.tensor([targets], dtype=torch.int32)
    aligned, scores = F.forced_align(log_probs, tgt, blank=0)
    aligned, scores = aligned[0], scores[0].exp()
    spans_tok = F.merge_tokens(aligned, scores, blank=0)
    frame_seconds = total_seconds / log_probs.shape[1]

    out: list[dict] = []
    for w, sp in zip(words, spans):
        entry = {"word": w["word"], "char_start": w["char_start"],
                 "char_end": w["char_end"], "start": None, "end": None,
                 "score": 0.0, "placed": False}
        if sp is not None:
            first, last = sp
            segs = spans_tok[first:last + 1]
            if segs:
                entry["start"] = round(segs[0].start * frame_seconds, 3)
                entry["end"] = round(segs[-1].end * frame_seconds, 3)
                entry["score"] = round(float(sum(s.score for s in segs) / len(segs)), 3)
                entry["placed"] = True
        out.append(entry)

    placed_idx = [i for i, e in enumerate(out) if e["placed"]]
    for i, e in enumerate(out):
        if e["placed"]:
            continue
        prev_end = next((out[j]["end"] for j in reversed(placed_idx) if j < i), 0.0)
        next_start = next((out[j]["start"] for j in placed_idx if j > i), total_seconds)
        e["start"], e["end"] = round(prev_end, 3), round(next_start, 3)
        e["score"] = 0.0

    return {
        "schema_version": LYRICS_VERSION,
        "aligner": {"name": ALIGNER_NAME, "torchaudio": torchaudio.__version__},
        "sample_rate": sr,
        "duration": round(total_seconds, 3),
        "lyrics": lyrics,
        "words": out,
    }
