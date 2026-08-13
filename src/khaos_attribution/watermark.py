"""Stage 6 watermarking contract, shared by every Khaos issuer.

This module is the single source for how a watermark_id is built and
which issuer may allocate which IDs. The audio machinery (AudioSeal)
is imported lazily so the schema package stays dependency-light — the
codec and ranges work everywhere; embed/detect need the heavy extras
(torch + torchaudio + audioseal) that generation machines already have.

Design decisions (carried over from the Platform implementation where
they were established — see that repo's git history for the evidence):

1. AudioSeal 16-bit models (facebook/audioseal, MIT). The provenance
   schema's watermark_id range (0-65535) is exactly AudioSeal's 16-bit
   message payload — the ID embedded in the audio IS the watermark_id
   in the record, no separate mapping table to drift.

2. The 16 bits carry an extended-Hamming (16,11) SECDED codeword, not
   a raw ID. Measured behaviour: AudioSeal's detector occasionally
   flips a single payload bit even at detection probability 1.0.
   SECDED corrects any single-bit error and refuses (rather than
   mis-attributes) double errors. Cost: 2048 IDs instead of 65536.

3. The 2048-payload space is partitioned by issuer so two registries
   can never hand out colliding IDs:
       platform  payloads    0-1023   (Listening Space outputs)
       app       payloads 1024-2047   (Workshop test renders)
   A decoded codeword's issuer is recoverable via watermark_issuer().

4. Models are 16 kHz-native; outputs are 44.1/48 kHz. The watermark
   delta is computed at 16 kHz and resampled onto the untouched
   original — the music itself is never resampled.

5. NO_TORCH_COMPILE is set before AudioSeal runs: its vendored moshi
   wrapper calls torch.compile, whose C++ codegen fails on macOS
   clang. Eager mode is more than fast enough.
"""

import logging
import os
import secrets
import threading

os.environ.setdefault("NO_TORCH_COMPILE", "1")

log = logging.getLogger("khaos_attribution.watermark")

# ---- extended Hamming (16,11) SECDED codec ----
#
# Bit layout follows the classic scheme: positions 1..15 hold parity at
# powers of two (1, 2, 4, 8) and data elsewhere; position 0 is overall
# parity. Single-bit errors are corrected via the syndrome; a non-zero
# syndrome with even overall parity means two errors — uncorrectable.

_DATA_POSITIONS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]  # 11 data bits
_PARITY_POSITIONS = [1, 2, 4, 8]

PAYLOAD_SPACE = 1 << len(_DATA_POSITIONS)  # 2048 distinct IDs

# Issuer partition — the contract between the Platform and the App.
PAYLOAD_RANGES: dict[str, range] = {
    "platform": range(0, 1024),
    "app": range(1024, 2048),
}


def encode_payload(payload: int) -> int:
    """11-bit payload -> 16-bit SECDED codeword (the watermark_id)."""
    if not 0 <= payload < PAYLOAD_SPACE:
        raise ValueError(f"Payload out of range: {payload}")
    bits = [0] * 16
    for i, pos in enumerate(_DATA_POSITIONS):
        bits[pos] = (payload >> (len(_DATA_POSITIONS) - 1 - i)) & 1
    for p in _PARITY_POSITIONS:
        bits[p] = sum(bits[pos] for pos in range(1, 16) if pos & p and pos != p) % 2
    bits[0] = sum(bits[1:]) % 2
    return int("".join(map(str, bits)), 2)


def decode_codeword(codeword: int) -> tuple[int, int, bool] | None:
    """16-bit word (possibly corrupted) -> (payload, clean codeword,
    corrected?) or None if uncorrectable (2+ bit errors)."""
    bits = [(codeword >> (15 - i)) & 1 for i in range(16)]
    syndrome = 0
    for pos in range(1, 16):
        if bits[pos]:
            syndrome ^= pos
    parity_ok = sum(bits) % 2 == 0
    corrected = False
    if syndrome and parity_ok:
        return None  # double error: detectable, not correctable
    if syndrome:
        bits[syndrome] ^= 1
        corrected = True
    elif not parity_ok:
        bits[0] ^= 1
        corrected = True
    payload = 0
    for pos in _DATA_POSITIONS:
        payload = (payload << 1) | bits[pos]
    return payload, int("".join(map(str, bits)), 2), corrected


def watermark_issuer(codeword: int) -> str | None:
    """Which issuer a (clean) codeword belongs to, or None if invalid."""
    decoded = decode_codeword(codeword)
    if decoded is None or decoded[1] != codeword:
        return None
    payload = decoded[0]
    for issuer, payload_range in PAYLOAD_RANGES.items():
        if payload in payload_range:
            return issuer
    return None


def allocate_watermark_id(used: set[int], issuer: str = "platform") -> int | None:
    """A fresh codeword from the issuer's partition that no stored output
    already carries, or None when the partition is exhausted."""
    payload_range = PAYLOAD_RANGES[issuer]
    span = len(payload_range)
    for _ in range(64):
        candidate = encode_payload(payload_range.start + secrets.randbelow(span))
        if candidate not in used:
            return candidate
    # dense registry: walk the partition deterministically instead of gambling
    for payload in payload_range:
        candidate = encode_payload(payload)
        if candidate not in used:
            return candidate
    log.error("Watermark partition %r exhausted (%d IDs)", issuer, span)
    return None


# ---- AudioSeal wrapper (heavy deps, all lazy) ----

_SAMPLE_RATE = 16000  # AudioSeal-native

# Model cache is process-global: the models are stateless once loaded.
_MODELS_LOCK = threading.Lock()
_MODELS = None


def _load_models():
    global _MODELS
    with _MODELS_LOCK:
        if _MODELS is None:
            from audioseal import AudioSeal

            log.info("Loading AudioSeal 16-bit generator + detector")
            _MODELS = (
                AudioSeal.load_generator("audioseal_wm_16bits"),
                AudioSeal.load_detector("audioseal_detector_16bits"),
            )
        return _MODELS


class Watermarker:
    """Embed/detect on in-memory audio arrays.

    Arrays are numpy float32 shaped (frames, channels) — the layout
    soundfile produces and consumes. File-based callers load first and
    save after, keeping container/metadata handling on their side.
    """

    @staticmethod
    def available() -> bool:
        try:
            import audioseal  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def embed_array(audio, sample_rate: int, watermark_id: int):
        """Return a watermarked copy of (frames, channels) float32 audio."""
        import torch
        import torchaudio

        generator, _ = _load_models()
        wav = torch.from_numpy(audio).float().T  # (channels, frames)
        mono = wav.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, T)
        message = torch.tensor([[(watermark_id >> (15 - i)) & 1 for i in range(16)]])
        mono16 = torchaudio.functional.resample(mono, sample_rate, _SAMPLE_RATE)
        with torch.inference_mode():
            delta16 = generator.get_watermark(mono16, _SAMPLE_RATE, message=message)
        delta = torchaudio.functional.resample(delta16, _SAMPLE_RATE, sample_rate)[0, 0]
        marked = (wav + delta[: wav.shape[-1]]).clamp(-1.0, 1.0)
        return marked.T.contiguous().numpy()

    @staticmethod
    def detect_array(audio, sample_rate: int) -> tuple[float, int | None, bool]:
        """(probability, clean codeword or None, corrected?) for audio
        shaped (frames, channels) float32."""
        import torch
        import torchaudio

        _, detector = _load_models()
        wav = torch.from_numpy(audio).float().T
        mono = wav.mean(dim=0, keepdim=True).unsqueeze(0)
        mono16 = torchaudio.functional.resample(mono, sample_rate, _SAMPLE_RATE)
        with torch.inference_mode():
            probability, message = detector.detect_watermark(mono16, _SAMPLE_RATE)
        raw = 0
        for bit in message[0].tolist():
            raw = (raw << 1) | int(bit)
        decoded = decode_codeword(raw)
        if decoded is None:
            return float(probability), None, False
        _, codeword, corrected = decoded
        return float(probability), codeword, corrected
