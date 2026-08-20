"""Phase 0 of watermarking v2: does the AudioSeal ID survive distribution?

Embeds a known calibration codeword (reserved band) into real generated
outputs, pushes each through the transforms of normal distribution, and
measures (a) detection probability and (b) whether the DECODED ID comes
back right at the current (16,11) SECDED — the evidence behind the
2048-IDs/1-bit-correction decision (docs/watermarking-v2.md §2/§9).

The Wwise-conversion leg proper waits for the Wwise MCP; Vorbis q4 is
included because Wwise's default music codec is Vorbis.

Usage:
    <workshop-venv-python> scripts/watermark_survival.py <wav> [<wav> ...]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from khaos_attribution.watermark import Watermarker, encode_payload  # noqa: E402

CALIBRATION_PAYLOAD = 7          # reserved band 0-31 (spec §3)
TRANSFORMS = {
    # name: ffmpeg args producing the transformed file (input/output added)
    "pcm_baseline":  ["-c:a", "pcm_s16le"],
    "mp3_128k":      ["-c:a", "libmp3lame", "-b:a", "128k"],
    "mp3_320k":      ["-c:a", "libmp3lame", "-b:a", "320k"],
    "aac_128k":      ["-c:a", "aac", "-b:a", "128k"],
    "opus_96k":      ["-c:a", "libopus", "-b:a", "96k"],
    "vorbis_q4":     ["-c:a", "libvorbis", "-q:a", "4"],   # Wwise's music codec
    "resample_44k1": ["-ar", "44100", "-c:a", "pcm_s16le"],
    "loudnorm":      ["-af", "loudnorm=I=-14:TP=-1", "-c:a", "pcm_s16le"],
    "trim_10s_mid":  None,   # handled specially: 10 s excerpt from the middle
}
EXTENSIONS = {"mp3_128k": ".mp3", "mp3_320k": ".mp3", "aac_128k": ".m4a",
              "opus_96k": ".opus", "vorbis_q4": ".ogg"}


def _ffmpeg(args: list) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel",
                             "error", *args], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode()[:500])


def _to_wav(src: Path, dst: Path) -> None:
    _ffmpeg(["-i", str(src), "-c:a", "pcm_s16le", str(dst)])


def run(paths: list) -> int:
    wm = Watermarker()
    codeword = encode_payload(CALIBRATION_PAYLOAD)
    print(f"calibration codeword: {codeword} (payload {CALIBRATION_PAYLOAD})")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, src in enumerate(map(Path, paths)):
            audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
            marked = wm.embed_array(audio, sr, codeword)
            master = tmp / f"master_{i}.wav"
            sf.write(str(master), marked, sr)

            for name, args in TRANSFORMS.items():
                out = tmp / f"{name}_{i}{EXTENSIONS.get(name, '.wav')}"
                if name == "trim_10s_mid":
                    dur = len(marked) / sr
                    start = max(0.0, dur / 2 - 5)
                    _ffmpeg(["-ss", f"{start:.2f}", "-t", "10",
                             "-i", str(master), "-c:a", "pcm_s16le", str(out)])
                else:
                    _ffmpeg(["-i", str(master), *args, str(out)])
                back = out
                if out.suffix != ".wav":
                    back = tmp / f"{name}_{i}_back.wav"
                    _to_wav(out, back)
                t_audio, t_sr = sf.read(str(back), dtype="float32", always_2d=True)
                prob, got, corrected = wm.detect_array(t_audio, t_sr)
                rows.append({"file": src.name, "transform": name,
                             "prob": round(float(prob), 3),
                             "id_ok": got == codeword,
                             "decoded": got,
                             "secded_corrected": bool(corrected)})
                print(f"  {src.name[:24]:<24} {name:<14} prob={prob:.3f} "
                      f"id_ok={got == codeword} corrected={corrected}")

    ok = sum(1 for r in rows if r["prob"] >= 0.5 and r["id_ok"])
    print(f"\nSURVIVAL: {ok}/{len(rows)} transform runs decoded the right ID")
    print(json.dumps(rows, indent=1))
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(run(sys.argv[1:]))
