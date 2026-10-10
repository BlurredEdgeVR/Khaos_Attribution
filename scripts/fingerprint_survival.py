"""Does the fingerprint still name an output after distribution, and refuse audio it never saw?
The bar is docs/fingerprint-bar.md, written before this ran; the record lands beside it.
    <workshop-venv-python> scripts/fingerprint_survival.py --outputs <dir>... --decoys <dir>... --machine laptop
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf  # noqa: E402

from khaos_attribution.fingerprint import _HOP_S, fingerprint_array, is_confident, match_stats  # noqa: E402

TRANSFORMS = {
    "pcm_baseline":  ["-c:a", "pcm_s16le"],
    "mp3_128k":      ["-c:a", "libmp3lame", "-b:a", "128k"],
    "mp3_320k":      ["-c:a", "libmp3lame", "-b:a", "320k"],
    "aac_128k":      ["-c:a", "aac", "-b:a", "128k"],
    "opus_96k":      ["-c:a", "libopus", "-b:a", "96k"],
    "vorbis_q4":     ["-c:a", "libvorbis", "-q:a", "4"],
    "resample_44k1": ["-ar", "44100", "-c:a", "pcm_s16le"],
    "loudnorm":      ["-af", "loudnorm=I=-14:TP=-1", "-c:a", "pcm_s16le"],
    "clip_20s_mid":  ("clip", 20.0, None),
    "clip_10s_mid":  ("clip", 10.0, None),
    "clip_5s_mid":   ("clip", 5.0, None),
    "mp3_128k_clip_10s": ("clip", 10.0, ["-c:a", "libmp3lame", "-b:a", "128k"]),
}
EXTENSIONS = {"mp3_128k": ".mp3", "mp3_320k": ".mp3", "aac_128k": ".m4a", "opus_96k": ".opus", "vorbis_q4": ".ogg"}
NOT_COUNTED = {"clip_5s_mid"}
QUERY_PHASES = 4   # a clip starts anywhere on the analysis grid; the query is fingerprinted at four sub-hop offsets
WORKERS = 4
BAR_OVERALL = 0.95
BAR_PER_TRANSFORM = 0.90
DOMINANCE = 2.0


def _ffmpeg(args: list) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:500])


def transformed(src: Path, name: str, tmp: Path) -> Path:
    """The file after one transform, back as WAV for fingerprinting."""
    spec = TRANSFORMS[name]
    if isinstance(spec, tuple):
        _, seconds, codec = spec
        info = sf.info(str(src))
        start = max(0.0, info.duration / 2 - seconds / 2)
        mid = tmp / f"{name}.wav"
        _ffmpeg(["-ss", f"{start:.2f}", "-t", f"{seconds}", "-i", str(src), "-c:a", "pcm_s16le", str(mid)])
        if not codec:
            return mid
        enc = tmp / f"{name}.mp3"
        _ffmpeg(["-i", str(mid), *codec, str(enc)])
        back = tmp / f"{name}_back.wav"
        _ffmpeg(["-i", str(enc), "-c:a", "pcm_s16le", str(back)])
        return back
    out = tmp / f"{name}{EXTENSIONS.get(name, '.wav')}"
    _ffmpeg(["-i", str(src), *spec, str(out)])
    if out.suffix == ".wav":
        return out
    back = tmp / f"{name}_back.wav"
    _ffmpeg(["-i", str(out), "-c:a", "pcm_s16le", str(back)])
    return back


def query_phases(audio, sr: int) -> list:
    """The query at QUERY_PHASES sub-hop offsets; the stored fingerprint and the index are untouched."""
    hop = max(1, int(sr * _HOP_S))
    return [fingerprint_array(audio[k * hop // QUERY_PHASES:], sr) for k in range(QUERY_PHASES)]


def best_match(phases: list, index: dict) -> tuple:
    """(winner, votes, distinct, runner_up_votes) over the whole index, each candidate scored by its best phase."""
    scored = sorted(((max((match_stats(q, fps) for q in phases), key=lambda t: t[0]), name) for name, fps in index.items()),
                    key=lambda t: t[0][0], reverse=True)
    (votes, distinct), winner = scored[0]
    runner = scored[1][0][0] if len(scored) > 1 else 0
    return winner, votes, distinct, runner


def is_hit(query: list, winner: str, expected: str | None, votes: int, distinct: int, runner: int) -> bool:
    confident = is_confident(votes, len(query), distinct)
    dominant = votes >= DOMINANCE * max(runner, 1)
    return confident and dominant and (expected is None or winner == expected)


def wavs(dirs: list) -> list:
    """Every WAV under the folders, one per distinct content: an index holds a fingerprint once however many copies exist."""
    import hashlib  # noqa: PLC0415
    seen, out = set(), []
    for d in dirs:
        for p in sorted(p for p in Path(d).rglob("*.wav") if not p.name.startswith("._")):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                out.append(p)
    return out


_INDEX: dict = {}


def _one_file(job: tuple) -> list:
    kind, src, tmp = job
    rows = []
    work = Path(tmp) / f"{kind}_{src.stem}"
    work.mkdir(exist_ok=True)
    for name in TRANSFORMS:
        try:
            phases = query_phases(*_read(transformed(src, name, work)))
        except Exception as exc:  # noqa: BLE001 — a transform that cannot be made is a row, not a crash
            rows.append({"kind": kind, "file": src.name, "transform": name, "error": str(exc)[:200], "hit": False})
            continue
        winner, votes, distinct, runner = best_match(phases, _INDEX)
        expected = str(src) if kind == "output" else None
        hit = is_hit(phases[0], winner, expected, votes, distinct, runner)
        rows.append({"kind": kind, "file": src.name, "transform": name, "votes": votes, "distinct": distinct,
                     "runner_up": runner, "query_landmarks": len(phases[0]), "winner_is_self": winner == expected, "hit": hit})
    return rows


def run(outputs: list, decoys: list, machine: str, record_dir: Path) -> int:
    from multiprocessing import get_context  # noqa: PLC0415
    print(f"indexing {len(outputs)} outputs")
    _INDEX.update({str(p): fingerprint_array(*_read(p)) for p in outputs})
    rows = []
    with tempfile.TemporaryDirectory() as td:
        jobs = [(kind, src, td) for kind, files in (("output", outputs), ("decoy", decoys)) for src in files]
        with get_context("fork").Pool(WORKERS) as pool:
            for i, got in enumerate(pool.imap(_one_file, jobs)):
                rows.extend(got)
                print(f"  {i + 1}/{len(jobs)} {jobs[i][0]} {jobs[i][1].name[:40]}", flush=True)
    return report(rows, machine, record_dir)


def _read(p: Path):
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return audio, sr


def report(rows: list, machine: str, record_dir: Path) -> int:
    errors = [r for r in rows if "error" in r]
    if errors:
        # A row that could not be measured is not a miss; a verdict over it would be a false one.
        print(f"measurement incomplete: {len(errors)} of {len(rows)} rows could not be made; first: {errors[0]['error'][:160]}")
        return 2
    out_rows = [r for r in rows if r["kind"] == "output"]
    counted = [r for r in out_rows if r["transform"] not in NOT_COUNTED]
    per = {}
    for r in out_rows:
        per.setdefault(r["transform"], [0, 0])
        per[r["transform"]][1] += 1
        per[r["transform"]][0] += int(r["hit"])
    overall = sum(int(r["hit"]) for r in counted), len(counted)
    false_hits = [r for r in rows if r["kind"] == "decoy" and r["hit"]]
    per_ok = all(h / n >= BAR_PER_TRANSFORM for t, (h, n) in per.items() if t not in NOT_COUNTED and n)
    passed = overall[1] > 0 and overall[0] / overall[1] >= BAR_OVERALL and per_ok and not false_hits
    record = {
        "bar": "docs/fingerprint-bar.md", "machine": machine, "host": platform.node(),
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": len({r["file"] for r in out_rows}), "decoys": len({r["file"] for r in rows if r["kind"] == "decoy"}),
        "identification": {"hits": overall[0], "pairs": overall[1], "rate": round(overall[0] / overall[1], 4) if overall[1] else None},
        "per_transform": {t: {"hits": h, "pairs": n, "rate": round(h / n, 4) if n else None, "counted": t not in NOT_COUNTED} for t, (h, n) in per.items()},
        "false_matches": [{"file": r["file"], "transform": r["transform"], "votes": r["votes"]} for r in false_hits],
        "passed": passed, "rows": rows,
    }
    path = record_dir / f"fingerprint-bar.{machine}.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"\nidentification {overall[0]}/{overall[1]} ({record['identification']['rate']})")
    for t, (h, n) in per.items():
        print(f"  {t:<20} {h:>4}/{n:<4} {'' if t not in NOT_COUNTED else '(reported, not counted)'}")
    print(f"false matches: {len(false_hits)}")
    print(f"{'PASSED' if passed else 'FAILED'} the bar; record {path}")
    return 0 if passed else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--outputs", nargs="+", required=True, help="folders of rendered outputs (searched for *.wav)")
    p.add_argument("--decoys", nargs="+", required=True, help="folders of audio no output is (the catalogue's originals)")
    p.add_argument("--machine", required=True)
    p.add_argument("--record-dir", type=Path, default=Path(__file__).resolve().parent.parent / "docs")
    a = p.parse_args(argv)
    outs, decs = wavs(a.outputs), wavs(a.decoys)
    if not outs or not decs:
        raise SystemExit("need at least one output and one decoy")
    return run(outs, decs, a.machine, a.record_dir)


if __name__ == "__main__":
    raise SystemExit(main())
