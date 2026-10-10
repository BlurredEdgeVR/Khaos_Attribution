"""The survival script's rules: a hit is the right output, confident and dominant; the bar is read as written."""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fingerprint_survival.py"
spec = importlib.util.spec_from_file_location("fingerprint_survival", SCRIPT)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _fps(seed: int, n: int = 60) -> list:
    return [((seed * 1000 + i) % 2 ** 25, i * 50) for i in range(n)]


def test_a_hit_needs_the_right_winner_confidence_and_dominance():
    q = _fps(1)
    assert S.is_hit(q, "a", "a", votes=40, distinct=40, runner=10)
    assert not S.is_hit(q, "b", "a", votes=40, distinct=40, runner=10), "the wrong output is never a hit"
    assert not S.is_hit(q, "a", "a", votes=40, distinct=40, runner=30), "a close runner-up is not dominance"
    assert not S.is_hit(q, "a", "a", votes=5, distinct=5, runner=0), "below the module's confidence floor"
    assert S.is_hit(q, "a", None, votes=40, distinct=40, runner=10), "a decoy that matches confidently is a false hit"


def test_best_match_ranks_the_whole_index_and_names_the_runner_up():
    index = {"a": _fps(1), "b": _fps(2), "c": _fps(3)}
    winner, votes, distinct, runner = S.best_match(_fps(2), index)
    assert winner == "b" and votes == 60 and distinct == 60 and runner == 0


def test_the_bar_is_read_as_written(tmp_path):
    def rows(hit_rate_counted, five_s_hits, false_match):
        out = []
        for i in range(100):
            for t in S.TRANSFORMS:
                hit = (i < hit_rate_counted * 100) if t != "clip_5s_mid" else (i < five_s_hits)
                out.append({"kind": "output", "file": f"o{i}", "transform": t, "hit": hit, "votes": 30})
        out.append({"kind": "decoy", "file": "d", "transform": "mp3_128k", "hit": false_match, "votes": 30})
        return out
    assert S.report(rows(0.96, 0, False), "t", tmp_path) == 0, "96% with 5 s clips at zero still passes: 5 s is reported, not counted"
    assert S.report(rows(0.94, 100, False), "t", tmp_path) == 1, "94% fails the 95% bar"
    assert S.report(rows(1.0, 100, True), "t", tmp_path) == 1, "one false match fails the bar"
    broken = rows(1.0, 100, False) + [{"kind": "output", "file": "x", "transform": "mp3_128k", "error": "ffmpeg", "hit": False}]
    assert S.report(broken, "t2", tmp_path) == 2 and not (tmp_path / "fingerprint-bar.t2.json").exists(), "an unmeasured row is no verdict"
    record = json.loads((tmp_path / "fingerprint-bar.t.json").read_text(encoding="utf-8"))
    assert record["false_matches"] and record["passed"] is False and record["per_transform"]["clip_5s_mid"]["counted"] is False
