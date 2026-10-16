"""Length in bars: the one arithmetic both rooms' forms and routes use, so a bar count means the same seconds
everywhere. Rounding is half-up, as the forms' JavaScript does.
"""

from __future__ import annotations

import math


def beats_per_bar(timesignature) -> int:
    """Beats per bar from the form's time signature ("4", "6", "6/8"); 4 when unusable."""
    try:
        beats = int(str(timesignature or "").strip().split("/")[0])
    except ValueError:
        return 4
    return beats if beats > 0 else 4


def bars_to_seconds(bars, bpm, beats_per_bar_count=4):
    """bars × beats × 60 / BPM, unrounded."""
    return bars * beats_per_bar_count * 60 / bpm


def fit_bars(bars, bpm, beats, lo, hi):
    """The whole seconds the engine is asked for, held to [lo, hi], and a line saying what happened. (seconds, line)."""
    exact = bars_to_seconds(bars, bpm, beats)
    whole = int(exact + 0.5)
    seconds = min(hi, max(lo, whole))
    shown = math.floor(exact * 10 + 0.5) / 10
    line = f"{bars} bar{'' if bars == 1 else 's'} at {bpm} BPM = {shown:.1f} s"
    if seconds > whole:
        line += f" — held at the {lo} s floor"
    elif seconds < whole:
        line += f" — held at the {hi} s ceiling"
    elif abs(exact - seconds) >= 0.05:
        line += f" — sent as {seconds} s"
    return seconds, line
