"""``python -m khaos_attribution.seal seal NUMBER MARK VERSION LETTER OUT.svg``
``python -m khaos_attribution.seal marks MARK VERSION LETTER OUT.svg [--reverse]``"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from khaos_attribution.seal import SealRefused, render_marks_row, render_seal


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m khaos_attribution.seal",
                                description="Write the Guild seal or the marks row as SVG.")
    sub = p.add_subparsers(dest="asset", required=True)
    s = sub.add_parser("seal", help="the ceremonial seal")
    s.add_argument("number"); s.add_argument("artist_mark"); s.add_argument("standard_version")
    s.add_argument("date_letter"); s.add_argument("out", type=Path)
    s.add_argument("--size", type=int, default=800)
    s.add_argument("--rough", action="store_true")
    m = sub.add_parser("marks", help="the marks row")
    m.add_argument("artist_mark"); m.add_argument("standard_version"); m.add_argument("date_letter")
    m.add_argument("out", type=Path)
    m.add_argument("--height", type=int, default=238)
    m.add_argument("--reverse", action="store_true", help="white on an ink rectangle")
    m.add_argument("--rough", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.asset == "seal":
            svg = render_seal(a.number, a.artist_mark, a.standard_version, a.date_letter,
                              size=a.size, rough=a.rough)
        else:
            svg = render_marks_row(a.artist_mark, a.standard_version, a.date_letter,
                                   height=a.height, rough=a.rough, reverse=a.reverse)
    except SealRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    a.out.write_text(svg, encoding="utf-8")
    print(f"{a.out} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
