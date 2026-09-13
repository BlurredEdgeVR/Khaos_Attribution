"""``python -m khaos_attribution.seal NUMBER OUT.svg [--size N] [--rough] [--seed S]``"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from khaos_attribution.seal import SealRefused, render_seal


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m khaos_attribution.seal",
                                description="Write the Guild seal for a registration number as SVG.")
    p.add_argument("number", help="the registration number (A-Z 0-9 : . - / and space)")
    p.add_argument("out", type=Path, help="where to write the SVG")
    p.add_argument("--size", type=int, default=800, help="rendered px (default 800)")
    p.add_argument("--rough", action="store_true", help="enable the optional turbulence filter")
    p.add_argument("--seed", type=int, default=None, help="seed for the filter only")
    a = p.parse_args(argv)
    try:
        svg = render_seal(a.number, seed=a.seed, size=a.size, rough=a.rough)
    except SealRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    a.out.write_text(svg, encoding="utf-8")
    print(f"{a.out} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
