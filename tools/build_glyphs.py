#!/usr/bin/env python
"""Build-time only: Red Hat Display Bold outlines -> khaos_attribution/seal/glyphs.json.

Run by hand when the charset or the font changes; commit the result:

    <any python with fontTools> tools/build_glyphs.py path/to/RedHatDisplay-Bold.ttf

Every glyph is normalised to a 1000-unit em. Red Hat Display is SIL OFL 1.1
with Reserved Font Name "Red Hat": the OFL travels beside the table
(seal/OFL.txt) and the derived file carries no reserved name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Inputs are uppercase-only (the renderer refuses anything else); the one
# lowercase glyph is for the ring's fixed label "REGISTER No: ".
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.-/ o"
OUT = Path(__file__).resolve().parents[1] / "src" / "khaos_attribution" / "seal" / "glyphs.json"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(argv[1])
    upem = font["head"].unitsPerEm
    k = 1000.0 / upem
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    names = font["name"]
    missing = [c for c in CHARSET if ord(c) not in cmap]
    if missing:
        print(f"font lacks {missing!r}", file=sys.stderr)
        return 1
    glyphs: dict[str, dict] = {}
    for ch in CHARSET:
        gname = cmap[ord(ch)]
        glyph = glyph_set[gname]
        pen = SVGPathPen(glyph_set, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
        glyph.draw(TransformPen(pen, (k, 0, 0, k, 0, 0)))
        bounds = BoundsPen(glyph_set)
        glyph.draw(TransformPen(bounds, (k, 0, 0, k, 0, 0)))
        box = [round(v, 1) for v in bounds.bounds] if bounds.bounds else None   # None: a space
        glyphs[ch] = {"d": pen.getCommands(), "advance": round(glyph.width * k, 1), "bounds": box}
    table = {
        "font": f"{names.getDebugName(1)} {names.getDebugName(2)} {names.getDebugName(5)}",
        "licence": "SIL Open Font License 1.1 — see OFL.txt beside this file",
        "upem": 1000,
        "charset": CHARSET,
        "glyphs": glyphs,
    }
    OUT.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(glyphs)} glyphs from {table['font']} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
