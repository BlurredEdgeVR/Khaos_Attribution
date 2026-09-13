"""The Guild seal and the marks row — hallmarks for a registered adapter.

Two assets from one set of punch primitives:

``render_seal(number, artist_mark, standard_version, date_letter)``
    Ceremonial. A circular stamp: the registration number set on an arc
    around the field, the full interlaced Khaos star presiding, and three
    hand-struck punches beneath it — artist, standard, year. Certificates,
    the Register page header, anything physical.
``render_marks_row(artist_mark, standard_version, date_letter)``
    Small and functional. Four punches in a line — artist, standard, Guild,
    year — no ring, no field. Model cards, model pages, cover art. The
    Guild punch carries a REDUCED solid star: the interlaced artwork does
    not survive reduction and is never drawn small (SPEC.md).

Pure geometry, deterministic, dependency-free. Every character is a vector
path from ``glyphs.json`` (built once from Red Hat Display Bold by
``tools/build_glyphs.py``); the runtime has no font, no ``<text>``, no
``<textPath>``, no ``@font-face`` and nothing to fetch. Text on a curved
path fails silently across SVG renderers and has cost this project time.

``seal_for_provenance`` / ``marks_row_for_provenance`` draw for a VERIFIED
provenance document and refuse — raising ``SealRefused`` — unless
``embedded_agrees_with_sidecar`` is literally ``True``. The values are read
from the record, never made here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib import resources
from typing import Sequence

__all__ = [
    "CHARSET", "SealRefused", "render_seal", "render_marks_row",
    "seal_for_provenance", "marks_row_for_provenance", "provenance_values",
]

# ── the seal's canvas (SPEC.md is the reference; these are its numbers) ────
CANVAS = 800
CENTRE = 400.0
FIELD_R = 396
OUTER_RULE_R = 352
INNER_RULE_R = 276
RULE_WIDTH = 7
BASELINE_R = 290.0
TYPE_SIZE = 46.0
TRACKING = 6.0
FIELD_FILL = "#0b0f0e"
INK = "#0b0f0e"
PAPER = "#ffffff"
STAR_TRANSFORM = "translate(269.00,167.00) scale(0.42429)"    # 262 wide, centred (400, 298)
STAR_STROKE_WIDTH = 3
FOOT_POLYGON = "400,684 409,703 428,712 409,721 400,740 391,721 372,712 391,703"
FOOT_DOTS = ((268, 684), (532, 684))
SEAL_PUNCH_TRANSFORM = "translate(229.80,467.10) scale(0.74)"  # three cells, row 460 native, centred (400, 530)
SEAL_PUNCH_PITCH = 160        # cell 140 + gap 20
SEAL_PUNCH_STROKE = 9

# ── the marks row's canvas ──────────────────────────────────────────────────
MARKS_CELL_W, MARKS_CELL_H = 140, 170
MARKS_GAP, MARKS_PAD = 14, 34
MARKS_W = MARKS_PAD * 2 + 4 * MARKS_CELL_W + 3 * MARKS_GAP      # 670
MARKS_H = MARKS_PAD * 2 + MARKS_CELL_H                          # 238
MARKS_PUNCH_STROKE = 7

# ── punches (local cell coordinates, 140 × 170, centre (70, 85)) ───────────
PUNCH_CX, PUNCH_CY = 70.0, 85.0
PUNCH_SHAPES = {
    "canted": "M 28,10 L 112,10 L 130,28 L 130,142 L 112,160 L 28,160 L 10,142 L 10,28 Z",
    "shield": "M 10,10 L 130,10 L 130,100 C 130,138 104,150 70,161 C 36,150 10,138 10,100 Z",
    "cartouche": "M 10,64 C 10,28 36,10 70,10 C 104,10 130,28 130,64 L 130,144 L 112,160 L 28,160 L 10,144 Z",
}
PUNCH_CIRCLE_R = 61
PUNCH_TEXT_MAX_W = 100.0      # the widest a punch's content may run (cell inner width, less air)
# (shape, type size, baseline, rotation, dx, dy) — measured off the approved design.
SEAL_PUNCHES = (
    ("canted",    44.0, 101.0, -2.4, 0.0, 0.0),
    ("shield",    44.0,  97.0,  1.9, 1.0, 3.0),
    ("cartouche", 92.0, 118.0, -1.4, -1.0, 2.0),
)
MARKS_PUNCHES = (
    ("canted",    44.0, 101.0, -2.4, 0.0, 0.0),
    ("shield",    44.0,  97.0,  1.7, 1.0, 3.0),
    ("circle",    None,  None,  -1.2, -1.0, -2.0),
    ("cartouche", 92.0, 118.0,  2.9, 1.0, 4.0),
)
# jitter_from_record draws within these ranges — the constants above sit inside them.
JITTER_ROTATION = (-3.0, 3.0)
JITTER_DX = (-1.0, 1.0)
JITTER_DY = (-2.0, 4.0)

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.-/ "
DATE_LETTER_EPOCH = 2026      # the year whose letter is A; see provenance_values


class SealRefused(ValueError):
    """The mark was not drawn. The message says why, in one sentence."""


# ── package data, read once ─────────────────────────────────────────────────

_GLYPHS: dict | None = None
_STAR: list[str] | None = None


def _glyphs() -> dict:
    global _GLYPHS
    if _GLYPHS is None:
        table = json.loads(resources.files(__package__).joinpath("glyphs.json").read_text(encoding="utf-8"))
        if table.get("upem") != 1000:
            raise RuntimeError("glyphs.json is not normalised to a 1000-unit em")
        _GLYPHS = table["glyphs"]
    return _GLYPHS


def _star_elements() -> list[str]:
    """The interlaced star's geometry (``<polygon points>`` / ``<path d>``)
    from star.svg, without colour: the renderer paints it."""
    global _STAR
    if _STAR is None:
        text = resources.files(__package__).joinpath("star.svg").read_text(encoding="utf-8")
        elems = [s.strip()[:-2].rstrip() for s in text.splitlines()
                 if s.strip().startswith(("<polygon ", "<path ")) and s.strip().endswith("/>")]
        if not elems:
            raise RuntimeError("star.svg holds no polygon or path elements")
        _STAR = elems
    return _STAR


def _num(v: float) -> str:
    """Three fixed decimals, no exponent, no negative zero: the same input
    is the same bytes on every machine."""
    s = f"{v:.3f}"
    return "0.000" if s == "-0.000" else s


def _text(value, what: str, *, min_len: int = 1, max_len: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SealRefused(f"{what} is missing")
    bad = sorted({c for c in value if c not in CHARSET})
    if bad:
        raise SealRefused(f"{what} {value!r} uses characters the punches cannot set: "
                          f"{''.join(bad)!r} (allowed: A-Z 0-9 : . - / and space)")
    if len(value) < min_len or (max_len is not None and len(value) > max_len):
        raise SealRefused(f"{what} {value!r} must be "
                          + (f"{min_len}–{max_len}" if max_len else f"at least {min_len}") + " characters")
    return value


# ── the ring type ───────────────────────────────────────────────────────────

def layout(number: str) -> list[dict]:
    """Where each ring glyph goes: the exact arc layout from SPEC.md.

    Advances are px (glyph advance × type size / 1000). The run's total arc
    length, tracking included, becomes an angle at the baseline radius and
    is centred on the top; each glyph sits at the middle of its own advance
    and is rotated to the tangent, facing outward.
    """
    glyphs = _glyphs()
    scale = TYPE_SIZE / 1000.0
    advances = [glyphs[c]["advance"] * scale for c in number]
    total = sum(advances) + TRACKING * (len(number) - 1)
    span = total / BASELINE_R
    start = -span / 2.0
    out: list[dict] = []
    run_before = 0.0
    for ch, adv in zip(number, advances):
        theta = start + (run_before + adv / 2.0) / BASELINE_R
        out.append({"char": ch, "advance_px": adv, "theta": theta,
                    "px": CENTRE + BASELINE_R * math.sin(theta),
                    "py": CENTRE - BASELINE_R * math.cos(theta), "d": glyphs[ch]["d"]})
        run_before += adv + TRACKING
    return out


def _ring_type(number: str) -> str:
    scale = TYPE_SIZE / 1000.0
    paths = []
    for g in layout(number):
        if g["char"] == " ":
            continue                      # an advance, not a mark
        t = (f"translate({_num(g['px'])},{_num(g['py'])}) rotate({_num(math.degrees(g['theta']))}) "
             f"scale({_num(scale)},{_num(-scale)}) translate({_num(-g['advance_px'] / (2.0 * scale))},0)")
        paths.append(f'      <path transform="{t}" d="{g["d"]}"/>')
    return "\n".join(paths)


# ── punch primitives (shared by both assets) ────────────────────────────────

def punch_text_layout(text: str, size: float, baseline: float) -> tuple[float, list[dict]]:
    """Straight type centred on the punch's centre line.

    Returns (type size actually used, glyphs). The size is the MAXIMUM: a
    run wider than ``PUNCH_TEXT_MAX_W`` is set smaller so it fits the punch
    — a hallmark is cut to its die, and a version string is not allowed to
    escape the shield.
    """
    glyphs = _glyphs()
    units = sum(glyphs[c]["advance"] for c in text)
    used = size
    if units * size / 1000.0 > PUNCH_TEXT_MAX_W:
        used = PUNCH_TEXT_MAX_W * 1000.0 / units
    s = used / 1000.0
    x = PUNCH_CX - units * s / 2.0
    out = []
    for c in text:
        out.append({"char": c, "x": x, "baseline": baseline, "scale": s,
                    "advance_px": glyphs[c]["advance"] * s, "d": glyphs[c]["d"],
                    "bounds": glyphs[c]["bounds"]})
        x += glyphs[c]["advance"] * s
    return used, out


def _punch_text(text: str, size: float, baseline: float) -> str:
    _, glyphs = punch_text_layout(text, size, baseline)
    return "\n".join(
        f'        <path transform="translate({_num(g["x"])},{_num(g["baseline"])}) '
        f'scale({_num(g["scale"])},{_num(-g["scale"])})" d="{g["d"]}"/>'
        for g in glyphs if g["char"] != " ")


def reduced_star_points() -> str:
    """The Guild punch's star: a solid sixteen-point polygon, generated —
    never the interlaced artwork scaled down (SPEC.md)."""
    pts = []
    for i in range(16):
        a = math.radians(i * 22.5)
        r = 50 if i % 2 == 0 else 13
        pts.append(f"{_num(PUNCH_CX + r * math.sin(a))},{_num(PUNCH_CY - r * math.cos(a))}")
    return " ".join(pts)


def _punch(shape: str, content: str, *, x: float, rotation: float, dx: float, dy: float,
           ink: str, stroke_width: int) -> str:
    outline = (f'      <circle cx="{PUNCH_CX:g}" cy="{PUNCH_CY:g}" r="{PUNCH_CIRCLE_R}" '
               f'fill="none" stroke="{ink}" stroke-width="{stroke_width}"/>'
               if shape == "circle" else
               f'      <path d="{PUNCH_SHAPES[shape]}" fill="none" stroke="{ink}" '
               f'stroke-width="{stroke_width}" stroke-linejoin="miter"/>')
    return (f'    <g transform="translate({_num(x + dx)},{_num(dy)}) '
            f'rotate({_num(rotation)},{PUNCH_CX:g},{PUNCH_CY:g})">\n'
            f'{outline}\n      <g fill="{ink}">\n{content}\n      </g>\n    </g>')


def _punch_contents(artist_mark: str, standard_version: str, date_letter: str,
                    guild: bool) -> list[str]:
    star = f'        <polygon points="{reduced_star_points()}"/>'
    out = [_punch_text(artist_mark, 44.0, 101.0), _punch_text(standard_version, 44.0, 97.0)]
    if guild:
        out.append(star)
    out.append(_punch_text(date_letter, 92.0, 118.0))
    return out


def _resolve_jitter(spec, jitter: Sequence[tuple[float, float, float]] | None):
    if jitter is None:
        return [(rot, dx, dy) for (_s, _sz, _b, rot, dx, dy) in spec]
    if len(jitter) != len(spec):
        raise SealRefused(f"jitter needs {len(spec)} (rotation, dx, dy) triples")
    return [tuple(float(v) for v in j) for j in jitter]


def _filter(rough: bool) -> tuple[str, str]:
    if not rough:
        return "", ""
    return ('  <defs>\n    <filter id="struck" x="-6%" y="-6%" width="112%" height="112%">\n'
            '      <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="3" seed="11" result="n"/>\n'
            '      <feDisplacementMap in="SourceGraphic" in2="n" scale="3.2" xChannelSelector="R" '
            'yChannelSelector="G"/>\n    </filter>\n  </defs>\n', ' filter="url(#struck)"')


# ── the two assets ──────────────────────────────────────────────────────────

def _validate_values(artist_mark, standard_version, date_letter) -> tuple[str, str, str]:
    return (_text(artist_mark, "artist_mark", min_len=2, max_len=3),
            _text(standard_version, "standard_version"),
            _text(date_letter, "date_letter", min_len=1, max_len=1))


def render_seal(number: str, artist_mark: str, standard_version: str, date_letter: str, *,
                size: int = 800, rough: bool = False,
                jitter: Sequence[tuple[float, float, float]] | None = None) -> str:
    """The ceremonial seal as SVG source.

    ``size`` is the rendered width and height in px (the 800-unit canvas
    scaled by size / 800). ``rough`` enables the optional turbulence filter
    — off by default; several renderers ignore filters, so the clean drawing
    is canonical and the filter never carries meaning. ``jitter`` overrides
    the three punches' (rotation, dx, dy); the provenance wrapper derives it
    from the record when asked to.
    """
    number = _text(number, "registration number")
    artist_mark, standard_version, date_letter = _validate_values(artist_mark, standard_version, date_letter)
    if not isinstance(size, int) or size <= 0:
        raise SealRefused("size must be a positive integer")
    jit = _resolve_jitter(SEAL_PUNCHES, jitter)
    contents = _punch_contents(artist_mark, standard_version, date_letter, guild=False)
    punches = "\n".join(
        _punch(shape, content, x=i * SEAL_PUNCH_PITCH, rotation=rot, dx=dx, dy=dy,
               ink=PAPER, stroke_width=SEAL_PUNCH_STROKE)
        for i, ((shape, _sz, _b, *_), content, (rot, dx, dy)) in enumerate(zip(SEAL_PUNCHES, contents, jit)))
    star = "\n".join(
        f'      {e} fill="{PAPER}" stroke="{PAPER}" stroke-width="{STAR_STROKE_WIDTH}" stroke-miterlimit="10"/>'
        for e in _star_elements())
    dots = "\n".join(f'    <circle cx="{cx}" cy="{cy}" r="8" fill="{PAPER}"/>' for cx, cy in FOOT_DOTS)
    defs, attr = _filter(rough)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {CANVAS} {CANVAS}">\n'
        f'{defs}  <g{attr}>\n'
        f'    <circle cx="{CENTRE:g}" cy="{CENTRE:g}" r="{FIELD_R}" fill="{FIELD_FILL}"/>\n'
        f'    <circle cx="{CENTRE:g}" cy="{CENTRE:g}" r="{OUTER_RULE_R}" fill="none" stroke="{PAPER}" stroke-width="{RULE_WIDTH}"/>\n'
        f'    <circle cx="{CENTRE:g}" cy="{CENTRE:g}" r="{INNER_RULE_R}" fill="none" stroke="{PAPER}" stroke-width="{RULE_WIDTH}"/>\n'
        f'    <g fill="{PAPER}">\n{_ring_type(number)}\n    </g>\n'
        f'    <g transform="{STAR_TRANSFORM}">\n{star}\n    </g>\n'
        f'    <g transform="{SEAL_PUNCH_TRANSFORM}">\n{punches}\n    </g>\n'
        f'    <polygon points="{FOOT_POLYGON}" fill="{PAPER}"/>\n{dots}\n'
        '  </g>\n</svg>\n')


def render_marks_row(artist_mark: str, standard_version: str, date_letter: str, *,
                     height: int = 238, rough: bool = False, reverse: bool = False,
                     jitter: Sequence[tuple[float, float, float]] | None = None) -> str:
    """The marks row as SVG source: four punches, no ring, no field.

    Ink on transparent by default; ``reverse`` draws it white on an ink
    rectangle. ``height`` is the rendered height in px (width follows the
    670 × 238 canvas).
    """
    artist_mark, standard_version, date_letter = _validate_values(artist_mark, standard_version, date_letter)
    if not isinstance(height, int) or height <= 0:
        raise SealRefused("height must be a positive integer")
    ink = PAPER if reverse else INK
    jit = _resolve_jitter(MARKS_PUNCHES, jitter)
    contents = _punch_contents(artist_mark, standard_version, date_letter, guild=True)
    punches = "\n".join(
        _punch(shape, content, x=MARKS_PAD + i * (MARKS_CELL_W + MARKS_GAP), rotation=rot, dx=dx, dy=MARKS_PAD + dy,
               ink=ink, stroke_width=MARKS_PUNCH_STROKE)
        for i, ((shape, _sz, _b, *_), content, (rot, dx, dy)) in enumerate(zip(MARKS_PUNCHES, contents, jit)))
    width = round(height * MARKS_W / MARKS_H)
    ground = f'    <rect width="{MARKS_W}" height="{MARKS_H}" fill="{INK}"/>\n' if reverse else ""
    defs, attr = _filter(rough)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {MARKS_W} {MARKS_H}">\n'
        f'{defs}  <g{attr}>\n{ground}{punches}\n  </g>\n</svg>\n')


# ── provenance binding ──────────────────────────────────────────────────────

def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _record_digest(record: dict) -> bytes:
    return hashlib.sha256(_canonical(record)).digest()


def _jitter_from_digest(digest: bytes, n: int) -> list[tuple[float, float, float]]:
    """n (rotation, dx, dy) triples within the design's ranges, from the
    record's hash — reproducible for a record, different across records."""
    out = []
    for i in range(n):
        chunk = digest[3 * i: 3 * i + 3]
        if len(chunk) < 3:
            chunk = hashlib.sha256(digest + bytes([i])).digest()[:3]
        pick = lambda b, lo, hi: round(lo + (b / 255.0) * (hi - lo), 2)   # noqa: E731
        out.append((pick(chunk[0], *JITTER_ROTATION), pick(chunk[1], *JITTER_DX), pick(chunk[2], *JITTER_DY)))
    return out


_INITIALS_SPLIT = re.compile(r"[\s\-_/.]+")


def provenance_values(document: dict, *, number_key: str = "watermark_id") -> dict:
    """The four values a mark is struck with, read from a verified document.

    Refuses unless ``embedded_agrees_with_sidecar`` is literally ``True`` —
    the mark asserts the check passed. Then, from the record:

    - ``number``: ``record[number_key]`` (the run's watermark ID), required;
    - ``artist_mark``: ``record["artist_mark"]`` when the record carries one,
      else the initials of ``artist_name`` (two or three A–Z/0–9 characters);
    - ``standard_version``: ``record["standard_version"]`` when present, else
      the record's ``schema_version`` as major.minor — the attribution
      standard this record conforms to, as displayed;
    - ``date_letter``: ``record["date_letter"]`` when present, else the year
      letter of ``timestamp`` (``DATE_LETTER_EPOCH`` → A, then B, C…).

    Nothing is generated or assigned here: every value is a reading of the
    record, and the reading is the same for the same record.
    """
    if not isinstance(document, dict):
        raise SealRefused("a provenance document is a JSON object")
    if document.get("embedded_agrees_with_sidecar") is not True:
        raise SealRefused("the mark asserts that the embedded watermark agrees with the sidecar, "
                          "and this document does not say so — not drawn")
    record = document.get("record") if isinstance(document.get("record"), dict) else document
    number = record.get(number_key)
    if number is None or number == "":
        raise SealRefused(f"the record carries no {number_key!r} to seal")
    mark = record.get("artist_mark")
    if not mark:
        name = str(record.get("artist_name") or "")
        words = [w for w in _INITIALS_SPLIT.split(name.upper()) if w]
        letters = "".join(w[0] for w in words if w[0] in CHARSET and w[0] != " ")[:3]
        if len(letters) < 2:
            letters = "".join(c for c in name.upper() if c.isalnum() and c in CHARSET)[:3]
        mark = letters
    version = record.get("standard_version")
    if not version:
        parts = str(record.get("schema_version") or "").split(".")
        version = ".".join(parts[:2]) if len(parts) >= 2 else ""
    letter = record.get("date_letter")
    if not letter:
        m = re.match(r"(\d{4})", str(record.get("timestamp") or ""))
        if m:
            letter = chr(ord("A") + (int(m.group(1)) - DATE_LETTER_EPOCH) % 26)
    values = {"number": str(number).upper(), "artist_mark": str(mark).upper(),
              "standard_version": str(version), "date_letter": str(letter or "").upper()}
    for key, what in (("artist_mark", "an artist mark"), ("standard_version", "a standard version"),
                      ("date_letter", "a date letter")):
        if not values[key]:
            raise SealRefused(f"the record yields no {what} to strike")
    values["_digest"] = _record_digest(record)
    return values


def seal_for_provenance(document: dict, *, size: int = 800, rough: bool = False,
                        jitter_from_record: bool = False, number_key: str = "watermark_id") -> str:
    """The seal for one verified provenance document (see provenance_values)."""
    v = provenance_values(document, number_key=number_key)
    jitter = _jitter_from_digest(v["_digest"], len(SEAL_PUNCHES)) if jitter_from_record else None
    return render_seal(v["number"], v["artist_mark"], v["standard_version"], v["date_letter"],
                       size=size, rough=rough, jitter=jitter)


def marks_row_for_provenance(document: dict, *, height: int = 238, rough: bool = False,
                             reverse: bool = False, jitter_from_record: bool = False,
                             number_key: str = "watermark_id") -> str:
    """The marks row for one verified provenance document (see provenance_values)."""
    v = provenance_values(document, number_key=number_key)
    jitter = _jitter_from_digest(v["_digest"], len(MARKS_PUNCHES)) if jitter_from_record else None
    return render_marks_row(v["artist_mark"], v["standard_version"], v["date_letter"],
                            height=height, rough=rough, reverse=reverse, jitter=jitter)
