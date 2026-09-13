"""The Guild seal: a circular stamp carrying a registration number around the
Khaos star, rendered onto model cards, certificates and cover art.

Pure geometry, deterministic, dependency-free. Every character is a vector
path from a precomputed glyph table (``glyphs.json``, built once from Red Hat
Display Bold by ``tools/build_glyphs.py``); the runtime has no font, no
``<text>``, no ``<textPath>``, no ``@font-face`` and nothing to fetch. Text on
a curved path fails silently across SVG renderers and has cost this project
time before — SPEC.md records the constraint and the geometry.

Two entry points:

``render_seal(number)``
    draws the seal for a registration number and returns SVG source.
``seal_for_provenance(document)``
    draws the seal for a VERIFIED provenance document — and refuses, raising
    ``SealRefused``, unless ``embedded_agrees_with_sidecar`` is literally
    ``True``. The seal asserts that the check passed; it must be impossible
    to draw one otherwise. The number is read from the record, never made
    here, and the seed is a hash of the canonicalised record, so one record
    always produces one byte-identical seal.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib import resources

__all__ = ["CHARSET", "SealRefused", "render_seal", "seal_for_provenance"]

# ── geometry (SPEC.md is the reference; these are its numbers) ─────────────
CANVAS = 800
CENTRE = 400.0
FIELD_R = 396
OUTER_RULE_R = 352
INNER_RULE_R = 276
RULE_WIDTH = 7
BASELINE_R = 290.0
TYPE_SIZE = 46.0
TRACKING = 6.0
STAR_TRANSFORM = "translate(215,215) scale(0.5992)"
STAR_STROKE_WIDTH = 3
FIELD_FILL = "#0b0f0e"
INK = "#ffffff"
FOOT_POLYGON = "400,684 409,703 428,712 409,721 400,740 391,721 372,712 391,703"
FOOT_DOTS = ((268, 684), (532, 684))

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.-/ "


class SealRefused(ValueError):
    """The seal was not drawn. The message says why, in one sentence."""


# ── package data, read once ─────────────────────────────────────────────────

_GLYPHS: dict | None = None
_STAR: list[str] | None = None


def _glyphs() -> dict:
    global _GLYPHS
    if _GLYPHS is None:
        text = resources.files(__package__).joinpath("glyphs.json").read_text(encoding="utf-8")
        table = json.loads(text)
        if table.get("upem") != 1000:
            raise RuntimeError("glyphs.json is not normalised to a 1000-unit em")
        _GLYPHS = table["glyphs"]
    return _GLYPHS


def _star_elements() -> list[str]:
    """The star's geometry elements (``<polygon points>`` / ``<path d>``), as
    written in star.svg, without colour: the renderer paints them."""
    global _STAR
    if _STAR is None:
        text = resources.files(__package__).joinpath("star.svg").read_text(encoding="utf-8")
        elems: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("<polygon ", "<path ")):
                elems.append(s[:-2].rstrip() if s.endswith("/>") else s)
        if not elems:
            raise RuntimeError("star.svg holds no polygon or path elements")
        _STAR = elems
    return _STAR


# ── the layout ──────────────────────────────────────────────────────────────

def _num(v: float) -> str:
    """Fixed formatting: three decimals, no exponent, no negative zero — the
    same input must give the same bytes on every machine."""
    s = f"{v:.3f}"
    if s in ("-0.000",):
        s = "0.000"
    return s


def layout(number: str) -> list[dict]:
    """Where each glyph goes: the exact arc layout from SPEC.md.

    Advances are in px (glyph advance × type size / 1000). The run's total
    arc length, tracking included, becomes an angle at the baseline radius
    and is centred on the top; each glyph sits at the middle of its own
    advance and is rotated to the tangent, facing outward.
    """
    glyphs = _glyphs()
    scale = TYPE_SIZE / 1000.0
    advances = [glyphs[c]["advance"] * scale for c in number]
    total = sum(advances) + TRACKING * (len(number) - 1)
    span = total / BASELINE_R
    start = -span / 2.0
    out: list[dict] = []
    run_before = 0.0
    for i, (ch, adv) in enumerate(zip(number, advances)):
        theta = start + (run_before + adv / 2.0) / BASELINE_R
        px = CENTRE + BASELINE_R * math.sin(theta)
        py = CENTRE - BASELINE_R * math.cos(theta)
        out.append({"char": ch, "advance_px": adv, "theta": theta, "px": px, "py": py,
                    "d": glyphs[ch]["d"]})
        run_before += adv + TRACKING
    return out


def _validate_number(number: str) -> str:
    if not isinstance(number, str) or not number.strip():
        raise SealRefused("a seal needs a registration number")
    bad = sorted({c for c in number if c not in CHARSET})
    if bad:
        raise SealRefused(f"the number {number!r} uses characters the seal cannot set: "
                          f"{''.join(bad)!r} (allowed: A-Z 0-9 : . - / and space)")
    return number


def render_seal(number: str, *, seed: int | None = None, size: int = 800,
                rough: bool = False) -> str:
    """The seal for ``number`` as SVG source.

    ``size`` is the rendered width and height in px; the drawing is the
    800-unit canvas scaled by size / 800. ``rough`` enables the optional
    turbulence filter (off by default; several renderers ignore filters, so
    the clean drawing is the canonical one and the filter never carries
    meaning). ``seed`` feeds only that filter.
    """
    number = _validate_number(number)
    if not isinstance(size, int) or size <= 0:
        raise SealRefused("size must be a positive integer")
    scale = TYPE_SIZE / 1000.0
    glyph_paths = []
    for g in layout(number):
        if g["char"] == " ":
            continue                      # an advance, not a mark
        transform = (f"translate({_num(g['px'])},{_num(g['py'])}) "
                     f"rotate({_num(math.degrees(g['theta']))}) "
                     f"scale({_num(scale)},{_num(-scale)}) "
                     f"translate({_num(-g['advance_px'] / (2.0 * scale))},0)")
        glyph_paths.append(f'    <path transform="{transform}" d="{g["d"]}"/>')
    star = "\n".join(f"    {e} fill=\"{INK}\" stroke=\"{INK}\" stroke-width=\"{STAR_STROKE_WIDTH}\"/>"
                     for e in _star_elements())
    dots = "\n".join(f'    <circle cx="{cx}" cy="{cy}" r="8" fill="{INK}"/>' for cx, cy in FOOT_DOTS)
    filter_def = ""
    group_attr = ""
    if rough:
        filter_def = (
            '  <defs>\n'
            '    <filter id="rough" x="-6%" y="-6%" width="112%" height="112%">\n'
            f'      <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="3" '
            f'seed="{int(seed or 0)}" result="noise"/>\n'
            '      <feDisplacementMap in="SourceGraphic" in2="noise" scale="4" '
            'xChannelSelector="R" yChannelSelector="G"/>\n'
            '    </filter>\n'
            '  </defs>\n')
        group_attr = ' filter="url(#rough)"'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {CANVAS} {CANVAS}">\n'
        f'{filter_def}'
        f'  <g{group_attr}>\n'
        f'    <circle cx="{CENTRE:g}" cy="{CENTRE:g}" r="{FIELD_R}" fill="{FIELD_FILL}"/>\n'
        f'    <circle cx="{CENTRE:g}" cy="{CENTRE:g}" r="{OUTER_RULE_R}" fill="none" '
        f'stroke="{INK}" stroke-width="{RULE_WIDTH}"/>\n'
        f'    <circle cx="{CENTRE:g}" cy="{CENTRE:g}" r="{INNER_RULE_R}" fill="none" '
        f'stroke="{INK}" stroke-width="{RULE_WIDTH}"/>\n'
        f'    <g fill="{INK}">\n' + "\n".join(glyph_paths) + '\n    </g>\n'
        f'    <g transform="{STAR_TRANSFORM}">\n{star}\n    </g>\n'
        f'    <polygon points="{FOOT_POLYGON}" fill="{INK}"/>\n'
        f'{dots}\n'
        '  </g>\n'
        '</svg>\n'
    )


# ── provenance binding ──────────────────────────────────────────────────────

def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def seal_for_provenance(document: dict, *, size: int = 800, rough: bool = False,
                        number_key: str = "watermark_id") -> str:
    """The seal for one verified provenance document.

    ``document`` is what the Listening Space's provenance endpoint returns:
    the provenance ``record`` plus ``embedded_agrees_with_sidecar``, the
    verifier's word that the audio on disk still carries the record it
    claims. Anything but a literal ``True`` refuses — a missing flag, a
    falsy one, a string "true" — because the seal asserts the check passed.

    The number is ``record[number_key]`` (the run's watermark ID by
    default); nothing is generated or assigned here. The filter seed is a
    hash of the canonicalised record, so the same record is always the same
    bytes.
    """
    if not isinstance(document, dict):
        raise SealRefused("a provenance document is a JSON object")
    if document.get("embedded_agrees_with_sidecar") is not True:
        raise SealRefused("the seal asserts that the embedded watermark agrees with the "
                          "sidecar, and this document does not say so — not drawn")
    record = document.get("record") if isinstance(document.get("record"), dict) else document
    value = record.get(number_key)
    if value is None or value == "":
        raise SealRefused(f"the record carries no {number_key!r} to seal")
    number = str(value).upper()
    digest = hashlib.sha256(_canonical(record)).digest()
    seed = int.from_bytes(digest[:4], "big")
    return render_seal(number, seed=seed, size=size, rough=rough)
