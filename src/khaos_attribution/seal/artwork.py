"""The seal on a model's hero artwork — composited at display or export
time, never burned in.

``place_seal_on_artwork`` renders the seal with ``render_seal``, chooses a
colourway against the corner it will sit in, and returns a NEW image: an
SVG document that embeds the artwork untouched and lays the seal over a
corner (top right by default), rotated −4° about its own centre. The
clean master is never modified — the artist keeps their own art without the mark, and an entry
struck from the Register loses its seal without anyone re-rendering
anything. Cache the composite if you cache at all; never the source.

Pillow (the ``artwork`` extra) is needed to read the artwork's size and to
sample the corner. The core renderers need nothing.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

from khaos_attribution.seal import COLOURWAYS, SealRefused, render_marks_row, render_seal

__all__ = ["SealedArtwork", "place_seal_on_artwork", "choose_colourway", "CORNERS",
           "SEAL_DIAMETER", "INSET", "ROTATION", "MIN_LONG_EDGE", "MARKS_HEIGHT", "MARKS_MIN_HEIGHT"]

# Measured against the artwork's WIDTH, whatever its shape.
SEAL_DIAMETER = 0.15       # of W
INSET = 0.05               # of W, to the seal's bounding box, from the two edges of its corner
ROTATION = -4.0            # degrees, about the seal's own centre — deliberate; not a UI badge
MIN_LONG_EDGE = 400        # px: below this the ring number stops being legible; the marks row goes on instead
MARKS_HEIGHT = 0.12        # of W, the marks row's height when it stands in
MARKS_MIN_HEIGHT = 40      # px: below this the punches merge; place nothing rather than something illegible
_LUMA_MIDPOINT = 127.5


@dataclass(frozen=True)
class SealedArtwork:
    """A new image (SVG source) and what was placed on it, so a caller can
    say — and a test can check — exactly what happened."""
    svg: str
    width: int
    height: int
    placed: str | None            # "seal" | "marks" | None
    colourway: str | None         # "ink" | "paper" | None
    chosen_by: str | None         # "override" | "sampled" | None
    region_luminance: float | None
    box: tuple[float, float, float, float] | None   # x, y, w, h of what was placed, before rotation
    rotation: float


def _pil():
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SealRefused("placing the seal on artwork needs Pillow: install the 'artwork' extra") from exc
    return Image


def _load(artwork) -> tuple[bytes, str, "object"]:
    """(the artwork's own bytes, its MIME type, a Pillow image) — the bytes
    are what the composite embeds, exactly as read; nothing is re-encoded
    unless a Pillow image object (no bytes of its own) was passed."""
    Image = _pil()
    if isinstance(artwork, (str, Path)):
        data = Path(artwork).read_bytes()
    elif isinstance(artwork, (bytes, bytearray)):
        data = bytes(artwork)
    elif hasattr(artwork, "save") and hasattr(artwork, "size"):
        buf = io.BytesIO(); artwork.save(buf, format="PNG"); data = buf.getvalue()
    else:
        raise SealRefused("artwork must be a path, bytes, or a Pillow image")
    image = Image.open(io.BytesIO(data)); image.load()
    fmt = (image.format or "PNG").upper()
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}.get(fmt)
    if mime is None:
        raise SealRefused(f"artwork format {fmt} is not one the composite can embed (PNG, JPEG, WEBP, GIF)")
    return data, mime, image


def region_luminance(image, box: tuple[float, float, float, float]) -> float:
    """Mean luminance (0–255) of the region the seal will occupy — never the
    whole image: a dark sleeve with a bright corner is common."""
    x, y, w, h = box
    crop = image.convert("L").crop((int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))))
    px = list(crop.getdata())
    return sum(px) / len(px) if px else 0.0


def choose_colourway(luminance: float) -> str:
    """The variant with the greater contrast against the region: a light
    corner takes the ink disc, a dark one the paper disc."""
    return "ink" if luminance >= _LUMA_MIDPOINT else "paper"


def _num(v: float) -> str:
    s = f"{v:.3f}"
    return "0.000" if s == "-0.000" else s


def _nest(svg: str, x: float, y: float, w: float, h: float) -> str:
    """The rendered asset as a nested <svg> at a place and size — its own
    viewBox does the scaling, so its geometry is untouched."""
    head, sep, rest = svg.partition(">")
    assert head.startswith("<svg ")
    view = re.search(r'viewBox="[^"]+"', head).group(0)
    return (f'<svg x="{_num(x)}" y="{_num(y)}" width="{_num(w)}" height="{_num(h)}" {view}>'
            + sep + rest.rstrip("\n"))


CORNERS = ("top-right", "bottom-right")


def place_seal_on_artwork(artwork, number: str, artist_mark: str, standard_version: str,
                          date_letter: str, *, colourway: str | None = None,
                          corner: str = "top-right") -> SealedArtwork:
    """Composite the seal onto hero artwork — a new image; the
    source is never modified.

    ``artwork`` is a path, bytes or a Pillow image. ``colourway`` forces
    ``"ink"`` or ``"paper"`` (an artist's call); otherwise it is chosen by
    the mean luminance of the corner the seal will occupy, and the choice
    is recorded either way. ``corner`` is ``"top-right"`` (the design's
    default) or ``"bottom-right"``; the insets are the same, measured to
    the seal's bounding box from the two edges it sits against.

    Below ``MIN_LONG_EDGE`` px on the output's long edge the seal's ring
    number is not legible, so the marks row is placed instead, bottom
    right at the same inset; below ``MARKS_MIN_HEIGHT`` for that row,
    nothing is placed — the artwork comes back unmarked, and ``placed``
    says so.
    """
    if colourway is not None and colourway not in COLOURWAYS:
        raise SealRefused(f"colourway must be one of {sorted(COLOURWAYS)}, not {colourway!r}")
    if corner not in CORNERS:
        raise SealRefused(f"corner must be one of {CORNERS}, not {corner!r}")
    data, mime, image = _load(artwork)
    W, H = image.size
    d = SEAL_DIAMETER * W
    inset = INSET * W
    href = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    ground = f'  <image href="{href}" x="0" y="0" width="{W}" height="{H}"/>\n'

    placed = None; box = None; chosen = None; used = None; luma = None; overlay = ""
    if max(W, H) >= MIN_LONG_EDGE and inset + d <= H:
        box = (W - inset - d, inset if corner == "top-right" else H - inset - d, d, d)
        if colourway is None:
            luma = region_luminance(image, box)
            used, chosen = choose_colourway(luma), "sampled"
        else:
            used, chosen = colourway, "override"
        seal = render_seal(number, artist_mark, standard_version, date_letter, colourway=used)
        cx, cy = box[0] + d / 2, box[1] + d / 2
        overlay = (f'  <g transform="rotate({_num(ROTATION)},{_num(cx)},{_num(cy)})">\n'
                   f'    {_nest(seal, *box)}\n  </g>\n')
        placed = "seal"
    else:
        mh = MARKS_HEIGHT * W
        mw = mh * 670 / 238
        if mh >= MARKS_MIN_HEIGHT and inset + mh <= H and inset + mw <= W:
            box = (W - inset - mw, H - inset - mh, mw, mh)
            if colourway is None:
                luma = region_luminance(image, box)
                used, chosen = choose_colourway(luma), "sampled"
            else:
                used, chosen = colourway, "override"
            # The row has no paper-ground variant: ink marks sit on the art
            # itself (legible only on a light corner), and the reversed row
            # carries its own ink ground (legible anywhere). So the corner
            # decides the other way round from the seal: a light corner takes
            # the ink marks on the art ("paper" — the art is the ground), a
            # dark corner the white-on-ink row ("ink").
            if chosen == "sampled":
                used = "paper" if used == "ink" else "ink"
            marks = render_marks_row(artist_mark, standard_version, date_letter, reverse=(used == "ink"))
            overlay = f'  {_nest(marks, *box)}\n'
            placed = "marks"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
           f'  <title>Guild of Fine Tuners — sealed artwork</title>\n' + ground + overlay + "</svg>\n")
    return SealedArtwork(svg=svg, width=W, height=H, placed=placed, colourway=used, chosen_by=chosen,
                         region_luminance=luma, box=box, rotation=ROTATION if placed == "seal" else 0.0)
