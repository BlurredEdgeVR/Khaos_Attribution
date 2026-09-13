"""The seal on hero artwork: exact placement, a colourway chosen against the
corner it sits in, the source untouched, and the marks row below the size at
which the ring stops being legible."""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL.Image")
from PIL import Image  # noqa: E402

from khaos_attribution import seal  # noqa: E402
from khaos_attribution.seal import SealRefused, artwork as A  # noqa: E402

VALUES = ("123456", "CG", "1.0", "A")
ASSETS = Path(__file__).resolve().parents[1] / "src" / "khaos_attribution" / "seal" / "assets"


def _png(w, h, base, corner=None, corner_frac=0.2):
    """A flat image, optionally with a differently lit top-right corner."""
    im = Image.new("RGB", (w, h), base)
    if corner is not None:
        cw, ch = int(w * corner_frac), int(h * corner_frac)
        im.paste(Image.new("RGB", (cw, ch), corner), (w - cw, 0))
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()


def _nested(svg):
    m = re.search(r'<svg x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" viewBox="0 0 (\d+) (\d+)"', svg)
    return tuple(float(v) for v in m.groups()) if m else None


@pytest.mark.parametrize("w,h", [(800, 800), (1200, 1200), (1600, 900), (600, 1000), (2000, 2000)])
def test_placement_is_measured_against_the_width_at_every_size(w, h):
    out = A.place_seal_on_artwork(_png(w, h, (200, 200, 200)), *VALUES)
    d, inset = 0.15 * w, 0.05 * w
    assert out.placed == "seal" and out.box == (w - inset - d, inset, d, d)
    x, y, bw, bh, vw, vh = _nested(out.svg)
    assert (x, y, bw, bh, vw, vh) == pytest.approx((w - inset - d, inset, d, d, 800, 800), abs=1e-3)
    cx, cy = w - inset - d / 2, inset + d / 2
    assert f'rotate(-4.000,{cx:.3f},{cy:.3f})' in out.svg and out.rotation == -4.0
    assert f'width="{w}" height="{h}" viewBox="0 0 {w} {h}"' in out.svg
    assert '<image href="data:image/png;base64,' in out.svg and f'width="{w}" height="{h}"/>' in out.svg


@pytest.mark.parametrize("w,h", [(800, 800), (1600, 900), (600, 1000)])
def test_the_seal_can_sit_bottom_right_with_the_same_insets(w, h):
    out = A.place_seal_on_artwork(_png(w, h, (200, 200, 200)), *VALUES, corner="bottom-right")
    d, inset = 0.15 * w, 0.05 * w
    assert out.placed == "seal" and out.box == (w - inset - d, h - inset - d, d, d)
    x, y, *_ = _nested(out.svg)
    assert (x, y) == pytest.approx((w - inset - d, h - inset - d), abs=1e-3)
    assert out.rotation == -4.0
    with pytest.raises(SealRefused):
        A.place_seal_on_artwork(_png(w, h, (200, 200, 200)), *VALUES, corner="middle")


def test_the_colourway_is_chosen_by_the_corner_not_the_image_average():
    # a dark sleeve with a bright top-right corner: the average says dark, the corner says light
    dark_with_light_corner = _png(1000, 1000, (20, 20, 20), corner=(240, 240, 240))
    out = A.place_seal_on_artwork(dark_with_light_corner, *VALUES)
    assert out.colourway == "ink" and out.chosen_by == "sampled" and out.region_luminance > 200
    assert '<circle cx="400" cy="400" r="396" fill="#0b0f0e"/>' in out.svg      # white art on an ink disc
    light_with_dark_corner = _png(1000, 1000, (235, 235, 235), corner=(15, 15, 15))
    out = A.place_seal_on_artwork(light_with_dark_corner, *VALUES)
    assert out.colourway == "paper" and out.region_luminance < 40
    assert '<circle cx="400" cy="400" r="396" fill="#f4f1ea"/>' in out.svg      # ink art on a paper disc
    assert A.place_seal_on_artwork(_png(800, 800, (250, 250, 250)), *VALUES).colourway == "ink"
    assert A.place_seal_on_artwork(_png(800, 800, (5, 5, 5)), *VALUES).colourway == "paper"


def test_the_override_wins_over_the_automatic_choice():
    light = _png(800, 800, (250, 250, 250))
    out = A.place_seal_on_artwork(light, *VALUES, colourway="paper")
    assert out.colourway == "paper" and out.chosen_by == "override" and out.region_luminance is None
    assert '#f4f1ea' in out.svg
    with pytest.raises(SealRefused):
        A.place_seal_on_artwork(light, *VALUES, colourway="sepia")


def test_the_source_artwork_is_byte_identical_after_the_call(tmp_path):
    data = _png(900, 700, (90, 120, 160))
    path = tmp_path / "hero.png"; path.write_bytes(data)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    out = A.place_seal_on_artwork(path, *VALUES)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert hashlib.sha256(data).hexdigest() == before
    # and the composite embeds exactly those bytes, not a re-encoding
    import base64
    embedded = re.search(r'base64,([A-Za-z0-9+/=]+)"', out.svg).group(1)
    assert base64.b64decode(embedded) == data
    assert out.svg != A.place_seal_on_artwork(path, "654321", "CG", "1.0", "A").svg


def test_below_the_threshold_the_marks_row_stands_in_bottom_right_and_never_a_shrunken_seal():
    out = A.place_seal_on_artwork(_png(399, 399, (240, 240, 240)), *VALUES)
    assert out.placed == "marks" and out.rotation == 0.0
    assert "REGISTER" not in out.svg and 'viewBox="0 0 800 800"' not in out.svg
    mh = 0.12 * 399; mw = mh * 670 / 238; inset = 0.05 * 399
    assert out.box == pytest.approx((399 - inset - mw, 399 - inset - mh, mw, mh))
    x, y, bw, bh, vw, vh = _nested(out.svg)
    assert (vw, vh) == (670, 238) and (x, y) == pytest.approx((399 - inset - mw, 399 - inset - mh), abs=1e-3)
    assert out.colourway == "paper"
    # a light corner takes the ink marks on the art; a dark corner the reversed row (its own ground)
    assert '<rect width="670" height="238"' not in A.place_seal_on_artwork(_png(399, 399, (240, 240, 240)), *VALUES).svg
    assert '<rect width="670" height="238" fill="#0b0f0e"/>' in A.place_seal_on_artwork(_png(399, 399, (10, 10, 10)), *VALUES).svg


def test_when_nothing_is_legible_nothing_is_placed():
    out = A.place_seal_on_artwork(_png(200, 200, (128, 128, 128)), *VALUES)
    assert out.placed is None and out.box is None and out.colourway is None
    assert "<svg x=" not in out.svg and "<image" in out.svg
    # a wide banner too short for the seal at 5 % + 15 % of its width falls back the same way
    out = A.place_seal_on_artwork(_png(1600, 200, (128, 128, 128)), *VALUES)
    assert out.placed != "seal"


def test_the_renderer_reproduces_the_approved_references_geometry():
    """assets/guild-seal.svg and guild-seal-light.svg are the approved
    design. Our render must place every ring glyph and every punch where
    they do (to a thousandth), in the same colours — the references are
    the acceptance, not a picture to resemble."""
    def transforms(svg):   # flat: x, y, rotation per ring glyph, in order
        return [float(v) for m in re.findall(r'translate\(([-\d.]+),([-\d.]+)\) rotate\(([-\d.]+)\) scale\(0\.044', svg)
                for v in m]
    def punches(svg):
        return [float(v) for m in re.findall(r'translate\(([-\d.]+),([-\d.]+)\) rotate\(([-\d.]+),70,85\)', svg)
                for v in m]
    for name, colourway, field, art in (("guild-seal.svg", "ink", "#0b0f0e", "#ffffff"),
                                        ("guild-seal-light.svg", "paper", "#f4f1ea", "#0b0f0e")):
        ref = (ASSETS / name).read_text(encoding="utf-8")
        ours = seal.render_seal(*VALUES, colourway=colourway)
        assert len(transforms(ref)) == 17 * 3      # REGISTER No: 123456 — spaces set nothing
        assert transforms(ours) == pytest.approx(transforms(ref), abs=1e-3)
        assert len(punches(ref)) == 9 and punches(ours) == pytest.approx(punches(ref), abs=1e-3)
        assert f'r="396" fill="{field}"' in ref and f'r="396" fill="{field}"' in ours
        assert ref.count(art) > 5 and ours.count(art) > 5
        assert 'translate(269.00,167.00) scale(0.42429)' in ref and 'translate(269.00,167.00) scale(0.42429)' in ours
        assert "<text" not in ref
