"""The Guild seal: byte-identical, font-free, and impossible to draw for an
output whose watermark did not agree with its sidecar."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from khaos_attribution import seal
from khaos_attribution.seal import SealRefused, render_seal, seal_for_provenance

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden" / "seal_123456.svg"


def test_the_same_input_is_the_same_bytes():
    assert render_seal("123456") == render_seal("123456")
    assert render_seal("A-1/2", seed=3, rough=True) == render_seal("A-1/2", seed=3, rough=True)


def test_the_golden_seal_has_not_moved():
    """Committed reference. A geometry change is a deliberate act: re-render
    the golden, look at it, commit it with the reason."""
    assert render_seal("123456") == GOLDEN.read_text(encoding="utf-8")


def test_no_text_element_font_or_fetch_in_the_output():
    for number in ("123456", "GUILD/2026-000123", "A B"):
        svg = render_seal(number)
        assert "<text" not in svg and "<textPath" not in svg
        assert "font" not in svg.lower() and "@import" not in svg and "http" not in svg.replace(
            'xmlns="http://www.w3.org/2000/svg"', "")


def test_rendering_needs_no_font_and_no_environment():
    """A fresh interpreter, isolated (-I), an empty environment: no HOME, no
    fontconfig, no PATH. The renderer reads only its own package data."""
    code = ("import sys; sys.path.insert(0, %r); import khaos_attribution.seal as s; "
            "svg = s.render_seal('123456'); "
            "assert 'fontTools' not in sys.modules; "
            "sys.stdout.write(str(len(svg)))") % str(ROOT / "src")
    r = subprocess.run([sys.executable, "-I", "-c", code], env={}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) == len(GOLDEN.read_text(encoding="utf-8"))


def test_the_filter_is_off_by_default_and_seeded_when_on():
    assert "<filter" not in render_seal("1")
    rough = render_seal("1", rough=True, seed=42)
    assert 'seed="42"' in rough and 'filter="url(#rough)"' in rough


def test_a_number_outside_the_charset_is_refused_not_guessed():
    for bad in ("abc", "12_34", "", "  ", "№1"):
        with pytest.raises(SealRefused):
            render_seal(bad)


# ── provenance binding ──────────────────────────────────────────────────────

RECORD = {"schema_version": "1.0.0", "artist_id": "art", "artist_name": "Ava",
          "adapter_version": "run_1", "adapter_hash": "ab" * 32, "prompt": "x",
          "timestamp": "2026-09-13T00:00:00Z", "watermark_id": 36388}


def test_binding_refuses_unless_the_agreement_flag_is_literally_true():
    for doc in ({"record": RECORD},
                {"record": RECORD, "embedded_agrees_with_sidecar": False},
                {"record": RECORD, "embedded_agrees_with_sidecar": None},
                {"record": RECORD, "embedded_agrees_with_sidecar": "true"},
                {"record": RECORD, "embedded_agrees_with_sidecar": 1},
                RECORD, [], "nope"):
        with pytest.raises(SealRefused):
            seal_for_provenance(doc)


def test_binding_reads_the_number_from_the_record_and_never_invents_one():
    doc = {"record": RECORD, "embedded_agrees_with_sidecar": True}
    assert "36388" not in seal_for_provenance(doc)          # the number is drawn, not written
    assert seal_for_provenance(doc) == render_seal("36388")   # the clean seal is the clean seal
    assert seal_for_provenance(doc) == seal_for_provenance(json.loads(json.dumps(doc)))
    with pytest.raises(SealRefused, match="carries no 'watermark_id'"):
        seal_for_provenance({"record": {**RECORD, "watermark_id": None},
                             "embedded_agrees_with_sidecar": True})
    other = {"record": {**RECORD, "generation_id": "g-7"}, "embedded_agrees_with_sidecar": True}
    assert seal_for_provenance(other, number_key="generation_id") == render_seal("G-7")


def test_binding_seed_follows_the_record_so_the_filter_is_deterministic_too():
    a = seal_for_provenance({"record": RECORD, "embedded_agrees_with_sidecar": True}, rough=True)
    b = seal_for_provenance({"record": dict(reversed(list(RECORD.items()))),
                             "embedded_agrees_with_sidecar": True}, rough=True)
    assert a == b, "key order is not part of the record"
    c = seal_for_provenance({"record": {**RECORD, "prompt": "y"}, "embedded_agrees_with_sidecar": True},
                            rough=True)
    assert a != c


# ── centring and the band ───────────────────────────────────────────────────

@pytest.mark.parametrize("number", ["7", "123456", "GUILD/2026-000123", "A B C 1 2 3"])
def test_the_run_is_centred_on_the_top_and_stays_inside_the_band(number):
    glyphs = seal.layout(number)
    r = seal.BASELINE_R
    first_edge = glyphs[0]["theta"] - glyphs[0]["advance_px"] / 2 / r
    last_edge = glyphs[-1]["theta"] + glyphs[-1]["advance_px"] / 2 / r
    assert abs(first_edge + last_edge) < 1e-9, "the run is not centred on the top"
    assert last_edge < math.pi / 2, "the run wraps past the sides"
    s = seal.TYPE_SIZE / 1000.0
    inner_edge = seal.INNER_RULE_R + seal.RULE_WIDTH / 2
    outer_edge = seal.OUTER_RULE_R - seal.RULE_WIDTH / 2
    for g in glyphs:
        box = seal._glyphs()[g["char"]]["bounds"]
        if box is None:
            continue
        top = r + box[3] * s              # font y up = outward
        bottom = r + box[1] * s
        assert inner_edge < bottom and top < outer_edge, (g["char"], bottom, top)


def test_every_glyph_in_the_table_fits_the_band():
    s = seal.TYPE_SIZE / 1000.0
    for ch, g in seal._glyphs().items():
        if g["bounds"] is None:
            continue
        assert seal.INNER_RULE_R + 3.5 < seal.BASELINE_R + g["bounds"][1] * s
        assert seal.BASELINE_R + g["bounds"][3] * s < seal.OUTER_RULE_R - 3.5, ch


def test_size_scales_the_canvas_not_the_geometry():
    small = render_seal("123456", size=200)
    assert 'width="200" height="200" viewBox="0 0 800 800"' in small
    assert re.sub(r'width="\d+" height="\d+"', "", small) == re.sub(r'width="\d+" height="\d+"', "",
                                                                      render_seal("123456"))


def test_the_cli_writes_the_same_bytes(tmp_path):
    out = tmp_path / "s.svg"
    r = subprocess.run([sys.executable, "-m", "khaos_attribution.seal", "123456", str(out)],
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == GOLDEN.read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "khaos_attribution.seal", "bad_id", str(out)],
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED" in r.stderr


def test_the_licence_travels_with_the_table():
    pkg = ROOT / "src" / "khaos_attribution" / "seal"
    assert (pkg / "OFL.txt").read_text().startswith("Copyright 2021 Red Hat, Inc.")
    assert "Reserved Font Name Red Hat" in (pkg / "OFL.txt").read_text()
    assert not list(pkg.glob("*.ttf")) and not list(ROOT.rglob("RedHatDisplay-Bold.ttf"))
