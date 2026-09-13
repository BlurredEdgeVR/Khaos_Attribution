# The Guild seal — geometry and the one hard constraint

A circular stamp carrying a registration number, set on an arc around the
Khaos star. Rendered by `khaos_attribution.seal.render_seal`; bound to a
verified provenance document by `seal_for_provenance`.

## The constraint: no font, ever

The renderer has **no font dependency and no text element at runtime**: no
`<text>`, no `<textPath>`, no `@font-face`, no network fetch. Text on a
curved path in SVG fails silently across renderers — a browser, Illustrator
and a headless rasteriser each place, kern or drop it differently, and one
of them will do so on the machine that matters. This project has already
lost time to it. Every character is a **vector path** from `glyphs.json`, a
table built once from Red Hat Display Bold by `tools/build_glyphs.py`
(fontTools, build time only; the runtime imports nothing but the standard
library). `tests/test_seal.py` refuses any output containing `<text` or
`<textPath`. Do not reintroduce either; if a new glyph is needed, add it
to the charset and rebuild the table.

Charset: `A-Z 0-9 : . - /` and space. Each entry: `d` (SVG path, 1000-unit
em, y up as the font has it), `advance`, `bounds` (`[xmin, ymin, xmax,
ymax]`, `null` for space).

## Canvas

`viewBox="0 0 800 800"`, centre (400, 400). The whole drawing scales by
`size / 800` through `width`/`height`; the viewBox never changes.

| Element | Value |
|---|---|
| Field circle | r 396, fill `#0b0f0e` |
| Outer band rule | r 352, stroke `#ffffff`, width 7, no fill |
| Inner band rule | r 276, stroke `#ffffff`, width 7, no fill |
| Type baseline radius | 290 |
| Type size | 46 (scale `s = 46 / 1000`) |
| Tracking | 6 px after each glyph, none trailing |
| Star | `translate(215,215) scale(0.5992)`, fill and stroke `#ffffff`, stroke-width 3 |
| Foot ornament | polygon `400,684 409,703 428,712 409,721 400,740 391,721 372,712 391,703`; dots r 8 at (268, 684) and (532, 684) |

## Arc layout (exact; implement as given)

Angle `theta` is 0 at the top and positive clockwise.

```
adv_i     = glyph advance × s                # px
total     = Σ adv_i + tracking × (n − 1)
span      = total / 290                      # radians
start     = −span / 2                        # centres the run on the top
theta_i   = start + (run_before_i + adv_i / 2) / 290
px        = 400 + 290 · sin(theta_i)
py        = 400 − 290 · cos(theta_i)

glyph transform:
  translate(px, py) rotate(degrees(theta_i)) scale(s, −s) translate(−adv_i / (2·s), 0)
```

`scale(s, −s)` flips the font's y-up outlines into SVG's y-down canvas.
The final translate centres each glyph on its own advance; letters face
outward as a consequence of the convention, not a special case. A space is
an advance with no mark. The baseline sits at r 290; the tallest glyph
reaches r ≈ 323 and nothing descends below r ≈ 289, so the run stays
inside the band (rules' inner edges at 279.5 and 348.5) at any length up
to the canvas's half circle.

Numbers are formatted to three decimals with no exponent, so the same
input is the same bytes on every machine.

## The optional roughening filter

`render_seal(..., rough=True)` wraps the outer group in a turbulence /
displacement filter seeded by `seed`. It is **off by default**, and several
renderers ignore filters entirely, so the clean drawing is the canonical
one. The filter must never carry meaning.

## Provenance binding

`seal_for_provenance(document)` takes what the Listening Space's provenance
endpoint returns — the `record` plus `embedded_agrees_with_sidecar`. It
**raises `SealRefused`** unless that flag is literally `True`; the seal
asserts the check passed, so drawing one otherwise must be impossible. The
number is `record["watermark_id"]` (override with `number_key`); this module
never generates, assigns, looks up or revokes numbers. The filter seed is
the first four bytes of `sha256` over the record canonicalised as compact,
key-sorted JSON, so one record is always one byte-identical seal.

## Licence

Red Hat Display is © Red Hat, Inc., SIL Open Font License 1.1, Reserved
Font Name "Red Hat". `glyphs.json` is a derived work of that Font Software:
permitted under the OFL provided the licence travels with it (`OFL.txt`,
beside this file) and the derived file carries no reserved name. The TTF
itself is not committed.
