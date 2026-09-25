# Studio style guide

This is the contract that every style module and every core primitive is built
against. When several people or agents work in parallel, this document decides
disputes.

## Shared rules (all styles)

1. **Reproducible.** All randomness comes from `paint.core.rng.Rng(seed, *labels)`.
   Never call `np.random` global functions or `random`. Derive sub-streams with
   `rng.child("label")` so adding a new consumer never shifts existing ones.
2. **Vectorised.** Per-pixel work is numpy array math. Python loops are only
   allowed over *strokes*, *objects*, or *layers*, never over pixels.
3. **Canvas units.** Scene coordinates are normalised: `x, y` in `[0, 1]`,
   origin top-left, `size` is a fraction of the canvas *height*. Styles convert
   with `paint.core.geom.Frame`.
4. **Float working space.** Images are `float32` arrays in `[0, 1]`, shape
   `(H, W, 3)` (colour) or `(H, W)` (masks, ink density). Convert to 8-bit only
   in `paint.core.io.save_png`.
5. **Masks are anti-aliased.** Shapes come from `paint.scene.shapes` as signed
   distance fields; `paint.core.masks.sdf_to_mask` converts them with a
   one-pixel smooth edge. When an SDF is unavailable, rasterise at 3x and
   downsample (`supersample_mask`).
6. **Simplify on purpose.** Each style has a *tradition* that tells us what to
   leave out. We never fake photoreal lighting on primitive shapes. Light
   direction is expressed through the medium: a halftone ramp, a dry-brush
   side, a warm/cool stroke split.
7. **Budget.** A 1024 px render must finish in < 60 s on a laptop CPU and stay
   below 2 GB RAM. The harness enforces 300 s / 4 GB hard limits.
8. **No copies.** We paint *in the manner of* traditions and pre-1929 painters'
   techniques. No specific artwork, character, logo or brand is reproduced.

## Scene objects are style-neutral

A style decides *how* to draw a `sun`, but every style must draw every object
kind in `paint.scene.shapes.KINDS` in a way a stranger would name correctly.
The silhouette comes from the shared SDF so the three styles agree on *what*
is where; styles vary the *treatment*.

## Screenprint (mid-century)

* **Inks:** 3-5 flat inks from the scene palette, plus paper. Never more than
  the rubric's limit (default 6 including paper).
* **Overprint:** layers composite with multiply, in printing order light to dark.
  Overlaps produce a third colour for free; use it.
* **Tone:** only via halftone dots or line screens on a single ink. Shadow side
  is away from `light.direction`.
* **Imperfection:** every ink plate has a small seeded misregistration offset
  (1-4 px at 1024), ink-starved speckle, and slightly uneven density.
* **Paper:** warm off-white with fibre noise; paper shows through as a colour.
* **Composition:** big flat shapes, strong silhouettes, clear figure/ground.

## Sumi-e (ink wash)

* **Monochrome.** Black ink on warm paper; at most one small accent seal or
  wash colour (red seal, or a faint indigo). Saturation stays low.
* **Economy.** 40-70 % of the paper stays empty. Emptiness is the sky, water,
  mist.
* **Strokes:** bristle brushes that load, run dry, split and taper. Each object
  is a handful of confident strokes, not an outline.
* **Washes:** diluted ink pools, with darker edges where the water dried
  (the "coffee ring" effect) and soft bleeding into paper fibres.
* **Distance:** far things are paler and wetter; near things are darker, drier.

## Impasto (in the manner of late-1880s post-impressionism)

* **Everything is a stroke.** No flat fills: the canvas is covered by short,
  thick, directional strokes whose orientation follows a flow field.
* **Flow:** skies swirl around vortices (suns, moons, stars, cloud centres);
  ground strokes follow the terrain; objects are stroked along their own form.
* **Colour:** saturated, complementary contrasts (cobalt/ultramarine vs.
  chrome yellow, viridian vs. vermilion). Stroke colour jitters in hue and
  value.
* **Contours:** dark, broken outlines (Prussian blue / near black) around
  major objects.
* **Relief:** strokes have ridges; a fake emboss from the stroke height map
  gives the paint body.

## Critique vocabulary

When critiquing a render (generator loop, not the judge), use these headings:
**reads wrong** (object misidentified), **flat** (lacks tonal structure),
**cluttered** (too many competing marks), **unrecognizable**, **style drift**
(looks like another style), **technical** (artefacts, seams, cropping).
