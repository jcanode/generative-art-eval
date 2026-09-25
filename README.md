# paint: a local code-painting studio with an eval suite

Python programs that paint pixel by pixel in a chosen tradition. They use only
numpy, Pillow and scipy: no image models, no reference images, no art
software. A second half measures quality, so changes can be compared instead
of eyeballed.

```
pip install -e .            # numpy, pillow, scipy, pyyaml  (+ "anthropic" for the judge: pip install -e .[judge])
paint render evals/suite/01_harbour_dusk.yaml --style screenprint --seed 7 --out out/
paint loop   evals/suite/07_lone_tree.yaml --style screenprint --critic manual
paint eval   --suite                      # all styles x all suite + holdout scenes -> runs/<id>/report.html
paint serve                               # web studio on http://127.0.0.1:8000
```

(`python3 -m paint ...` works without installing.)

### Running locally (macOS / Linux / Windows)

```
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[judge]"                                # Python 3.10+
python -m pytest -q tests                                # optional: pip install pytest first
paint serve                                              # open http://127.0.0.1:8000, paste your API key in Settings
```

Renders run in a child process: `fork` on Linux, `spawn` on macOS and Windows
(override with `PAINT_MP_START`). The 300 s timeout is enforced everywhere. The
4 GiB memory cap relies on `RLIMIT_AS`, which only Linux enforces, so on
macOS/Windows it is best-effort. A full `paint eval --suite` (3 styles × 13
scenes, with determinism re-renders) takes about 3 minutes on 4 cores;
`PAINT_WORKERS` sets the parallelism.

## Layout

```
paint/
  scene/     spec.py (YAML/JSON scene format), shapes.py (26 object kinds as signed distance fields)
  core/      rng (seeded, label-scoped), guard (timeout + memory cap), geom, sdf, masks (analytic AA +
             supersampling), noise, texture (paper, canvas, emboss), halftone (dots, line screens),
             flow (flow-field builder + vectorised streamlines), brush (bristle stroke engine), color, io
  styles/    screenprint.py, sumie.py, impasto.py  (each: render(scene, seed) -> PNG bytes)
  loop/      agent.py (render -> look -> critique -> revise, max 4, every version kept), critics.py
  evals/     technical.py, judges.py, fidelity.py, pairwise.py (+Elo), calibration.py, suite.py, report.py
  site/      server.py + static/index.html (local web studio)
evals/
  suite/     10 fixed regression scenes      holdout/  3 scenes the generator never iterates on
  rubrics/   <style>.md (edit these)          calibration/  your ratings (ratings.json)
docs/        STYLE_GUIDE.md (the contract every style is built against), STORYBOARD.md
tests/       pytest suite
```

## Scene specs: what to draw, not how

```yaml
title: Harbour at dusk
canvas: {width: 1024, height: 768}
seed: 7
light: {azimuth: 200, elevation: 20, warmth: 0.9}   # azimuth = where light comes FROM (0 right, 90 below, 180 left, 270 above)
palette: {hints: ["#f2c14e", "#e8643c", "#2a8c88", "#1d2d4a"], mood: warm}
horizon: 0.6
objects:
  - {kind: sun, x: 0.68, y: 0.42, size: 0.22}      # x, y in [0,1]; size = fraction of canvas height
  - {kind: sea}                                    # region kinds default to the horizon
  - {kind: boat, x: 0.32, y: 0.76, size: 0.34, variant: sail}
  - {kind: bird, x: 0.3, y: 0.2, size: 0.08, count: 3}
```

Kinds: sun, moon, stars, cloud, mountain, hills, sea, field, river, table, cliff,
tree, pine, cypress, bamboo, reeds, house, lighthouse, windmill, boat, bird, vase,
fruit, teapot, cup, rain. All three styles read the same geometry and vary only
the treatment.

## Styles

| style | how it works |
|---|---|
| `screenprint` | Each ink is a plate (a coverage map). Shapes knock out or overprint (multiply); tone comes only from halftone dots and line screens; each plate gets misregistration, ink starvation and density drift. A figure/ground rule stops an object being printed in the ink behind it. Checks: at most 5 inks and a bounded colour-cluster count. |
| `sumie` | Built against the style guide by a parallel agent. See the module docstring. |
| `impasto` | Built against the style guide by a parallel agent. See the module docstring. |

`paint render ... -p knob=value` overrides any knob. `DEFAULTS` and `KNOBS` in
each module list them with ranges.

## The loop

`paint loop` renders, runs the technical checks, then a critic looks at the PNG
and writes `vN/critique.json` + `critique.md`. The headings are: reads wrong,
flat, cluttered, unrecognizable, technical, strengths. The critic also gives a
1-10 score and concrete revisions (knob changes, object edits). Every version
is kept in `v1/`…`v4/`; `best.png` is the highest-scoring version, and `index.html`
shows them side by side.

Critics: `claude` (vision model; needs `ANTHROPIC_API_KEY`), `manual` (step mode:
the loop stops after each render until an agent or human who opened the image
writes the critique; re-run to continue; see CLAUDE.md), and `heuristic`
(offline pixel statistics, labelled as not a visual critique). Guaranteed exit:
at most 4 versions, a per-render timeout (300 s, process killed), a memory cap
(4 GiB `RLIMIT_AS`, fails loudly), a per-call critic timeout, and a whole-loop
deadline. Holdout scenes are refused.

## Evals

1. **Technical checks** (every render; `paint render` runs them by default):
   renders, time budget, identical pixels on a same-seed re-render, not
   blank/near-uniform, no subject cropped off the canvas (computed from the
   scene geometry on a padded frame), and the style's `RULES` (ink count,
   monochrome + empty paper, stroke texture + saturation).
2. **Subject fidelity (blind):** the judge sees only pixels. Images are
   re-encoded without metadata and sent without file names, spec, code, or the
   generator's critique. It lists what it sees; names are matched to spec kinds
   through a synonym table. Score = the share of kinds named. Names that match
   nothing are reported as *unexpected* (the "helmet that reads as a phone" case).
   The judge also guesses the medium, which gives a blind check for style drift.
3. **Style fidelity:** 1-5 per criterion with a one-line reason, against
   `evals/rubrics/<style>.md`. Criteria are parsed from `##` headings, and each
   rubric's hash is recorded in every run.
4. **Pairwise preference:** every pair is judged twice, in random order and then
   swapped. Only consistent verdicts count and move Elo (`runs/elo.json`,
   players `style@run-label`). The suite compares each render with the previous
   run of the same style and scene (using the rubric), and each style with the
   screenprint baseline (overall quality).
5. **Human calibration:** `paint calibrate --init` renders ~20 varied images,
   including deliberately degraded ones. Rate them in `evals/calibration/ratings.json`
   or on the studio's Calibration tab. `paint calibrate` then reports
   Spearman, Kendall, quadratic kappa, exact and within-one agreement, MAE, judge
   bias and pair concordance, plus a plain-language verdict on how far to trust
   the judge.

`paint eval --suite` does all of this and writes `runs/<id>/report.html`:
thumbnails, check results, scores, what the judge saw, the pairwise verdicts
with reasons, Elo, calibration, and a diff against the previous run (pixel
change, score deltas, checks fixed or regressed, previous thumbnail). Holdout
scores are reported separately. Judge answers are cached by pixel hash in
`runs/judge_cache.json`, so unchanged images are never re-judged.

Without an API key the judge falls back to an offline **mock** that only runs
the pipeline. It cannot see subjects, and the report shows a banner saying its
numbers are meaningless.

The judge model defaults to `claude-opus-5` (`--model` or `PAINT_JUDGE_MODEL`
changes it). Calls use vision plus structured JSON outputs, with server-side
refusal fallbacks enabled (`PAINT_JUDGE_FALLBACKS=0` turns them off).

## Web studio

`paint serve` runs a local site on 127.0.0.1 with tabs for:

- **Render:** scene picker and YAML editor, knob sliders, and check results.
- **Loop:** includes a form for being the critic yourself.
- **Evals:** run the suite, list reports, and see Elo.
- **Judge images:** upload your own generated images and get a blind judgement or a pairwise compare.
- **Calibration:** rate images 1-5 and measure the judge against you.
- **Settings:** enter your Anthropic API key. It stays in server memory only, never on disk or in logs.

## Guarding against gaming the eval

- The judge sees pixels only: PNG text chunks are stripped, and file names, spec, code and critiques are never sent.
- The generator's critic and the eval judge are separate: the critic may see the spec, the judge never does.
- Holdout scenes are refused by the loop and reported separately.
- Pairwise verdicts must survive a position swap.
- Rubric hashes and a code fingerprint are recorded in every run.
- Human calibration shows how far to trust the judge at all.
