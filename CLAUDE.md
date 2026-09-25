# Working in this repo (for Claude Code and other agents)

This is a procedural "code-painting" studio: numpy/Pillow/scipy only, no image
models, no reference images. Read `docs/STYLE_GUIDE.md` before touching a style.

## The rule that matters most

**Never report a render as good from code alone. Open the PNG and look at it.**
Use the Read tool on the image (or a contact sheet from `python3 -m paint sheet`).
Then write the critique before changing anything.

## The loop, driven by an agent

    python3 -m paint loop evals/suite/07_lone_tree.yaml --style screenprint --critic manual

1. The command renders `v1/image.png`, runs the technical checks, and stops with
   status `awaiting_critique`.
2. Open `v1/image.png`. Write `v1/critique.json`. `critique.template.json` lists
   the fields, the knobs with their ranges, and the object indices. Use the
   headings: reads_wrong / flat / cluttered / unrecognizable / technical /
   strengths. Put knob changes in `adjustments`. If the fix needs code, change
   the code and describe it in `code_changes`.
3. Re-run the same command. It applies the revisions and renders v2. The loop
   is capped at 4 versions and keeps all of them; `best.png` is the
   highest-scoring version, and earlier versions win ties.

## Rules

- Holdout scenes in `evals/holdout/` are never iterated on; the loop refuses them.
- All randomness goes through `paint.core.rng.Rng` children. Never use global `np.random`.
- No Python loops over pixels. Loops over strokes, objects or layers are fine.
- Budget: a 1024 px render should take under 60 s and 2 GB. The guard kills at 300 s / 4 GiB.
- Don't copy specific artworks, characters or logos. Painting "in the manner of" a
  pre-1929 technique is fine.
- Model-written programs (arena) only ever run through `paint.arena.sandbox.run_program`, never
  imported or exec'd in the studio process.
- Tests: `python3 -m pytest -q tests`.
