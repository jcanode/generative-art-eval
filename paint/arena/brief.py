"""What every model is told. Identical for all models, so the comparison is fair."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..loop.critics import _style_guide_section
from .policy import ALLOWED_MODULES

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = """You are a generative artist who paints with code. You write a complete, self-contained Python program
that produces an image pixel by pixel. There are no image models, no reference images, and no drawing software:
only numpy, scipy and Pillow's Image/ImageDraw/ImageFilter/ImageChops/ImageOps/ImageEnhance/ImageColor/ImageMath,
plus these standard-library modules: math, random, colorsys, itertools, functools, collections, dataclasses,
typing, heapq, bisect, statistics, operator, enum, string, re, fractions, decimal.

Contract (the harness enforces it):
- Define a top-level function `paint(width, height, seed)` that RETURNS an RGB image as a numpy array of shape
  (height, width, 3), either uint8 or float in [0, 1]. The harness saves the PNG; do not save or show anything.
- All randomness must come from `seed` (e.g. `np.random.default_rng(seed)`), so the same seed gives the same image.
- The program runs in a sandbox: no files, network, subprocesses, environment, getattr/eval/exec, or
  dunder attributes. Imports outside the list above fail.
- Budget: finish in under 60 seconds on one CPU core and under 2 GB of memory at the requested size.
  Vectorise per-pixel work with numpy; Python loops over strokes, shapes or layers are fine, loops over
  pixels are too slow.
- Paint "in the manner of" a tradition. Never reproduce a specific existing artwork, character or logo.

How to answer: a few sentences on your plan, then exactly ONE fenced ```python block containing the whole
program. When you revise, always send the complete program again, not a diff."""

CRITIQUE_REQUEST = """Above is the image your program v{n} rendered (seed {seed}, {width}x{height}), plus the
harness's technical checks. Look at it carefully, as a demanding art director would.

1. Critique it under these headings, with short concrete bullets: reads wrong (an object that reads as
   something else), flat, cluttered, unrecognizable (things a stranger could not name), technical, strengths.
2. Give it a line exactly like `Score: N/10` as a finished piece in the requested tradition.
3. Then send a complete revised program that fixes the most important problems. If you believe it cannot be
   improved further, say so and send the same program again.

This is revision {n} of at most {max_iters}; the last successful version is what gets judged."""

ERROR_REQUEST = """Your program v{n} did not produce an image.

{feedback}

Fix it and send the complete corrected program. This counts as one of your {max_iters} attempts."""


def describe_scene(scene_dict: dict) -> str:
    """The scene in plain words plus the exact spec, so models know what to draw and where."""
    objs = scene_dict.get("objects", [])
    words = ", ".join(
        f"{o.get('count', 1)} {o['kind']}{'s' if o.get('count', 1) > 1 else ''}" if o.get("count", 1) > 1 else o["kind"]
        for o in objs)
    spec = {k: scene_dict[k] for k in ("title", "canvas", "light", "palette", "horizon", "objects") if k in scene_dict}
    return f"""Scene: "{scene_dict.get('title', 'untitled')}" with {words}.

The exact layout (coordinates are fractions of the canvas from the top-left; `size` is a fraction of the canvas
height; for sea/field/table/hills `y` is where the region starts; for trees, buildings, boats, vases, cups,
teapots and fruit `(x, y)` is the base where the object touches the ground or water; light `azimuth` is the
direction light comes FROM in degrees, 0 = right, 90 = below, 180 = left, 270 = above; palette hints are
suggestions):

```yaml
{yaml.safe_dump(spec, sort_keys=False).strip()}
```"""


def task_message(task: dict, scene_dict: dict, width: int, height: int, seed: int) -> str:
    style = task.get("style")
    guide = _style_guide_section(style) if style else ""
    tradition = task.get("tradition") or (guide or f"Style: {style}")
    body = describe_scene(scene_dict) if scene_dict else f"Subject: {task.get('prompt', '')}"
    return f"""Task: {task.get('title', task['id'])}

Paint this in the following tradition:
{tradition}

{body}

Canvas: {width} x {height} pixels. Seed for this run: {seed}. Every object in the scene should be recognisable
to a stranger, and the image should read unmistakably as the tradition above."""


def load_tasks(path=None) -> list[dict]:
    p = Path(path) if path else ROOT / "evals" / "arena" / "tasks.yaml"
    data = yaml.safe_load(p.read_text())
    return data["tasks"]


def allowed_modules_text() -> str:
    return ", ".join(sorted(ALLOWED_MODULES))
