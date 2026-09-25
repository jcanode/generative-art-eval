"""Technical checks: automatic, cheap, run on every render.

1. renders       - the PNG exists and decodes
2. time_budget   - the render finished inside the budget
3. deterministic - re-rendering with the same seed gives identical pixels
4. not_blank     - not uniform / near-uniform
5. in_frame      - no subject is cropped off the canvas (from scene geometry)
6. style rules   - per-style RULES (ink count, monochrome, empty paper, ...)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.geom import Frame
from ..core.io import load_rgb
from ..core.rng import Rng
from ..scene.shapes import REGION_KINDS, build_shapes
from ..scene.spec import Scene, load_scene
from ..styles import get_style
from . import metrics

TIME_BUDGET_S = 60.0
IN_FRAME_MIN = 0.85
DECORATIVE = {"stars", "rain", "reeds"}  # scattered by design; partial cropping is fine


@dataclass(frozen=True)
class PaddedFrame(Frame):
    """A frame whose grid extends ``pad`` px beyond every edge (for off-canvas geometry)."""

    pad: int = 0

    @cached_property
    def grid(self):
        ys = np.arange(-self.pad, self.height + self.pad, dtype=np.float32) + 0.5
        xs = np.arange(-self.pad, self.width + self.pad, dtype=np.float32) + 0.5
        X, Y = np.meshgrid(xs, ys)
        return X, Y

    @property
    def shape(self):
        return (self.height + 2 * self.pad, self.width + 2 * self.pad)


def _check(name, passed, detail, value=None, **extra):
    d = {"name": name, "passed": bool(passed), "detail": detail}
    if value is not None:
        d["value"] = value
    d.update(extra)
    return d


def in_frame_fractions(scene: Scene, seed: int) -> list[dict]:
    """For every non-region object: the fraction of its silhouette that lies on the canvas."""
    scale = min(1.0, 384.0 / max(scene.width, scene.height))  # geometry at low res is plenty
    w, h = max(8, int(scene.width * scale)), max(8, int(scene.height * scale))
    pad = int(max(w, h) * 0.5)
    f = PaddedFrame(w, h, pad)
    out = []
    for shape in build_shapes(scene, f, Rng(seed, "geometry")):
        if shape.kind in REGION_KINDS or shape.kind in DECORATIVE or shape.kind == "hills":
            continue
        m = shape.mask
        total = float(m.sum())
        inside = float(m[pad : pad + h, pad : pad + w].sum())
        frac = inside / total if total > 0 else 0.0
        out.append({"kind": shape.kind, "fraction": round(frac, 3), "visible": total > 0})
    return out


def style_rule_checks(style: str, img: np.ndarray, info: dict) -> list[dict]:
    rules = getattr(get_style(style), "RULES", {})
    out = []
    for rule, limit in rules.items():
        if rule == "max_inks":
            n = info.get("ink_count")
            out.append(_check("style:max_inks", n is not None and n <= limit,
                              f"{n} inks used (max {limit})", n))
        elif rule in ("max_color_clusters", "max_color_clusters_per_ink"):
            if rule.endswith("per_ink"):
                if not info.get("ink_count"):
                    out.append(_check("style:flat_color", True, "ink count unknown for this image (skipped)", skipped=True))
                    continue
                limit = limit * int(info["ink_count"])
            n = metrics.color_clusters(img, bits=3, min_frac=0.01)
            out.append(_check("style:flat_color", n <= limit,
                              f"{n} colour clusters >=1% area at 3 bits/channel (max {limit}); continuous-tone images score far higher", n))
        elif rule == "max_mean_saturation":
            v = float(metrics.saturation(img).mean())
            out.append(_check("style:monochrome", v <= limit, f"mean saturation {v:.3f} (max {limit})", round(v, 4)))
        elif rule == "min_empty_fraction":
            v = metrics.empty_fraction(img)
            out.append(_check("style:empty_paper", v >= limit, f"{v:.0%} of the paper left empty (min {limit:.0%})", round(v, 4)))
        elif rule == "max_empty_fraction":
            v = metrics.empty_fraction(img)
            out.append(_check("style:not_too_empty", v <= limit, f"{v:.0%} empty (max {limit:.0%})", round(v, 4)))
        elif rule == "min_edge_density":
            v = metrics.edge_density(img)
            out.append(_check("style:stroke_texture", v >= limit, f"edge density {v:.3f} (min {limit}); flat fills score low", round(v, 4)))
        elif rule == "min_mean_saturation":
            v = float(metrics.saturation(img).mean())
            out.append(_check("style:saturated", v >= limit, f"mean saturation {v:.3f} (min {limit})", round(v, 4)))
        else:
            out.append(_check(f"style:{rule}", True, "no checker for this rule (skipped)", skipped=True))
    return out


def png_info(path) -> dict:
    try:
        text = Image.open(path).text  # type: ignore[attr-defined]
        return json.loads(text.get("paint:info", "{}"))
    except Exception:
        return {}


def run_checks(png, scene, style: str, seed: int | None = None, params: dict | None = None,
               seconds: float | None = None, determinism: bool = True,
               time_budget: float = TIME_BUDGET_S) -> dict:
    scene = scene if isinstance(scene, Scene) else load_scene(scene)
    seed = scene.seed if seed is None else int(seed)
    checks = []
    try:
        img = load_rgb(png)
        checks.append(_check("renders", True, f"{img.shape[1]}x{img.shape[0]} PNG decodes"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("renders", False, f"could not load PNG: {exc}"))
        return {"passed": False, "checks": checks, "metrics": {}}
    if (img.shape[1], img.shape[0]) != (scene.width, scene.height):
        checks.append(_check("canvas_size", False, f"{img.shape[1]}x{img.shape[0]} != spec {scene.width}x{scene.height}"))

    if seconds is not None:
        checks.append(_check("time_budget", seconds <= time_budget, f"{seconds:.1f}s (budget {time_budget:.0f}s)", round(seconds, 2)))

    if determinism:
        from ..render import render_bytes

        png2, _, _, _ = render_bytes(scene, style, seed, params)
        import io

        img2 = np.asarray(Image.open(io.BytesIO(png2)).convert("RGB"), dtype=np.float32) / 255.0
        same = img2.shape == img.shape and metrics.pixel_hash(img2) == metrics.pixel_hash(img)
        diff = metrics.mean_abs_diff(img, img2)
        checks.append(_check("deterministic", same, "identical pixels on re-render" if same else f"re-render differs (mean abs diff {diff:.4f})"))

    L = metrics.luminance(img)
    std = float(L.std())
    clusters = metrics.color_clusters(img, bits=4, min_frac=0.002)
    blank = std < 0.03 or clusters < 3
    checks.append(_check("not_blank", not blank, f"luminance std {std:.3f}, {clusters} colour clusters", round(std, 4)))

    fr = in_frame_fractions(scene, seed)
    cropped = [f"{d['kind']} ({d['fraction']:.0%} visible)" for d in fr if d["fraction"] < IN_FRAME_MIN]
    checks.append(_check("in_frame", not cropped,
                         "all subjects on canvas" if not cropped else "cropped: " + ", ".join(cropped),
                         objects=fr))

    checks.extend(style_rule_checks(style, img, png_info(png)))
    hard = [c for c in checks if not c.get("skipped")]
    return {"passed": all(c["passed"] for c in hard), "checks": checks, "metrics": metrics.summary(img)}
