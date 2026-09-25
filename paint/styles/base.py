"""Shared scaffolding for style modules.

A style module exposes:

    NAME: str
    DEFAULTS: dict            # revisable knobs, all JSON-serialisable
    KNOBS: dict               # knob -> (min, max, description) for the revision loop
    RULES: dict               # style-specific technical-check thresholds
    render(scene, seed, params=None) -> bytes          # PNG
    render_array(scene, seed, params=None) -> (img, info)

``info`` carries facts the technical checks rely on (ink count, etc.).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from ..core.color import hex_to_rgb, luminance
from ..core.geom import Frame
from ..core.guard import check_canvas
from ..core.io import png_bytes
from ..core.rng import Rng
from ..scene.shapes import Shape, build_shapes
from ..scene.spec import Scene


@dataclass
class Context:
    scene: Scene
    seed: int
    params: dict
    frame: Frame
    rng: Rng
    shapes: list[Shape]
    info: dict = field(default_factory=dict)

    @property
    def light_dir(self):
        return self.scene.light.direction

    def px(self, v: float) -> float:
        """Scale a length given in px-at-1024 to this canvas."""
        return v * self.frame.scale


def merge_params(defaults: dict, params: dict | None) -> dict:
    out = dict(defaults)
    for k, v in (params or {}).items():
        if k not in defaults:
            raise KeyError(f"unknown style parameter {k!r}; known: {sorted(defaults)}")
        out[k] = v
    return out


def clamp_params(params: dict, knobs: dict) -> dict:
    out = dict(params)
    for k, (lo, hi, _) in knobs.items():
        if k in out and isinstance(out[k], (int, float)) and not isinstance(out[k], bool):
            v = min(max(out[k], lo), hi)
            out[k] = type(params[k])(v) if isinstance(params[k], int) else float(v)
    return out


def make_context(scene: Scene, seed: int | None, params: dict, style_name: str) -> Context:
    seed = scene.seed if seed is None else int(seed)
    check_canvas(scene.width, scene.height)
    frame = Frame(scene.width, scene.height)
    rng = Rng(seed, style_name)
    # Geometry is seeded independently of style so all styles share one layout.
    shapes = build_shapes(scene, frame, Rng(seed, "geometry"))
    return Context(scene, seed, params, frame, rng, shapes)


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]


def encode(img: np.ndarray, style: str, ctx: Context) -> bytes:
    meta = {
        "paint:style": style,
        "paint:seed": ctx.seed,
        "paint:params": json.dumps(ctx.params, sort_keys=True, default=str),
        "paint:info": json.dumps(ctx.info, sort_keys=True, default=str),
    }
    return png_bytes(img, meta)


# Palette helpers -----------------------------------------------------------

MOOD_DEFAULTS = {
    "warm": ["#f2c14e", "#e8643c", "#2a8c88", "#1d2d4a"],
    "cool": ["#a8d0db", "#2a8c88", "#e8643c", "#1d2d4a"],
    "night": ["#f2d16b", "#3d5a80", "#98c1d9", "#141b2d"],
    "neutral": ["#e9c46a", "#d1495b", "#2e86ab", "#22223b"],
}

ROLE_TARGETS = {
    "sky": "#f0c987", "water": "#2e6f95", "ground": "#8a6a3a", "primary": "#2e6f95",
    "secondary": "#4f8a6e", "accent": "#e1552f", "dark": "#1b2238", "light": "#f4e3b5",
    "foliage": "#3c7a4a", "wood": "#6b4428", "stone": "#6f6a64",
}


def scene_hints(scene: Scene) -> list[np.ndarray]:
    hints = scene.palette.hints or MOOD_DEFAULTS.get(scene.palette.mood, MOOD_DEFAULTS["neutral"])
    return [hex_to_rgb(h) for h in hints]


def nearest(colors: list[np.ndarray], target) -> int:
    t = hex_to_rgb(target) if isinstance(target, str) else np.asarray(target)
    # Weighted RGB distance, with luminance mattering a bit more than hue.
    d = [float(np.sum((c - t) ** 2 * np.array([0.3, 0.45, 0.25])) + 0.6 * (luminance(c) - luminance(t)) ** 2) for c in colors]
    return int(np.argmin(d))
