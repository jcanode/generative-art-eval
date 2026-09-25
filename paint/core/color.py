"""Colour utilities: hex parsing, blending modes, HSV jitter, palettes."""
from __future__ import annotations

import colorsys

import numpy as np

NAMED = {
    "paper": "#f3ead8",
    "black": "#161514",
    "white": "#ffffff",
    "sumi": "#141312",
    "vermilion": "#d2402a",
    "chrome_yellow": "#f2b61c",
    "cobalt": "#1f4fa3",
    "ultramarine": "#2a3a8f",
    "prussian": "#15243a",
    "viridian": "#2f7d62",
    "ochre": "#c98a2e",
    "teal": "#2a8c88",
    "navy": "#1d2d4a",
    "coral": "#e8643c",
    "cream": "#f4e6c4",
    "mustard": "#d9a520",
    "olive": "#6b7a2e",
    "rust": "#a4462a",
}


def hex_to_rgb(c) -> np.ndarray:
    """'#rrggbb', a named colour, or an RGB triple (0-1 or 0-255) -> float32[3] in [0,1]."""
    if isinstance(c, str):
        c = NAMED.get(c.lower(), c)
        s = c.lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            raise ValueError(f"not a colour: {c!r}")
        return np.array([int(s[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    arr = np.asarray(c, dtype=np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


def rgb_to_hex(rgb) -> str:
    r, g, b = (np.clip(np.asarray(rgb), 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def luminance(rgb) -> float | np.ndarray:
    rgb = np.asarray(rgb)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def multiply(base, ink, density):
    """Multiply-blend an ink colour onto ``base`` (H,W,3) with per-pixel density (H,W)."""
    ink = np.asarray(ink, dtype=np.float32)
    return base * (1.0 - density[..., None] * (1.0 - ink))


def over(base, color, alpha):
    """Normal alpha compositing; ``color`` is RGB[3] or (H,W,3); ``alpha`` (H,W)."""
    a = alpha[..., None]
    return base * (1.0 - a) + np.asarray(color, dtype=np.float32) * a


def jitter_hsv(rgb, rng, dh=0.02, ds=0.08, dv=0.08):
    h, s, v = colorsys.rgb_to_hsv(*np.clip(rgb, 0, 1))
    h = (h + rng.normal(0, dh)) % 1.0
    s = float(np.clip(s + rng.normal(0, ds), 0, 1))
    v = float(np.clip(v + rng.normal(0, dv), 0, 1))
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def mix(a, b, t):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return a * (1 - t) + b * t


def shade(rgb, amount):
    """Darken (amount>0) or lighten (amount<0) an RGB colour."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb * (1 - amount) if amount >= 0 else rgb + (1 - rgb) * (-amount)
