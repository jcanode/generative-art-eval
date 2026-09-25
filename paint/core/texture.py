"""Substrate textures: paper for prints and ink, primed canvas for oil."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .color import hex_to_rgb
from .noise import blurred_white, fbm
from .rng import Rng


def paper_fibres(shape, rng: Rng, scale: float = 1.0):
    """Paper fibre structure in [0, 1]: long thin streaks at random angles plus pulp mottling."""
    H, W = shape
    fib = np.zeros(shape, dtype=np.float32)
    D = int(np.ceil(np.hypot(H, W))) + 2
    cy, cx = (D - H) // 2, (D - W) // 2
    for i, ang in enumerate(rng.uniform(0, np.pi, 3)):
        n = rng.child(f"fib{i}").normal(0, 1, (D, D)).astype(np.float32)
        # Anisotropic blur along a random direction: rotate, blur, rotate back.
        deg = float(np.degrees(ang))
        r = ndimage.rotate(n, deg, reshape=False, order=1, mode="reflect")
        r = ndimage.gaussian_filter(r, (0.6 * scale, 7.0 * scale))
        r = ndimage.rotate(r, -deg, reshape=False, order=1, mode="reflect")
        fib += r[cy : cy + H, cx : cx + W]
    fib /= fib.std() + 1e-6
    mottle = fbm(shape, 90 * scale, rng.child("mottle"), octaves=3)
    return np.clip(0.5 + 0.12 * fib + 0.6 * (mottle - 0.5), 0.0, 1.0).astype(np.float32)


def paper(shape, rng: Rng, tint="#f3ead8", strength: float = 1.0, scale: float = 1.0):
    """Warm paper as (H, W, 3) RGB and the fibre field that made it."""
    fib = paper_fibres(shape, rng, scale)
    grain = blurred_white(shape, rng.child("grain"), 0.7)
    base = hex_to_rgb(tint)
    v = 1.0 + strength * (0.06 * (fib - 0.5) + 0.03 * (grain - 0.5))
    img = np.clip(base[None, None, :] * v[..., None], 0, 1).astype(np.float32)
    return img, fib


def canvas_weave(shape, rng: Rng, period: float = 5.0):
    """Linen canvas weave in [0, 1] (0.5 mean): crossed threads with irregular thickness."""
    H, W = shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    wobble_x = (fbm(shape, 60, rng.child("wx"), octaves=2) - 0.5) * 3
    wobble_y = (fbm(shape, 60, rng.child("wy"), octaves=2) - 0.5) * 3
    warp = 0.5 + 0.5 * np.sin(2 * np.pi * (X + wobble_x) / period)
    weft = 0.5 + 0.5 * np.sin(2 * np.pi * (Y + wobble_y) / period)
    checker = np.sign(np.sin(np.pi * X / period) * np.sin(np.pi * Y / period))
    weave = np.where(checker > 0, warp * 0.7 + weft * 0.3, weft * 0.7 + warp * 0.3)
    slub = blurred_white(shape, rng.child("slub"), (1.0, 12.0))
    return np.clip(0.2 + 0.6 * weave + 0.3 * (slub - 0.5), 0, 1).astype(np.float32)


def emboss(height, light=(-0.6, -0.8), strength: float = 1.0):
    """Shading term (mean ~1) from a height map lit from ``light`` (image-space direction)."""
    gy, gx = np.gradient(height.astype(np.float32))
    lx, ly = light
    n = np.hypot(lx, ly) or 1.0
    # Surfaces facing the light (slope rising away from it) get brighter.
    d = -(gx * lx + gy * ly) / n
    return (1.0 + strength * d).astype(np.float32)
