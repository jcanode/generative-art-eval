"""Seeded, fully vectorised noise fields."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .rng import Rng


def value_noise(shape, cell: float, rng: Rng, order: int = 3):
    """Smooth noise in [0, 1]: a random lattice of spacing ``cell`` px, spline-upsampled."""
    H, W = shape
    gh, gw = int(np.ceil(H / cell)) + 4, int(np.ceil(W / cell)) + 4
    lattice = rng.random((gh, gw)).astype(np.float32)
    ys = np.arange(H, dtype=np.float32) / cell + 1.5
    xs = np.arange(W, dtype=np.float32) / cell + 1.5
    # map_coordinates over a separable grid via outer indexing.
    Yc, Xc = np.meshgrid(ys, xs, indexing="ij")
    out = ndimage.map_coordinates(lattice, [Yc, Xc], order=order, mode="reflect")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fbm(shape, cell: float, rng: Rng, octaves: int = 4, gain: float = 0.5, lacunarity: float = 2.0):
    """Fractal sum of value noise, normalised to [0, 1]."""
    total = np.zeros(shape, dtype=np.float32)
    amp, norm, c = 1.0, 0.0, float(cell)
    for i in range(octaves):
        total += amp * value_noise(shape, max(c, 1.0), rng.child(f"oct{i}"))
        norm += amp
        amp *= gain
        c /= lacunarity
    return total / norm


def white(shape, rng: Rng):
    return rng.random(shape).astype(np.float32)


def blurred_white(shape, rng: Rng, sigma):
    """Gaussian-filtered white noise, rescaled to roughly [0, 1]. ``sigma`` may be (sy, sx)."""
    n = ndimage.gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), sigma)
    s = n.std() or 1.0
    return np.clip(0.5 + n / (4 * s), 0.0, 1.0)


def normalize(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
