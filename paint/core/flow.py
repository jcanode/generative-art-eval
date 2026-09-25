"""Flow fields: vector fields that strokes follow.

Build a field by layering contributions, each a vector field with a weight
map, then normalise:

    field = (FlowBuilder(frame, rng)
             .noise(cell=220, weight=0.6)
             .uniform(angle=0.0, weight=1.0, mask=ground)
             .vortex(x=700, y=200, radius=180, strength=2.0)
             .along_sdf(sun_sdf, weight=1.5, reach=60)
             .build())
    paths = field.trace(starts, steps=12, step=3.0)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geom import Frame
from .noise import value_noise
from .rng import Rng


class FlowField:
    def __init__(self, vx: np.ndarray, vy: np.ndarray, fallback=(1.0, 0.0)):
        n = np.hypot(vx, vy)
        # Dead zones (e.g. flat noise plateaus have zero curl) would stall streamlines;
        # give them a fallback direction so every stroke still travels.
        dead = n < 1e-6
        if dead.any():
            vx = np.where(dead, fallback[0], vx)
            vy = np.where(dead, fallback[1], vy)
            n = np.where(dead, 1.0, n)
        self.vx = (vx / n).astype(np.float32)
        self.vy = (vy / n).astype(np.float32)
        self.H, self.W = vx.shape

    @property
    def angle(self) -> np.ndarray:
        return np.arctan2(self.vy, self.vx)

    def sample(self, x, y):
        """Bilinear sample at pixel coords (arrays). Returns unit (vx, vy)."""
        coords = [np.clip(np.asarray(y) - 0.5, 0, self.H - 1), np.clip(np.asarray(x) - 0.5, 0, self.W - 1)]
        vx = ndimage.map_coordinates(self.vx, coords, order=1, mode="nearest")
        vy = ndimage.map_coordinates(self.vy, coords, order=1, mode="nearest")
        n = np.hypot(vx, vy)
        n = np.where(n < 1e-6, 1.0, n)
        return vx / n, vy / n

    def trace(self, starts, steps: int, step: float, both: bool = True):
        """Integrate streamlines (RK2) from ``starts`` (N,2). Returns (N, P, 2) paths.

        Vectorised over particles; the only Python loop is over ``steps``.
        With ``both`` the path extends backwards too, centred on the start.
        """
        starts = np.asarray(starts, dtype=np.float32).reshape(-1, 2)

        def run(sign, n):
            pts = [starts.copy()]
            p = starts.copy()
            prev = None
            for _ in range(n):
                vx, vy = self.sample(p[:, 0], p[:, 1])
                v = np.stack([vx, vy], axis=1) * sign
                if prev is not None:  # keep orientation consistent (fields are sign-ambiguous)
                    flip = (v * prev).sum(axis=1) < 0
                    v[flip] *= -1
                mid = p + v * (step * 0.5)
                mvx, mvy = self.sample(mid[:, 0], mid[:, 1])
                mv = np.stack([mvx, mvy], axis=1)
                flip = (mv * v).sum(axis=1) < 0
                mv[flip] *= -1
                p = p + mv * step
                prev = mv
                pts.append(p.copy())
            return np.stack(pts, axis=1)

        if not both:
            return run(1.0, steps)
        half = max(1, steps // 2)
        fwd = run(1.0, half)
        bwd = run(-1.0, steps - half)
        return np.concatenate([bwd[:, :0:-1], fwd], axis=1)


class FlowBuilder:
    def __init__(self, frame: Frame, rng: Rng):
        self.frame = frame
        self.rng = rng
        self.vx = np.zeros(frame.shape, dtype=np.float32)
        self.vy = np.zeros(frame.shape, dtype=np.float32)
        self._n = 0

    def _add(self, vx, vy, weight):
        w = np.asarray(weight, dtype=np.float32)
        self.vx += vx * w
        self.vy += vy * w
        self._n += 1
        return self

    def uniform(self, angle: float, weight=1.0, mask=None):
        w = weight if mask is None else weight * mask
        return self._add(np.float32(np.cos(angle)), np.float32(np.sin(angle)), w)

    def noise(self, cell: float = 200.0, weight=1.0, curl: bool = True):
        """Smooth random directions. ``curl=True`` makes a divergence-free swirl."""
        r = self.rng.child(f"noise{self._n}")
        if curl:
            pot = value_noise(self.frame.shape, cell, r)
            gy, gx = np.gradient(pot)
            s = cell / 2.0
            return self._add(gy * s, -gx * s, weight)
        ang = value_noise(self.frame.shape, cell, r) * 4 * np.pi
        return self._add(np.cos(ang), np.sin(ang), weight)

    def vortex(self, x: float, y: float, radius: float, strength: float = 1.0,
               ccw: bool = True, inward: float = 0.15):
        """Swirl around (x, y) px. Strongest near ``radius``, fading beyond ~3 radii."""
        X, Y = self.frame.grid
        dx, dy = X - x, Y - y
        r = np.hypot(dx, dy) + 1e-3
        sgn = 1.0 if ccw else -1.0
        tx, ty = -dy / r * sgn, dx / r * sgn
        tx, ty = tx - inward * dx / r, ty - inward * dy / r
        fall = np.exp(-((r / (radius * 2.2)) ** 2)) * np.clip(r / (radius * 0.35), 0, 1)
        return self._add(tx, ty, strength * fall)

    def along_sdf(self, sdf, weight=1.0, reach: float = 40.0, inside: bool = True):
        """Flow tangent to an object's contour (strokes wrap around the form)."""
        g = ndimage.gaussian_filter(sdf, 1.5)
        gy, gx = np.gradient(g)
        n = np.hypot(gx, gy) + 1e-6
        tx, ty = -gy / n, gx / n
        d = np.abs(sdf) if inside else np.clip(sdf, 0, None)
        fall = np.exp(-d / reach)
        if not inside:
            fall = np.where(sdf < 0, 0.0, fall)
        return self._add(tx, ty, weight * fall)

    def radial(self, x: float, y: float, weight=1.0, radius: float = 200.0):
        """Rays out from a point (sun bursts)."""
        X, Y = self.frame.grid
        dx, dy = X - x, Y - y
        r = np.hypot(dx, dy) + 1e-3
        return self._add(dx / r, dy / r, weight * np.exp(-r / (radius * 3)))

    def build(self) -> FlowField:
        if self._n == 0:
            self.uniform(0.0)
        # Light smoothing removes seams where contributions meet.
        vx = ndimage.gaussian_filter(self.vx, 2.0)
        vy = ndimage.gaussian_filter(self.vy, 2.0)
        return FlowField(vx, vy)
