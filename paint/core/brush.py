"""Brush-stroke engine: tapering, pressure, bristles, and ink that runs dry.

A stroke is a polyline path plus a ``Brush``. Rendering maps every pixel in the
stroke's bounding box to stroke-local coordinates

    t in [0, 1]   distance along the stroke
    s in [-1, 1]  signed offset across the stroke, relative to local half-width

by finding the nearest segment of a variable-radius capsule chain. Bristle
behaviour is precomputed as a small (t, s) look-up table per stroke - each
bristle has an offset, a thickness, and an ink load that depletes along ``t`` -
and sampled at every pixel. So per-pixel work is pure numpy, and cost scales
with the stroke's bounding box, never with a Python loop over pixels.

``Footprint`` is what a style composites: an alpha window plus the (t, s)
coordinates so the style can vary colour or paint height along the stroke.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from .rng import Rng


@dataclass(frozen=True)
class Brush:
    width: float = 12.0          # full width in px at full pressure
    bristles: int = 18           # number of bristle clumps
    bristle_spread: float = 0.35 # random jitter of bristle offsets (fraction of spacing)
    taper_start: float = 0.12    # fraction of length over which the head swells in
    taper_end: float = 0.35      # fraction of length over which the tail thins out
    tip: float = 0.15            # width fraction remaining at the very ends
    load: float = 1.0            # ink/paint load at stroke start, 0-1
    dry_rate: float = 0.5        # load lost by the end of the stroke
    dry_texture: float = 0.6     # how strongly a dry brush breaks up on the grain
    opacity: float = 1.0
    wobble: float = 0.04         # bristle lateral waviness (fraction of width)
    edge: float = 1.0            # edge softness in px

    def with_(self, **kw) -> "Brush":
        return replace(self, **kw)


@dataclass
class Footprint:
    y0: int
    x0: int
    alpha: np.ndarray   # (h, w) coverage * ink, in [0, 1]
    t: np.ndarray       # (h, w) along-stroke parameter
    s: np.ndarray       # (h, w) across-stroke offset, -1..1 inside
    cover: np.ndarray   # (h, w) geometric coverage (no bristle/dry effects)

    @property
    def window(self):
        h, w = self.alpha.shape
        return slice(self.y0, self.y0 + h), slice(self.x0, self.x0 + w)


def resample(path, spacing: float, max_points: int = 64):
    """Resample a polyline to roughly uniform arclength spacing."""
    path = np.asarray(path, dtype=np.float32)
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-3:
        return path[:1].repeat(2, axis=0) + np.array([[0, 0], [0.5, 0]], dtype=np.float32)
    n = int(np.clip(np.ceil(total / max(spacing, 0.5)) + 1, 2, max_points))
    u = np.linspace(0.0, total, n)
    return np.stack([np.interp(u, cum, path[:, 0]), np.interp(u, cum, path[:, 1])], axis=1).astype(np.float32)


def bezier(p0, p1, p2, n: int = 16, p3=None):
    """Quadratic (or cubic with p3) Bezier sampled at n points."""
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, dtype=np.float32) for p in (p0, p1, p2))
    if p3 is None:
        return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t**2) * p2
    p3 = np.asarray(p3, dtype=np.float32)
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t**2) * p2 + (t**3) * p3


def width_profile(t, brush: Brush, pressure=None):
    """Half-width multiplier along t: swell in, hold, taper out, times pressure."""
    t = np.asarray(t, dtype=np.float32)
    head = np.clip(t / max(brush.taper_start, 1e-3), 0, 1)
    tail = np.clip((1 - t) / max(brush.taper_end, 1e-3), 0, 1)
    head = brush.tip + (1 - brush.tip) * np.sin(head * np.pi / 2)
    tail = brush.tip + (1 - brush.tip) * np.sin(tail * np.pi / 2) ** 0.8
    w = np.minimum(head, tail)
    if pressure is not None:
        pr = np.asarray(pressure, dtype=np.float32)
        if pr.ndim == 0:
            w = w * pr
        else:
            w = w * np.interp(t, np.linspace(0, 1, len(pr)), pr)
    return w


def bristle_lut(brush: Brush, rng: Rng, nt: int = 48, ns: int = 40):
    """(nt, ns) table: ink deposited at (t, s) by the bristle bundle."""
    t = np.linspace(0, 1, nt, dtype=np.float32)[:, None]
    s = np.linspace(-1, 1, ns, dtype=np.float32)[None, :]
    nb = max(1, brush.bristles)
    base = np.linspace(-0.92, 0.92, nb) if nb > 1 else np.zeros(1)
    spacing = 1.84 / max(nb - 1, 1)
    offs = base + rng.normal(0, brush.bristle_spread * spacing, nb)
    sig = spacing * rng.uniform(0.55, 1.0, nb) + 0.02
    load = np.clip(brush.load * rng.uniform(0.75, 1.1, nb), 0, 1.2)
    rate = brush.dry_rate * rng.uniform(0.6, 1.5, nb)
    # Outer bristles carry less ink and dry first.
    edge_pen = 1.0 - 0.35 * np.abs(base) ** 2
    phase = rng.uniform(0, 2 * np.pi, nb)
    freq = rng.uniform(1.0, 3.0, nb)
    lut = np.zeros((nt, ns), dtype=np.float32)
    for i in range(nb):  # loop over bristles (tens), not pixels
        o = offs[i] + brush.wobble * np.sin(2 * np.pi * freq[i] * t + phase[i])
        amount = np.clip((load[i] * edge_pen[i] - rate[i] * t) * 2.2, 0, 1)
        lut += amount * np.exp(-(((s - o) / sig[i]) ** 2))
    norm = np.percentile(lut[: max(2, nt // 6)], 70) if lut.max() > 0 else 1.0
    return np.clip(lut / max(norm, 1e-3), 0, 1)


def render_stroke(path, brush: Brush, rng: Rng, shape, pressure=None,
                  grain: np.ndarray | None = None) -> Footprint | None:
    """Rasterise one stroke into a Footprint (or None if it is off-canvas).

    ``grain`` is an optional full-canvas texture in [0, 1] (paper fibres or
    canvas weave); a dry brush skips over its low points.
    """
    H, W = shape
    pts = resample(path, spacing=max(brush.width * 0.35, 1.5))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(max(cum[-1], 1e-3))
    tk = cum / total
    hw = 0.5 * brush.width * width_profile(tk, brush, pressure)
    pad = float(hw.max()) + 2.0 + brush.edge
    x0 = int(max(0, np.floor(pts[:, 0].min() - pad)))
    x1 = int(min(W, np.ceil(pts[:, 0].max() + pad)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - pad)))
    y1 = int(min(H, np.ceil(pts[:, 1].max() + pad)))
    if x1 <= x0 or y1 <= y0:
        return None

    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    X += 0.5
    Y += 0.5
    A, B = pts[:-1], pts[1:]
    ex = (B[:, 0] - A[:, 0])[:, None, None]
    ey = (B[:, 1] - A[:, 1])[:, None, None]
    wx = X[None] - A[:, 0][:, None, None]
    wy = Y[None] - A[:, 1][:, None, None]
    L2 = np.maximum(ex * ex + ey * ey, 1e-6)
    h = np.clip((wx * ex + wy * ey) / L2, 0, 1)
    dx, dy = wx - ex * h, wy - ey * h
    dist = np.hypot(dx, dy)
    r = hw[:-1, None, None] + (hw[1:] - hw[:-1])[:, None, None] * h
    sd = dist - r
    j = np.argmin(sd, axis=0)
    take = lambda a: np.take_along_axis(a, j[None], axis=0)[0]  # noqa: E731
    sd_j, h_j, r_j, d_j = take(sd), take(h), take(r), take(dist)
    # Across-stroke offset = signed perpendicular distance to the segment's LINE. Using the
    # (clamped) point distance instead would bend bristle streaks into rings around the caps.
    seg_len = np.sqrt(take(L2))
    perp = take(ex * wy - ey * wx) / np.maximum(seg_len, 1e-6)
    t = (cum[:-1][j] + h_j * seg[j]) / total
    s = np.clip(perp / np.maximum(r_j, 1e-3), -1.2, 1.2)

    cover = np.clip(0.5 - sd_j / brush.edge, 0, 1).astype(np.float32)
    if not cover.any():
        return None

    lut = bristle_lut(brush, rng)
    nt, ns = lut.shape
    ink = ndimage.map_coordinates(
        lut, [t * (nt - 1), (np.clip(s, -1, 1) + 1) * 0.5 * (ns - 1)], order=1, mode="nearest"
    )
    if grain is not None and brush.dry_texture > 0:
        g = grain[y0:y1, x0:x1]
        # Where the brush is low on ink, only the high points of the grain take ink.
        need = (1.0 - ink) * brush.dry_texture
        take_ink = np.clip((g - need * 1.1 + 0.25) * 4.0, 0, 1)
        ink = ink * (1 - brush.dry_texture + brush.dry_texture * take_ink)
    alpha = np.clip(cover * ink * brush.opacity, 0, 1).astype(np.float32)
    return Footprint(y0, x0, alpha, t.astype(np.float32), s.astype(np.float32), cover)


def composite_color(canvas, fp: Footprint, color, alpha_scale: float = 1.0):
    """Paint an RGB colour (or (h,w,3) colour window) through a footprint, normal blend."""
    ys, xs = fp.window
    a = (fp.alpha * alpha_scale)[..., None]
    canvas[ys, xs] = canvas[ys, xs] * (1 - a) + np.asarray(color, dtype=np.float32) * a


def composite_ink(density, fp: Footprint, amount: float = 1.0):
    """Accumulate ink density like layered transparent washes: 1-(1-d)(1-a)."""
    ys, xs = fp.window
    density[ys, xs] = 1 - (1 - density[ys, xs]) * (1 - fp.alpha * amount)
