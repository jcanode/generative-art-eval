"""Helpers for the woodblock (moku-hanga manner) style.

Everything here is vectorised numpy; the only Python loops run over blocks,
marks or rows of a pattern, never over pixels.

* ``Window``           - a bbox slice of the canvas, so per-part work stays local.
* ``carved_outline``   - key-block line along an SDF contour, with a width that
                         swells and tapers (noise + light side) like a carved line.
* ``wood_grain``       - grain figure of a cherry block, one per colour block.
* ``baren_texture``    - rubbing-pad texture: short criss-cross streaks, patchy.
* ``wipe_ramp``        - a bokashi ramp whose wiped edge wanders and streaks.
* ``wave_rows``        - perspective rows of scalloped wave crests (+ foam band).
* ``parallel_lines``   - long straight parallel lines (rain), slot-hashed.
* ``stamp_arcs``       - small carved arcs (foliage clumps), windowed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .noise import blurred_white, fbm, value_noise
from .rng import Rng


class Window:
    """Pixel window (y0:y1, x0:x1) around a bbox, clipped to the canvas."""

    def __init__(self, bbox, shape, pad=0):
        H, W = shape
        x0, y0, x1, y1 = bbox
        self.x0, self.y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        self.x1, self.y1 = min(W, int(x1 + pad) + 1), min(H, int(y1 + pad) + 1)

    @property
    def empty(self):
        return self.x1 <= self.x0 or self.y1 <= self.y0

    @property
    def sl(self):
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def of(self, a):
        return a[..., self.y0:self.y1, self.x0:self.x1]


def hash01(i, salt: float = 0.0):
    """Deterministic pseudo-random in [0, 1) for integer arrays (pattern rows / slots)."""
    i = np.asarray(i, dtype=np.float64)
    return ((np.sin(i * 12.9898 + salt * 78.233) * 43758.5453) % 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Key block lines
# ---------------------------------------------------------------------------

def carved_outline(sdf, width, swell, light_dir=None, shadow_gain=0.35, rough=None, offset=0.0):
    """Coverage of a carved line on the contour ``sdf == -offset``.

    ``width`` is the nominal line width in px; ``swell`` (same shape as sdf, in
    [0, 1]) modulates it along the contour so the line tapers and swells.
    The side facing away from the light is cut thicker.
    """
    w = width * (0.55 + 0.9 * swell)
    if light_dir is not None and shadow_gain > 0:
        gy, gx = np.gradient(sdf)
        n = np.hypot(gx, gy) + 1e-6
        away = -(gx * light_dir[0] + gy * light_dir[1]) / n  # +1 where the outward normal faces away from light
        w = w * (1.0 + shadow_gain * np.clip(away, -1, 1))
    d = np.abs(sdf + offset) - w * 0.5
    if rough is not None:
        d = d + rough
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Block surface
# ---------------------------------------------------------------------------

def wood_grain(shape, rng: Rng, sc: float, warp, angle_deg: float = 0.0, period: float = 9.0):
    """Grain figure in [0, 1] (1 = grain line that prints lighter)."""
    H, W = shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.radians(angle_deg)
    v = -X * np.sin(a) + Y * np.cos(a)
    oy, ox = rng.integers(0, H), rng.integers(0, W)
    w = np.roll(warp, (int(oy), int(ox)), axis=(0, 1))
    p = period * sc
    phase = (v + w * p * 7.0) / p + rng.uniform(0, 1)
    lines = np.clip((np.cos(2 * np.pi * phase) - 0.55) * 2.6, 0, 1)
    # Ring width varies (early/late wood): modulate line strength slowly.
    strength = 0.55 + 0.45 * np.roll(warp, (int(ox) % H, int(oy) % W), axis=(0, 1))
    return (lines * strength).astype(np.float32)


def grain_warp(shape, rng: Rng, sc: float):
    """Shared low-frequency warp for all blocks' grain (rolled per block)."""
    return fbm(shape, 180 * sc, rng, octaves=3).astype(np.float32)


def baren_texture(shape, rng: Rng, sc: float):
    """Rubbing texture around 0.5: short horizontal / vertical streaks chosen patchily."""
    h = blurred_white(shape, rng.child("h"), (1.2 * sc, 5.0 * sc))
    v = blurred_white(shape, rng.child("v"), (5.0 * sc, 1.2 * sc))
    sel = value_noise(shape, 30 * sc, rng.child("sel"))
    sel = np.clip((sel - 0.5) * 5 + 0.5, 0, 1)
    mottle = fbm(shape, 60 * sc, rng.child("m"), octaves=3)
    return (h * sel + v * (1 - sel)) * 0.35 + 0.65 * mottle


def wipe_ramp(t, rng: Rng, shape, sc, wander=0.06, streak=0.12, gamma=1.4):
    """Bokashi ramp: 1 at t<=0, 0 at t>=1, with a wandering wiped edge and streaks.

    ``t`` is a normalised coordinate (0 = full ink edge .. 1 = wiped clean).
    """
    H, W = shape
    col = value_noise((1, W), 160 * sc, rng.child("col"))  # the wipe line wanders along x
    t = t + (col - 0.5) * 2 * wander
    r = np.clip(1.0 - t, 0, 1) ** gamma
    st = blurred_white(shape, rng.child("streak"), (0.6 * sc, 30 * sc))
    r = r * (1.0 - streak * (st - 0.5) * 2 * np.clip(r * (1 - r) * 4, 0, 1))
    return np.clip(r, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Stylised nature patterns
# ---------------------------------------------------------------------------

def wave_rows(X, Y, top, bottom, sc, s_near=26.0, s_far=3.5, line_w=1.0, rng: Rng | None = None, wob=None):
    """Rows of scalloped wave crests receding in perspective.

    Returns (line, foam, row, frac, spacing, t) over the given X, Y window.
    ``line`` is key-block coverage of the crest lines, ``foam`` a band just
    under each crest that is carved out of the water block (paper shows).
    """
    s0 = s_far * sc
    s1 = s_near * sc
    dy = np.clip(Y - top, 0, None)
    span = max(bottom - top, 1.0)
    b = (s1 - s0) / span
    q = np.log1p(b * dy / s0) / b
    row = np.floor(q)
    frac = q - row
    sp = s0 + b * dy
    t = np.clip(dy / span, 0, 1)
    p = sp * 3.2
    off = 0.5 * (row % 2) + (hash01(row, 1.7) - 0.5) * 0.3
    xw = X if wob is None else X + wob
    u = (xw / p + off) % 1.0
    arc = np.sqrt(np.clip(1.0 - (2 * u - 1) ** 2, 0, 1))
    curve = 0.78 - 0.5 * arc
    dist = (frac - curve) * sp
    lw = line_w * sc * (0.45 + 1.7 * t) * (0.3 + 0.7 * arc)
    fade = np.clip((sp - 2.5 * sc) / (3.0 * sc), 0.2, 1.0)
    line = np.clip(lw * 0.5 - np.abs(dist) + 0.5, 0, 1) * fade
    fw = sp * 0.16 * arc * np.clip((t - 0.12) / 0.3, 0, 1)
    foam = np.clip(np.minimum(dist - lw * 0.5, lw * 0.5 + fw - dist) + 0.5, 0, 1) * (dist > 0)
    return line.astype(np.float32), foam.astype(np.float32), row, frac, sp, t


def parallel_lines(X, Y, rng: Rng, sc, angle_deg=10.0, spacing=9.0, width=(0.6, 1.3), length=(0.15, 0.6),
                   present=0.75, H=1024):
    """Long straight parallel lines at ``angle_deg`` from vertical (rain), fully vectorised."""
    a = np.radians(angle_deg)
    u = X * np.cos(a) - Y * np.sin(a)
    v = X * np.sin(a) + Y * np.cos(a)
    d = spacing * sc
    umin = float(u.min()) - d
    slot = np.floor((u - umin) / d).astype(np.int64)
    n = int(slot.max()) + 2
    off = rng.uniform(0.15, 0.85, n)
    wid = rng.uniform(width[0], width[1], n) * sc
    ln = rng.uniform(length[0], length[1], n) * H
    vmin, vmax = float(v.min()), float(v.max())
    v0 = rng.uniform(vmin - 0.3 * H, vmax, n)
    on = rng.random(n) < present
    centre = umin + (slot + off[slot]) * d
    w = wid[slot]
    tv = (v - v0[slot]) / ln[slot]
    taper = np.clip(np.minimum(tv, 1 - tv) * 6, 0, 1)
    cov = np.clip(w * taper * 0.5 - np.abs(u - centre) + 0.5, 0, 1) * on[slot] * (tv >= 0) * (tv <= 1)
    return cov.astype(np.float32)


def stamp_arcs(canvas, frame, arcs, width, sc):
    """Carved upper arcs (foliage clumps). ``arcs`` = [(cx, cy, r, a0, a1)] with angles in radians.

    The line tapers toward both arc ends.
    """
    X, Y = frame.grid
    H, W = frame.shape
    for cx, cy, r, a0, a1 in arcs:
        pad = r + width * 2 + 2
        win = Window((cx - r, cy - r, cx + r, cy + r), (H, W), pad)
        if win.empty:
            continue
        Xw, Yw = X[win.sl], Y[win.sl]
        ang = np.arctan2(Yw - cy, Xw - cx)
        mid, half = (a0 + a1) / 2, (a1 - a0) / 2
        da = np.angle(np.exp(1j * (ang - mid)))
        along = 1 - np.clip(np.abs(da) / max(half, 1e-3), 0, 1.2)
        w = width * np.clip(along * 1.6, 0, 1)
        d = np.abs(np.hypot(Xw - cx, Yw - cy) - r) - w * 0.5
        cov = np.clip(0.5 - d, 0, 1) * (np.abs(da) <= half)
        canvas[win.sl] = np.maximum(canvas[win.sl], cov)
    return canvas


def stamp_strokes(canvas, frame, strokes, value=1.0):
    """Carved lens-shaped strokes: ``strokes`` = [(ax, ay, bx, by, w)], thickest mid-way, pointed ends."""
    X, Y = frame.grid
    H, W = frame.shape
    for ax, ay, bx, by, w in strokes:
        win = Window((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)), (H, W), w + 2)
        if win.empty:
            continue
        Xw, Yw = X[win.sl], Y[win.sl]
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey or 1.0
        h = np.clip(((Xw - ax) * ex + (Yw - ay) * ey) / L2, 0, 1)
        d = np.hypot(Xw - ax - ex * h, Yw - ay - ey * h)
        half = 0.5 * w * np.sqrt(np.clip(4 * h * (1 - h), 0, 1))
        cov = np.clip(half - d + 0.5, 0, 1) * value
        canvas[win.sl] = np.maximum(canvas[win.sl], cov)
    return canvas


def contour_strokes(sdf, rng: Rng, n, length, width, min_depth, max_depth, border_ok=None):
    """Short strokes inside a shape running parallel to its nearest edge (rock strata, ridges).

    Returns stroke tuples for ``stamp_strokes``. ``border_ok(x, y, depth)`` can veto
    points whose nearest edge is really the canvas border.
    """
    H, W = sdf.shape
    ys, xs = np.nonzero((sdf < -min_depth) & (sdf > -max_depth))
    if len(xs) == 0:
        return []
    idx = rng.integers(0, len(xs), n * 2)
    out = []
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        depth = float(-sdf[y, x])
        if border_ok is not None and not border_ok(x, y, depth):
            continue
        x0, x1 = max(0, x - 1), min(W - 1, x + 1)
        y0, y1 = max(0, y - 1), min(H - 1, y + 1)
        gx = float(sdf[y, x1] - sdf[y, x0])
        gy = float(sdf[y1, x] - sdf[y0, x])
        g = np.hypot(gx, gy)
        if g < 1e-3:
            continue
        tx, ty = -gy / g, gx / g
        L = length * rng.uniform(0.6, 1.3)
        c = rng.uniform(-0.3, 0.3)
        ax, ay = x - tx * L * (0.5 + c), y - ty * L * (0.5 + c)
        bx, by = x + tx * L * (0.5 - c), y + ty * L * (0.5 - c)
        out.append((ax, ay, bx, by, width * rng.uniform(0.7, 1.2)))
        if len(out) >= n:
            break
    return out


def soft_edges(plate, rng: Rng, sc, sigma=0.7, bleed=0.35):
    """Water-based pigment edge: a slight blur plus fibre-driven bleeding at edges."""
    p = ndimage.gaussian_filter(plate, sigma * sc)
    gy, gx = np.gradient(p)
    edge = np.clip(np.hypot(gx, gy) * 2.5 * sc, 0, 1)  # only true edges bleed, not bokashi ramps
    n = value_noise(plate.shape, 3.0 * sc, rng)
    return np.clip(p + (n - 0.5) * bleed * edge, 0, 1).astype(np.float32)
