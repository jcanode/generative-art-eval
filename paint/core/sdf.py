"""Signed distance functions in pixel space (negative inside).

All functions take coordinate arrays ``X, Y`` (any matching shape) and return a
float32 array. Distances are exact or close to exact near the boundary, which is
all that anti-aliasing and edge effects need.
"""
from __future__ import annotations

import numpy as np


def circle(X, Y, cx, cy, r):
    return np.hypot(X - cx, Y - cy) - r


def ellipse(X, Y, cx, cy, rx, ry):
    """Approximate ellipse SDF (gradient-normalised implicit), good near the edge."""
    dx, dy = (X - cx) / rx, (Y - cy) / ry
    k0 = np.hypot(dx, dy)
    k1 = np.hypot(dx / rx, dy / ry)
    return (k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)).astype(np.float32)


def box(X, Y, cx, cy, hw, hh, radius=0.0):
    qx = np.abs(X - cx) - (hw - radius)
    qy = np.abs(Y - cy) - (hh - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - radius


def segment(X, Y, ax, ay, bx, by, r):
    """Capsule: distance to segment AB minus radius (r may be an array)."""
    pax, pay = X - ax, Y - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay
    h = np.clip((pax * bax + pay * bay) / (denom if denom > 0 else 1.0), 0.0, 1.0)
    return np.hypot(pax - bax * h, pay - bay * h) - r


def tapered_segment(X, Y, ax, ay, bx, by, ra, rb):
    """Capsule whose radius interpolates linearly from ra at A to rb at B."""
    pax, pay = X - ax, Y - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay
    h = np.clip((pax * bax + pay * bay) / (denom if denom > 0 else 1.0), 0.0, 1.0)
    return np.hypot(pax - bax * h, pay - bay * h) - (ra + (rb - ra) * h)


def polygon(X, Y, pts):
    """Exact SDF of a simple polygon. Loops over vertices, never pixels."""
    pts = np.asarray(pts, dtype=np.float32)
    d = np.full(np.shape(X), np.inf, dtype=np.float32)
    inside = np.zeros(np.shape(X), dtype=bool)
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        denom = ex * ex + ey * ey
        h = np.clip((wx * ex + wy * ey) / (denom if denom > 0 else 1.0), 0.0, 1.0)
        d = np.minimum(d, np.hypot(wx - ex * h, wy - ey * h))
        # Even-odd crossing test.
        cond = (ay > Y) != (by > Y)
        xint = ax + (Y - ay) * (bx - ax) / ((by - ay) if by != ay else 1e-9)
        inside ^= cond & (X < xint)
    return np.where(inside, -d, d).astype(np.float32)


def union(*ds):
    out = ds[0]
    for d in ds[1:]:
        out = np.minimum(out, d)
    return out


def intersect(a, b):
    return np.maximum(a, b)


def subtract(a, b):
    """a minus b."""
    return np.maximum(a, -b)


def smooth_union(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def halfplane(X, Y, px, py, nx, ny):
    """Distance to the line through P with outward normal N (inside = behind the normal)."""
    n = np.hypot(nx, ny)
    return ((X - px) * nx + (Y - py) * ny) / n


def band_y(X, Y, y0, y1=None):
    """Horizontal band: inside where y0 <= Y (<= y1)."""
    d = y0 - Y
    if y1 is not None:
        d = np.maximum(d, Y - y1)
    return d.astype(np.float32)
