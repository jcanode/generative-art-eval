"""Ink-wash primitives: diluted-ink washes, wet bleeding, pooled edges, stroke helpers.

Model: ink is a single *density* field in [0, 1] per layer. Washes and wet
strokes are deposited into a scratch layer; ``bleed`` then lets the wet layer
creep into the paper along its fibres before the layer is merged, so the edge
of a wash is feathered and fibrous rather than a clean vector edge.

``pooled_edge`` is the "coffee ring": as a puddle of diluted ink dries, pigment
migrates to its rim, so the rim ends up darker than the middle.

Geometry helpers turn the shared SDF parts into brush paths: row/column
extents (scanlines clipped to a mask), top profiles (ridges), and a ridge
tracer that walks along a thin part (stems, reeds, leaves).

All per-pixel work is numpy; Python loops only run over strokes or path steps.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .brush import Brush, Footprint, render_stroke
from .rng import Rng


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def bbox_of(arr, thresh: float = 1e-3, pad: int = 0):
    """(y0, y1, x0, x1) slice bounds of ``arr > thresh`` padded and clipped, or None."""
    rows = np.nonzero((arr > thresh).any(axis=1))[0]
    if len(rows) == 0:
        return None
    cols = np.nonzero((arr[rows[0]: rows[-1] + 1] > thresh).any(axis=0))[0]
    H, W = arr.shape
    return (max(0, int(rows[0]) - pad), min(H, int(rows[-1]) + 1 + pad),
            max(0, int(cols[0]) - pad), min(W, int(cols[-1]) + 1 + pad))


def over(layer, d):
    """Accumulate density like layered transparent ink: 1-(1-a)(1-b). In place."""
    layer[...] = 1.0 - (1.0 - layer) * (1.0 - np.clip(d, 0, 1))
    return layer


# ---------------------------------------------------------------------------
# Wash physics
# ---------------------------------------------------------------------------

def pooled_edge(mask, width: float):
    """Rim term in [0, ~1]: high just inside the edge of a (soft) mask, 0 in the middle and outside."""
    blur = ndimage.gaussian_filter(mask, max(width, 0.5), mode="nearest")
    return np.clip((mask - blur) * 2.0, 0.0, 1.0) * mask


def deposit_wash(layer, mask, amount: float, fade=None, ring: float = 0.6, ring_width: float = 5.0,
                 granulation: float = 0.3, mottle=None):
    """Deposit a diluted-ink wash of ``mask`` into ``layer`` (in place), windowed to the mask.

    ``fade`` (full-canvas or None) multiplies density (mist, gradients) but does not
    create rims; rims come only from the mask's own silhouette.
    """
    win = bbox_of(mask, 1e-3, pad=int(3 * ring_width) + 2)
    if win is None:
        return layer
    y0, y1, x0, x1 = win
    m = mask[y0:y1, x0:x1]
    d = m * (1.0 + ring * 1.6 * pooled_edge(m, ring_width))
    if mottle is not None and granulation > 0:
        # Pigment settles unevenly: low-frequency blotches plus fibre granulation.
        d = d * (1.0 - granulation * (mottle[y0:y1, x0:x1] - 0.5) * 1.6)
    if fade is not None:
        d = d * fade[y0:y1, x0:x1]
    over(layer[y0:y1, x0:x1], d * amount)
    return layer


def bleed(layer, fibres, sigma: float, reach: float = 0.8):
    """Let a wet layer creep into the paper along its fibres (in place, windowed).

    Feathered spread = a blurred copy of the ink gated by the fibre field: ink
    wicks further where fibres are dense, leaving a hairy, irregular edge.
    """
    if sigma <= 0.3:
        return layer
    win = bbox_of(layer, 2e-3, pad=int(4 * sigma) + 2)
    if win is None:
        return layer
    y0, y1, x0, x1 = win
    w = layer[y0:y1, x0:x1]
    b = ndimage.gaussian_filter(w, sigma, mode="nearest")
    fib = fibres[y0:y1, x0:x1]
    gate = np.clip(0.2 + 1.6 * (fib - 0.35), 0.0, 1.25)
    feather = b * gate * reach
    # Slight softening of the wet body itself, then the feathers.
    soft = ndimage.gaussian_filter(w, max(0.6, sigma * 0.2), mode="nearest")
    layer[y0:y1, x0:x1] = np.maximum(soft, feather)
    return layer


# ---------------------------------------------------------------------------
# Strokes
# ---------------------------------------------------------------------------

def arclength(path):
    p = np.asarray(path, np.float32)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def split_path(path, max_len: float, overlap: float = 0.06):
    """Split a polyline into consecutive pieces of at most ``max_len`` px.

    Returns [(piece, t0, t1)] where t0/t1 are the piece's span of the full path's
    arclength parameter. Keeps each stroke's bounding box (and so the stroke
    engine's memory) small.
    """
    p = np.asarray(path, np.float32)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= max_len or total < 1e-3:
        return [(p, 0.0, 1.0)]
    n = int(np.ceil(total / max_len))
    out = []
    for i in range(n):
        a = max(0.0, (i / n - overlap * (i > 0)) * total)
        b = min(total, ((i + 1) / n) * total)
        u = np.linspace(a, b, max(3, int((b - a) / 4) + 2))
        piece = np.stack([np.interp(u, cum, p[:, 0]), np.interp(u, cum, p[:, 1])], axis=1).astype(np.float32)
        out.append((piece, a / total, b / total))
    return out


def wobble_path(path, amp: float, rng: Rng):
    """Hand tremor: a smooth random lateral offset along the path."""
    p = np.asarray(path, np.float32)
    if amp <= 0 or len(p) < 3:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = cum / max(cum[-1], 1e-3)
    tang = np.gradient(p, axis=0)
    n = np.hypot(tang[:, 0], tang[:, 1])[:, None] + 1e-6
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / n
    ph = rng.uniform(0, 2 * np.pi, 2)
    fr = rng.uniform(0.6, 2.2, 2)
    off = (np.sin(2 * np.pi * fr[0] * t + ph[0]) + 0.5 * np.sin(2 * np.pi * fr[1] * t + ph[1])) / 1.5
    off = off * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.5  # ends stay put
    return (p + nrm * (off * amp)[:, None]).astype(np.float32)


def ink_profile(fp: Footprint, dry: float, t0: float = 0.0, t1: float = 1.0, streak: float = 0.05,
                grain=None):
    """Re-shade a footprint the ink-painting way.

    The engine's alpha is a bristle pattern everywhere; for sumi-e a loaded
    brush lays a nearly solid body and only breaks into bristle streaks and
    flying white as it runs dry. ``d(t)`` rises along the stroke with ``dry``;
    the result blends solid coverage (wet) into the engine's pattern (dry).
    """
    cover = fp.cover
    pat = np.where(cover > 1e-3, fp.alpha / np.maximum(cover, 1e-3), 0.0)
    tg = t0 + fp.t * (t1 - t0)
    # Outer bristles give out first, so the edges of a stroke break up before its core.
    edge = np.clip(np.abs(fp.s), 0, 1) ** 2
    d = np.clip(dry * 1.25 * tg * (1.0 + 0.7 * edge) - 0.15, 0.0, 1.0) ** 1.2
    solid = (1.0 - streak) + streak * pat
    if grain is not None:
        g = grain[fp.y0: fp.y0 + cover.shape[0], fp.x0: fp.x0 + cover.shape[1]]
        solid = solid * (0.9 + 0.1 * g)
    return (cover * ((1.0 - d) * solid + d * pat)).astype(np.float32)


def _needs_split(path, width, budget=3.0e6):
    p = np.asarray(path, np.float32)
    span = p.max(axis=0) - p.min(axis=0) + width + 6
    npts = min(64, max(2, arclength(p) / max(width * 0.35, 1.5)))
    return float(span[0] * span[1]) * npts > budget


def stroke_into(layer, path, brush: Brush, rng: Rng, pressure=None, grain=None,
                amount: float = 1.0, max_len: float = 220.0, streak: float = 0.05, mode: str = "over") -> int:
    """Render a (possibly long) stroke into ``layer``.

    Long paths are split into overlapping sub-strokes that continue the ink
    load (keeps the engine's per-stroke memory small). Pieces are combined with
    ``max`` before compositing so their overlaps do not leave dark knots.
    ``mode="max"`` merges the stroke into the layer wet-in-wet (no darkening
    where it overlaps earlier strokes in the same layer).
    Returns the number of footprints drawn.
    """
    if _needs_split(path, brush.width):
        pieces = split_path(path, max(max_len, brush.width * 3))
    else:
        pieces = [(np.asarray(path, np.float32), 0.0, 1.0)]
    pr = None if pressure is None else np.asarray(pressure, np.float32)
    fps = []
    for i, (piece, t0, t1) in enumerate(pieces):
        b = brush
        p_piece = None
        if pr is not None and pr.ndim > 0:
            u = np.linspace(t0, t1, 8)
            p_piece = np.interp(u, np.linspace(0, 1, len(pr)), pr)
        elif pr is not None:
            p_piece = pr
        if len(pieces) > 1:
            load = max(0.25, brush.load - brush.dry_rate * t0)
            b = brush.with_(load=load, dry_rate=brush.dry_rate * (t1 - t0) * 1.2,
                            taper_start=brush.taper_start if i == 0 else 0.05,
                            taper_end=brush.taper_end if i == len(pieces) - 1 else 0.08,
                            tip=brush.tip if i in (0, len(pieces) - 1) else 0.3)
        fp = render_stroke(piece, b, rng.child(str(i)), layer.shape, p_piece, grain)
        if fp is None:
            continue
        fps.append((fp, ink_profile(fp, brush.dry_rate, t0, t1, streak, grain)))
    if not fps:
        return 0
    y0 = min(fp.y0 for fp, _ in fps)
    x0 = min(fp.x0 for fp, _ in fps)
    y1 = max(fp.y0 + a.shape[0] for fp, a in fps)
    x1 = max(fp.x0 + a.shape[1] for fp, a in fps)
    buf = np.zeros((y1 - y0, x1 - x0), np.float32)
    for fp, a in fps:
        h, w = a.shape
        sl = (slice(fp.y0 - y0, fp.y0 - y0 + h), slice(fp.x0 - x0, fp.x0 - x0 + w))
        buf[sl] = np.maximum(buf[sl], a)
    win = layer[y0:y1, x0:x1]
    if mode == "max":  # wet-in-wet: strokes merge into one pool instead of stacking
        np.maximum(win, np.clip(buf * amount, 0, 1), out=win)
    else:
        win[...] = 1.0 - (1.0 - win) * (1.0 - np.clip(buf * amount, 0, 1))
    return len(fps)


# ---------------------------------------------------------------------------
# Geometry from SDFs
# ---------------------------------------------------------------------------

def sample(arr, xs, ys):
    """Bilinear sample of a pixel-centred field at pixel coords (x, y)."""
    xs = np.atleast_1d(np.asarray(xs, np.float32))
    ys = np.atleast_1d(np.asarray(ys, np.float32))
    H, W = arr.shape
    return ndimage.map_coordinates(arr, [np.clip(ys - 0.5, 0, H - 1), np.clip(xs - 0.5, 0, W - 1)],
                                   order=1, mode="nearest")


def row_extents(sdf, ys):
    """For each y: (x_left, x_right) of the inside of ``sdf`` on that row, NaN if empty."""
    H, W = sdf.shape
    idx = np.clip(np.asarray(ys).astype(int), 0, H - 1)
    inside = sdf[idx] < 0
    has = inside.any(axis=1)
    left = np.argmax(inside, axis=1).astype(np.float32) + 0.5
    right = (W - 1 - np.argmax(inside[:, ::-1], axis=1)).astype(np.float32) + 0.5
    left[~has] = np.nan
    right[~has] = np.nan
    return left, right


def col_extents(sdf, xs):
    """For each x: (y_top, y_bottom) of the inside of ``sdf`` in that column, NaN if empty."""
    H, W = sdf.shape
    idx = np.clip(np.asarray(xs).astype(int), 0, W - 1)
    inside = sdf[:, idx] < 0
    has = inside.any(axis=0)
    top = np.argmax(inside, axis=0).astype(np.float32) + 0.5
    bot = (H - 1 - np.argmax(inside[::-1], axis=0)).astype(np.float32) + 0.5
    top[~has] = np.nan
    bot[~has] = np.nan
    return top, bot


def row_runs(sdf_row):
    """Inside runs of one row as [(x0, x1)] in pixel coordinates."""
    inside = np.concatenate([[False], sdf_row < 0, [False]])
    d = np.diff(inside.astype(np.int8))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return [(float(a), float(b)) for a, b in zip(starts, ends)]


def trace_ridge(sdf, start, direction, step: float, max_steps: int = 200, search: float | None = None,
                stop=None):
    """Walk along a thin part: step forward, re-centre on the deepest point across.

    ``stop(p) -> bool`` can end the walk early. Returns an (N, 2) path.
    """
    p = np.asarray(start, np.float32)
    d = np.asarray(direction, np.float32)
    d = d / (np.hypot(*d) + 1e-6)
    search = step * 1.2 if search is None else search
    offs = np.linspace(-search, search, 9, dtype=np.float32)
    pts = [p.copy()]
    misses = 0
    for _ in range(max_steps):  # loop over path steps of one stroke
        q = p + d * step
        n = np.array([-d[1], d[0]], np.float32)
        cand = q[None] + offs[:, None] * n[None]
        v = sample(sdf, cand[:, 0], cand[:, 1])
        k = int(np.argmin(v))
        q = cand[k]
        if v[k] > 0.5:
            misses += 1
            if misses > 1:
                break
        else:
            misses = 0
        nd = q - p
        nd = nd / (np.hypot(*nd) + 1e-6)
        d = 0.65 * d + 0.35 * nd
        d = d / (np.hypot(*d) + 1e-6)
        p = q
        pts.append(p.copy())
        if stop is not None and stop(p):
            break
    return np.array(pts, np.float32)


def smooth_path(path, iters: int = 2):
    p = np.asarray(path, np.float32).copy()
    for _ in range(iters):
        if len(p) < 3:
            break
        p[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return p


def arc(cx, cy, rx, ry, a0, a1, n: int = 24):
    """Points on an ellipse from angle a0 to a1 (radians, image coords: +y down)."""
    a = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1).astype(np.float32)


def line(p0, p1, n: int = 8, sag: float = 0.0):
    """Straight segment p0->p1 with an optional perpendicular bow of ``sag`` px."""
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    t = np.linspace(0, 1, n)[:, None]
    pts = p0 + (p1 - p0) * t
    if sag:
        d = p1 - p0
        nrm = np.array([-d[1], d[0]], np.float32) / (np.hypot(*d) + 1e-6)
        pts = pts + nrm * (sag * 4 * t * (1 - t))
    return pts.astype(np.float32)
