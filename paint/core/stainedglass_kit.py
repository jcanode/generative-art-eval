"""Building blocks for the stained-glass style: pane cutting, leading, glass.

A window is modelled as an integer *label map*: every pixel belongs to exactly
one piece of glass (label >= 1) or to the stone surround (label 0). Lead came
is drawn wherever the label changes, so leading is continuous by construction
and follows every cut. Pane cutting is vectorised: each cutter maps pixels to
an integer *key* (course/column, ring/sector, facet/band ...) and keys are
compacted into pane ids afterwards.
"""
from __future__ import annotations

import colorsys

import numpy as np
from scipy import ndimage

from .noise import value_noise
from .rng import Rng

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def hsv(rgb):
    return colorsys.rgb_to_hsv(*[float(x) for x in np.clip(rgb, 0, 1)])


def from_hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1))), np.float32)


def jewel(rgb, pale: bool = False, sat_floor: float = 0.62, boost: float = 1.3):
    """Push a colour to stained-glass jewel saturation (or a tinted 'white' glass)."""
    h, s, v = hsv(rgb)
    if pale:
        return from_hsv(h, np.clip(max(s, 0.1), 0.1, 0.32), max(v, 0.86))
    s = np.clip(max(s * boost, sat_floor), 0, 0.96)
    v = np.clip(0.3 + 0.7 * v, 0.34, 0.93)
    return from_hsv(h, s, v)


def vary(rgb, rng: Rng, dh=0.012, ds=0.05, dv=0.07):
    h, s, v = hsv(rgb)
    return from_hsv(h + rng.normal(0, dh), s + rng.normal(0, ds), v + rng.normal(0, dv))


def lighten(rgb, a):
    rgb = np.asarray(rgb, np.float32)
    return rgb + (1 - rgb) * a


def darken(rgb, a):
    return np.asarray(rgb, np.float32) * (1 - a)


# ---------------------------------------------------------------------------
# Pane cutting
# ---------------------------------------------------------------------------

KEY_BASE = 1 << 12  # course * KEY_BASE + column


def make_bounds(vmin, vmax, first, growth, rng: Rng, jitter=0.2, min_last=0.45):
    """Course boundaries between vmin and vmax: first height ``first``, growing by ``growth``."""
    out, v, hgt = [], vmin, first
    while True:
        step = hgt * (1 + rng.uniform(-jitter, jitter))
        if v + step > vmax - min_last * hgt:
            break
        v += step
        out.append(v)
        hgt *= growth
    return np.array(out, np.float64)


def courses(u, v, bounds, widths, rng: Rng, slant=0.35, wave=0.0, wave_len=600.0):
    """Staggered courses: rows split along ``v`` at ``bounds``; each row is cut into
    pieces of its own width along ``u``, with its own offset and a slant so
    joints never line up into a grid.

    ``wave`` > 0 makes every boundary its own gentle curve (amplitude as a
    fraction of the neighbouring gaps, wavelength around ``wave_len`` in u units),
    so courses swell and thin like ribbons instead of running parallel.
    """
    bounds = np.asarray(bounds, np.float64)
    n = len(bounds) + 1
    if wave > 0 and len(bounds):
        gaps = np.diff(np.concatenate([[bounds[0] - 1e9], bounds, [bounds[-1] + 1e9]]))
        room = np.minimum(gaps[:-1], gaps[1:])
        room = np.where(np.isfinite(room) & (room < 1e8), room, np.max(np.diff(bounds)) if len(bounds) > 1 else 50.0)
        wr = rng.child("wave")
        k = np.zeros(np.shape(v), np.int64)
        for j, bj in enumerate(bounds):
            amp = wave * 0.45 * room[j] * wr.uniform(0.5, 1.0)
            lam = wave_len * wr.uniform(0.6, 1.5)
            ph = wr.uniform(0, 2 * np.pi)
            k += (v > bj + amp * np.sin(2 * np.pi * u / lam + ph)).astype(np.int64)
    else:
        k = np.searchsorted(bounds, v).astype(np.int64)
    edges = np.concatenate([[bounds[0] - (bounds[1] - bounds[0] if len(bounds) > 1 else 50)] if len(bounds) else [0.0],
                            bounds, [bounds[-1] + 50 if len(bounds) else 100.0]])
    mids = 0.5 * (edges[:-1] + edges[1:])
    widths = np.broadcast_to(np.asarray(widths, np.float64), (n,)) if np.ndim(widths) == 0 else np.asarray(widths, np.float64)
    off = rng.uniform(0, 1, n) * widths
    sl = rng.uniform(-slant, slant, n)
    col = np.floor((u + off[k] + sl[k] * (v - mids[k])) / widths[k]).astype(np.int64)
    return k * KEY_BASE + (col + KEY_BASE // 2)


def rings_sectors(X, Y, cx, cy, radii, counts, rng: Rng, twist=0.0):
    """Sunburst: concentric rings (``radii`` boundaries) cut into radial sectors.
    Returns (key, ring index) with ring = -1 outside the outer radius."""
    d = np.hypot(X - cx, Y - cy)
    a = np.arctan2(Y - cy, X - cx)
    ring = np.searchsorted(np.asarray(radii), d).astype(np.int64)
    ring = np.where(ring >= len(radii), -1, ring)
    key = np.zeros(X.shape, np.int64)
    for i, n in enumerate(counts):
        off = rng.uniform(0, 2 * np.pi) + (np.pi / n if i % 2 else 0)
        sec = np.floor(((a + off + twist * d / max(radii[-1], 1)) % (2 * np.pi)) / (2 * np.pi / n)).astype(np.int64)
        key = np.where(ring == i, (i + 1) * KEY_BASE + sec, key)
    return key, ring


def principal_axis(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return None
    step = max(1, len(xs) // 4000)
    xs, ys = xs[::step].astype(np.float64), ys[::step].astype(np.float64)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    w, v = np.linalg.eigh(cov)
    return cx, cy, v[:, 1], float(np.sqrt(12 * max(w[1], 0))), float(np.sqrt(12 * max(w[0], 0)))


def cut_form(sdf, mask, ps, lead_w, light, rng: Rng, facets: float = 1.0, core: bool = True,
             along: bool = True, split_core: bool = True):
    """Contour-following cut of an object part (on a crop).

    * facets: pixels are grouped by the direction of the outward normal of the
      silhouette (the SDF gradient), so cut lines run along the medial axis:
      ridges of mountains, midribs of leaves, the centre line of a trunk.
    * rim / core: a band that follows the outline, and a core split in two by
      a line across the light direction (a lit piece and a shadow piece).
    * along: long shapes are also cut across their length.
    """
    key = np.zeros(sdf.shape, np.int64)
    if not mask.any():
        return key
    d = -sdf
    dmax = float(d[mask].max())
    area = float(mask.sum())
    if dmax < 2.6 * lead_w or area < (0.55 * ps) ** 2:
        return key
    er = ndimage.binary_erosion(mask)
    perim = float((mask & ~er).sum())
    nb = int(np.clip(round(perim / (1.25 * ps) * facets), 1, 12))
    gy, gx = np.gradient(sdf.astype(np.float32))
    ang = np.arctan2(gy, gx)
    if nb > 1:
        off = rng.uniform(0, 2 * np.pi)
        key += np.floor(((ang + off) % (2 * np.pi)) / (2 * np.pi / nb)).astype(np.int64)
    if core and dmax > 3.0 * lead_w and dmax > 0.3 * ps:
        rw = np.clip(0.5 * dmax, 1.8 * lead_w, 0.6 * ps)
        inner = d > rw
        ckey = 100
        if split_core and dmax - rw > 0.33 * ps:
            pa = principal_axis(mask)
            if pa is not None:
                cx, cy = pa[0], pa[1]
                H, W = sdf.shape
                Yc, Xc = np.mgrid[0:H, 0:W]
                lx, ly = light
                ckey = 100 + ((Xc - cx) * lx + (Yc - cy) * ly > 0).astype(np.int64)
        key = np.where(inner, ckey, key)
    if along:
        pa = principal_axis(mask)
        if pa is not None:
            cx, cy, ax, L, Wd = pa
            if L > 2.3 * ps and L > 2.2 * Wd:
                H, W = sdf.shape
                Yc, Xc = np.mgrid[0:H, 0:W]
                proj = (Xc - cx) * ax[0] + (Yc - cy) * ax[1]
                nseg = max(2, int(round(L / (1.25 * ps))))
                seg_len = L / nseg
                perp = (Xc - cx) * (-ax[1]) + (Yc - cy) * ax[0]
                sl = rng.uniform(-0.3, 0.3)
                seg = np.floor((proj + L / 2 + sl * perp) / seg_len + rng.uniform(-0.2, 0.2)).astype(np.int64)
                key = key + 1000 * (seg + 50)
    return key


def compact(key, mask):
    """Map arbitrary integer keys inside ``mask`` to 0..n-1. Returns (ids, unique keys)."""
    ids = np.zeros(key.shape, np.int64)
    if not mask.any():
        return ids, np.zeros(0, np.int64)
    uk, inv = np.unique(key[mask], return_inverse=True)
    ids[mask] = inv
    return ids, uk


# ---------------------------------------------------------------------------
# Glazier's clean-up: no hairline slivers
# ---------------------------------------------------------------------------


def edges_of(labels):
    """Pixels whose right or lower neighbour has a different label."""
    e = np.zeros(labels.shape, bool)
    e[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    e[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return e


def merge_slivers(labels, pane_group, min_area, min_half):
    """Merge pieces that are too small or too thin into a neighbouring piece of
    the same group (same object part), the way a glazier would re-draw the cut.

    ``pane_group[label]`` gives each pane's group. Returns the new label map.
    """
    e = edges_of(labels)
    comps, n = ndimage.label(~e)
    if n == 0:
        return labels
    idx = np.arange(1, n + 1)
    area = np.bincount(comps.ravel(), minlength=n + 1)[1:]
    half = ndimage.distance_transform_edt(~e)
    thick = ndimage.maximum(half, comps, idx)
    thick = np.asarray(thick, np.float64)
    # Representative label for each component.
    first = ndimage.minimum(np.arange(labels.size).reshape(labels.shape), comps, idx)
    comp_label = labels.ravel()[np.asarray(first, np.int64)]
    small = (area < min_area) | (thick < min_half)
    if not small.any():
        return labels
    comp_group = pane_group[comp_label]
    out = labels.copy()
    small_full = np.concatenate([[False], small])
    group_full = np.concatenate([[-1], comp_group])
    for g in np.unique(comp_group[small]):
        big_here = (~small_full) & (group_full == g)
        if not big_here[1:].any():
            continue
        gpix = pane_group[labels] == g
        seeds = big_here[comps] & gpix
        ys, xs = np.nonzero(gpix)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub_seed = seeds[y0:y1, x0:x1]
        if not sub_seed.any():
            continue
        _, (iy, ix) = ndimage.distance_transform_edt(~sub_seed, return_indices=True)
        target = labels[y0:y1, x0:x1][iy, ix]
        fix = (small_full[comps[y0:y1, x0:x1]] | (comps[y0:y1, x0:x1] == 0)) & gpix[y0:y1, x0:x1]
        blk = out[y0:y1, x0:x1]
        blk[fix] = target[fix]
    return out


# ---------------------------------------------------------------------------
# Glass surface
# ---------------------------------------------------------------------------


def streak_fields(shape, rng: Rng, scale: float):
    """Four streaky-glass noise fields (horizontal, vertical, two diagonals), each in ~[0, 1]."""
    H, W = shape
    s_long, s_short = 34 * scale, 2.2 * scale

    def norm(a):
        a = a - a.mean()
        return np.clip(0.5 + a / (4 * (a.std() + 1e-6)), 0, 1).astype(np.float32)

    hz = norm(ndimage.gaussian_filter(rng.child("h").normal(0, 1, (H + W, W)).astype(np.float32), (s_short, s_long)))
    vt = norm(ndimage.gaussian_filter(rng.child("v").normal(0, 1, (H, W)).astype(np.float32), (s_long, s_short)))
    yy, xx = np.mgrid[0:H, 0:W]
    d1 = hz[(yy + (xx * 0.8).astype(np.int64)) % (H + W), xx]
    d2 = hz[(yy + ((W - 1 - xx) * 0.8).astype(np.int64)) % (H + W), xx]
    return np.stack([hz[:H], vt, d1, d2])


def seeds_field(shape, rng: Rng, scale: float, density: float):
    """Seedy glass: tiny bubbles as (highlight, rim) fields."""
    pts = (rng.random(shape) < density).astype(np.float32)
    big = (rng.child("big").random(shape) < density * 0.25).astype(np.float32)
    core = ndimage.gaussian_filter(pts, 0.8 * scale) + ndimage.gaussian_filter(big, 1.6 * scale) * 2.5
    ring = ndimage.gaussian_filter(pts, 1.7 * scale) + ndimage.gaussian_filter(big, 3.0 * scale) * 2.5
    k = 1.0 / (ndimage.gaussian_filter(np.pad(np.ones((1, 1), np.float32), 8), 0.8 * scale).max() + 1e-6)
    return np.clip(core * k, 0, 1.5), np.clip(ring * k * 0.6, 0, 1)


def wobble(shape, rng: Rng, cell, amp):
    return (value_noise(shape, cell, rng) - 0.5) * 2 * amp


# ---------------------------------------------------------------------------
# Local SDF drawing (on a window, for small marks)
# ---------------------------------------------------------------------------


def window(shape, x0, y0, x1, y1, pad=2):
    H, W = shape
    xa, ya = max(0, int(np.floor(x0)) - pad), max(0, int(np.floor(y0)) - pad)
    xb, yb = min(W, int(np.ceil(x1)) + pad), min(H, int(np.ceil(y1)) + pad)
    if xb <= xa or yb <= ya:
        return None
    Y, X = np.mgrid[ya:yb, xa:xb].astype(np.float32) + 0.5
    return (slice(ya, yb), slice(xa, xb)), X, Y
