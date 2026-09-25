"""Relief-print primitives for the linocut style.

The block is an *ink map* in [0, 1]: 1 where the lino surface is left standing
(it takes ink), 0 where it has been carved away (paper shows). Everything here
edits that map with carving operations:

* ``line_screen`` - rows of carved lines laid along the isolines of a *phase*
  field (one phase unit per line). The cut's width follows a tone field, and
  each row is broken into tapered dashes by a per-row shifted noise field, so
  cuts start and end in points like a V-gouge, and in light tones the black
  ridges left between cuts break into lens-shaped slivers.
* ``gouge`` - batched, individually placed tapered gouge marks (straight or
  bent), used for chips, stipple, leaf flicks, rays and border nicks. They can
  carve (value 0) or leave ink (value 1, for stray ridges).
* ``plate_sdf`` - the irregular edge of a hand-cut plate.

All vectorised; the only Python loops are over batches and marks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .rng import Rng


class WrapNoise:
    """Smooth periodic value noise that can be sampled at arbitrary pixel coordinates."""

    def __init__(self, rng: Rng, cell: float, size: int = 96):
        lat = rng.random((size, size)).astype(np.float32)
        # Pre-filter once for cubic spline sampling with wrap-around.
        self.coef = ndimage.spline_filter(lat, order=3, mode="grid-wrap").astype(np.float32)
        self.cell = float(max(cell, 1.0))
        self.size = size

    def __call__(self, X, Y):
        c = [np.asarray(Y, np.float32) / self.cell, np.asarray(X, np.float32) / self.cell]
        v = ndimage.map_coordinates(self.coef, c, order=3, mode="grid-wrap", prefilter=False)
        # Spline-interpolated uniform noise clusters round 0.5; stretch it back towards [0, 1].
        return np.clip((v - 0.5) * 2.4 + 0.5, 0.0, 1.0).astype(np.float32)


def _hash(r, salt):
    return (np.sin(r * 12.9898 + salt * 78.233) * 43758.5453) % 1.0


def line_screen(phase, tone, X, Y, noise: WrapNoise, px=None, taper: float = 0.7,
                rough=None, salt: float = 0.0):
    """Ink coverage (1 = ink) of a carved line screen.

    phase  line coordinate: one unit per carved line (isolines are the cut centre lines
           at phase = k + 0.5; the ridges left standing sit at integer phase).
    tone   target white (carved) fraction in [0, 1]; 0 = solid black, 1 = cleared.
    px     pixels per phase unit (line spacing); estimated from the phase gradient if None.
    taper  0 = continuous ruled cuts, 1 = strongly broken, pointed dashes.
    rough  px offset field that frays the cut edges.
    """
    phase = np.asarray(phase, np.float32)
    if px is None:
        gy, gx = np.gradient(phase)
        g = np.hypot(gx, gy)
        med = float(np.median(g[g > 1e-4])) if np.any(g > 1e-4) else 0.1
        g = np.clip(g, med * 0.33, med * 3.0)
        px = ndimage.gaussian_filter(1.0 / g, 1.0)
    row = np.floor(phase)
    fr = phase - row
    d = np.abs(fr - 0.5)
    ox = _hash(row, salt) * 997.0
    oy = _hash(row + 0.37, salt + 1.3) * 991.0
    q = noise(X + ox, Y + oy)
    c = 0.38 * taper
    s = np.clip((q - c) / (1.0 - c), 0.0, 1.0) ** 0.75
    s = 1.0 - taper + taper * s  # taper 0: uniform ruled cuts
    # Threshold model: w = clip(s + k). Low tones carve only the crests of s (short
    # pointed cuts); high tones leave black only in its troughs (lens-shaped slivers).
    # k(tone) is calibrated on this call's own distribution of s so mean carve ~= tone.
    samp = s.ravel()[:: max(1, s.size // 4000)]
    ks = np.linspace(-1.05, 1.05, 43, dtype=np.float32)
    means = np.clip(samp[None, :] + ks[:, None], 0, 1).mean(axis=1)
    k = np.interp(np.asarray(tone, np.float32), means, ks).astype(np.float32)
    w = np.clip(s + k, 0.0, 1.0)
    hw = 0.5 * w * px
    cut = hw - d * px + 0.5
    if rough is not None:
        cut = cut + rough * np.minimum(w * 4.0, 1.0)
    cov = np.clip(cut, 0.0, 1.0)
    cov = np.where(w < 0.02, 0.0, cov)
    return (1.0 - cov).astype(np.float32)


def gouge(canvas, xs, ys, angs, lens, wids, value: float = 0.0, curve=None, clip=None,
          rng: Rng | None = None, skew: float = 0.85, point: float = 0.8, wobble: float = 0.15,
          budget: float = 3.0e6) -> int:
    """Stamp tapered gouge marks into ``canvas`` (H, W) in place.

    Each mark is centred at (x, y), runs along ``ang`` for ``len`` px and is ``wid``
    px wide at its widest; the width follows a skewed sine so the mark enters
    fairly bluntly and leaves in a long point. ``curve`` bends the mark (fraction
    of its half-length). ``value`` 0 carves (paper), 1 leaves ink. ``clip`` is an
    optional (H, W) coverage the marks are confined to.
    """
    xs = np.asarray(xs, np.float32).ravel()
    n = len(xs)
    if n == 0:
        return 0
    H, W = canvas.shape
    ys = np.asarray(ys, np.float32).ravel()
    angs = np.broadcast_to(np.asarray(angs, np.float32), (n,))
    lens = np.maximum(np.broadcast_to(np.asarray(lens, np.float32), (n,)), 1.0)
    wids = np.maximum(np.broadcast_to(np.asarray(wids, np.float32), (n,)), 0.3)
    curve = np.zeros(n, np.float32) if curve is None else np.broadcast_to(np.asarray(curve, np.float32), (n,))
    rng = rng or Rng(0, "gouge")
    phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    freq = rng.uniform(4.0, 9.0, n).astype(np.float32)
    reach = lens * 0.5 * (1.0 + np.abs(curve)) + wids + 3.0
    wins = (np.ceil(2 * reach / 4.0) * 4).astype(int)
    pad = int(wins.max()) + 2
    P = np.zeros((H + 2 * pad, W + 2 * pad), np.float32)
    P[pad:pad + H, pad:pad + W] = canvas
    if clip is not None:
        C = np.zeros_like(P)
        C[pad:pad + H, pad:pad + W] = clip
    for win in sorted(set(wins.tolist()), reverse=True):
        idx = np.nonzero(wins == win)[0]
        chunk = max(1, int(budget // (win * win)))
        ar = np.arange(win, dtype=np.float32) + 0.5
        for k0 in range(0, len(idx), chunk):
            ii = idx[k0:k0 + chunk]
            x0 = np.floor(xs[ii] - win / 2).astype(np.int64)
            y0 = np.floor(ys[ii] - win / 2).astype(np.int64)
            dx = (x0[:, None] + ar[None, :] - xs[ii, None])[:, None, :]
            dy = (y0[:, None] + ar[None, :] - ys[ii, None])[:, :, None]
            ca, sa = np.cos(angs[ii])[:, None, None], np.sin(angs[ii])[:, None, None]
            L = (lens[ii] * 0.5)[:, None, None]
            u = (dx * ca + dy * sa) / L
            v = -dx * sa + dy * ca
            v = v - curve[ii, None, None] * L * (u * u - 0.33)
            t = np.clip((u + 1.0) * 0.5, 0.0, 1.0)
            prof = np.sin(np.pi * t ** skew) ** point
            wob = 1.0 + wobble * np.sin(u * freq[ii, None, None] + phase[ii, None, None])
            hw = 0.5 * wids[ii, None, None] * prof * wob
            a = np.clip(hw - np.abs(v) + 0.5, 0.0, 1.0) * (np.abs(u) < 1.0)
            a = np.where(hw < 0.15, 0.0, a)
            ys0, xs0 = y0 + pad, x0 + pad
            for q in range(len(ii)):
                y, x = int(ys0[q]), int(xs0[q])
                if y < 0 or x < 0 or y + win > P.shape[0] or x + win > P.shape[1]:
                    continue
                aq = a[q]
                if clip is not None:
                    aq = aq * C[y:y + win, x:x + win]
                reg = P[y:y + win, x:x + win]
                if value <= 0.0:
                    np.minimum(reg, 1.0 - aq, out=reg)
                else:
                    np.maximum(reg, aq * value, out=reg)
    canvas[:] = P[pad:pad + H, pad:pad + W]
    return n


def scatter(mask, spacing: float, rng: Rng, density=None, offset=(0, 0), jitter: float = 0.9):
    """Jittered-grid points (global px) inside ``mask`` (a crop at ``offset``), kept with prob ``density``."""
    h, w = mask.shape
    sp = max(float(spacing), 1.0)
    gy, gx = np.mgrid[0:h:sp, 0:w:sp]
    gx = gx.ravel().astype(np.float32)
    gy = gy.ravel().astype(np.float32)
    # Stagger alternate rows so the grid never reads as a grid.
    gx = gx + ((gy / sp).astype(int) % 2) * sp * 0.5
    gx = gx + rng.uniform(-0.5, 0.5, gx.size).astype(np.float32) * sp * jitter
    gy = gy + rng.uniform(-0.5, 0.5, gy.size).astype(np.float32) * sp * jitter
    ix = np.clip(gx.astype(int), 0, w - 1)
    iy = np.clip(gy.astype(int), 0, h - 1)
    keep = mask[iy, ix] > 0.5
    if density is not None:
        dv = density[iy, ix] if np.ndim(density) else np.full(gx.shape, float(density))
        keep &= rng.random(gx.size) < dv
    return gx[keep] + offset[0], gy[keep] + offset[1]


def plate_sdf(shape, margin: float, rng: Rng, wobble: float = 2.0, fray: float = 0.8, corner: float = 3.0):
    """Signed distance (px, negative inside) to a hand-cut rectangular plate edge.

    The edges are not quite straight (low-frequency drift), slightly out of
    square, and frayed at the pixel scale.
    """
    from .noise import value_noise

    H, W = shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    sk = rng.uniform(-1, 1, 4) * wobble
    # Per-edge insets vary linearly along the edge (out of square) plus drift.
    left = margin + sk[0] + (Y / H) * sk[1]
    right = W - margin - sk[1] - (Y / H) * sk[0]
    top = margin + sk[2] + (X / W) * sk[3]
    bottom = H - margin - sk[3] - (X / W) * sk[2]
    qx = np.maximum(left - X, X - right)
    qy = np.maximum(top - Y, Y - bottom)
    d = np.where((qx > -corner) & (qy > -corner),
                 np.hypot(np.maximum(qx + corner, 0), np.maximum(qy + corner, 0)) - corner,
                 np.maximum(qx, qy))
    drift = (value_noise(shape, 120, rng.child("drift")) - 0.5) * 2 * wobble
    fr = (value_noise(shape, 3.0, rng.child("fray")) - 0.5) * 2 * fray
    return (d + drift + fr).astype(np.float32)


def local_mean(values, weight, sigma: float):
    """Normalised convolution: weighted local mean of ``values`` (0 where weight vanishes)."""
    num = ndimage.gaussian_filter(values * weight, sigma)
    den = ndimage.gaussian_filter(weight, sigma)
    return num / np.maximum(den, 1e-3)


__all__ = ["WrapNoise", "line_screen", "gouge", "scatter", "plate_sdf", "local_mean"]
