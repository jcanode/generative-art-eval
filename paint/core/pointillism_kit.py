"""Divisionist dot engine: pigment mixtures, dot sprites, batched splatting.

Three pieces, all vectorised:

* ``PigmentMixer`` - a fixed palette of *unmixed* pigments. For any target
  colour it solves a small non-negative least-squares problem (weights sum to
  one, a hue-aware ridge spreads the weight over neighbouring hues) and caches
  the result per 5-bit colour bin. Each dot then *samples* one pigment from its
  bin's weights, so an area's dots average to the target colour in the eye
  while every single dot stays a pure colour.
* ``DotBank`` - pre-rendered dab sprites (slightly irregular round / oval
  touches with a soft edge and a little paint texture) in a few size bins that
  share one window size.
* ``splat`` - composites a batch of dots onto a canvas in one go with
  ``np.bincount`` (coverage and colour are accumulated, overlaps inside a
  batch are normalised), so tens of thousands of dots cost a few array ops.
  Successive batches composite "over" each other like successive sessions of
  dotting.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .color import hex_to_rgb
from .paintbody import rgb_to_hsv
from .rng import Rng

# A divisionist palette: pure pigments, their white tints and deep shades. No
# earths, no black.
PIGMENTS = {
    "white": "#f7f2e4",
    "pale_yellow": "#f7eaa6",
    "lemon": "#f4dc3c",
    "chrome": "#f0ac1c",
    "orange": "#ec7424",
    "vermilion": "#d8382a",
    "rose": "#d44a78",
    "pale_rose": "#f2c0c4",
    "violet": "#7446a8",
    "pale_violet": "#c8b8e6",
    "ultramarine": "#2c3ea8",
    "cobalt": "#2f6cc8",
    "cerulean": "#4aa8dc",
    "pale_blue": "#b4d8f0",
    "emerald": "#2eaa5a",
    "yellow_green": "#9cc83a",
    "viridian": "#1c7a5e",
    "pale_green": "#c4e4b0",
    "deep_blue": "#1c2462",
    "deep_violet": "#3c2258",
    "deep_green": "#16452e",
    "deep_red": "#74202a",
}


class PigmentMixer:
    def __init__(self, names=None, division: float = 0.02, hue_bias: float = 6.0, bits: int = 5):
        names = list(names or PIGMENTS)
        self.names = names
        self.rgb = np.stack([hex_to_rgb(PIGMENTS[n]) for n in names]).astype(np.float32)  # (K, 3)
        self.hsv = rgb_to_hsv(self.rgb)
        self.lum = self.rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        self.K = len(names)
        self.division = float(division)
        self.hue_bias = float(hue_bias)
        self.bits = bits
        self._cache: dict[int, np.ndarray] = {}
        self.chroma = self.hsv[:, 1] > 0.3  # pigments with a clear hue

    # -- mixtures ----------------------------------------------------------
    def _solve(self, c):
        K = self.K
        lw = np.array([0.2126, 0.7152, 0.0722], np.float32)
        d2 = ((self.rgb - c) ** 2).sum(1)
        lam = self.division * (1.0 + self.hue_bias * d2)
        M = 30.0
        A = np.vstack([self.rgb.T, 2.0 * lw @ self.rgb.T, np.diag(np.sqrt(lam)), M * np.ones((1, K))])
        b = np.concatenate([c, [2.0 * float(lw @ c)], np.zeros(K), [M]])
        w, _ = nnls(A.astype(np.float64), b.astype(np.float64), maxiter=200)
        w = np.maximum(w, 0)
        w[w < 0.03] = 0.0  # no stray single dots of an unrelated hue
        s = w.sum()
        return (w / s if s > 0 else np.ones(K) / K).astype(np.float32)

    def bins(self, colors):
        q = 2 ** self.bits - 1
        i = np.clip((np.asarray(colors) * q + 0.5).astype(np.int32), 0, q)
        return (i[..., 0] << (2 * self.bits)) | (i[..., 1] << self.bits) | i[..., 2]

    def weights(self, colors):
        """(n, K) pigment weights for (n, 3) target colours (cached per colour bin)."""
        b = self.bins(colors)
        uniq, inv = np.unique(b, return_inverse=True)
        q = 2 ** self.bits - 1
        table = np.empty((len(uniq), self.K), np.float32)
        for j, code in enumerate(uniq.tolist()):  # loop over distinct colour bins (thousands)
            w = self._cache.get(code)
            if w is None:
                c = np.array([(code >> (2 * self.bits)) & q, (code >> self.bits) & q, code & q], np.float32) / q
                w = self._solve(c)
                self._cache[code] = w
            table[j] = w
        return table[inv.reshape(-1)]

    def sample(self, colors, rng: Rng):
        """Pick one pigment index per dot from its target colour's mixture."""
        W = self.weights(colors)
        cum = np.cumsum(W, axis=1)
        u = rng.random(len(W))[:, None] * cum[:, -1:]
        return np.minimum((cum < u).sum(1), self.K - 1)

    def by_hue(self, hue, lum, rng: Rng, spread: float = 0.04):
        """Chromatic pigment closest to (hue, luminance) per dot; used for complementary dots."""
        hue = np.asarray(hue, np.float32)[:, None]
        lum = np.asarray(lum, np.float32)[:, None]
        dh = np.abs(self.hsv[None, :, 0] - hue)
        dh = np.minimum(dh, 1 - dh)
        score = dh * 2.0 + np.abs(self.lum[None, :] - lum) * 1.0
        score = score + np.where(self.chroma[None, :], 0.0, 9.0)
        score = score + rng.uniform(0, spread, score.shape)
        return np.argmin(score, axis=1)

    def index(self, name):
        return self.names.index(name)


# ---------------------------------------------------------------------------
# Dot sprites
# ---------------------------------------------------------------------------

class DotBank:
    """Sprites (n_bins, n_var, win, win) for dots of diameter ``diameters[bin]``."""

    def __init__(self, diameters, rng: Rng, n_var: int = 24, round_: float = 0.8):
        diameters = np.asarray(diameters, np.float32)
        self.diameters = diameters
        self.win = int(np.ceil(diameters.max() * 1.25)) + 3
        w = self.win
        ar = np.arange(w, dtype=np.float32) - (w - 1) / 2
        X, Y = np.meshgrid(ar, ar)
        sprites = np.zeros((len(diameters), n_var, w, w), np.float32)
        for b, d in enumerate(diameters):  # loop over size bins
            r = rng.child("bin", str(b))
            for v in range(n_var):  # loop over sprite variants
                a = d / 2 * r.uniform(0.95, 1.12)
                bb = a * r.uniform(round_, 1.0)
                th = r.uniform(0, np.pi)
                ox, oy = r.uniform(-0.3, 0.3, 2)
                xr = (X - ox) * np.cos(th) + (Y - oy) * np.sin(th)
                yr = -(X - ox) * np.sin(th) + (Y - oy) * np.cos(th)
                # Irregular rim: a few low-frequency lobes on the radius.
                ang = np.arctan2(yr, xr)
                k = r.uniform(0.0, 0.08, 3)
                ph = r.uniform(0, 2 * np.pi, 3)
                wob = 1 + sum(k[i] * np.sin((i + 2) * ang + ph[i]) for i in range(3))
                rho = np.sqrt((xr / a) ** 2 + (yr / bb) ** 2) / wob
                edge = np.clip((1 - rho) * min(a, bb) / 0.8 + 0.5, 0, 1)
                # Paint texture: slightly heavier where the brush tip landed.
                tex = 0.86 + 0.14 * np.clip(1 - 0.6 * np.hypot(xr / a + 0.3, yr / bb + 0.2), 0, 1)
                tex = tex * (0.94 + 0.06 * r.random((w, w)))
                sprites[b, v] = np.clip(edge * tex, 0, 1)
        self.sprites = sprites
        self.n_var = n_var

    def bin_of(self, diam):
        d = np.asarray(diam, np.float32)[:, None]
        return np.argmin(np.abs(self.diameters[None, :] - d), axis=1)


def jittered_grid(shape, spacing: float, rng: Rng, jitter: float = 0.45, hex_: bool = True):
    """Dot centres on a (hex) lattice with random phase and jitter, covering the canvas."""
    H, W = shape
    ph = rng.uniform(0, spacing, 2)
    row_h = spacing * (0.866 if hex_ else 1.0)
    ys = np.arange(-row_h + ph[1], H + row_h, row_h)
    xs = np.arange(-spacing + ph[0], W + spacing, spacing)
    gx, gy = np.meshgrid(xs, ys)
    if hex_:
        gx = gx + (np.arange(len(ys)) % 2)[:, None] * spacing / 2
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    pts = pts + rng.uniform(-jitter, jitter, pts.shape) * spacing
    keep = (pts[:, 0] > -spacing / 2) & (pts[:, 0] < W + spacing / 2) & (pts[:, 1] > -spacing / 2) & (pts[:, 1] < H + spacing / 2)
    return pts[keep].astype(np.float32)


def splat(canvas, pts, colors, bank: DotBank, bins, variants, opacity=1.0, height=None):
    """Composite a batch of dots onto ``canvas`` (H, W, 3) in place.

    Coverage and colour are accumulated with ``np.bincount``; where dots of the
    same batch overlap, their colours are averaged (weighted by coverage).
    Returns the batch coverage map (H, W).
    """
    H, W = canvas.shape[:2]
    w = bank.win
    pad = w
    Hp, Wp = H + 2 * pad, W + 2 * pad
    n = len(pts)
    if n == 0:
        return np.zeros((H, W), np.float32)
    x0 = np.floor(pts[:, 0] - (w - 1) / 2 + 0.5).astype(np.int64) + pad
    y0 = np.floor(pts[:, 1] - (w - 1) / 2 + 0.5).astype(np.int64) + pad
    x0 = np.clip(x0, 0, Wp - w)
    y0 = np.clip(y0, 0, Hp - w)
    ar = np.arange(w)
    flat = ((y0[:, None, None] + ar[None, :, None]) * Wp + (x0[:, None, None] + ar[None, None, :])).ravel()
    alpha = bank.sprites[bins, variants] * np.broadcast_to(np.asarray(opacity, np.float32), (n,))[:, None, None]
    a = alpha.ravel()
    size = Hp * Wp
    A = np.bincount(flat, a, minlength=size)
    cols = np.asarray(colors, np.float32)
    C = np.stack([np.bincount(flat, (alpha * cols[:, c][:, None, None]).ravel(), minlength=size) for c in range(3)], -1)
    A = A.reshape(Hp, Wp)[pad:pad + H, pad:pad + W].astype(np.float32)
    C = C.reshape(Hp, Wp, 3)[pad:pad + H, pad:pad + W].astype(np.float32)
    over = np.maximum(A, 1.0)
    C /= over[..., None]
    A = A / over
    canvas *= (1 - A)[..., None]
    canvas += C
    if height is not None:
        Hm = np.bincount(flat, a, minlength=size).reshape(Hp, Wp)[pad:pad + H, pad:pad + W].astype(np.float32)
        height *= (1 - A)
        height += np.minimum(Hm, 1.2)
    return A


__all__ = ["PIGMENTS", "PigmentMixer", "DotBank", "jittered_grid", "splat"]
