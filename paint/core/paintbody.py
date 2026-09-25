"""Paint body: batched opaque strokes with a height map, for thick-paint styles.

``brush.render_stroke`` rasterises one stroke at a time; with thousands of
short impasto strokes the per-call overhead dominates. Here strokes are
rasterised in *batches* that share one square window size: the capsule-chain
distance, the (t, s) stroke coordinates and the bristle look-ups are computed
for the whole batch as (n, win, win) arrays. The only Python loop left is the
compositing of each window onto the canvas (a few slice operations), which is
what keeps the painter's order: later strokes cover earlier ones.

Every stroke also deposits *paint height*: bristle ridges across the stroke
(from a bristle look-up table sampled at ``s``) and a heavier load at the start
of the stroke. The accumulated height map is what an emboss pass lights to
give the paint body.

    bank = BristleBank(rng.child("bristles"))
    body = PaintBody((H, W), margin=64, ground=rgb, bank=bank)
    body.paint(paths, widths, colors, colors2, rng=rng.child("layer"))
    img, height = body.image, body.height
"""
from __future__ import annotations

import numpy as np

from .rng import Rng

NT, NS = 32, 40


class BristleBank:
    """A bank of pre-computed bristle tables, shared by all strokes of a render.

    For each brush ``i``:
      ink[i]    (NT, NS) paint deposited at (t, s): ~1 while loaded, breaking up
                on the outer bristles as the brush runs dry towards t = 1.
      ridge[i]  (NT, NS) height relief across the stroke: one ridge per bristle
                clump, grooves in between.
      streak[i] (NT, NS) in [0, 1]: which bristles carry the second colour of a
                two-colour loaded brush.
    """

    def __init__(self, rng: Rng, n: int = 48, bristles=(6, 13), dry: float = 0.5):
        t = np.linspace(0, 1, NT, dtype=np.float32)[:, None]
        s = np.linspace(-1, 1, NS, dtype=np.float32)[None, :]
        ink = np.zeros((n, NT, NS), np.float32)
        ridge = np.zeros((n, NT, NS), np.float32)
        streak = np.zeros((n, NT, NS), np.float32)
        for i in range(n):  # loop over brushes (tens)
            r = rng.child(str(i))
            nb = int(r.integers(bristles[0], bristles[1] + 1))
            spacing = 1.9 / max(nb - 1, 1)
            base = np.linspace(-0.95, 0.95, nb)
            offs = base + r.normal(0, 0.22 * spacing, nb)
            sig = spacing * r.uniform(0.35, 0.6, nb)
            load = r.uniform(0.85, 1.15, nb) * (1.0 - 0.3 * np.abs(base) ** 3)
            rate = dry * r.uniform(0.4, 1.7, nb) * (1.0 + 0.6 * np.abs(base))
            hgt = r.uniform(0.5, 1.0, nb)
            second = r.random(nb) < r.uniform(0.2, 0.5)
            phase = r.uniform(0, 2 * np.pi, nb)
            wob = 0.05 * spacing * np.sin(2 * np.pi * r.uniform(0.5, 1.5, nb)[None, :] * t + phase[None, :])
            for b in range(nb):  # loop over bristle clumps (tens)
                o = offs[b] + wob[:, b:b + 1]
                g_wide = np.exp(-(((s - o) / (sig[b] * 1.9)) ** 2))
                g_thin = np.exp(-(((s - o) / sig[b]) ** 2))
                amount = np.clip((load[b] - rate[b] * t) * 3.0, 0, 1)
                ink[i] += amount * g_wide
                ridge[i] = np.maximum(ridge[i], hgt[b] * g_thin * np.clip(amount * 1.5, 0.3, 1))
                if second[b]:
                    streak[i] = np.maximum(streak[i], g_thin)
        norm = np.percentile(ink[:, :4], 60, axis=(1, 2))[:, None, None]
        self.ink = np.clip(ink / np.maximum(norm, 1e-3), 0, 1).astype(np.float32)
        self.ridge = np.clip(ridge, 0, 1).astype(np.float32)
        self.streak = np.clip(streak, 0, 1).astype(np.float32)
        self.n = n


def width_profile(tk, tip: float = 0.45, head: float = 0.12, tail: float = 0.3):
    """Half-width multiplier along the stroke: quick swell in, long blunt taper out."""
    h = np.clip(tk / head, 0, 1)
    tl = np.clip((1 - tk) / tail, 0, 1)
    return tip + (1 - tip) * np.sin(np.minimum(h, tl) * np.pi / 2)


def stamp(paths, widths, brush_idx, win: int, bank: BristleBank, tip: float = 0.45):
    """Rasterise n strokes that share a (win, win) window.

    paths (n, P, 2) px, widths (n,) full width px. Windows are centred on the
    middle point of each path. Returns dict with y0, x0 (n,) window origins and
    (n, win, win) arrays alpha, height, streak, t.
    """
    paths = np.asarray(paths, np.float32)
    n, P, _ = paths.shape
    seg = np.hypot(*(paths[:, 1:] - paths[:, :-1]).transpose(2, 0, 1))  # (n, S)
    cum = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(seg, axis=1)], axis=1)
    total = np.maximum(cum[:, -1:], 1e-3)
    tk = cum / total
    hw = (0.5 * widths[:, None] * width_profile(tk, tip)).astype(np.float32)  # (n, P)
    c = paths[:, P // 2]
    y0 = np.floor(c[:, 1] - win / 2).astype(np.int64)
    x0 = np.floor(c[:, 0] - win / 2).astype(np.int64)
    ar = np.arange(win, dtype=np.float32) + 0.5
    X = (x0[:, None] + ar[None, :]).astype(np.float32)  # (n, win)
    Y = (y0[:, None] + ar[None, :]).astype(np.float32)
    A, B = paths[:, :-1], paths[:, 1:]
    ex = (B[..., 0] - A[..., 0])[:, :, None, None]
    ey = (B[..., 1] - A[..., 1])[:, :, None, None]
    wx = X[:, None, None, :] - A[..., 0][:, :, None, None]  # (n, S, 1, win)
    wy = Y[:, None, :, None] - A[..., 1][:, :, None, None]  # (n, S, win, 1)
    L2 = np.maximum(ex * ex + ey * ey, 1e-6)
    h = np.clip((wx * ex + wy * ey) / L2, 0, 1)  # (n, S, win, win)
    dist = np.hypot(wx - ex * h, wy - ey * h)
    r = hw[:, :-1, None, None] + (hw[:, 1:] - hw[:, :-1])[:, :, None, None] * h
    sd = dist - r
    j = np.argmin(sd, axis=1)[:, None]  # (n, 1, win, win)
    take = lambda a: np.take_along_axis(np.broadcast_to(a, sd.shape), j, axis=1)[:, 0]  # noqa: E731
    sd_j, h_j, r_j, d_j = take(sd), take(h), take(r), take(dist)
    cross = np.sign(take(ex * wy - ey * wx))
    jj = j[:, 0]
    nidx = np.arange(n)[:, None, None]
    cumA = cum[:, :-1][nidx, jj]
    segj = seg[nidx, jj]
    t = np.clip((cumA + h_j * segj) / total[:, :, None], 0, 1)
    s = np.clip(cross * d_j / np.maximum(r_j, 1e-3), -1, 1)
    cover = np.clip(0.5 - sd_j, 0, 1)
    ti = np.clip((t * (NT - 1) + 0.5).astype(np.int32), 0, NT - 1)
    si = np.clip(((s + 1) * 0.5 * (NS - 1) + 0.5).astype(np.int32), 0, NS - 1)
    bi = np.asarray(brush_idx)[:, None, None]
    ink = bank.ink[bi, ti, si]
    ridge = bank.ridge[bi, ti, si]
    streak = bank.streak[bi, ti, si]
    alpha = cover * ink
    # Heavier at the start of the stroke, where the loaded brush first lands.
    height = cover * (0.45 + 0.55 * ridge) * (1.25 - 0.6 * t) * np.clip(ink * 1.3, 0.2, 1)
    return {"y0": y0, "x0": x0, "alpha": alpha.astype(np.float32), "height": height.astype(np.float32),
            "streak": streak, "t": t.astype(np.float32)}


class PaintBody:
    """An opaque-paint canvas (padded by ``margin`` so windows never need clipping) plus a height map."""

    def __init__(self, shape, margin: int, ground, bank: BristleBank, budget: float = 2.5e6):
        H, W = shape
        self.H, self.W, self.M = H, W, int(margin)
        self.C = np.empty((H + 2 * self.M, W + 2 * self.M, 3), np.float32)
        self.C[:] = np.asarray(ground, np.float32)
        self.Hm = np.zeros((H + 2 * self.M, W + 2 * self.M), np.float32)
        self.bank = bank
        self.budget = budget
        self.count = 0

    @property
    def image(self):
        M = self.M
        return self.C[M:M + self.H, M:M + self.W]

    @property
    def height(self):
        M = self.M
        return self.Hm[M:M + self.H, M:M + self.W]

    def paint(self, paths, widths, colors, colors2=None, rng: Rng | None = None, streak_amt=0.5,
              clip_sdf=None, tol=None, opacity=1.0, body=1.0, tip=0.45):
        """Paint strokes in order (bucketed by window size, biggest first).

        paths (n, P, 2); widths (n,); colors / colors2 (n, 3); ``clip_sdf`` an
        (H, W) SDF (negative inside) the paint is confined to, loosened by
        ``tol`` (n,) px; ``opacity`` / ``body`` scalars or (n,).
        """
        paths = np.asarray(paths, np.float32)
        n = len(paths)
        if n == 0:
            return 0
        widths = np.asarray(widths, np.float32)
        colors = np.asarray(colors, np.float32)
        colors2 = colors if colors2 is None else np.asarray(colors2, np.float32)
        opacity = np.broadcast_to(np.asarray(opacity, np.float32), (n,))
        body = np.broadcast_to(np.asarray(body, np.float32), (n,))
        tol = np.zeros(n, np.float32) if tol is None else np.broadcast_to(np.asarray(tol, np.float32), (n,))
        streak_amt = np.broadcast_to(np.asarray(streak_amt, np.float32), (n,))
        rng = rng or Rng(0, "paintbody")
        bidx = rng.integers(0, self.bank.n, n)
        # Window: centred on the middle path point; must hold the whole stroke.
        mid = paths[:, paths.shape[1] // 2]
        reach = np.abs(paths - mid[:, None]).max(axis=(1, 2))
        need = 2 * (reach + widths * 0.5 + 2.0)
        wins = (np.ceil(need / 8.0) * 8).astype(int)
        wins = np.minimum(wins, 2 * self.M - 8)
        M = self.M
        for win in sorted(set(wins.tolist()), reverse=True):
            idx = np.nonzero(wins == win)[0]
            S = paths.shape[1] - 1
            chunk = max(1, int(self.budget // (S * win * win)))
            for k0 in range(0, len(idx), chunk):
                ii = idx[k0:k0 + chunk]
                fp = stamp(paths[ii], widths[ii], bidx[ii], win, self.bank, tip)
                a = fp["alpha"] * opacity[ii, None, None]
                if clip_sdf is not None:
                    yy = np.clip(fp["y0"][:, None] + np.arange(win), 0, self.H - 1)
                    xx = np.clip(fp["x0"][:, None] + np.arange(win), 0, self.W - 1)
                    d = clip_sdf[yy[:, :, None], xx[:, None, :]]
                    a = a * np.clip(0.5 - (d - tol[ii, None, None]), 0, 1)
                c1 = colors[ii][:, None, None, :]
                c2 = colors2[ii][:, None, None, :]
                m = (fp["streak"] * streak_amt[ii, None, None])[..., None]
                col = c1 + (c2 - c1) * m
                hs = fp["height"] * body[ii, None, None]
                ys0 = fp["y0"] + M
                xs0 = fp["x0"] + M
                for q in range(len(ii)):  # composite in painter's order
                    y, x = ys0[q], xs0[q]
                    if y < 0 or x < 0 or y + win > self.C.shape[0] or x + win > self.C.shape[1]:
                        continue
                    aq = a[q]
                    reg = self.C[y:y + win, x:x + win]
                    reg += (col[q] - reg) * aq[..., None]
                    hreg = self.Hm[y:y + win, x:x + win]
                    hreg += (hs[q] - 0.7 * hreg) * aq
        self.count += n
        return n


# Vectorised colour helpers ------------------------------------------------

def rgb_to_hsv(rgb):
    rgb = np.clip(np.asarray(rgb, np.float32), 0, 1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) / 6.0
    h = np.where(d < 1e-6, 0, h)
    return np.stack([h, s, mx], -1).astype(np.float32)


def hsv_to_rgb(hsv):
    hsv = np.asarray(hsv, np.float32)
    h, s, v = hsv[..., 0] % 1.0, np.clip(hsv[..., 1], 0, 1), np.clip(hsv[..., 2], 0, 1)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = [np.stack(c, -1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))]
    out = np.zeros(hsv.shape, np.float32)
    for k in range(6):
        out = np.where((i == k)[..., None], choices[k], out)
    return out


def jitter(rgb, rng: Rng, dh=0.02, ds=0.08, dv=0.08):
    """Per-row HSV jitter of an (n, 3) colour array."""
    hsv = rgb_to_hsv(rgb)
    n = hsv.shape[0]
    hsv[:, 0] += rng.normal(0, dh, n)
    hsv[:, 1] += rng.normal(0, ds, n)
    hsv[:, 2] += rng.normal(0, dv, n)
    return hsv_to_rgb(hsv)


def vivid(rgb, boost: float = 1.3, lift: float = 0.06):
    """Push saturation up (only for colours that already have a hue)."""
    hsv = rgb_to_hsv(np.asarray(rgb, np.float32))
    s = hsv[..., 1]
    hsv[..., 1] = np.where(s > 0.1, np.clip(s * boost + lift, 0, 1), s)
    return hsv_to_rgb(hsv)


__all__ = ["BristleBank", "PaintBody", "stamp", "width_profile", "rgb_to_hsv", "hsv_to_rgb", "jitter", "vivid"]
