"""Sumi-e: black ink on warm paper, painted with economy.

Painting model
--------------
* One ink. Everything is a *density* field in [0, 1]; the final image is the
  paper multiplied by ``1 - density * (1 - ink)``. At most one accent colour
  lives in its own density field (a red sun wash, or a small red seal).
* Objects are a handful of bristle strokes whose paths are derived from the
  shared SDF parts: scanlines/profiles clipped to the mask, top profiles
  (ridges), traced medial lines (stalks, reeds, leaves), or the known
  construction lines of the part (roof slopes, masts, handles).
* Washes are deposited into a wet scratch layer with a pooled (coffee-ring)
  rim and pigment mottling, then bleed into the paper fibres before merging.
* Distance: far objects are paler, wetter (more bleed) and less dry-brushed;
  near objects darker and drier (more flying white).
* Economy: regions (sea, field, river, table) are only suggested by a few
  bands and strokes; most of the paper stays empty.
* Before a solid object is painted, the ink beneath it is mostly "reserved"
  (a painter would have left that paper blank), so a white sail still reads
  against a grey mountain.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..core import sdf as S
from ..core.brush import Brush, bezier
from ..core.color import hex_to_rgb, mix
from ..core.masks import sdf_to_mask
from ..core.noise import blurred_white, fbm
from ..core.rng import Rng
from ..core.texture import paper
from ..core.wash import (arc, bleed, bbox_of, col_extents, deposit_wash, line, over, row_extents, row_runs,
                         sample, smooth_path, stroke_into, trace_ridge, wobble_path)
from .base import Context, clamp_params, encode, make_context, merge_params

NAME = "sumie"

DEFAULTS = {
    "ink": 1.0,             # overall ink strength
    "dryness": 0.6,         # how fast brushes run dry (flying white)
    "bleed": 1.0,           # wet feathering into paper fibres
    "coffee_ring": 0.45,    # darker pooled rims on washes
    "far_fade": 0.62,       # how much paler distant things are
    "wash_tone": 0.3,       # density of diluted-ink washes
    "brush_scale": 1.0,     # brush width multiplier
    "economy": 0.5,         # 0 = more marks, 1 = fewer marks / more empty paper
    "region_marks": 1.0,    # number of suggestion strokes on sea/field/river
    "ground_wash": 0.6,     # graded wash toward the viewer on water and ground
    "granulation": 0.35,    # pigment mottling in washes
    "paper_warmth": 0.5,    # 0 cool white .. 1 warm cream
    "wobble": 1.0,          # hand tremor on stroke paths
    "reserve": 0.85,        # how much ink under solid objects is left blank
    "seal_size": 0.055,     # seal side as a fraction of canvas height
    "accent": True,         # allow the single red accent (sun wash or seal)
}

KNOBS = {
    "ink": (0.5, 1.3, "overall ink strength; lower = paler, more atmospheric"),
    "dryness": (0.2, 1.2, "how quickly brushes run dry; higher = more flying white"),
    "bleed": (0.0, 2.5, "wet feathering of washes into paper fibres"),
    "coffee_ring": (0.0, 1.2, "darker pooled rims where washes dried"),
    "far_fade": (0.2, 0.85, "paleness of distant objects (aerial perspective)"),
    "wash_tone": (0.12, 0.55, "density of diluted-ink washes"),
    "brush_scale": (0.6, 1.6, "brush width multiplier"),
    "economy": (0.0, 1.0, "fewer marks and more empty paper as it rises"),
    "region_marks": (0.0, 2.0, "suggestion strokes on sea / field / river / table"),
    "ground_wash": (0.0, 1.2, "graded diluted wash on water / ground toward the viewer"),
    "granulation": (0.0, 0.8, "pigment mottling inside washes"),
    "paper_warmth": (0.0, 1.0, "paper tint from cool white to warm cream"),
    "wobble": (0.0, 2.5, "hand tremor on stroke paths"),
    "reserve": (0.0, 1.0, "how much background ink is left blank under solid objects"),
    "seal_size": (0.03, 0.08, "red seal size (fraction of canvas height)"),
}

RULES = {
    "max_mean_saturation": 0.18,   # monochrome ink plus one small accent
    "min_empty_fraction": 0.35,    # fraction of near-paper pixels
    "max_empty_fraction": 0.85,
    "max_accent_fraction": 0.05,   # fraction of pixels carrying the red accent
}

INK = hex_to_rgb("#161618")
ACCENT = hex_to_rgb("#b3342a")
SUN_ACCENT = hex_to_rgb("#c9472e")
PAPER_COOL = hex_to_rgb("#f2eee4")
PAPER_WARM = hex_to_rgb("#ecdfc4")

SOLID_RESERVE = {  # kind -> fraction of the reserve knob applied under it
    "house": 1.0, "lighthouse": 1.0, "windmill": 1.0, "boat": 1.0, "cup": 1.0, "teapot": 1.0,
    "vase": 1.0, "fruit": 1.0, "cliff": 0.7, "bird": 0.8, "tree": 0.5, "pine": 0.5, "cypress": 0.5,
    "bamboo": 0.4, "river": 0.9, "moon": 1.0, "sun": 0.6, "cloud": 0.5,
}


# ---------------------------------------------------------------------------
# Painter: ink layers, stroke and wash helpers
# ---------------------------------------------------------------------------

class Painter:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.f = ctx.frame
        self.sc = ctx.frame.scale
        self.P = ctx.params
        self.H, self.W = ctx.frame.shape
        rng = ctx.rng
        tint = mix(PAPER_COOL, PAPER_WARM, float(self.P["paper_warmth"]))
        self.paper_img, self.fib = paper(self.f.shape, rng.child("paper"), tint=tint, strength=1.2, scale=self.sc)
        tooth = blurred_white(self.f.shape, rng.child("tooth"), 0.6 * self.sc + 0.3)
        self.grain = np.clip(0.55 * self.fib + 0.45 * tooth, 0, 1).astype(np.float32)
        self.mottle = fbm(self.f.shape, 70 * self.sc, rng.child("mottle"), octaves=3)
        self.edge_noise = fbm(self.f.shape, 22 * self.sc, rng.child("edge"), octaves=2)
        # Horizontally stretched noise: water streaks and ground strata.
        self.hstreak = blurred_white(self.f.shape, rng.child("hstreak"), (2.0 * self.sc, 60 * self.sc))
        self.D = np.zeros(self.f.shape, np.float32)   # ink
        self.A = np.zeros(self.f.shape, np.float32)   # accent
        self.n_strokes = 0
        self.n_washes = 0
        self.kind_strokes: dict[str, int] = {}
        self._kind = "-"
        self._r = rng.child("objects")
        self._k = 0

    # -- per-object bookkeeping ------------------------------------------
    def begin(self, si: int, kind: str):
        self._r = self.ctx.rng.child("obj", str(si), kind)
        self._k = 0
        self._kind = kind

    def r(self, label: str):
        return self._r.child(label)

    def _next(self):
        self._k += 1
        return self._r.child("s", str(self._k))

    # -- tone model ---------------------------------------------------------
    def far(self, depth: float) -> float:
        return float(np.clip((depth - 0.3) / 0.65, 0, 1) ** 1.2)

    def tone(self, depth: float, base: float) -> float:
        return float(np.clip(base * self.P["ink"] * (1 - self.P["far_fade"] * self.far(depth)), 0, 1))

    def dry(self, depth: float, k: float = 1.0) -> float:
        return float(self.P["dryness"] * k * (0.55 + 0.75 * (1 - self.far(depth))))

    def bleed_px(self, depth: float, k: float = 1.0) -> float:
        return float(self.P["bleed"] * k * self.sc * (1.2 + 3.2 * self.far(depth)))

    def layer(self):
        return np.zeros(self.f.shape, np.float32)

    # -- marks ----------------------------------------------------------------
    def stroke(self, path, width, amount, layer=None, pressure=None, dry=0.6, wobble=1.0, **kw):
        """One bristle stroke. ``width`` in px (already canvas-scaled)."""
        path = np.asarray(path, np.float32)
        if len(path) < 2 or amount <= 0.003:
            return
        w = float(max(1.4 * self.sc, width * self.P["brush_scale"]))
        rng = self._next()
        amp = self.P["wobble"] * wobble * min(0.12 * w, 4.0 * self.sc)
        path = wobble_path(path, amp, rng.child("wob"))
        br = Brush(width=w, bristles=int(np.clip(w / 1.7, 5, 40)), bristle_spread=0.45,
                   taper_start=kw.get("taper_start", 0.12), taper_end=kw.get("taper_end", 0.35),
                   tip=min(0.25, kw.get("tip", 0.15)), load=kw.get("load", 1.0), dry_rate=float(dry),
                   dry_texture=kw.get("dry_texture", 0.8), opacity=1.0, wobble=kw.get("bwobble", 0.05),
                   edge=kw.get("edge", 1.0))
        target = self.D if layer is None else layer
        stroke_into(target, path, br, rng, pressure, grain=self.grain, amount=float(np.clip(amount, 0, 1)),
                    max_len=220 * self.sc, streak=kw.get("streak", 0.05), mode=kw.get("mode", "over"))
        self.n_strokes += 1
        self.kind_strokes[self._kind] = self.kind_strokes.get(self._kind, 0) + 1

    def dab(self, x, y, size, amount, angle=0.0, layer=None, dry=0.25, elong=1.3):
        """A short, loaded, round-ended touch of the brush (dots, knots, needles clusters)."""
        L = size * elong
        ca, sa = math.cos(angle), math.sin(angle)
        p = line((x - ca * L / 2, y - sa * L / 2), (x + ca * L / 2, y + sa * L / 2), 4)
        self.stroke(p, size, amount, layer=layer, dry=dry, wobble=0.3, taper_start=0.35, taper_end=0.45, tip=0.4)

    def wash_mask(self, mask, amount, layer, fade=None, ring=None, ring_width=None):
        ring = self.P["coffee_ring"] if ring is None else ring
        rw = (5.0 if ring_width is None else ring_width) * self.sc
        deposit_wash(layer, mask, float(amount), fade=fade, ring=ring, ring_width=rw,
                     granulation=self.P["granulation"], mottle=self.mottle)
        self.n_washes += 1

    def wash_sdf(self, sdf, amount, layer, fade=None, jitter=3.0, ring=None, ring_width=None):
        """Wash a part: its SDF edge is roughened by noise so the wash edge is organic."""
        j = jitter * self.sc
        m = sdf_to_mask(sdf + (self.edge_noise - 0.5) * 2 * j, softness=1.5)
        self.wash_mask(m, amount, layer, fade=fade, ring=ring, ring_width=ring_width)

    def commit(self, layer, sigma, target=None):
        bleed(layer, self.fib, sigma)
        over(self.D if target is None else target, layer)

    def reserve(self, mask, k):
        if k > 0:
            self.D *= 1.0 - np.clip(k * mask, 0, 1)

    # -- output ---------------------------------------------------------------
    def image(self):
        Dt = np.clip(self.D, 0, 1) * (0.92 + 0.16 * self.fib)
        img = self.paper_img * (1.0 - np.clip(Dt, 0, 1)[..., None] * (1.0 - INK))
        A = np.clip(self.A, 0, 1)
        acc = ACCENT if self.ctx.info.get("accent") == "seal" else SUN_ACCENT
        img = img * (1.0 - A[..., None] * (1.0 - acc))
        return np.clip(img, 0, 1).astype(np.float32), Dt, A


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def _shadow_x(pt: Painter) -> float:
    """+1 if the shadow side is to the right, -1 if left."""
    lx, _ = pt.ctx.light_dir
    return 1.0 if lx < 0 else -1.0


def _top_of_region(part) -> float:
    return float(part.sdf[0, 0] + 0.5)


def _profile_paths(sdf, y0, y1, us, n=26):
    """Vertical paths at fractions ``us`` (-1..1) across the row extents of a part."""
    ys = np.linspace(y0, y1, n)
    xl, xr = row_extents(sdf, ys)
    ok = ~np.isnan(xl)
    if ok.sum() < 3:
        return [], None
    ys, xl, xr = ys[ok], xl[ok], xr[ok]
    c, hw = (xl + xr) / 2, (xr - xl) / 2
    paths = [np.stack([c + u * hw, ys], axis=1) for u in us]
    return paths, hw


def _components(mask, min_px=3):
    lab, n = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        sub = lab[sl] == i + 1
        ys, xs = np.nonzero(sub)
        if len(xs) < min_px:
            continue
        out.append((xs + sl[1].start + 0.5, ys + sl[0].start + 0.5))
    return out


def _peaks(ys, window):
    """Indices of local minima (peaks in image space) of a top profile."""
    v = np.where(np.isnan(ys), np.inf, ys)
    mn = ndimage.minimum_filter1d(v, size=max(3, int(window)), mode="nearest")
    idx = np.nonzero((v == mn) & np.isfinite(v))[0]
    keep = []
    for i in idx:  # collapse plateaus
        if not keep or i - keep[-1] > window // 2:
            keep.append(int(i))
    return keep


# ---------------------------------------------------------------------------
# Object drawers: (pt, shape) -> None
# ---------------------------------------------------------------------------

def _draw_sun(pt: Painter, sh):
    part, r, sc = sh.parts[0], sh.radius, pt.sc
    wet = pt.layer()
    if pt.ctx.info.get("accent") == "sun":
        pt.wash_sdf(part.sdf, 0.55, wet, jitter=2.0, ring=0.5 * pt.P["coffee_ring"], ring_width=r * 0.12 / sc)
        pt.commit(wet, pt.bleed_px(0.6), target=pt.A)
        wet = pt.layer()
    else:
        pt.wash_sdf(part.sdf, pt.tone(0.5, 0.16), wet, jitter=2.0, ring_width=r * 0.1 / sc)
    if pt.P["economy"] < 0.85:
        # A thin band of mist drifting across the lower half of the disc.
        rr = pt.r("mist")
        y = sh.cy + r * rr.uniform(0.25, 0.5)
        p = line((sh.cx - r * 1.9, y + r * 0.04), (sh.cx + r * 1.5, y - r * 0.03), 14, sag=r * 0.05)
        pt.stroke(p, r * 0.14, pt.tone(0.6, 0.22), layer=wet, dry=0.15, taper_start=0.3, taper_end=0.5, tip=0.05)
    pt.commit(wet, pt.bleed_px(0.8, 1.4))


def _draw_moon(pt: Painter, sh):
    part, r, sc = sh.parts[0], sh.radius, pt.sc
    X, Y = pt.f.grid
    d = part.sdf
    # The moon is left as paper; a soft halo of wash around it makes it glow.
    halo = np.exp(-np.clip(d, 0, None) / (r * 0.9)) * np.clip(d / (2.0 * sc), 0, 1)
    halo *= np.hypot(X - sh.cx, Y - sh.cy) < r * 4.5
    wet = pt.layer()
    pt.wash_mask(halo.astype(np.float32), 0.26 * pt.P["ink"], wet, ring=0.0)
    pt.commit(wet, 2.0 * sc)
    if (sh.obj.variant or "crescent") == "full":
        a0 = pt.r("arc").uniform(0, 2 * np.pi)
        pt.stroke(arc(sh.cx, sh.cy, r * 1.01, r * 1.01, a0, a0 + 4.6, 40), 2.4 * sc, 0.38 * pt.P["ink"], dry=0.8)
    else:
        ang = math.radians(sh.obj.params.get("phase_angle", 35))
        lit = math.atan2(math.sin(ang), -math.cos(ang))  # lit limb faces away from the cut
        pt.stroke(arc(sh.cx, sh.cy, r * 1.0, r * 1.0, lit - 1.5, lit + 1.5, 36), 2.6 * sc, 0.4 * pt.P["ink"], dry=0.6)


def _draw_stars(pt: Painter, sh):
    part, sc = sh.parts[0], pt.sc
    comps = _components(part.sdf < 0)
    sizes = []
    for xs, ys in comps:
        rad = max(xs.max() - xs.min(), ys.max() - ys.min()) / 6.0  # rays are 3r long each way
        sizes.append(rad)
    med = float(np.median(sizes)) if sizes else 1.0
    rr = pt.r("stars")
    for (xs, ys), rad in zip(comps, sizes):
        x, y = float(xs.mean()), float(ys.mean())
        pt.dab(x, y, max(2.2 * sc, rad * 0.8 + 1.5 * sc), 0.7 * pt.P["ink"], angle=rr.uniform(0, np.pi), dry=0.1,
               elong=1.0)
        if rad > med:
            L = rad * 3.2
            a = rr.uniform(-0.3, 0.3)
            for aa in (a, a + np.pi / 2):
                ca, sa = math.cos(aa), math.sin(aa)
                pt.stroke(line((x - ca * L, y - sa * L), (x + ca * L, y + sa * L), 5), 1.3 * sc, 0.3 * pt.P["ink"],
                          dry=0.5, wobble=0.2, taper_start=0.45, taper_end=0.5, tip=0.05)


def _draw_cloud(pt: Painter, sh):
    part, sc = sh.parts[0], pt.sc
    b = part.stats()[3]
    X, Y = pt.f.grid
    fade = (0.45 + 0.55 * np.clip((Y - b[1]) / max(b[3] - b[1], 1), 0, 1)).astype(np.float32)
    wet = pt.layer()
    pt.wash_sdf(part.sdf, pt.P["wash_tone"] * 0.55 * pt.P["ink"], wet, fade=fade, jitter=5.0,
                ring=0.5 * pt.P["coffee_ring"], ring_width=10)
    pt.commit(wet, pt.bleed_px(sh.depth, 1.6))
    xs = np.linspace(b[0] + 2, b[2] - 2, 40)
    top, _ = col_extents(part.sdf, xs)
    ok = ~np.isnan(top)
    if ok.sum() > 5:
        path = np.stack([xs[ok], top[ok] + 2 * sc], axis=1)
        n = len(path)
        cut = int(n * pt.r("cut").uniform(0.4, 0.6))
        for seg in (path[: cut - 1], path[cut + 2:]):
            if len(seg) > 3:
                pt.stroke(seg, 3.2 * sc, pt.tone(0.5, 0.42), dry=pt.dry(sh.depth, 0.8), taper_end=0.5, tip=0.05)


def _draw_mountain(pt: Painter, sh):
    rock, snow = sh.part("rock"), sh.part("snow")
    sc, depth = pt.sc, sh.depth
    x0, y0, x1, y1 = rock.stats()[3]
    X, Y = pt.f.grid
    t = np.clip((Y - y0) / max(y1 - y0, 1), 0, 1)
    fade = 1.0 - 0.9 * t ** 1.1                     # mist swallows the foot of the mountain
    if snow is not None and pt.P["economy"] < 0.9:
        sm = ndimage.gaussian_filter(sdf_to_mask(snow.sdf), 3 * sc)
        fade = fade * (1.0 - 0.8 * sm)
    wet = pt.layer()
    pt.wash_sdf(rock.sdf, pt.tone(depth, pt.P["wash_tone"] * 1.45), wet, fade=fade.astype(np.float32),
                jitter=4.0, ring_width=6)
    pt.commit(wet, pt.bleed_px(depth, 1.2))

    # Dry-brush ridge along the top profile, in a few overlapping strokes.
    step = 4 * sc
    xs = np.arange(x0 + 1, x1 - 1, step, dtype=np.float32)
    top, _ = col_extents(rock.sdf, xs)
    ok = ~np.isnan(top)
    if ok.sum() < 4:
        return
    xs, top = xs[ok], top[ok]
    ridge = smooth_path(np.stack([xs, top + 2.5 * sc], axis=1), 2)
    rr = pt.r("ridge")
    near_k = 1.0 - pt.far(depth)
    w = (6.0 + 5.0 * near_k) * sc
    n = len(ridge)
    cuts = sorted(set([0, n] + [int(c) for c in rr.uniform(0.15, 0.85, 2) * n]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = ridge[max(0, a - 1): b]
        if len(seg) < 3:
            continue
        pr = 0.6 + 0.6 * rr.random(5)
        pt.stroke(seg, w, pt.tone(depth - 0.1, 0.95), pressure=pr, dry=pt.dry(depth, 1.3),
                  taper_start=0.08, taper_end=0.3)

    # Texture strokes (cun): shorter echoes of the ridge a little below it, near each peak.
    econ = pt.P["economy"]
    for pi_, i in enumerate(_peaks(top, int(len(top) * 0.25))):
        if top[i] > y0 + (y1 - y0) * 0.5:
            continue
        for side in (-1, 1):
            j = i + side * int(len(top) * rr.uniform(0.06, 0.14))
            j = int(np.clip(j, 0, len(top) - 1))
            for k in range(1 if econ > 0.6 else 2):
                lo, hi = sorted((i, j))
                seg = ridge[lo: hi + 1].copy()
                if len(seg) < 3:
                    continue
                drop = (y1 - y0) * rr.uniform(0.08, 0.2) * (k + 1)
                seg[:, 1] += drop
                seg[:, 0] += side * drop * 0.35
                if sample(rock.sdf, seg[:, 0], seg[:, 1]).max() > 0:
                    seg = seg[sample(rock.sdf, seg[:, 0], seg[:, 1]) < -2]
                    if len(seg) < 3:
                        continue
                pt.stroke(seg, w * 0.55, pt.tone(depth, 0.55), dry=pt.dry(depth, 1.6), taper_start=0.2,
                          taper_end=0.5, tip=0.05)


def _draw_hills(pt: Painter, sh):
    sc, H, W = pt.sc, pt.H, pt.W
    X, Y = pt.f.grid
    xs = np.arange(W, dtype=np.float32) + 0.5
    n = len(sh.parts)
    rr = pt.r("hills")
    for k, part in enumerate(sh.parts):
        top, _ = col_extents(part.sdf, xs)
        if np.isnan(top).all():
            continue
        top = np.where(np.isnan(top), np.nanmax(top), top).astype(np.float32)
        near_k = (k + 1) / n
        depth = sh.depth - 0.12 * k
        bh = H * (0.05 + 0.04 * near_k)
        band = np.exp(-np.clip(Y - top[None, :], 0, None) / bh).astype(np.float32)
        m = sdf_to_mask(part.sdf + (pt.edge_noise - 0.5) * 6 * sc, softness=1.5)
        if k > 0:
            pt.reserve(m * np.exp(-np.clip(Y - top[None, :], 0, None) / (bh * 2)), 0.6 * pt.P["reserve"])
        wet = pt.layer()
        pt.wash_mask(m, pt.tone(depth, pt.P["wash_tone"] * (0.9 + 0.5 * near_k)), wet, fade=band, ring_width=6)
        pt.commit(wet, pt.bleed_px(depth, 1.3))
        # Ridge line: a long dry stroke broken where the brush lifted.
        ridge = np.stack([xs[::6], top[::6] + 2 * sc], axis=1)
        cuts = sorted(set([0, len(ridge)] + [int(c) for c in rr.uniform(0.1, 0.9, 2 if k == n - 1 else 1) * len(ridge)]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            seg = ridge[a: max(a + 2, b - 3)]
            if len(seg) >= 3 and rr.random() < (0.95 if k == n - 1 else 0.6):
                pt.stroke(seg, (3.5 + 4.0 * near_k) * sc, pt.tone(depth, 0.75), pressure=0.6 + 0.6 * rr.random(5),
                          dry=pt.dry(depth, 1.4), taper_start=0.1, taper_end=0.35)


def _water_reflections(pt: Painter, top, bottom):
    """Broken horizontal strokes under sun / moon / tall objects standing on the water."""
    sc = pt.sc
    rr = pt.r("refl")
    for other in pt.ctx.shapes:
        if other.kind in ("sun", "moon") and other.cy < top:
            n = 5
            for i in range(n):
                y = top + (bottom - top) * (0.03 + 0.07 * i + 0.01 * rr.random())
                half = other.radius * (0.9 - 0.13 * i) * rr.uniform(0.7, 1.1)
                x = other.cx + rr.uniform(-0.15, 0.15) * other.radius
                pt.stroke(line((x - half, y), (x + half, y), 6), (2.0 + 0.8 * i) * sc, pt.tone(0.6, 0.3),
                          dry=0.9, taper_start=0.25, taper_end=0.4, tip=0.05)


def _draw_sea(pt: Painter, sh):
    part = sh.parts[0]
    sc, H, W = pt.sc, pt.H, pt.W
    top = _top_of_region(part)
    X, Y = pt.f.grid
    rr = pt.r("waves")
    # Soft band of diluted ink just under the horizon: the far water.
    wet = pt.layer()
    band = np.exp(-np.clip(Y - top, 0, None) / (0.035 * H)).astype(np.float32)
    pt.wash_mask(sdf_to_mask(part.sdf), pt.tone(0.75, pt.P["wash_tone"] * 0.55), wet, fade=band, ring=0.8, ring_width=3)
    # Graded water tone toward the viewer, broken into horizontal streaks.
    if pt.P["ground_wash"] > 0:
        g = np.clip((Y - top) / max(H - top, 1), 0, 1) ** 1.3
        streaks = np.clip(0.25 + 1.3 * (pt.hstreak - 0.5) + 0.5, 0, 1)
        fade = (g * streaks).astype(np.float32)
        pt.wash_mask(sdf_to_mask(part.sdf), pt.tone(0.3, pt.P["wash_tone"] * 0.7 * pt.P["ground_wash"]), wet,
                     fade=fade, ring=0.0)
    pt.commit(wet, pt.bleed_px(0.8))
    # Horizon line in two or three pieces.
    xa = 0.0
    for i in range(3):
        xb = xa + W * rr.uniform(0.25, 0.45)
        if rr.random() < 0.8:
            pt.stroke(line((xa, top + 1.5 * sc), (min(xb, W), top + 1.5 * sc), 12), 2.6 * sc, pt.tone(0.7, 0.55),
                      dry=pt.dry(0.7, 1.2), taper_start=0.15, taper_end=0.3, tip=0.1)
        xa = xb + W * rr.uniform(0.02, 0.08)
        if xa >= W:
            break
    # A few wave strokes: short and faint near the horizon, longer and darker near the viewer.
    n = int(round(11 * pt.P["region_marks"] * (1.25 - 0.5 * pt.P["economy"])))
    for i in range(n):
        u = (i + rr.random()) / max(n, 1)
        y = top + (H - top) * (0.06 + 0.9 * u ** 1.35)
        L = W * (0.03 + 0.11 * u) * rr.uniform(0.7, 1.3)
        x = rr.uniform(0.02 * W, W - L - 0.02 * W)
        k = 12
        xx = np.linspace(x, x + L, k)
        yy = y + np.sin(np.linspace(0, np.pi * rr.uniform(1.0, 2.5), k) + rr.uniform(0, 6)) * (1.5 + 4 * u) * sc
        pt.stroke(np.stack([xx, yy], 1), (2.0 + 6.0 * u) * sc, pt.tone(0.75 - 0.6 * u, 0.3 + 0.35 * u),
                  dry=pt.dry(0.75 - 0.6 * u, 1.3), taper_start=0.25, taper_end=0.45, tip=0.05)
    _water_reflections(pt, top, H)


def _draw_river(pt: Painter, sh):
    part = sh.parts[0]
    sc, H, W = pt.sc, pt.H, pt.W
    top = float(np.nanmin(col_extents(part.sdf, np.arange(0, W, 4))[0]))
    ys = np.arange(top + 2 * sc, H - 1, 4 * sc)
    xl, xr = row_extents(part.sdf, ys)
    ok = ~np.isnan(xl)
    ys, xl, xr = ys[ok], xl[ok], xr[ok]
    if len(ys) < 4:
        return
    rr = pt.r("river")
    t = (ys - ys[0]) / max(ys[-1] - ys[0], 1)
    marks = pt.P["region_marks"] * (1.25 - 0.5 * pt.P["economy"])
    # Water is suggested by horizontal wash strokes laid across the current (never by its edges):
    # far bands thin and close together, near bands broad and spaced out.
    wet = pt.layer()
    n = int(round(16 * marks))
    for i in range(n):
        u = (i + rr.random()) / max(n, 1)
        j = int(np.clip(u ** 1.3 * len(ys), 0, len(ys) - 1))
        wdt = xr[j] - xl[j]
        L = wdt * rr.uniform(0.5, 0.95)
        x = xl[j] + rr.uniform(-0.05, 1.0) * max(wdt - L, 1)
        pt.stroke(line((x, ys[j]), (x + L, ys[j] + rr.uniform(-1, 1) * sc), 10), (3 + 16 * t[j]) * sc,
                  pt.tone(0.75 - 0.5 * t[j], pt.P["wash_tone"] * 0.75), layer=wet, dry=pt.dry(0.5, 0.9),
                  taper_start=0.25, taper_end=0.4, tip=0.05)
    pt.commit(wet, pt.bleed_px(0.6, 1.3))
    # Ripples: thin dark wavy strokes, a few.
    for i in range(int(round(8 * marks))):
        j = int(np.clip(((i + rr.random()) / max(1, int(round(8 * marks)))) ** 0.8 * len(ys), 0, len(ys) - 1))
        wdt = xr[j] - xl[j]
        if wdt < 12 * sc:
            continue
        L = wdt * rr.uniform(0.12, 0.3)
        x = rr.uniform(xl[j] + 0.08 * wdt, xr[j] - 0.08 * wdt - L)
        k = 10
        xx = np.linspace(x, x + L, k)
        yy = ys[j] + np.sin(np.linspace(0, np.pi * 2, k) + rr.uniform(0, 6)) * (1 + 2 * t[j]) * sc
        pt.stroke(np.stack([xx, yy], 1), (1.5 + 3.5 * t[j]) * sc, pt.tone(0.7 - 0.5 * t[j], 0.6),
                  dry=pt.dry(0.5, 1.3), taper_start=0.3, taper_end=0.4, tip=0.05)
    # Banks: a few tufts of grass where water meets land.
    for i in range(int(round(4 * marks))):
        j = int(rr.integers(len(ys) // 4, len(ys)))
        side = 1 if rr.random() < 0.5 else -1
        bx = (xr[j] if side > 0 else xl[j]) + side * 4 * sc
        hgt = (8 + 30 * t[j]) * sc
        for k in range(int(rr.integers(3, 6))):
            x0 = bx + rr.uniform(-1, 1) * hgt * 0.3
            lean = rr.uniform(-0.4, 0.4) * hgt
            pt.stroke(line((x0, ys[j]), (x0 + lean, ys[j] - hgt * rr.uniform(0.6, 1.1)), 5, sag=lean * 0.2),
                      (1.5 + 2 * t[j]) * sc, pt.tone(0.6 - 0.4 * t[j], 0.7), dry=pt.dry(0.3, 0.8),
                      taper_start=0.05, taper_end=0.8, tip=0.02)


def _draw_field(pt: Painter, sh):
    part = sh.parts[0]
    sc, H, W = pt.sc, pt.H, pt.W
    xs = np.arange(W, dtype=np.float32) + 0.5
    top, _ = col_extents(part.sdf, xs)
    if np.isnan(top).all():
        return
    top = np.where(np.isnan(top), np.nanmin(top), top).astype(np.float32)
    X, Y = pt.f.grid
    rr = pt.r("field")
    wet = pt.layer()
    band = np.exp(-np.clip(Y - top[None, :], 0, None) / (0.03 * H)).astype(np.float32)
    pt.wash_mask(sdf_to_mask(part.sdf + (pt.edge_noise - 0.5) * 6 * sc, 1.5),
                 pt.tone(0.75, pt.P["wash_tone"] * 0.6), wet, fade=band, ring_width=4)
    if pt.P["ground_wash"] > 0:
        g = np.clip((Y - top[None, :]) / max(H - float(top.mean()), 1), 0, 1) ** 1.1
        patches = np.clip(0.35 + (pt.mottle - 0.4) * 1.4, 0, 1) * (0.7 + 0.6 * (pt.hstreak - 0.5))
        fade = (g * patches).astype(np.float32)
        pt.wash_mask(sdf_to_mask(part.sdf), pt.tone(0.3, pt.P["wash_tone"] * 1.1 * pt.P["ground_wash"]), wet,
                     fade=fade, ring=0.0)
    pt.commit(wet, pt.bleed_px(0.75))
    ridge = np.stack([xs[::8], top[::8] + 1.5 * sc], axis=1)
    cuts = sorted(set([0, len(ridge)] + [int(c) for c in rr.uniform(0.1, 0.9, 3) * len(ridge)]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = ridge[a: max(a + 2, b - 3)]
        if len(seg) >= 3 and rr.random() < 0.75:
            pt.stroke(seg, 3.0 * sc, pt.tone(0.7, 0.55), dry=pt.dry(0.7, 1.4), taper_start=0.15, taper_end=0.35)
    marks = pt.P["region_marks"] * (1.25 - 0.5 * pt.P["economy"])
    ymin = float(top.mean())
    # Ground strokes: a few near-horizontal dry touches, longer near the viewer.
    for i in range(int(round(5 * marks))):
        u = (i + rr.random()) / max(1, int(round(5 * marks)))
        y = ymin + (H - ymin) * (0.2 + 0.72 * u)
        L = W * (0.05 + 0.1 * u)
        x = rr.uniform(0.02 * W, W - L)
        pt.stroke(line((x, y), (x + L, y + rr.uniform(-0.02, 0.02) * L), 8, sag=rr.uniform(-2, 2) * sc),
                  (2.5 + 4 * u) * sc, pt.tone(0.6 - 0.4 * u, 0.4), dry=pt.dry(0.3, 1.5), taper_start=0.2,
                  taper_end=0.45, tip=0.05)
    # Grass tufts: a few quick upward flicks.
    for i in range(int(round(3 * marks))):
        u = rr.uniform(0.35, 0.95)
        y = ymin + (H - ymin) * u
        x = rr.uniform(0.05 * W, 0.95 * W)
        hgt = H * (0.02 + 0.035 * u)
        for j in range(int(rr.integers(3, 6))):
            bx = x + rr.uniform(-1, 1) * hgt * 0.4
            lean = rr.uniform(-0.5, 0.5) * hgt
            pt.stroke(line((bx, y), (bx + lean, y - hgt * rr.uniform(0.6, 1.1)), 6, sag=lean * 0.2),
                      (1.8 + 1.8 * u) * sc, pt.tone(0.5 - 0.3 * u, 0.7), dry=pt.dry(0.3, 0.8),
                      taper_start=0.05, taper_end=0.8, tip=0.02)


def _draw_table(pt: Painter, sh):
    top_part, front = sh.part("top"), sh.part("front")
    sc, H, W = pt.sc, pt.H, pt.W
    top = float(top_part.sdf[0, 0] + 0.5)
    edge = float(front.sdf[0, 0] + 0.5)
    X, Y = pt.f.grid
    rr = pt.r("table")
    wet = pt.layer()
    fade = np.exp(-np.clip(Y - edge, 0, None) / (0.09 * H)).astype(np.float32)
    pt.wash_sdf(front.sdf, pt.tone(0.35, pt.P["wash_tone"] * 0.9), wet, fade=fade, jitter=3, ring_width=4)
    pt.commit(wet, pt.bleed_px(0.4, 1.2))
    # The table edge: one long confident line, re-touched twice.
    xa = rr.uniform(0.0, 0.03) * W
    while xa < W * 0.97:
        xb = min(W, xa + W * rr.uniform(0.3, 0.55))
        pt.stroke(line((xa, edge + 1), (xb, edge + 1 + rr.uniform(-1.5, 1.5) * sc), 14), 5.0 * sc, pt.tone(0.35, 0.8),
                  pressure=0.7 + 0.5 * rr.random(4), dry=pt.dry(0.35, 1.1), taper_start=0.08, taper_end=0.25)
        xa = xb + rr.uniform(0.005, 0.03) * W


def _draw_cliff(pt: Painter, sh):
    rock = sh.parts[0]
    sc, H = pt.sc, pt.H
    s = sh.radius * 2
    side = -1 if sh.obj.x < 0.5 else 1
    depth = sh.depth
    ys = np.arange(sh.cy + 1, H - 1, 4 * sc)
    xl, xr = row_extents(rock.sdf, ys)
    ok = ~np.isnan(xl)
    ys, xl, xr = ys[ok], xl[ok], xr[ok]
    if len(ys) < 4:
        return
    edge = xl if side > 0 else xr
    X, Y = pt.f.grid
    ex = np.interp(Y[:, 0], ys, edge).astype(np.float32)[:, None]
    fade = np.exp(-np.abs(X - ex) / (0.35 * s)) * (1.0 - 0.45 * np.clip((Y - sh.cy) / (H - sh.cy), 0, 1))
    wet = pt.layer()
    pt.wash_sdf(rock.sdf, pt.tone(depth, pt.P["wash_tone"] * 1.3), wet, fade=fade.astype(np.float32), jitter=4)
    pt.commit(wet, pt.bleed_px(depth))
    rr = pt.r("cliff")
    path = np.stack([edge - side * 2 * sc, ys], axis=1)
    n = len(path)
    cuts = sorted(set([0, n] + [int(c) for c in rr.uniform(0.25, 0.75, 2) * n]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = path[max(0, a - 1): b]
        if len(seg) >= 3:
            pt.stroke(seg, 9 * sc, pt.tone(depth - 0.25, 0.9), pressure=0.6 + 0.6 * rr.random(5),
                      dry=pt.dry(depth - 0.2, 1.3), taper_start=0.08, taper_end=0.25)
    # Top edge of the headland.
    outer = pt.W if side > 0 else 0.0
    x_top = float(edge[0])
    pt.stroke(line((x_top, sh.cy + 2 * sc), (x_top + (outer - x_top) * 0.9, sh.cy + 2 * sc), 10), 5 * sc,
              pt.tone(depth - 0.2, 0.75), dry=pt.dry(depth, 1.2), taper_start=0.1, taper_end=0.5)
    # Texture strokes (cun) near the face, following its slope.
    for i in range(int(round(6 * (1.2 - 0.5 * pt.P["economy"])))):
        y = rr.uniform(sh.cy + 0.08 * s, H - 0.08 * s)
        x = float(np.interp(y, ys, edge)) + side * rr.uniform(0.04, 0.3) * s
        L = s * rr.uniform(0.08, 0.16)
        slope = float(np.interp(y + L, ys, edge) - np.interp(y, ys, edge))
        pt.stroke(line((x, y), (x + slope * 0.9 + side * L * 0.2, y + L), 6), 4 * sc, pt.tone(depth - 0.1, 0.55),
                  dry=pt.dry(depth, 1.7), taper_start=0.2, taper_end=0.6, tip=0.05)


def _draw_tree(pt: Painter, sh):
    trunk, crown = sh.part("trunk"), sh.part("crown")
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    rr = pt.r("tree")
    sx = _shadow_x(pt)
    # Anchor: a pale pool of shadow on the ground and a few grass flicks at the foot of the trunk.
    _ground_shadow(pt, cx + sx * s * 0.12, cy + s * 0.01, s * 0.32, s * 0.035, depth)
    for j in range(int(rr.integers(5, 9))):
        gx = cx + rr.uniform(-0.18, 0.2) * s
        hgt = s * rr.uniform(0.03, 0.07)
        lean = rr.uniform(-0.5, 0.5) * hgt
        pt.stroke(line((gx, cy + s * 0.012), (gx + lean, cy + s * 0.012 - hgt), 5, sag=lean * 0.2),
                  max(1.8 * sc, s * 0.008), pt.tone(depth, 0.75), dry=pt.dry(depth, 0.8), taper_start=0.05,
                  taper_end=0.8, tip=0.02)
    wet = pt.layer()
    # Pale wet mass of foliage underneath the dabs.
    pt.wash_sdf(crown.sdf, pt.tone(depth, pt.P["wash_tone"] * 0.5), wet, jitter=8, ring=0.3 * pt.P["coffee_ring"],
                ring_width=10)
    # Trunk: two strokes from the ground up, with dry-brush bark.
    tx, ty = cx + s * 0.02, cy - s * 0.6
    p = line((cx, cy + s * 0.01), (tx, ty), 12, sag=s * 0.02 * rr.uniform(-1, 1))
    pt.stroke(p, s * 0.1, pt.tone(depth, 0.85), pressure=[1.0, 0.85, 0.7, 0.55], dry=pt.dry(depth, 1.2),
              taper_start=0.05, taper_end=0.3, tip=0.4)
    pt.stroke(p + np.array([sx * s * 0.02, 0]), s * 0.05, pt.tone(depth, 0.6), pressure=[1.0, 0.8, 0.5],
              dry=pt.dry(depth, 1.6), taper_start=0.1, taper_end=0.4)
    # Branches forking into the crown.
    ccx, ccy = cx, cy - s * 0.68
    for i in range(4):
        t = rr.uniform(0.55, 0.92)
        bx, by = cx + (tx - cx) * t, cy + (ty - cy) * t
        ang = -np.pi / 2 + (1 if i % 2 else -1) * rr.uniform(0.35, 1.1)
        L = s * rr.uniform(0.14, 0.26)
        ex, ey = bx + math.cos(ang) * L, by + math.sin(ang) * L
        pt.stroke(line((bx, by), (ex, ey), 8, sag=L * 0.12 * rr.uniform(-1, 1)), s * 0.03, pt.tone(depth, 0.8),
                  dry=pt.dry(depth, 1.2), taper_start=0.05, taper_end=0.7, tip=0.05)
    # Foliage: clusters of loaded dabs, denser and darker on the shadow side.
    x0, y0, x1, y1 = crown.stats()[3]
    n_cl = int(round(12 * (1.2 - 0.5 * pt.P["economy"])))
    centres = []
    tries = 0
    while len(centres) < n_cl and tries < 200:  # rejection sampling of cluster centres
        tries += 1
        x, y = rr.uniform(x0, x1), rr.uniform(y0, y1)
        if sample(crown.sdf, x, y)[0] < -s * 0.04:
            centres.append((x, y))
    R = max(1.0, (x1 - x0) / 2)
    for (x, y) in centres:
        shade = float(np.clip(0.5 + 0.5 * sx * (x - ccx) / R + 0.25 * (y - ccy) / R, 0, 1))
        for j in range(int(rr.integers(6, 11))):
            dx, dy = rr.normal(0, s * 0.05, 2)
            px_, py_ = x + dx, y + dy * 0.8
            if sample(crown.sdf, px_, py_)[0] > s * 0.01:
                continue
            pt.dab(px_, py_, s * rr.uniform(0.028, 0.045), pt.tone(depth, rr.uniform(0.3, 0.55) + 0.35 * shade),
                   angle=rr.uniform(-0.6, 0.6), dry=pt.dry(depth, 0.5), elong=rr.uniform(1.2, 1.9))
    pt.commit(wet, pt.bleed_px(depth, 1.2))


def _draw_pine(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    needles = sh.part("needles")
    rr = pt.r("pine")
    wet = pt.layer()
    pt.wash_sdf(needles.sdf, pt.tone(depth, pt.P["wash_tone"] * 0.6), wet, jitter=5, ring_width=5)
    pt.commit(wet, pt.bleed_px(depth, 1.1))
    # Trunk: up through the tiers, slightly bent.
    top_y = cy - s * 0.1 - 3 * s * 0.19 - s * 0.12
    pt.stroke(line((cx, cy + s * 0.02), (cx + s * 0.015, top_y), 14, sag=s * 0.025 * rr.uniform(-1, 1)), s * 0.065,
              pt.tone(depth, 0.85), pressure=[1.0, 0.8, 0.55, 0.3], dry=pt.dry(depth, 1.2), taper_start=0.04,
              taper_end=0.5, tip=0.1)
    tiers = 4
    for i in range(tiers):
        yb = cy - s * 0.1 - i * s * 0.19
        w = s * 0.3 * (1 - i / (tiers + 0.8))
        th = s * 0.32
        for side in (-1, 1):
            # Branch drooping out to the tier tip.
            bx, by = cx, yb - th * 0.3
            tipx, tipy = cx + side * w * 0.95, yb - th * 0.02
            pt.stroke(line((bx, by), (tipx, tipy), 8, sag=-side * s * 0.02), s * 0.018, pt.tone(depth, 0.8),
                      dry=pt.dry(depth, 1.0), taper_start=0.05, taper_end=0.6, tip=0.1)
            # Needle clusters: fans of short strokes radiating upward-outward.
            for u in (0.45, 0.9):
                nx, ny = bx + (tipx - bx) * u, by + (tipy - by) * u
                n_nd = int(round(9 * (1.15 - 0.4 * pt.P["economy"])))
                ln = s * (0.1 - 0.012 * i) * (0.8 + 0.4 * u)
                for k in range(n_nd):
                    a = -np.pi / 2 + side * (-0.9 + 2.2 * k / max(1, n_nd - 1)) * 0.9 + rr.normal(0, 0.08)
                    ex, ey = nx + math.cos(a) * ln, ny + math.sin(a) * ln * 0.75
                    pt.stroke(line((nx, ny), (ex, ey), 4), max(1.6 * sc, s * 0.01), pt.tone(depth, 0.9),
                              dry=pt.dry(depth, 0.7), wobble=0.2, taper_start=0.05, taper_end=0.8, tip=0.02)


def _draw_cypress(pt: Painter, sh):
    part = sh.parts[0]
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    rr = pt.r("cypress")
    sx = _shadow_x(pt)
    # Body: three long loaded strokes rising along the flame silhouette, tapering to the tip.
    us = (-0.55, 0.0, 0.55)
    paths, hw = _profile_paths(part.sdf, cy - s * 0.02, cy - s * 0.985, us, n=40)
    wet = pt.layer()
    if paths:
        width = 2 * float(hw.max()) * 0.5
        for u, p in zip(us, paths):
            pt.stroke(p, width * 1.1, pt.tone(depth, 0.5 + 0.3 * sx * u), layer=wet, pressure=np.clip(hw / hw.max(), 0.15, 1),
                      dry=pt.dry(depth, 0.9), taper_start=0.03, taper_end=0.3, tip=0.02, mode="max")
    pt.commit(wet, pt.bleed_px(depth, 0.9))
    # Tufts: short upward flicks along both edges, like flames licking out.
    step = s * 0.045
    y = cy - step
    keep = 1.0 - 0.35 * pt.P["economy"]
    while y > cy - s * 0.9:
        xl, xr = row_extents(part.sdf, [y])
        if not np.isnan(xl[0]):
            for side, xe in ((-1, xl[0]), (1, xr[0])):
                if rr.random() > keep:
                    continue
                L = s * rr.uniform(0.06, 0.1)
                x0 = xe - side * s * 0.02
                pt.stroke(line((x0, y + L * 0.3), (x0 + side * L * 0.35, y - L * 0.7), 5, sag=-side * L * 0.15),
                          s * 0.03, pt.tone(depth, 0.55 + 0.3 * (side == sx)), dry=pt.dry(depth, 0.9), wobble=0.3,
                          taper_start=0.1, taper_end=0.75, tip=0.02)
        y -= step * rr.uniform(0.8, 1.2)


def _stalk_lines(st_sdf, base_y, s, sc):
    """Find straight stalks in a stalk SDF: base x from a row near the bottom, angle by ray casting."""
    runs = row_runs(st_sdf[int(np.clip(base_y, 0, st_sdf.shape[0] - 1))])
    out = []
    angs = np.linspace(-0.2, 0.2, 41)
    L = np.linspace(0, s * 1.2, 240)
    for a, b in runs:
        bx, wd = (a + b) / 2, b - a
        dx, dy = np.sin(angs)[:, None], -np.cos(angs)[:, None]
        xs, ys = bx + dx * L[None], base_y + dy * L[None]
        v = sample(st_sdf, xs.ravel(), ys.ravel()).reshape(xs.shape)
        outside = v > 0.5
        first_out = np.where(outside.any(axis=1), np.argmax(outside, axis=1), len(L) - 1)
        k = int(np.argmax(first_out))
        length = float(L[first_out[k]])
        out.append((bx, float(angs[k]), length, wd))
    return out


def _replay_bamboo(pt: Painter, sh):
    """Replay the shared bamboo construction (same labelled geometry stream as the shape builder)
    and validate it against the part SDFs. Returns (stalks, leaves) or None if it does not match,
    in which case the caller falls back to reading the SDFs directly."""
    ctx, o, f = pt.ctx, sh.obj, pt.f
    idx = next((k for k, ob in enumerate(ctx.scene.objects) if ob is o), None)
    if idx is None:
        return None
    rng = Rng(ctx.seed, "geometry").child("shape", str(idx), o.kind)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    n = max(1, o.count if o.count > 1 else 3)
    stalks, leaves = [], []
    for k in range(n):
        bx = cx + (k - (n - 1) / 2) * s * 0.16 + rng.uniform(-0.03, 0.03) * s
        lean = rng.uniform(-0.08, 0.08) * s
        top = cy - s * rng.uniform(0.85, 1.05)
        w = s * 0.022
        stalks.append((bx, cy, bx + lean, top, w))
        for j in range(2):
            t = rng.uniform(0.5, 0.95)
            ax, ay = bx + lean * t, cy + (top - cy) * t
            side = 1 if rng.random() < 0.5 else -1
            for m in range(int(rng.integers(3, 5))):
                ang = side * rng.uniform(0.15, 1.1) + np.pi / 2 * (1 - side) + 0.35 * side
                ln = s * rng.uniform(0.12, 0.2)
                leaves.append(((ax, ay), (ax + np.cos(ang) * ln, ay + abs(np.sin(ang)) * ln * 0.5 + ln * 0.2)))
    st, lv = sh.part("stalks").sdf, sh.part("leaves").sdf
    H, W = st.shape
    for x0, y0, x1, y1, w in stalks:
        ts = np.linspace(0.1, 0.9, 9)
        xs, ys = x0 + (x1 - x0) * ts, y0 + (y1 - y0) * ts
        ins = (xs > 0) & (xs < W) & (ys > 0) & (ys < H)
        if ins.any() and sample(st, xs[ins], ys[ins]).max() > -w * 0.5:
            return None
    for (ax, ay), (ex, ey) in leaves:
        mx, my = ax + (ex - ax) * 0.3, ay + (ey - ay) * 0.3
        if 0 < mx < W and 0 < my < H and sample(lv, mx, my)[0] > -s * 0.008:
            return None
    return stalks, leaves


def _draw_bamboo(pt: Painter, sh):
    st, leaves = sh.part("stalks"), sh.part("leaves")
    s, cy, depth, sc = sh.radius * 2, sh.cy, sh.depth, pt.sc
    H = pt.H
    rr = pt.r("bamboo")
    replay = _replay_bamboo(pt, sh)
    pt.ctx.info.setdefault("bamboo_geometry", []).append("replayed" if replay else "detected")
    if replay:
        stalks = [(x0, y0, x1, y1, 2 * w) for x0, y0, x1, y1, w in replay[0]]
        leaf_segs = _spread_fans(replay[1], 1.4)
    else:
        base_y = min(cy, H) - s * 0.03
        stalks = []
        for bx, ang, length, wd in _stalk_lines(st.sdf, base_y, s, sc):
            ux, uy = math.sin(ang), -math.cos(ang)
            back = (cy - base_y) / max(-uy, 1e-3)
            x0 = bx - ux * back
            stalks.append((x0, cy, x0 + ux * (length + back), cy + uy * (length + back), wd))
        leaf_segs = _leaf_segments(leaves.sdf, st.sdf, s)
    for x0, y0, x1, y1, wd in stalks:
        stone = rr.uniform(0.75, 1.05)
        gap = 0.006
        for j in range(7):
            t0, t1 = j / 7 + gap, (j + 1) / 7 - gap
            p0 = (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0)
            p1 = (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)
            pt.stroke(line(p0, p1, 8, sag=rr.normal(0, 0.6) * sc), wd * 1.08 * (1 - 0.25 * (j / 7) ** 2),
                      pt.tone(depth - 0.15, 0.6 * stone), pressure=[1.0, 0.92, 0.92, 1.0],
                      dry=pt.dry(depth, 0.55), taper_start=0.02, taper_end=0.03, tip=0.25, wobble=0.25, streak=0.3)
        for j in range(1, 7):
            t = j / 7
            nx, ny = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            pt.stroke(line((nx - wd * 0.72, ny + 1 * sc), (nx + wd * 0.72, ny - 1 * sc), 6, sag=wd * 0.16),
                      max(2.0 * sc, wd * 0.24), pt.tone(depth - 0.2, 0.95), dry=pt.dry(depth, 0.5),
                      taper_start=0.25, taper_end=0.25, tip=0.2, wobble=0.2)
    # Leaves: one stroke each from the base out to the tip, swelling then tapering to a point.
    for base, tip in leaf_segs:
        L = math.hypot(tip[0] - base[0], tip[1] - base[1])
        path = line(base, tip, 10, sag=L * rr.uniform(-0.08, 0.08))
        pt.stroke(path, s * 0.017 * rr.uniform(1.45, 1.9), pt.tone(depth - 0.25, rr.uniform(0.5, 0.9)),
                  pressure=[0.3, 1.0, 0.85, 0.55, 0.25],
                  dry=pt.dry(depth, 0.45), taper_start=0.15, taper_end=0.6, tip=0.02, wobble=0.5)


def _spread_fans(segs, k):
    """Open each fan of leaves sharing a base a little wider so the blades separate."""
    groups: dict = {}
    for base, tip in segs:
        groups.setdefault((round(base[0], 1), round(base[1], 1)), []).append((base, tip))
    out = []
    for grp in groups.values():
        angs = [math.atan2(t[1] - b[1], t[0] - b[0]) for b, t in grp]
        mean = math.atan2(np.mean(np.sin(angs)), np.mean(np.cos(angs)))
        for (b, t), a in zip(grp, angs):
            L = math.hypot(t[0] - b[0], t[1] - b[1])
            na = mean + (math.remainder(a - mean, 2 * math.pi)) * k
            out.append((b, (b[0] + math.cos(na) * L, b[1] + math.sin(na) * L)))
    return out


def _leaf_segments(leaf_sdf, stalk_sdf, s):
    lm = leaf_sdf < 0
    if not lm.any():
        return []
    win = bbox_of(lm.astype(np.float32), 0.5, pad=4)
    y0, y1, x0, x1 = win
    L = lm[y0:y1, x0:x1]
    att_mask = L & (stalk_sdf[y0:y1, x0:x1] < 0)
    atts = [(float(xs.mean()) + x0, float(ys.mean()) + y0) for xs, ys in _components(att_mask, min_px=2)]
    if not atts:
        return []
    Yw, Xw = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dist = np.full(L.shape, np.inf, np.float32)
    for ax, ay in atts:  # loop over attachment points
        dist = np.minimum(dist, np.hypot(Xw - ax, Yw - ay))
    dist = np.where(L, dist, -np.inf)
    mx = ndimage.maximum_filter(dist, size=max(5, int(s * 0.035)))
    cand = np.nonzero((dist == mx) & np.isfinite(dist) & (dist > s * 0.06))
    order = np.argsort(-dist[cand])
    out = []
    for i in order:
        tip = (cand[1][i] + x0 + 0.5, cand[0][i] + y0 + 0.5)
        if any(math.hypot(tip[0] - t[0], tip[1] - t[1]) < s * 0.04 for _, t in out):
            continue
        base = min(atts, key=lambda a: math.hypot(a[0] - tip[0], a[1] - tip[1]))
        out.append((base, tip))
    return out


def _draw_reeds(pt: Painter, sh):
    part = sh.parts[0]
    s, cy, depth, sc = sh.radius * 2, sh.cy, sh.depth, pt.sc
    y1 = cy - s * 0.4
    runs = row_runs(part.sdf[int(np.clip(y1, 0, pt.H - 1))])
    for a, b in runs:
        x = (a + b) / 2
        up = trace_ridge(part.sdf, (x, y1), (0, -1), step=s * 0.02, max_steps=80, search=s * 0.012)
        down = trace_ridge(part.sdf, (x, y1), (0, 1), step=s * 0.02, max_steps=80, search=s * 0.012,
                           stop=lambda q: q[1] >= cy)
        path = np.concatenate([down[::-1], up[1:]], axis=0)
        if len(path) < 5:
            continue
        pt.stroke(smooth_path(path, 2), s * 0.028, pt.tone(depth - 0.1, 0.85), pressure=[1, 0.9, 0.7, 0.4],
                  dry=pt.dry(depth, 1.0), taper_start=0.03, taper_end=0.75, tip=0.02, wobble=0.5)


def _draw_house(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    w, h, roof_h = s * 0.5, s * 0.42, s * 0.32
    sx = _shadow_x(pt)
    rr = pt.r("house")
    ey = cy - h
    apex = (cx, ey - roof_h)
    wet = pt.layer()
    # Roof: a wet wash of the gable, a darker half on the shadow side, thin slope edges, dark eave.
    X, Y = pt.f.grid
    roof = S.polygon(X, Y, [(cx - w * 0.62, ey), (cx + w * 0.62, ey), apex])
    pt.wash_sdf(roof, pt.tone(depth, 0.42), wet, jitter=1.5, ring_width=3)
    pt.wash_sdf(S.intersect(roof, -sx * (X - cx)), pt.tone(depth, 0.25), wet, jitter=1.0, ring=0.2)
    pt.commit(wet, pt.bleed_px(depth, 0.6))
    for side in (-1, 1):
        pt.stroke(line(apex, (cx + side * w * 0.66, ey + s * 0.005), 6), s * 0.022, pt.tone(depth, 0.85),
                  dry=pt.dry(depth, 1.0), taper_start=0.05, taper_end=0.3, tip=0.1)
    pt.stroke(line((cx - w * 0.68, ey), (cx + w * 0.68, ey), 6, sag=rr.normal(0, 1) * sc), s * 0.045,
              pt.tone(depth, 0.95), dry=pt.dry(depth, 0.9), taper_start=0.08, taper_end=0.2, tip=0.3)
    # Chimney.
    pt.stroke(line((cx + w * 0.25, ey - roof_h * 0.85), (cx + w * 0.25, ey - roof_h * 0.35), 4), s * 0.06,
              pt.tone(depth, 0.8), dry=pt.dry(depth, 0.8), taper_start=0.2, taper_end=0.2, tip=0.6)
    # Walls: two thin verticals, the shadow side heavier, and a pale shadow wash.
    for side in (-1, 1):
        heavy = side == sx
        pt.stroke(line((cx + side * w / 2, ey + s * 0.02), (cx + side * w / 2, cy), 6), s * (0.035 if heavy else 0.022),
                  pt.tone(depth, 0.85 if heavy else 0.55), dry=pt.dry(depth, 1.2), taper_start=0.1, taper_end=0.3)
    wet = pt.layer()
    X, Y = pt.f.grid
    body = S.box(X, Y, cx + sx * w * 0.25, cy - h / 2, w * 0.25, h / 2)
    pt.wash_sdf(body, pt.tone(depth, pt.P["wash_tone"] * 0.5), wet, jitter=1.5, ring_width=3)
    pt.commit(wet, pt.bleed_px(depth, 0.6))
    # Door and window: single dabs.
    pt.stroke(line((cx - w * 0.15, cy - h * 0.48), (cx - w * 0.15, cy), 4), w * 0.19, pt.tone(depth, 0.95),
              dry=pt.dry(depth, 0.4), taper_start=0.1, taper_end=0.1, tip=0.7, wobble=0.2)
    pt.dab(cx + w * 0.2, cy - h * 0.55, w * 0.18, pt.tone(depth, 0.85), angle=np.pi / 2, dry=0.2, elong=1.2)
    # Ground line.
    pt.stroke(line((cx - w * 0.75, cy + 1), (cx + w * 0.85, cy + 1), 6), s * 0.02, pt.tone(depth, 0.5),
              dry=pt.dry(depth, 1.4), taper_start=0.2, taper_end=0.5, tip=0.05)


def _draw_lighthouse(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    hb, ht, h = s * 0.12, s * 0.075, s * 0.72
    sx = _shadow_x(pt)
    # Contours: two tapering side strokes, shadow side heavier.
    for side in (-1, 1):
        heavy = side == sx
        pt.stroke(line((cx + side * hb, cy), (cx + side * ht, cy - h), 10), s * (0.03 if heavy else 0.018),
                  pt.tone(depth - 0.3, 0.9 if heavy else 0.6), dry=pt.dry(depth, 1.1), taper_start=0.05, taper_end=0.2)
    # Shadow wash on one half of the tower.
    X, Y = pt.f.grid
    tower = S.polygon(X, Y, [(cx - hb, cy), (cx + hb, cy), (cx + ht, cy - h), (cx - ht, cy - h)])
    half = S.intersect(tower, -sx * (X - cx))
    wet = pt.layer()
    pt.wash_sdf(half, pt.tone(depth, pt.P["wash_tone"] * 0.5), wet, jitter=1.0, ring_width=3)
    # Stripes: broad horizontal bands, bowed to show the round tower.
    band = h / 5
    for k in range(3):
        yc = cy - (1 + 2 * k) * band
        if yc < cy - h + band * 0.3:
            yc = cy - h + band * 0.3
        t = (cy - yc) / h
        hw = hb + (ht - hb) * t
        pt.stroke(line((cx - hw * 0.98, yc), (cx + hw * 0.98, yc), 8, sag=hw * 0.12), band * (0.85 if k < 2 else 0.55),
                  pt.tone(depth - 0.2, 0.62), layer=wet, dry=pt.dry(depth, 0.8), taper_start=0.04, taper_end=0.06,
                  tip=0.85, wobble=0.2)
    pt.commit(wet, pt.bleed_px(depth, 0.6))
    # Gallery, lantern frame and cap.
    gy = cy - h - s * 0.015
    pt.stroke(line((cx - ht * 1.5, gy), (cx + ht * 1.5, gy), 6), s * 0.034, pt.tone(depth - 0.3, 0.95),
              dry=pt.dry(depth, 0.6), taper_start=0.1, taper_end=0.15, tip=0.4)
    ly = cy - h - s * 0.08
    for side in (-1, 1):
        pt.stroke(line((cx + side * ht * 0.8, gy - s * 0.02), (cx + side * ht * 0.8, ly - s * 0.045), 4),
                  max(2 * sc, s * 0.012), pt.tone(depth - 0.3, 0.85), dry=0.4, taper_start=0.1, taper_end=0.1, tip=0.5)
    apex = (cx, cy - h - s * 0.22)
    for side in (-1, 1):
        pt.stroke(line(apex, (cx + side * ht * 1.1, cy - h - s * 0.13), 6), s * 0.04, pt.tone(depth - 0.3, 0.95),
                  dry=pt.dry(depth, 0.5), taper_start=0.15, taper_end=0.2, tip=0.3)
    # The lamp: a small pale glow left mostly as paper.
    wet = pt.layer()
    pt.wash_sdf(S.box(X, Y, cx, ly, ht * 0.8, s * 0.05), pt.tone(depth, 0.08), wet, jitter=1.0, ring=1.0,
                ring_width=2)
    pt.commit(wet, 1.0 * sc)


def _windmill_rotation(sh, hx, hy, s):
    if sh.obj.rotation:
        return math.radians(sh.obj.rotation)
    spars = sh.part("spars")
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    R = s * 0.35
    v = sample(spars.sdf, hx + R * np.cos(th), hy + R * np.sin(th))
    inside = v < 0
    if not inside.any():
        return 0.3
    # Circular mean of 4*theta gives the cross orientation modulo 90 degrees.
    a4 = np.angle(np.sum(np.exp(4j * th[inside])))
    return float(a4 / 4)


def _draw_windmill(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    h = s * 0.62
    sx = _shadow_x(pt)
    X, Y = pt.f.grid
    for side in (-1, 1):
        heavy = side == sx
        pt.stroke(line((cx + side * s * 0.13, cy), (cx + side * s * 0.07, cy - h), 10), s * (0.028 if heavy else 0.018),
                  pt.tone(depth - 0.2, 0.9 if heavy else 0.6), dry=pt.dry(depth, 1.1), taper_start=0.05, taper_end=0.2)
    wet = pt.layer()
    tower = S.polygon(X, Y, [(cx - s * 0.13, cy), (cx + s * 0.13, cy), (cx + s * 0.07, cy - h), (cx - s * 0.07, cy - h)])
    pt.wash_sdf(S.intersect(tower, -sx * (X - cx)), pt.tone(depth, pt.P["wash_tone"] * 0.5), wet, jitter=1.5)
    pt.wash_sdf(sh.part("sails").sdf, pt.tone(depth, pt.P["wash_tone"] * 0.4), wet, jitter=1.0, ring=1.0, ring_width=3)
    pt.commit(wet, pt.bleed_px(depth, 0.6))
    # Cap: a dark wedge.
    apex = (cx, cy - h - s * 0.1)
    for u in (-1, 0, 1):
        pt.stroke(line(apex, (cx + u * s * 0.1, cy - h + s * 0.005), 5), s * 0.05, pt.tone(depth - 0.2, 0.85),
                  dry=pt.dry(depth, 0.5), taper_start=0.2, taper_end=0.2, tip=0.4)
    pt.stroke(line((cx, cy - s * 0.12), (cx, cy), 3), s * 0.06, pt.tone(depth - 0.2, 0.9), dry=0.3, tip=0.7,
              taper_start=0.1, taper_end=0.1)
    # Sails: spar + outer edge + a couple of lattice ribs each.
    hx, hy = cx, cy - h - s * 0.02
    rot = _windmill_rotation(sh, hx, hy, s)
    L = s * 0.5
    for k in range(4):
        a = rot + k * np.pi / 2
        ux, uy = math.cos(a), math.sin(a)
        px_, py_ = -uy, ux
        pt.stroke(line((hx, hy), (hx + ux * L, hy + uy * L), 8), s * 0.022, pt.tone(depth - 0.2, 0.9),
                  dry=pt.dry(depth, 0.9), taper_start=0.05, taper_end=0.3, tip=0.2)
        o = s * 0.1
        pt.stroke(line((hx + ux * 0.18 * L + px_ * o, hy + uy * 0.18 * L + py_ * o), (hx + ux * L + px_ * o, hy + uy * L + py_ * o), 8),
                  s * 0.012, pt.tone(depth - 0.2, 0.6), dry=pt.dry(depth, 1.2), taper_start=0.1, taper_end=0.2)
        for u in (0.45, 0.75):
            b0 = (hx + ux * u * L, hy + uy * u * L)
            pt.stroke(line(b0, (b0[0] + px_ * o, b0[1] + py_ * o), 3), s * 0.009, pt.tone(depth - 0.2, 0.5),
                      dry=0.6, wobble=0.1, taper_start=0.1, taper_end=0.2)
    pt.dab(hx, hy, s * 0.05, pt.tone(depth - 0.2, 0.95), dry=0.1, elong=1.0)


def _draw_boat(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    o = sh.obj
    L = s * 0.9
    hh = s * 0.13
    flip = -1 if o.params.get("facing", "right") == "left" else 1
    rr = pt.r("boat")
    stern = (cx - flip * L / 2, cy - hh)
    bow = (cx + flip * L / 2 * 1.08, cy - hh * 1.25)
    # Reflection first (under the hull), faint broken horizontals.
    if any(x.kind in ("sea", "river") for x in pt.ctx.shapes):
        for i, (dy, fr) in enumerate(((0.55, 0.65), (1.3, 0.45), (2.1, 0.28))):
            half = L * fr / 2
            x = cx + rr.normal(0, L * 0.04)
            pt.stroke(line((x - half, cy + hh * dy), (x + half, cy + hh * dy), 6), s * (0.035 - 0.008 * i),
                      pt.tone(depth + 0.1, 0.35), dry=pt.dry(depth, 1.4), taper_start=0.25, taper_end=0.35, tip=0.05)
    # Hull: one broad loaded stroke from stern to bow, then gunwale and keel lines.
    body = np.array([(cx - flip * L * 0.44, cy - hh * 0.45), (cx, cy - hh * 0.4), (cx + flip * L * 0.5, cy - hh * 0.95)])
    body = line(body[0], body[1], 6)[:-1].tolist() + line(body[1], body[2], 6).tolist()
    pt.stroke(np.array(body), hh * 1.15, pt.tone(depth, 0.75), pressure=[0.9, 1.0, 0.9, 0.45],
              dry=pt.dry(depth, 0.8), taper_start=0.06, taper_end=0.35, tip=0.3)
    pt.stroke(line(stern, bow, 10, sag=flip * hh * 0.25), s * 0.035, pt.tone(depth, 0.95), dry=pt.dry(depth, 1.0),
              taper_start=0.05, taper_end=0.4, tip=0.1)
    keel = [(cx - flip * L * 0.4, cy + hh * 0.05), (cx + flip * L * 0.36, cy + hh * 0.1), bow]
    pt.stroke(np.array(line(keel[0], keel[1], 6)[:-1].tolist() + line(keel[1], keel[2], 5).tolist()), s * 0.025,
              pt.tone(depth, 0.85), dry=pt.dry(depth, 1.2), taper_start=0.08, taper_end=0.4, tip=0.05)
    variant = o.variant or "sail"
    if variant in ("sail", "sailboat"):
        mx = cx - flip * L * 0.02
        mtop = cy - hh - s * 0.9
        wet = pt.layer()
        pt.wash_sdf(sh.part("sail").sdf, pt.tone(depth, pt.P["wash_tone"] * 0.3), wet, jitter=1.5, ring=1.0, ring_width=3)
        pt.commit(wet, pt.bleed_px(depth, 0.6))
        pt.stroke(line((mx, cy - hh), (mx, mtop), 12), max(2.0 * sc, s * 0.016), pt.tone(depth, 0.9),
                  dry=pt.dry(depth, 0.8), taper_start=0.03, taper_end=0.3, tip=0.3)
        for sgn, top_off, bot_off, reach in ((1, 0.05, 0.08, 0.46), (-1, 0.12, 0.1, 0.42)):
            a = (mx + sgn * flip * s * 0.03, mtop + s * top_off)
            b = (mx + sgn * flip * s * 0.03, cy - hh - s * bot_off)
            c = (mx + sgn * flip * L * reach, cy - hh - s * bot_off)
            # Leading edge bellies outward; foot is a short dry stroke.
            pt.stroke(line(a, c, 12, sag=-sgn * flip * s * 0.05), s * 0.022, pt.tone(depth, 0.75),
                      dry=pt.dry(depth, 1.0), taper_start=0.05, taper_end=0.35, tip=0.1)
            pt.stroke(line(b, c, 6, sag=s * 0.01), s * 0.014, pt.tone(depth, 0.55), dry=pt.dry(depth, 1.3),
                      taper_start=0.1, taper_end=0.3)
    elif variant == "fishing":
        bx = cx - flip * L * 0.1
        top_y = cy - hh - s * 0.2
        wet = pt.layer()
        X, Y = pt.f.grid
        pt.wash_sdf(S.box(X, Y, bx, cy - hh - s * 0.1, L * 0.14, s * 0.1), pt.tone(depth, 0.18), wet, jitter=1.5)
        pt.commit(wet, pt.bleed_px(depth, 0.6))
        pt.stroke(line((bx - L * 0.17, top_y), (bx + L * 0.17, top_y), 6, sag=s * 0.01), s * 0.04,
                  pt.tone(depth, 0.95), dry=pt.dry(depth, 0.7), taper_start=0.08, taper_end=0.2, tip=0.3)
        for side in (-1, 1):
            pt.stroke(line((bx + side * L * 0.13, top_y), (bx + side * L * 0.13, cy - hh), 4), s * 0.018,
                      pt.tone(depth, 0.7), dry=pt.dry(depth, 0.8), taper_start=0.1, taper_end=0.2)
        pt.dab(bx + flip * L * 0.04, cy - hh - s * 0.1, s * 0.04, pt.tone(depth, 0.8), dry=0.2)
        mx = cx + flip * L * 0.2
        mt = (mx, cy - hh - s * 0.45)
        pt.stroke(line((mx, cy - hh), mt, 8), max(2 * sc, s * 0.014), pt.tone(depth, 0.9), dry=pt.dry(depth, 0.8),
                  taper_start=0.03, taper_end=0.3, tip=0.3)
        end = (cx + flip * L * 0.62, cy - hh * 0.2)
        pt.stroke(line(mt, end, 14, sag=flip * s * 0.06), max(1.3 * sc, s * 0.005), pt.tone(depth, 0.6), dry=0.5,
                  wobble=0.2, taper_start=0.05, taper_end=0.3, tip=0.2)
    else:
        # Rowing boat: an oar dipping into the water, a hunched rower as two touches.
        ox = cx + flip * L * 0.05
        pt.stroke(line((ox - flip * L * 0.08, cy - hh * 1.9), (ox + flip * L * 0.3, cy + hh * 0.5), 8), s * 0.016,
                  pt.tone(depth, 0.85), dry=pt.dry(depth, 0.9), taper_start=0.05, taper_end=0.2, tip=0.3)
        pt.dab(ox - flip * L * 0.05, cy - hh * 2.0, s * 0.07, pt.tone(depth, 0.85), angle=np.pi / 2, dry=0.3, elong=1.6)
        pt.dab(ox - flip * L * 0.05, cy - hh * 3.1, s * 0.05, pt.tone(depth, 0.9), dry=0.2, elong=1.0)


def _draw_bird(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    if (sh.obj.variant or "flying") == "perched":
        return _draw_perched(pt, sh)
    part = sh.parts[0]
    for xs, ys in _components(part.sdf < 0, min_px=6):
        x0, x1 = xs.min(), xs.max()
        wdt = x1 - x0
        mid = np.abs(xs - (x0 + x1) / 2) < wdt * 0.08
        body = (float(xs[mid].mean()), float(ys[mid].max())) if mid.any() else ((x0 + x1) / 2, ys.max())
        left, right = xs < body[0] - wdt * 0.1, xs > body[0] + wdt * 0.1
        if not left.any() or not right.any():
            continue
        tipL = (float(x0), float(ys[xs <= x0 + wdt * 0.04].mean()))
        tipR = (float(x1), float(ys[xs >= x1 - wdt * 0.04].mean()))
        iL, iR = np.argmin(np.where(left, ys, np.inf)), np.argmin(np.where(right, ys, np.inf))
        elL, elR = (float(xs[iL]), float(ys[iL])), (float(xs[iR]), float(ys[iR]))
        wd = max(2.2 * sc, wdt * 0.075)
        for el, tip in ((elL, tipL), (elR, tipR)):
            path = bezier(body, el, tip, n=12)
            pt.stroke(path, wd, pt.tone(depth, 0.9), pressure=[0.8, 1.0, 0.7, 0.4], dry=pt.dry(depth, 0.7),
                      taper_start=0.15, taper_end=0.75, tip=0.03, wobble=0.3)


def _draw_perched(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    tone = lambda b: pt.tone(depth, b)  # noqa: E731
    X, Y = pt.f.grid
    # Pale belly wash, then the back and wing in two broad strokes.
    wet = pt.layer()
    belly = S.intersect(S.ellipse(X, Y, cx, cy, s * 0.28, s * 0.16), cy - s * 0.02 - Y)
    pt.wash_sdf(belly, tone(0.14), wet, jitter=1.5, ring_width=3)
    back = np.array([(cx + s * 0.2, cy - s * 0.15), (cx + s * 0.02, cy - s * 0.17), (cx - s * 0.2, cy - s * 0.09)])
    pt.stroke(bezier(*back, n=10), s * 0.13, tone(0.55), layer=wet, dry=pt.dry(depth, 0.6), taper_start=0.2,
              taper_end=0.5, tip=0.2)
    pt.commit(wet, pt.bleed_px(depth, 0.8))
    pt.stroke(bezier((cx + s * 0.13, cy - s * 0.09), (cx - s * 0.03, cy - s * 0.02), (cx - s * 0.27, cy - s * 0.02), n=10),
              s * 0.12, tone(0.9), dry=pt.dry(depth, 1.0), taper_start=0.15, taper_end=0.6, tip=0.05)
    # Head: one loaded dab, the eye left with a ring of paper around a dark pupil.
    hx, hy = cx + s * 0.24, cy - s * 0.14
    head = pt.layer()
    pt.stroke(line((hx - s * 0.05, hy + s * 0.01), (hx + s * 0.05, hy - s * 0.01), 4), s * 0.19, tone(0.8), layer=head,
              dry=0.25, taper_start=0.3, taper_end=0.35, tip=0.45, wobble=0.2)
    ex, ey = cx + s * 0.27, cy - s * 0.16
    ring = np.hypot(X - ex, Y - ey)
    head *= 1.0 - 0.95 * np.clip((s * 0.042 - ring) / (1.0 * sc + 0.5), 0, 1)
    pt.commit(head, 0.8 * sc)
    pt.dab(ex, ey, max(2.5 * sc, s * 0.028), tone(1.0), dry=0.0, elong=1.0)
    # Beak: two short strokes meeting at the tip.
    tip_ = (cx + s * 0.45, cy - s * 0.13)
    for by in (-0.17, -0.1):
        pt.stroke(line((cx + s * 0.31, cy + s * by), tip_, 4), s * 0.03, tone(0.95), dry=0.3, taper_start=0.05,
                  taper_end=0.6, tip=0.05, wobble=0.1)
    # Belly line and tail.
    pt.stroke(arc(cx, cy, s * 0.28, s * 0.16, 0.25, 2.6, 16), s * 0.02, tone(0.55), dry=pt.dry(depth, 1.2),
              taper_start=0.2, taper_end=0.5, tip=0.05)
    for tip2 in ((cx - s * 0.55, cy + s * 0.06), (cx - s * 0.5, cy + s * 0.15)):
        pt.stroke(line((cx - s * 0.2, cy - s * 0.01), tip2, 6), s * 0.075, tone(0.9), dry=pt.dry(depth, 0.9),
                  taper_start=0.1, taper_end=0.6, tip=0.05)
    # Legs and a twig to stand on.
    for lx0, lx1 in ((0.0, -0.02), (0.08, 0.07)):
        pt.stroke(line((cx + s * lx0, cy + s * 0.12), (cx + s * lx1, cy + s * 0.26), 4), max(1.8 * sc, s * 0.014),
                  tone(0.95), dry=0.4, taper_start=0.05, taper_end=0.3, tip=0.3, wobble=0.1)
    pt.stroke(line((cx - s * 0.45, cy + s * 0.3), (cx + s * 0.5, cy + s * 0.23), 10, sag=s * 0.02), s * 0.03, tone(0.7),
              dry=pt.dry(depth, 1.3), taper_start=0.1, taper_end=0.7, tip=0.05)


def _ground_shadow(pt: Painter, cx, cy, rx, ry, depth):
    X, Y = pt.f.grid
    lx, _ = pt.ctx.light_dir
    wet = pt.layer()
    pt.wash_sdf(S.ellipse(X, Y, cx - lx * rx * 0.25, cy, rx, ry), pt.tone(depth, pt.P["wash_tone"] * 0.5), wet,
                jitter=2.0, ring_width=3)
    pt.commit(wet, pt.bleed_px(depth, 1.2))


def _draw_vase(pt: Painter, sh):
    body, band = sh.part("body"), sh.part("band")
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    sx = _shadow_x(pt)
    top = cy - s
    _ground_shadow(pt, cx, cy, s * 0.36, s * 0.045, depth)
    # Body: one wet wash of the silhouette (pooled rim), then a loaded stroke down the shadow side
    # following the profile, and a dry contour on that edge.
    wet = pt.layer()
    pt.wash_sdf(body.sdf, pt.tone(depth, pt.P["wash_tone"] * 0.95), wet, jitter=2.0, ring_width=5)
    us = (0.62 * sx, 0.1 * sx)
    paths, hw = _profile_paths(body.sdf, cy - s * 0.02, top + s * 0.12, us)
    if paths:
        width = 2 * float(hw.max()) * 0.36
        for k, p in enumerate(paths):
            pt.stroke(p, width, pt.tone(depth, 0.5 - 0.22 * k), layer=wet, pressure=np.clip(hw / hw.max(), 0.3, 1),
                      dry=pt.dry(depth, 0.9 + 0.4 * k), taper_start=0.06, taper_end=0.35, tip=0.05, mode="max")
    pt.commit(wet, pt.bleed_px(depth, 0.7))
    if paths:
        edge = paths[0].copy()
        c = edge[:, 0] - 0.62 * sx * hw
        edge[:, 0] = c + sx * hw
        pt.stroke(edge[::-1], s * 0.02, pt.tone(depth, 0.9), dry=pt.dry(depth, 1.3), taper_start=0.08, taper_end=0.3)
    # Lip ellipse and a dark band.
    pt.stroke(arc(cx, top + s * 0.01, s * 0.16, s * 0.025, 0.1, np.pi - 0.1, 14), s * 0.022, pt.tone(depth, 0.95),
              dry=0.5, taper_start=0.1, taper_end=0.2)
    pt.stroke(arc(cx, top + s * 0.01, s * 0.16, s * 0.025, np.pi + 0.2, 2 * np.pi - 0.2, 14), s * 0.014,
              pt.tone(depth, 0.6), dry=0.8, taper_start=0.2, taper_end=0.3)
    by = top + s * 0.45
    xl, xr = row_extents(band.sdf, [by])
    if not np.isnan(xl[0]):
        pt.stroke(line((xl[0] + 2, by), (xr[0] - 2, by), 10, sag=s * 0.025), s * 0.035, pt.tone(depth, 0.85),
                  dry=pt.dry(depth, 1.4), taper_start=0.15, taper_end=0.3, tip=0.1)


def _fruit_centres(bodies_sdf, sh, variant):
    o = sh.obj
    s = sh.radius * 2
    n = max(1, o.count)
    cx0, cy0 = sh.cx, sh.cy
    out = []
    for k in range(n):
        nx = cx0 + (k - (n - 1) / 2) * s * 0.85
        ny = cy0 + (abs(k - (n - 1) / 2) * s * 0.06 if n > 1 else 0) - s * 0.5
        if variant == "pear":
            ny += s * 0.5 * 0.25
        # Refine: the deepest point of the silhouette near the nominal centre.
        R = int(s * 0.12) + 1
        y0, y1 = int(max(0, ny - R)), int(min(bodies_sdf.shape[0], ny + R))
        x0, x1 = int(max(0, nx - R)), int(min(bodies_sdf.shape[1], nx + R))
        win = bodies_sdf[y0:y1, x0:x1]
        if win.size == 0:
            continue
        j = np.unravel_index(np.argmin(win), win.shape)
        fx, fy, rad = x0 + j[1] + 0.5, y0 + j[0] + 0.5, float(-win[j])
        if rad <= 1:
            continue
        r = rad / 0.78 if variant == "pear" else (rad / 0.92 if variant == "orange" else rad)
        if variant == "pear":
            fy -= r * 0.25
        out.append((fx, fy, min(r, s * 0.55)))
    return out


def _draw_fruit(pt: Painter, sh):
    bodies = sh.part("fruit")
    s, depth, sc = sh.radius * 2, sh.depth, pt.sc
    sx = _shadow_x(pt)
    variant = sh.obj.variant or "apple"
    fr = _fruit_centres(bodies.sdf, sh, variant)
    mid = (len(fr) - 1) / 2
    order = sorted(range(len(fr)), key=lambda i: -abs(i - mid))[::-1]  # middle (farthest) first
    X, Y = pt.f.grid
    for i in order:
        fx, fy, r = fr[i]
        _ground_shadow(pt, fx, fy + r * (1.0 if variant != "pear" else 1.25), r * 0.8, r * 0.14, depth)
        own = S.circle(X, Y, fx, fy + (0.25 * r if variant == "pear" else 0), r * 1.02)
        pt.reserve(sdf_to_mask(own), pt.P["reserve"])
        wet = pt.layer()
        local = np.maximum(bodies.sdf, np.hypot(X - fx, Y - fy - (0.25 * r if variant == "pear" else 0)) - r * 1.05)
        pt.wash_sdf(local, pt.tone(depth, pt.P["wash_tone"] * 1.1), wet, jitter=1.5, ring_width=4)
        # One loaded crescent stroke wrapping the shadow side, from the stem round to the base.
        cy_b = fy + (0.25 * r if variant == "pear" else 0.0)
        rb = r * (0.78 if variant == "pear" else 1.0)
        R = rb * 0.62
        if sx > 0:
            a0, a1 = -0.5 * np.pi + 0.35, 0.5 * np.pi + 0.2
        else:
            a0, a1 = 1.5 * np.pi - 0.35, 0.5 * np.pi - 0.2
        pt.stroke(arc(fx, cy_b, R, R, a0, a1, 18), rb * 0.62, pt.tone(depth, 0.62), layer=wet,
                  pressure=[0.4, 1.0, 1.0, 0.5], dry=pt.dry(depth, 0.8), taper_start=0.25, taper_end=0.4, tip=0.05,
                  mode="max")
        pt.commit(wet, pt.bleed_px(depth, 0.7))
        st_top = fy - r * (1.0 if variant == "pear" else (0.92 if variant == "orange" else 0.85))
        if variant == "pear":
            st_top = fy - r * 0.55 - r * 0.45 + r * 0.08
        if variant != "orange":
            pt.stroke(arc(fx, st_top + r * 0.08, r * 0.14, r * 0.06, 0.2, np.pi - 0.2, 8), r * 0.07, pt.tone(depth, 0.9),
                      dry=0.4, taper_start=0.2, taper_end=0.3)
        pt.stroke(line((fx, st_top + r * 0.05), (fx + r * 0.1, st_top - r * 0.28), 5), max(2 * sc, r * 0.08),
                  pt.tone(depth, 0.95), dry=0.4, taper_start=0.05, taper_end=0.3, tip=0.4)
        if variant != "orange":
            lx, ly = fx + r * 0.1, st_top - r * 0.2
            pt.stroke(line((lx, ly), (lx + r * 0.5, ly - r * 0.14), 6, sag=-r * 0.05), r * 0.24, pt.tone(depth, 0.75),
                      dry=pt.dry(depth, 0.5), taper_start=0.2, taper_end=0.7, tip=0.02)


def _draw_teapot(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    sx = _shadow_x(pt)
    bw, bh = s * 0.36, s * 0.27
    by = cy - bh
    flat = by - bh * 0.72
    _ground_shadow(pt, cx, cy, s * 0.45, s * 0.045, depth)
    wet = pt.layer()
    body_part, lid_part = sh.part("body"), sh.part("lid")
    # Belly, spout and handle: one wet wash of the silhouette; lid a paler one.
    pt.wash_sdf(body_part.sdf, pt.tone(depth, pt.P["wash_tone"] * 0.9), wet, jitter=1.5, ring_width=5)
    pt.wash_sdf(lid_part.sdf, pt.tone(depth, pt.P["wash_tone"] * 0.55), wet, jitter=1.0, ring_width=3)
    # One loaded crescent wrapping the shadow side of the belly.
    if sx > 0:
        a0, a1 = math.radians(-70), math.radians(80)
    else:
        a0, a1 = math.radians(250), math.radians(100)
    pt.stroke(arc(cx, by + bh * 0.05, bw * 0.72, bh * 0.62, a0, a1, 20), bh * 0.6, pt.tone(depth, 0.5), layer=wet,
              pressure=[0.35, 1.0, 1.0, 0.6], dry=pt.dry(depth, 0.9), taper_start=0.25, taper_end=0.35, tip=0.05,
              mode="max")
    # Spout: one stroke from the body out to the lip, narrowing.
    sp = np.array(line((cx + bw * 0.7, by + bh * 0.25), (cx + bw * 1.2, by - bh * 0.6), 8, sag=bw * 0.05).tolist()
                  + line((cx + bw * 1.2, by - bh * 0.6), (cx + bw * 1.38, by - bh * 0.8), 4).tolist()[1:])
    pt.stroke(sp, s * 0.1, pt.tone(depth, 0.45), layer=wet, pressure=[1.0, 0.6, 0.42, 0.36], dry=pt.dry(depth, 1.0),
              taper_start=0.03, taper_end=0.08, tip=0.8, mode="max")
    pt.commit(wet, pt.bleed_px(depth, 0.7))
    # Handle: a single loop.
    pt.stroke(arc(cx - bw * 0.95, by - bh * 0.05, s * 0.15, s * 0.17, math.radians(-75), math.radians(-285), 20),
              s * 0.06, pt.tone(depth, 0.8), dry=pt.dry(depth, 0.9), taper_start=0.1, taper_end=0.2, tip=0.4)
    # Contour on the shadow side of the belly.
    a0, a1 = (math.radians(-60), math.radians(85)) if sx > 0 else (math.radians(240), math.radians(95))
    pt.stroke(arc(cx, by, bw, bh, a0, a1, 18), s * 0.025, pt.tone(depth, 0.9), dry=pt.dry(depth, 1.3),
              taper_start=0.1, taper_end=0.3)
    # Rim, lid dome, knob, foot.
    pt.stroke(line((cx - bw * 0.6, flat), (cx + bw * 0.6, flat), 8), s * 0.03, pt.tone(depth, 0.95), dry=0.5,
              taper_start=0.08, taper_end=0.15, tip=0.3)
    pt.stroke(arc(cx, flat, bw * 0.46, bh * 0.3, np.pi + 0.15, 2 * np.pi - 0.15, 14), s * 0.035, pt.tone(depth, 0.7),
              dry=pt.dry(depth, 0.8), taper_start=0.1, taper_end=0.2)
    pt.dab(cx, flat - bh * 0.38, s * 0.075, pt.tone(depth, 0.95), dry=0.1, elong=1.0)
    pt.stroke(line((cx - bw * 0.55, cy - s * 0.02), (cx + bw * 0.55, cy - s * 0.02), 8), s * 0.035, pt.tone(depth, 0.8),
              dry=pt.dry(depth, 0.9), taper_start=0.1, taper_end=0.2, tip=0.4)


def _draw_cup(pt: Painter, sh):
    s, cx, cy, depth, sc = sh.radius * 2, sh.cx, sh.cy, sh.depth, pt.sc
    sx = _shadow_x(pt)
    w_top, w_bot, h = s * 0.3, s * 0.22, s * 0.42
    X, Y = pt.f.grid
    wet = pt.layer()
    pt.wash_sdf(S.ellipse(X, Y, cx, cy, s * 0.5, s * 0.07), pt.tone(depth, pt.P["wash_tone"] * 0.45), wet, jitter=1.5,
                ring_width=3)
    body = S.polygon(X, Y, [(cx - w_top, cy - h), (cx + w_top, cy - h), (cx + w_bot, cy - s * 0.04), (cx - w_bot, cy - s * 0.04)])
    pt.wash_sdf(S.intersect(body, -sx * (X - cx - sx * w_top * 0.1)), pt.tone(depth, pt.P["wash_tone"] * 0.6), wet,
                jitter=1.0, ring_width=3)
    pt.wash_sdf(S.ellipse(X, Y, cx, cy - h + s * 0.005, w_top * 0.9, s * 0.035), pt.tone(depth, 0.6), wet, jitter=0.5,
                ring_width=2)
    pt.commit(wet, pt.bleed_px(depth, 0.6))
    # Saucer: front arc strong, back arcs peeking out either side of the cup.
    pt.stroke(arc(cx, cy, s * 0.5, s * 0.07, 0.08, np.pi - 0.08, 20), s * 0.024, pt.tone(depth, 0.85),
              dry=pt.dry(depth, 1.0), taper_start=0.1, taper_end=0.2)
    for a0, a1 in ((np.pi + 0.05, np.pi + 0.6), (2 * np.pi - 0.6, 2 * np.pi - 0.05)):
        pt.stroke(arc(cx, cy, s * 0.5, s * 0.07, a0, a1, 8), s * 0.014, pt.tone(depth, 0.55), dry=0.8,
                  taper_start=0.2, taper_end=0.3)
    # Sides, rim, handle.
    for side in (-1, 1):
        heavy = side == sx
        p = line((cx + side * w_top, cy - h), (cx + side * w_bot, cy - s * 0.04), 8, sag=-side * s * 0.01)
        pt.stroke(p, s * (0.03 if heavy else 0.02), pt.tone(depth, 0.9 if heavy else 0.65), dry=pt.dry(depth, 1.0),
                  taper_start=0.05, taper_end=0.2)
    pt.stroke(arc(cx, cy - h, w_top, s * 0.05, 0.0, 2 * np.pi, 30), s * 0.018, pt.tone(depth, 0.85), dry=0.6,
              taper_start=0.05, taper_end=0.1, tip=0.4)
    pt.stroke(arc(cx + w_top, cy - h * 0.55, s * 0.12, s * 0.12, math.radians(-110), math.radians(110), 16), s * 0.05,
              pt.tone(depth, 0.8), dry=pt.dry(depth, 0.9), taper_start=0.1, taper_end=0.2, tip=0.4)
    pt.stroke(line((cx - w_bot * 0.8, cy - s * 0.012), (cx + w_bot * 0.8, cy - s * 0.012), 6), s * 0.02,
              pt.tone(depth, 0.7), dry=pt.dry(depth, 1.0), taper_start=0.1, taper_end=0.2)


def _draw_rain(pt: Painter, sh):
    part = sh.parts[0]
    sc = pt.sc
    rr = pt.r("rain")
    keep = 1.0 - 0.4 * pt.P["economy"]
    for xs, ys in _components(part.sdf < 0, min_px=4):
        if rr.random() > keep:
            continue
        i0, i1 = np.argmin(ys), np.argmax(ys)
        pt.stroke(line((xs[i0], ys[i0]), (xs[i1], ys[i1]), 4), 1.8 * sc, pt.tone(0.5, rr.uniform(0.25, 0.45)),
                  dry=0.9, wobble=0.0, taper_start=0.3, taper_end=0.5, tip=0.05)


DRAWERS = {
    "sun": _draw_sun, "moon": _draw_moon, "stars": _draw_stars, "cloud": _draw_cloud,
    "mountain": _draw_mountain, "hills": _draw_hills, "sea": _draw_sea, "field": _draw_field,
    "river": _draw_river, "table": _draw_table, "cliff": _draw_cliff,
    "tree": _draw_tree, "pine": _draw_pine, "cypress": _draw_cypress, "bamboo": _draw_bamboo, "reeds": _draw_reeds,
    "house": _draw_house, "lighthouse": _draw_lighthouse, "windmill": _draw_windmill,
    "boat": _draw_boat, "bird": _draw_bird,
    "vase": _draw_vase, "fruit": _draw_fruit, "teapot": _draw_teapot, "cup": _draw_cup,
    "rain": _draw_rain,
}


# ---------------------------------------------------------------------------
# Seal
# ---------------------------------------------------------------------------

def _seal(pt: Painter):
    """A small red chop with an abstract carved glyph (lines on a 3x3 lattice, no real text)."""
    H, W, sc = pt.H, pt.W, pt.sc
    size = pt.P["seal_size"] * H
    m = 0.045 * min(H, W)
    cands = [(W - m - size / 2, H - m - size / 2), (W - m - size / 2, H * 0.7), (m + size / 2, H - m - size / 2),
             (W - m - size / 2, m + size), (m + size / 2, m + size)]
    best, best_v = cands[0], np.inf
    for x, y in cands:
        y0, y1 = int(max(0, y - size)), int(min(H, y + size))
        x0, x1 = int(max(0, x - size)), int(min(W, x + size))
        v = float(pt.D[y0:y1, x0:x1].mean()) + 0.5 * float(pt.D[y0:y1, x0:x1].max())
        if v < 0.06:
            best = (x, y)
            break
        if v < best_v:
            best, best_v = (x, y), v
    x, y = best
    rr = pt.r("seal")
    hw = size / 2
    pad = int(size * 0.2) + 2
    y0, y1 = int(max(0, y - hw - pad)), int(min(H, y + hw + pad))
    x0, x1 = int(max(0, x - hw - pad)), int(min(W, x + hw + pad))
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    rough = (pt.edge_noise[y0:y1, x0:x1] - 0.5) * 2.5 * sc
    body = S.box(X, Y, x, y, hw, hw * 1.08, radius=size * 0.06) + rough
    inner = np.abs(S.box(X, Y, x, y, hw * 0.84, hw * 0.92)) - size * 0.028   # carved border line
    # Glyph: a random connected set of bars on a 3x3 lattice.
    g = np.linspace(-0.5, 0.5, 3) * size * 0.62
    edges = [((i, j), (i + 1, j)) for i in range(2) for j in range(3)] + [((i, j), (i, j + 1)) for i in range(3) for j in range(2)]
    picks = rr.permutation(len(edges))[: int(rr.integers(4, 7))]
    glyph = np.full(X.shape, np.inf, np.float32)
    bw = size * 0.05
    for e in picks:
        (a, b), (c, d) = edges[e]
        glyph = np.minimum(glyph, S.segment(X, Y, x + g[a], y + g[b] * 1.08, x + g[c], y + g[d] * 1.08, bw))
    glyph = np.minimum(glyph, S.circle(X, Y, x + g[int(rr.integers(0, 3))], y + g[int(rr.integers(0, 3))] * 1.08, bw * 1.3))
    cover = sdf_to_mask(body, 1.2) * (1 - sdf_to_mask(inner + rough * 0.4, 1.0)) * (1 - sdf_to_mask(glyph + rough * 0.4, 1.0))
    # Stamp texture: uneven paste and paper fibres showing through.
    tex = 0.72 + 0.28 * pt.grain[y0:y1, x0:x1]
    holes = np.clip((pt.mottle[y0:y1, x0:x1] - 0.72) * 6, 0, 0.6)
    over(pt.A[y0:y1, x0:x1], cover * tex * (1 - holes) * 0.92)
    return (round(float(x), 1), round(float(y), 1), round(float(size), 1))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def render_array(scene, seed=None, params=None):
    params = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, params, NAME)
    pt = Painter(ctx)
    kinds = [s.kind for s in ctx.shapes]
    accent_on = bool(params["accent"])
    ctx.info["accent"] = ("sun" if "sun" in kinds else "seal") if accent_on else None
    for si, shape in enumerate(ctx.shapes):
        pt.begin(si, shape.kind)
        k = SOLID_RESERVE.get(shape.kind, 0.0) * params["reserve"]
        if k > 0 and shape.kind != "fruit":
            m = ndimage.gaussian_filter(shape.mask, 1.5 * pt.sc)
            pt.reserve(m, k)
        DRAWERS[shape.kind](pt, shape)
    if ctx.info["accent"] == "seal":
        ctx.info["seal"] = _seal(pt)
    img, Dt, A = pt.image()
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    ctx.info.update({
        "accent_used": bool(ctx.info["accent"]) and float(A.max()) > 0.05,
        "accent_fraction": round(float((A > 0.1).mean()), 4),
        "strokes": pt.n_strokes,
        "washes": pt.n_washes,
        "strokes_by_kind": dict(sorted(pt.kind_strokes.items())),
        "empty_fraction": round(float(((Dt < 0.06) & (A < 0.05)).mean()), 4),
        "mean_saturation": round(float(sat.mean()), 4),
        "ink_coverage": round(float(Dt.mean()), 4),
    })
    return img, ctx


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)
