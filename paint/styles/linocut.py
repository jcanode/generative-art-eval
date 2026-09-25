"""Linocut relief print.

Printing model: a single lino block starts fully inked (1 = standing surface).
Every tone is made by *carving it away*: rows of V-gouge cuts laid along a
per-object phase field (contours of the silhouette, cross-contour bands on
turned forms, rays round the sun, perspective furrows on fields, waves on
water), with the cut width driven by a tone field - wide cuts on the lit side,
none in the core shadow. Individually stamped gouge marks add chips, leaf
flicks, grass and stars. Objects are carved far to near; each silhouette gets
either a carved white separation line (dark on dark) or a black keyline (light
shapes). Finally the plate edge is cut by hand, stray ridges and chips are
left in, and the block is printed: uneven roller density, paper grain showing
through the solids, fibres in the paper. An optional second block prints one
flat colour under the black key.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core import linocut_kit as K
from ..core.color import hex_to_rgb, luminance
from ..core.masks import sdf_to_mask
from ..core.noise import blurred_white, fbm, value_noise
from ..core.texture import paper
from ..scene.shapes import Part, shade_field
from .base import Context, clamp_params, encode, make_context, merge_params, scene_hints

NAME = "linocut"

DEFAULTS = {
    "cut_spacing": 7.5,      # carved line pitch in px at 1024
    "tone_contrast": 0.8,    # how strongly light direction opens up the cuts
    "cut_taper": 0.7,        # how broken / pointed the cut dashes are
    "line_wobble": 0.5,      # hand wobble of the cut rows
    "sky_cut": 0.6,          # how much of a daytime sky is carved away
    "outline": 1.0,          # width of keylines / carved separation lines
    "chip_density": 0.7,     # leaf flicks, stipple chips, grass
    "edge_ragged": 0.8,      # fraying of every cut edge
    "ink_texture": 0.6,      # roller unevenness and paper grain in the solids
    "plate_margin": 0.028,   # blank paper round the plate (fraction of height)
    "chatter": 0.5,          # stray ridges left standing in cleared areas
    "second_ink": 0,         # 1 = print a flat colour block under the black key
    "ink": "#161412",
    "paper": "#f1ebdf",
    "color": None,           # second-ink colour (hex); None = warmest palette hint
}

KNOBS = {
    "cut_spacing": (5.0, 12.0, "pitch of the carved lines; bigger reads bolder and cruder"),
    "tone_contrast": (0.3, 1.0, "how much the lit side is carved open vs the shadow side"),
    "cut_taper": (0.0, 1.0, "0 = long ruled cuts, 1 = short pointed gouge dashes"),
    "line_wobble": (0.0, 1.0, "hand wobble in the cut rows"),
    "sky_cut": (0.2, 1.0, "how much of a day sky is cleared to paper"),
    "outline": (0.0, 2.0, "keyline / carved separation line width"),
    "chip_density": (0.0, 1.0, "density of chips, leaf flicks and grass cuts"),
    "edge_ragged": (0.0, 1.5, "fraying of cut edges"),
    "ink_texture": (0.0, 1.0, "roller unevenness and paper grain in solid blacks"),
    "plate_margin": (0.0, 0.06, "paper margin round the plate"),
    "chatter": (0.0, 1.0, "stray uncut ridges in cleared areas"),
    "second_ink": (0, 1, "1 = add a second (colour) block"),
}

RULES = {"max_inks": 2, "max_mean_saturation": 0.2}


# ---------------------------------------------------------------------------
# Render state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.f = ctx.frame
        self.P = ctx.params
        self.sc = self.f.scale
        self.H, self.W = self.f.shape
        self.X, self.Y = self.f.grid
        self.ink = np.ones(self.f.shape, np.float32)
        self.color = np.zeros(self.f.shape, np.float32)
        r = ctx.rng
        self.noise = K.WrapNoise(r.child("dash"), cell=44 * self.sc)
        self.rough = (value_noise(self.f.shape, 2.2 * self.sc, r.child("rough")) - 0.5) * 1.6 * self.P["edge_ragged"]
        self.wob = (value_noise(self.f.shape, 90 * self.sc, r.child("wob")) - 0.5) * 0.9 * self.P["line_wobble"] \
            + (value_noise(self.f.shape, 23 * self.sc, r.child("wob2")) - 0.5) * 0.25 * self.P["line_wobble"]
        self.silh = (value_noise(self.f.shape, 7 * self.sc, r.child("silh")) - 0.5) * 2.0 * self.sc
        sc = ctx.scene
        self.hz = sc.horizon * self.H
        self.night = sc.palette.mood == "night"
        kinds = {s.kind for s in ctx.shapes}
        self.interior = "table" in kinds and not ({"sea", "field", "river", "sun", "moon"} & kinds)
        self.lights = [s for s in ctx.shapes if s.kind in ("sun", "moon")]
        self.pitch = self.P["cut_spacing"] * self.sc
        # Second block: accent parts and suns; failing those, the main subject's body.
        self.color_target = None
        if not any(p.role == "accent" or s.kind == "sun" for s in ctx.shapes for p in s.parts
                   if s.kind != "lighthouse"):
            objs = [s for s in ctx.shapes if not s.region and s.kind not in ("rain", "stars", "bird")]
            if objs:
                big = max(objs, key=lambda s: float(s.mask.sum()))
                self.color_target = big.parts[0]
        self.salt = 0.0
        self.dim = False
        self.lift = False
        self.marks = 0

    def px(self, v):
        return v * self.sc

    def crop(self, part: Part, pad: float = 14.0):
        x0, y0, x1, y1 = part.stats()[3]
        p = int(np.ceil(pad * self.sc)) + 4
        x0, y0 = max(0, x0 - p), max(0, y0 - p)
        x1, y1 = min(self.W, x1 + p), min(self.H, y1 + p)
        return (slice(y0, y1), slice(x0, x1))

    def screen(self, sl, phase, tone, px=None, taper=None):
        self.salt += 1.0
        # Push tones apart: a relief print wants big blacks and big whites, few mid greys.
        tone = np.clip(0.5 + (np.asarray(tone, np.float32) - 0.5) * 1.2, 0, 1)
        if self.night and self.dim:
            tone = tone * 0.55
        if self.lift:
            tone = 1.0 - (1.0 - tone) * 0.3
        return K.line_screen(phase + self.wob[sl], tone, self.X[sl], self.Y[sl], self.noise, px=px,
                             taper=self.P["cut_taper"] if taper is None else taper,
                             rough=self.rough[sl], salt=self.salt)

    def contour(self, sl, sdf, tone, pitch=None, taper=None, offset=0.0):
        p = pitch or self.pitch
        s = ndimage.gaussian_filter(sdf, 1.2 * self.sc)
        return self.screen(sl, -s / p + offset, tone, px=p, taper=taper)

    def parallel(self, sl, angle, tone, pitch=None, taper=None):
        p = pitch or self.pitch
        c, s = np.cos(angle), np.sin(angle)
        ph = (-self.X[sl] * s + self.Y[sl] * c) / p
        return self.screen(sl, ph, tone, px=p, taper=taper)

    def cross(self, sl, cx, rx, tone, bow=0.14, pitch=None, taper=None):
        """Cross-contour bands on a turned form (ellipse sections seen from slightly above)."""
        p = pitch or self.pitch
        u = np.clip((self.X[sl] - cx) / max(rx, 1.0), -1, 1)
        ph = (self.Y[sl] - bow * rx * np.sqrt(1 - u * u)) / p
        return self.screen(sl, ph, tone, px=p, taper=taper)

    def turned(self, sl, sdf, tone, bow=0.3, pitch=None, taper=None):
        """Cross-contour bands on a solid of revolution, using each row's own width."""
        p = pitch or self.pitch
        inside = sdf < 0
        h, w = sdf.shape
        xs = np.arange(w, dtype=np.float32)[None, :]
        cnt = inside.sum(1)
        ok = cnt > 0
        lo = np.where(inside, xs, 1e9).min(1)
        hi = np.where(inside, xs, -1e9).max(1)
        lo, hi = np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)
        if not ok.any():
            return np.ones_like(sdf)
        rows = np.arange(h)
        cx = np.interp(rows, rows[ok], ((lo + hi) / 2)[ok])
        rx = np.interp(rows, rows[ok], ((hi - lo) / 2 + 1)[ok])
        cx = ndimage.gaussian_filter1d(cx, 3)[:, None]
        rx = np.maximum(ndimage.gaussian_filter1d(rx, 3), 2.0)[:, None]
        u = np.clip((xs - cx) / rx, -1, 1)
        ph = (self.Y[sl] - bow * rx * np.sqrt(1 - u * u)) / p
        return self.screen(sl, ph, tone, px=p, taper=taper)

    def shade(self, part, softness=1.0):
        return shade_field(part, self.ctx.light_dir, self.f, softness)

    def gouge(self, xs, ys, angs, lens, wids, value=0.0, curve=None, clip=None, label="g", **kw):
        n = K.gouge(self.ink, xs, ys, angs, lens, wids, value=value, curve=curve, clip=clip,
                    rng=self.ctx.rng.child("gouge", label, str(self.marks)), **kw)
        self.marks += n
        return n

    def tangent(self, sdf, xs, ys, sl):
        """Contour tangent angle of a crop SDF at global points."""
        g = ndimage.gaussian_filter(sdf, 2.0 * self.sc)
        gy, gx = np.gradient(g)
        ix = np.clip((xs - sl[1].start).astype(int), 0, g.shape[1] - 1)
        iy = np.clip((ys - sl[0].start).astype(int), 0, g.shape[0] - 1)
        return np.arctan2(gy[iy, ix], gx[iy, ix]) + np.pi / 2


def _tone(base, gain, shade, contrast):
    return np.clip(base + gain * contrast * (1.0 - shade), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Background: sky / wall
# ---------------------------------------------------------------------------

def _background(st: _State):
    P, sc = st.P, st.sc
    X, Y = st.X, st.Y
    full = (slice(0, st.H), slice(0, st.W))
    v = np.clip(Y / max(st.hz, 1.0), 0, 1)
    if st.interior:
        # A dark wall carved in echo lines that ripple out from the still life, opening
        # into a light halo round the objects and closing to solid black at the edges.
        objs = [s for s in st.ctx.shapes if not s.region]
        d = np.full(st.f.shape, np.inf, np.float32)
        for s in objs:
            d = np.minimum(d, s.sdf)
        d = np.maximum(d, 0.0)
        d = ndimage.gaussian_filter(np.minimum(d, 400 * sc), 6 * sc)
        tone = np.clip(0.7 * np.exp(-d / (70 * sc)) + 0.04, 0, 1)
        p = st.pitch * 1.25
        st.ink[:] = st.screen(full, d / p - st.wob * 0.5, tone, px=p, taper=0.55)
        return
    if st.night:
        tone = 0.08 + 0.42 * v ** 3
    else:
        top = 0.35 + 0.5 * P["sky_cut"]
        tone = top + (min(0.97, top + 0.3) - top) * v ** 1.2
    pitch = st.pitch * 1.2
    ph = (Y + 3 * sc * np.sin(X / (140 * sc))) / pitch
    st.ink[:] = st.screen(full, ph, tone, px=pitch)
    for L in st.lights:
        if L.cy > st.hz + L.radius:
            continue
        R = L.radius
        r = np.hypot(X - L.cx, Y - L.cy)
        edge = (st.silh / sc) * R * 0.08
        if L.kind == "sun" and not st.night:
            # Sunburst: carved rays fanning out, the black wedges thinning to points.
            outer = R * 2.2
            zone = sdf_to_mask(r - outer + edge * 3)
            n = int(np.clip(2 * np.pi * R / (st.pitch * 2.2), 24, 56))
            ang = np.arctan2(Y - L.cy, X - L.cx)
            ph = ang * n / (2 * np.pi) + 0.25
            px = np.maximum(r * 2 * np.pi / n, 1.0)
            t = np.clip(0.42 + 0.5 * (r - R) / (outer - R), 0, 1)
            rays = st.screen(full, ph - st.wob * 0.6, t, px=px, taper=0.25)
            st.ink[:] = st.ink * (1 - zone) + rays * zone
        else:
            # Moon (or night sun): concentric rings, carved wider close to the disc.
            outer = R * 3.2
            zone = sdf_to_mask(r - outer + edge * 3)
            t = np.clip(tone + 0.6 * np.exp(-np.clip(r - R, 0, None) / (0.9 * R)), 0, 1)
            t = np.where(r < R, tone, t)  # the dark limb of a crescent stays night sky
            rings = st.screen(full, (r - R) / st.pitch, t, px=st.pitch)
            st.ink[:] = st.ink * (1 - zone) + rings * zone


# ---------------------------------------------------------------------------
# Treatments per kind. Each returns the ink pattern over the crop ``sl``.
# ---------------------------------------------------------------------------

def _white(st, sl):
    return np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), np.float32)


def _water(st, shape, part, sl, m):
    sc = st.sc
    X, Y = st.X[sl], st.Y[sl]
    Hh = max(st.H - st.hz, 1.0)
    dt = np.clip((Y - st.hz) / Hh, 0, 1)
    k = 3.0
    sp0 = st.pitch * 0.55
    amp = (1.0 + 7.0 * dt) * sc
    lam = (40 + 160 * dt) * sc
    wn = value_noise(st.f.shape, 70 * sc, st.ctx.rng.child("wave", shape.kind, part.name, str(int(shape.obj.x * 1000))))[sl]
    Yw = Y + amp * np.sin(2 * np.pi * X / lam + 5.0 * wn)
    dtw = np.clip((Yw - st.hz) / Hh, 0, 1)
    ph = Hh / (k * sp0) * np.log1p(k * dtw)
    px = sp0 * (1 + k * dtw)
    if st.night:
        t = 0.3 - 0.18 * dt
    else:
        t = 0.62 - 0.36 * dt
    glint = np.zeros_like(t)
    for L in st.lights:
        if L.cy > st.hz:
            continue
        wcol = L.radius * (0.7 + 1.6 * dt)
        col = np.exp(-((X - L.cx) / wcol) ** 2)
        glint = np.maximum(glint, col)
    t = np.clip(t + 0.55 * glint * (0.6 + 0.8 * (st.noise(X * 0.5, Y * 3.0) - 0.5)), 0, 1)
    pat = st.screen(sl, ph, t, px=px, taper=min(1.0, st.P["cut_taper"] + 0.2))
    # Glints: short horizontal flicks carved into the reflection column.
    if glint.max() > 0.2:
        r = st.ctx.rng.child("glint", str(st.salt))
        xs, ys = K.scatter(m * (glint > 0.3), 7 * sc, r, density=glint * 0.5, offset=(sl[1].start, sl[0].start))
        if len(xs):
            dd = np.clip((ys - st.hz) / Hh, 0, 1)
            sub = np.ones_like(pat)
            K.gouge(sub, xs - sl[1].start, ys - sl[0].start, r.normal(0, 0.05, len(xs)),
                    (8 + 22 * dd) * sc * r.uniform(0.6, 1.3, len(xs)), (1.6 + 2.5 * dd) * sc, rng=r.child("m"))
            pat = np.minimum(pat, sub)
            st.marks += len(xs)
    return pat


def _field(st, shape, part, sl, m):
    sc = st.sc
    X, Y = st.X[sl], st.Y[sl]
    Hh = max(st.H - st.hz, 1.0)
    dt = np.clip((Y - st.hz) / Hh, 0, 1)
    r = st.ctx.rng.child("field")
    vx = st.W * r.uniform(0.35, 0.65)
    vy = st.hz - 0.03 * st.H
    theta = np.arctan2(X - vx, np.maximum(Y - vy, 1.0))
    n = 40
    dist = np.hypot(X - vx, Y - vy)
    px0 = np.maximum(dist * np.pi / n, 0.5)
    # Carvers add furrows as they fan out: double the count wherever the spacing
    # gets too wide (old lines run on, new ones start between them).
    jit = (st.noise(X * 0.25 + 211, Y * 0.25) - 0.5) * 0.35
    lev = np.clip(np.floor(np.log2(np.maximum(px0 / (2.0 * st.pitch), 1.0)) + jit), 0, 3)
    mult = 2.0 ** lev
    ph = theta * n * mult / np.pi - st.wob[sl] * 0.5
    px = px0 / mult
    t = 0.26 + 0.12 * (1 - dt)
    if st.night:
        t = t * 0.6
    t = t * np.clip((px0 - 2.5 * sc) / (3.0 * sc), 0, 1)
    pat = st.screen(sl, ph, t, px=px, taper=0.35)
    # Grass tufts: upward flicks in the foreground, bigger towards the viewer.
    dens = st.P["chip_density"]
    if dens > 0:
        xs, ys = K.scatter(m * (dt > 0.12), 13 * sc, r.child("pts"), density=np.clip(dt * 0.9, 0, 1) * dens,
                           offset=(sl[1].start, sl[0].start))
        if len(xs):
            dd = np.clip((ys - st.hz) / Hh, 0, 1)
            sub = np.ones_like(pat)
            K.gouge(sub, xs - sl[1].start, ys - sl[0].start, -np.pi / 2 + r.normal(0, 0.25, len(xs)),
                    (7 + 16 * dd) * sc * r.uniform(0.7, 1.3, len(xs)), (2.0 + 2.2 * dd) * sc,
                    curve=r.normal(0, 0.3, len(xs)), rng=r.child("m"))
            pat = np.minimum(pat, sub)
            st.marks += len(xs)
    return pat


def _table(st, shape, part, sl, m):
    if part.name == "front":
        return st.parallel(sl, 0.0, 0.1, pitch=st.pitch * 1.1)
    X, Y = st.X[sl], st.Y[sl]
    top = shape.obj.y * st.H
    dt = np.clip((Y - top) / max(st.H * 0.25, 1), 0, 1)
    # Wood grain: long wavy cuts.
    g = value_noise(st.f.shape, 160 * st.sc, st.ctx.rng.child("grain"))[sl]
    ph = (Y + 14 * st.sc * np.sin(X / (90 * st.sc) + 4 * g)) / (st.pitch * 0.9)
    t = 0.72 - 0.22 * dt
    # Cast shadows: solid black pools on the side away from the light.
    lx, ly = st.ctx.light_dir
    shadow = np.zeros_like(X)
    for o in st.ctx.shapes:
        if o.region or o.obj.y <= shape.obj.y:
            continue
        x0, y0, x1, y1 = o.bbox()
        rw, hgt = (x1 - x0) / 2, (y1 - y0)
        cx = (x0 + x1) / 2 - np.sign(lx) * rw * 0.55
        ry = max(hgt * 0.07, 4 * st.sc)
        e = ((X - cx) / (rw * 1.15)) ** 2 + ((Y - (y1 - ry * 0.4)) / ry) ** 2
        shadow = np.maximum(shadow, np.clip((1.0 - e) * 6, 0, 1))
    t = t * (1 - shadow) + 0.03 * shadow
    return st.screen(sl, ph, t, px=st.pitch * 0.9, taper=0.3)


def _hills(st, shape, part, sl, m):
    k = int(part.name[-1]) if part.name[-1].isdigit() else 0
    n = max(1, shape.obj.count)
    far = (n - 1 - k) / max(n - 1, 1) if n > 1 else 0.0
    base = (0.3 if st.night else 0.24) + 0.3 * far
    sdf = part.sdf[sl]
    t = np.clip(base + 0.25 * np.exp(sdf / (40 * st.sc)), 0, 1)  # lighter along the ridge
    return st.contour(sl, sdf, t, pitch=st.pitch * (0.85 + 0.2 * (1 - far)))


def _ridge_phase(st, sl, sdf, pitch):
    """Strata lines that echo the ridge line: distance below each column's skyline."""
    inside = sdf < 0
    h, w = sdf.shape
    rows = np.arange(h, dtype=np.float32)[:, None]
    top = np.where(inside, rows, 1e9).min(0)
    ok = top < 1e8
    if not ok.any():
        return None
    cols = np.arange(w)
    top = np.interp(cols, cols[ok], top[ok])
    top = ndimage.gaussian_filter1d(top, 2.5 * st.sc)[None, :]
    return (rows + sl[0].start - (top + sl[0].start)) / pitch


def _mountain(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    rock = shape.part("rock") or part
    p = st.pitch * (1.1 if part.name == "snow" else 1.0)
    ph = _ridge_phase(st, sl, rock.sdf[sl], p)
    if part.name == "snow":
        t = np.clip(0.98 - 0.55 * sh * c, 0, 1)
    else:
        t = _tone(0.08, 0.8, sh, c)
    if ph is None:
        return st.contour(sl, part.sdf[sl], t, pitch=p)
    return st.screen(sl, ph, t, px=p)


def _cliff(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    t = _tone(0.08, 0.6, sh, st.P["tone_contrast"])
    side = -1 if shape.obj.x < 0.5 else 1
    pat = st.parallel(sl, np.radians(-18 * side), t, pitch=st.pitch * 1.05)
    return _chips(st, sl, m, pat, t, spacing=16, length=9, width=3.0, label="cliff")


def _chips(st, sl, m, pat, t, spacing=12, length=8, width=2.6, label="chips", ang=None, curve=0.0):
    dens = st.P["chip_density"]
    if dens <= 0:
        return pat
    sc = st.sc
    r = st.ctx.rng.child("chips", label, str(st.salt))
    xs, ys = K.scatter(m, spacing * sc, r, density=np.clip(t * 1.4, 0, 1) * dens, offset=(sl[1].start, sl[0].start))
    if len(xs) == 0:
        return pat
    a = r.uniform(0, np.pi, len(xs)) if ang is None else ang(xs, ys) + r.normal(0, 0.35, len(xs))
    sub = np.ones_like(pat)
    K.gouge(sub, xs - sl[1].start, ys - sl[0].start, a, length * sc * r.uniform(0.6, 1.4, len(xs)),
            width * sc * r.uniform(0.7, 1.2, len(xs)), curve=r.normal(curve, 0.2, len(xs)), rng=r.child("m"))
    st.marks += len(xs)
    return np.minimum(pat, np.maximum(sub, 1 - m))


def _tree(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    sdf = part.sdf[sl]
    if part.name == "trunk":
        t = _tone(0.04, 0.45, sh, c)
        return st.screen(sl, st.X[sl] / (st.pitch * 0.9) + 0.02 * st.Y[sl] / st.pitch, t, px=st.pitch * 0.9, taper=0.8)
    # Crown: a black mass; broken rim cuts on the lit side, leaf flicks carved into the light.
    t = _tone(0.0, 0.62, sh, c) ** 1.2
    rim = np.exp(np.minimum(sdf, 0) / (22 * st.sc))
    pat = st.contour(sl, sdf, t * rim, pitch=st.pitch * 1.1)
    return _chips(st, sl, m, pat, t, spacing=10, length=17, width=5.0, label="leaf",
                  ang=lambda xs, ys: st.tangent(sdf, xs, ys, sl), curve=0.5)


def _chevron(st, sl, cx, slope, tone, pitch=None):
    p = pitch or st.pitch
    X, Y = st.X[sl], st.Y[sl]
    ph = (Y - slope * np.abs(X - cx)) / p
    return st.screen(sl, ph, tone, px=p / np.sqrt(1 + slope * slope), taper=0.6)


def _pine(st, shape, part, sl, m):
    if part.name == "trunk":
        return np.ones_like(st.X[sl])
    sh = st.shade(part)[sl]
    t = _tone(0.03, 0.55, sh, st.P["tone_contrast"])
    return _chevron(st, sl, shape.cx, -0.55, t, pitch=st.pitch * 1.05)


def _cypress(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    t = _tone(0.03, 0.55, sh, st.P["tone_contrast"])
    return _chevron(st, sl, shape.cx, -1.1, t)


def _bamboo(st, shape, part, sl, m):
    sdf = part.sdf[sl]
    if part.name == "stalks":
        sh = st.shade(part, softness=2.0)[sl]
        t = np.clip(0.02 + 0.8 * st.P["tone_contrast"] * (1 - sh) ** 2, 0, 1)
        # Cuts run along the stalk.
        return st.contour(sl, sdf, t, pitch=st.pitch * 0.8, taper=0.5)
    # Leaves: black with a carved midrib.
    return st.contour(sl, sdf, 0.12, pitch=st.pitch * 0.9, taper=0.8, offset=0.3)


def _house(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    if part.name == "walls":
        t = np.clip(0.97 - 0.8 * c * sh, 0, 1)
        return st.parallel(sl, 0.0, t, pitch=st.pitch * 0.9, taper=0.4)
    t = _tone(0.06, 0.4, sh, c)
    return st.contour(sl, part.sdf[sl], t, pitch=st.pitch * 0.9)


def _tower_part(st, shape, part, sl, light: bool):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    t = np.clip(0.97 - 0.85 * c * sh, 0, 1) if light else _tone(0.03, 0.3, sh, c)
    body = shape.part("tower") or part
    return st.turned(sl, body.sdf[sl], t, bow=0.3, pitch=st.pitch * 0.85)


def _lighthouse(st, shape, part, sl, m):
    if part.name == "tower":
        return _tower_part(st, shape, part, sl, True)
    if part.name == "stripes":
        return _tower_part(st, shape, part, sl, False)
    if part.name == "gallery":
        return np.ones_like(st.X[sl])
    return _white(st, sl)  # lantern, beam


def _windmill(st, shape, part, sl, m):
    if part.name == "tower":
        return _tower_part(st, shape, part, sl, True)
    if part.name == "cap":
        return st.contour(sl, part.sdf[sl], 0.12)
    if part.name == "sails":
        return st.contour(sl, part.sdf[sl], 0.72, pitch=st.pitch * 1.3, taper=0.1)
    return np.ones_like(st.X[sl])


def _boat(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    sdf = part.sdf[sl]
    if part.name == "hull":
        t = _tone(0.0, 0.3, sh, c) * np.exp(np.minimum(sdf, 0) / (6 * st.sc))
        return st.contour(sl, sdf, t, pitch=st.pitch * 0.8, taper=0.5, offset=0.5)
    if part.name == "cabin":
        return _white(st, sl)
    if part.name == "sail":
        # Canvas seams: cuts parallel to the mast, closing up on the shadowed cloth.
        t = np.clip(0.98 - 0.7 * c * sh, 0, 1)
        return st.parallel(sl, np.radians(88), t, pitch=st.pitch * 1.05, taper=0.4)
    return np.ones_like(sdf)


def _bird(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    sdf = part.sdf[sl]
    if part.name == "body":
        t = _tone(0.05, 0.6, sh, c)
        pat = st.contour(sl, sdf, t, pitch=st.pitch * 0.8)
        return pat
    if part.name == "wing":
        return st.contour(sl, sdf, 0.2, pitch=st.pitch * 0.75, taper=0.3)
    return _white(st, sl)  # beak


def _vase(st, shape, part, sl, m):
    if part.name == "band":
        # A carved white band with a zigzag left standing in it.
        X, Y = st.X[sl], st.Y[sl]
        cy = float(Y[part.sdf[sl] < 0].mean()) if np.any(part.sdf[sl] < 0) else shape.cy
        z = np.abs(((X - shape.cx) / (st.pitch * 1.6)) % 2 - 1) - 0.5
        zig = np.abs(Y - cy - z * st.pitch * 1.3) - 1.4 * st.sc
        return sdf_to_mask(zig)
    sh = st.shade(part)[sl]
    t = np.clip(0.04 + 1.0 * st.P["tone_contrast"] * (1 - sh) ** 1.4, 0, 1)
    return st.turned(sl, part.sdf[sl], t, bow=0.3, pitch=st.pitch * 0.9)


def _components(part: Part):
    """Split a group of (possibly overlapping) round fruit into one Part per fruit.

    Fruit centres are the peaks of the inside distance; each pixel goes to the
    fruit whose inscribed circle it is deepest in, and the cell walls become part
    of that fruit's SDF, so each fruit gets its own shading and outline.
    """
    inner = np.clip(-part.sdf, 0, None)
    rmax = float(inner.max())
    if rmax <= 2:
        return [part]
    size = max(3, int(rmax * 1.1))
    peaks = (inner == ndimage.maximum_filter(inner, size=size)) & (inner > 0.7 * rmax)
    lab, n = ndimage.label(peaks)
    if n <= 1:
        return [part]
    cs = ndimage.center_of_mass(peaks, lab, range(1, n + 1))
    H, W = part.sdf.shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    ds = []
    for cy, cx in cs:
        r = float(inner[int(round(cy)), int(round(cx))])
        ds.append(np.hypot(X - cx, Y - cy) - r)
    ds = np.stack(ds)
    out = []
    for i in range(n):
        others = np.min(np.delete(ds, i, axis=0), axis=0)
        cell = (ds[i] - others) * 0.5
        out.append(Part(part.name, np.maximum(part.sdf, cell).astype(np.float32), part.role, part.detail, part.shade))
    return out


def _fruit(st, shape, part, sl, m):
    sdf = part.sdf[sl]
    if part.name == "leaf":
        return st.contour(sl, sdf, 0.15, pitch=st.pitch * 0.8, taper=0.8, offset=0.3)
    sh = st.shade(part)[sl]
    t = np.clip(0.04 + 0.85 * st.P["tone_contrast"] * (1 - sh) ** 1.4, 0, 1)
    ccx, _, cr, _ = part.stats()
    return st.cross(sl, ccx, cr, t, bow=0.45, pitch=st.pitch * 0.8)


def _teapot(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    if part.name == "lid":
        t = np.clip(0.35 + 0.55 * c * (1 - sh), 0, 1)
        return st.parallel(sl, 0.0, t, pitch=st.pitch * 0.8, taper=0.4)
    t = np.clip(0.04 + 0.85 * c * (1 - sh) ** 1.4, 0, 1)
    return st.cross(sl, shape.cx, shape.radius * 0.72, t, bow=0.3, pitch=st.pitch * 0.9)


def _cup(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    c = st.P["tone_contrast"]
    if part.name == "tea":
        return np.ones_like(sh)
    if part.name == "saucer":
        t = np.clip(0.45 - 0.4 * c * sh, 0, 1)
        return st.contour(sl, part.sdf[sl], t, pitch=st.pitch * 0.8, taper=0.3)
    t = np.clip(0.97 - 0.8 * c * sh, 0, 1)
    return st.turned(sl, part.sdf[sl], t, bow=0.3, pitch=st.pitch * 0.85)


def _cloud(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    base = 0.55 if st.night else 1.0
    t = np.clip(base - 0.6 * sh ** 1.5, 0, 1)
    return st.contour(sl, part.sdf[sl], t, pitch=st.pitch * 1.1, taper=0.4)


def _generic(st, shape, part, sl, m):
    sh = st.shade(part)[sl]
    base = {"light": 0.8, "sky": 0.8, "dark": 0.0, "accent": 0.3, "foliage": 0.05, "wood": 0.05}.get(part.role, 0.1)
    t = _tone(base, 0.5, sh, st.P["tone_contrast"])
    return st.contour(sl, part.sdf[sl], t)


TREAT = {
    "sun": lambda st, sh, p, sl, m: _white(st, sl),
    "moon": lambda st, sh, p, sl, m: _white(st, sl),
    "cloud": _cloud,
    "mountain": _mountain, "hills": _hills, "sea": _water, "river": _water, "field": _field,
    "table": _table, "cliff": _cliff,
    "tree": _tree, "pine": _pine, "cypress": _cypress, "bamboo": _bamboo,
    "house": _house, "lighthouse": _lighthouse, "windmill": _windmill,
    "boat": _boat, "bird": _bird,
    "vase": _vase, "fruit": _fruit, "teapot": _teapot, "cup": _cup,
}


# ---------------------------------------------------------------------------
# Part compositing and outlines
# ---------------------------------------------------------------------------

def _detail(st, shape, part, sl, before):
    """Thin parts are cut in whichever of black or white contrasts with what is under them."""
    m = sdf_to_mask(part.sdf[sl])
    if shape.kind == "rain":
        bg = ndimage.gaussian_filter(before, 3 * st.sc)
        return m, (bg < 0.5).astype(np.float32)
    grow = sdf_to_mask(part.sdf[sl] - 4 * st.sc)
    under = float((before * grow).sum() / max(grow.sum(), 1e-3))
    return m, np.full_like(m, 1.0 if under < 0.5 else 0.0)


def _outline(st, sl, sdf, m, before, pat, is_region):
    w = st.P["outline"]
    if w <= 0:
        return
    sc = st.sc
    sig = 5.0 * sc
    inside = K.local_mean(pat, m, sig)
    outside = K.local_mean(before, 1 - m, sig)
    wob = 1.0 + 0.35 * (st.noise(st.X[sl] * 3.1 + 500, st.Y[sl] * 3.1) - 0.5)
    # Dark against dark: carve a white separation line just outside the silhouette.
    both_dark = np.clip((np.minimum(inside, outside) - 0.55) / 0.2, 0, 1)
    ww = 2.8 * sc * w * wob
    band_w = np.clip(ww / 2 - np.abs(sdf - ww / 2 - 0.4 * sc) + 0.5, 0, 1)
    # ...and a broken echo line further out, the carver's way of lifting a silhouette.
    echo = np.clip(0.4 * ww - np.abs(sdf - 3.4 * ww) + 0.5, 0, 1) * (st.noise(st.X[sl] * 1.7, st.Y[sl] * 1.7 + 300) > 0.35)
    band_w = np.maximum(band_w, echo) * both_dark
    # Light shape: a black keyline on the silhouette.
    light_in = np.clip((0.62 - inside) / 0.25, 0, 1)
    if is_region:
        light_in = light_in * np.clip((0.6 - outside) / 0.25, 0, 1)
    else:
        # Objects keep a black keyline unless they sit dark-on-dark (then the white cut separates).
        light_in = np.maximum(light_in, 1.0 - both_dark)
    wk = 2.0 * sc * w * wob
    band_k = np.clip(wk / 2 - np.abs(sdf) + 0.5, 0, 1) * light_in
    ink = st.ink[sl]
    ink[:] = np.minimum(ink, 1 - band_w)
    ink[:] = np.maximum(ink, band_k)


def pi_first(shape, part):
    return shape.parts and part is shape.parts[0]


def _carve_part(st, shape, part):
    if shape.kind == "fruit" and part.name == "fruit" and not getattr(part, "_split", False):
        comps = _components(part)
        comps.sort(key=lambda p: p.stats()[1])  # back (higher) fruit first
        for c in comps:
            c._split = True
            _carve_part(st, shape, c)
        return
    sl = st.crop(part, pad=18 if not shape.region else 4)
    if sl[0].stop <= sl[0].start or sl[1].stop <= sl[1].start:
        return
    sdf = part.sdf[sl]
    if not np.any(sdf < 1):
        return
    before = st.ink[sl].copy()
    if part.detail:
        m, pat = _detail(st, shape, part, sl, before)
        if shape.kind == "stars":
            pat = np.zeros_like(pat) if st.night or before.mean() > 0.5 else pat
        st.ink[sl] = before * (1 - m) + pat * m
        return
    m = sdf_to_mask(sdf + st.silh[sl] * (0.3 if shape.region else 1.0))
    fn = TREAT.get(shape.kind, _generic)
    # At night, land and objects print as near-black silhouettes against the glow.
    st.dim = shape.kind not in ("sun", "moon", "stars", "cloud", "sea", "river")
    # A small object on a dark ground is cut light (keylined), or it would vanish.
    if not shape.region and shape.kind not in ("sun", "moon", "cloud", "stars"):
        x0, y0, x1, y1 = shape.bbox()
        small = (x1 - x0) * (y1 - y0) < 0.02 * st.H * st.W
        if small and pi_first(shape, part):
            ring = sdf_to_mask(shape.sdf[sl] - 10 * st.sc) - sdf_to_mask(shape.sdf[sl])
            shape._lift = float((before * ring).sum() / max(ring.sum(), 1e-3)) > 0.7
        st.lift = bool(small and getattr(shape, "_lift", False))
        if st.lift:
            st.dim = False
    pat = fn(st, shape, part, sl, m)
    st.dim = False
    st.lift = False
    if st.P["second_ink"] and (part is st.color_target or (st.color_target is None and (
            part.role == "accent" or shape.kind == "sun") and shape.kind != "lighthouse")):
        st.color[sl] = np.maximum(st.color[sl], sdf_to_mask(sdf - 1.5 * st.sc))
        if shape.kind != "sun":
            # Over the colour block the key keeps only its shadow-side cuts.
            pat = pat * (ndimage.gaussian_filter(pat, 4 * st.sc) > 0.55)
    st.ink[sl] = before * (1 - m) + pat * m
    _outline(st, sl, sdf, m, before, pat, shape.region)


# ---------------------------------------------------------------------------
# Extra carving: stars, chatter, plate edge
# ---------------------------------------------------------------------------

def _stardust(st):
    r = st.ctx.rng.child("stardust")
    sky = (st.Y < st.hz * 0.95) & (st.ink > 0.5)
    xs, ys = K.scatter(sky.astype(np.float32), 26 * st.sc, r, density=0.35 * st.P["chip_density"] + 0.1)
    if len(xs):
        n = len(xs)
        st.gouge(xs, ys, r.uniform(0, np.pi, n), 4 * st.sc * r.uniform(0.6, 1.5, n), 2.2 * st.sc, label="dust")


def _chatter(st):
    if st.P["chatter"] <= 0:
        return
    sc = st.sc
    clear = ndimage.gaussian_filter(st.ink, 6 * sc) < 0.06
    r = st.ctx.rng.child("chatter")
    xs, ys = K.scatter(clear.astype(np.float32), 30 * sc, r, density=0.18 * st.P["chatter"])
    if len(xs):
        n = len(xs)
        ang = r.normal(0, 0.25, n) + (r.random(n) < 0.3) * np.pi / 2
        st.gouge(xs, ys, ang, 14 * sc * r.uniform(0.5, 1.6, n), 1.6 * sc * r.uniform(0.6, 1.2, n), value=1.0,
                 clip=clear.astype(np.float32), label="chatter")


def _plate(st):
    """Hand-cut plate edge: blank paper outside, an uncut rim with chips inside."""
    sc = st.sc
    margin = st.P["plate_margin"] * st.H
    r = st.ctx.rng.child("plate")
    d = K.plate_sdf(st.f.shape, margin, r, wobble=2.2 * sc, fray=0.9 * sc * st.P["edge_ragged"] + 0.2)
    rim_w = 5.5 * sc
    rim = sdf_to_mask(np.abs(d + rim_w / 2) - rim_w / 2)
    st.ink = np.maximum(st.ink, rim)
    # Chips knocked out of the rim and the plate corner.
    rim_band = (np.abs(d + rim_w / 2) < rim_w).astype(np.float32)
    xs, ys = K.scatter(rim_band, 22 * sc, r.child("chips"), density=0.4)
    if len(xs):
        n = len(xs)
        gy, gx = np.gradient(ndimage.gaussian_filter(d, 3))
        ix, iy = np.clip(xs.astype(int), 0, st.W - 1), np.clip(ys.astype(int), 0, st.H - 1)
        ang = np.arctan2(gy[iy, ix], gx[iy, ix]) + r.normal(0, 0.5, n)
        st.gouge(xs, ys, ang, 7 * sc * r.uniform(0.5, 1.5, n), 3.0 * sc * r.uniform(0.6, 1.3, n), label="rimchip")
    st.ink = st.ink * sdf_to_mask(d)
    return d


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def _second_color(scene, params):
    if params.get("color"):
        return hex_to_rgb(params["color"])
    hints = scene_hints(scene)

    def warmth(c):
        return (c[0] - c[2]) + 0.5 * (max(c) - min(c)) - 0.3 * abs(luminance(c) - 0.55)

    return max(hints, key=warmth)


def _print(st, plate_d):
    P, sc = st.P, st.sc
    r = st.ctx.rng.child("print")
    # Fray every cut edge a little: soften, then re-threshold against fine noise.
    fine = value_noise(st.f.shape, 1.6 * sc, r.child("fine")) - 0.5
    b = ndimage.gaussian_filter(st.ink, 0.7 * sc)
    ink = np.clip((b - 0.5 + 0.28 * P["edge_ragged"] * fine) * 2.2 + 0.5, 0, 1)
    paper_hex = P["paper"]
    pimg, fib = paper(st.f.shape, r.child("paper"), tint=paper_hex, strength=0.8, scale=sc)
    tex = P["ink_texture"]
    # Roller: broad density drift and faint bands along the roll direction.
    roll = 1.0 - 0.1 * tex * value_noise(st.f.shape, 260 * sc, r.child("roll"))
    bands = blurred_white((st.H, 1), r.child("bands"), (18 * sc, 0))
    roll = roll * (1.0 - 0.07 * tex * np.abs(bands - 0.5) * 2)
    # Paper grain showing through the solids: ink misses the valleys of the sheet.
    starve = fbm(st.f.shape, 40 * sc, r.child("starve"), octaves=3)
    pits = blurred_white(st.f.shape, r.child("pits"), 0.9 * sc)
    grain = (pits - 0.5) * 2.0 + (starve - 0.5) * 1.0 + (fib - 0.5) * 0.3
    holes = np.clip((grain - (1.08 - 0.3 * tex)) * 4.0, 0, 1)
    speck = (r.random(st.f.shape) < 0.004 * tex).astype(np.float32)
    speck = ndimage.maximum_filter(speck, size=max(1, int(round(1.6 * sc))))
    density = ink * roll * (1.0 - 0.85 * holes) * (1.0 - 0.9 * speck)
    ink_rgb = hex_to_rgb(P["ink"])
    out = pimg.copy()
    n_inks = 1
    if P["second_ink"]:
        crgb = _second_color(st.ctx.scene, P)
        cm = ndimage.gaussian_filter(st.color, 0.8 * sc)
        cm = np.clip((cm - 0.5 + 0.3 * fine) * 2.5 + 0.5, 0, 1) * sdf_to_mask(plate_d)
        cden = cm * (0.92 + 0.08 * roll) * (1.0 - 0.6 * holes)
        # Slightly off-register under the key block.
        cden = ndimage.shift(cden, r.uniform(-1.5, 1.5, 2) * sc, order=1, mode="nearest")
        out = out * (1.0 - cden[..., None] * (1.0 - crgb))
        n_inks = 2
        st.ctx.info["second_color"] = "#%02x%02x%02x" % tuple(int(v * 255) for v in crgb)
    out = out * (1.0 - np.clip(density, 0, 1)[..., None] * (1.0 - ink_rgb))
    st.ctx.info["black_fraction"] = round(float((ink > 0.5).mean()), 3)
    return np.clip(out, 0, 1).astype(np.float32), n_inks


def render_array(scene, seed=None, params=None):
    params = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, params, NAME)
    st = _State(ctx)
    _background(st)
    for shape in ctx.shapes:
        for part in shape.parts:
            _carve_part(st, shape, part)
        if shape.kind == "stars":
            _stardust(st)
    _chatter(st)
    plate_d = _plate(st)
    img, n_inks = _print(st, plate_d)
    ctx.info.update({
        "ink_count": n_inks,
        "ink": params["ink"],
        "paper": params["paper"],
        "gouge_marks": int(st.marks),
        "interior": st.interior,
    })
    return img, ctx


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)
