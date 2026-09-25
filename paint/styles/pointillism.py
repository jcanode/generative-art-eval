"""Pointillism, in the manner of 1880s divisionism.

Painting model
--------------
1. **Study (target image).** The scene is first worked out as a calm, simplified
   colour study: flat local colours per part, a warm-lit / cool-shadow split
   from ``shade_field``, a graded sky with glows around lights, water that
   reflects the sky and the sun, aerial perspective on far forms. This study
   is never shown; it only tells the dots what to average to.
2. **Irradiation.** The study's luminance is pushed away from its blurred self
   (lighter beside dark, darker beside light), which paints the halo that
   divisionists put along every contrasting edge.
3. **Division.** A fixed palette of unmixed pigments (pure hues, their white
   tints and deep shades; no earths, no black). Each colour of the study is
   decomposed into a pigment mixture (``PigmentMixer``: NNLS with a
   hue-aware ridge that spreads weight over neighbouring hues); every dot
   samples one pigment, so e.g. a green is emerald + yellow-green + cerulean +
   lemon dots side by side. On top of that, shadows receive dots of the
   *complement* of the local colour, lit surfaces receive dots of the light's
   colour, and the halo seams get complementary dots of the colour across the
   edge.
4. **Dotting.** Several sessions of dots on jittered hex lattices, each batch
   splatted with ``np.bincount`` (``pointillism_kit.splat``). Dot size is
   nearly constant (a little larger toward the bottom). A final session of
   smaller dots follows thin detail parts (masts, windows, stems, stars, birds,
   rain) so they survive the dot pitch. Silhouettes are made only by where the
   dot colours change: no outlines.
5. **Canvas.** A warm primed ground shows between dots; a faint canvas weave and
   dab relief finish the surface.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.color import hex_to_rgb, luminance, mix
from ..core.noise import value_noise
from ..core.paintbody import hsv_to_rgb, rgb_to_hsv, vivid
from ..core.pointillism_kit import DotBank, PigmentMixer, jittered_grid, splat
from ..core.texture import canvas_weave, emboss
from ..scene.shapes import SKY_KINDS, shade_field
from .base import ROLE_TARGETS, clamp_params, encode, make_context, merge_params, scene_hints

NAME = "pointillism"

DEFAULTS = {
    "dot_size": 6.5,          # dot diameter in px at 1024 px height
    "spacing": 1.3,           # lattice pitch / dot diameter inside one session
    "passes": 3,              # dotting sessions (more = denser, less ground)
    "foreground_grow": 0.25,  # dots this much larger at the bottom than at the top
    "division": 0.015,        # how widely a colour is split over neighbouring pigments
    "complement": 0.3,        # complementary dots in shadows
    "sunlight": 0.18,         # dots of the light's colour on lit surfaces
    "halo": 1.0,              # irradiation strength along light/dark edges
    "halo_radius": 2.2,       # halo width in dot sizes
    "seam": 0.35,             # complementary dots on halo seams
    "saturation": 1.35,       # saturation boost of the study colours
    "shade": 0.8,             # strength of the lit/shadow split
    "glow": 1.0,              # glow around suns, moons, stars, lanterns
    "detail_scale": 0.55,     # detail dots relative to main dots
    "ground_show": 1.0,       # canvas texture strength
    "relief": 0.3,            # dab relief (emboss)
    "dark_fill": 0.9,         # extra dotting session in dark areas
}

KNOBS = {
    "dot_size": (3.5, 11.0, "dot diameter; bigger reads more mosaic-like, smaller more tonal"),
    "spacing": (1.0, 1.8, "dot pitch in a session; higher shows more ground"),
    "passes": (2, 5, "number of dotting sessions; more covers the ground"),
    "foreground_grow": (0.0, 0.6, "how much bigger foreground dots are"),
    "division": (0.0, 0.15, "colour division: spread of a colour over neighbouring pigments"),
    "complement": (0.0, 0.7, "complementary dots in shadows"),
    "sunlight": (0.0, 0.5, "light-coloured dots on lit surfaces"),
    "halo": (0.0, 1.2, "irradiation halo strength at light/dark edges"),
    "halo_radius": (1.0, 5.0, "halo width in dot sizes"),
    "seam": (0.0, 0.8, "complementary dots along contrasting edges"),
    "saturation": (0.8, 1.6, "saturation boost of the colour study"),
    "shade": (0.0, 1.0, "lit/shadow contrast on objects"),
    "glow": (0.0, 2.0, "glow around light sources"),
    "detail_scale": (0.35, 0.9, "size of the dots that carry thin details"),
    "ground_show": (0.0, 2.0, "canvas weave visibility"),
    "relief": (0.0, 1.0, "paint relief of the dabs"),
    "dark_fill": (0.0, 1.0, "extra dots in dark areas (less ground showing there)"),
}

RULES = {"min_mean_saturation": 0.25, "min_edge_density": 0.3}

CANON = {k: hex_to_rgb(v) for k, v in {
    "sky": "#9cc4e4", "water": "#3a78b8", "ground": "#b8a24c", "primary": "#2f64b4", "secondary": "#4f9a64",
    "accent": "#e0502a", "dark": "#2a2c62", "light": "#f4ead0", "foliage": "#3c8c46", "wood": "#8c5a30",
    "stone": "#8c84a0",
}.items()}
PULL = {"sky": 0.35, "water": 0.3, "ground": 0.3, "primary": 0.12, "secondary": 0.15, "accent": 0.15, "dark": 0.45,
        "light": 0.25, "foliage": 0.4, "wood": 0.35, "stone": 0.3}
EMITTERS = {"sun", "moon", "stars"}
INTERIOR_KINDS = {"table", "vase", "teapot", "cup", "fruit"}
WATER = {"sea", "river"}


def _lum(c):
    return float(luminance(np.asarray(c)))


class _Palette:
    def __init__(self, scene, P):
        self.hints = scene_hints(scene)
        self.mood = scene.palette.mood
        self.warmth = float(scene.light.warmth)
        self.P = P
        self.rainy = any(o.kind == "rain" for o in scene.objects)
        kinds = {o.kind for o in scene.objects}
        self.interior = bool(kinds & INTERIOR_KINDS) and not (kinds & (SKY_KINDS | {"sea", "field", "hills", "mountain"}))
        self.sun_rgb = mix(hex_to_rgb("#f8f0c8"), hex_to_rgb("#f6c050"), self.warmth)
        self.shadow_rgb = mix(hex_to_rgb("#3c3c96"), hex_to_rgb("#5a3a8c"), self.warmth)
        if self.mood == "night":
            self.sun_rgb = hex_to_rgb("#e8e8c0")

    def v(self, c):
        return np.clip(vivid(np.asarray(c, np.float32), self.P["saturation"], 0.03), 0, 1).astype(np.float32)

    def role(self, role, obj_color=None):
        if obj_color and obj_color.startswith("#"):
            return self.v(hex_to_rgb(obj_color))
        if obj_color and obj_color in CANON:
            role = obj_color
        target = hex_to_rgb(ROLE_TARGETS.get(role, ROLE_TARGETS["primary"]))
        w = np.array([0.3, 0.45, 0.25], np.float32)
        d = [float(np.sum((c - target) ** 2 * w) + 0.6 * (_lum(c) - _lum(target)) ** 2) for c in self.hints]
        h = self.hints[int(np.argmin(d))]
        return self.v(mix(h, CANON.get(role, CANON["primary"]), PULL.get(role, 0.2)))

    def sky(self):
        h = self.hints
        lightest = max(h, key=_lum)
        darkest = min(h, key=_lum)
        bluest = max(h, key=lambda c: (c[2] - c[0]) - 0.2 * abs(_lum(c) - 0.55))
        if self.interior:
            wall = mix(lightest, hex_to_rgb("#f2dca0") if self.warmth >= 0.5 else hex_to_rgb("#c8dcec"), 0.45)
            top = mix(wall, hex_to_rgb("#c89a70") if self.warmth >= 0.5 else hex_to_rgb("#8ca0c8"), 0.35)
            return self.v(top), self.v(wall)
        if self.mood == "night":
            top = mix(darkest, hex_to_rgb("#1c2466"), 0.55)
            bottom = mix(bluest, hex_to_rgb("#3a58a8"), 0.5)
            return self.v(top), self.v(bottom)
        if self.mood == "warm":
            sk = self.role("sky")
            top = mix(sk, hex_to_rgb("#e4806a"), 0.4)
            bottom = mix(lightest, hex_to_rgb("#f2c060"), 0.55)
            out = self.v(top), self.v(bottom)
        else:
            k = 0.5 if (bluest[2] - bluest[0]) > 0.15 else 0.75
            top = mix(bluest, hex_to_rgb("#88b8e4"), k)
            top = top + (1 - top) * 0.15
            bottom = mix(lightest, hex_to_rgb("#f2ecd0"), 0.5)
            out = self.v(top), self.v(bottom)
        if self.rainy:
            grey = hex_to_rgb("#98a0b4")
            out = tuple(mix(c, grey * (0.8 + 0.4 * _lum(c)), 0.45).astype(np.float32) for c in out)
        return out

    def lit(self, c):
        c = np.asarray(c, np.float32)
        a = 0.22 * self.P["shade"]
        w = c + (self.sun_rgb - c) * a
        return np.clip(w + (1 - w) * 0.05, 0, 1)

    def dark(self, c):
        c = np.asarray(c, np.float32)
        a = self.P["shade"]
        d = c * (1 - 0.38 * a)
        return np.clip(d + (self.shadow_rgb * 0.8 - d) * 0.3 * a, 0, 1)


class _Study:
    """The hidden colour study the dots average to."""

    def __init__(self, ctx, pal: _Palette):
        self.ctx = ctx
        self.pal = pal
        self.P = ctx.params
        self.f = ctx.frame
        self.H, self.W = ctx.frame.shape
        self.T = np.zeros((self.H, self.W, 3), np.float32)
        self.S = np.zeros((self.H, self.W), np.float32)       # shadow amount (0 lit .. 1 shadow)
        self.lit = np.zeros((self.H, self.W), np.float32)     # sunlit surface weight
        self.local = np.zeros((self.H, self.W, 3), np.float32)  # unshaded local colour
        self.glow = np.zeros((self.H, self.W), np.float32)
        self.details = []  # (mask, sdf, color-map-or-rgb)
        self.horizon = ctx.scene.horizon * self.H
        self.sky_top, self.sky_bottom = pal.sky()
        if pal.mood == "night":
            self.haze = mix(self.sky_top, hex_to_rgb("#1e1c4a"), 0.5)
        else:
            self.haze = mix(self.sky_bottom, hex_to_rgb("#8480c8"), 0.72)

    def put(self, m, color, S=None, lit=None, local=None):
        m = m[..., None]
        col = np.asarray(color, np.float32)
        self.T += (col - self.T) * m
        loc = col if local is None else np.asarray(local, np.float32)
        self.local += (loc - self.local) * m
        mm = m[..., 0]
        self.S += ((0 if S is None else S) - self.S) * mm
        self.lit += ((0 if lit is None else lit) - self.lit) * mm

    # -- background ----------------------------------------------------------
    def paint_sky(self):
        X, Y = self.f.grid
        top, bottom = self.sky_top, self.sky_bottom
        span = max(self.horizon, self.H * 0.3) if not self.pal.interior else self.H
        t = np.clip(Y / span, 0, 1) ** 0.85
        n = value_noise(self.f.shape, 180 * self.f.scale, self.ctx.rng.child("skynoise"))
        t = np.clip(t + (n - 0.5) * 0.25, 0, 1)
        col = top[None, None] * (1 - t[..., None]) + bottom[None, None] * t[..., None]
        self.T[:] = col
        self.local[:] = col
        # Glows around lights.
        g = self.P["glow"]
        for sh in self.ctx.shapes:
            if sh.kind in ("sun", "moon"):
                cx, cy, r, _ = sh.parts[0].stats()
                if r <= 0:
                    continue
                d = np.hypot(X - cx, Y - cy)
                fall = np.exp(-np.clip(d - r, 0, None) / (r * (0.9 if sh.kind == "sun" else 0.8)))
                gc = mix(self.pal.sun_rgb, hex_to_rgb("#f09030"), 0.3) if sh.kind == "sun" else \
                    mix(hex_to_rgb("#e0dcb0"), self.sky_bottom, 0.4)
                amt = np.clip(fall * (0.55 if sh.kind == "sun" else 0.45) * g, 0, 0.8)
                self.T += (gc - self.T) * amt[..., None]
                self.glow = np.maximum(self.glow, fall)

    # -- parts ----------------------------------------------------------------
    def paint_region(self, sh):
        X, Y = self.f.grid
        pal = self.pal
        rng = self.ctx.rng.child("region", sh.kind, f"{sh.obj.x:.3f}", f"{sh.obj.y:.3f}")
        for p in sh.parts:
            m = p.mask
            if m.max() <= 0:
                continue
            base = pal.role(p.role, sh.obj.color)
            if sh.kind == "table" and p.name == "top" and not sh.obj.color:
                lightest = max(pal.hints, key=_lum)
                base = mix(lightest, hex_to_rgb("#7480c0") if pal.warmth >= 0.4 else hex_to_rgb("#c08a58"), 0.7)
            top = self.horizon if sh.kind != "table" else p.stats()[3][1]
            depth = np.clip((Y - top) / max(self.H - top, 1), 0, 1)
            if sh.kind in WATER:
                near = pal.v(mix(base, hex_to_rgb("#1c3c8c"), 0.25)) * 0.9
                far = mix(base, self.sky_bottom, 0.5)
                col = far[None, None] * (1 - depth[..., None] ** 0.7) + near[None, None] * depth[..., None] ** 0.7
                cell = 30 * self.f.scale
                band = value_noise((self.H, max(2, int(self.W / 8))), cell, rng.child("waves"))
                band = ndimage.zoom(band, (1, self.W / band.shape[1]), order=1)[:self.H, :self.W]
                if band.shape != (self.H, self.W):
                    band = np.pad(band, ((0, self.H - band.shape[0]), (0, self.W - band.shape[1])), mode="edge")
                col = col * (0.9 + 0.2 * band[..., None])
                self.put(m, col, lit=0.4, local=base)
                # Reflections of lights: a column of broken light under suns and moons.
                for s2 in self.ctx.shapes:
                    if s2.kind in ("sun", "moon") and s2.parts[0].stats()[2] > 0:
                        cx, cy, r, _ = s2.parts[0].stats()
                        wdt = r * (0.5 + 1.2 * depth)
                        colm = np.exp(-((X - cx) / np.maximum(wdt, 1)) ** 2) * np.clip(band * 1.6 - 0.35, 0, 1)
                        amt = np.clip(colm * 0.8 * self.P["glow"], 0, 0.9)
                        gl = mix(pal.sun_rgb, hex_to_rgb("#fff8e0"), 0.3)
                        self.T += (gl - self.T) * (m * amt)[..., None]
            else:
                if sh.kind == "table" and p.name == "front":
                    col = pal.dark(base)[None, None] * np.ones((self.H, self.W, 1), np.float32)
                    self.put(m, col, S=0.6, local=base)
                    continue
                far = mix(pal.lit(base), self.sky_bottom, 0.25) if sh.kind != "table" else base
                near = pal.v(base * 0.92)
                n = value_noise(self.f.shape, 90 * self.f.scale, rng.child("mottle"))
                col = far[None, None] * (1 - depth[..., None]) + near[None, None] * depth[..., None]
                col = col * (0.92 + 0.16 * n[..., None])
                self.put(m, col, lit=0.8, local=base)

    def paint_shape(self, sh):
        pal = self.pal
        light = self.ctx.light_dir
        aerial = float(np.clip((sh.depth - 0.72) * 1.6, 0, 0.35)) if sh.kind not in SKY_KINDS else 0.0
        nparts = len(sh.parts)
        for pi, p in enumerate(sh.parts):
            m = p.mask
            if m.max() <= 0:
                continue
            role = p.role
            base = pal.role(role, sh.obj.color if p.role not in ("dark", "accent") or p.name in ("hull", "body", "birds") else None)
            if sh.kind == "sun":
                base = mix(pal.sun_rgb, hex_to_rgb("#fffbea"), 0.75)
            elif sh.kind == "moon":
                base = hex_to_rgb("#f6f0cc")
            elif sh.kind == "stars":
                base = hex_to_rgb("#fbf4d0")
            elif p.name == "lantern":
                base = hex_to_rgb("#fae070")
            elif p.name == "beam":
                base = mix(pal.sun_rgb, hex_to_rgb("#fff8e0"), 0.6)
            elif p.name == "snow":
                base = mix(hex_to_rgb("#f6f4ee"), self.sky_top, 0.15)
            elif sh.kind == "cloud":
                base = mix(hex_to_rgb("#f8f2e4"), pal.sun_rgb, 0.25)
            elif sh.kind == "fruit" and p.name == "fruit" and not sh.obj.color:
                v = sh.obj.variant or "apple"
                base = pal.v(hex_to_rgb({"orange": "#ec8a1c", "pear": "#b8c83a"}.get(v, "#d23a2a")))
            elif p.name == "nodes":
                base = mix(base, pal.role("foliage"), 0.45)
            elif sh.kind == "rain":
                base = mix(pal.role("dark"), hex_to_rgb("#6878a0"), 0.5)
            # Keep figure apart from ground: compare with what is already under the part.
            if sh.kind not in EMITTERS and p.name not in ("beam", "snow", "stripes", "band", "nodes", "tea"):
                base = self._separate(base, m)
            if aerial > 0:
                a = aerial
                if sh.kind == "hills":
                    a = max(a, 0.4) + 0.2 * (nparts - 1 - pi) / max(nparts - 1, 1)
                base = mix(base, self.haze, a)
            alpha = m
            if p.name == "beam":
                alpha = m * 0.45
            if p.detail or sh.kind in EMITTERS or p.name == "lantern":
                self.put(alpha, base, local=base, lit=0.0)
                if p.detail:
                    self.details.append((p, base, sh.kind))
                if sh.kind == "stars" or p.name == "lantern":
                    fall = np.exp(-np.clip(p.sdf, 0, None) / (7 * self.f.scale + (4 if p.name == "lantern" else 0)))
                    amt = np.clip(fall * 0.6 * self.P["glow"], 0, 0.8) * (1 - m)
                    self.T += (hex_to_rgb("#f4ecc0") - self.T) * amt[..., None]
                    self.glow = np.maximum(self.glow, fall)
                continue
            if p.shade:
                S = shade_field(p, light, self.f)
                lit, dk = pal.lit(base), pal.dark(base)
                s = np.clip(S, 0, 1) ** 1.1
                col = lit[None, None] * (1 - s[..., None]) + dk[None, None] * s[..., None]
                self.put(alpha, col, S=S, lit=1 - S, local=base)
            else:
                self.put(alpha, base, S=0.15, lit=0.5, local=base)
            if self._small(p):
                self.details.append((p, None, sh.kind))

    def _separate(self, base, m):
        """Push a part's colour away from what is already under it (checked per half)."""
        ys = np.nonzero(m.max(axis=1) > 0.5)[0]
        if len(ys) == 0:
            return base
        mid = int(ys.mean())
        worst, under = 9.0, None
        for sl in (slice(0, mid), slice(mid, self.H)):
            mm = m[sl]
            ws = float(mm.sum())
            if ws < 20:
                continue
            u = (self.T[sl] * mm[..., None]).reshape(-1, 3).sum(0) / ws
            dl = abs(_lum(base) - _lum(u))
            dc = float(np.sqrt(np.sum((base - u) ** 2)))
            score = dl + 0.25 * dc
            if score < worst:
                worst, under = score, u
        thr = 0.26
        if under is None or worst >= thr:
            return base
        k = float(np.clip((thr - worst) / thr, 0, 1))
        lb, lu = _lum(base), _lum(under)
        up = (lb > lu and lu < 0.7) or lu < 0.3
        t = min(lu + 0.28, 0.92) if up else max(lu - 0.3, 0.06)
        t = lb + (t - lb) * min(1.0, 0.4 + k)
        if t >= lb:
            a = (t - lb) / max(1 - lb, 1e-3)
            out = base + (1 - base) * a
        else:
            out = base * (t / max(lb, 1e-3))
        # Also turn the hue away from the ground, toward its complement.
        hsv = rgb_to_hsv(under)
        comp = hsv_to_rgb(np.array([(hsv[0] + 0.5) % 1.0, max(hsv[1], 0.5), _lum(out) + 0.25], np.float32))
        out = mix(out, comp * (_lum(out) / max(_lum(comp), 1e-3)), 0.35 * k)
        return np.clip(out, 0, 1).astype(np.float32)

    def _small(self, p):
        cx, cy, r, bbox = p.stats()
        if r <= 0:
            return False
        inner = float(-p.sdf.min())
        g = self.P["dot_size"] * self.f.scale
        return inner < 2.2 * g

    def build(self):
        self.paint_sky()
        for sh in self.ctx.shapes:
            if sh.kind in ("sun", "moon", "stars", "cloud") or not sh.region:
                self.paint_shape(sh)
            else:
                self.paint_region(sh)
        return self


def _halo(T, P, g):
    L = luminance(T).astype(np.float32)
    sig = max(1.0, P["halo_radius"] * g)
    Lb = ndimage.gaussian_filter(L, sig)
    delta = (L - Lb) * P["halo"] * 1.4
    up = np.clip(delta, 0, None)[..., None]
    dn = np.clip(-delta, 0, None)[..., None]
    out = T + (1 - T) * up - T * dn
    return np.clip(out, 0, 1).astype(np.float32), delta, Lb


def _sample(arr, pts):
    H, W = arr.shape[:2]
    xi = np.clip(pts[:, 0].astype(np.int64), 0, W - 1)
    yi = np.clip(pts[:, 1].astype(np.int64), 0, H - 1)
    return arr[yi, xi]


def render_array(scene, seed=None, params=None):
    P = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, P, NAME)
    H, W = ctx.frame.shape
    sc = ctx.frame.scale
    rng = ctx.rng
    pal = _Palette(scene, P)
    g = P["dot_size"] * sc

    study = _Study(ctx, pal).build()
    T, delta, Lb = _halo(study.T, P, g)
    Tb = ndimage.gaussian_filter(study.T, (P["halo_radius"] * g, P["halo_radius"] * g, 0))

    mixer = PigmentMixer(division=P["division"])
    grow = P["foreground_grow"]
    main_d = np.array([g * (1 - grow / 2), g, g * (1 + grow / 2)], np.float32)
    det_d = g * P["detail_scale"]
    diams = np.unique(np.round(np.concatenate([np.linspace(main_d[0], main_d[-1], 5), [det_d, det_d * 1.25]]), 2))
    bank = DotBank(diams, rng.child("dots"))

    # Ground: a warm primed canvas.
    weave = canvas_weave((H, W), rng.child("weave"), period=max(3.0, 4.5 * sc))
    ground_rgb = hex_to_rgb("#efe4c8") if not scene.palette.mood == "night" else hex_to_rgb("#d8ccb0")
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = ground_rgb
    canvas *= (0.94 + 0.12 * weave * P["ground_show"] * 0.5 + (1 - 0.5 * P["ground_show"]) * 0.06)[..., None]
    height = np.zeros((H, W), np.float32)

    sun_hue = float(rgb_to_hsv(pal.sun_rgb)[0])
    counts = {"main": 0, "detail": 0, "complement": 0, "sun": 0, "seam": 0}
    passes = int(P["passes"])
    Ls = luminance(T)
    for k in range(passes + 1):
        r = rng.child("pass", str(k))
        pts = jittered_grid((H, W), g * P["spacing"], r.child("grid"))
        if k == passes:
            # A last session only where the study is dark, so the ground does not sparkle there.
            keep = r.random(len(pts)) < P["dark_fill"] * np.clip((0.45 - _sample(Ls, pts)) / 0.3, 0, 1)
            pts = pts[keep]
        pts = pts[r.permutation(len(pts))]
        n = len(pts)
        tc = _sample(T, pts)
        pig = mixer.sample(tc + r.normal(0, 0.012, tc.shape), r.child("mix"))
        s = _sample(study.S, pts)
        lit = _sample(study.lit, pts)
        loc = _sample(study.local, pts)
        lum_t = luminance(tc)
        u = r.random((n, 3))
        # Complementary dots in the shadows.
        pc = P["complement"] * np.clip((s - 0.3) / 0.6, 0, 1)
        cm = u[:, 0] < pc
        if cm.any():
            hue = (rgb_to_hsv(loc[cm])[:, 0] + 0.5) % 1.0
            pig[cm] = mixer.by_hue(hue, lum_t[cm] * 0.9, r.child("comp"))
            counts["complement"] += int(cm.sum())
        # Dots of the light's colour on lit surfaces.
        ps = P["sunlight"] * lit * (0.5 + 0.5 * pal.warmth) * (scene.palette.mood != "night")
        sm = (u[:, 1] < ps) & ~cm
        if sm.any():
            pig[sm] = mixer.by_hue(np.full(sm.sum(), sun_hue), np.clip(lum_t[sm] + 0.12, 0, 0.95), r.child("sun"), 0.08)
            counts["sun"] += int(sm.sum())
        # Seams: complement of the colour across a contrasting edge.
        dl = _sample(delta, pts)
        pe = P["seam"] * np.clip((np.abs(dl) - 0.03) * 5, 0, 1)
        em = (u[:, 2] < pe) & ~cm & ~sm
        if em.any():
            other = np.clip(_sample(Tb, pts[em]) * 2 - _sample(study.T, pts[em]), 0, 1)
            hue = (rgb_to_hsv(other)[:, 0] + 0.5) % 1.0
            pig[em] = mixer.by_hue(hue, lum_t[em], r.child("seam"))
            counts["seam"] += int(em.sum())
        cols = mixer.rgb[pig] * (1 + r.normal(0, 0.03, (n, 1))).astype(np.float32)
        cols = np.clip(cols, 0, 1)
        yfrac = np.clip(pts[:, 1] / H, 0, 1)
        dd = g * (1 + grow * (yfrac - 0.5)) * r.uniform(0.92, 1.08, n)
        bins = bank.bin_of(dd)
        var = r.integers(0, bank.n_var, n)
        splat(canvas, pts, cols, bank, bins, var, opacity=r.uniform(0.9, 1.0, n), height=height)
        counts["main"] += n

    # Detail session: small dots that follow thin parts.
    r = rng.child("detail")
    dpts, dcols = [], []
    for i, (p, base, kind) in enumerate(study.details):
        inside = p.sdf < 0
        area = float(inside.sum())
        if area <= 0:
            continue
        wid = max(1.0, 2.0 * float(-p.sdf.min()))
        nd = int(np.clip(area / (det_d * min(det_d, wid)) * 1.3, 3, 40000))
        ys, xs = np.nonzero(inside)
        rr = r.child(str(i))
        pick = rr.integers(0, len(xs), nd)
        pts = np.stack([xs[pick] + rr.random(nd), ys[pick] + rr.random(nd)], 1).astype(np.float32)
        tc = _sample(T, pts) if base is None else np.broadcast_to(np.asarray(base, np.float32), (nd, 3))
        pig = mixer.sample(tc, rr.child("mix"))
        dpts.append(pts)
        dcols.append(mixer.rgb[pig])
    if dpts:
        pts = np.concatenate(dpts)
        cols = np.concatenate(dcols)
        n = len(pts)
        order = r.permutation(n)
        pts, cols = pts[order], cols[order]
        dd = det_d * r.uniform(0.9, 1.25, n)
        splat(canvas, pts, cols, bank, bank.bin_of(dd), r.integers(0, bank.n_var, n), height=height)
        counts["detail"] += n

    # Surface: canvas weave where the dots are thin, faint dab relief.
    if P["relief"] > 0:
        hb = ndimage.gaussian_filter(height, 0.7)
        canvas *= emboss(hb, strength=0.35 * P["relief"])[..., None]
    img = np.clip(canvas, 0, 1).astype(np.float32)

    ctx.info.update({
        "dots": counts, "dot_px": round(float(g), 2), "pigments": len(mixer.names),
        "ink_count": len(mixer.names), "mixture_bins": len(mixer._cache), "details": len(study.details),
    })
    return img, ctx


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)
