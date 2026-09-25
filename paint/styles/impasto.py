"""Impasto oil, in the manner of late-1880s post-impressionist technique.

Painting model
--------------
Everything is an opaque, thick, directional stroke laid with the batched
stamping engine in ``core.paintbody``; nothing is a flat fill.

* **Zones.** The sky (or, indoors, the wall) is one zone covering the whole
  canvas; every shape part is another. Zones are painted far to near. Each
  zone has its own *colour field* (what colour a stroke picks up where it
  lands) and its own *flow field* (which way strokes run). Strokes are
  clipped to their zone's SDF with a little tolerance, so silhouettes stay
  exact while edges stay ragged.
* **Layers per zone.** Underpainting of big, darker strokes; the main layer of
  strokes sized to the zone (small objects get small strokes so they keep
  their shape); an accent layer of thick warm highlights on the lit side and
  cool strokes in the shadow.
* **Flow.** The sky flow is the curl of a stream function (wind + noise +
  vortices at suns, moons, stars and cloud centres), so colour bands drawn on
  the stream function's level sets line up exactly with the strokes. Ground
  and water run horizontally along the terrain; objects are stroked along
  their own form (tangent to their SDF contours, ``FlowBuilder.along_sdf``)
  plus a kind-specific bias (trunks and towers vertical, hulls horizontal).
* **Colour.** Scene palette hints are pulled towards canonical pigments per
  role and pushed in saturation. Form is warm lit side vs. cool shadow side,
  quantised into three bands from ``shade_field``; strokes sample it and
  jitter in hue and value, with a second colour streaked through the bristles.
* **Contours.** Broken Prussian-blue dashes traced along each major part's
  zero contour (traced along the SDF tangent, then snapped back onto it).
* **Relief.** Every stroke deposits height (bristle ridges, heavier at the
  start); the height map is lit with ``emboss``, and the canvas weave shows
  where the paint is thin.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.color import hex_to_rgb, luminance, mix
from ..core.flow import FlowField
from ..core.noise import value_noise
from ..core.paintbody import BristleBank, PaintBody, jitter, rgb_to_hsv, vivid
from ..core.texture import canvas_weave, emboss
from ..scene.shapes import shade_field
from .base import ROLE_TARGETS, clamp_params, encode, make_context, merge_params, scene_hints

NAME = "impasto"

DEFAULTS = {
    "stroke_width": 13.0,      # max stroke width in px at 1024 px height (sky / ground)
    "stroke_aspect": 3.6,      # stroke length / width
    "min_width": 2.6,          # smallest stroke (details), px at 1024
    "object_scale": 0.55,      # stroke width as a fraction of a part's inner radius
    "density": 1.7,            # main-layer coverage (strokes per unit area of stroke)
    "underpaint": 0.8,         # underpainting coverage
    "underpaint_scale": 1.8,   # underpainting stroke size multiplier
    "accent": 0.4,             # accent (highlight / shadow) layer coverage on objects
    "swirl": 1.0,              # vortex strength in skies
    "curl": 0.5,               # noise wander of all flows
    "contour_weight": 1.0,     # dark broken outline width multiplier (0 = off)
    "contour_gaps": 0.3,       # fraction of contour dashes left out
    "relief": 1.3,             # emboss strength of the paint height map
    "color_jitter": 1.0,       # per-stroke hue / value jitter
    "saturation": 1.35,        # saturation boost on palette colours
    "warm_cool": 0.75,         # warm lit side vs cool shadow side split
    "two_color": 0.55,         # second colour streaked through the bristles
    "canvas_show": 0.6,        # canvas weave visibility where paint is thin
    "halo_rings": 1.0,         # concentric halo strokes around suns, moons, stars
}

KNOBS = {
    "stroke_width": (6.0, 24.0, "max stroke width; bigger is bolder and coarser"),
    "stroke_aspect": (1.5, 6.0, "stroke length relative to width; short dabs vs long dashes"),
    "min_width": (1.5, 5.0, "smallest stroke for thin details"),
    "object_scale": (0.25, 1.0, "object stroke width relative to object thickness"),
    "density": (0.8, 3.0, "main-layer stroke coverage"),
    "underpaint": (0.0, 1.5, "underpainting coverage"),
    "underpaint_scale": (1.0, 3.0, "underpainting stroke size"),
    "accent": (0.0, 1.0, "highlight / shadow accent strokes on objects"),
    "swirl": (0.0, 2.5, "vortex strength around suns, moons, stars, clouds"),
    "curl": (0.0, 1.5, "noise wander in the flow fields"),
    "contour_weight": (0.0, 2.5, "dark broken contour width"),
    "contour_gaps": (0.0, 0.8, "how broken the contours are"),
    "relief": (0.0, 2.5, "paint relief / emboss strength"),
    "color_jitter": (0.0, 2.5, "per-stroke hue and value jitter"),
    "saturation": (0.8, 1.8, "palette saturation boost"),
    "warm_cool": (0.0, 1.0, "warm lit / cool shadow colour split"),
    "two_color": (0.0, 1.0, "two-colour brush loading"),
    "canvas_show": (0.0, 1.5, "canvas weave in thin paint"),
    "halo_rings": (0.0, 1.5, "halo ring strength around lights"),
}

RULES = {"min_edge_density": 0.08, "min_mean_saturation": 0.25}

PIG = {k: hex_to_rgb(v) for k, v in {
    "cobalt": "#1f4fa3", "ultramarine": "#2a3a8f", "prussian": "#15243a", "cerulean": "#4a8ccc",
    "viridian": "#2f7d62", "sap": "#4f8a2e", "chrome": "#f2b61c", "lemon": "#f6e27a", "ochre": "#c98a2e",
    "vermilion": "#d2402a", "orange": "#e8792b", "violet": "#4b3d8f", "lead_white": "#f4ecd8",
    "umber": "#5e3d22", "sienna": "#8a4a26", "ground": "#7a6446", "wood": "#a8622e",
}.items()}

ROLE_PIG = {"sky": "cerulean", "water": "cobalt", "ground": "ochre", "primary": "cobalt", "secondary": "viridian",
            "accent": "vermilion", "dark": "prussian", "light": "lead_white", "foliage": "viridian",
            "wood": "sienna", "stone": "violet"}
ROLE_PULL = {"foliage": 0.45, "wood": 0.45, "stone": 0.25, "dark": 0.55, "water": 0.3, "ground": 0.25,
             "light": 0.25, "sky": 0.3, "accent": 0.2, "primary": 0.12, "secondary": 0.12}

# Parts that get no dark outline (tone bands, glows, regions, tiny marks).
NO_CONTOUR = {"stripes", "band", "snow", "beam", "water", "ground", "stars", "lantern", "tea", "disc",
              "crescent", "cloud", "front", "leaves", "leaf", "nodes", "reeds", "rain", "birds"}
REGION_WATER = {"sea", "river"}


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

class _Palette:
    def __init__(self, scene, P):
        self.hints = scene_hints(scene)
        self.mood = scene.palette.mood
        self.P = P
        self.boost = P["saturation"]
        self.warmth = scene.light.warmth
        lum = [luminance(h) for h in self.hints]
        self.darkest = self.hints[int(np.argmin(lum))]
        self.lightest = self.hints[int(np.argmax(lum))]
        self.rainy = any(o.kind == "rain" for o in scene.objects)

    def v(self, c):
        return vivid(np.asarray(c, np.float32), self.boost, 0.04)

    def role(self, role: str, avoid=None, obj_color=None):
        if obj_color and obj_color.startswith("#"):
            return self.v(hex_to_rgb(obj_color))
        if obj_color and obj_color in ROLE_TARGETS:
            role = obj_color
        target = hex_to_rgb(ROLE_TARGETS.get(role, ROLE_TARGETS["primary"]))
        w = np.array([0.3, 0.45, 0.25], np.float32)
        dist = [float(np.sum((c - target) ** 2 * w) + 0.6 * (luminance(c) - luminance(target)) ** 2) for c in self.hints]
        pig = PIG[ROLE_PIG.get(role, "cobalt")]
        pull = ROLE_PULL.get(role, 0.15)
        out = None
        for i in np.argsort(dist):
            c = self.v(mix(self.hints[i], pig, pull))
            if avoid is None or np.sqrt(np.sum((c - avoid) ** 2)) > 0.22:
                out = c
                break
        if out is None:
            c = self.v(mix(self.hints[int(np.argmin(dist))], pig, pull))
            # Still too close to what is behind: push value away from it.
            out = c * 0.55 if luminance(avoid) > 0.45 else c + (1 - c) * 0.45
        return np.clip(out, 0, 1).astype(np.float32)

    def lit(self, c, amt=1.0):
        """Warm, lighter version of c ((3,) or (..., 3))."""
        c = np.asarray(c, np.float32)
        a = self.P["warm_cool"] * (0.25 + 0.3 * self.warmth) * amt
        hsv = rgb_to_hsv(c)
        cool_c = (hsv[..., 0] > 0.4) & (hsv[..., 0] < 0.8) & (hsv[..., 1] > 0.15)
        warm = np.where((luminance(c) > 0.5)[..., None], PIG["lemon"], PIG["chrome"])
        # Blues and greens lit with yellow turn to mud; lift them towards a warm white instead.
        pale = mix(PIG["lead_white"], PIG["lemon"], 0.25)
        tgt = np.where(cool_c[..., None], pale, warm)
        a = np.where(cool_c, a * 0.8, a)[..., None] if np.ndim(c) > 1 else (a * 0.8 if cool_c else a)
        w = c + (tgt - c) * a
        return np.clip(w + (1 - w) * 0.08 * amt, 0, 1).astype(np.float32)

    def shadow(self, c, amt=1.0):
        """Cool, darker version of c: warm colours get violet shadows, blues deepen to ultramarine."""
        c = np.asarray(c, np.float32)
        a = self.P["warm_cool"] * (0.35 + 0.2 * (1 - self.warmth)) * amt
        hsv = rgb_to_hsv(c)
        is_warm = (hsv[..., 0] < 0.45) | (hsv[..., 1] < 0.15)
        cool = np.where(is_warm[..., None], PIG["violet"], PIG["ultramarine"])
        d = c * (1 - 0.3 * amt)
        return np.clip(d + (cool - d) * a, 0, 1).astype(np.float32)

    def pick_contrast(self, avoid, pull_to="ochre", pull=0.25, extra=()):
        """The candidate colour farthest from everything in ``avoid`` (hints preferred)."""
        cands = [(self.v(mix(h, PIG[pull_to], pull)), 0.06) for h in self.hints]
        cands += [(self.v(PIG[k]), 0.0) for k in extra]
        cands = [(c, b) for c, b in cands if 0.18 < luminance(c) < 0.8]
        if not cands:
            return self.v(PIG[pull_to])
        score = [min(np.sqrt(np.sum((c - a) ** 2)) for a in avoid) + b if avoid else b for c, b in cands]
        return cands[int(np.argmax(score))][0].astype(np.float32)

    def sky(self, interior: bool):
        out = self._sky(interior)
        if self.rainy and not interior:
            grey = hex_to_rgb("#8c95a3")
            out = tuple(np.clip(mix(c, grey * (0.8 + 0.4 * luminance(c)), 0.4), 0, 1).astype(np.float32) for c in out)
        return out

    def _sky(self, interior: bool):
        h = self.hints
        if interior:
            # A wall behind a still life: a light, strongly hued ground.
            wall = self.role("light")
            wall = self.v(mix(wall, PIG["chrome"] if self.warmth >= 0.5 else PIG["cerulean"], 0.35))
            return wall, self.v(mix(wall, PIG["ochre"], 0.25)), mix(wall, PIG["lead_white"], 0.45), \
                self.shadow(wall, 0.7)
        if self.mood == "night":
            top = self.v(mix(self.darkest, PIG["ultramarine"], 0.55))
            blues = sorted(h, key=lambda c: -(c[2] - c[0]))
            bottom = self.v(mix(blues[0], PIG["cobalt"], 0.4))
            return top, bottom, self.v(mix(bottom, PIG["cerulean"], 0.5)) * 1.15, top * 0.7
        if self.mood == "warm":
            sk = self.role("sky")
            top = self.v(mix(sk, PIG["orange"], 0.3))
            bottom = self.v(mix(sk, PIG["lemon"], 0.35))
            return top, bottom, mix(bottom, PIG["lead_white"], 0.4), self.v(mix(top, PIG["vermilion"], 0.3))
        bluest = sorted(h, key=lambda c: -(c[2] - c[0]) + 0.2 * abs(luminance(c) - 0.55))[0]
        # A palette with no real blue still gets a sky: lean harder on cerulean.
        k = 0.5 if (bluest[2] - bluest[0]) > 0.15 else 0.72
        blue = self.v(mix(bluest, PIG["cerulean"], k))
        top = np.clip(blue * 1.05, 0, 1)
        bottom = self.v(mix(self.lightest, PIG["cerulean"], 0.25))
        return top, bottom, mix(bottom, PIG["lead_white"], 0.5), self.v(mix(top, PIG["cobalt"], 0.5))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aniso_noise(shape, cy, cx, rng):
    """Noise stretched horizontally (cells cy tall, cx wide)."""
    H, W = shape
    w = max(2, int(np.ceil(W * cy / cx)))
    n = value_noise((H, w), cy, rng)
    out = ndimage.zoom(n, (1, W / w), order=1)
    if out.shape[1] < W:
        out = np.pad(out, ((0, 0), (0, W - out.shape[1])), mode="edge")
    return out[:, :W]


def _tangent(sdf, sigma=1.5):
    g = ndimage.gaussian_filter(np.clip(sdf, -4000, 4000).astype(np.float32), sigma)
    gy, gx = np.gradient(g)
    n = np.hypot(gx, gy) + 1e-6
    return -gy / n, gx / n, gx / n, gy / n


def _sample(arr, xs, ys):
    H, W = arr.shape[:2]
    yi = np.clip(ys.astype(int), 0, H - 1)
    xi = np.clip(xs.astype(int), 0, W - 1)
    return arr[yi, xi]


def _seeds(sdf, bbox, spacing, rng, thin_ok=True):
    """Jittered-grid seed points inside ``sdf < 0`` (plus random pixels for thin parts)."""
    x0, y0, x1, y1 = bbox
    H, W = sdf.shape
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    pts = np.zeros((0, 2), np.float32)
    if len(xs) and len(ys):
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        pts = pts + rng.uniform(-0.5, 0.5, pts.shape) * spacing
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        pts = pts[_sample(sdf, pts[:, 0], pts[:, 1]) < 0]
    if thin_ok:
        ys_, xs_ = np.nonzero(sdf[y0:y1, x0:x1] < 0)
        want = int(len(xs_) / (spacing * spacing))
        if len(xs_) and len(pts) < 0.7 * want + 2:
            k = max(want - len(pts), 3)
            pick = rng.integers(0, len(xs_), k)
            extra = np.stack([xs_[pick] + x0 + rng.random(k), ys_[pick] + y0 + rng.random(k)], 1)
            pts = np.concatenate([pts, extra])
    return pts[rng.permutation(len(pts))].astype(np.float32)


class _Painter:
    """Per-render state: canvas, flows, palette, stroke bookkeeping."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.P = P = ctx.params
        self.f = ctx.frame
        self.sc = ctx.frame.scale
        self.H, self.W = ctx.frame.shape
        self.rng = ctx.rng
        self.pal = _Palette(ctx.scene, P)
        wmax = P["stroke_width"] * self.sc
        lmax = wmax * P["underpaint_scale"] * 1.2 * P["stroke_aspect"]
        margin = int(0.62 * max(lmax, 110 * self.sc) + wmax * P["underpaint_scale"] * 1.2 + 16)
        self.bank = BristleBank(self.rng.child("bristles"))
        self.body = PaintBody((self.H, self.W), margin, PIG["ground"], self.bank)
        # Two shared curl-noise flows, reused by every zone.
        self.gnoise = []
        for i, cell in enumerate((260.0, 90.0)):
            c = cell * self.sc
            pot = value_noise(self.f.shape, c, self.rng.child("gnoise", str(i)))
            gy, gx = np.gradient(pot)
            n = np.hypot(gx, gy).mean() + 1e-9
            self.gnoise.append((gy / n, -gx / n))
        self.dither = value_noise(self.f.shape, 14 * self.sc, self.rng.child("dither"))
        self.counts = {"under": 0, "main": 0, "accent": 0, "contour": 0, "star": 0}
        self.horizon_px = ctx.scene.horizon * self.H

    # -- strokes -----------------------------------------------------------
    def strokes(self, layer, seeds, flow, widths, lengths, field, rng, clip_sdf=None, tol=0.8,
                color_fn=None, body=1.0, opacity=1.0, jit=1.0, tip=0.6, steps=5):
        n = len(seeds)
        if n == 0:
            return
        P = self.P
        # FlowField.trace broadcasts ``step``, so each stroke integrates at its own step length.
        paths = flow.trace(seeds, steps=steps, step=(lengths / steps)[:, None].astype(np.float32), both=True)
        c1 = _sample(field, seeds[:, 0], seeds[:, 1])
        # The second colour comes from a neighbour across the flow (picks up adjacent bands).
        d = paths[:, -1] - paths[:, 0]
        dn = np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / dn
        off = nrm * (widths[:, None] * rng.uniform(0.8, 2.0, (n, 1)) * np.sign(rng.normal(size=(n, 1))))
        c2 = _sample(field, seeds[:, 0] + off[:, 0], seeds[:, 1] + off[:, 1])
        if color_fn is not None:
            c1, c2 = color_fn(c1), color_fn(c2)
        cj = P["color_jitter"] * jit
        c1 = jitter(c1, rng.child("j1"), 0.012 * cj, 0.06 * cj, 0.06 * cj)
        c2 = jitter(c2, rng.child("j2"), 0.02 * cj, 0.08 * cj, 0.08 * cj)
        self.body.paint(paths, widths, c1, c2, rng=rng.child("paint"), streak_amt=P["two_color"],
                        clip_sdf=clip_sdf, tol=tol, opacity=opacity, body=body, tip=tip)
        self.counts[layer] += n

    def zone(self, sdf, bbox, flow, field, w, rng, inr, perspective=None, accent_shade=None, detail=False,
             under=True, clip=True, tol=0.9, aspect=1.0):
        """Paint one zone: underpainting, main layer, accents."""
        P, sc = self.P, self.sc
        asp = P["stroke_aspect"] * aspect
        clip_sdf = sdf if clip else None

        def sizes(pts, scale, r):
            n = len(pts)
            ws = np.full(n, w * scale, np.float32) * r.uniform(0.8, 1.15, n)
            if perspective is not None:
                ws *= perspective(pts[:, 1])
            ws = np.maximum(ws, P["min_width"] * sc * 0.8)
            ls = ws * asp * r.uniform(0.75, 1.25, n)
            if detail:
                ls = np.maximum(ls, np.maximum(4 * inr, 16 * sc))
            return ws.astype(np.float32), ls.astype(np.float32)

        def seeds(sp, r, thin_ok=True):
            if perspective is None:
                return _seeds(sdf, bbox, sp, r, thin_ok)
            # Strokes shrink with distance, so seed spacing shrinks with them (even coverage).
            out = []
            ys = np.linspace(bbox[1], bbox[3], 7)
            for k in range(6):
                y0, y1 = int(ys[k]), int(ys[k + 1])
                if y1 > y0:
                    fac = float(perspective(np.array([(y0 + y1) / 2.0]))[0])
                    out.append(_seeds(sdf, (bbox[0], y0, bbox[2], y1), sp * fac, r.child(str(k)), thin_ok))
            pts = np.concatenate(out) if out else np.zeros((0, 2), np.float32)
            return pts[r.permutation(len(pts))]

        if under and P["underpaint"] > 0 and inr > 2.5 * w:
            wu = w * P["underpaint_scale"]
            r = rng.child("under")
            sp = np.sqrt(wu * wu * asp * 0.7 / P["underpaint"])
            pts = seeds(sp, r, thin_ok=False)
            ws, ls = sizes(pts, P["underpaint_scale"], r)
            self.strokes("under", pts, flow, ws, ls, field, r, clip_sdf, tol + 1.5 * sc,
                         color_fn=lambda c: mix(c * 0.85, PIG["umber"], 0.12), body=0.8, jit=1.3)
        r = rng.child("main")
        sp = np.sqrt(w * w * asp * 0.7 / P["density"])
        pts = seeds(sp, r)
        ws, ls = sizes(pts, 1.0, r)
        self.strokes("main", pts, flow, ws, ls, field, r, clip_sdf, tol * sc)
        if accent_shade is not None and P["accent"] > 0:
            r = rng.child("accent")
            wa = w * 0.75
            sp = np.sqrt(wa * wa * asp * 0.7 / P["accent"])
            pts = _seeds(sdf, bbox, sp, r)
            if len(pts):
                sh = _sample(accent_shade, pts[:, 0], pts[:, 1])
                lit = pts[sh < 0.3]
                dark = pts[sh > 0.7][: max(1, len(lit) // 2)]
                for name, pp, fn, body in (
                    ("lit", lit, lambda c: self.pal.lit(c, 1.0), 1.7),
                    ("dark", dark, lambda c: self.pal.shadow(c, 0.6), 1.2),
                ):
                    if len(pp):
                        rr = r.child(name)
                        ws, ls = sizes(pp, 0.75, rr)
                        self.strokes("accent", pp, flow, ws, ls * 0.8, field, rr, clip_sdf, 0.0,
                                     color_fn=fn, body=body)

    # -- flows -------------------------------------------------------------
    def part_flow(self, sdf, reach, uni=None, uni_w=0.0, curl=None, fine=False, rng=None):
        tx, ty, _, _ = _tangent(sdf)
        fall = np.exp(-np.abs(np.clip(sdf, -4000, 4000)) / max(reach, 1.0)).astype(np.float32)
        curl = self.P["curl"] if curl is None else curl
        nx, ny = self.gnoise[1 if fine else 0]
        # (+1e-3: a fallback direction where every contribution vanishes, see _sky.)
        vx = tx * fall + nx * curl * 0.35 + 1e-3
        vy = ty * fall + ny * curl * 0.35
        if uni is not None and uni_w > 0:
            vx = vx + np.cos(uni) * uni_w
            vy = vy + np.sin(uni) * uni_w
        return FlowField(vx.astype(np.float32), vy.astype(np.float32))


# ---------------------------------------------------------------------------
# Lights (sun, moon, stars, clouds) -> vortices and halos
# ---------------------------------------------------------------------------

def _lights(ctx):
    out = []  # (kind, x, y, R)
    for s in ctx.shapes:
        if s.kind in ("sun", "moon"):
            out.append((s.kind, s.cx, s.cy, s.radius))
        elif s.kind == "cloud":
            out.append(("cloud", s.cx, s.cy - s.radius * 0.1, s.radius * 0.7))
        elif s.kind == "stars":
            part = s.parts[0]
            lab, n = ndimage.label(part.sdf < 0)
            if n:
                idx = np.arange(1, n + 1)
                pos = ndimage.minimum_position(part.sdf, lab, idx)
                mins = ndimage.minimum(part.sdf, lab, idx)
                for (y, x), m in zip(pos, mins):
                    out.append(("star", float(x) + 0.5, float(y) + 0.5, max(float(-m), 1.5)))
    return out


def _sky(p: _Painter, ctx, lights, interior):
    """Stream-function sky: flow = curl(psi); colour bands on psi's level sets."""
    P, sc, f = p.P, p.sc, p.f
    H, W = f.shape
    X, Y = f.grid
    rng = p.rng.child("sky")
    hz = max(p.horizon_px, H * 0.2)
    # Indoors a faint drift still keeps every streamline moving: value_noise is clipped to [0, 1], so its
    # curl is exactly zero on the clipped plateaus and strokes there would stall into dots.
    psi = (Y * (0.2 if interior else 1.0)).astype(np.float32)
    cell = 300 * sc
    psi += (value_noise(f.shape, cell, rng.child("psi")) - 0.5) * cell * (0.5 + P["curl"] * (1.5 if interior else 0.8))
    vr = rng.child("vortex")
    for i, (kind, x, y, R) in enumerate(lights):
        if kind in ("sun", "moon"):
            Rv, st = R * 1.4, 2.4
        elif kind == "star":
            Rv, st = R * 4.5, 2.0
        else:
            Rv, st = R * 0.9, 1.6
        sgn = 1.0 if vr.random() < 0.6 else -1.0
        r = np.hypot(X - x, Y - y)
        psi += sgn * P["swirl"] * st * Rv * 2.0 * (1 - np.exp(-(r / (2 * Rv)) ** 2))
    gy, gx = np.gradient(psi)
    flow = FlowField((gy + 1e-3).astype(np.float32), (-gx).astype(np.float32))

    top, bottom, light, dark = p.pal.sky(interior)
    t = np.clip(Y / hz, 0, 1)[..., None] ** 0.9
    base = top[None, None] * (1 - t) + bottom[None, None] * t
    lam = 42 * sc
    band = np.sin(2 * np.pi * psi / lam + (p.dither - 0.5) * 2.5)
    lb = (band > (0.7 if interior else 0.5))[..., None]
    db = (band < (-0.8 if interior else -0.6))[..., None]
    field = np.where(lb, base * 0.6 + light * 0.4, np.where(db, base * 0.7 + dark * 0.3, base))
    # Halos: concentric rings of light around suns, moons and stars.
    night = ctx.scene.palette.mood == "night"
    for kind, x, y, R in lights:
        if kind == "cloud" or P["halo_rings"] <= 0:
            continue
        r = np.hypot(X - x, Y - y)
        if kind == "star":
            Rh, per = R * 5.0, R * 1.4
            c1, c2 = PIG["lemon"], mix(PIG["lead_white"], PIG["lemon"], 0.3)
            fade = np.exp(-np.clip(r - R, 0, None) / (Rh * 0.45))
        else:
            Rh, per = R * 2.6, R * 0.42
            if kind == "sun":
                c1, c2 = PIG["lemon"], mix(PIG["chrome"], PIG["orange"], 0.3)
            else:
                c1, c2 = mix(PIG["lemon"], PIG["lead_white"], 0.4), (PIG["chrome"] if night else PIG["lemon"])
            fade = np.exp(-np.clip(r - R, 0, None) / (R * 1.1))
        ring = np.cos(2 * np.pi * (r - R) / per + (p.dither - 0.5) * 1.5)
        hc = np.where((ring > 0)[..., None], c1, c2)
        a = np.clip(fade * 0.9 * P["halo_rings"], 0, 1) * (r < Rh * 1.6)
        a = np.where(ring[..., None] < -0.7, a[..., None] * 0.3, a[..., None])
        field = field * (1 - a) + hc * a
    field = field.astype(np.float32)
    w = P["stroke_width"] * sc
    # The sky stops a little below the highest full-width ground region (sea, field, table), so its
    # colours never glint through the dry-brush gaps of the ground painted over it.
    tops = [s.parts[0].stats()[3][1] for s in ctx.shapes if s.kind in ("sea", "field", "table") and s.parts]
    if tops:
        cut = min(tops) + 10 * sc
        zone_sdf = (Y - cut).astype(np.float32)
        bbox = (0, 0, W, int(min(H, cut + 1)))
    else:
        zone_sdf, bbox = np.full(f.shape, -1e4, np.float32), (0, 0, W, H)
    p.zone(zone_sdf, bbox, flow, field, w, rng.child("zone"), inr=1e4, clip=bool(tops), aspect=1.25)
    return flow


# ---------------------------------------------------------------------------
# Colour fields per part
# ---------------------------------------------------------------------------

def _water_field(p, part, base, lights, rng):
    f, sc = p.f, p.sc
    H, W = f.shape
    X, Y = f.grid
    _, y0, _, _ = part.stats()[3]
    top = max(y0, 0)
    dt = np.clip((Y - top) / max(H - top, 1), 0, 1)
    n1 = _aniso_noise(f.shape, 7 * sc, 60 * sc, rng.child("n1"))
    n2 = _aniso_noise(f.shape, 22 * sc, 150 * sc, rng.child("n2"))
    n = n1 * (1 - dt) + n2 * dt
    deep = p.pal.shadow(base, 0.9)
    light = mix(base, PIG["cerulean"] * 1.1, 0.45)
    top_, bottom, sky_light, _ = p.pal.sky(False)
    field = np.where((n < 0.36)[..., None], deep, np.where((n < 0.62)[..., None], base, light))
    near_h = (np.exp(-dt / 0.12) * 0.55)[..., None]
    field = field * (1 - near_h) + mix(bottom, base, 0.35) * near_h
    for kind, x, y, R in lights:
        if kind not in ("sun", "moon") or y > top:
            continue
        half = R * (0.5 + 0.9 * dt) * (0.6 + 0.8 * n2)
        dash = _aniso_noise(f.shape, 5 * sc, 26 * sc, rng.child("refl", str(int(x))))
        m = ((np.abs(X - x) < half) & (dash > 0.42) & (Y > top + 2))[..., None]
        refl = PIG["lemon"] if kind == "sun" else mix(PIG["lemon"], PIG["lead_white"], 0.4)
        refl2 = PIG["chrome"] if kind == "sun" else PIG["lemon"]
        field = np.where(m, np.where((dash > 0.6)[..., None], refl, refl2), field)
    return field.astype(np.float32)


def _ground_field(p, part, base, rng):
    f, sc = p.f, p.sc
    H, W = f.shape
    Y = f.grid[1]
    _, y0, _, _ = part.stats()[3]
    dt = np.clip((Y - y0) / max(H - y0, 1), 0, 1)
    n = _aniso_noise(f.shape, 10 * sc, 90 * sc, rng.child("n1")) * (1 - dt) + \
        _aniso_noise(f.shape, 30 * sc, 160 * sc, rng.child("n2")) * dt
    green = p.pal.role("foliage")
    yellow = p.pal.v(mix(base, PIG["chrome"], 0.45))
    tones = [p.pal.shadow(base, 0.5), mix(base, green, 0.55), base, yellow, p.pal.lit(yellow, 0.6)]
    q = np.clip((n * 1.25 - 0.12) * len(tones), 0, len(tones) - 1).astype(int)
    field = np.zeros(f.shape + (3,), np.float32)
    for i, c in enumerate(tones):
        field = np.where((q == i)[..., None], c, field)
    far = (np.exp(-dt / 0.15) * 0.3)[..., None]
    return (field * (1 - far) + mix(base, PIG["cerulean"], 0.4) * far).astype(np.float32)


def _banded_field(p, part, base, rng, period):
    """Tone bands parallel to the part's outline (hills, table top)."""
    dd = np.clip(-part.sdf, 0, None)
    n = p.dither
    q = np.floor(dd / period + n * 1.8).astype(int) % 3
    tones = [base, p.pal.shadow(base, 0.45), p.pal.lit(base, 0.7)]
    field = np.zeros(p.f.shape + (3,), np.float32)
    for i, c in enumerate(tones):
        field = np.where((q == i)[..., None], c, field)
    return field


def _cloth_field(p, part, base, rng):
    """Table surface: long horizontal bands of the cloth colour, its shadow and a slightly lighter tone."""
    f, sc = p.f, p.sc
    n = _aniso_noise(f.shape, 9 * sc, 120 * sc, rng.child("n"))
    tones = [p.pal.shadow(base, 0.5), base, np.clip(base * 1.12 + 0.03, 0, 1)]
    q = np.clip(n * 3, 0, 2).astype(int)
    field = np.zeros(f.shape + (3,), np.float32)
    for i, c in enumerate(tones):
        field = np.where((q == i)[..., None], c, field)
    return field


def _table_color(p, ctx, table, interior):
    """A cloth / wood colour that contrasts with the wall and every object standing on it."""
    avoid = [p.pal.sky(interior)[0], p.pal.sky(interior)[1]]
    for s in ctx.shapes:
        if s.region or s.depth >= table.depth or not s.parts:
            continue
        pt = s.parts[0]
        avoid.append(p.pal.role(pt.role, obj_color=s.obj.color))
    return p.pal.pick_contrast(avoid, "ochre", 0.2, extra=("wood", "viridian", "vermilion"))


def _form_field(p, part, base, rng, light_dir, amt=1.0):
    """Warm lit / mid / cool shadow bands from shade_field, dithered at stroke scale."""
    sh = shade_field(part, light_dir, p.f)
    s2 = sh + (p.dither - 0.5) * 0.3
    lit, sha = p.pal.lit(base, amt), p.pal.shadow(base, amt)
    field = np.where((s2 < 0.36)[..., None], lit, np.where((s2 < 0.6)[..., None], base, sha))
    return field.astype(np.float32), sh


def _flat_field(p, base, spread=0.08):
    d = (p.dither - 0.5)[..., None] * 2 * spread
    return np.clip(base[None, None] * (1 + d), 0, 1).astype(np.float32)


def _disc_field(p, shape, part, kind):
    X, Y = p.f.grid
    r = np.hypot(X - shape.cx, Y - shape.cy) / max(shape.radius, 1)
    if kind == "sun":
        cs = [mix(PIG["lemon"], PIG["lead_white"], 0.35), PIG["lemon"], PIG["chrome"], mix(PIG["chrome"], PIG["orange"], 0.5)]
    else:
        cs = [mix(PIG["lemon"], PIG["lead_white"], 0.6), mix(PIG["lemon"], PIG["lead_white"], 0.3), PIG["lemon"],
              mix(PIG["lemon"], PIG["chrome"], 0.4)]
    q = np.clip((r + (p.dither - 0.5) * 0.3) * 4, 0, 3).astype(int)
    field = np.zeros(p.f.shape + (3,), np.float32)
    for i, c in enumerate(cs):
        field = np.where((q == i)[..., None], c, field)
    return field


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------

def _contour(p: _Painter, sdf, width, rng):
    P, sc = p.P, p.sc
    if P["contour_weight"] <= 0:
        return
    band = np.abs(sdf) < 0.6
    ys, xs = np.nonzero(band)
    if len(xs) < 6:
        return
    dash = float(np.clip(width * 8, 14 * sc, 42 * sc))
    n = int(len(xs) / dash * 1.4) + 1
    pick = rng.integers(0, len(xs), n)
    keep = rng.random(n) >= P["contour_gaps"]
    pick = pick[keep]
    if len(pick) == 0:
        return
    starts = np.stack([xs[pick] + 0.5, ys[pick] + 0.5], 1).astype(np.float32)
    tx, ty, nx, ny = _tangent(sdf, 1.0)
    flow = FlowField(tx.astype(np.float32), ty.astype(np.float32))
    steps = 4
    paths = flow.trace(starts, steps=steps, step=dash / steps, both=True)
    # Snap back onto the zero contour.
    for _ in range(2):
        q = paths.reshape(-1, 2)
        coords = [np.clip(q[:, 1] - 0.5, 0, p.H - 1), np.clip(q[:, 0] - 0.5, 0, p.W - 1)]
        d = ndimage.map_coordinates(sdf, coords, order=1, mode="nearest")
        gx = ndimage.map_coordinates(nx, coords, order=1, mode="nearest")
        gy = ndimage.map_coordinates(ny, coords, order=1, mode="nearest")
        d = np.clip(d, -3 * width, 3 * width)
        q = q - np.stack([gx * d, gy * d], 1)
        paths = q.reshape(paths.shape).astype(np.float32)
    m = len(paths)
    widths = (width * rng.uniform(0.6, 1.3, m)).astype(np.float32)
    base = np.tile(mix(PIG["prussian"], np.array([0.05, 0.05, 0.08], np.float32), 0.3), (m, 1))
    cols = jitter(base, rng.child("c"), 0.02, 0.1, 0.05)
    cols2 = np.tile(PIG["ultramarine"] * 0.7, (m, 1))
    p.body.paint(paths, widths, cols, cols2, rng=rng.child("paint"), streak_amt=0.35, body=1.3, tip=0.3)
    p.counts["contour"] += m


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _inradius(sdf):
    return float(max(-np.min(sdf), 0.0))


def _stars(p: _Painter, lights, rng):
    """Star cores: tight whorls of pale strokes, clipped to a disc around each star."""
    X, Y = p.f.grid
    sc = p.sc
    stars = [(x, y, R) for k, x, y, R in lights if k == "star"]
    for i, (x, y, R) in enumerate(stars):
        r = rng.child(str(i))
        Rc = R * 2.0 + 1.5 * sc
        x0, y0 = int(max(0, x - Rc * 2)), int(max(0, y - Rc * 2))
        x1, y1 = int(min(p.W, x + Rc * 2 + 1)), int(min(p.H, y + Rc * 2 + 1))
        sdf = np.full(p.f.shape, 1e3, np.float32)
        sdf[y0:y1, x0:x1] = np.hypot(X[y0:y1, x0:x1] - x, Y[y0:y1, x0:x1] - y) - Rc
        k = 6 + int(R)
        ang = r.uniform(0, 2 * np.pi, k)
        rad = Rc * np.sqrt(r.uniform(0, 0.8, k))
        pts = np.stack([x + np.cos(ang) * rad, y + np.sin(ang) * rad], 1).astype(np.float32)
        paths = []
        for (px, py), a0 in zip(pts, ang):
            aa = a0 + np.linspace(-0.9, 0.9, 5)
            rr = np.hypot(px - x, py - y) + 1.0
            paths.append(np.stack([x + np.cos(aa) * rr, y + np.sin(aa) * rr], 1))
        paths = np.array(paths, np.float32)
        ws = np.full(k, max(Rc * 0.6, p.P["min_width"] * sc), np.float32)
        base = np.tile(mix(PIG["lead_white"], PIG["lemon"], 0.35), (k, 1))
        cols = jitter(base, r.child("c"), 0.01, 0.1, 0.04)
        cols2 = np.tile(PIG["lemon"], (k, 1))
        p.body.paint(paths, ws, cols, cols2, rng=r.child("p"), clip_sdf=sdf, tol=1.0, body=1.6, tip=0.6)
        p.counts["star"] += k


def _rain_sheet(p: _Painter, part, base, rng):
    """A veil of fine, translucent rain streaks over the whole scene, parallel to the drawn rain."""
    sc = p.sc
    n = int(260 * p.W * p.H / (1024 * 768))
    ang = np.radians(100)
    d = np.array([np.cos(ang), np.sin(ang)], np.float32)
    seeds = np.stack([rng.uniform(-0.05, 1.05, n) * p.W, rng.uniform(0, 1, n) * p.H], 1).astype(np.float32)
    ln = rng.uniform(30, 80, n).astype(np.float32) * sc
    u = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
    paths = seeds[:, None, :] + (ln[:, None] * u[None, :])[..., None] * d[None, None, :]
    light = mix(p.pal.sky(False)[2], PIG["lead_white"], 0.3)
    cols = np.where((rng.random(n) < 0.5)[:, None], light, base).astype(np.float32)
    cols = jitter(cols, rng.child("c"), 0.01, 0.05, 0.06)
    ws = rng.uniform(1.3, 2.2, n).astype(np.float32) * sc
    p.body.paint(paths, ws, cols, cols, rng=rng.child("p"), opacity=rng.uniform(0.35, 0.7, n), body=0.4, tip=0.3)
    p.counts["accent"] += n


def _kind_bias(kind, name):
    """(angle, weight) of a uniform bias added to the along-form flow."""
    V, Hz = np.pi / 2, 0.0
    if kind in ("sea", "river", "field", "table") or name in ("hull", "top", "front", "gallery"):
        return Hz, 1.2
    if name in ("trunk", "tower", "stalks", "flame") or kind in ("cliff",):
        return V, 0.8 if name != "flame" else 0.6
    if name in ("walls",):
        return V, 0.5
    if kind == "rain":
        return np.radians(100), 3.0
    return None, 0.0


def render_array(scene, seed=None, params=None):
    params = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, params, NAME)
    P = params
    p = _Painter(ctx)
    f, sc = p.f, p.sc
    H, W = f.shape
    lights = _lights(ctx)
    interior = any(s.kind == "table" for s in ctx.shapes)
    table_col = None
    # Toned ground (imprimatura): the sky's mean colour knocked down with umber, so any gap between
    # strokes reads as a shadowed crevice in the paint rather than a speck of bright canvas.
    st, sb = p.pal.sky(interior)[:2]
    p.body.C[:] = mix(mix(st, sb, 0.5), PIG["umber"], 0.45)
    night = scene.palette.mood == "night"
    _sky(p, ctx, lights, interior)
    lx, ly = ctx.light_dir
    wmax = P["stroke_width"] * sc
    orng = p.rng.child("objects")

    for si, shape in enumerate(ctx.shapes):
        rs = orng.child(str(si), shape.kind)
        kind = shape.kind
        if kind == "stars":
            _stars(p, lights, rs.child("stars"))
            continue
        far = shape.depth >= 0.84 and kind in ("mountain", "hills")
        for pi, part in enumerate(shape.parts):
            if not np.isfinite(part.sdf).any() or not (part.sdf < 0).any():
                continue
            rp = rs.child(str(pi), part.name)
            if kind == "field":
                # The ground's wavy top can dip below things standing on the horizon (mountain
                # bases); let the ground overlap them instead of leaving a seam of sky.
                part = type(part)(part.name, part.sdf - 0.012 * H - 2.0, part.role, part.detail, part.shade)
            bbox = part.stats()[3]
            inr = _inradius(part.sdf)
            region = shape.region
            # Background colour under the part, so an object never melts into what is behind it.
            x0, y0, x1, y1 = bbox
            m = part.sdf[y0:y1, x0:x1] < 0
            bg = p.body.image[y0:y1, x0:x1][m].mean(0) if m.any() else None
            role = part.role
            ocol = shape.obj.color if pi == 0 else None
            base = p.pal.role(role, avoid=None if (region or part.detail or kind in ("rain",)) else bg, obj_color=ocol)
            accent_shade = None
            perspective = None
            if kind in REGION_WATER:
                field = _water_field(p, part, base, lights, rp.child("field"))
            elif kind == "field":
                base = p.pal.v(mix(base, PIG["ochre"], 0.4))
                field = _ground_field(p, part, base, rp.child("field"))
            elif kind == "table":
                if table_col is None:
                    table_col = _table_color(p, ctx, shape, interior)
                base = table_col if part.name == "top" else p.pal.shadow(table_col, 1.3) * 0.8
                field = _cloth_field(p, part, base, rp)
            elif kind == "hills":
                base = p.pal.v(mix(base, PIG["viridian"], 0.2))
                layer = int(part.name[-1]) if part.name[-1].isdigit() else 0
                nl = sum(1 for q in shape.parts)
                if nl > 1 and layer < nl - 1:
                    base = mix(base, p.pal.sky(interior)[1], 0.3 * (nl - 1 - layer) / (nl - 1))
                field = _banded_field(p, part, base, rp, 11 * sc)
            elif kind in ("sun", "moon"):
                field = _disc_field(p, shape, part, kind)
            elif kind == "rain":
                base = mix(p.pal.role("dark"), p.pal.sky(interior)[2], 0.5)
                field = _flat_field(p, base, 0.12)
            elif part.shade and not part.detail:
                field, accent_shade = _form_field(p, part, base, rp, (lx, ly), 0.55 if night else 1.0)
            else:
                field = _flat_field(p, base, 0.1)
            if part.name == "snow":
                field, accent_shade = _form_field(p, part, mix(PIG["lead_white"], PIG["cerulean"], 0.12), rp, (lx, ly), 0.6)

            # Stroke size: scale with the part's thickness; perspective for regions.
            if region and not (kind == "table" and part.name == "front"):
                w = wmax
                top = max(bbox[1], 0)

                def perspective(y, top=top):
                    return 0.4 + 0.75 * np.clip((y - top) / max(H - top, 1), 0, 1)
            else:
                cap = wmax * (0.7 if far else (1.0 if region else 0.85))
                w = float(np.clip(inr * P["object_scale"], P["min_width"] * sc, cap))
            ang, uw = _kind_bias(kind, part.name)
            reach = max(inr * 0.7, 6 * sc) if not region else 60 * sc
            flow = p.part_flow(part.sdf, reach, ang, uw, fine=inr < 60 * sc, rng=rp)
            detail = part.detail or kind == "rain"
            tol = 0.3 if detail else 0.9
            p.zone(part.sdf, bbox, flow, field, w, rp.child("zone"), inr, perspective=perspective,
                   accent_shade=accent_shade if not region else None, detail=detail,
                   under=not detail, tol=tol)
            # Contour immediately, so nearer things painted later cover it.
            if (part.name not in NO_CONTOUR and not part.detail and kind not in ("sun", "moon", "cloud")
                    and (not region or kind in ("table", "river"))):
                cw = P["contour_weight"] * float(np.clip(shape.radius * 0.022, 2.0, 5.0)) * sc
                if far or kind == "river":
                    cw *= 0.6
                _contour(p, part.sdf, cw, rp.child("contour"))
            if kind == "rain":
                _rain_sheet(p, part, base, rp.child("sheet"))

    # Relief: light the height map, show the canvas weave where paint is thin.
    img = p.body.image.copy()
    hmap = ndimage.gaussian_filter(p.body.height, 0.7 * max(sc, 0.5))
    ln = np.hypot(lx, ly) or 1.0
    shade = emboss(hmap, light=(lx / ln, min(ly / ln, -0.35)), strength=0.55 * P["relief"])
    shade = np.clip(shade, 0.55, 1.45)
    spec = np.clip(shade - 1.15, 0, None) * 0.5
    img = img * shade[..., None] + spec[..., None]
    weave = canvas_weave(f.shape, p.rng.child("weave"), period=4.5 * max(sc, 0.6))
    thin = np.clip(1.0 - hmap / 0.45, 0, 1)
    img = img * (1 + P["canvas_show"] * 0.35 * (weave - 0.5) * (0.2 + thin))[..., None]
    img = np.clip(img, 0, 1).astype(np.float32)

    total = sum(p.counts.values())
    ctx.info.update({
        "strokes": int(total),
        "stroke_layers": {k: int(v) for k, v in p.counts.items()},
        "vortices": len(lights),
        "interior": bool(interior),
        "bare_canvas_fraction": round(float((p.body.height < 0.05).mean()), 4),
    })
    return img, ctx


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)
