"""Leaded stained-glass window.

Construction model
------------------
The picture is *glazed*: every pixel of the window belongs to one piece of
glass, stored in an integer label map, and lead came is drawn wherever the
label changes, so the leading is continuous and follows every cut.

* **Cartoon.** The scene is laid out inside a window opening (stone surround
  plus a border of coloured glass), using the shared geometry scaled into the
  opening. Background first: an outdoor sky is cut into gently domed,
  staggered courses, broken into a sunburst of rings and rays around a sun or
  moon; an interior wall is glazed in diamond quarries. Regions are cut the
  way they lie: sea in wave-edged courses that widen toward the viewer, fields
  in courses crossed by furrow lines converging on a vanishing point, hills
  in bands that follow the ridge line, the table in converging planks.
  Objects are cut along their own form: pieces are grouped by the direction
  of the silhouette normal (so cuts run along ridges, midribs and centre
  lines), with a rim/core split and cross-cuts on long shapes.
* **Glazier's clean-up.** Pieces that are too small or too thin to cut are
  merged into a neighbour of the same object.
* **Glass.** Each piece is one saturated colour chosen from the scene hints
  pushed to jewel saturation, varied piece to piece; shadow-side pieces are
  darker glass. Inside a piece: streaks in one of four directions, seeds
  (bubbles), thickness mottling and a thickness ramp. Backlit: pieces glow
  lighter at the centre and light bleeds (halation) over the lead's edges.
* **Paint.** Sparse vitreous paint: trace lines for masts, spars, eyes,
  window mullions, feathers, planks; a stippled matt on the shadow side of
  objects.
* **Window.** Stone surround, a border of alternating coloured pieces with
  corner blocks, and iron saddle bars.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core import stainedglass_kit as K
from ..core import sdf as S
from ..core.color import hex_to_rgb, luminance, mix
from ..core.geom import Frame
from ..core.masks import sdf_to_mask
from ..core.noise import fbm, value_noise
from ..core.rng import Rng
from ..scene.shapes import build_shapes
from .base import ROLE_TARGETS, clamp_params, encode, make_context, merge_params, nearest, scene_hints

NAME = "stainedglass"

DEFAULTS = {
    "pane_size": 105.0,     # typical piece size, px at 1024
    "lead_width": 7.0,      # lead came width, px at 1024
    "lead_wobble": 0.3,     # irregularity of the came width
    "border": 0.065,        # window border (stone + glass band) as a fraction of the short side
    "saddle_bars": 2,       # horizontal iron bars across the window
    "saturation": 1.3,      # jewel saturation boost of palette colours
    "glow": 1.3,            # backlit glow: lighter piece centres
    "halation": 0.5,        # light bleeding over the lead edges
    "bloom": 0.3,           # soft light spill around the brightest glass
    "streaks": 0.6,         # streaky glass
    "seeds": 0.6,           # bubbles in the glass
    "mottle": 0.6,          # thickness mottling
    "piece_jitter": 1.0,    # colour variation from piece to piece
    "form_shading": 1.0,    # darker glass on the shadow side of objects
    "matting": 0.45,        # painted matt shading on the shadow side
    "detail_paint": 1.0,    # painted trace lines (mullions, feathers, planks)
    "sunburst": 1.0,        # rings and rays around sun / moon (0 = off)
}

KNOBS = {
    "pane_size": (60.0, 180.0, "typical glass piece size; smaller = more, busier leading"),
    "lead_width": (3.0, 12.0, "lead came width"),
    "lead_wobble": (0.0, 0.8, "hand-made irregularity of the lead"),
    "border": (0.0, 0.12, "window border width"),
    "saddle_bars": (0, 4, "number of iron saddle bars"),
    "saturation": (0.9, 1.8, "jewel saturation of the glass"),
    "glow": (0.0, 2.0, "backlit glow at piece centres"),
    "halation": (0.0, 1.0, "light bleeding over lead edges"),
    "bloom": (0.0, 1.0, "soft spill of light around the brightest glass"),
    "streaks": (0.0, 1.5, "streaks in the glass"),
    "seeds": (0.0, 1.5, "bubbles in the glass"),
    "mottle": (0.0, 1.5, "thickness mottling"),
    "piece_jitter": (0.0, 2.0, "colour variation between pieces"),
    "form_shading": (0.0, 2.0, "shadow-side pieces in darker glass"),
    "matting": (0.0, 1.0, "painted shadow matt on objects"),
    "detail_paint": (0.0, 1.5, "painted trace lines"),
    "sunburst": (0.0, 1.5, "sunburst leading around lights"),
}

RULES = {"min_mean_saturation": 0.4, "min_edge_density": 0.07}

ROLE_GLASS = {k: hex_to_rgb(v) for k, v in {
    "sky": "#3f86d6", "water": "#1f5fb8", "ground": "#7a8f1f", "foliage": "#1f8a3a", "wood": "#8a4a14",
    "stone": "#6a5a8a", "accent": "#c0182a", "primary": "#1f4fb8", "secondary": "#2a8a4a",
    "dark": "#1a2466", "light": "#f0ecd0",
}.items()}
ROLE_PULL = {"sky": 0.25, "water": 0.3, "ground": 0.5, "foliage": 0.4, "wood": 0.45, "stone": 0.35,
             "accent": 0.2, "primary": 0.12, "secondary": 0.12, "dark": 0.35, "light": 0.25}

# Thin detail parts drawn as lead (dark, opaque lines) with a minimum half-width (x lead width).
LEAD_LINES = {"mast": 0.36, "spars": 0.36, "birds": 0.5, "nodes": 0.3, "reeds": 0.22, "stem": 0.28}
PAINT_LINES = {"rain": 0.62, "legs": 0.9, "eye": 1.0}
FRUIT_GLASS = {"orange": "#f08a12", "pear": "#b9c22a"}
LEAD_RGB = np.array([0.045, 0.043, 0.045], np.float32)
PAINT_RGB = np.array([0.10, 0.07, 0.05], np.float32)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

class _Palette:
    def __init__(self, scene, P):
        self.hints = scene_hints(scene)
        self.mood = scene.palette.mood
        self.boost = P["saturation"]
        lum = [luminance(h) for h in self.hints]
        self.darkest = self.hints[int(np.argmin(lum))]
        self.lightest = self.hints[int(np.argmax(lum))]
        self.rainy = any(o.kind == "rain" for o in scene.objects)

    def j(self, c, pale=False):
        return K.jewel(c, pale=pale, boost=self.boost)

    def role(self, role, obj_color=None):
        if obj_color and obj_color.startswith("#"):
            return self.j(hex_to_rgb(obj_color))
        if obj_color and obj_color in ROLE_TARGETS:
            role = obj_color
        role = role if role in ROLE_GLASS else "primary"
        c = self.hints[nearest(self.hints, ROLE_TARGETS[role])]
        return self.j(mix(c, ROLE_GLASS[role], ROLE_PULL[role]), pale=role == "light")

    def candidates(self, role):
        out = [self.role(role)]
        out += [self.j(h) for h in self.hints]
        out += [self.j(ROLE_GLASS[k]) for k in ("primary", "accent", "secondary", "wood", "stone")]
        return out

    def contrast(self, role, behind, obj_color=None, min_d=0.3, base=None, water=False):
        """Role colour, or the nearest alternative that stands clear of everything
        behind it (``behind`` is a list of colours). A darker or lighter glass of
        the same hue is preferred over a change of hue."""
        base = self.role(role, obj_color) if base is None else base
        if not behind:
            return base

        def clear(c):
            return min(np.linalg.norm(c - b) for b in behind)
        if clear(base) >= min_d:
            return base
        hh, ss, vv = K.hsv(base)
        if water:
            # Water under a sky of the same blue: shift toward teal / lighten, as glaziers do.
            tones = [K.from_hsv(hh - dh, ss, min(0.95, vv * f)) for dh, f in ((0.07, 1.3), (0.1, 1.5), (0.0, 1.45), (-0.06, 1.2))]
        else:
            tones = [K.from_hsv(hh, min(0.95, ss * 1.05), vv * 0.7), K.from_hsv(hh, ss, min(0.97, vv * 1.25 + 0.1)),
                     K.from_hsv(hh, min(0.95, ss * 1.05), vv * 0.5)]
        ok = [c for c in tones if clear(c) >= min_d]
        if ok:
            return ok[0]
        cands = self.candidates(role)
        ok = [c for c in cands if clear(c) >= min_d]
        if ok:
            return min(ok, key=lambda c: np.linalg.norm(c - base))
        return max(cands + tones, key=clear)

    def sun(self):
        gold = hex_to_rgb("#ffc81e")
        c = self.hints[nearest(self.hints, "#f2c14e")]
        return K.lighten(self.j(mix(gold, c, 0.3)), 0.22)

    def light_colour(self, shape):
        return self.sun() if shape.kind == "sun" else K.lighten(self.role("light"), 0.05)

    def sky(self):
        """(horizon colour, zenith colour) of the sky glass."""
        h = self.hints
        if self.mood == "night":
            blues = sorted(h, key=lambda c: float(c[2] - c[0]) - 0.5 * luminance(c), reverse=True)
            low = self.j(mix(blues[0], hex_to_rgb("#2a4fa8"), 0.3))
            top = self.j(mix(self.darkest, hex_to_rgb("#1a2270"), 0.5))
            return K.lighten(low, 0.05), K.darken(top, 0.1)
        blues = [c for c in h if 0.47 < K.hsv(c)[0] < 0.7 and K.hsv(c)[1] > 0.2]
        if self.mood == "cool" or self.rainy or (self.mood == "neutral" and blues):
            c = h[nearest(h, "#6fa8dc")]
            base = self.j(mix(c, ROLE_GLASS["sky"], 0.3))
            hh, s, v = K.hsv(base)
            low = K.from_hsv(hh - 0.02, s * 0.62, min(0.95, v + 0.12))
            top = K.from_hsv(hh + 0.03, min(0.95, s * 1.1), v * 0.8)
            return low, top
        base = self.role("sky")
        if self.mood == "warm":
            top = self.j(mix(h[nearest(h, ROLE_TARGETS["accent"])], base, 0.25))
            return K.lighten(base, 0.08), top
        hh, s, v = K.hsv(base)
        return K.lighten(base, 0.1), K.from_hsv(hh - 0.03, s, v * 0.82)


# ---------------------------------------------------------------------------
# The glazier: label map + piece table
# ---------------------------------------------------------------------------

class _Glazier:
    def __init__(self, shape):
        self.labels = np.zeros(shape, np.int64)
        self.cols = [np.zeros(3, np.float32)]   # label 0 = stone / none
        self.orient = [0]
        self.group = [-1]
        self.groups = []                        # group -> dict(part, kind, shade, ...)
        self.creases = []                       # painted separations inside single pieces

    def new_group(self, **info):
        self.groups.append(info)
        return len(self.groups) - 1

    def add(self, key, mask, sl, group, colour_fn, orient_fn=None):
        """Register pieces for ``mask`` (crop at ``sl``) keyed by ``key``."""
        if not mask.any():
            return
        ids, uk = K.compact(key, mask)
        n = len(uk)
        base = len(self.cols)
        blk = self.labels[sl]
        blk[mask] = base + ids[mask]
        yy, xx = np.nonzero(mask)
        yy = yy + sl[0].start
        xx = xx + sl[1].start
        iv = ids[mask]
        cnt = np.maximum(np.bincount(iv, minlength=n), 1)
        cx = np.bincount(iv, weights=xx, minlength=n) / cnt
        cy = np.bincount(iv, weights=yy, minlength=n) / cnt
        cols = colour_fn(uk, cx, cy, cnt)
        orients = orient_fn(uk, cx, cy) if orient_fn else np.zeros(n, np.int64)
        for i in range(n):
            self.cols.append(np.clip(np.asarray(cols[i], np.float32), 0, 1))
            self.orient.append(int(orients[i]))
            self.group.append(group)

    def behind(self, mask, sl, share=0.12, kinds=None):
        """Mean colour of each object part covering more than ``share`` of the mask
        (optionally only parts of the given ``kinds``)."""
        lab = self.labels[sl][mask]
        if lab.size == 0:
            return []
        cols = np.asarray(self.cols)
        grp = np.asarray(self.group)[lab]
        out = []
        for g in np.unique(grp):
            sel = grp == g
            if kinds is not None and (g < 0 or self.groups[g]["kind"] not in kinds):
                continue
            if sel.mean() >= share:
                out.append(cols[lab[sel]].mean(0))
        return out


def _crop(part, shape, pad=4):
    x0, y0, x1, y1 = part.stats()[3]
    H, W = shape
    return (slice(max(0, y0 - pad), min(H, y1 + pad)), slice(max(0, x0 - pad), min(W, x1 + pad)))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_array(scene, seed=None, params=None):
    P = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, P, NAME)
    rng = ctx.rng
    H, W = ctx.frame.shape
    sc = ctx.frame.scale
    b = int(round(P["border"] * min(H, W)))
    h, w = H - 2 * b, W - 2 * b
    inner = Frame(w, h)
    shapes = build_shapes(scene, inner, Rng(ctx.seed, "geometry")) if b else ctx.shapes
    ps = P["pane_size"] * sc
    lw = P["lead_width"] * sc
    pal = _Palette(scene, P)
    gz = _Glazier((h, w))
    X, Y = inner.grid
    hor = scene.horizon * h
    light = ctx.light_dir
    jit = P["piece_jitter"]
    indoor = any(s.kind == "table" for s in shapes)
    lights = [s for s in shapes if s.kind in ("sun", "moon") and s.cy < hor + s.radius * 0.5]

    # ---- background ------------------------------------------------------
    brng = rng.child("background")
    full = (slice(0, h), slice(0, w))
    allmask = np.ones((h, w), bool)
    if indoor:
        _wall(gz, pal, X, Y, ps, brng, jit, full, allmask)
    else:
        _sky(gz, pal, X, Y, ps, hor, lights, brng, jit, full, allmask, P, sc)

    # ---- objects, far to near ---------------------------------------------
    line_parts, paint_parts, form_groups = [], [], []
    orng = rng.child("objects")
    for si, shape in enumerate(shapes):
        r = orng.child(str(si))
        if shape.kind == "stars":
            _stars(gz, pal, shape.parts[0], inner, lw, r, jit)
            continue
        for pi, part in enumerate(shape.parts):
            name = part.name
            if part.stats()[2] <= 0:
                continue
            if name in LEAD_LINES or (shape.kind == "reeds"):
                line_parts.append((part, LEAD_LINES.get(name, 0.22)))
                continue
            if name in PAINT_LINES:
                paint_parts.append((part, PAINT_LINES[name]))
                continue
            sl = _crop(part, (h, w))
            sdf_c = part.sdf[sl]
            dmax = float(-sdf_c.min())
            if not shape.region and 0 < dmax < 1.1 * lw:
                # Too thin to glaze: widen the piece so glass still shows between the leads.
                sdf_c = sdf_c - min(1.1 * lw - dmax, 0.8 * lw)
            m = sdf_c < 0
            if not m.any():
                continue
            pr = r.child(name)
            obj_color = shape.obj.color if (pi == 0 and shape.obj.color) else None
            behind = gz.behind(m, sl, kinds=("sky",) if shape.kind in ("sea", "river") else None)
            g = gz.new_group(kind=shape.kind, part=part, region=shape.region, shade=part.shade and not shape.region)
            Xc, Yc = X[sl], Y[sl]
            key, colour_fn, orient_fn = _cut(shape, part, pi, sdf_c, m, Xc, Yc, sl, ps, lw, light, pr, pal,
                                             behind, obj_color, hor, lights, h, w, jit, P, gz)
            gz.add(key, m, sl, g, colour_fn, orient_fn)
            if shape.kind in ("tree", "pine", "cypress", "boat", "house", "bird", "vase", "fruit", "teapot",
                              "cup", "mountain", "cliff", "windmill", "lighthouse", "cloud", "bamboo"):
                form_groups.append((g, shape, part))

    # ---- glazier's clean-up ------------------------------------------------
    pane_group = np.asarray(gz.group, np.int64)
    pane_group[0] = -1
    labels = K.merge_slivers(gz.labels, np.where(pane_group < 0, 10 ** 6, pane_group),
                             min_area=(0.42 * ps) ** 2, min_half=0.85 * lw)

    # ---- the whole window: stone, border band, picture ---------------------
    L = np.zeros((H, W), np.int64)
    L[b:b + h, b:b + w] = labels
    cols = list(gz.cols)
    orient = list(gz.orient)
    if b:
        _border(L, cols, orient, pal, b, H, W, lw)
    cols = np.asarray(cols, np.float32)
    orient = np.asarray(orient, np.int64)
    n_pieces = len(np.unique(L)) - 1

    # ---- lead came -----------------------------------------------------------
    lrng = rng.child("lead")
    E = K.edges_of(L)
    dist = ndimage.distance_transform_edt(~E).astype(np.float32)
    hw = 0.5 * lw * (1 + P["lead_wobble"] * (value_noise((H, W), 60 * sc, lrng.child("w")) - 0.5) * 2)
    lead = np.clip(hw - dist + 0.5, 0, 1)
    dglass = np.maximum(dist - hw, 0)

    # ---- glass -------------------------------------------------------------
    grng = rng.child("glass")
    G = cols[L]
    FX, FY = ctx.frame.grid
    cnt = np.maximum(np.bincount(L.ravel(), minlength=len(cols)), 1)
    pcx = (np.bincount(L.ravel(), weights=FX.ravel(), minlength=len(cols)) / cnt).astype(np.float32)
    pcy = (np.bincount(L.ravel(), weights=FY.ravel(), minlength=len(cols)) / cnt).astype(np.float32)
    th = grng.child("ramp").uniform(0, 2 * np.pi, len(cols)).astype(np.float32)
    ramp = np.clip(((FX - pcx[L]) * np.cos(th)[L] + (FY - pcy[L]) * np.sin(th)[L]) / ps, -1, 1)
    streaks = K.streak_fields((H, W), grng.child("streak"), sc)
    st = np.take_along_axis(streaks, orient[L][None], 0)[0]
    mott = fbm((H, W), 70 * sc, grng.child("mottle"), octaves=3)
    T = (1 + P["mottle"] * 0.3 * (mott - 0.5) + P["streaks"] * 0.26 * (st - 0.5) + 0.06 * ramp)
    G = G * T[..., None]
    gl = 1 - np.exp(-dglass / (0.22 * ps))
    G = G * (1 - P["glow"] * (0.28 * (1 - gl) - 0.06 * gl))[..., None]
    G = G + (1 - G) * (0.09 * P["glow"] * gl ** 2)[..., None]
    core, ring = K.seeds_field((H, W), grng.child("seeds"), sc, 0.0009 * P["seeds"])
    G = G * (1 - 0.22 * ring)[..., None] + (1 - G) * (0.45 * core)[..., None]
    back = 0.92 + 0.16 * value_noise((H, W), 380 * sc, grng.child("back"))
    G = G * back[..., None]

    # ---- vitreous paint: shadow matt and trace lines ------------------------
    prng = rng.child("paint")
    paint = np.zeros((H, W), np.float32)
    stip = 0.55 + 0.45 * fbm((H, W), 6 * sc, prng.child("stipple"), octaves=2)
    inner_sl = (slice(b, b + h), slice(b, b + w))
    if P["matting"] > 0:
        lab_group = pane_group[np.clip(labels, 0, len(pane_group) - 1)]
        for g, shape, part in form_groups:
            if not part.shade:
                continue
            sl = _crop(part, (h, w), pad=2)
            m = lab_group[sl] == g
            if not m.any():
                continue
            sf = _shade_crop(part, light, sl)
            blk = paint[inner_sl][sl]
            blk[:] = np.maximum(blk, m * np.clip(sf - 0.35, 0, 1) * 0.75 * P["matting"] * stip[inner_sl][sl])
    detail = np.zeros((H, W), np.float32)
    if P["detail_paint"] > 0:
        for g, shape, part in form_groups:
            _trace(detail[inner_sl], shape, part, labels, pane_group, g, lw, ps, prng.child(str(g)), light)
        for part, a in paint_parts:
            sl = _crop(part, (h, w), pad=3)
            dmax = float(-part.sdf[sl].min())
            grow = max(0.0, 0.9 * sc - dmax) if part.name != "eye" else max(0.0, 2.2 * sc - dmax)
            blk = detail[inner_sl][sl]
            blk[:] = np.maximum(blk, a * sdf_to_mask(part.sdf[sl] - grow))
        for sl, cr in gz.creases:
            blk = detail[inner_sl][sl]
            blk[:] = np.maximum(blk, cr * 0.85)
        detail *= P["detail_paint"]
    paint = np.clip(paint, 0, 1)
    G = G * (1 - 0.7 * paint)[..., None]
    G = G * (1 - np.clip(detail, 0, 1))[..., None] + PAINT_RGB * np.clip(detail, 0, 1)[..., None]

    # ---- lead-weight line work (masts, spars, gulls, reeds) ----------------
    lines = np.zeros((H, W), np.float32)
    for part, tw in line_parts:
        sl = _crop(part, (h, w), pad=int(lw) + 3)
        dmax = float(-part.sdf[sl].min())
        grow = max(0.0, tw * lw - dmax)
        blk = lines[inner_sl][sl]
        blk[:] = np.maximum(blk, sdf_to_mask(part.sdf[sl] - grow))
    cov = np.maximum(lead, lines)

    # ---- composite ------------------------------------------------------------
    glassmask = (L > 0).astype(np.float32)
    stone = _stone((H, W), rng.child("stone"), sc, L, b)
    base = G * glassmask[..., None] + stone * (1 - glassmask)[..., None]
    lead_rgb = LEAD_RGB + 0.05 * np.clip(1 - dist / np.maximum(hw, 1e-3), 0, 1)[..., None] * 0.5
    out = base * (1 - cov)[..., None] + lead_rgb * cov[..., None]
    if P["halation"] > 0:
        lit = G * (glassmask * (1 - cov))[..., None]
        s = max(0.9 * 0.5 * lw, 1.0)
        bleed = np.stack([ndimage.gaussian_filter(lit[..., c], s) for c in range(3)], -1)
        out = out + P["halation"] * 1.1 * bleed * (cov * glassmask)[..., None]
    if P["bloom"] > 0:
        lum = luminance(out)
        hot = np.clip(lum - 0.62, 0, None) * glassmask
        glow_img = np.stack([ndimage.gaussian_filter(out[..., c] * hot, 18 * sc) for c in range(3)], -1)
        out = 1 - (1 - out) * (1 - np.clip(P["bloom"] * 2.2 * glow_img, 0, 0.6))
    if P["saddle_bars"] > 0:
        _saddle_bars(out, int(P["saddle_bars"]), b, h, W, lw, sc)
    out = np.clip(out, 0, 1).astype(np.float32)

    ctx.info.update({
        "pieces": int(n_pieces),
        "lead_width_px": round(float(lw), 2),
        "border_px": b,
        "indoor": bool(indoor),
        "sky": [K.hsv(c) for c in pal.sky()] if not indoor else None,
    })
    return out, ctx


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)


# ---------------------------------------------------------------------------
# Background glazing
# ---------------------------------------------------------------------------

def _sky(gz, pal, X, Y, ps, hor, lights, rng, jit, sl, mask, P, sc):
    h, w = X.shape
    R = 2.2 * max(w, h)
    cx, cy = w * (0.5 + rng.uniform(-0.15, 0.15)), hor + R
    wob_u = K.wobble((h, w), rng.child("wu"), 70 * sc, 3.0 * sc)
    wob_v = K.wobble((h, w), rng.child("wv"), 90 * sc, 4.0 * sc)
    r = np.hypot(X - cx, Y - cy)
    v = R - r + wob_v
    u = np.arctan2(X - cx, cy - Y) * R + wob_u
    bounds = K.make_bounds(float(v.min()), float(v.max()), 0.7 * ps, 1.03, rng.child("b"), jitter=0.3)
    widths = rng.child("w").uniform(1.4, 2.5, len(bounds) + 1) * ps
    key = K.courses(u, v, bounds, widths, rng.child("c"), slant=0.9, wave=0.5, wave_len=5.0 * ps)
    burst = np.full((h, w), -1, np.int64)
    if P["sunburst"] > 0:
        for li, L in enumerate(lights):
            rr = L.radius
            n_r = 2 if P["sunburst"] < 1.2 else 3
            radii = [rr * (1.0 + 0.7 * P["sunburst"] * (i + 1) * (1.0 if L.kind == "sun" else 0.75)) + ps * 0.12 * (i + 1)
                     for i in range(n_r)]
            counts = []
            for i in range(n_r):
                mid = 0.5 * ((radii[i - 1] if i else rr) + radii[i])
                n = int(max(8, round(2 * np.pi * mid / (0.75 * ps))))
                counts.append(n + n % 2)
            k2, ring = K.rings_sectors(X, Y, L.cx, L.cy, radii, counts, rng.child("burst", str(li)),
                                       twist=0.35 if L.kind == "sun" else 0.0)
            sel = ring >= 0
            key = np.where(sel, (li + 1) * 10 ** 8 + k2, key)
            burst = np.where(sel, li, burst)
    low, top = pal.sky()
    crng = rng.child("colour")
    light_cols = [pal.light_colour(L) for L in lights]
    accent = pal.role("accent")

    def colour(uk, pcx, pcy, cnt):
        out = []
        for i, k in enumerate(uk):
            t = np.clip(pcy[i] / max(hor, 1), 0, 1.3)
            c = mix(top, low, min(1.0, t) ** 0.9)
            if k >= 10 ** 8:
                li = int(k // 10 ** 8) - 1
                ring = int((k % 10 ** 8) // K.KEY_BASE) - 1
                sec = int(k % K.KEY_BASE)
                sun = light_cols[li]
                is_sun = lights[li].kind == "sun"
                a = [0.5, 0.72, 0.86][ring] if is_sun else [0.72, 0.85, 0.93][ring]
                a = a + (0.12 if sec % 2 else -0.06) * (1 if is_sun else 0.5)
                ray = sun if (sec % 2 == 0 or not is_sun) else mix(sun, accent, 0.55)
                c = mix(ray, c, float(np.clip(a - (0.1 if is_sun else 0), 0, 1)))
            c = K.vary(c, crng.child(str(int(k))), dh=0.012 * jit, ds=0.05 * jit, dv=0.06 * jit)
            out.append(c)
        return out

    def orient(uk, pcx, pcy):
        return np.array([0 if k < 10 ** 8 else (2 + int(k) % 2) for k in uk])

    gz.add(key, mask, sl, gz.new_group(kind="sky", part=None, region=True, shade=False), colour, orient)


def _wall(gz, pal, X, Y, ps, rng, jit, sl, mask):
    h, w = X.shape
    qx, qy = 0.85 * ps, 1.3 * ps
    ox, oy = rng.uniform(0, qx), rng.uniform(0, qy)
    a = np.floor((X + ox) / qx + (Y + oy) / qy).astype(np.int64)
    bq = np.floor((X + ox) / qx - (Y + oy) / qy).astype(np.int64)
    key = (a + 2048) * K.KEY_BASE * 4 + (bq + 2048)
    wall = pal.role("light")
    hh, s, v = K.hsv(pal.j(pal.lightest))
    wall_a = K.from_hsv(hh, max(0.38, s * 0.8), 0.9)
    wall_b = K.from_hsv(hh + 0.03, max(0.3, s * 0.6), 0.95)
    crng = rng.child("colour")

    def colour(uk, pcx, pcy, cnt):
        out = []
        for i, k in enumerate(uk):
            ai, bi = int(k // (K.KEY_BASE * 4)), int(k % (K.KEY_BASE * 4))
            c = wall_a if (ai + bi) % 2 else wall_b
            t = np.clip(pcy[i] / h, 0, 1)
            c = K.darken(c, 0.12 * t)
            out.append(K.vary(c, crng.child(str(int(k))), dh=0.01 * jit, ds=0.04 * jit, dv=0.04 * jit))
        return out

    def orient(uk, pcx, pcy):
        return np.array([int(k) % 4 for k in uk])

    gz.add(key, mask, sl, gz.new_group(kind="wall", part=None, region=True, shade=False), colour, orient)
    gz.wall = (wall_a, wall_b)


def _stars(gz, pal, part, frame, lw, rng, jit):
    h, w = frame.shape
    core = part.sdf < -0.5
    lab, n = ndimage.label(core)
    if n == 0:
        return
    idx = np.arange(1, n + 1)
    cy_, cx_ = np.asarray(ndimage.center_of_mass(core, lab, idx)).T
    area = ndimage.sum(core, lab, idx)
    gold = pal.j(mix(pal.hints[nearest(pal.hints, "#f2d16b")], hex_to_rgb("#f2c21a"), 0.4))
    g = gz.new_group(kind="stars", part=None, region=False, shade=False)
    for i in range(n):
        r0 = np.sqrt(area[i] / np.pi)
        Ro = max(3.2 * r0, 2.6 * lw)
        Ri = Ro * 0.42
        rot = rng.uniform(-0.3, 0.3)
        pts = [(cx_[i] + np.cos(rot + k * np.pi / 4) * (Ro if k % 2 == 0 else Ri),
                cy_[i] + np.sin(rot + k * np.pi / 4) * (Ro if k % 2 == 0 else Ri)) for k in range(8)]
        wdw = K.window((h, w), cx_[i] - Ro, cy_[i] - Ro, cx_[i] + Ro, cy_[i] + Ro)
        if wdw is None:
            continue
        sl, Xl, Yl = wdw
        m = S.polygon(Xl, Yl, pts) < 0
        c = K.vary(gold, rng.child(str(i)), dh=0.01 * jit, dv=0.05 * jit)
        gz.add(np.zeros(m.shape, np.int64), m, sl, g, lambda uk, a, b_, n_, c=c: [c] * len(uk))


# ---------------------------------------------------------------------------
# Cutting objects and regions
# ---------------------------------------------------------------------------

def _cut(shape, part, pi, sdf, m, X, Y, sl, ps, lw, light, rng, pal, behind, obj_color, hor, lights, h, w, jit, P, gz):
    kind, name = shape.kind, part.name
    role = part.role
    if kind == "table" and name == "top" and not obj_color:
        role = "wood"
    base = None
    if kind == "fruit" and name == "fruit" and not obj_color:
        v = shape.obj.variant or "apple"
        if v in FRUIT_GLASS:
            base = pal.j(mix(hex_to_rgb(FRUIT_GLASS[v]), pal.role("accent"), 0.15))
    base = pal.contrast(role, behind, obj_color, min_d=0.3 if shape.region else 0.25, base=base,
                        water=kind in ("sea", "river"))
    if kind == "sun":
        base = pal.sun()
    if kind == "hills":
        k = int(name[-1]) if name[-1].isdigit() else 0
        n_layers = sum(1 for p in shape.parts)
        far = 1.0 - k / max(n_layers, 1)
        base = mix(base, pal.sky()[0], 0.28 * far)
    if kind == "mountain" and name == "rock":
        base = mix(base, pal.sky()[0], 0.3)
    crng = rng.child("colour")
    cxp, cyp, rad, _ = part.stats()
    lx, ly = light
    fs = P["form_shading"]

    def obj_colour(alt=None, core_lift=0.1):
        def colour(uk, pcx, pcy, cnt):
            out = []
            for i, k in enumerate(uk):
                c = base
                if alt is not None and alt(k, i):
                    c = K.lighten(c, 0.1) if luminance(c) < 0.5 else K.darken(c, 0.08)
                u = ((pcx[i] - cxp) * lx + (pcy[i] - cyp) * ly) / max(rad, 1)
                c = c * (1 + 0.22 * fs * float(np.clip(u, -1, 1)))
                if 100 <= k % 1000 < 102:
                    c = K.lighten(c, core_lift)
                out.append(K.vary(c, crng.child(str(int(k))), dh=0.01 * jit, ds=0.05 * jit, dv=0.06 * jit))
            return out
        return colour

    def rand_orient(uk, pcx, pcy):
        return rng.child("o").integers(0, 4, len(uk))

    wob = lambda cell, amp, lab: K.wobble(X.shape, rng.child(lab), cell, amp)  # noqa: E731
    sc = ps / max(P["pane_size"], 1e-6)

    # Regions ---------------------------------------------------------------
    if kind in ("sea", "river"):
        top = hor if kind == "river" else float(Y[m].min())
        t = np.clip((Y - top) / max(h - top, 1), 0, 1)
        lam = (0.35 + 0.9 * t) * ps * 2.2
        v = Y + (0.05 + 0.3 * t) * ps * np.sin(2 * np.pi * X / lam + rng.uniform(0, 6)) + wob(50 * sc, 2 * sc, "wv")
        bounds = K.make_bounds(top, float(h), 0.28 * ps, 1.28, rng.child("b"), min_last=0.3)
        tb = np.clip((np.concatenate([[top], bounds]) - top) / max(h - top, 1), 0, 1)
        widths = (1.0 + 1.7 * tb) * ps * rng.child("w").uniform(0.9, 1.3, len(tb))
        key = K.courses(X + wob(60 * sc, 3 * sc, "wu"), v, bounds, widths, rng.child("c"), slant=0.5)
        sun_cols = [(Lt, pal.light_colour(Lt)) for Lt in lights]
        c2 = K.from_hsv(K.hsv(base)[0] + 0.035, K.hsv(base)[1], K.hsv(base)[2] * 0.8)
        sky_low = pal.sky()[0]

        def colour(uk, pcx, pcy, cnt):
            out = []
            for i, k in enumerate(uk):
                course = int(k // K.KEY_BASE)
                tt = float(np.clip((pcy[i] - top) / max(h - top, 1), 0, 1))
                c = base if (course + int(k % K.KEY_BASE) % 3 == 0) % 2 == 0 else c2
                c = mix(c, sky_low, 0.3 * (1 - tt) ** 2)
                for Lt, lc in sun_cols:
                    dx = abs(pcx[i] - Lt.cx) / max(Lt.radius * (0.7 + 0.8 * tt), 1)
                    wgt = float(np.exp(-dx ** 2 * 2) * (1 - 0.4 * tt))
                    if wgt > 0.3 + 0.25 * crng.child("r", str(int(k))).random():
                        c = mix(c, lc, 0.72) if Lt.kind == "sun" else K.lighten(c, 0.3)
                out.append(K.vary(c, crng.child(str(int(k))), dh=0.01 * jit, ds=0.05 * jit, dv=0.07 * jit))
            return out
        return key, colour, lambda uk, a, b_: np.zeros(len(uk), np.int64)

    if kind == "field":
        top = float(Y[m].min())
        vx, vy = w * (0.5 + rng.uniform(-0.2, 0.2)), top - 0.6 * h
        u = (X - vx) / np.maximum(Y - vy, 1) * (h - vy)
        bounds = K.make_bounds(top, float(h), 0.3 * ps, 1.3, rng.child("b"), min_last=0.3)
        v = Y + wob(80 * sc, 3 * sc, "wv") + 0.1 * ps * np.sin(X / (0.9 * w) * 2 * np.pi + rng.uniform(0, 6))
        widths = rng.child("w").uniform(1.1, 1.6, len(bounds) + 1) * ps
        key = K.courses(u + wob(60 * sc, 4 * sc, "wu"), v, bounds, widths, rng.child("c"), slant=0.15)
        hh, s, vv = K.hsv(base)
        variants = [base, K.from_hsv(hh - 0.04, s, min(0.95, vv * 1.12)), K.from_hsv(hh + 0.03, s, vv * 0.84)]
        sky_low = pal.sky()[0]

        def colour(uk, pcx, pcy, cnt):
            out = []
            for i, k in enumerate(uk):
                course, col = int(k // K.KEY_BASE), int(k % K.KEY_BASE)
                c = variants[(col + 2 * course) % 3]
                tt = float(np.clip((pcy[i] - top) / max(h - top, 1), 0, 1))
                c = mix(c, sky_low, 0.18 * (1 - tt) ** 2)
                out.append(K.vary(c, crng.child(str(int(k))), dh=0.01 * jit, ds=0.05 * jit, dv=0.06 * jit))
            return out
        return key, colour, lambda uk, a, b_: np.full(len(uk), 1, np.int64)

    if kind == "table":
        top = float(Y[m].min())
        if name == "top":
            vx, vy = w * 0.5, top - 1.2 * h
            u = (X - vx) / np.maximum(Y - vy, 1) * (h - vy)
            v = Y
            bounds = K.make_bounds(top, float(Y[m].max()) + 1, 0.9 * ps, 1.2, rng.child("b"), min_last=0.4)
            key = K.courses(u, v, bounds, 0.75 * ps, rng.child("c"), slant=0.0)
        else:
            key = np.floor((X + rng.uniform(0, ps)) / (1.3 * ps)).astype(np.int64)
        hh, s, vv = K.hsv(base)
        b2 = K.from_hsv(hh + 0.02, s, vv * 0.85)

        def colour(uk, pcx, pcy, cnt):
            return [K.vary(base if (int(k % K.KEY_BASE) % 2) else b2, crng.child(str(int(k))),
                           dh=0.008 * jit, ds=0.04 * jit, dv=0.05 * jit) for k in uk]
        return key, colour, lambda uk, a, b_: np.full(len(uk), 1 if name == "front" else 1, np.int64)

    if kind == "hills":
        depth = -sdf
        v = depth + wob(70 * sc, 3 * sc, "wv")
        bounds = K.make_bounds(0, float(depth[m].max()), 0.42 * ps, 1.35, rng.child("b"), min_last=0.3)
        key = K.courses(X + wob(50 * sc, 3 * sc, "wu"), v, bounds, rng.child("w").uniform(1.2, 1.9, len(bounds) + 1) * ps,
                        rng.child("c"), slant=0.6)
        return key, obj_colour(alt=lambda k, i: (k // K.KEY_BASE) % 2 == 1), lambda uk, a, b_: np.zeros(len(uk), np.int64)

    # Buildings: cut the way a glazier would draw masonry and roofs -----------
    if kind == "house" and name == "walls":
        x0, x1 = float(X[m].min()), float(X[m].max())
        n = max(1, int(round((x1 - x0) / (0.9 * ps))))
        key = np.floor((X - x0) / max((x1 - x0) / n, 1)).astype(np.int64)
        return key, obj_colour(alt=lambda k, i: k % 2 == 1), rand_orient
    if kind == "house" and name == "roof":
        y0, y1 = float(Y[m].min()), float(Y[m].max())
        n = max(1, int(round((y1 - y0) / (0.45 * ps))))
        key = np.floor((Y - y0) / max((y1 - y0) / n, 1)).astype(np.int64)
        x_mid = float(X[m].mean())
        key = key * 2 + (X > x_mid)
        return key, obj_colour(alt=lambda k, i: k % 2 == 1), lambda uk, a, b_: np.zeros(len(uk), np.int64)
    if kind in ("windmill",) and name == "tower":
        y0, y1 = float(Y[m].min()), float(Y[m].max())
        n = max(1, int(round((y1 - y0) / (0.8 * ps))))
        key = np.floor((Y - y0) / max((y1 - y0) / n, 1)).astype(np.int64) * 2 + (X > float(X[m].mean()))
        return key, obj_colour(alt=lambda k, i: k % 2 == 1), lambda uk, a, b_: np.full(len(uk), 1, np.int64)
    if kind == "lighthouse" or name in ("door", "windows", "tea", "beak", "lantern", "gallery", "cap", "lid", "band",
                                         "snow", "sails", "beam", "cabin"):
        key = np.zeros(sdf.shape, np.int64)
        if kind == "lighthouse" and name == "tower":
            key = (X > float(X[m].mean())).astype(np.int64)
        if name in ("snow", "sails") and float((-sdf[m]).max()) > 2.5 * lw:
            key = K.cut_form(sdf, m, ps * 1.1, lw, light, rng.child("cut"), core=False, along=False)
        if name == "windows":
            base_w = pal.j(hex_to_rgb("#f2b51c"))
            return key, lambda uk, a, b_, n_: [base_w] * len(uk), rand_orient
        return key, obj_colour(), rand_orient

    if name in ("leaves",):
        # Each leaf of a spray gets its own piece: group pixels by the orientation
        # of the silhouette normal (sign ignored, so both halves of a leaf agree).
        gy, gx = np.gradient(sdf.astype(np.float32))
        ori = np.arctan2(gy, gx) % np.pi
        ob = np.floor((ori + rng.uniform(0, np.pi)) % np.pi / (np.pi / 7)).astype(np.int64)
        crease = K.edges_of(np.where(m, ob, -1)) & m & (sdf < -1.5)
        crease = ndimage.binary_dilation(crease) & m
        gz.creases.append((sl, crease))
        key = np.zeros(sdf.shape, np.int64)
        hh, ss, vv = K.hsv(base)
        base = K.from_hsv(hh - 0.04, ss, min(0.95, vv * 1.12))
        return key, obj_colour(alt=lambda k, i: k % 2 == 1), rand_orient

    # Generic form cut ---------------------------------------------------------
    facets = 1.0
    core = True
    if kind in ("mountain", "cliff"):
        facets, core = 1.3, False
    split = True
    if kind in ("sun", "moon"):
        facets, split = 0.9, False
    if kind in ("boat",) and name == "hull":
        facets, core = 0.8, False
    key = K.cut_form(sdf, m, ps, lw, light, rng.child("cut"), facets=facets, core=core, split_core=split)
    return key, obj_colour(), rand_orient


def _shade_crop(part, light, sl):
    cx, cy, r, _ = part.stats()
    ys = np.arange(sl[0].start, sl[0].stop, dtype=np.float32) + 0.5
    xs = np.arange(sl[1].start, sl[1].stop, dtype=np.float32) + 0.5
    Xc, Yc = np.meshgrid(xs, ys)
    lx, ly = light
    u = ((Xc - cx) * lx + (Yc - cy) * ly) / max(r, 1)
    ramp = np.clip(0.5 - 0.5 * u, 0, 1)
    sdf = part.sdf[sl]
    edge = np.exp(-np.clip(-sdf, 0, None) / max(r * 0.25, 1.0))
    core = edge * np.clip(-u, 0, 1)
    return np.clip(0.75 * ramp + 0.5 * core, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Painted trace lines
# ---------------------------------------------------------------------------

def _seg(canvas, ax, ay, bx, by, r, a=1.0):
    wdw = K.window(canvas.shape, min(ax, bx) - r - 2, min(ay, by) - r - 2, max(ax, bx) + r + 2, max(ay, by) + r + 2)
    if wdw is None:
        return
    sl, Xl, Yl = wdw
    m = sdf_to_mask(S.segment(Xl, Yl, ax, ay, bx, by, r)) * a
    canvas[sl] = np.maximum(canvas[sl], m)


def _arc(canvas, cx, cy, rad, a0, a1, r, a=1.0, n=10):
    ts = np.linspace(a0, a1, n)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        _seg(canvas, cx + np.cos(t0) * rad, cy + np.sin(t0) * rad, cx + np.cos(t1) * rad, cy + np.sin(t1) * rad, r, a)


def _trace(canvas, shape, part, labels, pane_group, g, lw, ps, rng, light):
    """Sparse painted details on glass (in the inner picture's coordinates)."""
    kind, name = shape.kind, part.name
    fine = max(0.7, 0.13 * lw)
    x0, y0, x1, y1 = part.stats()[3]
    inside = part.sdf < -fine * 2
    local = np.zeros_like(canvas)
    if kind == "house" and name == "windows":
        lab, n = ndimage.label(part.sdf < 0)
        for s in ndimage.find_objects(lab):
            ya, yb, xa, xb = s[0].start, s[0].stop, s[1].start, s[1].stop
            mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
            _seg(local, mx, ya, mx, yb, fine * 1.3)
            _seg(local, xa, my, xb, my, fine * 1.3)
    elif kind == "house" and name == "door":
        mx = 0.5 * (x0 + x1)
        _seg(local, mx, y0 + 2, mx, y1 - 2, fine)
        _seg(local, x0 + (x1 - x0) * 0.75, 0.5 * (y0 + y1), x0 + (x1 - x0) * 0.75 + 0.1, 0.5 * (y0 + y1), fine * 1.8)
    elif kind == "house" and name == "roof":
        for t in np.linspace(0.25, 0.85, 4):
            y = y0 + (y1 - y0) * t
            _seg(local, x0, y, x1, y, fine * 0.8, 0.7)
    elif kind == "boat" and name == "hull":
        for t in (0.35, 0.62):
            y = y0 + (y1 - y0) * t
            _seg(local, x0, y - (y1 - y0) * 0.1, x1, y, fine)
    elif kind == "boat" and name == "sail":
        for t in np.linspace(0.2, 0.85, 4):
            y = y0 + (y1 - y0) * t
            _seg(local, x0, y + (y1 - y0) * 0.03, x1, y, fine * 0.8, 0.6)
    elif kind == "bird" and name == "wing":
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        rw = 0.5 * (x1 - x0)
        for j, t in enumerate((0.25, 0.5, 0.75)):
            _arc(local, cx - rw * 0.3 + rw * t * 0.8, cy - rw * 0.5, rw * 0.6, np.pi * 0.35, np.pi * 0.75, fine)
    elif kind == "tree" and name == "crown":
        pts = np.argwhere(inside)
        if len(pts):
            n = int(np.clip(len(pts) / (0.35 * ps) ** 2, 6, 40))
            sel = pts[rng.integers(0, len(pts), n)]
            for (yy, xx) in sel:
                rr = rng.uniform(0.08, 0.14) * ps
                a0 = rng.uniform(0.1, 0.6) * np.pi
                _arc(local, xx, yy - rr, rr, a0, a0 + np.pi * 0.5, fine * 0.8, 0.8, n=6)
    elif kind in ("pine", "cypress"):
        pts = np.argwhere(inside)
        if len(pts):
            n = int(np.clip(len(pts) / (0.3 * ps) ** 2, 5, 30))
            sel = pts[rng.integers(0, len(pts), n)]
            for (yy, xx) in sel:
                ln = rng.uniform(0.08, 0.16) * ps
                d = rng.choice([-1, 1])
                _seg(local, xx, yy, xx + d * ln * 0.7, yy + ln * 0.4, fine * 0.8, 0.8)
    elif kind == "fruit" and name == "leaf":
        lab, n = ndimage.label(part.sdf < 0)
        for s in ndimage.find_objects(lab):
            _seg(local, s[1].start, s[0].stop - 1, s[1].stop - 1, s[0].start, fine * 0.8)
    elif kind == "bamboo" and name == "leaves":
        pass
    elif kind == "windmill" and name == "sails":
        Yl, Xl = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]].astype(np.float32)
        sp = max(0.12 * ps, 4.0)
        f1, f2 = ((Xl + Yl) / sp) % 1, ((Xl - Yl) / sp) % 1
        dd = np.minimum(np.minimum(f1, 1 - f1), np.minimum(f2, 1 - f2)) * sp / 1.414
        local = np.maximum(local, np.clip(fine - dd + 0.5, 0, 1) * (part.sdf < -1) * 0.8)
    if local.any():
        own = pane_group[np.clip(labels, 0, len(pane_group) - 1)] == g
        canvas[:] = np.maximum(canvas, local * own)


# ---------------------------------------------------------------------------
# Window structure
# ---------------------------------------------------------------------------

def _border(L, cols, orient, pal, b, H, W, lw):
    so = int(round(0.36 * b))
    bw = b - so
    a = pal.role("accent")
    c_b = pal.role("primary")
    if np.linalg.norm(a - c_b) < 0.3:
        c_b = pal.j(hex_to_rgb("#1f4fb8"))
    gold = pal.j(hex_to_rgb("#f0b21a"))
    rng = Rng(int(H * 7 + W), "border")

    def new(c, o=0):
        cols.append(np.clip(K.vary(c, rng.child(str(len(cols))), dh=0.01, dv=0.06), 0, 1))
        orient.append(o)
        return len(cols) - 1

    # Corner blocks.
    for (ys, xs) in (((so, b), (so, b)), ((so, b), (W - b, W - so)), ((H - b, H - so), (so, b)), ((H - b, H - so), (W - b, W - so))):
        L[ys[0]:ys[1], xs[0]:xs[1]] = new(gold, 2)
    # Runs of alternating pieces.
    for side in range(4):
        horiz = side in (0, 1)
        length = W - 2 * b if horiz else H - 2 * b
        n = max(1, int(round(length / (1.9 * bw))))
        n += (n + 1) % 2  # odd count so the run is symmetric
        edges = np.round(np.linspace(0, length, n + 1)).astype(int)
        for i in range(n):
            c = a if i % 2 == 0 else c_b
            lab = new(c, 0 if horiz else 1)
            if side == 0:
                L[so:b, b + edges[i]:b + edges[i + 1]] = lab
            elif side == 1:
                L[H - b:H - so, b + edges[i]:b + edges[i + 1]] = lab
            elif side == 2:
                L[b + edges[i]:b + edges[i + 1], so:b] = lab
            else:
                L[b + edges[i]:b + edges[i + 1], W - b:W - so] = lab


def _stone(shape, rng, sc, L, b):
    H, W = shape
    t = fbm(shape, 40 * sc, rng.child("t"), octaves=4)
    base = np.array([0.24, 0.215, 0.19], np.float32)
    img = base * (0.75 + 0.5 * t)[..., None]
    if b:
        # The reveal darkens toward the glass.
        d = ndimage.distance_transform_edt(L == 0).astype(np.float32)
        img *= (0.55 + 0.45 * np.clip(d / max(0.36 * b, 1), 0, 1))[..., None]
    return img.astype(np.float32)


def _saddle_bars(out, n, b, h, W, lw, sc):
    H = out.shape[0]
    ys = [b + h * (k + 1) / (n + 1) for k in range(n)]
    Y = np.arange(H, dtype=np.float32)[:, None] + 0.5
    half = 0.62 * lw
    for y in ys:
        d = np.abs(Y - y)
        m = np.clip(half - d + 0.5, 0, 1)
        hi = np.clip(1 - np.abs(Y - (y - half * 0.4)) / max(half * 0.35, 0.7), 0, 1) * m
        col = np.array([0.03, 0.03, 0.035], np.float32)
        x_in = np.zeros(W, np.float32)
        x_in[max(0, int(0.15 * b)):W - int(0.15 * b)] = 1
        mm = (m * x_in[None, :])[..., None]
        out[:] = out * (1 - mm) + (col + 0.08 * hi[..., None]) * mm
