"""Colour woodblock print in the moku-hanga manner.

Printing model
--------------
* **Colour blocks.** Each ink is a block, a coverage map in [0, 1]. Objects are
  cut far-to-near: a nearer shape clears the blocks beneath it and puts its own
  ink down, so every area is one flat colour (paper counts as a colour).
  Density is flat except for *bokashi*: graded wiping on the sky (dark toward
  the top, warm glow at the horizon), on the water (deepening toward the
  horizon), on mountain peaks and hill crests.
* **Key block.** A dark sumi block of carved outlines along every part contour.
  The line swells and tapers along the contour and is cut thicker on the side
  away from the light. Thin parts (masts, windows, stems, birds) are carved
  solid. A nearer shape clears the key lines of what lies behind it.
* **Pattern.** Water is rows of scalloped crests with a paper foam line carved
  out of the water block; rain is long straight parallel lines; clouds carry an
  inner contour line and a banded bokashi base; foliage carries clump arcs.
* **Printing.** Each colour block is shifted by its own small misregistration,
  softened at the edges, printed with its own wood grain and baren texture,
  and multiplied onto warm washi, lightest ink first, key block last.
"""
from __future__ import annotations

import colorsys

import numpy as np
from scipy import ndimage

from ..core import woodblock_kit as K
from ..core.color import hex_to_rgb, luminance, mix, rgb_to_hex
from ..core.masks import sdf_to_mask
from ..core.noise import fbm, value_noise
from ..core.texture import paper
from .base import ROLE_TARGETS, clamp_params, encode, make_context, merge_params, nearest, scene_hints

NAME = "woodblock"

DEFAULTS = {
    "key_width": 3.4,        # key-block line width in px at 1024
    "line_swell": 0.8,       # how much the carved line swells / tapers along the contour
    "max_color_inks": 6,     # colour blocks (the key block comes on top of these)
    "misregistration": 2.0,  # max colour-block offset in px at 1024
    "bokashi": 0.8,          # strength of graded wiping (sky, water, peaks)
    "wood_grain": 0.5,       # visibility of the wood grain in flat areas
    "baren": 0.5,            # rubbing-pad texture
    "wave_density": 1.0,     # scale of wave rows (higher = more, smaller rows)
    "foam": 0.8,             # paper foam line carved under each wave crest
    "pigment_mix": 0.4,      # pull of scene colours toward traditional pigments
    "rain_density": 1.0,     # density of the parallel rain lines
    "border": True,          # print a keyline border around the image
    "kasumi": True,          # mist bands across the foot of mountains
}

KNOBS = {
    "key_width": (1.2, 5.0, "key-block line width; thicker = bolder, more graphic"),
    "line_swell": (0.0, 1.0, "carved line taper/swell along the contour; 0 = uniform"),
    "max_color_inks": (3, 6, "number of colour blocks"),
    "misregistration": (0.0, 5.0, "colour block offset; imperfection vs. crispness"),
    "bokashi": (0.0, 1.0, "graded wiping strength on sky, water and peaks"),
    "wood_grain": (0.0, 1.0, "wood grain visible in the flat colour areas"),
    "baren": (0.0, 1.0, "baren rubbing texture"),
    "wave_density": (0.6, 1.6, "density of the stylised wave rows"),
    "foam": (0.0, 1.0, "paper foam lines under wave crests"),
    "pigment_mix": (0.0, 0.8, "how far scene colours are pulled toward traditional pigments"),
    "rain_density": (0.3, 2.0, "density of rain lines"),
}

RULES = {"max_inks": 8, "max_color_clusters": 26}

KEY_INK = "#221f22"

# Traditional-leaning pigments per role (Prussian blue, indigo, beni, ochre, ...).
TRAD = {
    "sky": "#285a94", "night": "#1c2a4f", "water": "#2d6490", "ground": "#aaa261", "foliage": "#587a40",
    "wood": "#7a4f33", "stone": "#8a8172", "accent": "#c8432e", "glow": "#e6a04e", "light": "#f0dca2",
    "dark": "#2c3348", "mountain": "#5d7690",
}
LANDSCAPE = {"water", "ground", "foliage", "wood", "stone", "mountain", "glow"}

NO_OUTLINE = {"sun", "moon", "stars", "rain"}


def part_role(shape, part, pi):
    """The palette role of a part, with woodblock overrides (wood tables, blue-grey mountains)."""
    if shape.obj.color and pi == 0:
        return shape.obj.color
    if shape.kind == "table" and part.name == "top":
        return "wood"
    if shape.kind == "mountain" and part.name == "rock":
        return "mountain"
    if shape.kind == "fruit" and part.name == "fruit":
        return {"orange": "#e0842e", "pear": "#c8b449"}.get(shape.obj.variant or "apple", part.role)
    return part.role


def _pigment(rgb, desat=0.9, vmax=0.93):
    h, s, v = colorsys.rgb_to_hsv(*np.clip(rgb, 0, 1))
    return np.array(colorsys.hsv_to_rgb(h, s * desat, min(v, vmax)), np.float32)


class Palette:
    """Role -> colour-block index (or None for paper), with a capped, merged ink set."""

    def __init__(self, scene, shapes, P):
        hints = scene_hints(scene)
        self.night = scene.palette.mood == "night"
        self.warm = scene.palette.mood in ("warm",) or any(s.kind == "sun" for s in shapes)
        pm = P["pigment_mix"]
        roles = set()
        for s in shapes:
            for pi, p in enumerate(s.parts):
                if not (p.detail and p.role != "light"):
                    roles.add(part_role(s, p, pi))
        if any(s.kind in ("sea", "river") for s in shapes):
            roles.add("water")
        roles.add("skyink")
        if not self.night and self.warm:
            roles.add("glow")
        if any(s.kind in ("moon", "stars") or s.part("beam") is not None for s in shapes):
            roles.add("lightink")
        roles.discard("light")
        self.role_rgb = {}
        for r in sorted(roles):
            self.role_rgb[r] = self._role_colour(r, hints, pm)
        # Merge near-identical colours, then fold the smallest-area roles into
        # their nearest block until the cap holds (big areas keep their colour).
        H, W = shapes[0].parts[0].sdf.shape if shapes else (1, 1)
        area = {r: 0.0 for r in self.role_rgb}
        for s in shapes:
            for pi, p in enumerate(s.parts):
                rr = part_role(s, p, pi)
                if rr in area:
                    area[rr] += float((p.sdf < 0).sum())
        area["skyink"] = area.get("skyink", 0) + 0.4 * H * W
        if "glow" in area:
            area["glow"] += 0.15 * H * W
        names = list(self.role_rgb)
        groups = [[n] for n in names]
        cols = [self.role_rgb[n] for n in names]
        areas = [area[n] + 1.0 for n in names]

        def dist(a, b):
            return float(np.sqrt(np.sum((a - b) ** 2)))

        def fold(i, j):
            """Merge group j into group i (area-weighted colour)."""
            wi, wj = areas[i], areas[j]
            cols[i] = (cols[i] * wi + cols[j] * wj) / (wi + wj) if wj > 0.25 * wi else cols[i]
            groups[i] += groups[j]
            areas[i] += areas[j]
            del groups[j], cols[j], areas[j]

        while len(cols) > 1:
            best = min(((dist(cols[i], cols[j]), i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))))
            if best[0] > 0.13:
                break
            _, i, j = best
            if areas[j] > areas[i]:
                i, j = j, i
            fold(i, j)
        while len(cols) > int(P["max_color_inks"]):
            j = int(np.argmin(areas))
            i = min((x for x in range(len(cols)) if x != j), key=lambda x: dist(cols[x], cols[j]))
            fold(i, j)
        # Print order: lightest first.
        order = sorted(range(len(cols)), key=lambda i: -float(luminance(cols[i])))
        self.inks = [cols[i].astype(np.float32) for i in order]
        self.role_ink = {}
        for new, old in enumerate(order):
            for n in groups[old]:
                self.role_ink[n] = new
        self.key = hex_to_rgb(KEY_INK)

    def _role_colour(self, r, hints, pm):
        if r.startswith("#"):
            return _pigment(hex_to_rgb(r))
        if r == "skyink":
            t = hex_to_rgb(TRAD["night" if self.night else "sky"])
        elif r == "lightink":
            t = hex_to_rgb(TRAD["light"])
            return _pigment(mix(hints[int(np.argmax([luminance(h) for h in hints]))], t, 0.6), 0.95, 0.97)
        elif r in TRAD:
            t = hex_to_rgb(TRAD[r])
        else:
            t = hex_to_rgb(ROLE_TARGETS.get(r, ROLE_TARGETS["primary"]))
            h = hints[nearest(hints, t)]
            return _pigment(mix(h, t, pm * 0.4))
        if r.startswith("#"):
            return _pigment(hex_to_rgb(r))
        h = hints[nearest(hints, t)]
        # The sky bokashi is the signature Prussian blue; pull it harder.
        w = pm
        if r == "skyink":
            w = min(1.0, pm * 1.6)
        elif r in LANDSCAPE and float(np.sqrt(np.sum((h - t) ** 2))) > 0.2:
            w = max(pm, 0.85)  # no hint near the traditional pigment: use the pigment
        c = mix(h, t, w)
        if r == "skyink" and not self.night and luminance(c) > 0.45:
            c = mix(c, t, 0.6)
        return _pigment(c, 1.0 if r == "skyink" else 0.9)

    def ink(self, role):
        if role.startswith("#"):
            rgb = hex_to_rgb(role)
            return int(np.argmin([np.sum((c - rgb) ** 2) for c in self.inks]))
        return self.role_ink.get(role, self.role_ink.get("skyink"))

    def ranked(self, role):
        """All inks ordered by closeness to the role's colour."""
        rgb = self.role_rgb.get(role)
        if rgb is None:
            rgb = hex_to_rgb(role) if role.startswith("#") else self.inks[self.ink(role)]
        return sorted(range(len(self.inks)), key=lambda i: float(np.sum((self.inks[i] - rgb) ** 2)))


def render_array(scene, seed=None, params=None):
    params = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, params, NAME)
    f, rng, P = ctx.frame, ctx.rng, params
    H, W = f.shape
    X, Y = f.grid
    sc = f.scale
    ld = ctx.light_dir

    pal = Palette(scene, ctx.shapes, P)
    n = len(pal.inks)
    cov = np.zeros((n, H, W), np.float32)
    key = np.zeros((H, W), np.float32)

    swell = np.clip(0.5 + (value_noise(f.shape, 34 * sc, rng.child("swell")) - 0.5) * 1.6 * P["line_swell"]
                    + (value_noise(f.shape, 110 * sc, rng.child("swell2")) - 0.5) * 0.8 * P["line_swell"], 0.05, 1.0)
    rough = (value_noise(f.shape, 2.5 * sc, rng.child("rough")) - 0.5) * 0.9
    edge_wob = (value_noise(f.shape, 7 * sc, rng.child("ewob")) - 0.5) * 0.9 * sc

    def lay(k, m, dens=1.0, win=None):
        """Clear all blocks under m, then put ink k (None = paper) down with density ``dens``."""
        sl = win.sl if win is not None else (slice(None), slice(None))
        mm = m if win is None or m.shape == (win.y1 - win.y0, win.x1 - win.x0) else m[sl]
        c = cov[(slice(None),) + sl]
        c *= 1.0 - mm[None]
        if k is not None:
            d = dens if np.isscalar(dens) else (dens if dens.shape == mm.shape else dens[sl])
            c[k] += mm * d

    def overprint(k, m, dens, win=None):
        """Add ink k over whatever is there (for bokashi wipes and shadows)."""
        sl = win.sl if win is not None else (slice(None), slice(None))
        c = cov[(k,) + sl]
        np.maximum(c, m * dens, out=c)

    def clear_key(m, win):
        key[win.sl] *= 1.0 - m

    def outline(sdf_w, win, width, offset=0.0, shadow=0.35):
        line = K.carved_outline(sdf_w, width, win.of(swell), ld, shadow, win.of(rough) * sc, offset)
        key[win.sl] = np.maximum(key[win.sl], line)
        return line

    def window(part_or_bbox, pad=None):
        bbox = part_or_bbox.stats()[3] if hasattr(part_or_bbox, "stats") else part_or_bbox
        return K.Window(bbox, f.shape, pad if pad is not None else 6 * sc + P["key_width"] * 2 * sc)

    horizon_px = scene.horizon * H
    has_ground = any(s.kind in ("sea", "field", "river", "hills") for s in ctx.shapes)
    indoor = any(s.kind == "table" for s in ctx.shapes) and not has_ground
    raining = any(s.kind == "rain" for s in ctx.shapes)
    kw = P["key_width"] * sc
    bok = P["bokashi"]

    # Sky ----------------------------------------------------------------
    sky_k = pal.ink("skyink")
    sky_bottom = horizon_px
    if not any(s_.region or s_.kind == "hills" for s_ in ctx.shapes):
        sky_bottom = float(H)  # nothing is printed below the horizon: the sky block runs to the edge
    brng = rng.child("bokashi")
    if pal.night:
        t = Y / max(sky_bottom, 1)
        dens = 0.8 + 0.2 * bok * K.wipe_ramp(t / 0.7, brng.child("sky"), f.shape, sc)
        lay(sky_k, np.ones(f.shape, np.float32), dens)
    else:
        band = 0.42 if not raining else 0.75
        top = bok * K.wipe_ramp(Y / (sky_bottom * band), brng.child("sky"), f.shape, sc, gamma=1.2)
        # Indoors the "sky" is a wall: its top bokashi is printed from the (wood) table block.
        wall_k = pal.ink("wood") if (indoor and "wood" in pal.role_ink) else sky_k
        lay(wall_k, np.ones(f.shape, np.float32), np.clip((0.95 if not indoor else 0.5) * top, 0, 1))
        if "glow" in pal.role_ink and not indoor:
            g = pal.ink("glow")
            if g != sky_k:
                t = (sky_bottom - Y) / (sky_bottom * 0.45)
                glow = K.wipe_ramp(t, brng.child("glow"), f.shape, sc, gamma=1.1) * (Y < sky_bottom + 4) * bok
                glow = glow * (1 - np.clip(cov[sky_k] * 1.5, 0, 1))
                overprint(g, np.ones(f.shape, np.float32), glow * 0.9)
        elif not indoor:
            # Cool / neutral days: a pale wash of the sky block low on the horizon.
            t = (sky_bottom - Y) / (sky_bottom * 0.3)
            low = K.wipe_ramp(t, brng.child("low"), f.shape, sc, gamma=1.3) * (Y < sky_bottom + 4) * 0.35 * bok
            overprint(sky_k, np.ones(f.shape, np.float32), low)
    if raining or pal.night:
        # Sumi bokashi at the very top of the sky (ichimonji band).
        t = Y / (sky_bottom * (0.35 if not raining else 0.6))
        top_key = K.wipe_ramp(t, brng.child("ichimonji"), f.shape, sc, gamma=1.5) * (0.55 if raining else 0.45) * bok
        key[:] = np.maximum(key, top_key)
    ctx.info["sky_ink"] = sky_k

    # Objects ------------------------------------------------------------
    orng = rng.child("objects")
    lights = [s for s in ctx.shapes if s.kind in ("sun", "moon")]
    rain_shapes = []
    mist_done = not (P["kasumi"] and any(s_.kind == "mountain" for s_ in ctx.shapes))
    for si, shape in enumerate(ctx.shapes):
        r = orng.child(str(si))
        kind = shape.kind
        if not mist_done and shape.depth < 0.85:
            mist_done = True
            _kasumi(ctx, lay, key, pal, brng.child("kasumi"))
        if kind == "rain":
            rain_shapes.append(shape)
            continue
        sm = shape.mask
        area = max(float(sm.sum()), 1.0)
        under = (cov * sm[None]).reshape(n, -1).sum(1) / area
        for pi, part in enumerate(shape.parts):
            win = window(part)
            if win.empty:
                continue
            sd = win.of(part.sdf)
            if not (sd < 0.5).any():
                continue
            thick = float(-sd.min())
            m = sdf_to_mask(sd + win.of(edge_wob))
            role = part_role(shape, part, pi)
            lw = min(kw * (0.8 if thick < 12 * sc else 1.0), max(thick * 0.45, 0.8 * sc))

            # Detail parts are carved solid in the key block (or cut to paper if light).
            if part.detail:
                if role == "light":
                    k = pal.ink("lightink") if "lightink" in pal.role_ink else None
                    lay(k, m, 1.0, win)
                    clear_key(m, win)
                else:
                    key[win.sl] = np.maximum(key[win.sl], sdf_to_mask(sd + win.of(rough) * sc * 0.6))
                continue

            # Resolve the colour block.
            if role == "light":
                if kind in ("moon",) or part.name == "beam":
                    k = pal.ink("lightink") if "lightink" in pal.role_ink else None
                elif kind == "lighthouse" and part.name == "tower" and float(under.sum()) < 0.3 and "lightink" in pal.role_ink:
                    k = pal.ink("lightink")
                else:
                    k = None
            elif role == "dark" and "dark" not in pal.role_ink:
                k = None
            else:
                ranked = pal.ranked(role)
                k = ranked[0]
                if not shape.region and kind not in ("sun",):
                    dom = [j for j in range(n) if under[j] > 0.35]
                    if k in dom:
                        alt = [j for j in ranked[1:3] if j not in dom]
                        if alt and np.sum((pal.inks[alt[0]] - pal.inks[k]) ** 2) < 0.35:
                            k = alt[0]
            dens = 1.0
            Yw, Xw = Y[win.sl], X[win.sl]

            if kind == "hills":
                layer = int(part.name[-1]) if part.name[-1].isdigit() else 0
                nl = sum(1 for p in shape.parts)
                base = 0.5 + 0.4 * (layer + 1) / nl
                crest = K.wipe_ramp(-sd / (f.height * 0.06), brng.child("hill", str(si), str(layer)), sd.shape, sc, wander=0.1)
                dens = np.clip(base + (1 - base) * crest * bok, 0, 1)
            elif kind in ("sea", "river"):
                x0, y0, x1, y1 = part.stats()[3]
                t = (Yw - y0) / max(H - y0, 1)
                deep = K.wipe_ramp(t / 0.45, brng.child("water", str(si)), sd.shape, sc, wander=0.04, gamma=1.6)
                dens = np.clip(0.6 + 0.4 * deep * bok + 0.4 * (1 - bok), 0, 1)
            elif kind == "field":
                crest = K.wipe_ramp(-sd / (f.height * 0.07), brng.child("field", str(si)), sd.shape, sc, wander=0.08)
                dens = np.clip(0.82 + 0.18 * crest * bok, 0, 1)
            elif kind == "table" and part.name == "front":
                k = pal.ranked(part_role(shape, shape.parts[0], 0))[0]
            elif part.name == "beam":
                # The beam is light wiped out along its length.
                x0 = part.stats()[3][0]
                t = (Xw - x0) / max(part.stats()[2] * 2, 1)
                m = m * K.wipe_ramp(t, brng.child("beam", str(si)), sd.shape, sc, wander=0.02, gamma=0.8)
                dens = 0.7

            lay(k, m, dens, win)
            clear_key(sdf_to_mask(sd + 0.5), win)

            # Per-kind extras ---------------------------------------------
            if kind == "mountain" and part.name == "rock":
                y0 = part.stats()[3][1]
                hgt = shape.radius * 2
                t = (Yw - y0) / (hgt * 0.55)
                peak = K.wipe_ramp(t, brng.child("peak", str(si)), sd.shape, sc, wander=0.05) * bok
                deep = pal.ink("skyink")
                if deep != k:
                    overprint(deep, m, peak * 0.75, win)
                # A few carved ridge strokes following the slopes, kept to the upper rock.
                _strata(ctx, key, part, r, lw, n=int(10 + shape.radius / (14 * sc)),
                        length=shape.radius * 0.35, ymax=shape.cy + shape.radius * 0.4)
            if kind in ("field", "cliff") and "foliage" in pal.role_ink and pal.ink("foliage") != k:
                # Green bokashi along the top edge of the ground / the cliff's grassy crown.
                band = f.height * (0.05 if kind == "field" else 0.035)
                moss = K.wipe_ramp(-sd / band, brng.child("moss", str(si)), sd.shape, sc, wander=0.12, gamma=1.3)
                if kind == "cliff":
                    moss = moss * (Yw < shape.cy + shape.radius * 0.6)
                overprint(pal.ink("foliage"), m, moss * 0.8 * bok, win)
            if kind == "table" and part.name == "front":
                overprint(k, m, 1.0, win)
                key[win.sl] = np.maximum(key[win.sl], m * 0.32)
            if kind == "cliff":
                _strata(ctx, key, part, r, lw, n=int(14 + shape.radius / (10 * sc)), length=shape.radius * 0.3)
            if kind in ("sea", "river"):
                _water(ctx, part, win, cov, key, pal, k, lights, r, sd, m)
            if kind == "cloud":
                # Banded base: a pale bokashi band rising from the flat bottom.
                y1 = part.stats()[3][3]
                hgt = part.stats()[2] * 2
                t = (y1 - Yw) / (hgt * 0.45)
                band = K.wipe_ramp(t, brng.child("cloud", str(si)), sd.shape, sc, wander=0.03) * 0.55 * bok
                bk = pal.ink("glow") if "glow" in pal.role_ink else pal.ink("skyink")
                overprint(bk, m, band, win)
                inner = K.carved_outline(sd, lw * 0.45, win.of(swell), None, 0, win.of(rough) * sc, offset=thick * 0.3)
                key[win.sl] = np.maximum(key[win.sl], inner * (Yw < y1 - hgt * 0.2))
            if kind == "tree" and part.name == "crown":
                _foliage_arcs(ctx, key, part, r, lw)
            if kind == "pine" and part.name == "needles":
                _pine_lines(ctx, key, part, r, lw, win, sd)
            if kind == "cypress":
                _cypress_lines(ctx, key, shape, part, r, lw, win, sd)

            # Key-block outline.
            if kind in NO_OUTLINE or part.name == "beam":
                continue
            if kind == "sea":
                continue
            wmul = 0.7 if kind in ("cloud", "hills") else 1.0
            outline(sd, win, lw * wmul, shadow=0.35)

    # Rain: long straight parallel lines over everything -----------------
    if rain_shapes:
        rr = rng.child("rain")
        dens = P["rain_density"] * max(1.0, rain_shapes[0].obj.count / 90)
        a = parallel_lines_layer(X, Y, rr.child("a"), sc, 12.0, 8.0 / dens, (0.6, 1.2), (0.25, 0.7), 0.7, H)
        b = parallel_lines_layer(X, Y, rr.child("b"), sc, 17.0, 11.0 / dens, (0.5, 1.0), (0.2, 0.5), 0.6, H)
        key[:] = np.maximum(key, np.maximum(a * 0.9, b * 0.5))

    # Border keyline -------------------------------------------------------
    if P["border"]:
        inset = 0.012 * min(H, W)
        d_in = np.minimum(np.minimum(X - inset, W - inset - X), np.minimum(Y - inset, H - inset - Y))
        outside = np.clip(0.5 - d_in, 0, 1)
        cov *= 1 - outside[None]
        key *= 1 - outside
        bw = kw * 1.2
        key[:] = np.maximum(key, np.clip(0.5 - (np.abs(d_in - bw * 0.5) - bw * 0.5) + (rough * 0.6 * sc), 0, 1) * (d_in > -1))

    # Print ----------------------------------------------------------------
    out, info = _print(ctx, pal, cov, key)
    ctx.info.update(info)
    return np.clip(out, 0, 1).astype(np.float32), ctx


def _kasumi(ctx, lay, key, pal, rng):
    """Kasumi: flat horizontal mist bands with rounded ends, left as paper, across the mountains' foot."""
    f = ctx.frame
    H, W = f.shape
    X, Y = f.grid
    sc = f.scale
    mts = [s for s in ctx.shapes if s.kind == "mountain"]
    base = max(s.cy + s.radius for s in mts)
    top = min(s.parts[0].stats()[3][1] for s in mts)
    hgt = base - top
    bands = [(rng.uniform(-0.15, 0.0) * W, rng.uniform(0.4, 0.52) * W, base - hgt * rng.uniform(0.1, 0.14), hgt * 0.06),
             (rng.uniform(0.62, 0.72) * W, W * 1.15, base - hgt * rng.uniform(0.2, 0.26), hgt * 0.045)]
    wav = (value_noise(f.shape, 90 * sc, rng.child("w")) - 0.5) * hgt * 0.035
    d = np.full(f.shape, np.inf, np.float32)
    for x0, x1, yc, hh in bands:
        # Stadium: a segment with round caps.
        dx = np.maximum(np.maximum(x0 - X, X - x1), 0)
        d = np.minimum(d, np.hypot(dx, Y - yc) - hh)
    d = d + wav
    # Crisp band, printed with a pale graded tint (bokashi inside the band, strongest at its top).
    m = sdf_to_mask(d)
    g = pal.ink("glow") if "glow" in pal.role_ink else pal.ink("skyink")
    tint = np.zeros(f.shape, np.float32)
    for x0, x1, yc, hh in bands:
        tint = np.maximum(tint, np.clip(1 - (Y - (yc - hh)) / (2 * hh), 0, 1) * (np.abs(Y - yc) < hh * 1.5))
    lay(g, m, 0.35 * tint * ctx.params["bokashi"])
    key *= 1.0 - sdf_to_mask(d + 0.5)


def parallel_lines_layer(X, Y, rng, sc, ang, spacing, width, length, present, H):
    return K.parallel_lines(X, Y, rng, sc, ang, spacing, width, length, present, H)


def _water(ctx, part, win, cov, key, pal, k, lights, r, sd, m):
    """Scalloped wave rows, carved foam, and broken reflections of sun / moon."""
    f, P = ctx.frame, ctx.params
    sc = f.scale
    H = f.height
    X, Y = f.grid
    Xw, Yw = X[win.sl], Y[win.sl]
    top = float(part.stats()[3][1])
    dens = P["wave_density"]
    wob = (value_noise(sd.shape, 80 * sc, r.child("wob")) - 0.5) * 10 * sc
    line, foam, row, frac, sp, t = K.wave_rows(Xw, Yw, top, H, sc, s_near=26.0 / dens, s_far=3.5 / dens,
                                               line_w=P["key_width"] * 0.55, wob=wob)
    inside = sdf_to_mask(sd + 2.0 * sc)
    key[win.sl] = np.maximum(key[win.sl], line * inside)
    if k is not None and P["foam"] > 0:
        c = cov[(k,) + win.sl]
        c *= 1.0 - foam * P["foam"] * inside
    for L in lights:
        if L.cy > top:
            continue
        lk = pal.ink("glow") if (L.kind == "sun" and "glow" in pal.role_ink) else None
        if L.kind == "moon":
            lk = pal.ink("lightink") if "lightink" in pal.role_ink else None
        rl = K.hash01(row, 3.1)
        ro = (K.hash01(row, 5.3) - 0.5) * L.radius * 0.5
        half = L.radius * (0.2 + 0.9 * rl) * (0.55 + 0.7 * t)
        bar = (np.abs(Xw - L.cx - ro) < half) & (frac > 0.25) & (frac < 0.62) & (rl > 0.15)
        refl = bar.astype(np.float32) * inside * np.clip(1.2 - t * 1.1, 0, 1).astype(np.float32)
        refl = (refl > 0.5).astype(np.float32) * inside
        c = cov[(slice(None),) + win.sl]
        c *= 1.0 - refl[None]
        if lk is not None:
            cov[(lk,) + win.sl] += refl


def _strata(ctx, key, part, r, lw, n, length, ymax=None):
    """Carved rock strokes parallel to the nearest (non-canvas) edge."""
    f = ctx.frame
    H, W = f.shape
    sc = f.scale

    def ok(x, y, depth):
        if ymax is not None and y > ymax:
            return False
        return min(x, W - x, H - y, y) > depth + 6 * sc

    strokes = K.contour_strokes(part.sdf, r.child("strata"), n, length, lw * 0.75, 6 * sc, length * 1.2, ok)
    canvas = np.zeros(f.shape, np.float32)
    K.stamp_strokes(canvas, f, strokes)
    key[:] = np.maximum(key, canvas * sdf_to_mask(part.sdf + 2.0 * sc))


def _foliage_arcs(ctx, key, part, r, lw):
    """Clumps of foliage as rows of small carved arcs (the crown's leaf masses)."""
    f = ctx.frame
    cx, cy, rad, (x0, y0, x1, y1) = part.stats()
    n = int(np.clip(rad * rad / (18 * f.scale) ** 2 * 0.9, 12, 90))
    xs = r.uniform(x0, x1, n * 3)
    ys = r.uniform(y0, y1, n * 3)
    iy = np.clip(ys.astype(int), 0, f.height - 1)
    ix = np.clip(xs.astype(int), 0, f.width - 1)
    rr = rad * r.uniform(0.09, 0.15, n * 3)
    ok = part.sdf[iy, ix] < -rr * 0.9
    arcs = []
    for x, y, a in list(zip(xs[ok], ys[ok], rr[ok]))[:n]:
        a0 = -np.pi + r.uniform(0.25, 0.6)
        a1 = -r.uniform(0.25, 0.6)
        arcs.append((x, y, a, a0, a1))
    canvas = np.zeros(f.shape, np.float32)
    K.stamp_arcs(canvas, f, arcs, lw * 0.6, f.scale)
    key[:] = np.maximum(key, canvas * 0.9)


def _pine_lines(ctx, key, part, r, lw, win, sd):
    """Short carved needle strokes hanging under each tier edge."""
    f = ctx.frame
    sc = f.scale
    X, Y = f.grid
    Xw, Yw = X[win.sl], Y[win.sl]
    inner = K.carved_outline(sd, lw * 0.5, np.full(sd.shape, 0.6, np.float32), None, 0, None, offset=max(3.0 * sc, lw * 2))
    comb = (np.sin(Xw / (5.0 * sc) + np.sin(Yw / (9 * sc)) * 1.2) > 0.2)
    gate = value_noise(sd.shape, 20 * sc, r.child("pg")) > 0.45
    key[win.sl] = np.maximum(key[win.sl], inner * comb * gate)


def _cypress_lines(ctx, key, shape, part, r, lw, win, sd):
    """Flame-like carved strokes running up the cypress."""
    f = ctx.frame
    sc = f.scale
    X, Y = f.grid
    Xw = X[win.sl]
    Yw = Y[win.sl]
    cx = shape.cx
    ph = np.sin(Yw / (22 * sc)) * 5 * sc
    u = (Xw - cx + ph) / (9 * sc)
    stroke = np.clip(1.0 - np.abs(u - np.round(u)) * 9 / max(lw / sc, 1), 0, 1)
    gate = value_noise(sd.shape, 26 * sc, r.child("cg")) > 0.55
    inside = sd < -3 * sc
    key[win.sl] = np.maximum(key[win.sl], stroke * gate * inside * 0.85)


def _print(ctx, pal, cov, key):
    """Misregister, soften, texture and multiply the blocks onto washi; key block last."""
    f, P, rng = ctx.frame, ctx.params, ctx.rng
    H, W = f.shape
    sc = f.scale
    paper_hex = ctx.scene.palette.paper or "#f1e6cc"
    img, fib = paper(f.shape, rng.child("paper"), tint=paper_hex, strength=1.3, scale=sc)
    out = img.copy()
    prng = rng.child("print")
    warp = K.grain_warp(f.shape, prng.child("warp"), sc)
    baren = K.baren_texture(f.shape, prng.child("baren"), sc)
    offsets, used = [], []
    for k in range(len(pal.inks)):
        pr = prng.child(str(k))
        plate = np.clip(cov[k], 0, 1)
        if float(plate.max()) < 0.02 or float((plate > 0.05).mean()) < 0.0005:
            offsets.append((0.0, 0.0))
            continue
        used.append(k)
        dx, dy = pr.uniform(-1, 1, 2) * P["misregistration"] * sc
        offsets.append((round(float(dx), 2), round(float(dy), 2)))
        plate = ndimage.shift(plate, (dy, dx), order=1, mode="nearest")
        plate = K.soft_edges(plate, pr.child("edge"), sc)
        ang = pr.uniform(-6, 6)
        grain = K.wood_grain(f.shape, pr.child("grain"), sc, warp, ang, pr.uniform(7, 12))
        br = np.roll(baren, (int(pr.integers(0, H)), int(pr.integers(0, W))), axis=(0, 1))
        dens = plate * (1.0 - 0.22 * P["wood_grain"] * grain) * (1.0 + 0.22 * P["baren"] * (br - 0.5) * 2)
        dens *= 0.92 + 0.16 * (fib - 0.5)
        out = out * (1.0 - np.clip(dens, 0, 1)[..., None] * (1.0 - pal.inks[k]))
    # Key block: printed dense, crisp, with a few ink-starved specks.
    kr = prng.child("key")
    kplate = ndimage.gaussian_filter(np.clip(key, 0, 1), 0.35 * sc)
    starve = fbm(f.shape, 6 * sc, kr.child("starve"), octaves=2)
    holes = np.clip((starve - 0.74) * 6, 0, 1)
    kd = kplate * 0.96 * (1.0 - 0.3 * holes) * (0.95 + 0.1 * (fib - 0.5))
    out = out * (1.0 - np.clip(kd, 0, 1)[..., None] * (1.0 - pal.key))
    info = {
        "inks": [rgb_to_hex(c) for c in pal.inks],
        "roles": {r_: int(i) for r_, i in sorted(pal.role_ink.items())},
        "key": KEY_INK,
        "used_blocks": used,
        "ink_count": len(used) + 1,
        "paper": paper_hex,
        "offsets": offsets,
    }
    return out, info


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)
