"""Mid-century screenprint.

Printing model: each ink is a *plate*, a coverage map in [0, 1]. Objects are
drawn far-to-near; an object either knocks out the plates beneath it (the
paper shows through, then its own ink goes down) or overprints (its ink simply
multiplies with what is already there, so two inks make a third colour).
Tone never comes from gradients - only from halftone dots and line screens
on a single ink. Finally each plate gets its own misregistration offset, ink
starvation, and squeegee density drift, and the plates multiply onto paper
in printing order.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core import halftone
from ..core.color import hex_to_rgb, luminance
from ..core.masks import sdf_to_mask
from ..core.noise import blurred_white, fbm, value_noise
from ..core.texture import paper
from ..scene.shapes import SKY_KINDS, shade_field
from .base import ROLE_TARGETS, Context, clamp_params, encode, make_context, merge_params, nearest, scene_hints

NAME = "screenprint"

DEFAULTS = {
    "inks": 4,               # number of ink plates (excluding paper)
    "halftone_cell": 7.0,    # dot pitch in px at 1024 px height
    "shade_strength": 0.55,  # how much halftone shadow on the side away from the light
    "misregistration": 2.5,  # max plate offset in px at 1024
    "overprint": 0.35,       # probability an object overprints instead of knocking out
    "ink_texture": 0.5,      # ink starvation / speckle amount
    "edge_roughness": 0.9,   # px of wobble on shape edges
    "sky_gradient": 0.35,    # halftone ramp of a second ink toward the horizon
    "halo_rings": True,      # concentric rings around sun / moon
    "water_lines": 0.5,      # density of wave line-work on water
    "far_tint": 0.55,        # halftone tint used for far objects (atmospheric depth)
    "key_details": True,     # print thin details in the darkest (key) ink
}

KNOBS = {
    "inks": (2, 5, "number of ink plates"),
    "halftone_cell": (4.0, 14.0, "halftone dot size; bigger reads more graphic, smaller more tonal"),
    "shade_strength": (0.0, 0.9, "strength of halftone shading on the shadow side"),
    "misregistration": (0.0, 6.0, "plate offset; imperfection vs. crispness"),
    "overprint": (0.0, 1.0, "how often shapes overprint (third colours) instead of knocking out"),
    "ink_texture": (0.0, 1.0, "ink starvation speckle and density drift"),
    "edge_roughness": (0.0, 3.0, "shape edge wobble"),
    "sky_gradient": (0.0, 0.8, "halftone gradient in the sky"),
    "water_lines": (0.0, 1.0, "wave line-work on water"),
    "far_tint": (0.2, 1.0, "tint of far objects; lower = more atmospheric depth"),
}

RULES = {"max_inks": 5, "max_color_clusters_per_ink": 6}

SCREEN_ANGLES = [15.0, 75.0, 0.0, 45.0, 30.0]


def _choose_inks(ctx: Context) -> list[np.ndarray]:
    hints = scene_hints(ctx.scene)
    n = int(ctx.params["inks"])
    if len(hints) > n:
        # Keep the lightest and darkest, then the most mutually distinct of the rest.
        order = sorted(range(len(hints)), key=lambda i: luminance(hints[i]))
        chosen = [order[0], order[-1]]
        rest = [i for i in order if i not in chosen]
        while len(chosen) < n and rest:
            best = max(rest, key=lambda i: min(np.sum((hints[i] - hints[j]) ** 2) for j in chosen))
            chosen.append(best)
            rest.remove(best)
        hints = [hints[i] for i in chosen]
    # Print order: lightest first.
    return sorted(hints, key=lambda c: -luminance(c))


class _Roles:
    """Resolve palette roles to (ink index | None for paper, tint)."""

    def __init__(self, inks, paper_rgb, mood):
        self.inks = inks
        self.dark = int(np.argmin([luminance(c) for c in inks]))
        self.light = int(np.argmax([luminance(c) for c in inks]))
        night = mood == "night"
        self.sky = self.dark if night else nearest(inks, ROLE_TARGETS["sky"])
        self.sky_tint = 0.85 if night else 1.0
        self.paper_rgb = paper_rgb
        self.night = night

    def resolve(self, role: str, avoid=()):
        if role.startswith("#"):
            return nearest(self.inks, role), 1.0
        if role == "light":
            if luminance(self.inks[self.light]) > 0.72 and self.light != self.sky:
                return self.light, 1.0
            return None, 0.0  # knock out to paper
        if role == "dark":
            return self.dark, 1.0
        if role == "sky":
            return self.sky, self.sky_tint
        candidates = [i for i in range(len(self.inks)) if i != self.sky or len(self.inks) <= 2]
        if len([c for c in candidates if c not in avoid]) >= 1:
            candidates = [c for c in candidates if c not in avoid]
        sub = [self.inks[i] for i in candidates]
        target = ROLE_TARGETS.get(role, ROLE_TARGETS["primary"])
        k = candidates[nearest(sub, target)]
        return k, 1.0

    def shadow_of(self, k):
        """The ink that prints the shadow halftone for a part in ink k."""
        if k is None:
            order = sorted(range(len(self.inks)), key=lambda i: luminance(self.inks[i]))
            return order[len(order) // 2]
        darker = [i for i in range(len(self.inks)) if luminance(self.inks[i]) < luminance(self.inks[k]) - 0.08]
        if not darker:
            return None
        return max(darker, key=lambda i: luminance(self.inks[i]))  # the next-darker ink, not the key


def render_array(scene, seed=None, params=None):
    params = clamp_params(merge_params(DEFAULTS, params), KNOBS)
    ctx = make_context(scene, seed, params, NAME)
    f, rng, P = ctx.frame, ctx.rng, params
    H, W = f.shape
    X, Y = f.grid
    sc = f.scale

    inks = _choose_inks(ctx)
    n = len(inks)
    paper_rgb_hex = scene.palette.paper or "#f1e6cf"
    roles = _Roles(inks, hex_to_rgb(paper_rgb_hex), scene.palette.mood)
    plates = np.zeros((n, H, W), dtype=np.float32)
    cell = P["halftone_cell"] * sc
    angle = lambda k: SCREEN_ANGLES[k % len(SCREEN_ANGLES)]  # noqa: E731
    rough = (value_noise(f.shape, 9 * sc, rng.child("rough")) - 0.5) * 2 * P["edge_roughness"] * sc

    def screen(k, tone):
        return halftone.dots(tone, f, cell, angle(k))

    def lay(k, m, cov=1.0, knockout=True):
        """Put ink k down under mask m. k=None knocks out to paper."""
        if knockout:
            keep = 1.0 - m
            for j in range(n):
                if j != k:
                    plates[j] *= keep
        if k is not None:
            plates[k] = plates[k] * (1.0 - m) + m * cov if knockout else np.maximum(plates[k], m * cov)

    horizon_px = scene.horizon * H

    # Sky ------------------------------------------------------------------
    sky_k, sky_t = roles.resolve("sky")
    plates[sky_k] = 1.0 if sky_t >= 0.99 else screen(sky_k, np.full(f.shape, sky_t, np.float32))
    if P["sky_gradient"] > 0:
        warm = [i for i in range(n) if i not in (sky_k, roles.dark)]
        if warm:
            g = warm[0] if roles.night else max(warm, key=lambda i: inks[i][0] - inks[i][2])
            ramp = np.clip(1.0 - (horizon_px - Y) / max(horizon_px * 0.75, 1), 0, 1) ** 1.6
            ramp = np.where(Y > horizon_px, 0.0, ramp) * P["sky_gradient"]
            plates[g] = np.maximum(plates[g], screen(g, ramp))
            ctx.info["sky_gradient_ink"] = g

    # Objects ---------------------------------------------------------------
    orng = rng.child("objects")
    lights = [s for s in ctx.shapes if s.kind in ("sun", "moon")]
    for si, shape in enumerate(ctx.shapes):
        r = orng.child(str(si))
        overprint = (not shape.region and shape.kind not in SKY_KINDS and shape.kind not in ("bird", "rain")
                     and r.random() < P["overprint"])
        far = shape.depth >= 0.85 and shape.kind in ("mountain", "hills")
        # What is already printed behind this object (measured once, before any of its own parts).
        sm = shape.mask
        under = (plates * sm[None]).reshape(n, -1).sum(1) / max(float(sm.sum()), 1.0)
        for pi, part in enumerate(shape.parts):
            m = sdf_to_mask(part.sdf + rough)
            if not m.any():
                continue
            role = shape.obj.color if (shape.obj.color and pi == 0 and shape.obj.color.startswith("#")) else part.role
            avoid = ()
            if not shape.region and not part.detail:
                # Figure/ground: never print an object in the same ink that dominates behind it.
                avoid = tuple(int(j) for j in np.nonzero(under > 0.25)[0])
                if overprint and pi == 0:
                    # Overprinting onto dark ink makes mud; only overprint onto light grounds.
                    lum_under = float(np.prod([1 - u * (1 - luminance(inks[j])) for j, u in enumerate(under)]))
                    if lum_under < 0.55:
                        overprint = False
            k, tint = roles.resolve(role, avoid)
            if part.detail and P["key_details"] and part.role == "dark":
                k, tint = roles.dark, 1.0
            if far and k is not None:
                layer = int(part.name[-1]) if part.name[-1].isdigit() else 0
                tint = min(1.0, P["far_tint"] + 0.2 * layer)
            cov = 1.0 if (k is None or tint >= 0.99) else screen(k, np.full(f.shape, tint, np.float32))
            lay(k, m, cov, knockout=k is None or not (overprint and not part.detail))

            if part.shade and P["shade_strength"] > 0 and not shape.region:
                sh = roles.shadow_of(k)
                if sh is not None:
                    tone = shade_field(part, ctx.light_dir, f) * P["shade_strength"] * (0.6 if k is None else 1.0)
                    plates[sh] = np.maximum(plates[sh], m * screen(sh, tone))

            if shape.kind in ("sea", "river") and P["water_lines"] > 0:
                _water(plates, m, part, k, lights, roles, ctx, screen, horizon_px)
            if shape.kind == "field":
                g = roles.shadow_of(k)
                if g is not None:
                    depth_t = np.clip((Y - horizon_px) / max(H - horizon_px, 1), 0, 1)
                    wob = (value_noise(f.shape, 40 * sc, r.child("fw")) - 0.5) * 8 * sc
                    # Furrows: sparse, thin, widening toward the viewer, broken by noise.
                    fade = np.clip((depth_t - 0.04) / 0.25, 0, 1)
                    # Continuous furrow rows converging to the horizon, thinning with distance.
                    lines = halftone.lines((0.06 + 0.2 * depth_t ** 1.5) * fade, f, spacing=cell * (1.6 + 3.5 * depth_t), angle_deg=0, wobble=wob)
                    plates[g] = np.maximum(plates[g], m * lines)

        if shape.kind in ("sun", "moon") and P["halo_rings"]:
            k, _ = roles.resolve(shape.parts[0].role)
            k = k if k is not None else roles.light
            dist = np.hypot(X - shape.cx, Y - shape.cy) - shape.radius
            rings = np.zeros(f.shape, np.float32)
            for j, gap in enumerate((0.35, 0.8, 1.4)):
                w = shape.radius * (0.1 - 0.025 * j)
                rings = np.maximum(rings, sdf_to_mask(np.abs(dist - shape.radius * gap) - w) * (0.9 - 0.2 * j))
            rings *= (Y < horizon_px) if shape.cy < horizon_px else 1.0
            plates[k] = np.maximum(plates[k], rings)

    # Plate physics: misregistration, ink starvation, density drift --------
    prng = rng.child("plates")
    fib_img, fib = paper(f.shape, rng.child("paper"), tint=paper_rgb_hex, strength=1.0, scale=sc)
    out = fib_img.copy()
    offsets = []
    for k in range(n):
        pr = prng.child(str(k))
        dx, dy = pr.uniform(-1, 1, 2) * P["misregistration"] * sc
        offsets.append((round(float(dx), 2), round(float(dy), 2)))
        plate = ndimage.shift(plates[k], (dy, dx), order=1, mode="nearest")
        drift = 1.0 - 0.12 * P["ink_texture"] * value_noise(f.shape, 300 * sc, pr.child("drift"))
        streak = blurred_white((1, W), pr.child("streak"), (0, 25 * sc))[0][None, :]
        drift *= 1.0 - 0.08 * P["ink_texture"] * (streak - 0.5)
        starve = fbm(f.shape, 18 * sc, pr.child("starve"), octaves=3)
        holes = np.clip((starve - (1.0 - 0.18 * P["ink_texture"])) * 12.0, 0, 1)
        speck = (pr.random(f.shape) < 0.012 * P["ink_texture"]).astype(np.float32)
        speck = ndimage.maximum_filter(speck, size=max(1, int(round(1.5 * sc))))
        density = plate * drift * (1.0 - holes) * (1.0 - 0.8 * speck) * (0.93 + 0.07 * fib)
        out = out * (1.0 - np.clip(density, 0, 1)[..., None] * (1.0 - inks[k]))
    ctx.info.update({
        "inks": [f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in inks],
        "ink_count": n,
        "paper": paper_rgb_hex,
        "offsets": offsets,
    })
    return np.clip(out, 0, 1), ctx


def _water(plates, m, part, k, lights, roles, ctx, screen, horizon_px):
    """Wave line-work and broken reflections of sun/moon on water."""
    f, P = ctx.frame, ctx.params
    X, Y = f.grid
    sc = f.scale
    r = ctx.rng.child("water")
    depth_t = np.clip((Y - horizon_px) / max(f.height - horizon_px, 1), 0, 1)
    wob = (value_noise(f.shape, 90 * sc, r.child("w")) - 0.5) * 6 * sc
    # Perspective: line spacing grows toward the viewer.
    spacing = (4 + 22 * depth_t ** 1.3) * sc
    row = np.floor((Y + wob) / spacing)
    phase = ((Y + wob) / spacing) % 1.0
    # Each wave row is broken into dashes by 1-D noise along x, re-seeded per row.
    dash_n = value_noise(f.shape, 55 * sc, r.child("b"))
    row_jit = (np.sin(row * 12.9898) * 43758.5453) % 1.0
    dash = ((dash_n + row_jit * 0.6) % 1.0)
    amount = P["water_lines"] * (0.35 + 0.65 * depth_t)
    darker = roles.shadow_of(k) if k is not None else roles.dark
    lighter = roles.light if roles.light != k else None
    thin = np.abs(phase - 0.5) < (0.07 + 0.05 * depth_t)
    if darker is not None:
        trough = m * thin * (dash < 0.55 * amount)
        plates[darker] = np.maximum(plates[darker], trough.astype(np.float32))
    if lighter is not None and luminance(roles.inks[lighter]) > 0.7:
        crest = (m * (np.abs(((Y + wob) / spacing + 0.5) % 1.0 - 0.5) < 0.06) * (dash > 1.0 - 0.3 * amount)).astype(np.float32)
        for j in range(len(plates)):
            if j != lighter:
                plates[j] *= 1 - crest
        plates[lighter] = np.maximum(plates[lighter], crest)
    for L in lights:
        if L.cy > horizon_px:
            continue
        lk, _ = roles.resolve(L.parts[0].role)
        lk = lk if lk is not None else roles.light
        # Broken reflection: one dash per wave row, its half-length set per row.
        row_len = (np.sin(row * 78.233 + 1.3) * 12345.678) % 1.0
        row_off = ((np.sin(row * 39.425 + 0.7) * 24634.634) % 1.0 - 0.5) * L.radius * 0.5
        half = L.radius * (0.25 + 0.95 * row_len) * (0.5 + 0.8 * depth_t) * (1.0 - 0.5 * (dash_n - 0.5))
        bar = np.abs(X - L.cx - row_off) < half
        refl = (m * bar * (np.abs(phase - 0.5) < 0.2) * (Y > horizon_px + 2) * (row_len > 0.12)).astype(np.float32)
        for j in range(len(plates)):
            if j != lk:
                plates[j] *= 1 - refl
        plates[lk] = np.maximum(plates[lk], refl)


def render(scene, seed=None, params=None) -> bytes:
    img, ctx = render_array(scene, seed, params)
    return encode(img, NAME, ctx)
