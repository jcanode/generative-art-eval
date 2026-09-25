"""The shared object vocabulary: style-neutral silhouettes as signed distance fields.

Every style reads the same geometry, so the three styles agree on WHAT is WHERE
and differ only in treatment. Each kind builds a ``Shape`` made of named
``Part``s; a part has a pixel-space SDF (negative inside) and a palette *role*
that the style resolves to an ink or paint colour.

Roles: sky, water, ground, primary, secondary, accent, dark, light, foliage,
wood, stone. Parts flagged ``detail`` are thin (masts, windows, stems) and
styles usually draw them as line work rather than fills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import sdf as S
from ..core.geom import Frame, rotate
from ..core.rng import Rng
from .spec import Scene, SceneObject

SKY_KINDS = {"sun", "moon", "stars", "cloud"}
REGION_KINDS = {"sea", "field", "table", "river"}


@dataclass
class Part:
    name: str
    sdf: np.ndarray
    role: str
    detail: bool = False
    shade: bool = True
    _stats: tuple | None = field(default=None, repr=False)

    @property
    def mask(self) -> np.ndarray:
        return np.clip(0.5 - self.sdf, 0.0, 1.0).astype(np.float32)

    def stats(self):
        """(cx, cy, radius, bbox) of the inside region, in px. bbox = (x0, y0, x1, y1)."""
        if self._stats is None:
            ys, xs = np.nonzero(self.sdf < 0)
            if len(xs) == 0:
                self._stats = (0.0, 0.0, 0.0, (0, 0, 0, 0))
            else:
                cx, cy = float(xs.mean()), float(ys.mean())
                r = float(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 1)
                self._stats = (cx, cy, r, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        return self._stats


@dataclass
class Shape:
    obj: SceneObject
    kind: str
    parts: list[Part]
    cx: float
    cy: float
    radius: float
    depth: float
    region: bool = False

    @property
    def sdf(self) -> np.ndarray:
        return S.union(*[p.sdf for p in self.parts])

    @property
    def mask(self) -> np.ndarray:
        return np.clip(0.5 - self.sdf, 0.0, 1.0).astype(np.float32)

    def bbox(self):
        boxes = [p.stats()[3] for p in self.parts if p.stats()[2] > 0]
        if not boxes:
            return (0, 0, 0, 0)
        b = np.array(boxes)
        return (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max()))

    def part(self, name) -> Part | None:
        return next((p for p in self.parts if p.name == name), None)


def shade_field(part: Part, light_dir, frame: Frame, softness: float = 1.0):
    """Stylised form shading in [0, 1] (1 = shadow side), the print/paint way.

    Not a physical lighting model: tone ramps across the part away from the
    light, with a crescent of core shadow near the far edge. Reads as volume
    without pretending to be a photograph.
    """
    cx, cy, r, _ = part.stats()
    if r <= 0:
        return np.zeros(frame.shape, np.float32)
    X, Y = frame.grid
    lx, ly = light_dir
    u = ((X - cx) * lx + (Y - cy) * ly) / r  # +1 toward the light
    ramp = np.clip(0.5 - 0.5 * u * softness, 0, 1)
    edge = np.exp(-np.clip(-part.sdf, 0, None) / max(r * 0.25, 1.0))
    core = edge * np.clip(-u, 0, 1)
    return np.clip(0.75 * ramp + 0.5 * core, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Kind builders. Signature: (obj, frame, scene, rng) -> list[Part]
# Coordinates: (cx, cy) px centre, s px size (fraction of canvas height).
# ---------------------------------------------------------------------------

def _grid(frame):
    return frame.grid


def _sun(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    r = f.size(o.size) / 2
    return [Part("disc", S.circle(X, Y, cx, cy, r), "accent", shade=False)]


def _moon(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    r = f.size(o.size) / 2
    disc = S.circle(X, Y, cx, cy, r)
    if (o.variant or "crescent") == "full":
        return [Part("disc", disc, "light", shade=False)]
    ang = math.radians(o.params.get("phase_angle", 35))
    cut = S.circle(X, Y, cx + r * 0.55 * math.cos(ang), cy - r * 0.55 * math.sin(ang), r * 0.92)
    return [Part("crescent", S.subtract(disc, cut), "light", shade=False)]


def _stars(o, f, sc, rng):
    X, Y = _grid(f)
    n = max(1, o.count if o.count > 1 else 14)
    top = f.height * sc.horizon * 0.85
    xs = rng.uniform(0.03, 0.97, n) * f.width
    ys = rng.uniform(0.03, 1.0, n) * top
    rs = f.size(o.size) * rng.uniform(0.012, 0.03, n) + 1.0
    d = np.full(f.shape, np.inf, np.float32)
    for x, y, r in zip(xs, ys, rs):
        core = S.circle(X, Y, x, y, r)
        ray_h = S.tapered_segment(X, Y, x - r * 3, y, x + r * 3, y, 0.3, 0.3)
        ray_v = S.tapered_segment(X, Y, x, y - r * 3, x, y + r * 3, 0.3, 0.3)
        d = np.minimum(d, S.union(core, ray_h, ray_v))
    return [Part("stars", d, "light", detail=True, shade=False)]


def _cloud(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    n = 5
    offs = np.linspace(-0.5, 0.5, n)
    d = np.full(f.shape, np.inf, np.float32)
    for i, u in enumerate(offs):
        rr = s * (0.22 + 0.16 * (1 - abs(u) * 1.8)) * rng.uniform(0.85, 1.15)
        puff = S.circle(X, Y, cx + u * s * 1.2, cy - rr * 0.5 + s * 0.1, rr)
        d = puff if i == 0 else S.smooth_union(d, puff, s * 0.08)
    d = S.intersect(d, Y - (cy + s * 0.12))
    return [Part("cloud", d, "light")]


def _mountain(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    base = cy + s / 2
    peaks = max(1, o.count)
    parts = []
    d_all = np.full(f.shape, np.inf, np.float32)
    snow_all = np.full(f.shape, np.inf, np.float32)
    for k in range(peaks):
        px_ = cx + (k - (peaks - 1) / 2) * s * 0.9 + rng.uniform(-0.1, 0.1) * s
        h = s * (1.0 if k == peaks // 2 else rng.uniform(0.6, 0.85))
        top = base - h
        half = h * rng.uniform(1.0, 1.3)
        n = 9
        left = [(px_ - half * (1 - i / n) + rng.uniform(-0.03, 0.03) * s,
                 base - h * (i / n) + (rng.uniform(-0.04, 0.04) * s if 0 < i < n else 0)) for i in range(n)]
        right = [(px_ + half * (i / n) + rng.uniform(-0.03, 0.03) * s,
                  top + h * (i / n) + (rng.uniform(-0.04, 0.04) * s if 0 < i < n else 0)) for i in range(1, n + 1)]
        pts = left + [(px_, top)] + right
        d = S.polygon(X, Y, pts)
        snowline = top + h * rng.uniform(0.22, 0.3)
        wav = np.sin((X - px_) / (s * 0.06)) * s * 0.02
        snow = S.intersect(d, Y - snowline - wav)
        d_all = np.minimum(d_all, d)
        snow_all = np.minimum(snow_all, snow)
    parts.append(Part("rock", d_all, o.color or "secondary"))
    if o.params.get("snow", True):
        parts.append(Part("snow", snow_all, "light", shade=False))
    return parts


def _hills(o, f, sc, rng):
    X, Y = _grid(f)
    s = f.size(o.size)
    base = f.height * o.y
    layers = max(1, o.count)
    parts = []
    for k in range(layers):
        amp = s * (1.0 - 0.25 * k)
        ph = rng.uniform(0, 2 * np.pi, 3)
        fr = rng.uniform(0.7, 1.6, 3) * 2 * np.pi / f.width
        prof = sum(np.sin(X * fr[i] + ph[i]) * (0.5 ** i) for i in range(3)) / 1.75
        top = base - amp * (0.55 + 0.45 * prof) + k * s * 0.45
        d = S.intersect(top - Y, Y - f.height * 1.5)
        parts.append(Part(f"hill{k}", d.astype(np.float32), o.color or "secondary"))
    return parts


def _region_top(o, f, sc):
    return f.height * o.y


def _sea(o, f, sc, rng):
    X, Y = _grid(f)
    top = _region_top(o, f, sc)
    return [Part("water", (top - Y).astype(np.float32), o.color or "water", shade=False)]


def _field(o, f, sc, rng):
    X, Y = _grid(f)
    top = _region_top(o, f, sc)
    wav = np.sin(X / f.width * 2 * np.pi * rng.uniform(0.6, 1.2) + rng.uniform(0, 6)) * f.height * 0.012
    return [Part("ground", (top + wav - Y).astype(np.float32), o.color or "ground", shade=False)]


def _river(o, f, sc, rng):
    X, Y = _grid(f)
    top = f.height * sc.horizon
    cx = f.width * o.x
    t = np.clip((Y - top) / (f.height - top), 0, 1)
    half = f.width * (0.01 + o.size * 0.8 * t)
    bend = np.sin(t * 3.0 + rng.uniform(0, 3)) * f.width * 0.08 * t
    d = np.maximum(np.abs(X - cx - bend) - half, top - Y)
    return [Part("river", d.astype(np.float32), o.color or "water", shade=False)]


def _table(o, f, sc, rng):
    X, Y = _grid(f)
    top = _region_top(o, f, sc)
    edge = top + (f.height - top) * 0.62
    surface = np.maximum(top - Y, Y - edge)
    front = edge - Y
    return [Part("top", surface.astype(np.float32), o.color or "secondary", shade=False),
            Part("front", front.astype(np.float32), "dark", shade=False)]


def _cliff(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    side = -1 if o.x < 0.5 else 1
    outer = f.width if side > 0 else 0
    pts = [(outer, f.height), (outer, cy), (cx + side * s * 0.1, cy),
           (cx - side * s * 0.12, cy + s * 0.25), (cx - side * s * 0.05, cy + s * 0.5),
           (cx - side * s * 0.3, cy + s * 0.8), (cx - side * s * 0.35, f.height)]
    return [Part("rock", S.polygon(X, Y, pts), o.color or "stone")]


def _tree(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # (x, y) = base of the trunk
    s = f.size(o.size)
    top = cy - s * 0.5
    # Tapering trunk with a flat base and a slight root flare.
    v = np.clip((cy - Y) / (s * 0.5), 0, 1)
    half = s * (0.028 + 0.03 * (1 - v) ** 4 + 0.012 * (1 - v))
    trunk = np.maximum(np.abs(X - cx - s * 0.02 * v) - half, np.maximum(Y - cy, top - Y)).astype(np.float32)
    # Two short limbs into the crown.
    limbs = S.union(S.tapered_segment(X, Y, cx, cy - s * 0.35, cx - s * 0.13, cy - s * 0.55, s * 0.02, s * 0.01),
                    S.tapered_segment(X, Y, cx + s * 0.01, cy - s * 0.4, cx + s * 0.14, cy - s * 0.58, s * 0.018, s * 0.01))
    crown_c = (cx + s * 0.01, cy - s * 0.7)
    d = S.ellipse(X, Y, crown_c[0], crown_c[1], s * 0.27, s * 0.2)
    for i in range(11):
        a = i / 11 * 2 * np.pi + rng.uniform(-0.25, 0.25)
        rr = s * rng.uniform(0.09, 0.14)
        px_ = crown_c[0] + np.cos(a) * s * 0.27
        py_ = crown_c[1] + np.sin(a) * s * 0.19 * (0.8 if np.sin(a) > 0 else 1.0)
        d = S.smooth_union(d, S.circle(X, Y, px_, py_, rr), s * 0.03)
    return [Part("trunk", S.union(trunk, limbs), "wood"), Part("crown", d, o.color or "foliage")]


def _pine(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    trunk = S.box(X, Y, cx, cy - s * 0.06, s * 0.035, s * 0.07)
    d = None
    tiers = 4
    for i in range(tiers):
        yb = cy - s * 0.1 - i * s * 0.19
        w = s * 0.3 * (1 - i / (tiers + 0.8))
        th = s * 0.32
        tri = S.polygon(X, Y, [(cx - w, yb), (cx - w * 0.7, yb - th * 0.08), (cx + w * 0.7, yb - th * 0.08), (cx + w, yb),
                               (cx + w * 0.3, yb - th * 0.5), (cx, yb - th), (cx - w * 0.3, yb - th * 0.5)])
        d = tri if d is None else np.minimum(d, tri)
    return [Part("trunk", trunk, "wood"), Part("needles", d, o.color or "foliage")]


def _cypress(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    v = np.clip((cy - Y) / s, 0, 1)  # 0 base .. 1 tip
    # Flame silhouette: widest a third of the way up, drawn to a sharp, slightly leaning point.
    half = s * 0.115 * np.clip(np.sin(np.pi * np.clip(0.18 + v * 0.95, 0, 1)), 0, 1) ** 0.9 * (1 - v ** 1.6)
    lean = s * 0.06 * v ** 2
    # Tufts: alternating lobes up both sides give the flickering outline.
    lobes = (0.5 + 0.5 * np.sin(v * 38 + rng.uniform(0, 6))) * s * 0.03 * (1 - v)
    d = np.maximum(np.abs(X - cx - lean) - half - lobes, np.maximum(Y - cy, (cy - s) - Y))
    trunk = S.box(X, Y, cx, cy - s * 0.01, s * 0.02, s * 0.03)
    return [Part("flame", np.minimum(d, trunk).astype(np.float32), o.color or "foliage")]


def _leaf(X, Y, ax, ay, bx, by, w):
    """Lanceolate leaf from base A to tip B: narrow stalk, widest at 30 %, needle tip."""
    mx, my = ax + (bx - ax) * 0.3, ay + (by - ay) * 0.3
    return np.minimum(S.tapered_segment(X, Y, ax, ay, mx, my, w * 0.25, w),
                      S.tapered_segment(X, Y, mx, my, bx, by, w, 0.3))


def _bamboo(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    n = max(1, o.count if o.count > 1 else 3)
    stalks = np.full(f.shape, np.inf, np.float32)
    nodes = np.full(f.shape, np.inf, np.float32)
    leaves = np.full(f.shape, np.inf, np.float32)
    for k in range(n):
        bx = cx + (k - (n - 1) / 2) * s * 0.16 + rng.uniform(-0.03, 0.03) * s
        lean = rng.uniform(-0.08, 0.08) * s
        top = cy - s * rng.uniform(0.85, 1.05)
        w = s * 0.022
        stalks = np.minimum(stalks, S.segment(X, Y, bx, cy, bx + lean, top, w))
        for j in range(1, 7):
            t = j / 7
            ny = cy + (top - cy) * t
            nx = bx + lean * t
            nodes = np.minimum(nodes, S.box(X, Y, nx, ny, w * 1.5, max(1.0, s * 0.005)))
        for j in range(2):
            t = rng.uniform(0.5, 0.95)
            ax_, ay_ = bx + lean * t, cy + (top - cy) * t
            side = 1 if rng.random() < 0.5 else -1
            for m in range(int(rng.integers(3, 5))):
                # Leaves fan out and droop from a node, all to one side.
                ang = side * rng.uniform(0.15, 1.1) + np.pi / 2 * (1 - side) + 0.35 * side
                ln = s * rng.uniform(0.12, 0.2)
                ex_, ey_ = ax_ + np.cos(ang) * ln, ay_ + abs(np.sin(ang)) * ln * 0.5 + ln * 0.2
                leaves = np.minimum(leaves, _leaf(X, Y, ax_, ay_, ex_, ey_, s * 0.017))
    return [Part("stalks", stalks, o.color or "foliage"), Part("nodes", nodes, "dark", detail=True, shade=False),
            Part("leaves", leaves, o.color or "foliage", shade=False)]


def _reeds(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    n = max(3, o.count if o.count > 1 else 9)
    d = np.full(f.shape, np.inf, np.float32)
    for k in range(n):
        bx = cx + rng.uniform(-0.3, 0.3) * s
        h = s * rng.uniform(0.6, 1.0)
        lean = rng.uniform(-0.25, 0.25) * s
        d = np.minimum(d, S.tapered_segment(X, Y, bx, cy, bx + lean, cy - h, s * 0.012, 0.4))
    return [Part("reeds", d, o.color or "foliage", detail=True, shade=False)]


def _house(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # base centre
    s = f.size(o.size)
    w, h = s * 0.5, s * 0.42
    body = S.box(X, Y, cx, cy - h / 2, w / 2, h / 2)
    roof_h = s * 0.32
    roof = S.polygon(X, Y, [(cx - w * 0.62, cy - h), (cx + w * 0.62, cy - h), (cx, cy - h - roof_h)])
    chim = S.box(X, Y, cx + w * 0.25, cy - h - roof_h * 0.55, s * 0.035, roof_h * 0.35)
    door = S.box(X, Y, cx - w * 0.15, cy - h * 0.25, w * 0.1, h * 0.25)
    win = S.union(S.box(X, Y, cx + w * 0.2, cy - h * 0.55, w * 0.1, h * 0.12),
                  S.box(X, Y, cx - w * 0.15, cy - h * 0.72, w * 0.07, h * 0.08) if rng.random() < 0.3 else np.full(f.shape, np.inf, np.float32))
    return [Part("walls", body, o.color or "light"), Part("roof", S.union(roof, chim), "accent"),
            Part("door", door, "dark", detail=True, shade=False), Part("windows", win, "dark", detail=True, shade=False)]


def _lighthouse(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # base centre
    s = f.size(o.size)
    hb, ht, h = s * 0.12, s * 0.075, s * 0.72
    tower = S.polygon(X, Y, [(cx - hb, cy), (cx + hb, cy), (cx + ht, cy - h), (cx - ht, cy - h)])
    stripes = S.intersect(tower, (np.abs(((cy - Y) / (h / 5)) % 2 - 1) - 0.5) * (h / 5))
    gallery = S.box(X, Y, cx, cy - h - s * 0.015, ht * 1.45, s * 0.018)
    lantern = S.box(X, Y, cx, cy - h - s * 0.08, ht * 0.8, s * 0.05)
    cap = S.polygon(X, Y, [(cx - ht * 1.05, cy - h - s * 0.13), (cx + ht * 1.05, cy - h - s * 0.13), (cx, cy - h - s * 0.22)])
    parts = [Part("tower", tower, o.color or "light"), Part("stripes", stripes, "accent"),
             Part("gallery", S.union(gallery, cap), "dark", shade=False),
             Part("lantern", lantern, "accent", shade=False)]
    if o.params.get("beam", True):
        lx, ly = cx, cy - h - s * 0.08
        beam = S.intersect(S.polygon(X, Y, [(lx, ly - 2), (lx + s * 1.6, ly - s * 0.22), (lx + s * 1.6, ly + s * 0.12), (lx, ly + 2)]),
                           -S.circle(X, Y, lx, ly, ht * 0.8))
        parts.insert(0, Part("beam", beam, "light", shade=False))
    return parts


def _windmill(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)
    s = f.size(o.size)
    h = s * 0.62
    tower = S.polygon(X, Y, [(cx - s * 0.13, cy), (cx + s * 0.13, cy), (cx + s * 0.07, cy - h), (cx - s * 0.07, cy - h)])
    cap = S.polygon(X, Y, [(cx - s * 0.1, cy - h), (cx + s * 0.1, cy - h), (cx, cy - h - s * 0.1)])
    door = S.box(X, Y, cx, cy - s * 0.06, s * 0.035, s * 0.06)
    hx, hy = cx, cy - h - s * 0.02
    rot = math.radians(o.rotation or rng.uniform(10, 35))
    sails = np.full(f.shape, np.inf, np.float32)
    spars = np.full(f.shape, np.inf, np.float32)
    for k in range(4):
        a = rot + k * np.pi / 2
        ux, uy = math.cos(a), math.sin(a)
        L = s * 0.5
        spars = np.minimum(spars, S.segment(X, Y, hx, hy, hx + ux * L, hy + uy * L, s * 0.012))
        # sail lattice panel to one side of the spar
        px_, py_ = -uy, ux
        a0, a1 = 0.18 * L, L
        quad = [(hx + ux * a0, hy + uy * a0), (hx + ux * a1, hy + uy * a1),
                (hx + ux * a1 + px_ * s * 0.1, hy + uy * a1 + py_ * s * 0.1),
                (hx + ux * a0 + px_ * s * 0.1, hy + uy * a0 + py_ * s * 0.1)]
        sails = np.minimum(sails, S.polygon(X, Y, quad))
    hub = S.circle(X, Y, hx, hy, s * 0.025)
    return [Part("tower", tower, o.color or "light"), Part("cap", cap, "accent"),
            Part("door", door, "dark", detail=True, shade=False),
            Part("sails", sails, "light", shade=False),
            Part("spars", S.union(spars, hub), "dark", detail=True, shade=False)]


def _boat(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # waterline centre
    s = f.size(o.size)
    L = s * 0.9
    flip = -1 if o.params.get("facing", "right") == "left" else 1
    hull_h = s * 0.13
    hull = S.polygon(X, Y, [
        (cx - L / 2, cy - hull_h), (cx + L / 2 * 1.08, cy - hull_h * 1.25),
        (cx + L * 0.36, cy + hull_h * 0.15), (cx - L * 0.38, cy + hull_h * 0.15),
    ])
    if flip < 0:
        hull = S.polygon(X, Y, [
            (cx + L / 2, cy - hull_h), (cx - L / 2 * 1.08, cy - hull_h * 1.25),
            (cx - L * 0.36, cy + hull_h * 0.15), (cx + L * 0.38, cy + hull_h * 0.15)])
    parts = [Part("hull", hull, o.color or "dark")]
    if (o.variant or "sail") in ("sail", "sailboat"):
        mx = cx - flip * L * 0.02
        mtop = cy - hull_h - s * 0.9
        mast = S.segment(X, Y, mx, cy - hull_h, mx, mtop, max(1.2, s * 0.008))
        main = S.polygon(X, Y, [(mx + flip * s * 0.03, mtop + s * 0.05), (mx + flip * s * 0.03, cy - hull_h - s * 0.08),
                                (mx + flip * L * 0.46, cy - hull_h - s * 0.08)])
        jib = S.polygon(X, Y, [(mx - flip * s * 0.03, mtop + s * 0.12), (mx - flip * s * 0.03, cy - hull_h - s * 0.1),
                               (mx - flip * L * 0.42, cy - hull_h - s * 0.1)])
        parts += [Part("sail", S.union(main, jib), "light"), Part("mast", mast, "dark", detail=True, shade=False)]
    elif o.variant == "fishing":
        cab = S.box(X, Y, cx - flip * L * 0.1, cy - hull_h - s * 0.1, L * 0.14, s * 0.1)
        mast = S.segment(X, Y, cx + flip * L * 0.2, cy - hull_h, cx + flip * L * 0.2, cy - hull_h - s * 0.45, max(1.2, s * 0.01))
        line = S.segment(X, Y, cx + flip * L * 0.2, cy - hull_h - s * 0.45, cx + flip * L * 0.62, cy - hull_h * 0.2, max(0.8, s * 0.004))
        parts += [Part("cabin", cab, "light"), Part("mast", S.union(mast, line), "dark", detail=True, shade=False)]
    return parts


def _bird(o, f, sc, rng):
    X, Y = _grid(f)
    s = f.size(o.size)
    n = max(1, o.count)
    d = np.full(f.shape, np.inf, np.float32)
    if (o.variant or "flying") == "perched":
        cx, cy = f.px(o.x, o.y)
        body = S.ellipse(X, Y, cx, cy, s * 0.28, s * 0.16)
        head = S.circle(X, Y, cx + s * 0.24, cy - s * 0.14, s * 0.1)
        beak = S.polygon(X, Y, [(cx + s * 0.32, cy - s * 0.17), (cx + s * 0.45, cy - s * 0.13), (cx + s * 0.32, cy - s * 0.1)])
        tail = S.polygon(X, Y, [(cx - s * 0.2, cy - s * 0.04), (cx - s * 0.55, cy + s * 0.06), (cx - s * 0.5, cy + s * 0.15), (cx - s * 0.18, cy + s * 0.08)])
        wing = S.ellipse(X, Y, cx - s * 0.05, cy - s * 0.02, s * 0.2, s * 0.09)
        eye = S.circle(X, Y, cx + s * 0.27, cy - s * 0.16, max(1.0, s * 0.018))
        legs = S.union(S.segment(X, Y, cx, cy + s * 0.12, cx - s * 0.02, cy + s * 0.26, max(0.8, s * 0.012)),
                       S.segment(X, Y, cx + s * 0.08, cy + s * 0.12, cx + s * 0.07, cy + s * 0.26, max(0.8, s * 0.012)))
        return [Part("body", S.union(body, head, tail), o.color or "primary"), Part("beak", beak, "accent", shade=False),
                Part("wing", wing, "dark"), Part("eye", eye, "dark", detail=True, shade=False),
                Part("legs", legs, "dark", detail=True, shade=False)]
    for k in range(n):
        if n == 1:
            cx, cy, ss = *f.px(o.x, o.y), s
        else:
            cx = (o.x + rng.uniform(-0.12, 0.12)) * f.width
            cy = (o.y + rng.uniform(-0.07, 0.07)) * f.height
            ss = s * rng.uniform(0.6, 1.0)
        flap = rng.uniform(0.12, 0.3)
        w = S.union(
            S.tapered_segment(X, Y, cx, cy, cx - ss * 0.22, cy - ss * flap, ss * 0.035, ss * 0.02),
            S.tapered_segment(X, Y, cx - ss * 0.22, cy - ss * flap, cx - ss * 0.5, cy - ss * flap * 0.3, ss * 0.02, 0.5),
            S.tapered_segment(X, Y, cx, cy, cx + ss * 0.22, cy - ss * flap, ss * 0.035, ss * 0.02),
            S.tapered_segment(X, Y, cx + ss * 0.22, cy - ss * flap, cx + ss * 0.5, cy - ss * flap * 0.3, ss * 0.02, 0.5),
        )
        d = np.minimum(d, w)
    return [Part("birds", d, o.color or "dark", detail=True, shade=False)]


def _profile_sdf(X, Y, cx, top, height, vs, rs):
    """Solid of revolution silhouette: half-width r(v) interpolated from control points."""
    v = (Y - top) / height
    r = np.interp(np.clip(v, 0, 1), vs, rs)
    return np.maximum(np.abs(X - cx) - r, np.maximum(-v, v - 1) * height).astype(np.float32)


def _vase(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # base centre
    s = f.size(o.size)
    top = cy - s
    body = _profile_sdf(X, Y, cx, top, s, [0, 0.05, 0.2, 0.33, 0.62, 0.9, 1.0],
                        np.array([0.15, 0.13, 0.1, 0.16, 0.3, 0.2, 0.15]) * s)
    lip = S.ellipse(X, Y, cx, top + s * 0.01, s * 0.16, s * 0.025)
    band = S.intersect(body, np.abs(Y - (top + s * 0.45)) - s * 0.035)
    return [Part("body", S.union(body, lip), o.color or "primary"), Part("band", band, "accent", shade=False)]


def _fruit(o, f, sc, rng):
    X, Y = _grid(f)
    s = f.size(o.size)
    n = max(1, o.count)
    variant = o.variant or "apple"
    bodies = np.full(f.shape, np.inf, np.float32)
    stems = np.full(f.shape, np.inf, np.float32)
    leaves = np.full(f.shape, np.inf, np.float32)
    cx0, cy0 = f.px(o.x, o.y)
    for k in range(n):
        cx = cx0 + (k - (n - 1) / 2) * s * 0.85 + rng.uniform(-0.05, 0.05) * s
        cy = cy0 + (abs(k - (n - 1) / 2) * s * 0.06 if n > 1 else 0) - s * 0.5
        r = s * 0.5 * rng.uniform(0.88, 1.05)
        if variant == "pear":
            b = S.smooth_union(S.circle(X, Y, cx, cy + r * 0.25, r * 0.78), S.circle(X, Y, cx + r * 0.08, cy - r * 0.55, r * 0.45), r * 0.4)
            st_top = cy - r * 1.0
        elif variant == "orange":
            b = S.circle(X, Y, cx, cy, r * 0.92)
            st_top = cy - r * 0.92
        else:
            b = S.subtract(S.circle(X, Y, cx, cy, r), S.circle(X, Y, cx, cy - r * 1.05, r * 0.2))
            st_top = cy - r * 0.85
        bodies = np.minimum(bodies, b)
        stems = np.minimum(stems, S.tapered_segment(X, Y, cx, st_top + r * 0.05, cx + r * 0.1, st_top - r * 0.28, max(1.0, r * 0.05), max(0.8, r * 0.035)))
        if variant != "orange":
            lx, ly = cx + r * 0.1, st_top - r * 0.2
            leaves = np.minimum(leaves, S.tapered_segment(X, Y, lx, ly, lx + r * 0.42, ly - r * 0.12, r * 0.12, 0.5))
    return [Part("fruit", bodies, o.color or "accent"), Part("stem", stems, "wood", detail=True, shade=False),
            Part("leaf", leaves, "foliage", shade=False)]


def _teapot(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # base centre
    s = f.size(o.size)
    bw, bh = s * 0.36, s * 0.27
    by = cy - bh
    flat = by - bh * 0.72
    body = S.intersect(S.intersect(S.ellipse(X, Y, cx, by, bw, bh), Y - (cy - s * 0.01)), flat - Y)
    foot = S.box(X, Y, cx, cy - s * 0.02, bw * 0.55, s * 0.025)
    lid = S.intersect(S.ellipse(X, Y, cx, flat, bw * 0.46, bh * 0.3), Y - flat - 1.0)
    rim = S.box(X, Y, cx, flat, bw * 0.58, s * 0.014)
    knob = S.circle(X, Y, cx, flat - bh * 0.36, s * 0.04)
    spout = S.union(S.tapered_segment(X, Y, cx + bw * 0.7, by + bh * 0.25, cx + bw * 1.2, by - bh * 0.6, s * 0.09, s * 0.035),
                    S.tapered_segment(X, Y, cx + bw * 1.2, by - bh * 0.6, cx + bw * 1.38, by - bh * 0.8, s * 0.035, s * 0.028))
    ring = np.abs(S.ellipse(X, Y, cx - bw * 0.95, by - bh * 0.05, s * 0.15, s * 0.17)) - s * 0.032
    handle = S.intersect(ring, X - (cx - bw * 0.8))
    return [Part("body", S.union(body, foot, spout, handle), o.color or "primary"),
            Part("lid", S.union(lid, knob, rim), "secondary")]


def _cup(o, f, sc, rng):
    X, Y = _grid(f)
    cx, cy = f.px(o.x, o.y)  # base centre
    s = f.size(o.size)
    w_top, w_bot, h = s * 0.3, s * 0.22, s * 0.42
    body = S.polygon(X, Y, [(cx - w_top, cy - h), (cx + w_top, cy - h), (cx + w_bot, cy - s * 0.04),
                            (cx + w_bot * 0.8, cy - s * 0.01), (cx - w_bot * 0.8, cy - s * 0.01), (cx - w_bot, cy - s * 0.04)])
    rim = S.ellipse(X, Y, cx, cy - h, w_top, s * 0.05)
    inside = S.ellipse(X, Y, cx, cy - h + s * 0.005, w_top * 0.9, s * 0.035)
    ring = np.abs(S.ellipse(X, Y, cx + w_top * 1.0, cy - h * 0.55, s * 0.12, s * 0.12)) - s * 0.028
    handle = S.intersect(ring, (cx + w_top * 0.85) - X)
    saucer = S.ellipse(X, Y, cx, cy, s * 0.5, s * 0.07)
    return [Part("saucer", saucer, "secondary"), Part("cup", S.union(body, rim, handle), o.color or "light"),
            Part("tea", inside, "dark", shade=False)]


def _rain(o, f, sc, rng):
    X, Y = _grid(f)
    n = max(10, o.count if o.count > 1 else 80)
    xs = rng.uniform(-0.1, 1.05, n) * f.width
    ys = rng.uniform(0, 1.0, n) * f.height
    ln = f.size(o.size) * rng.uniform(0.2, 0.4, n)
    ang = math.radians(100)
    d = np.full(f.shape, np.inf, np.float32)
    for x, y, l in zip(xs, ys, ln):
        d = np.minimum(d, S.segment(X, Y, x, y, x + math.cos(ang) * l, y + math.sin(ang) * l, 0.7 * f.scale + 0.3))
    return [Part("rain", d, o.color or "dark", detail=True, shade=False)]


BUILDERS = {
    "sun": _sun, "moon": _moon, "stars": _stars, "cloud": _cloud,
    "mountain": _mountain, "hills": _hills, "sea": _sea, "field": _field, "river": _river,
    "table": _table, "cliff": _cliff,
    "tree": _tree, "pine": _pine, "cypress": _cypress, "bamboo": _bamboo, "reeds": _reeds,
    "house": _house, "lighthouse": _lighthouse, "windmill": _windmill,
    "boat": _boat, "bird": _bird,
    "vase": _vase, "fruit": _fruit, "teapot": _teapot, "cup": _cup,
    "rain": _rain,
}
KINDS = frozenset(BUILDERS)

# Plain-language names a blind judge might use for each kind (for subject-fidelity scoring).
SYNONYMS = {
    "sun": ["sun", "sunset", "sunrise", "orb"],
    "moon": ["moon", "crescent"],
    "stars": ["star", "stars"],
    "cloud": ["cloud", "clouds"],
    "mountain": ["mountain", "mountains", "peak", "peaks", "alps"],
    "hills": ["hill", "hills", "dunes", "slopes"],
    "sea": ["sea", "ocean", "water", "lake", "waves", "bay", "harbor", "harbour"],
    "field": ["field", "meadow", "ground", "grass", "land", "plain", "earth"],
    "river": ["river", "stream", "creek", "water"],
    "table": ["table", "tabletop", "surface", "shelf", "cloth", "tablecloth"],
    "cliff": ["cliff", "rock", "rocks", "headland", "bluff", "crag"],
    "tree": ["tree", "trees", "oak"],
    "pine": ["pine", "fir", "conifer", "evergreen", "spruce", "tree"],
    "cypress": ["cypress", "tree", "poplar", "flame", "conifer"],
    "bamboo": ["bamboo", "stalk", "stalks", "cane", "reeds"],
    "reeds": ["reed", "reeds", "grass", "rushes", "cattail"],
    "house": ["house", "houses", "cottage", "cabin", "building", "home", "village", "hut"],
    "lighthouse": ["lighthouse", "tower", "beacon"],
    "windmill": ["windmill", "mill"],
    "boat": ["boat", "boats", "sailboat", "sailboats", "ship", "yacht", "vessel", "sail", "dinghy"],
    "bird": ["bird", "birds", "gull", "gulls", "seagull", "seagulls", "swallow", "crow", "sparrow"],
    "vase": ["vase", "jug", "pitcher", "urn", "pot", "jar", "amphora", "vessel"],
    "fruit": ["fruit", "fruits", "apple", "apples", "orange", "oranges", "pear", "pears", "peach"],
    "teapot": ["teapot", "kettle", "pot"],
    "cup": ["cup", "teacup", "mug", "saucer"],
    "rain": ["rain", "raining", "rainfall", "drizzle", "storm", "shower"],
}

_DEFAULT_DEPTH = {"stars": 1.0, "sun": 0.98, "moon": 0.98, "cloud": 0.95, "mountain": 0.9, "hills": 0.85,
                  "sea": 0.8, "field": 0.8, "river": 0.78, "table": 0.75, "cliff": 0.6, "rain": 0.02, "bird": 0.05}


def infer_depth(o: SceneObject, scene: Scene) -> float:
    if o.depth is not None:
        return float(o.depth)
    if o.kind in _DEFAULT_DEPTH:
        return _DEFAULT_DEPTH[o.kind]
    # Lower on the canvas = nearer. Keep all objects in front of regions.
    return float(np.clip(0.55 - (o.y - scene.horizon) * 0.8, 0.1, 0.7))


def build_shapes(scene: Scene, frame: Frame, rng: Rng) -> list[Shape]:
    """All objects as Shapes, sorted far-to-near (draw order)."""
    shapes = []
    for i, o in enumerate(scene.objects):
        r = rng.child("shape", str(i), o.kind)
        parts = BUILDERS[o.kind](o, frame, scene, r)
        parts = [Part(p.name, p.sdf.astype(np.float32), p.role, p.detail, p.shade) for p in parts]
        cx, cy = frame.px(o.x, o.y)
        shapes.append(Shape(o, o.kind, parts, cx, cy, frame.size(o.size) / 2, infer_depth(o, scene),
                            region=o.kind in REGION_KINDS))
    order = sorted(range(len(shapes)), key=lambda i: (-shapes[i].depth, i))
    return [shapes[i] for i in order]
