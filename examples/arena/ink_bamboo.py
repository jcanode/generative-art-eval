"""Example hand-written contestant: `paint arena --models program:examples/arena/ink_bamboo.py,...`

A compact sumi-e bamboo painter in ~60 lines, the kind of program a model is asked to write.
"""
import numpy as np
from scipy import ndimage


def stroke(canvas, x0, y0, x1, y1, width, rng, load=1.0, taper=True):
    """A tapered, dry-brushed stroke from (x0, y0) to (x1, y1), vectorised over its bounding box."""
    h, w = canvas.shape
    pad = int(width) + 2
    xa, xb = int(max(0, min(x0, x1) - pad)), int(min(w, max(x0, x1) + pad))
    ya, yb = int(max(0, min(y0, y1) - pad)), int(min(h, max(y0, y1) + pad))
    if xa >= xb or ya >= yb:
        return
    Y, X = np.mgrid[ya:yb, xa:xb].astype(float)
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
    d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
    half = width / 2 * (np.sin(np.pi * np.clip(t * 0.9 + 0.08, 0, 1)) ** 0.5 if taper else 1.0 - 0.15 * np.abs(2 * t - 1) ** 6)
    across = np.clip(d / np.maximum(half, 1e-3), 0, 1)
    bristles = 0.8 + 0.2 * np.sin(across * 11 + rng.uniform(0, 6))
    ink = np.clip((load - 0.7 * t) * 1.6, 0, 1) * bristles
    dry = rng.random(ink.shape) < (0.25 + 0.75 * ink)
    cover = np.clip(half - d + 0.5, 0, 1) * np.where(dry, ink, ink * 0.2)
    canvas[ya:yb, xa:xb] = 1 - (1 - canvas[ya:yb, xa:xb]) * (1 - cover)


def paint(width, height, seed):
    rng = np.random.default_rng(seed)
    ink = np.zeros((height, width))
    s = height
    for k, bx in enumerate((0.25, 0.4, 0.52)):
        x = bx * width
        lean = rng.uniform(-0.04, 0.04) * width
        top = s * rng.uniform(0.02, 0.12)
        segs = 7
        for i in range(segs):  # stalk segments with gaps at the nodes
            t0, t1 = i / segs, (i + 1) / segs - 0.012
            y0, y1 = s - t0 * (s - top), s - t1 * (s - top)
            stroke(ink, x + lean * t0, y0, x + lean * t1, y1, s * 0.035, rng, load=0.9 - 0.25 * k, taper=False)
            stroke(ink, x + lean * t1 - s * 0.02, y1, x + lean * t1 + s * 0.02, y1 - 2, s * 0.008, rng, 1.0)
        for j in range(rng.integers(1, 3)):  # leaf sprays
            t = rng.uniform(0.35, 0.8)
            ax, ay = x + lean * t, s - t * (s - top)
            side = rng.choice([-1, 1])
            for m in range(4):
                ang = side * rng.uniform(0.2, 1.1) + (np.pi if side < 0 else 0) + 0.4 * side
                ln = s * rng.uniform(0.1, 0.17)
                stroke(ink, ax, ay, ax + np.cos(ang) * ln, ay + abs(np.sin(ang)) * ln * 0.6 + ln * 0.2, s * 0.03, rng, 1.0)
    # the bird: body, head, beak, tail as a few loaded dabs on a twig
    bx, by = 0.68 * width, 0.42 * height
    stroke(ink, 0.54 * width, by + s * 0.05, 0.8 * width, by + s * 0.04, s * 0.008, rng, 1.0)
    Y, X = np.mgrid[0:height, 0:width]
    body = ((X - bx) / (s * 0.07)) ** 2 + ((Y - by) / (s * 0.04)) ** 2 < 1
    head = (X - (bx + s * 0.065)) ** 2 + (Y - (by - s * 0.035)) ** 2 < (s * 0.028) ** 2
    ink[body] = np.maximum(ink[body], 0.55)
    ink[head] = np.maximum(ink[head], 0.9)
    stroke(ink, bx - s * 0.02, by - s * 0.015, bx + s * 0.05, by - s * 0.01, s * 0.035, rng, 1.0)  # wing
    stroke(ink, bx - s * 0.06, by + s * 0.005, bx - s * 0.14, by + s * 0.04, s * 0.025, rng, 1.0)  # tail
    stroke(ink, bx + s * 0.09, by - s * 0.04, bx + s * 0.115, by - s * 0.033, s * 0.012, rng, 1.0)  # beak
    eye = (X - (bx + s * 0.075)) ** 2 + (Y - (by - s * 0.042)) ** 2 < (s * 0.006) ** 2
    ink[eye] = 0.0
    wash = ndimage.gaussian_filter(ink, 3) * 0.25
    ink = np.clip(ink + wash, 0, 1)
    paper = np.array([0.95, 0.92, 0.85]) * (0.97 + 0.03 * ndimage.gaussian_filter(rng.random((height, width)), 1.2)[..., None])
    img = paper * (1 - ink[..., None] * np.array([0.92, 0.92, 0.9]))
    seal = (np.abs(Y - 0.88 * height) < 0.022 * s) & (np.abs(X - 0.85 * width) < 0.022 * s)
    img[seal] = [0.72, 0.18, 0.14]
    return np.clip(img, 0, 1)
