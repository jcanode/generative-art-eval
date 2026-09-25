"""Anti-aliased coverage masks.

Two routes to smooth edges:

* ``sdf_to_mask`` - analytic: a signed distance field gives sub-pixel coverage
  directly (a 1 px linear ramp across the boundary).
* ``supersample_mask`` - brute force: evaluate a shape function on an NxN
  subpixel grid and box-filter down. Used where the SDF is only approximate
  (smooth unions, thin details) or a shape is defined by a predicate.
"""
from __future__ import annotations

import numpy as np

from .geom import Frame


def sdf_to_mask(sdf, softness: float = 1.0):
    """Coverage in [0, 1] from a pixel-space SDF (negative inside)."""
    return np.clip(0.5 - sdf / max(softness, 1e-6), 0.0, 1.0).astype(np.float32)


def downsample(arr, factor: int):
    """Box-filter an (H*f, W*f[, C]) array down to (H, W[, C])."""
    if factor == 1:
        return arr
    h, w = arr.shape[0] // factor, arr.shape[1] // factor
    rest = arr.shape[2:]
    return arr[: h * factor, : w * factor].reshape(h, factor, w, factor, *rest).mean(axis=(1, 3))


def supersample_mask(shape_fn, frame: Frame, factor: int = 3, bbox=None):
    """Coverage of ``shape_fn(X, Y) -> sdf or bool`` via ``factor``x``factor`` supersampling.

    ``bbox = (x0, y0, x1, y1)`` in pixels restricts work to a window; outside it
    the mask is zero.
    """
    H, W = frame.shape
    x0, y0, x1, y1 = (0, 0, W, H) if bbox is None else _clip_bbox(bbox, W, H)
    out = np.zeros((H, W), dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    offs = (np.arange(factor, dtype=np.float32) + 0.5) / factor
    xs = (np.arange(x0, x1, dtype=np.float32)[:, None] + offs[None, :]).ravel()
    ys = (np.arange(y0, y1, dtype=np.float32)[:, None] + offs[None, :]).ravel()
    X, Y = np.meshgrid(xs, ys)
    v = shape_fn(X, Y)
    inside = (v if v.dtype == bool else v <= 0).astype(np.float32)
    out[y0:y1, x0:x1] = downsample(inside, factor)
    return out


def _clip_bbox(bbox, W, H):
    x0, y0, x1, y1 = bbox
    return (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
            min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1))))


def edge_band(sdf, width: float):
    """1 at the boundary falling to 0 ``width`` px inside; 0 outside. For pooled edges."""
    inside = np.clip(-sdf, 0, None)
    return (np.exp(-inside / max(width, 1e-6)) * (sdf <= 0.5)).astype(np.float32)
