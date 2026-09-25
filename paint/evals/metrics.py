"""Pixel measurements shared by technical checks, the heuristic critic, and the report."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import ndimage

from ..core.io import load_rgb


def pixel_hash(path_or_img) -> str:
    img = load_rgb(path_or_img) if not isinstance(path_or_img, np.ndarray) else path_or_img
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return hashlib.sha256(u8.tobytes() + str(u8.shape).encode()).hexdigest()[:16]


def luminance(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def saturation(img):
    mx, mn = img.max(axis=2), img.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def edge_density(img, thresh: float = 0.08):
    """Fraction of pixels with a strong local luminance gradient (Sobel)."""
    L = luminance(img)
    g = np.hypot(ndimage.sobel(L, axis=0), ndimage.sobel(L, axis=1)) / 4.0
    return float((g > thresh).mean())


def color_clusters(img, bits: int = 4, min_frac: float = 0.005):
    """Number of quantised colours that each cover at least ``min_frac`` of the image."""
    q = (np.clip(img, 0, 1) * (2**bits - 1) + 0.5).astype(np.int32)
    code = (q[..., 0] << (2 * bits)) | (q[..., 1] << bits) | q[..., 2]
    counts = np.bincount(code.ravel())
    return int((counts >= min_frac * code.size).sum())


def paper_color(img):
    """Estimate the substrate colour: the most common colour among the brightest 30 % of pixels."""
    L = luminance(img)
    bright = img[L >= np.percentile(L, 70)]
    q = (bright * 31 + 0.5).astype(np.int32)
    code = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    mode = np.bincount(code).argmax()
    return np.array([(mode >> 10) & 31, (mode >> 5) & 31, mode & 31], dtype=np.float32) / 31.0


def empty_fraction(img, tol: float = 0.07):
    """Fraction of pixels within ``tol`` (RGB distance) of the estimated paper colour."""
    p = paper_color(img)
    d = np.sqrt(((img - p) ** 2).sum(axis=2) / 3.0)
    return float((d < tol).mean())


def colorfulness(img):
    """Hasler-Suesstrunk colourfulness, roughly 0 (grey) .. 1+ (vivid)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def summary(img) -> dict:
    L = luminance(img)
    S = saturation(img)
    return {
        "lum_mean": round(float(L.mean()), 4),
        "lum_std": round(float(L.std()), 4),
        "lum_p5": round(float(np.percentile(L, 5)), 4),
        "lum_p95": round(float(np.percentile(L, 95)), 4),
        "sat_mean": round(float(S.mean()), 4),
        "edge_density": round(edge_density(img), 4),
        "color_clusters": color_clusters(img),
        "empty_fraction": round(empty_fraction(img), 4),
        "colorfulness": round(colorfulness(img), 4),
    }


def mean_abs_diff(a, b) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(np.abs(a - b).mean())
