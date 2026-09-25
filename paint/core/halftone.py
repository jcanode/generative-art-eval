"""Halftone screens: turn a continuous tone into printable marks.

``tone`` arrays are in [0, 1], where 1 means full ink. All screens are
anti-aliased and fully vectorised.
"""
from __future__ import annotations

import numpy as np

from .geom import Frame


def _screen_coords(frame: Frame, cell: float, angle_deg: float):
    X, Y = frame.grid
    a = np.radians(angle_deg)
    u = (X * np.cos(a) + Y * np.sin(a)) / cell
    v = (-X * np.sin(a) + Y * np.cos(a)) / cell
    return u, v


def dots(tone, frame: Frame, cell: float = 8.0, angle_deg: float = 45.0, gain: float = 0.0):
    """Amplitude-modulated round dots (Euclidean spot) -> ink coverage in [0, 1].

    Dot area tracks ``tone``; dots merge into a checkerboard at 50 % and into
    holes above that, like a real screen. ``gain`` simulates dot gain.
    """
    u, v = _screen_coords(frame, cell, angle_deg)
    spot = 0.5 * (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v))  # in [-1, 1]
    t = np.clip(np.asarray(tone, dtype=np.float32) + gain, 0.0, 1.0)
    # Map tone to a threshold on the spot function so covered area ~= tone.
    thr = np.cos(np.pi * t)  # t=0 -> 1 (no ink), t=1 -> -1 (full)
    width = 2.2 * np.pi / cell  # approx |grad spot| in spot units per px
    cov = np.clip((spot - thr) / width + 0.5, 0.0, 1.0)
    # Very light tones: anti-aliasing would give every dot a visible haze; fade in instead.
    cov = cov * np.clip(t / 0.08, 0.0, 1.0)
    return np.where(t <= 0.001, 0.0, np.where(t >= 0.999, 1.0, cov)).astype(np.float32)


def lines(tone, frame: Frame, spacing: float = 6.0, angle_deg: float = 0.0, wobble=None):
    """Line screen: parallel lines whose thickness tracks tone. ``wobble`` (H,W) px offsets."""
    u, _ = _screen_coords(frame, spacing, angle_deg)
    if wobble is not None:
        u = u + wobble / spacing
    phase = np.abs((u % 1.0) - 0.5) * 2.0  # 0 at line centre, 1 between lines
    t = np.clip(np.asarray(tone, dtype=np.float32), 0.0, 1.0)
    edge = 2.0 / spacing  # phase units per pixel
    cov = np.clip((t - phase) / edge + 0.5, 0.0, 1.0)
    return np.where(t <= 0.001, 0.0, cov).astype(np.float32)


def stipple(tone, rng, shape):
    """Frequency-modulated dither (random stipple): coverage is binary noise with P(ink)=tone."""
    return (rng.random(shape) < np.clip(tone, 0, 1)).astype(np.float32)
