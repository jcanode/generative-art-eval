"""Canvas frames: normalised scene coordinates <-> pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Frame:
    """A pixel canvas. Scene units: x, y in [0, 1]; sizes are fractions of height."""

    width: int
    height: int

    @cached_property
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre coordinates ``(X, Y)``, each ``(H, W)`` float32."""
        ys = np.arange(self.height, dtype=np.float32) + 0.5
        xs = np.arange(self.width, dtype=np.float32) + 0.5
        X, Y = np.meshgrid(xs, ys)
        return X, Y

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def unit(self) -> float:
        """Pixels per scene size unit (canvas height)."""
        return float(self.height)

    @property
    def scale(self) -> float:
        """Resolution scale relative to a 1024 px reference height, for sizing marks."""
        return self.height / 1024.0

    def px(self, x: float, y: float) -> tuple[float, float]:
        return x * self.width, y * self.height

    def size(self, s: float) -> float:
        return s * self.unit

    def scaled(self, factor: int) -> "Frame":
        return Frame(self.width * factor, self.height * factor)


def rotate(X, Y, cx, cy, angle):
    """Rotate coordinates about (cx, cy) by -angle, i.e. into a shape's local frame."""
    c, s = np.cos(angle), np.sin(angle)
    dx, dy = X - cx, Y - cy
    return c * dx + s * dy, -s * dx + c * dy
