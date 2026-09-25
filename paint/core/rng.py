"""Seeded, hierarchical random number generation.

Every random decision in a render flows from one integer seed. Sub-streams are
derived by *label*, not by call order, so adding a new consumer (say, a new
texture pass) never changes the numbers an existing consumer receives.

    rng = Rng(7)
    paper = rng.child("paper")        # independent of...
    strokes = rng.child("strokes")    # ...this, regardless of creation order
"""
from __future__ import annotations

import hashlib

import numpy as np


def _derive(seed: int, labels: tuple[str, ...]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(seed)).encode())
    for label in labels:
        h.update(b"\x00")
        h.update(str(label).encode())
    return int.from_bytes(h.digest(), "little")


class Rng:
    """A labelled wrapper around ``numpy.random.Generator`` (PCG64)."""

    def __init__(self, seed: int, *labels: str):
        self.seed = int(seed)
        self.labels = tuple(str(l) for l in labels)
        self.gen = np.random.Generator(np.random.PCG64(_derive(self.seed, self.labels)))

    def child(self, *labels: str) -> "Rng":
        return Rng(self.seed, *self.labels, *labels)

    # Thin delegates so call sites read naturally.
    def uniform(self, low=0.0, high=1.0, size=None):
        return self.gen.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.gen.normal(loc, scale, size)

    def integers(self, low, high=None, size=None):
        return self.gen.integers(low, high, size)

    def random(self, size=None):
        return self.gen.random(size)

    def choice(self, a, size=None, replace=True, p=None):
        return self.gen.choice(a, size=size, replace=replace, p=p)

    def permutation(self, x):
        return self.gen.permutation(x)

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed}, labels={'/'.join(self.labels) or '-'})"
