"""Pairwise preference with position-swap consistency, and an Elo table across versions.

Each pair is judged twice: once in a random order, once swapped. Only a
verdict that survives the swap counts. An inconsistent pair (the judge prefers
whichever image is in the same slot both times) is recorded but does not move
Elo.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

K_FACTOR = 24.0
BASE = 1500.0


def compare_pair(judge, a, b, rubric: dict | None = None, seed: int = 0) -> dict:
    """Judge A vs B twice with positions swapped. Returns the consistent verdict or 'inconsistent'."""
    rnd = random.Random(f"{seed}|{a}|{b}")
    first_a = rnd.random() < 0.5
    order1 = (a, b) if first_a else (b, a)
    v1 = judge.compare(order1[0], order1[1], rubric)
    v2 = judge.compare(order1[1], order1[0], rubric)

    def winner(v, order):
        w = v.get("winner")
        if w == "tie":
            return "tie"
        pick = order[0] if w == "1" else order[1]
        return "A" if pick == a else "B"

    w1 = winner(v1, order1)
    w2 = winner(v2, (order1[1], order1[0]))
    consistent = w1 == w2
    return {
        "a": str(a), "b": str(b),
        "verdict": w1 if consistent else "inconsistent",
        "consistent": consistent,
        "runs": [
            {"order": ["A", "B"] if first_a else ["B", "A"], "winner": w1, "reason": v1.get("reason", "")},
            {"order": ["B", "A"] if first_a else ["A", "B"], "winner": w2, "reason": v2.get("reason", "")},
        ],
    }


class EloTable:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.ratings: dict[str, float] = {}
        self.games: dict[str, dict] = {}
        self.history: list[dict] = []
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.ratings = d.get("ratings", {})
            self.games = d.get("games", {})
            self.history = d.get("history", [])

    def _g(self, p):
        return self.games.setdefault(p, {"w": 0, "l": 0, "t": 0, "inconsistent": 0})

    def expected(self, a, b):
        ra, rb = self.ratings.get(a, BASE), self.ratings.get(b, BASE)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def record(self, a: str, b: str, verdict: str, context: str = ""):
        """verdict: 'A' (a wins), 'B', 'tie', or 'inconsistent' (logged, no rating change)."""
        self.ratings.setdefault(a, BASE)
        self.ratings.setdefault(b, BASE)
        self.history.append({"a": a, "b": b, "verdict": verdict, "context": context})
        if verdict == "inconsistent":
            self._g(a)["inconsistent"] += 1
            self._g(b)["inconsistent"] += 1
            return
        sa = {"A": 1.0, "B": 0.0, "tie": 0.5}[verdict]
        ea = self.expected(a, b)
        self.ratings[a] += K_FACTOR * (sa - ea)
        self.ratings[b] += K_FACTOR * ((1 - sa) - (1 - ea))
        if verdict == "tie":
            self._g(a)["t"] += 1
            self._g(b)["t"] += 1
        else:
            win, lose = (a, b) if verdict == "A" else (b, a)
            self._g(win)["w"] += 1
            self._g(lose)["l"] += 1

    def table(self):
        rows = []
        for p, r in sorted(self.ratings.items(), key=lambda kv: -kv[1]):
            g = self._g(p)
            rows.append({"player": p, "elo": round(r, 1), **g})
        return rows

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"ratings": self.ratings, "games": self.games,
                                             "history": self.history[-5000:]}, indent=1))
