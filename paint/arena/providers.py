"""Code-writing contestants.

A Coder holds one conversation: ``start(task_text)`` returns the first program and
``revise(content_blocks)`` returns the next one, where the blocks carry the
rendered image (or the error) back to the model.

* AnthropicCoder - any Claude model via the Messages API (streaming, adaptive thinking where supported,
  prompt caching on the growing conversation). The conversation is append-only: assistant turns are sent
  back exactly as received.
* NaiveCoder     - an offline, deterministic baseline that writes a flat-shapes program from the scene
  spec. It never improves. It keeps the arena runnable without a key and marks the floor.
* ScriptedCoder  - replays given programs, for tests.

To add another vendor, implement the same two methods; nothing else in the arena changes.
"""
from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass, field

from .brief import SYSTEM

# Per-model API settings and prices (USD per million tokens, input/output). Keep in sync with the
# model table in the Claude API docs.
MODELS = {
    "claude-opus-5": {"price": (5.0, 25.0), "thinking": {"type": "adaptive"}, "effort": "high"},
    "claude-sonnet-5": {"price": (2.0, 10.0), "thinking": {"type": "adaptive"}, "effort": "high"},
    "claude-haiku-4-5": {"price": (1.0, 5.0), "thinking": None, "effort": None},
    "claude-fable-5-1": {"price": (10.0, 50.0), "thinking": None, "effort": "high"},   # thinking is always on
    "claude-opus-5-5": {"price": (4.0, 20.0), "thinking": None, "effort": "high"},     # thinking always on; default effort is medium
    "claude-opus-4-8": {"price": (5.0, 25.0), "thinking": {"type": "adaptive"}, "effort": "high"},
}
DEFAULT_LINEUP = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


@dataclass
class Reply:
    code: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str = ""
    extras: dict = field(default_factory=dict)


def extract_code(text: str) -> str:
    blocks = CODE_RE.findall(text)
    with_paint = [b for b in blocks if "def paint" in b]
    if with_paint:
        return with_paint[-1]
    if blocks:
        return max(blocks, key=len)
    return text if "def paint" in text else ""


def image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode()}}


class AnthropicCoder:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 32000, effort: str | None = None):
        import anthropic

        self.model = model
        self.name = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key, timeout=600.0) if key else anthropic.Anthropic(timeout=600.0)
        self.cfg = MODELS.get(model, {"price": (5.0, 25.0), "thinking": {"type": "adaptive"}, "effort": "high"})
        if effort:
            self.cfg = {**self.cfg, "effort": effort}
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def _call(self) -> Reply:
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM, messages=self.messages,
                  cache_control={"type": "ephemeral"})
        if self.cfg.get("thinking"):
            kw["thinking"] = self.cfg["thinking"]
        if self.cfg.get("effort"):
            kw["output_config"] = {"effort": self.cfg["effort"]}
        t0 = time.perf_counter()
        with self.client.messages.stream(**kw) as stream:
            msg = stream.get_final_message()
        # Append-only history: send the assistant turn back exactly as received (thinking blocks included).
        self.messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason == "refusal":
            raise RuntimeError("the model declined this request (refusal)")
        text = "\n".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        return Reply(extract_code(text), text, u.input_tokens + (getattr(u, "cache_creation_input_tokens", 0) or 0),
                     u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0, time.perf_counter() - t0,
                     msg.stop_reason or "")

    def start(self, task_text: str) -> Reply:
        self.messages = [{"role": "user", "content": task_text}]
        return self._call()

    def revise(self, blocks: list[dict]) -> Reply:
        self.messages.append({"role": "user", "content": blocks})
        return self._call()

    def cost(self, r: Reply) -> float:
        pin, pout = self.cfg["price"]
        # Cache reads bill at ~10% of the input price.
        return (r.input_tokens * pin + r.cache_read_tokens * pin * 0.1 + r.output_tokens * pout) / 1e6


class ScriptedCoder:
    """Replays programs in order (for tests and demos)."""

    def __init__(self, programs: list[str], name: str = "scripted"):
        self.programs = list(programs)
        self.name = name
        self.i = 0

    def _next(self) -> Reply:
        code = self.programs[min(self.i, len(self.programs) - 1)]
        self.i += 1
        return Reply(code, f"Score: {min(5 + self.i, 9)}/10\n```python\n{code}\n```")

    def start(self, task_text):
        return self._next()

    def revise(self, blocks):
        return self._next()

    def cost(self, r):
        return 0.0


class NaiveCoder:
    """Offline floor: flat shapes at the right places in the palette colours. Never revises."""

    name = "baseline:naive"

    def __init__(self, scene: dict):
        self.scene = scene

    def _program(self) -> str:
        s = self.scene
        return NAIVE_TEMPLATE.format(objects=repr(s.get("objects", [])), hints=repr((s.get("palette") or {}).get("hints") or
                                     ["#f2c14e", "#e8643c", "#2a8c88", "#1d2d4a"]), horizon=float(s.get("horizon", 0.62)))

    def start(self, task_text):
        code = self._program()
        return Reply(code, "Flat-shape baseline. Score: 3/10\n```python\n" + code + "\n```")

    def revise(self, blocks):
        return self.start("")

    def cost(self, r):
        return 0.0


NAIVE_TEMPLATE = '''
import numpy as np

OBJECTS = {objects}
HINTS = {hints}
HORIZON = {horizon}


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def paint(width, height, seed):
    rng = np.random.default_rng(seed)
    cols = [hexrgb(h) for h in HINTS]
    lum = [c @ np.array([0.3, 0.6, 0.1]) for c in cols]
    light, dark = cols[int(np.argmax(lum))], cols[int(np.argmin(lum))]
    mid = cols[len(cols) // 2]
    Y, X = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.ones((height, width, 3)) * light
    hz = HORIZON * height
    img[Y >= hz] = mid
    for i, o in enumerate(OBJECTS):
        k = o["kind"]
        if k in ("sea", "field", "table", "river", "hills", "stars", "rain"):
            continue
        cx, cy = o.get("x", 0.5) * width, o.get("y", 0.5) * height
        s = o.get("size", 0.2) * height
        c = cols[(i + 1) % len(cols)]
        if k in ("sun", "moon", "cloud", "fruit"):
            m = (X - cx) ** 2 + (Y - cy + (s / 2 if k == "fruit" else 0)) ** 2 < (s / 2) ** 2
        elif k in ("mountain",):
            m = (Y > cy - s / 2) & (Y < cy + s / 2) & (np.abs(X - cx) < (Y - (cy - s / 2)))
        elif k == "bird":
            m = (np.abs(Y - cy) < 2) & (np.abs(X - cx) < s / 2)
        else:
            m = (np.abs(X - cx) < s * 0.25) & (Y < cy) & (Y > cy - s)
        img[m] = dark if k in ("bird", "boat", "tree", "pine", "cypress") else c
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0, 1)
'''


class FileCoder(ScriptedCoder):
    """A hand-written program entered as a contestant: ``program:path/to/file.py``."""

    def __init__(self, path: str):
        from pathlib import Path

        super().__init__([Path(path).read_text()], name=f"program:{Path(path).stem}")


def make_coder(name: str, api_key: str | None = None, scene: dict | None = None, effort: str | None = None):
    if name == "baseline:naive":
        return NaiveCoder(scene or {})
    if name.startswith("program:"):
        return FileCoder(name.split(":", 1)[1])
    if name.startswith("claude-"):
        return AnthropicCoder(name, api_key=api_key, effort=effort)
    raise ValueError(f"unknown contestant {name!r}; use a claude-* model id, baseline:naive, or program:<file.py>")
