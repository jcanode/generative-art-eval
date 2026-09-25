"""Blind vision judges.

The judge sees ONLY pixels: images are decoded, downscaled and re-encoded
without metadata (our PNGs carry style/params in text chunks), sent without
file names, and never accompanied by the scene spec, the code, or the
generator's own critique. The only text context it gets is a task prompt and,
for style scoring, the rubric.

Backends:
    ClaudeJudge  - Anthropic Messages API with vision + structured outputs.
    MockJudge    - offline, deterministic, pixel-statistics stand-in. It exists
                   so the pipeline runs without an API key; its "opinions" are
                   placeholders and the report labels them as such.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from . import metrics

PROMPT_VERSION = "v1"
DEFAULT_MODEL = os.environ.get("PAINT_JUDGE_MODEL", "claude-opus-5")
RUBRIC_DIR = Path(__file__).resolve().parents[2] / "evals" / "rubrics"
MAX_SIDE = 1024


class JudgeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

def load_rubric(style: str, rubric_dir: Path | None = None) -> dict:
    """Parse evals/rubrics/<style>.md into {title, intro, criteria: [{name, definition}], text, hash}."""
    path = (rubric_dir or RUBRIC_DIR) / f"{style}.md"
    text = path.read_text()
    title, intro, criteria, cur = style, [], [], None
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            cur = {"name": line[3:].strip(), "definition": ""}
            criteria.append(cur)
        elif cur is not None:
            cur["definition"] = (cur["definition"] + " " + line.strip()).strip()
        elif line.strip():
            intro.append(line.strip())
    if not criteria:
        raise JudgeError(f"rubric {path} has no '## ' criteria")
    return {"style": style, "title": title, "intro": " ".join(intro), "criteria": criteria,
            "text": text, "hash": hashlib.sha1(text.encode()).hexdigest()[:10]}


# ---------------------------------------------------------------------------
# Image preparation (strips everything but pixels)
# ---------------------------------------------------------------------------

def blind_png(path_or_array) -> tuple[bytes, str]:
    """Pixels only: decode, downscale, re-encode with no metadata. Returns (png_bytes, pixel_hash)."""
    if isinstance(path_or_array, np.ndarray):
        im = Image.fromarray((np.clip(path_or_array, 0, 1) * 255 + 0.5).astype(np.uint8))
    else:
        im = Image.open(path_or_array).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = buf.getvalue()
    return data, hashlib.sha256(np.asarray(im).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Response cache (keyed by pixels + task, so unchanged images are never re-judged)
# ---------------------------------------------------------------------------

class JudgeCache:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.data = {}
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def get(self, key):
        with self.lock:
            return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data))
                tmp.replace(self.path)


def _key(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Prompts and schemas
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a careful, impartial art critic and visual analyst. You judge only what is visible in the "
    "image(s) you are shown. You have no other information about how they were made. Be literal about "
    "what you see; do not guess at what the maker intended."
)

DESCRIBE_PROMPT = (
    "List every distinct thing you can see depicted in this image: objects, living things, landscape "
    "features, weather. Use plain common nouns (e.g. 'boat', 'mountain', 'teapot'), one entry per kind "
    "of thing, and say how confident you are. If a shape is ambiguous, name what it most looks like, "
    "not what it might be meant to be. Then give a one-sentence summary, and name the medium or "
    "technique the image most resembles."
)

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["name", "confidence"], "additionalProperties": False}},
        "summary": {"type": "string"},
        "medium": {"type": "string"},
    },
    "required": ["objects", "summary", "medium"],
    "additionalProperties": False,
}

SCORE_ENUM = {"type": "integer", "enum": [1, 2, 3, 4, 5]}


def style_prompt(rubric: dict) -> str:
    lines = [f"Score this image against the rubric below. {rubric['intro']}", "",
             "For each criterion give an integer score from 1 to 5 and a one-line reason that points at "
             "something specific in the image. Judge what is there, not what might have been intended. "
             "Use the full scale.", ""]
    for c in rubric["criteria"]:
        lines.append(f"- {c['name']}: {c['definition']}")
    return "\n".join(lines)


def style_schema(rubric: dict) -> dict:
    names = [c["name"] for c in rubric["criteria"]]
    return {
        "type": "object",
        "properties": {"scores": {"type": "array", "items": {
            "type": "object",
            "properties": {"criterion": {"type": "string", "enum": names}, "reason": {"type": "string"},
                           "score": SCORE_ENUM},
            "required": ["criterion", "reason", "score"], "additionalProperties": False}}},
        "required": ["scores"],
        "additionalProperties": False,
    }


PAIR_PROMPT = (
    "You are shown two images, Image 1 and Image 2. Decide which is the better piece of artwork{focus}. "
    "Consider how clearly the subjects read, composition, and craft within its medium. Position is "
    "irrelevant; do not prefer an image because it came first or second. Explain briefly, then answer "
    "'1', '2', or 'tie' (only if they are genuinely equal)."
)

PAIR_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "winner": {"type": "string", "enum": ["1", "2", "tie"]}},
    "required": ["reason", "winner"],
    "additionalProperties": False,
}

OVERALL_PROMPT = (
    "Rate this image as a finished artwork on a 1-5 scale: 1 = broken or unreadable, 2 = weak, "
    "3 = competent but unremarkable, 4 = good, 5 = excellent. Consider subject clarity, composition, "
    "and craft within its medium. Give a one-line reason first."
)

OVERALL_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "score": SCORE_ENUM},
    "required": ["reason", "score"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

class Judge:
    name = "judge"
    is_mock = False

    def __init__(self, cache: JudgeCache | None = None, model: str | None = None):
        self.cache = cache or JudgeCache(None)
        self.model = model or DEFAULT_MODEL

    # Subclasses implement _ask(images, prompt, schema) -> dict
    def _ask(self, images: list[bytes], prompt: str, schema: dict) -> dict:
        raise NotImplementedError

    def _cached(self, task, hashes, extra, images, prompt, schema):
        key = _key(self.name, self.model, PROMPT_VERSION, task, *hashes, extra)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = self._ask(images, prompt, schema)
        self.cache.put(key, out)
        return out

    def describe(self, image) -> dict:
        png, h = blind_png(image)
        return self._cached("describe", [h], "", [png], DESCRIBE_PROMPT, DESCRIBE_SCHEMA)

    def score_style(self, image, rubric: dict) -> dict:
        png, h = blind_png(image)
        out = self._cached("style", [h], rubric["hash"], [png], style_prompt(rubric), style_schema(rubric))
        by_name = {s["criterion"]: s for s in out.get("scores", [])}
        scores = []
        for c in rubric["criteria"]:
            s = by_name.get(c["name"])
            scores.append({"criterion": c["name"], "score": int(s["score"]) if s else None,
                           "reason": s["reason"] if s else "(judge omitted this criterion)"})
        valid = [s["score"] for s in scores if s["score"] is not None]
        return {"scores": scores, "mean": round(float(np.mean(valid)), 3) if valid else None,
                "rubric_hash": rubric["hash"]}

    def compare(self, image_1, image_2, rubric: dict | None = None) -> dict:
        p1, h1 = blind_png(image_1)
        p2, h2 = blind_png(image_2)
        focus = f" as {rubric['title'].lower()}" if rubric else ""
        prompt = PAIR_PROMPT.format(focus=focus)
        if rubric:
            prompt += "\n\nWhat the tradition values:\n" + "\n".join(f"- {c['name']}: {c['definition']}" for c in rubric["criteria"])
        return self._cached("pair", [h1, h2], rubric["hash"] if rubric else "", [p1, p2], prompt, PAIR_SCHEMA)

    def rate_overall(self, image) -> dict:
        png, h = blind_png(image)
        return self._cached("overall", [h], "", [png], OVERALL_PROMPT, OVERALL_SCHEMA)


class ClaudeJudge(Judge):
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None, cache: JudgeCache | None = None,
                 fallbacks: bool | None = None):
        super().__init__(cache, model)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise JudgeError("the 'anthropic' package is required for the Claude judge: pip install anthropic") from exc
        self._anthropic = anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        # Server-side refusal fallbacks (on by default; PAINT_JUDGE_FALLBACKS=0 disables).
        self.fallbacks = (os.environ.get("PAINT_JUDGE_FALLBACKS", "1") != "0") if fallbacks is None else fallbacks

    def _ask(self, images, prompt, schema):
        content = []
        for i, png in enumerate(images):
            if len(images) > 1:
                content.append({"type": "text", "text": f"Image {i + 1}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.standard_b64encode(png).decode()}})
        content.append({"type": "text", "text": prompt})
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if self.fallbacks:
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        a = self._anthropic
        try:
            resp = self.client.messages.create(**kwargs)
        except a.AuthenticationError as exc:
            raise JudgeError(f"API key rejected: {exc}") from exc
        except a.BadRequestError as exc:
            if self.fallbacks and "fallback" in str(exc).lower():
                self.fallbacks = False  # account/model without fallback support: retry once without it
                return self._ask(images, prompt, schema)
            raise JudgeError(f"bad request: {exc}") from exc
        except a.RateLimitError as exc:
            raise JudgeError(f"rate limited (SDK retries exhausted): {exc}") from exc
        except a.APIStatusError as exc:
            raise JudgeError(f"API error {exc.status_code}: {exc}") from exc
        except a.APIConnectionError as exc:
            raise JudgeError(f"could not reach the API: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise JudgeError("judge declined to answer (refusal)")
        if resp.stop_reason == "max_tokens":
            raise JudgeError("judge response truncated (max_tokens)")
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise JudgeError("judge returned no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise JudgeError(f"judge returned non-JSON: {text[:200]}")
            return json.loads(m.group(0))


class MockJudge(Judge):
    """Offline stand-in. Deterministic, derived from pixel statistics, and NOT a real opinion.

    It cannot see subjects, so ``describe`` returns no objects. Style and overall
    scores are coarse functions of contrast/texture/colour; pairwise prefers the
    image with more tonal structure. Use it to exercise the pipeline, never to
    make decisions.
    """

    name = "mock"
    is_mock = True

    def __init__(self, cache: JudgeCache | None = None, model: str | None = None):
        super().__init__(cache, "mock")

    @staticmethod
    def _stats(png: bytes) -> dict:
        img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0
        return metrics.summary(img)

    @staticmethod
    def _quality(m: dict) -> float:
        # Reward mid-high contrast and some texture; penalise extremes. Arbitrary but monotone-ish.
        c = min(m["lum_std"] / 0.2, 1.0)
        t = 1.0 - abs(m["edge_density"] - 0.15) / 0.3
        return float(np.clip(0.5 * c + 0.5 * t, 0, 1))

    def _ask(self, images, prompt, schema):
        stats = [self._stats(p) for p in images]
        props = schema["properties"]
        if "objects" in props:
            return {"objects": [], "summary": "(mock judge cannot see subjects)", "medium": "unknown (mock)"}
        if "winner" in props:
            q1, q2 = (self._quality(s) for s in stats)
            w = "tie" if abs(q1 - q2) < 0.01 else ("1" if q1 > q2 else "2")
            return {"reason": f"mock: tonal-structure score {q1:.2f} vs {q2:.2f}", "winner": w}
        if "scores" in props:
            names = props["scores"]["items"]["properties"]["criterion"]["enum"]
            q = self._quality(stats[0])
            out = []
            for i, n in enumerate(names):
                h = int(hashlib.md5(n.encode()).hexdigest(), 16) % 3 - 1  # stable per-criterion offset
                out.append({"criterion": n, "score": int(np.clip(round(1 + 4 * q) + (h if i % 2 else 0), 1, 5)),
                            "reason": "mock score from pixel statistics"})
            return {"scores": out}
        q = self._quality(stats[0])
        return {"reason": f"mock: tonal-structure score {q:.2f}", "score": int(np.clip(round(1 + 4 * q), 1, 5))}


def make_judge(kind: str = "auto", model: str | None = None, api_key: str | None = None,
               cache_path: str | Path | None = "runs/judge_cache.json") -> Judge:
    cache = JudgeCache(Path(cache_path)) if cache_path else JudgeCache(None)
    if kind == "auto":
        kind = "claude" if (api_key or os.environ.get("ANTHROPIC_API_KEY")) else "mock"
    if kind == "claude":
        return ClaudeJudge(api_key=api_key, model=model, cache=cache)
    if kind == "mock":
        return MockJudge(cache=cache)
    raise JudgeError(f"unknown judge {kind!r}")
