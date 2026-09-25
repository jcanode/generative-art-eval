"""Critics for the render -> look -> critique -> revise loop.

A critic looks at the rendered PNG and returns a ``Critique``: what reads
wrong, what is flat, what is cluttered, what is unrecognisable, plus concrete
revisions (style-knob changes and object edits) and a 1-10 self-score.

* ``ClaudeCritic``    - a vision model looks at the image (API key required).
* ``ManualCritic``    - step mode for an agent (e.g. Claude Code) or a human: the loop stops
                        after each render and waits for ``vN/critique.json`` to be written by
                        someone who has actually opened the PNG. Re-run the command to continue.
* ``HeuristicCritic`` - offline fallback that measures pixels. It is NOT a visual critique and
                        says so in every critique it writes.

Unlike the eval judge, the critic is part of the generator: it may see the
scene spec and the knobs. The eval judge never sees any of this.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..evals import metrics
from ..core.io import load_rgb

HEADINGS = ("reads_wrong", "flat", "cluttered", "unrecognizable", "technical", "strengths")


@dataclass
class Critique:
    critic: str
    looked_at: str                      # path + pixel hash of the image that was examined
    summary: str = ""
    reads_wrong: list[str] = field(default_factory=list)
    flat: list[str] = field(default_factory=list)
    cluttered: list[str] = field(default_factory=list)
    unrecognizable: list[str] = field(default_factory=list)
    technical: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    score: float = 5.0                  # critic's own 1-10 rating of this version
    adjustments: dict = field(default_factory=dict)       # knob -> new value
    object_edits: list[dict] = field(default_factory=list)  # {index, size_scale?, dx?, dy?, note?}
    code_changes: str = ""              # free text: revisions made in code rather than knobs
    stop: bool = False                  # the critic is satisfied

    def to_markdown(self, version: int) -> str:
        lines = [f"# v{version} critique ({self.critic})", "", f"_Looked at: {self.looked_at}_", "",
                 f"**Score:** {self.score}/10", "", self.summary, ""]
        for h in HEADINGS:
            items = getattr(self, h)
            if items:
                lines.append(f"**{h.replace('_', ' ').capitalize()}**")
                lines += [f"- {x}" for x in items]
                lines.append("")
        if self.adjustments:
            lines.append("**Revisions (knobs):** " + ", ".join(f"`{k}` -> {v}" for k, v in self.adjustments.items()))
        if self.object_edits:
            lines.append("**Revisions (objects):** " + "; ".join(json.dumps(e) for e in self.object_edits))
        if self.code_changes:
            lines.append(f"**Code changes:** {self.code_changes}")
        if self.stop:
            lines.append("\n**Critic is satisfied; loop stops here.**")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, d: dict, critic: str, looked_at: str) -> "Critique":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k not in ("critic", "looked_at")}
        c = cls(critic=d.get("critic", critic), looked_at=d.get("looked_at", looked_at), **known)
        c.score = float(c.score)
        return c

    def to_dict(self):
        return asdict(self)


CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        **{h: {"type": "array", "items": {"type": "string"}} for h in HEADINGS},
        "score": {"type": "integer", "enum": list(range(1, 11))},
        "adjustments": {"type": "array", "items": {
            "type": "object",
            "properties": {"knob": {"type": "string"}, "value": {"type": "number"}, "why": {"type": "string"}},
            "required": ["knob", "value", "why"], "additionalProperties": False}},
        "object_edits": {"type": "array", "items": {
            "type": "object",
            "properties": {"index": {"type": "integer"}, "size_scale": {"type": "number"}, "dx": {"type": "number"},
                           "dy": {"type": "number"}, "note": {"type": "string"}},
            "required": ["index", "size_scale", "dx", "dy", "note"], "additionalProperties": False}},
        "stop": {"type": "boolean"},
    },
    "required": ["summary", *HEADINGS, "score", "adjustments", "object_edits", "stop"],
    "additionalProperties": False,
}


def _style_guide_section(style: str) -> str:
    guide = (Path(__file__).resolve().parents[2] / "docs" / "STYLE_GUIDE.md")
    if not guide.exists():
        return ""
    text = guide.read_text()
    key = {"screenprint": "## Screenprint", "sumie": "## Sumi-e", "impasto": "## Impasto"}.get(style)
    if not key or key not in text:
        return ""
    start = text.index(key)
    end = text.find("\n## ", start + 4)
    return text[start: end if end > 0 else None]


def _looked(png: Path) -> str:
    return f"{png.name} (pixels {metrics.pixel_hash(png)})"


class ClaudeCritic:
    name = "claude"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key, timeout=180.0) if key else anthropic.Anthropic(timeout=180.0)
        self.model = model or os.environ.get("PAINT_CRITIC_MODEL", "claude-opus-5")

    def critique(self, png: Path, scene, style_mod, params: dict, checks: dict, history: list[dict]) -> Critique:
        knobs = "\n".join(f"- {k}: current {params.get(k)!r}, range [{lo}, {hi}] - {desc}"
                          for k, (lo, hi, desc) in style_mod.KNOBS.items())
        objects = "\n".join(f"  [{i}] {o.kind} at ({o.x:.2f}, {o.y:.2f}) size {o.size:.2f}" for i, o in enumerate(scene.objects))
        failed = [f"{c['name']}: {c['detail']}" for c in checks.get("checks", []) if not c["passed"]]
        prev = "\n".join(f"v{h['version']}: score {h['score']} - {h['summary']}" for h in history) or "(first version)"
        prompt = f"""You are the critic inside a procedural painting studio. The image was rendered by code in the
'{style_mod.NAME}' style from this scene:
title: {scene.title}
objects:
{objects}

Style guide:
{_style_guide_section(style_mod.NAME)}

Look hard at the image. Critique it under these headings (short concrete bullets, empty list if nothing):
reads_wrong (an object that reads as something else), flat (lacks tonal structure / depth), cluttered
(competing marks, noise), unrecognizable (objects a stranger could not name), technical (artefacts, seams,
cropping), strengths. Score it 1-10 as a finished piece in this tradition.

Then propose revisions using ONLY these knobs (value must stay in range; change at most 4):
{knobs}
and optional object edits (index from the list; size_scale multiplies size; dx, dy shift in canvas fractions,
use 1.0/0/0 for no change). Set stop=true only if another iteration is unlikely to improve it.

Failed technical checks: {failed or 'none'}
Earlier versions:
{prev}"""
        data = base64.standard_b64encode(png.read_bytes()).decode()
        resp = self.client.messages.create(
            model=self.model, max_tokens=16000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": prompt}]}],
            output_config={"format": {"type": "json_schema", "schema": CRITIQUE_SCHEMA}},
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"critic stopped: {resp.stop_reason}")
        d = json.loads(next(b.text for b in resp.content if getattr(b, "type", None) == "text"))
        d["adjustments"] = {a["knob"]: a["value"] for a in d.get("adjustments", []) if a["knob"] in style_mod.KNOBS}
        return Critique.from_dict(d, f"claude:{self.model}", _looked(png))


class ManualCritic:
    """Step mode: returns None until someone who opened the PNG writes vN/critique.json."""

    name = "manual"

    def critique(self, png: Path, scene, style_mod, params, checks, history):
        path = png.parent / "critique.json"
        if not path.exists():
            template = {
                "summary": "", "reads_wrong": [], "flat": [], "cluttered": [], "unrecognizable": [],
                "technical": [], "strengths": [], "score": 5, "adjustments": {}, "object_edits": [],
                "code_changes": "", "stop": False,
                "_knobs": {k: {"current": params.get(k), "range": [lo, hi], "about": desc}
                           for k, (lo, hi, desc) in style_mod.KNOBS.items()},
                "_objects": [f"[{i}] {o.kind} ({o.x:.2f}, {o.y:.2f}) size {o.size:.2f}" for i, o in enumerate(scene.objects)],
            }
            (png.parent / "critique.template.json").write_text(json.dumps(template, indent=2))
            return None
        d = json.loads(path.read_text())
        d.pop("_knobs", None)
        d.pop("_objects", None)
        return Critique.from_dict(d, "manual", _looked(png))


class HeuristicCritic:
    """Offline fallback: rules over pixel statistics. Not a visual critique."""

    name = "heuristic"

    def critique(self, png: Path, scene, style_mod, params, checks, history):
        img = load_rgb(png)
        m = metrics.summary(img)
        c = Critique(critic="heuristic (pixel statistics only; not a visual critique)", looked_at=_looked(png))
        knobs = style_mod.KNOBS
        adj = {}

        def nudge(knob, delta):
            if knob in knobs:
                lo, hi, _ = knobs[knob]
                cur = adj.get(knob, params.get(knob))
                if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                    new = min(max(cur + delta, lo), hi)
                    adj[knob] = round(new, 3) if isinstance(cur, float) else int(round(new))

        score = 6.0
        if m["lum_std"] < 0.12:
            c.flat.append(f"low tonal range (luminance std {m['lum_std']:.2f})")
            nudge("shade_strength", 0.15)
            nudge("contrast", 0.15)
            score -= 1
        if m["edge_density"] > 0.35 and style_mod.NAME != "impasto":
            c.cluttered.append(f"very busy surface (edge density {m['edge_density']:.2f})")
            for k in ("water_lines", "ink_texture", "stroke_density", "density", "texture"):
                nudge(k, -0.15)
            score -= 1
        rules = getattr(style_mod, "RULES", {})
        if "min_empty_fraction" in rules and m["empty_fraction"] < rules["min_empty_fraction"]:
            c.cluttered.append(f"too little empty paper ({m['empty_fraction']:.0%})")
            for k in ("density", "stroke_density", "wash_amount", "wash"):
                nudge(k, -0.15)
            score -= 1
        for ch in checks.get("checks", []):
            if not ch["passed"]:
                c.technical.append(f"{ch['name']}: {ch['detail']}")
                score -= 1
                if ch["name"] == "in_frame":
                    for i, o in enumerate(scene.objects):
                        if any(d["kind"] == o.kind and d["fraction"] < 0.85 for d in ch.get("objects", [])):
                            c.object_edits.append({"index": i, "size_scale": 0.85,
                                                   "dx": round((0.5 - o.x) * 0.2, 3), "dy": round((0.5 - o.y) * 0.2, 3),
                                                   "note": "pull cropped object inward"})
        c.adjustments = adj
        c.score = max(1.0, score)
        c.summary = ("Heuristic pass over pixel statistics: " + json.dumps(m)) if adj or c.object_edits else \
            "Heuristic pass found nothing to change; a visual critic is needed for anything subtler."
        c.stop = not adj and not c.object_edits
        return c


def make_critic(kind: str, model=None, api_key=None):
    if kind == "auto":
        kind = "claude" if (api_key or os.environ.get("ANTHROPIC_API_KEY")) else "manual"
    if kind == "claude":
        return ClaudeCritic(model=model, api_key=api_key)
    if kind == "manual":
        return ManualCritic()
    if kind == "heuristic":
        return HeuristicCritic()
    raise ValueError(f"unknown critic {kind!r}")
