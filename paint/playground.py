"""Playground: freestyle scene generation.

Three ways to get a scene without writing YAML:

* ``surprise(seed)``         - a random but composed scene (setting, objects, palette, light).
* ``from_prompt(text)``      - a sentence becomes a scene. With an API key, Claude writes the
                               spec (structured output against the scene schema). Without one,
                               an offline keyword parser picks kinds, mood and palette, and the
                               composer lays them out.
* ``compose(kinds, ...)``    - the layout engine both use: horizon, depth, rule-of-thirds
                               placement, perspective sizing, and indoor vs outdoor logic.

The result is always a normal scene dict, so anything made here can be saved,
rendered in any style, iterated in the loop, or added to the suite.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .core.rng import Rng
from .scene.shapes import KINDS, SYNONYMS

SKY = ["sun", "moon", "stars", "cloud"]
BACKDROP = ["mountain", "hills"]
STANDING = ["tree", "pine", "cypress", "house", "lighthouse", "windmill", "bamboo", "reeds"]
TABLETOP = ["vase", "fruit", "teapot", "cup"]
# How far above its base each kind reaches, as a multiple of its size (keeps tall things on canvas).
REACH = {"tree": 0.95, "pine": 0.85, "cypress": 1.0, "house": 0.75, "lighthouse": 1.0, "windmill": 1.25,
         "bamboo": 1.05, "reeds": 1.0, "vase": 1.05, "teapot": 0.85, "cup": 0.5, "fruit": 1.2}
WATERCRAFT = ["boat"]

PALETTES = [
    {"name": "harbour dusk", "mood": "warm", "hints": ["#f2c14e", "#e8643c", "#2a8c88", "#1d2d4a"], "tags": ["dusk", "sunset", "warm", "sea"]},
    {"name": "alpine morning", "mood": "cool", "hints": ["#f4e3b5", "#d1495b", "#3c7a4a", "#2e5c8a", "#1b2238"], "tags": ["mountain", "morning", "cool"]},
    {"name": "moonlight", "mood": "night", "hints": ["#f2d16b", "#3d5a80", "#98c1d9", "#141b2d"], "tags": ["night", "moon", "stars"]},
    {"name": "tea house", "mood": "neutral", "hints": ["#f3e1bd", "#2a9d8f", "#e76f51", "#264653"], "tags": ["indoor", "tea", "still"]},
    {"name": "orchard", "mood": "warm", "hints": ["#f6d78b", "#e07a3f", "#5b8c3a", "#2a3d2a"], "tags": ["field", "summer", "tree"]},
    {"name": "rainy river", "mood": "cool", "hints": ["#d9dcd6", "#81a4cd", "#3e7cb1", "#1c2541"], "tags": ["rain", "grey", "river"]},
    {"name": "coral reef", "mood": "warm", "hints": ["#ffe8d6", "#ff7f50", "#00a6a6", "#1b4965"], "tags": ["tropical", "bright"]},
    {"name": "autumn", "mood": "warm", "hints": ["#f4d35e", "#ee964b", "#a23e48", "#3d2b1f"], "tags": ["autumn", "fall", "leaves"]},
    {"name": "winter", "mood": "cool", "hints": ["#f1f5f9", "#9fb8c8", "#577590", "#1e2a38"], "tags": ["winter", "snow", "cold"]},
    {"name": "spring", "mood": "neutral", "hints": ["#fdf0d5", "#f4a6b8", "#8cb369", "#3a5a40"], "tags": ["spring", "blossom", "garden"]},
    {"name": "desert", "mood": "warm", "hints": ["#f7e1b5", "#e9a15a", "#c0603b", "#4a2c2a"], "tags": ["desert", "dry", "hot"]},
    {"name": "storm", "mood": "cool", "hints": ["#cfd8dc", "#78909c", "#455a64", "#1c262b"], "tags": ["storm", "stormy", "dark"]},
]

SETTINGS = {
    "coast": {"base": "sea", "pick": [(["boat"], 0.9), (["lighthouse"], 0.35), (["bird"], 0.6), (["cloud"], 0.5), (["sun", "moon"], 0.7), (["hills", "mountain"], 0.5), (["cliff"], 0.25)]},
    "countryside": {"base": "field", "pick": [(["tree", "pine"], 0.9), (["house"], 0.6), (["windmill"], 0.3), (["sun"], 0.6), (["cloud"], 0.6), (["hills", "mountain"], 0.6), (["bird"], 0.4), (["river"], 0.25)]},
    "mountains": {"base": "field", "pick": [(["mountain"], 1.0), (["pine"], 0.9), (["house"], 0.5), (["cloud"], 0.4), (["moon", "sun"], 0.4), (["bird"], 0.3)]},
    "night": {"base": "sea", "pick": [(["moon"], 1.0), (["stars"], 1.0), (["hills"], 0.7), (["pine", "cypress", "house"], 0.8), (["boat"], 0.4)]},
    "still life": {"base": "table", "pick": [(["vase", "teapot"], 1.0), (["fruit"], 0.8), (["cup"], 0.6)]},
    "garden": {"base": "field", "pick": [(["bamboo"], 0.7), (["bird"], 0.7), (["reeds"], 0.5), (["tree"], 0.5), (["sun", "moon"], 0.4)]},
    "rain": {"base": "field", "pick": [(["rain"], 1.0), (["river"], 0.7), (["boat"], 0.6), (["reeds"], 0.6), (["hills"], 0.7), (["house"], 0.3)]},
}

MOOD_WORDS = {
    "night": ["night", "midnight", "moonlit", "moonlight", "starry", "stars", "dark"],
    "warm": ["dusk", "sunset", "sunrise", "evening", "golden", "warm", "autumn", "summer", "desert"],
    "cool": ["morning", "winter", "cold", "snow", "mist", "misty", "rain", "rainy", "storm", "grey", "gray", "fog"],
}

COLOR_WORDS = {
    "red": "#c8412f", "orange": "#e8843c", "yellow": "#f2c14e", "gold": "#e0a526", "green": "#4f8a4a",
    "teal": "#2a8c88", "blue": "#2e5c8a", "navy": "#1d2d4a", "purple": "#6b4c9a", "violet": "#7d5ba6",
    "pink": "#f4a6b8", "brown": "#7a5230", "black": "#1a1a1a", "white": "#f4f1ea", "grey": "#8a8f94", "gray": "#8a8f94",
}


def _plain(obj):
    """Numpy scalars -> plain Python, so scenes serialise to YAML/JSON cleanly."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _r(v, nd=3):
    return float(round(float(v), nd))


def _pick_palette(rng: Rng, mood: str | None, words: set[str]):
    scored = []
    for p in PALETTES:
        s = len(words & set(p["tags"])) * 2 + (1 if mood and p["mood"] == mood else 0)
        scored.append((s + rng.uniform(0, 0.5), p))
    return max(scored, key=lambda t: t[0])[1]


def compose(kinds: list[str], rng: Rng, mood: str | None = None, palette: dict | None = None,
            width: int = 1024, height: int = 768, counts: dict | None = None, title: str = "playground",
            variants: dict | None = None) -> dict:
    """Lay out a list of object kinds as a well-composed scene dict."""
    kinds = [k for k in kinds if k in KINDS]
    counts = counts or {}
    indoor = any(k in TABLETOP for k in kinds) and not any(k in STANDING + WATERCRAFT + BACKDROP for k in kinds)
    if indoor:
        kinds = [k for k in kinds if k in TABLETOP or k == "table"]
        if "table" not in kinds:
            kinds.insert(0, "table")
    elif not any(k in ("sea", "field", "river") for k in kinds):
        kinds.insert(0, "sea" if any(k in ("boat", "lighthouse", "cliff") for k in kinds) else "field")
    if "river" in kinds and "field" not in kinds:
        kinds.insert(0, "field")
    if "lighthouse" in kinds and "sea" in kinds and "cliff" not in kinds:
        kinds.insert(kinds.index("lighthouse"), "cliff")  # lighthouses stand on rock, not open water
    if mood is None:
        mood = "night" if ("moon" in kinds or "stars" in kinds) and "sun" not in kinds else rng.choice(["warm", "cool", "neutral"])
    horizon = _r(rng.uniform(0.56, 0.64) if indoor else rng.uniform(0.5, 0.7))
    if "mountain" in kinds:
        horizon = _r(max(horizon, 0.62))

    objs = []
    used_x = []

    def free_x(lo=0.18, hi=0.82, min_gap=0.14):
        for _ in range(40):
            x = rng.uniform(lo, hi)
            if all(abs(x - u) > min_gap for u in used_x):
                used_x.append(x)
                return x
        x = rng.uniform(lo, hi)
        used_x.append(x)
        return x

    thirds = [1 / 3, 2 / 3]
    for k in kinds:
        n = int(counts.get(k, 1))
        if k in ("sea", "field", "table"):
            objs.append({"kind": k, "y": horizon})
        elif k == "river":
            objs.append({"kind": "river", "x": _r(rng.uniform(0.4, 0.6)), "size": _r(rng.uniform(0.4, 0.6))})
        elif k == "hills":
            objs.append({"kind": "hills", "y": horizon, "size": _r(rng.uniform(0.07, 0.12)), "count": int(rng.integers(1, 3))})
        elif k == "mountain":
            s = rng.uniform(0.3, 0.45)
            objs.append({"kind": "mountain", "x": _r(rng.uniform(0.35, 0.65)), "y": _r(horizon - s / 2 + 0.01), "size": _r(s),
                         "count": int(rng.integers(1, 4))})
        elif k in ("sun", "moon"):
            x = thirds[int(rng.integers(0, 2))] + rng.uniform(-0.08, 0.08)
            o = {"kind": k, "x": _r(x), "y": _r(rng.uniform(0.15, horizon - 0.18)), "size": _r(rng.uniform(0.1, 0.2))}
            if k == "moon" and rng.random() < 0.5:
                o["variant"] = "full"
            objs.append(o)
        elif k == "stars":
            objs.append({"kind": "stars", "size": 0.5, "count": int(counts.get(k, rng.integers(10, 20)))})
        elif k == "cloud":
            for i in range(max(n, int(rng.integers(1, 3)))):
                objs.append({"kind": "cloud", "x": _r(rng.uniform(0.15, 0.85)), "y": _r(rng.uniform(0.12, horizon - 0.25)),
                             "size": _r(rng.uniform(0.08, 0.14))})
        elif k == "bird":
            objs.append({"kind": "bird", "x": _r(rng.uniform(0.2, 0.8)), "y": _r(rng.uniform(0.15, horizon - 0.2)),
                         "size": _r(rng.uniform(0.05, 0.08)), "count": max(n, int(rng.integers(2, 5)))})
        elif k == "rain":
            objs.append({"kind": "rain", "size": 0.3, "count": 90})
        elif k == "cliff":
            side = 0.62 if rng.random() < 0.5 else 0.38
            objs.append({"kind": "cliff", "x": side, "y": _r(horizon - 0.04), "size": 0.4})
        elif k == "boat":
            for i in range(n):
                y = rng.uniform(horizon + 0.08, 0.86)
                depth_t = (y - horizon) / (1 - horizon)
                o = {"kind": "boat", "x": _r(free_x(0.2, 0.8)), "y": _r(y), "size": _r(min(0.12 + 0.25 * depth_t, (y - 0.04) / 1.2))}
                if (variants or {}).get("boat"):
                    o["variant"] = variants["boat"]
                if rng.random() < 0.5:
                    o["facing"] = "left"
                objs.append(o)
        elif k == "lighthouse":
            cliff = next((o for o in objs if o["kind"] == "cliff"), None)
            if cliff:
                objs.append({"kind": "lighthouse", "x": _r(cliff["x"] + (0.14 if cliff["x"] > 0.5 else -0.14)), "y": _r(cliff["y"] + 0.01),
                             "size": _r(min(0.45, (cliff["y"] - 0.03) / REACH["lighthouse"])), "beam": False})
            else:
                y = horizon + 0.03
                objs.append({"kind": "lighthouse", "x": _r(free_x(0.15, 0.85)), "y": _r(y),
                             "size": _r(min(rng.uniform(0.35, 0.5), (y - 0.04) / REACH["lighthouse"])), "beam": False})
        elif k in STANDING:
            for i in range(n):
                y = rng.uniform(horizon + 0.06, 0.95)
                depth_t = (y - horizon) / (1 - horizon)
                base = {"tree": 0.45, "pine": 0.32, "cypress": 0.55, "house": 0.18, "windmill": 0.55, "bamboo": 0.9, "reeds": 0.3}[k]
                size = base * (0.55 + 0.6 * depth_t)
                yb = y if k != "bamboo" else 1.0
                o = {"kind": k, "x": _r(free_x()), "y": _r(yb), "size": _r(min(size, (yb - 0.04) / REACH[k]))}
                if k in ("bamboo", "reeds"):
                    o["count"] = int(rng.integers(2, 4)) if k == "bamboo" else 10
                objs.append(o)
        elif k in TABLETOP:
            for i in range(n):
                base_y = rng.uniform(horizon + 0.18, horizon + 0.26)
                size = {"vase": 0.45, "teapot": 0.45, "cup": 0.28, "fruit": 0.12}[k]
                o = {"kind": k, "x": _r(free_x(0.2, 0.8, 0.2)), "y": _r(base_y), "size": _r(size * rng.uniform(0.85, 1.1))}
                if k == "fruit":
                    o["count"] = max(n, int(rng.integers(1, 4)))
                    o["variant"] = (variants or {}).get("fruit") or str(rng.choice(["apple", "orange", "pear"]))
                objs.append(o)
                if k == "fruit":
                    break

    # Focal subject: the first object that isn't setting goes on a third and is enlarged.
    focal = next((o for o in objs if o["kind"] in STANDING + TABLETOP + WATERCRAFT + ["lighthouse"]), None)
    if focal and focal["kind"] not in ("bamboo", "reeds", "lighthouse"):
        third = 1 / 3 if rng.random() < 0.5 else 2 / 3
        others = [o for o in objs if o is not focal and "x" in o and o["kind"] in STANDING + TABLETOP + WATERCRAFT]
        focal["x"] = _r(third + rng.uniform(-0.04, 0.04))
        for o in others:  # keep the rest out of the focal subject's way
            if abs(o["x"] - focal["x"]) < 0.16:
                o["x"] = _r(min(max(1 - focal["x"] + rng.uniform(-0.1, 0.1), 0.15), 0.85))
        reach = REACH.get(focal["kind"], 1.2)
        focal["size"] = _r(min(focal["size"] * 1.3, (focal["y"] - 0.04) / reach))

    # Keep sun/moon clear of tall subjects.
    tall = [o for o in objs if o["kind"] in ("lighthouse", "windmill", "tree", "cypress", "pine", "bamboo") and "x" in o]
    for o in objs:
        if o["kind"] in ("sun", "moon") and any(abs(o["x"] - t["x"]) < 0.2 for t in tall):
            cands = [c for c in (0.2, 0.35, 0.5, 0.65, 0.8) if all(abs(c - t["x"]) >= 0.2 for t in tall)]
            if cands:
                o["x"] = _r(min(cands, key=lambda c: abs(c - (1 - o["x"]))))

    words = set(re.findall(r"[a-z]+", title.lower()))
    pal = palette or _pick_palette(rng, mood, words | set(kinds))
    az = float(rng.choice([200, 215, 330, 300, 250]))
    return _plain({
        "title": title,
        "canvas": {"width": int(width), "height": int(height)},
        "seed": int(rng.integers(0, 10_000)),
        "light": {"azimuth": az, "elevation": _r(rng.uniform(20, 55), 1), "warmth": 0.8 if mood == "warm" else 0.3},
        "palette": {"hints": list(pal["hints"]), "mood": mood if mood in ("warm", "cool", "night", "neutral") else pal["mood"]},
        "horizon": horizon,
        "objects": objs,
    })


def surprise(seed: int | None = None, setting: str | None = None, width=1024, height=768) -> dict:
    seed = int(seed if seed is not None else np.random.SeedSequence().entropy % 1_000_000)
    rng = Rng(seed, "surprise")
    setting = setting or str(rng.choice(sorted(SETTINGS)))
    spec = SETTINGS[setting]
    kinds = [spec["base"]]
    for i, (options, p) in enumerate(spec["pick"]):
        if i == 0 or rng.random() < p:  # the first pick is the focal subject: always present
            kinds.append(str(rng.choice(options)))
    mood = "night" if setting == "night" else ("cool" if setting in ("rain", "mountains") else None)
    title = f"{setting.title()} (surprise #{seed})"
    scene = compose(kinds, rng.child("compose"), mood=mood, width=width, height=height, title=title)
    scene["seed"] = seed
    scene["notes"] = f"playground surprise: setting={setting}"
    return _plain(scene)


def parse_prompt(text: str, seed: int = 0, width=1024, height=768) -> dict:
    """Offline prompt -> scene: keyword matching against the object vocabulary."""
    t = text.lower()
    variants = {}
    for word, kind, var in (("sailboat", "boat", "sail"), ("sailing", "boat", "sail"), ("fishing", "boat", "fishing"),
                            ("rowboat", "boat", "row"), ("rowing", "boat", "row"), ("apple", "fruit", "apple"),
                            ("pear", "fruit", "pear"), ("orange", "fruit", "orange")):
        if re.search(rf"\b{word}", t):
            variants.setdefault(kind, var)
    # "pine trees" / "cypress trees" name one kind, not two.
    t = re.sub(r"\b(pine|cypress|fir|palm)\s+trees?\b", r"\1", t)
    words = re.findall(r"[a-z]+", t)
    wset = set(words)
    found: list[str] = []
    counts: dict[str, int] = {}
    numbers = {"two": 2, "three": 3, "four": 4, "five": 5, "several": 3, "many": 5, "a": 1, "one": 1}
    # Most specific kinds first so "lighthouse" wins over "house", "teapot" over "pot".
    order = sorted(KINDS, key=lambda k: -max(len(s) for s in SYNONYMS.get(k, [k])))
    for k in order:
        for syn in SYNONYMS.get(k, [k]):
            m = re.search(rf"\b(?:(\w+)\s+)?{re.escape(syn)}\b", t)
            if m and k not in found:
                if k == "house" and "lighthouse" in t and not re.search(r"\bhouses?\b", t):
                    continue
                if k == "tree" and any(w in wset for w in ("pine", "pines", "cypress", "fir")) and not re.search(r"\btrees?\b", t):
                    continue
                found.append(k)
                if m.group(1) in numbers:
                    counts[k] = numbers[m.group(1)]
                elif syn.endswith("s") and syn not in ("stars", "reeds") and len(syn) > 3:
                    counts[k] = 2
                break
    for num, k in re.findall(r"\b(\d+)\s+(\w+)", t):
        for kind in found:
            if any(k.rstrip("s") == s.rstrip("s") for s in SYNONYMS.get(kind, [])):
                counts[kind] = min(int(num), 6)
    mood = next((m for m, ws in MOOD_WORDS.items() if wset & set(ws)), None)
    if mood == "warm" and "sun" not in found and wset & {"sunset", "sunrise", "dusk"}:
        found.append("sun")
    if mood == "night" and "moon" not in found and "stars" not in found:
        found.append("moon")
    if wset & {"rain", "rainy", "raining", "storm"} and "rain" not in found:
        found.append("rain")
    if not found:
        return surprise(seed, width=width, height=height) | {"title": text[:80] or "untitled",
                                                            "notes": "prompt matched no known objects; surprise scene"}
    rng = Rng(seed, "prompt", t)
    pal = None
    colors = [COLOR_WORDS[w] for w in words if w in COLOR_WORDS]
    if colors:
        base = _pick_palette(rng, mood, wset)
        hints = list(dict.fromkeys(colors + base["hints"]))[:5]
        pal = {"hints": hints, "mood": base["mood"]}
    scene = compose(found, rng, mood=mood, palette=pal, width=width, height=height, counts=counts,
                    title=text.strip()[:80], variants=variants)
    scene["seed"] = seed
    scene["notes"] = f"offline prompt parser: matched {', '.join(found)}"
    return _plain(scene)


SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "horizon": {"type": "number"},
        "mood": {"type": "string", "enum": ["warm", "cool", "night", "neutral"]},
        "palette": {"type": "array", "items": {"type": "string"}},
        "light_azimuth": {"type": "number"},
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": sorted(KINDS)},
                "x": {"type": "number"}, "y": {"type": "number"}, "size": {"type": "number"},
                "count": {"type": "integer"},
                "variant": {"type": "string"},
            },
            "required": ["kind", "x", "y", "size", "count", "variant"],
            "additionalProperties": False}},
    },
    "required": ["title", "horizon", "mood", "palette", "light_azimuth", "objects"],
    "additionalProperties": False,
}

PROMPT_GUIDE = """Turn the user's idea into a scene spec for a procedural painter. You can only use these object kinds:
{kinds}

Coordinates: x, y in [0, 1] from the top-left of the canvas; size is a fraction of canvas HEIGHT.
- sea / field / table: regions from y down to the bottom; set y = horizon.
- hills: y = horizon, size = hill height (0.05-0.15), count = layers (1-2).
- mountain: (x, y) is the centre, so y ~ horizon - size/2; size 0.25-0.5; count = number of peaks.
- tree, pine, cypress, house, lighthouse, windmill, bamboo, reeds, vase, teapot, cup, fruit: (x, y) is the BASE
  (where it touches the ground or table); it extends UP by roughly `size`. The base must be below the horizon.
- boat: (x, y) is the waterline centre, below the horizon; size 0.12-0.35.
- sun, moon, cloud, bird: centre; birds use count for a flock. stars: size 0.5, count 8-20. rain: count 60-120.
- river: x = centre; size 0.3-0.6 = width at the bottom.
- Tabletop things need a table; don't mix them with landscape objects.
Variants: boat sail|fishing|row; moon crescent|full; bird flying|perched; fruit apple|pear|orange; otherwise "".
Keep everything inside the canvas, give the picture one clear focal subject on a third, and don't overcrowd
(3-8 objects). Palette: 3-5 hex colours that suit the mood. light_azimuth: degrees the light comes FROM
(0 right, 90 below, 180 left, 270 above). Use count 1 when not relevant."""


def prompt_with_claude(text: str, api_key: str | None = None, model: str | None = None, seed: int = 0,
                       width=1024, height=768) -> dict:
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key, timeout=120.0) if key else anthropic.Anthropic(timeout=120.0)
    resp = client.messages.create(
        model=model or os.environ.get("PAINT_JUDGE_MODEL", "claude-opus-5"),
        max_tokens=16000,
        system=PROMPT_GUIDE.format(kinds=", ".join(sorted(KINDS))),
        messages=[{"role": "user", "content": text}],
        output_config={"format": {"type": "json_schema", "schema": SCENE_SCHEMA}},
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"scene writer stopped: {resp.stop_reason}")
    d = json.loads(next(b.text for b in resp.content if getattr(b, "type", None) == "text"))
    objs = []
    for o in d["objects"]:
        o = {k: v for k, v in o.items() if not (k == "variant" and not v) and not (k == "count" and v == 1)}
        o["x"], o["y"] = _r(min(max(o["x"], 0.0), 1.0)), _r(min(max(o["y"], 0.0), 1.1))
        o["size"] = _r(min(max(o["size"], 0.02), 1.2))
        objs.append(o)
    hints = [h for h in d["palette"] if re.fullmatch(r"#[0-9a-fA-F]{6}", h)][:5] or PALETTES[0]["hints"]
    return {"title": d["title"][:80], "canvas": {"width": width, "height": height}, "seed": seed,
            "light": {"azimuth": float(d["light_azimuth"]) % 360, "elevation": 35, "warmth": 0.8 if d["mood"] == "warm" else 0.4},
            "palette": {"hints": hints, "mood": d["mood"]}, "horizon": _r(min(max(d["horizon"], 0.3), 0.85)),
            "objects": objs, "notes": f"written by {model or 'claude'} from prompt: {text[:200]}"}


def from_prompt(text: str, api_key: str | None = None, model: str | None = None, seed: int = 0,
                use_claude: bool | None = None, width=1024, height=768) -> tuple[dict, str]:
    """Returns (scene_dict, source) where source is 'claude' or 'offline'."""
    if use_claude is None:
        use_claude = bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if use_claude:
        try:
            return prompt_with_claude(text, api_key, model, seed, width, height), "claude"
        except Exception as exc:  # noqa: BLE001 - fall back, but say so
            scene = parse_prompt(text, seed, width, height)
            scene["notes"] += f" (Claude scene writer failed: {type(exc).__name__}: {str(exc)[:160]})"
            return scene, "offline"
    return parse_prompt(text, seed, width, height), "offline"


def preview_scene(scene: dict, max_side: int = 384) -> dict:
    """The same scene at a smaller canvas (coordinates are normalised, so layout is identical)."""
    d = json.loads(json.dumps(scene))
    c = d.setdefault("canvas", {})
    w, h = int(c.get("width", 1024)), int(c.get("height", 768))
    k = min(1.0, max_side / max(w, h))
    c["width"], c["height"] = max(64, int(w * k)), max(64, int(h * k))
    return d
