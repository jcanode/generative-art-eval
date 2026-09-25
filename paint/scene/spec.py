"""Scene specs: WHAT to draw, independent of style.

A scene is YAML or JSON:

    title: Harbour at dusk
    canvas: {width: 1024, height: 768}
    seed: 7
    light: {azimuth: 225, elevation: 30, warmth: 0.8}
    palette: {hints: ["#e8643c", "#1d3557", "#f4a261"], mood: warm}
    horizon: 0.62
    objects:
      - {kind: sun, x: 0.72, y: 0.36, size: 0.2}
      - {kind: sea}
      - {kind: boat, x: 0.35, y: 0.74, size: 0.22}

Coordinates are normalised: x, y in [0, 1] from the top-left; ``size`` is a
fraction of canvas height. ``depth`` (0 near .. 1 far) orders drawing; if it
is omitted it is inferred from the kind and y position.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path



_HORIZON_KINDS = {"sea", "field", "table", "hills", "river"}


class SceneError(ValueError):
    pass


@dataclass
class Light:
    azimuth: float = 225.0   # degrees, where the light comes FROM in image space: 0 right, 90 below, 180 left, 270 above
    elevation: float = 35.0  # degrees above the picture plane
    warmth: float = 0.5      # 0 cool .. 1 warm

    @property
    def direction(self) -> tuple[float, float]:
        """Unit vector pointing FROM the scene TOWARDS the light, image coords (y down)."""
        a = math.radians(self.azimuth)
        return (math.cos(a), math.sin(a))


@dataclass
class Palette:
    hints: list[str] = field(default_factory=list)
    mood: str = "neutral"     # warm | cool | neutral | night
    paper: str | None = None


@dataclass
class SceneObject:
    kind: str
    x: float = 0.5
    y: float = 0.5
    size: float = 0.2
    rotation: float = 0.0     # degrees
    depth: float | None = None
    color: str | None = None  # palette role (primary/secondary/accent/dark/light) or hex
    count: int = 1
    variant: str | None = None
    id: str | None = None
    params: dict = field(default_factory=dict)


@dataclass
class Scene:
    title: str = "untitled"
    width: int = 1024
    height: int = 768
    seed: int = 0
    light: Light = field(default_factory=Light)
    palette: Palette = field(default_factory=Palette)
    horizon: float = 0.62
    objects: list[SceneObject] = field(default_factory=list)
    notes: str = ""
    source: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source", None)
        d["canvas"] = {"width": d.pop("width"), "height": d.pop("height")}
        return d

    @property
    def kinds(self) -> list[str]:
        return [o.kind for o in self.objects]

    def with_seed(self, seed: int) -> "Scene":
        from copy import deepcopy

        s = deepcopy(self)
        s.seed = int(seed)
        return s


def _build(d: dict, source: str | None) -> Scene:
    from .shapes import KINDS

    if not isinstance(d, dict):
        raise SceneError("scene must be a mapping")
    canvas = d.get("canvas", {}) or {}
    light = Light(**(d.get("light") or {}))
    pal = d.get("palette") or {}
    if isinstance(pal, list):
        pal = {"hints": pal}
    palette = Palette(**pal)
    objs = []
    for i, o in enumerate(d.get("objects") or []):
        if "kind" not in o:
            raise SceneError(f"object #{i} has no kind")
        o = dict(o)
        known = set(SceneObject.__dataclass_fields__)
        params = {k: o.pop(k) for k in list(o) if k not in known}
        if "y" not in o and o["kind"] in _HORIZON_KINDS:
            o["y"] = float(d.get("horizon", 0.62))
        obj = SceneObject(**o)
        obj.params.update(params)
        if obj.kind not in KINDS:
            raise SceneError(f"object #{i}: unknown kind {obj.kind!r}; known: {sorted(KINDS)}")
        if obj.size <= 0:
            raise SceneError(f"object #{i} ({obj.kind}): size must be positive")
        objs.append(obj)
    scene = Scene(
        title=d.get("title", "untitled"),
        width=int(canvas.get("width", d.get("width", 1024))),
        height=int(canvas.get("height", d.get("height", 768))),
        seed=int(d.get("seed", 0)),
        light=light,
        palette=palette,
        horizon=float(d.get("horizon", 0.62)),
        objects=objs,
        notes=d.get("notes", ""),
        source=source,
    )
    return scene


def load_scene(path_or_dict) -> Scene:
    if isinstance(path_or_dict, dict):
        return _build(path_or_dict, None)
    p = Path(path_or_dict)
    text = p.read_text()
    if p.suffix == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    return _build(data, str(p))


def dump_scene(scene: Scene, path) -> None:
    p = Path(path)
    if p.suffix == ".json":
        p.write_text(json.dumps(scene.to_dict(), indent=2))
    else:
        import yaml

        p.write_text(yaml.safe_dump(scene.to_dict(), sort_keys=False))
