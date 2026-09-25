"""Render service: one guarded render to a PNG plus a JSON sidecar."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .core.guard import DEFAULT_MEMORY_BYTES, DEFAULT_TIMEOUT_S, RenderFailure, run_guarded
from .scene.spec import Scene, load_scene
from .styles import get_style


@dataclass
class RenderRecord:
    scene: str
    style: str
    seed: int
    params: dict
    png: str
    seconds: float
    peak_rss_mb: float
    info: dict
    ok: bool = True
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _render_bytes(scene: Scene, style: str, seed: int, params: dict | None) -> tuple[bytes, dict]:
    mod = get_style(style)
    img, ctx = mod.render_array(scene, seed, params)
    from .styles.base import encode

    return encode(img, mod.NAME, ctx), {"info": ctx.info, "params": ctx.params}


def render_bytes(scene: Scene, style: str, seed: int | None = None, params: dict | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S, mem_bytes: int = DEFAULT_MEMORY_BYTES):
    """Guarded render -> (png_bytes, meta, seconds, peak_rss_mb). Raises RenderFailure."""
    seed = scene.seed if seed is None else int(seed)
    res = run_guarded(_render_bytes, scene, style, seed, params, timeout=timeout, mem_bytes=mem_bytes)
    png, meta = res.value
    return png, meta, res.seconds, res.peak_rss_mb


def default_name(scene: Scene, style: str, seed: int) -> str:
    stem = Path(scene.source).stem if scene.source else scene.title.lower().replace(" ", "_")
    return f"{stem}__{style}__s{seed}"


def render_to_file(scene_or_path, style: str, seed: int | None = None, out_dir="out",
                   params: dict | None = None, name: str | None = None,
                   timeout: float = DEFAULT_TIMEOUT_S, raise_on_error: bool = True) -> RenderRecord:
    scene = scene_or_path if isinstance(scene_or_path, Scene) else load_scene(scene_or_path)
    seed = scene.seed if seed is None else int(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or default_name(scene, style, seed)
    png_path = out_dir / f"{name}.png"
    t0 = time.perf_counter()
    try:
        png, meta, secs, rss = render_bytes(scene, style, seed, params, timeout=timeout)
    except RenderFailure as exc:
        rec = RenderRecord(scene.source or scene.title, style, seed, params or {}, str(png_path),
                           time.perf_counter() - t0, float("nan"), {}, ok=False, error=str(exc))
        (out_dir / f"{name}.json").write_text(rec.to_json())
        if raise_on_error:
            raise
        return rec
    png_path.write_bytes(png)
    rec = RenderRecord(scene.source or scene.title, style, seed, meta["params"], str(png_path),
                       round(secs, 3), round(rss, 1), meta["info"])
    (out_dir / f"{name}.json").write_text(rec.to_json())
    return rec
