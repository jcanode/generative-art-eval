"""The render -> look -> critique -> revise loop.

For every iteration:
  1. render (guarded: timeout + memory cap) and run the technical checks
  2. a critic LOOKS at the PNG (a vision model, or an agent/human in step mode)
  3. the critique is saved next to the image (critique.json + critique.md)
  4. knob adjustments and object edits are applied, and the next version renders

Hard limits guarantee exit: at most 4 versions (MAX_ITERS), a wall-clock
deadline for the whole loop, a timeout per render, and a timeout per critic
call. Every version is kept (v1/, v2/, ...) because earlier ones are sometimes
better; ``best.png`` is a copy of the highest-scoring version, and ties go to
the earlier version.

Holdout scenes (evals/holdout/) are refused: the generator never iterates on them.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.guard import DEFAULT_TIMEOUT_S
from ..render import render_to_file
from ..scene.spec import Scene, dump_scene, load_scene
from ..styles import get_style
from ..styles.base import clamp_params, merge_params
from .critics import Critique, make_critic

MAX_ITERS = 4
DEFAULT_DEADLINE_S = 45 * 60
HOLDOUT_DIR = Path(__file__).resolve().parents[2] / "evals" / "holdout"


class HoldoutError(RuntimeError):
    pass


@dataclass
class LoopResult:
    run_dir: Path
    status: str                      # done | awaiting_critique | deadline | error
    versions: list[dict] = field(default_factory=list)
    best: int | None = None
    message: str = ""

    def summary(self) -> dict:
        return {"run_dir": str(self.run_dir), "status": self.status, "best": self.best,
                "message": self.message, "versions": self.versions}


def apply_revisions(scene: Scene, params: dict, crit: Critique, style_mod) -> tuple[Scene, dict]:
    new_params = dict(params)
    for k, v in (crit.adjustments or {}).items():
        if k in style_mod.DEFAULTS:
            new_params[k] = v
    new_params = clamp_params(new_params, style_mod.KNOBS)
    new_scene = copy.deepcopy(scene)
    for e in crit.object_edits or []:
        i = int(e.get("index", -1))
        if 0 <= i < len(new_scene.objects):
            o = new_scene.objects[i]
            o.size = float(min(max(o.size * float(e.get("size_scale", 1.0) or 1.0), 0.01), 2.0))
            o.x = float(min(max(o.x + float(e.get("dx", 0.0) or 0.0), 0.0), 1.0))
            o.y = float(min(max(o.y + float(e.get("dy", 0.0) or 0.0), 0.0), 1.2))
    return new_scene, new_params


def _is_holdout(path) -> bool:
    try:
        Path(path).resolve().relative_to(HOLDOUT_DIR.resolve())
        return True
    except ValueError:
        return False


def run_loop(scene_path, style: str, seed: int | None = None, out_dir="runs/loop", critic="auto",
             max_iters: int = MAX_ITERS, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT_S,
             model: str | None = None, api_key: str | None = None, deadline_s: float = DEFAULT_DEADLINE_S,
             allow_holdout: bool = False, progress=None) -> LoopResult:
    if _is_holdout(scene_path) and not allow_holdout:
        raise HoldoutError(f"{scene_path} is a holdout scene; the generator loop never iterates on holdouts")
    say = progress or (lambda m: print(m, flush=True))
    max_iters = max(1, min(int(max_iters), MAX_ITERS))
    style_mod = get_style(style)
    base_scene = load_scene(scene_path)
    seed = base_scene.seed if seed is None else int(seed)
    stem = Path(scene_path).stem
    run_dir = Path(out_dir) / f"{stem}__{style_mod.NAME}__s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    critic_obj = make_critic(critic, model=model, api_key=api_key)
    t_start = time.time()

    state_path = run_dir / "loop.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "scene": str(scene_path), "style": style_mod.NAME, "seed": seed, "critic": critic_obj.name, "versions": []}
    result = LoopResult(run_dir, "running", state["versions"])

    scene = base_scene
    cur_params = clamp_params(merge_params(style_mod.DEFAULTS, params), style_mod.KNOBS)
    # Resume: rebuild scene/params from the last saved version.
    if state["versions"]:
        last = state["versions"][-1]
        scene = load_scene(run_dir / f"v{last['version']}" / "scene.yaml")
        cur_params = last["params"]

    def save():
        state_path.write_text(json.dumps(state, indent=2, default=str))

    for i in range(1, max_iters + 1):
        vdir = run_dir / f"v{i}"
        existing = next((v for v in state["versions"] if v["version"] == i), None)
        if existing and existing.get("critique"):
            if existing["critique"].get("stop"):
                break
            scene, cur_params = apply_revisions(
                load_scene(vdir / "scene.yaml"), existing["params"],
                Critique.from_dict(existing["critique"], "", ""), style_mod)
            continue
        if time.time() - t_start > deadline_s:
            result.status, result.message = "deadline", f"loop deadline of {deadline_s:.0f}s reached"
            break

        png = vdir / "image.png"
        if not existing:
            vdir.mkdir(parents=True, exist_ok=True)
            dump_scene(scene, vdir / "scene.yaml")
            say(f"v{i}: rendering {stem} in {style_mod.NAME} (seed {seed})")
            rec = render_to_file(scene, style_mod.NAME, seed, vdir, cur_params, name="image",
                                 timeout=timeout, raise_on_error=False)
            if not rec.ok:
                existing = {"version": i, "params": cur_params, "ok": False, "error": rec.error}
                state["versions"].append(existing)
                save()
                result.status, result.message = "error", f"v{i} render failed: {rec.error}"
                return result
            from ..evals.technical import run_checks

            checks = run_checks(png, scene, style_mod.NAME, seed=seed, params=cur_params,
                                seconds=rec.seconds, determinism=(i == 1))
            (vdir / "checks.json").write_text(json.dumps(checks, indent=2))
            existing = {"version": i, "params": rec.params, "seconds": rec.seconds, "ok": True,
                        "technical_passed": checks["passed"], "image": str(png)}
            state["versions"].append(existing)
            save()
        checks = json.loads((vdir / "checks.json").read_text())
        history = [{"version": v["version"], "score": v["critique"]["score"], "summary": v["critique"]["summary"]}
                   for v in state["versions"] if v.get("critique")]
        crit = critic_obj.critique(png, scene, style_mod, existing["params"], checks, history)
        if crit is None:
            result.status = "awaiting_critique"
            result.message = (f"v{i} rendered. Open {png} and LOOK at it, then write {vdir / 'critique.json'} "
                              f"(template: {vdir / 'critique.template.json'}) and re-run the same command.")
            say(result.message)
            return result
        (vdir / "critique.json").write_text(json.dumps(crit.to_dict(), indent=2))
        (vdir / "critique.md").write_text(crit.to_markdown(i))
        existing["critique"] = crit.to_dict()
        save()
        say(f"v{i}: score {crit.score}/10 - {crit.summary[:140]}")
        if crit.stop:
            break
        if i < max_iters:
            scene, cur_params = apply_revisions(scene, existing["params"], crit, style_mod)

    scored = [v for v in state["versions"] if v.get("critique")]
    if scored:
        best = max(scored, key=lambda v: (v["critique"]["score"], -v["version"]))
        result.best = best["version"]
        shutil.copyfile(run_dir / f"v{best['version']}" / "image.png", run_dir / "best.png")
        state["best"] = best["version"]
    save()
    if result.status == "running":
        result.status = "done"
        result.message = f"{len(state['versions'])} version(s); best is v{result.best}"
    _write_index(run_dir, state)
    return result


def _write_index(run_dir: Path, state: dict):
    """A small side-by-side HTML page of every version and its critique."""
    import html

    cells = []
    for v in state["versions"]:
        c = v.get("critique") or {}
        md = (run_dir / f"v{v['version']}" / "critique.md")
        crit = html.escape(md.read_text()) if md.exists() else "(no critique)"
        star = " ★ best" if state.get("best") == v["version"] else ""
        cells.append(f"<figure><img src='v{v['version']}/image.png' alt='version {v['version']}'>"
                     f"<figcaption><b>v{v['version']}{star}</b> · score {c.get('score', '–')} · {v.get('seconds', '–')}s"
                     f"<pre>{crit}</pre></figcaption></figure>")
    (run_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Loop versions</title><style>body{font:14px system-ui;margin:16px;background:#f7f5f0;color:#1d1c1a}"
        "@media(prefers-color-scheme:dark){body{background:#161513;color:#ece9e2}}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}"
        "img{width:100%;border-radius:4px}pre{white-space:pre-wrap;font-size:12px}</style>"
        f"<h1>{html.escape(state['scene'])} · {html.escape(state['style'])}</h1><main>{''.join(cells)}</main>")
