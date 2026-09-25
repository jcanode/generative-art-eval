"""Model arena: which model paints best when it has to write the whole program?

    paint arena --models claude-opus-5,claude-sonnet-5,claude-haiku-4-5 --iterations 4

For every task x contestant: run_session (write -> sandbox -> look -> revise).
Then every final image goes through the same blind evaluation as the studio's own
styles: technical checks, subject fidelity (judge lists what it sees), rubric
score, and position-swapped pairwise comparisons between all contestants on the
same task, which update an Elo table. The studio's hand-built styles take part as
the ``house`` player, a reference point that is judged exactly like the models.

Output: runs/arena/<run-id>/{results.json, report.html, <task>/<contestant>/v1..v4/}
and runs/arena/elo.json (persistent across runs).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ..core.io import load_rgb, thumbnail
from ..evals import metrics
from ..evals.fidelity import score_subjects
from ..evals.judges import JudgeError, load_rubric, make_judge
from ..evals.pairwise import EloTable, compare_pair
from ..evals.technical import style_rule_checks
from ..scene.spec import load_scene
from .brief import load_tasks, task_message
from .providers import DEFAULT_LINEUP, make_coder
from .sandbox import backend_name
from .session import run_session

ROOT = Path(__file__).resolve().parents[2]
HOUSE = "house"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def evaluate_image(judge, png: Path, scene, style: str | None, info_rules=True) -> dict:
    img = load_rgb(png)
    L = metrics.luminance(img)
    blank = float(L.std()) < 0.03 or metrics.color_clusters(img, 4, 0.002) < 3
    checks = [{"name": "not_blank", "passed": not blank, "detail": f"luminance std {L.std():.3f}"}]
    if style:
        from ..evals.technical import png_info

        info = png_info(png)  # house renders carry their facts (ink count); model programs don't
        checks += [c for c in style_rule_checks(style, img, info) if not (c["name"] == "style:max_inks" and not info)]
    out = {"technical": {"passed": all(c["passed"] for c in checks if not c.get("skipped")), "checks": checks},
           "metrics": metrics.summary(img)}
    if judge is not None:
        try:
            if judge.is_mock:
                out["fidelity"] = {"score": None, "note": "mock judge cannot see subjects"}
            else:
                out["fidelity"] = score_subjects(scene, judge.describe(png))
            if style:
                out["style_score"] = judge.score_style(png, load_rubric(style))
            out["overall"] = judge.rate_overall(png)
        except JudgeError as exc:
            out["judge_error"] = str(exc)
    return out


def run_arena(models=None, tasks=None, iterations=4, judge="auto", judge_model=None, api_key=None,
              out_root="runs/arena", label=None, include_house=True, backend=None, width=None, height=None,
              pairwise=True, max_cost_per_session=None, effort=None, progress=None, tasks_file=None) -> Path:
    say = progress or (lambda m: print(m, flush=True))
    models = models or (DEFAULT_LINEUP if (api_key or os.environ.get("ANTHROPIC_API_KEY")) else ["baseline:naive"])
    all_tasks = load_tasks(tasks_file)
    if tasks:
        wanted = set(tasks)
        all_tasks = [t for t in all_tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in all_tasks}
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(sorted(missing))}")
    root = Path(out_root) if Path(out_root).is_absolute() else ROOT / out_root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = label or stamp
    run_id = f"{stamp}_{_safe(label)}" if label != stamp else stamp
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or backend_name()
    judge_obj = None if judge == "none" else make_judge(judge, model=judge_model, api_key=api_key,
                                                        cache_path=ROOT / "runs" / "judge_cache.json")
    say(f"arena {run_id}: {len(models)} contestant(s) x {len(all_tasks)} task(s), {iterations} iteration(s), "
        f"sandbox={backend}, judge={judge_obj.name if judge_obj else 'none'}")

    jobs = []
    for t in all_tasks:
        scene_dict = yaml.safe_load((ROOT / t["scene"]).read_text()) if t.get("scene") else {}
        if width and height:
            scene_dict.setdefault("canvas", {}).update({"width": width, "height": height})
        W = int(scene_dict.get("canvas", {}).get("width", 1024))
        H = int(scene_dict.get("canvas", {}).get("height", 768))
        seed = int(scene_dict.get("seed", 0))
        text = task_message(t, scene_dict, W, H, seed)
        (run_dir / _safe(t["id"])).mkdir(exist_ok=True)
        (run_dir / _safe(t["id"]) / "brief.md").write_text(text)
        for m in models:
            jobs.append((t, scene_dict, text, W, H, seed, m))

    def do(job):
        t, scene_dict, text, W, H, seed, m = job
        coder = make_coder(m, api_key=api_key, scene=scene_dict, effort=effort)
        out = run_dir / _safe(t["id"]) / _safe(m)
        return run_session(coder, t, text, out, W, H, seed, iterations=iterations, backend=backend,
                           max_cost=max_cost_per_session, progress=say)

    workers = int(os.environ.get("PAINT_ARENA_WORKERS", min(4, max(1, len(jobs)))))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sessions = list(ex.map(do, jobs))

    # House player: the studio's own hand-built style for the same scene and seed.
    house = {}
    if include_house:
        from ..render import render_to_file

        for t in all_tasks:
            if not t.get("scene") or not t.get("style"):
                continue
            sc = load_scene(ROOT / t["scene"])
            if width and height:
                sc.width, sc.height = width, height
            rec = render_to_file(sc, t["style"], sc.seed, run_dir / _safe(t["id"]) / HOUSE, name="image", raise_on_error=False)
            if rec.ok:
                house[t["id"]] = rec.png
        say(f"  house renders: {len(house)} of {len(all_tasks)}")

    # Evaluate every final image (and each contestant's first working version, to measure self-improvement).
    entries = []
    for s in sessions:
        t = next(x for x in all_tasks if x["id"] == s.task_id)
        e = {"task": s.task_id, "player": s.contestant, "session": s.to_dict(), "style": t.get("style")}
        if s.final:
            sc = load_scene(ROOT / t["scene"]) if t.get("scene") else None
            final_png = Path(s.dir) / f"v{s.final}" / "image.png"
            e["image"] = os.path.relpath(final_png, run_dir)
            thumbnail(final_png, Path(s.dir) / "thumb.png", 360)
            e["thumb"] = os.path.relpath(Path(s.dir) / "thumb.png", run_dir)
            e["eval"] = evaluate_image(judge_obj, final_png, sc, t.get("style"))
            if s.first_ok and s.first_ok != s.final and judge_obj is not None and t.get("style"):
                try:
                    e["first_style"] = judge_obj.score_style(Path(s.dir) / f"v{s.first_ok}" / "image.png", load_rubric(t["style"]))
                except JudgeError as exc:
                    e["first_style_error"] = str(exc)
        entries.append(e)
    for tid, png in house.items():
        t = next(x for x in all_tasks if x["id"] == tid)
        thumbnail(png, Path(png).parent / "thumb.png", 360)
        entries.append({"task": tid, "player": HOUSE, "style": t.get("style"), "image": os.path.relpath(png, run_dir),
                        "thumb": os.path.relpath(Path(png).parent / "thumb.png", run_dir),
                        "eval": evaluate_image(judge_obj, Path(png), load_scene(ROOT / t["scene"]), t.get("style"))})

    pairs = []
    elo = EloTable(root / "elo.json")
    if pairwise and judge_obj is not None:
        by_task = {}
        for e in entries:
            if e.get("image"):
                by_task.setdefault(e["task"], []).append(e)
        tasks_pairs = [(tid, a, b) for tid, es in by_task.items() for a, b in itertools.combinations(es, 2)]

        def cmp(tp):
            tid, a, b = tp
            style = a.get("style")
            try:
                r = compare_pair(judge_obj, run_dir / a["image"], run_dir / b["image"],
                                 rubric=load_rubric(style) if style else None,
                                 seed=int(hashlib.md5(f"{tid}|{a['player']}|{b['player']}".encode()).hexdigest()[:6], 16))
            except JudgeError as exc:
                return {"task": tid, "error": str(exc)}
            r.update({"task": tid, "player_a": a["player"], "player_b": b["player"],
                      "a": a["image"], "b": b["image"]})
            return r

        with ThreadPoolExecutor(max_workers=4) as ex:
            for r in ex.map(cmp, tasks_pairs):
                pairs.append(r)
                if "verdict" in r:
                    elo.record(r["player_a"], r["player_b"], r["verdict"], f"arena:{r['task']}")
        elo.save()
        say(f"  pairwise: {len(pairs)} pairs, {sum(1 for p in pairs if p.get('consistent'))} consistent")

    results = {"run_id": run_id, "label": label, "created": stamp, "models": models, "iterations": iterations,
               "tasks": all_tasks, "sandbox": backend,
               "judge": {"name": judge_obj.name if judge_obj else "none", "model": judge_obj.model if judge_obj else None,
                         "mock": bool(judge_obj and judge_obj.is_mock)},
               "entries": entries, "pairs": pairs, "elo": elo.table(), "leaderboard": leaderboard(entries, elo)}
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    from .report import write_arena_report

    path = write_arena_report(results, run_dir)
    say(f"done -> {path}")
    return path


def leaderboard(entries, elo) -> list[dict]:
    ratings = dict((r["player"], r["elo"]) for r in elo.table())
    rows = []
    for p in sorted({e["player"] for e in entries}):
        es = [e for e in entries if e["player"] == p]
        sess = [e["session"] for e in es if "session" in e]

        def mean(vals):
            vals = [v for v in vals if v is not None]
            return round(float(np.mean(vals)), 3) if vals else None

        rows.append({
            "player": p, "tasks": len(es), "elo": ratings.get(p),
            "rendered": round(sum(1 for e in es if e.get("image")) / len(es), 3),
            "first_try": round(sum(1 for s in sess if s.get("first_ok") == 1) / len(sess), 3) if sess else None,
            "attempts_to_render": mean([s.get("first_ok") for s in sess]),
            "deterministic": round(sum(1 for s in sess if s.get("deterministic")) / len(sess), 3) if sess else None,
            "tech_pass": mean([1.0 if e.get("eval", {}).get("technical", {}).get("passed") else 0.0 for e in es if e.get("eval")]),
            "fidelity": mean([(e.get("eval", {}).get("fidelity") or {}).get("score") for e in es]),
            "style": mean([(e.get("eval", {}).get("style_score") or {}).get("mean") for e in es]),
            "style_first": mean([(e.get("first_style") or (e.get("eval", {}).get("style_score") if e.get("session", {}).get("first_ok") == e.get("session", {}).get("final") else None) or {}).get("mean") for e in es if "session" in e]),
            "overall": mean([(e.get("eval", {}).get("overall") or {}).get("score") for e in es]),
            "cost": round(sum(s.get("cost", 0) for s in sess), 3) if sess else None,
            "output_tokens": sum(v["output_tokens"] for s in sess for v in s.get("versions", [])) if sess else None,
        })
    rows.sort(key=lambda r: -(r["elo"] or 0))
    return rows
