"""Regression suite runner: render everything, check everything, judge, compare, report.

    paint eval --suite

Layout under ``runs/``:

    index.json            one entry per run (for diffs against the previous run)
    elo.json              Elo ratings across versions (style@run-label)
    judge_cache.json      judge answers keyed by image pixels + task
    <run-id>/renders/     PNG + JSON sidecar per scene x style
    <run-id>/results.json
    <run-id>/report.html
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..core.io import load_rgb, thumbnail
from ..scene.spec import load_scene
from ..styles import available
from . import metrics
from .fidelity import score_subjects
from .judges import JudgeError, load_rubric, make_judge
from .pairwise import EloTable, compare_pair
from .technical import run_checks

ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = ROOT / "evals" / "suite"
HOLDOUT_DIR = ROOT / "evals" / "holdout"
BASELINE_STYLE = "screenprint"


def suite_scenes(include_holdout=True):
    out = [(p, "suite") for p in sorted(SUITE_DIR.glob("*.yaml"))]
    if include_holdout:
        out += [(p, "holdout") for p in sorted(HOLDOUT_DIR.glob("*.yaml"))]
    return out


def code_fingerprint() -> str:
    h = hashlib.sha1()
    for p in sorted((ROOT / "paint").rglob("*.py")):
        h.update(p.read_bytes())
    for p in sorted((ROOT / "evals" / "rubrics").glob("*.md")):
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def _load_index(root: Path) -> list[dict]:
    p = root / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


def _save_index(root: Path, idx: list[dict]):
    (root / "index.json").write_text(json.dumps(idx, indent=1))


def previous_run(root: Path, exclude: str | None = None) -> dict | None:
    for entry in reversed(_load_index(root)):
        if entry["id"] != exclude and (root / entry["id"] / "results.json").exists():
            return json.loads((root / entry["id"] / "results.json").read_text())
    return None


def _item_key(it):
    return f"{it['scene_id']}::{it['style']}"


def run_suite(out_root="runs", styles=None, judge="auto", model=None, include_holdout=True,
              pairwise=True, timeout=300.0, label=None, params_by_style=None, api_key=None,
              progress=None, determinism=True, scenes=None) -> Path:
    root = Path(out_root)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    styles = styles or available()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = label or stamp
    run_id = f"{stamp}_{label}" if label != stamp else stamp
    run_dir = root / run_id
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    (run_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    say = progress or (lambda msg: print(msg, flush=True))

    judge_obj = None
    if judge != "none":
        judge_obj = make_judge(judge, model=model, api_key=api_key, cache_path=root / "judge_cache.json")
    rubrics = {}
    for s in styles:
        try:
            rubrics[s] = load_rubric(s)
        except FileNotFoundError:
            rubrics[s] = None

    scene_list = scenes or suite_scenes(include_holdout)
    jobs = [(Path(p), split, s) for p, split in scene_list for s in styles]
    say(f"run {run_id}: {len(jobs)} renders ({len(styles)} styles x {len(scene_list)} scenes), judge={judge_obj.name if judge_obj else 'none'}")

    def do(job):
        path, split, style = job
        from ..render import render_to_file

        scene = load_scene(path)
        name = f"{path.stem}__{style}"
        params = (params_by_style or {}).get(style)
        rec = render_to_file(scene, style, scene.seed, run_dir / "renders", params, name=name,
                             timeout=timeout, raise_on_error=False)
        item = {"scene_id": path.stem, "scene_title": scene.title, "split": split, "style": style,
                "seed": scene.seed, "kinds": sorted(set(scene.kinds)), "png": None, "ok": rec.ok,
                "error": rec.error, "seconds": rec.seconds}
        if not rec.ok:
            item["technical"] = {"passed": False, "checks": [{"name": "renders", "passed": False, "detail": rec.error}]}
            say(f"  FAIL {name}: {rec.error.splitlines()[0] if rec.error else ''}")
            return item
        item["png"] = str(Path(rec.png).relative_to(run_dir))
        thumbnail(rec.png, run_dir / "thumbs" / f"{name}.png", 360)
        item["thumb"] = f"thumbs/{name}.png"
        item["pixel_hash"] = metrics.pixel_hash(rec.png)
        item["technical"] = run_checks(rec.png, scene, style, seed=scene.seed, params=rec.params,
                                       seconds=rec.seconds, determinism=determinism)
        if judge_obj is not None:
            try:
                if judge_obj.is_mock:
                    item["fidelity"] = {"score": None, "note": "mock judge cannot see subjects"}
                else:
                    item["fidelity"] = score_subjects(scene, judge_obj.describe(rec.png))
                if rubrics.get(style):
                    item["style_score"] = judge_obj.score_style(rec.png, rubrics[style])
            except JudgeError as exc:
                item["judge_error"] = str(exc)
        t = item["technical"]
        say(f"  {'ok  ' if t['passed'] else 'WARN'} {name} {rec.seconds:.1f}s "
            f"fid={item.get('fidelity', {}).get('score')} style={item.get('style_score', {}).get('mean')}")
        return item

    workers = int(os.environ.get("PAINT_WORKERS", min(4, os.cpu_count() or 1)))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        items = list(ex.map(do, jobs))

    prev = previous_run(root, exclude=run_id)
    prev_items = {_item_key(it): it for it in (prev or {}).get("items", [])}
    for it in items:
        it["diff"] = _diff_item(it, prev_items.get(_item_key(it)), run_dir, root / prev["run_id"] if prev else None)

    pairs = []
    elo = EloTable(root / "elo.json")
    if pairwise and judge_obj is not None:
        pairs = _pairwise(judge_obj, items, prev, run_dir, root, label, rubrics, elo, say)
        elo.save()

    results = {
        "run_id": run_id, "label": label, "created": stamp, "styles": styles,
        "judge": {"name": judge_obj.name if judge_obj else "none", "model": judge_obj.model if judge_obj else None,
                  "mock": bool(judge_obj and judge_obj.is_mock)},
        "rubrics": {s: (r["hash"] if r else None) for s, r in rubrics.items()},
        "code": code_fingerprint(), "seconds": round(time.time() - t0, 1),
        "previous": prev["run_id"] if prev else None,
        "items": items, "pairs": pairs, "elo": elo.table(),
        "summary": summarize(items), "previous_summary": (prev or {}).get("summary"),
    }
    cal = ROOT / "evals" / "calibration" / "last_calibration.json"
    if cal.exists():
        results["calibration"] = json.loads(cal.read_text()).get("agreement")
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    idx = _load_index(root)
    idx.append({"id": run_id, "label": label, "created": stamp, "styles": styles, "judge": results["judge"]["name"]})
    _save_index(root, idx)

    from .report import write_report

    path = write_report(results, run_dir)
    say(f"done in {results['seconds']}s -> {path}")
    return path


def _diff_item(it, prev, run_dir, prev_dir):
    if not prev:
        return {"status": "new"}
    d = {"status": "same" if prev.get("pixel_hash") == it.get("pixel_hash") else "changed",
         "prev_run": prev_dir.name if prev_dir else None}
    if d["status"] == "changed" and it.get("png") and prev.get("png") and prev_dir:
        try:
            a = load_rgb(run_dir / it["png"])
            b = load_rgb(prev_dir / prev["png"])
            d["mean_abs_diff"] = round(metrics.mean_abs_diff(a, b), 4)
        except FileNotFoundError:
            pass
    for key, get in (("fidelity", lambda x: (x.get("fidelity") or {}).get("score")),
                     ("style", lambda x: (x.get("style_score") or {}).get("mean")),
                     ("seconds", lambda x: x.get("seconds"))):
        a, b = get(it), get(prev)
        if a is not None and b is not None:
            d[key] = round(a - b, 3)
    was, now = prev.get("technical", {}).get("passed"), it.get("technical", {}).get("passed")
    if was is not None and now is not None and was != now:
        d["technical"] = "fixed" if now else "regressed"
    if prev.get("thumb") and prev_dir:
        d["prev_thumb"] = os.path.relpath(prev_dir / prev["thumb"], run_dir)
    return d


def _pairwise(judge, items, prev, run_dir, root, label, rubrics, elo, say):
    pairs = []
    by = {(it["scene_id"], it["style"]): it for it in items if it.get("png")}
    prev_by = {(it["scene_id"], it["style"]): it for it in (prev or {}).get("items", []) if it.get("png")}
    prev_label = (prev or {}).get("label")
    tasks = []
    # (1) Regression: this version vs the previous run, same scene and style (holdout included, judged blind).
    for key, it in by.items():
        p = prev_by.get(key)
        if p and p.get("pixel_hash") != it.get("pixel_hash") and prev_label != label:
            tasks.append(("regression", it, run_dir / it["png"], root / prev["run_id"] / p["png"],
                          f"{it['style']}@{label}", f"{it['style']}@{prev_label}", rubrics.get(it["style"])))
    # (2) Cross-style: each style vs the screenprint baseline on the same scene (no rubric: overall quality).
    for (scene_id, style), it in by.items():
        base = by.get((scene_id, BASELINE_STYLE))
        if style != BASELINE_STYLE and base:
            tasks.append(("vs_baseline", it, run_dir / it["png"], run_dir / base["png"],
                          f"{style}@{label}", f"{BASELINE_STYLE}@{label}", None))

    def go(t):
        kind, it, a, b, pa, pb, rubric = t
        try:
            seed = int(hashlib.md5(f"{it['scene_id']}|{kind}|{it['style']}".encode()).hexdigest()[:6], 16)
            res = compare_pair(judge, a, b, rubric=rubric, seed=seed)
        except JudgeError as exc:
            return {"kind": kind, "scene_id": it["scene_id"], "error": str(exc)}
        res.update({"kind": kind, "scene_id": it["scene_id"], "split": it["split"], "player_a": pa, "player_b": pb,
                    "a": os.path.relpath(a, run_dir), "b": os.path.relpath(b, run_dir)})
        return res

    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(go, tasks):
            pairs.append(res)
            if "verdict" in res:
                elo.record(res["player_a"], res["player_b"], res["verdict"], f"{res['kind']}:{res['scene_id']}")
    if tasks:
        n_cons = sum(1 for p in pairs if p.get("consistent"))
        say(f"  pairwise: {len(tasks)} pairs, {n_cons} consistent after position swap")
    return pairs


def summarize(items) -> dict:
    out = {}
    for style in sorted({it["style"] for it in items}):
        for split in ("suite", "holdout"):
            rows = [it for it in items if it["style"] == style and it["split"] == split]
            if not rows:
                continue

            def mean(vals):
                vals = [v for v in vals if v is not None]
                return round(sum(vals) / len(vals), 3) if vals else None

            out.setdefault(style, {})[split] = {
                "n": len(rows),
                "technical_pass": round(sum(1 for r in rows if r.get("technical", {}).get("passed")) / len(rows), 3),
                "fidelity": mean([(r.get("fidelity") or {}).get("score") for r in rows]),
                "fidelity_subjects": mean([(r.get("fidelity") or {}).get("subjects_score") for r in rows]),
                "style": mean([(r.get("style_score") or {}).get("mean") for r in rows]),
                "seconds": mean([r.get("seconds") for r in rows]),
            }
    return out
