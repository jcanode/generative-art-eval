"""Human calibration: how well does the judge agree with you?

``evals/calibration/ratings.json`` holds ~20 images you rate yourself (1-5):

    {"items": [{"image": "evals/calibration/images/c01.png", "rating": 4, "notes": "..."}, ...]}

``paint calibrate --init`` renders a varied set (styles, scenes, and some
deliberately degraded knob settings so the set spans good to bad) and writes
the file with empty ratings. Fill them in by hand or in the web studio. Then
``paint calibrate`` asks the judge for its own blind 1-5 rating of each image
and reports agreement.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = ROOT / "evals" / "calibration"


def load_ratings(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("items", [])


def save_ratings(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"items": items}, indent=2))


def agreement(human: list[float], judge: list[float]) -> dict:
    h, j = np.asarray(human, float), np.asarray(judge, float)
    n = len(h)
    out = {"n": n}
    if n < 3:
        out["note"] = "need at least 3 rated images"
        return out
    rho, p_rho = stats.spearmanr(h, j)
    tau, p_tau = stats.kendalltau(h, j)
    r, _ = stats.pearsonr(h, j) if h.std() > 0 and j.std() > 0 else (float("nan"), None)
    out.update({
        "spearman": round(float(rho), 3), "spearman_p": round(float(p_rho), 4),
        "kendall_tau": round(float(tau), 3),
        "pearson": round(float(r), 3),
        "exact_agreement": round(float((h == j).mean()), 3),
        "within_one": round(float((np.abs(h - j) <= 1).mean()), 3),
        "mae": round(float(np.abs(h - j).mean()), 3),
        "judge_bias": round(float((j - h).mean()), 3),
        "quadratic_kappa": round(quadratic_kappa(h.astype(int), j.astype(int)), 3),
    })
    # Pairwise concordance: for pairs you rated differently, how often does the judge order them the same way?
    agree = total = 0
    for a, b in combinations(range(n), 2):
        if h[a] == h[b]:
            continue
        total += 1
        agree += (j[a] - j[b]) * (h[a] - h[b]) > 0
    out["pair_concordance"] = round(agree / total, 3) if total else None
    out["verdict"] = _verdict(out)
    return out


def quadratic_kappa(a, b, lo=1, hi=5) -> float:
    k = hi - lo + 1
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x - lo, y - lo] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else float("nan")


def _verdict(a: dict) -> str:
    rho, kappa = a.get("spearman", 0), a.get("quadratic_kappa", 0)
    if np.isnan(rho):
        return "undefined (no variance in ratings)"
    if rho >= 0.7 and kappa >= 0.6:
        return "trustworthy: judge ranks images much like you do"
    if rho >= 0.4:
        return "partly trustworthy: use for large differences, not fine-grained ones"
    return "not trustworthy on this set: don't make decisions from judge scores alone"


def calibrate(ratings_path, judge="auto", model=None, out=None, api_key=None) -> dict:
    from .judges import make_judge

    items = [it for it in load_ratings(ratings_path) if it.get("rating") is not None]
    if not items:
        return {"error": f"no rated images in {ratings_path}; run `paint calibrate --init` and add ratings"}
    j = make_judge(judge, model=model, api_key=api_key)
    rows = []
    for it in items:
        img = ROOT / it["image"] if not Path(it["image"]).is_absolute() else Path(it["image"])
        r = j.rate_overall(img)
        rows.append({"image": it["image"], "human": int(it["rating"]), "judge": int(r["score"]), "reason": r.get("reason", "")})
    result = {"judge": j.name, "model": j.model, "mock": j.is_mock,
              "agreement": agreement([r["human"] for r in rows], [r["judge"] for r in rows]), "rows": rows}
    path = Path(out) if out else CAL_DIR / "last_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


DEGRADED = {
    "screenprint": [{"misregistration": 6.0, "edge_roughness": 3.0, "ink_texture": 1.0}, {"inks": 2, "shade_strength": 0.0}],
    "sumie": [],
    "impasto": [],
}


def init_set(n_target: int = 20, out_dir: Path | None = None, styles=None) -> Path:
    """Render a varied calibration set and write ratings.json with empty ratings."""
    from ..render import render_to_file
    from ..styles import available, get_style

    out_dir = out_dir or CAL_DIR / "images"
    styles = styles or available()
    scenes = sorted((ROOT / "evals" / "suite").glob("*.yaml"))
    i = 0
    plan: list = []
    for si, scene in enumerate(scenes):
        style = styles[si % len(styles)]
        plan.append((scene, style, None, None))
    for style in styles:
        mod = get_style(style)
        # Deliberately bad settings so the set spans weak to strong work.
        degraded = DEGRADED.get(style) or [{k: lo for k, (lo, hi, _) in mod.KNOBS.items()},
                                           {k: hi for k, (lo, hi, _) in mod.KNOBS.items()}]
        for d in degraded:
            plan.append((scenes[(len(plan) * 3) % len(scenes)], style, 1000 + len(plan), d))
    while len(plan) < n_target:
        i += 1
        plan.append((scenes[(i * 7) % len(scenes)], styles[i % len(styles)], 2000 + i, None))
    items = []
    for k, (scene, style, seed, params) in enumerate(plan[:n_target]):
        rec = render_to_file(scene, style, seed, out_dir, params, name=f"c{k + 1:02d}", raise_on_error=False)
        if rec.ok:
            items.append({"image": str(Path(rec.png).resolve().relative_to(ROOT)), "rating": None, "notes": ""})
    path = CAL_DIR / "ratings.json"
    existing = {it["image"]: it for it in load_ratings(path)}
    for it in items:
        if it["image"] in existing:
            it.update({k: existing[it["image"]][k] for k in ("rating", "notes")})
    save_ratings(path, items)
    return path
