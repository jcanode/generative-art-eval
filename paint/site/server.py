"""Local web studio: a small stdlib HTTP server + single-page UI.

    paint serve --port 8000    ->  http://127.0.0.1:8000

What it does:
* takes an Anthropic API key (held in server memory only; never written to disk or logged)
* renders scenes in any style with knob sliders, and runs technical checks
* runs the render -> look -> critique -> revise loop (Claude critic, heuristic, or you as the critic)
* runs the eval suite and links the HTML reports
* accepts uploaded images and judges them blind (subjects, rubric, pairwise)
* collects your 1-5 ratings for judge calibration

Binds to 127.0.0.1 by default. Long jobs run in background threads and are polled.
"""
from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "runs"
UPLOADS = RUNS / "uploads"
STUDIO = RUNS / "studio"
SERVE_ROOTS = [RUNS, ROOT / "evals" / "calibration"]
MAX_UPLOAD = 20 * 1024 * 1024


class KeyStore:
    """The API key lives only in this process's memory."""

    def __init__(self):
        self._key = os.environ.get("ANTHROPIC_API_KEY") or None
        self.source = "env" if self._key else None
        self.lock = threading.Lock()

    def set(self, key: str | None):
        with self.lock:
            self._key = (key or "").strip() or None
            self.source = "ui" if self._key else None

    def get(self) -> str | None:
        with self.lock:
            return self._key

    def masked(self) -> str | None:
        k = self.get()
        return None if not k else f"{k[:7]}…{k[-4:]}"


KEYS = KeyStore()
SETTINGS = {"model": os.environ.get("PAINT_JUDGE_MODEL", "claude-opus-5")}


class Jobs:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, kind: str, fn, *args, **kwargs) -> str:
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "kind": kind, "status": "running", "log": [], "result": None, "error": None,
               "started": time.time()}
        with self.lock:
            self.jobs[jid] = job

        def log(msg):
            with self.lock:
                job["log"].append(str(msg))
                job["log"] = job["log"][-400:]

        def run():
            try:
                job["result"] = fn(*args, progress=log, **kwargs)
                job["status"] = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                log(traceback.format_exc(limit=3))
            job["finished"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            return json.loads(json.dumps(j, default=str)) if j else None


JOBS = Jobs()


def _url_for(path: Path) -> str | None:
    p = Path(path).resolve()
    for root in SERVE_ROOTS:
        try:
            rel = p.relative_to(root.resolve())
            return f"/files/{root.name}/{rel.as_posix()}"
        except ValueError:
            continue
    return None


def _resolve_file(url_path: str) -> Path | None:
    parts = unquote(url_path).split("/", 3)  # ['', 'files', '<root>', 'rest']
    if len(parts) < 4:
        return None
    root = next((r for r in SERVE_ROOTS if r.name == parts[2]), None)
    if root is None:
        return None
    target = (root / parts[3]).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None  # path traversal
    return target if target.is_file() else None


# ---------------------------------------------------------------------------
# API implementations
# ---------------------------------------------------------------------------

def api_state(_):
    from ..styles import available, get_style

    styles = {}
    for s in available():
        m = get_style(s)
        styles[s] = {"defaults": m.DEFAULTS, "knobs": {k: {"min": lo, "max": hi, "about": d} for k, (lo, hi, d) in m.KNOBS.items()},
                     "rules": getattr(m, "RULES", {})}
    scenes = []
    for folder, split in ((ROOT / "evals" / "suite", "suite"), (ROOT / "evals" / "holdout", "holdout"), (ROOT / "scenes", "scenes"),
                          (ROOT / "scenes" / "playground", "playground")):
        for p in sorted(folder.glob("*.y*ml")) + sorted(folder.glob("*.json")):
            scenes.append({"path": str(p.relative_to(ROOT)), "name": p.stem, "split": split, "text": p.read_text()})
    return {"key": {"set": KEYS.get() is not None, "masked": KEYS.masked(), "source": KEYS.source},
            "model": SETTINGS["model"], "styles": styles, "scenes": scenes}


def api_key(body):
    if body.get("clear"):
        KEYS.set(None)
    else:
        key = (body.get("api_key") or "").strip()
        if not key:
            raise ValueError("api_key is empty")
        KEYS.set(key)
    if body.get("model"):
        SETTINGS["model"] = body["model"]
    return {"set": KEYS.get() is not None, "masked": KEYS.masked(), "model": SETTINGS["model"]}


def api_key_test(_):
    key = KEYS.get()
    if not key:
        return {"ok": False, "detail": "no key set"}
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key, timeout=20.0, max_retries=0)
        m = client.models.retrieve(SETTINGS["model"])
        return {"ok": True, "detail": f"key works; model {m.id} available"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _scene_from(body):
    from ..scene.spec import load_scene

    if body.get("scene_text"):
        import yaml

        data = yaml.safe_load(body["scene_text"])
        scene = load_scene(data)
        name = body.get("scene_name") or "custom"
        scene.source = str(STUDIO / f"{name}.yaml")
        return scene
    path = ROOT / body["scene"]
    return load_scene(path)


def api_render(body):
    from ..evals.technical import run_checks
    from ..render import render_to_file

    scene = _scene_from(body)
    style = body.get("style", "screenprint")
    seed = int(body["seed"]) if body.get("seed") not in (None, "") else scene.seed
    params = body.get("params") or {}
    out = STUDIO / time.strftime("%Y%m%d")
    name = f"{Path(scene.source or 'scene').stem}__{style}__s{seed}__{uuid.uuid4().hex[:6]}"
    rec = render_to_file(scene, style, seed, out, params, name=name, raise_on_error=False)
    if not rec.ok:
        return {"ok": False, "error": rec.error}
    checks = run_checks(rec.png, scene, style, seed=seed, params=rec.params, seconds=rec.seconds,
                        determinism=bool(body.get("determinism", False)))
    return {"ok": True, "image": _url_for(Path(rec.png)), "path": str(Path(rec.png).relative_to(ROOT)),
            "seconds": rec.seconds, "checks": checks, "params": rec.params, "info": rec.info}


def _judge(kind=None):
    from ..evals.judges import make_judge

    kind = kind or ("claude" if KEYS.get() else "mock")
    return make_judge(kind, model=SETTINGS["model"], api_key=KEYS.get(), cache_path=RUNS / "judge_cache.json")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _local_path(p: str) -> Path:
    """Accept a repo-relative path or a /files/ URL; refuse anything outside the served roots."""
    if p.startswith("/files/"):
        t = _resolve_file(p)
    else:
        t = (ROOT / p).resolve()
        if not any(_inside(t, r) for r in SERVE_ROOTS):
            t = None
    if t is None or not t.exists():
        raise ValueError(f"image not found or not allowed: {p}")
    return t


def api_judge(body):
    from ..evals.fidelity import score_subjects
    from ..evals.judges import load_rubric
    from ..scene.spec import load_scene

    judge = _judge(body.get("judge"))
    img = _local_path(body["image"])
    out = {"judge": judge.name, "mock": judge.is_mock, "model": judge.model}
    desc = judge.describe(img)
    out["describe"] = desc
    if body.get("scene") and not judge.is_mock:
        out["fidelity"] = score_subjects(load_scene(ROOT / body["scene"]), desc)
    if body.get("rubric"):
        out["style"] = judge.score_style(img, load_rubric(body["rubric"]))
    out["overall"] = judge.rate_overall(img)
    return out


def api_compare(body):
    from ..evals.judges import load_rubric
    from ..evals.pairwise import EloTable, compare_pair

    judge = _judge(body.get("judge"))
    a, b = _local_path(body["a"]), _local_path(body["b"])
    rubric = load_rubric(body["rubric"]) if body.get("rubric") else None
    res = compare_pair(judge, a, b, rubric=rubric, seed=int(time.time()))
    if body.get("elo_a") and body.get("elo_b"):
        elo = EloTable(RUNS / "elo.json")
        elo.record(body["elo_a"], body["elo_b"], res["verdict"], "studio")
        elo.save()
        res["elo"] = elo.table()
    res.update({"judge": judge.name, "mock": judge.is_mock})
    return res


def api_upload(body):
    """JSON body: {files: [{name, data_base64}]} -> saved PNGs (re-encoded, metadata stripped)."""
    from io import BytesIO

    from PIL import Image

    UPLOADS.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in body.get("files", [])[:24]:
        raw = f.get("data", "")
        if "," in raw[:100]:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{f.get('name')}: not base64") from exc
        if len(data) > MAX_UPLOAD:
            raise ValueError(f"{f.get('name')}: larger than {MAX_UPLOAD // 1024 // 1024} MB")
        im = Image.open(BytesIO(data)).convert("RGB")
        if max(im.size) > 4096:
            im.thumbnail((4096, 4096))
        stem = "".join(ch for ch in Path(f.get("name", "upload")).stem if ch.isalnum() or ch in "-_")[:40] or "upload"
        out = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}_{stem}.png"
        im.save(out, format="PNG")
        saved.append({"name": f.get("name"), "image": _url_for(out), "path": str(out.relative_to(ROOT)),
                      "size": list(im.size)})
    return {"files": saved}


def api_uploads(_):
    items = list(UPLOADS.glob("*.png")) if UPLOADS.exists() else []
    items += list(STUDIO.rglob("*.png")) if STUDIO.exists() else []
    items.sort(key=lambda p: -p.stat().st_mtime)
    return {"files": [{"image": _url_for(p), "path": str(p.relative_to(ROOT)),
                       "name": ("upload: " if UPLOADS in p.parents else "render: ") + p.name} for p in items[:200]]}


def api_loop_start(body):
    from ..loop.agent import run_loop

    critic = body.get("critic") or ("claude" if KEYS.get() else "heuristic")
    if critic == "claude" and not KEYS.get():
        raise ValueError("the Claude critic needs an API key (Settings)")
    scene = ROOT / body["scene"]
    jid = JOBS.start("loop", run_loop, str(scene), body.get("style", "screenprint"),
                     seed=int(body["seed"]) if body.get("seed") not in (None, "") else None,
                     out_dir=str(RUNS / "loop"), critic=critic, max_iters=int(body.get("iterations", 4)),
                     params=body.get("params") or None, model=SETTINGS["model"], api_key=KEYS.get())
    return {"job": jid}


def _loop_view(run_dir: Path):
    state = json.loads((run_dir / "loop.json").read_text()) if (run_dir / "loop.json").exists() else {"versions": []}
    versions = []
    for v in state.get("versions", []):
        vd = run_dir / f"v{v['version']}"
        tmpl = vd / "critique.template.json"
        versions.append({**{k: v.get(k) for k in ("version", "seconds", "technical_passed", "critique", "params", "error")},
                         "image": _url_for(vd / "image.png") if (vd / "image.png").exists() else None,
                         "template": json.loads(tmpl.read_text()) if tmpl.exists() and not v.get("critique") else None})
    return {"run_dir": str(run_dir.relative_to(ROOT)), "best": state.get("best"), "versions": versions,
            "style": state.get("style"), "scene": state.get("scene"), "critic": state.get("critic")}


def api_loops(_):
    root = RUNS / "loop"
    dirs = sorted((p for p in root.glob("*") if (p / "loop.json").exists()), key=lambda p: -p.stat().st_mtime) if root.exists() else []
    return {"loops": [_loop_view(d) for d in dirs[:30]]}


def api_loop_critique(body):
    """You are the critic: write vN/critique.json for a manual loop and continue it."""
    run_dir = (ROOT / body["run_dir"]).resolve()
    run_dir.relative_to((RUNS / "loop").resolve())
    state = json.loads((run_dir / "loop.json").read_text())
    v = int(body["version"])
    crit = body["critique"]
    crit.setdefault("adjustments", {})
    crit.setdefault("object_edits", [])
    (run_dir / f"v{v}" / "critique.json").write_text(json.dumps(crit, indent=2))
    from ..loop.agent import run_loop

    jid = JOBS.start("loop", run_loop, state["scene"], state["style"], seed=state["seed"],
                     out_dir=str(run_dir.parent), critic="manual")
    return {"job": jid}


def api_eval(body):
    from ..evals.suite import run_suite

    judge = body.get("judge") or ("claude" if KEYS.get() else "mock")
    if judge == "claude" and not KEYS.get():
        raise ValueError("the Claude judge needs an API key (Settings)")

    def job(progress):
        path = run_suite(out_root=str(RUNS), styles=body.get("styles") or None, judge=judge, model=SETTINGS["model"],
                         include_holdout=body.get("holdout", True), pairwise=body.get("pairwise", True),
                         label=body.get("label") or None, api_key=KEYS.get(), progress=progress)
        return {"report": _url_for(path)}

    return {"job": JOBS.start("eval", lambda progress: job(progress))}


def api_runs(_):
    idx = json.loads((RUNS / "index.json").read_text()) if (RUNS / "index.json").exists() else []
    out = []
    for e in reversed(idx[-50:]):
        rep = RUNS / e["id"] / "report.html"
        res = RUNS / e["id"] / "results.json"
        summary = json.loads(res.read_text()).get("summary") if res.exists() else None
        out.append({**e, "report": _url_for(rep) if rep.exists() else None, "summary": summary})
    elo = json.loads((RUNS / "elo.json").read_text()) if (RUNS / "elo.json").exists() else {}
    return {"runs": out, "elo": sorted(({"player": k, "elo": round(v, 1)} for k, v in elo.get("ratings", {}).items()),
                                       key=lambda r: -r["elo"])}


def api_calibration(_):
    from ..evals.calibration import CAL_DIR, load_ratings

    items = load_ratings(CAL_DIR / "ratings.json")
    for it in items:
        it["url"] = _url_for(ROOT / it["image"])
    last = CAL_DIR / "last_calibration.json"
    return {"items": items, "last": json.loads(last.read_text()) if last.exists() else None}


def api_calibration_rate(body):
    from ..evals.calibration import CAL_DIR, load_ratings, save_ratings

    path = CAL_DIR / "ratings.json"
    items = load_ratings(path)
    target = body["image"]
    found = False
    for it in items:
        if it["image"] == target:
            r = body.get("rating")
            it["rating"] = int(r) if r not in (None, "") else None
            if "notes" in body:
                it["notes"] = body["notes"]
            found = True
    if not found and body.get("add"):
        items.append({"image": target, "rating": body.get("rating"), "notes": body.get("notes", "")})
    save_ratings(path, items)
    return {"ok": True, "rated": sum(1 for it in items if it.get("rating") is not None), "total": len(items)}


def api_calibration_init(body):
    from ..evals.calibration import init_set

    def job(progress):
        progress("rendering calibration set…")
        return {"path": str(init_set(int(body.get("n", 20))).relative_to(ROOT))}

    return {"job": JOBS.start("calibration-init", job)}


def api_calibration_run(body):
    from ..evals.calibration import CAL_DIR, calibrate

    judge = body.get("judge") or ("claude" if KEYS.get() else "mock")

    def job(progress):
        progress(f"asking the {judge} judge to rate each image blind…")
        return calibrate(CAL_DIR / "ratings.json", judge=judge, model=SETTINGS["model"], api_key=KEYS.get())

    return {"job": JOBS.start("calibration", job)}


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------

PLAY = RUNS / "playground"


def api_play_meta(_):
    from .. import playground as pg
    from ..scene.shapes import KINDS

    centred = {"sun", "moon", "cloud", "bird", "mountain", "stars", "rain"}
    regions = {"sea", "field", "table", "river", "hills"}
    return {"kinds": sorted(KINDS), "reach": pg.REACH, "centred": sorted(centred), "regions": sorted(regions),
            "settings": sorted(pg.SETTINGS), "palettes": pg.PALETTES,
            "variants": {"boat": ["sail", "fishing", "row"], "moon": ["crescent", "full"], "bird": ["flying", "perched"],
                         "fruit": ["apple", "pear", "orange"]}}


def api_play_surprise(body):
    from .. import playground as pg

    seed = body.get("seed")
    return {"scene": pg.surprise(int(seed) if seed not in (None, "") else None, body.get("setting") or None,
                                 int(body.get("width", 1024)), int(body.get("height", 768)))}


def api_play_prompt(body):
    from .. import playground as pg

    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("describe a scene first")
    use_claude = body.get("use_claude")
    if use_claude and not KEYS.get():
        raise ValueError("the Claude scene writer needs an API key (Settings)")
    scene, source = pg.from_prompt(text, api_key=KEYS.get(), model=SETTINGS["model"], seed=int(body.get("seed") or 0),
                                   use_claude=(bool(KEYS.get()) if use_claude is None else bool(use_claude)),
                                   width=int(body.get("width", 1024)), height=int(body.get("height", 768)))
    return {"scene": scene, "source": source}


def _play_render_one(scene_dict, style, seed, preview, params):
    from .. import playground as pg
    from ..render import render_to_file
    from ..scene.spec import load_scene

    d = pg.preview_scene(scene_dict, 360) if preview else scene_dict
    scene = load_scene(d)
    out = PLAY / ("preview" if preview else time.strftime("%Y%m%d"))
    name = f"{style}__{uuid.uuid4().hex[:8]}"
    rec = render_to_file(scene, style, seed, out, params or None, name=name, timeout=60 if preview else 300,
                         raise_on_error=False)
    return {"style": style, "ok": rec.ok, "error": (rec.error or "").splitlines()[0] if rec.error else None,
            "image": _url_for(Path(rec.png)) if rec.ok else None, "seconds": rec.seconds,
            "path": str(Path(rec.png).relative_to(ROOT)) if rec.ok else None}


def api_play_render(body):
    from concurrent.futures import ThreadPoolExecutor

    from ..scene.spec import load_scene
    from ..styles import available

    scene = body.get("scene") or {}
    load_scene(scene)  # validate early: unknown kinds etc. become a 400
    styles = [s for s in (body.get("styles") or ["screenprint"]) if s in available()][:8]
    seed = int(body["seed"]) if body.get("seed") not in (None, "") else int(scene.get("seed", 0))
    preview = bool(body.get("preview", False))
    if preview:  # previews are disposable: keep the folder small
        old = sorted((PLAY / "preview").glob("*.png"), key=lambda p: p.stat().st_mtime) if (PLAY / "preview").exists() else []
        for p in old[:-80]:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
    workers = max(1, min(len(styles), int(os.environ.get("PAINT_WORKERS", os.cpu_count() or 2))))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda s: _play_render_one(scene, s, seed, preview, (body.get("params") or {}).get(s)), styles))
    return {"results": results, "preview": preview}


def api_play_save(body):
    import re

    import yaml

    from ..scene.spec import load_scene

    scene = body["scene"]
    load_scene(scene)
    name = re.sub(r"[^a-z0-9_-]+", "_", (body.get("name") or scene.get("title") or "scene").lower()).strip("_")[:50] or "scene"
    folder = ROOT / "scenes" / "playground"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(scene, sort_keys=False))
    return {"path": str(path.relative_to(ROOT))}


GET_ROUTES = {"/api/state": api_state, "/api/runs": api_runs, "/api/calibration": api_calibration,
              "/api/uploads": api_uploads, "/api/loops": api_loops, "/api/play/meta": api_play_meta}
POST_ROUTES = {"/api/key": api_key, "/api/key/test": api_key_test, "/api/render": api_render,
               "/api/judge": api_judge, "/api/compare": api_compare, "/api/upload": api_upload,
               "/api/loop": api_loop_start, "/api/loop/critique": api_loop_critique, "/api/eval": api_eval,
               "/api/calibration/rate": api_calibration_rate, "/api/calibration/init": api_calibration_init,
               "/api/calibration/run": api_calibration_run, "/api/play/surprise": api_play_surprise,
               "/api/play/prompt": api_play_prompt, "/api/play/render": api_play_render, "/api/play/save": api_play_save}


class Handler(BaseHTTPRequestHandler):
    server_version = "PaintStudio/0.1"

    def log_message(self, fmt, *args):  # keep request bodies (and keys) out of logs
        if os.environ.get("PAINT_HTTP_LOG"):
            super().log_message(fmt, *args)

    def _send(self, code, body: bytes, ctype="application/json", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=str).encode())

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path.startswith("/static/"):
            p = (STATIC / u.path[len("/static/"):]).resolve()
            if p.is_file() and _inside(p, STATIC):
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            return self._json(404, {"error": "not found"})
        if u.path.startswith("/files/"):
            p = _resolve_file(u.path)
            if not p:
                return self._json(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream", cache=True)
        if u.path.startswith("/api/jobs/"):
            j = JOBS.get(u.path.rsplit("/", 1)[-1])
            return self._json(200 if j else 404, j or {"error": "no such job"})
        fn = GET_ROUTES.get(u.path)
        if not fn:
            return self._json(404, {"error": "not found"})
        try:
            return self._json(200, fn(parse_qs(u.query)))
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        u = urlparse(self.path)
        fn = POST_ROUTES.get(u.path)
        if not fn:
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD * 2:
            return self._json(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            return self._json(200, fn(body))
        except (ValueError, KeyError) as exc:
            return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "127.0.0.1", port: int = 8000):
    for d in (RUNS, UPLOADS, STUDIO):
        d.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"paint studio on http://{host}:{port}  (API key: {'from env' if KEYS.get() else 'not set - add it in Settings'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
