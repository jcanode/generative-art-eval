"""Command line interface.

    paint render scene.yaml --style screenprint --seed 7 --out out/
    paint sheet --style sumie --out sheet.png [scene ...]
    paint loop scene.yaml --style screenprint --critic claude --out runs/loop
    paint check out/x.png --scene scene.yaml --style screenprint
    paint eval --suite [--judge claude|mock] [--styles a,b] [--out runs/]
    paint compare a.png b.png --scene scene.yaml [--judge claude]
    paint calibrate --ratings evals/calibration/ratings.json
    paint play "a lighthouse at night with gulls" --styles screenprint,woodblock
    paint play --surprise --setting coast
    paint serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.guard import DEFAULT_TIMEOUT_S, RenderFailure

ROOT = Path(__file__).resolve().parent.parent


def _parse_params(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_render(a):
    from .render import render_to_file

    try:
        rec = render_to_file(a.scene, a.style, a.seed, a.out, _parse_params(a.param), timeout=a.timeout)
    except RenderFailure as exc:
        print(f"RENDER FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"{rec.png}  ({rec.seconds:.1f}s, peak {rec.peak_rss_mb:.0f} MB)")
    if not a.no_check:
        from .evals.technical import run_checks

        report = run_checks(rec.png, a.scene, a.style, seed=rec.seed, params=rec.params, seconds=rec.seconds,
                            determinism=not a.fast)
        for c in report["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
        return 0 if report["passed"] else 1
    return 0


def cmd_sheet(a):
    from .render import render_to_file
    from .sheet import contact_sheet

    scenes = a.scenes or sorted(str(p) for p in (ROOT / "evals" / "suite").glob("*.yaml"))
    tmp = Path(a.out).with_suffix("")
    paths, labels = [], []
    for s in scenes:
        rec = render_to_file(s, a.style, a.seed, tmp, _parse_params(a.param), timeout=a.timeout)
        paths.append(rec.png)
        labels.append(f"{Path(s).stem} {rec.seconds:.1f}s")
    print(contact_sheet(paths, a.out, labels=labels))
    return 0


def cmd_loop(a):
    from .loop.agent import run_loop

    result = run_loop(a.scene, a.style, seed=a.seed, out_dir=a.out, critic=a.critic,
                      max_iters=a.iterations, params=_parse_params(a.param), timeout=a.timeout,
                      model=a.model)
    print(json.dumps(result.summary(), indent=2))
    return 0


def cmd_check(a):
    from .evals.technical import run_checks

    report = run_checks(a.png, a.scene, a.style, seed=a.seed, determinism=not a.fast)
    for c in report["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return 0 if report["passed"] else 1


def cmd_eval(a):
    from .evals.suite import run_suite

    styles = a.styles.split(",") if a.styles else None
    report = run_suite(out_root=a.out, styles=styles, judge=a.judge, model=a.model,
                       include_holdout=not a.no_holdout, pairwise=not a.no_pairwise,
                       timeout=a.timeout, label=a.label, params_by_style=None)
    print(f"report: {report}")
    return 0


def cmd_compare(a):
    from .evals.judges import make_judge
    from .evals.pairwise import compare_pair

    judge = make_judge(a.judge, model=a.model)
    res = compare_pair(judge, a.a, a.b, style=a.style, seed=a.seed)
    print(json.dumps(res, indent=2))
    return 0


def cmd_calibrate(a):
    from .evals.calibration import calibrate, init_set

    if a.init:
        path = init_set(a.n)
        print(f"wrote {path}; rate each image 1-5 (edit the file or use `paint serve`), then run `paint calibrate`")
        return 0
    print(json.dumps(calibrate(a.ratings, judge=a.judge, model=a.model, out=a.out), indent=2))
    return 0


def cmd_play(a):
    """Freestyle: a prompt (or --surprise) -> scene -> render in one or more styles."""
    import yaml

    from . import playground as pg
    from .render import render_to_file
    from .scene.spec import load_scene

    if a.surprise or not a.prompt:
        scene, source = pg.surprise(a.seed, a.setting), "surprise"
    else:
        use = {"auto": None, "claude": True, "offline": False}[a.writer]
        scene, source = pg.from_prompt(" ".join(a.prompt), seed=a.seed or 0, use_claude=use, model=a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch.isalnum() else "_" for ch in scene["title"].lower())[:40].strip("_") or "scene"
    (out / f"{stem}.yaml").write_text(yaml.safe_dump(scene, sort_keys=False))
    print(f"scene ({source}): {out / (stem + '.yaml')}  objects: {', '.join(o['kind'] for o in scene['objects'])}")
    sc = load_scene(scene)
    for style in a.styles.split(","):
        rec = render_to_file(sc, style.strip(), None, out, name=f"{stem}__{style.strip()}", timeout=a.timeout, raise_on_error=False)
        print(f"  {style}: {rec.png if rec.ok else 'FAILED ' + (rec.error or '').splitlines()[0]}" + (f" ({rec.seconds:.1f}s)" if rec.ok else ""))
    return 0


def cmd_serve(a):
    from .site.server import serve

    serve(a.host, a.port)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="paint", description="Procedural code-painting harness and eval suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, scene=True):
        if scene:
            sp.add_argument("scene")
        sp.add_argument("--style", default="screenprint")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--param", "-p", action="append", help="style knob override key=value")
        sp.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)

    r = sub.add_parser("render", help="render one scene")
    common(r)
    r.add_argument("--out", default="out")
    r.add_argument("--no-check", action="store_true", help="skip technical checks")
    r.add_argument("--fast", action="store_true", help="skip the determinism re-render")
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("sheet", help="contact sheet of many scenes in one style")
    common(s, scene=False)
    s.add_argument("scenes", nargs="*")
    s.add_argument("--out", default="out/sheet.png")
    s.set_defaults(fn=cmd_sheet)

    lp = sub.add_parser("loop", help="render -> look -> critique -> revise, up to 4 times")
    common(lp)
    lp.add_argument("--out", default="runs/loop")
    lp.add_argument("--critic", default="auto", choices=["auto", "claude", "manual", "heuristic"],
                    help="auto = claude if ANTHROPIC_API_KEY is set, else manual")
    lp.add_argument("--iterations", type=int, default=4)
    lp.add_argument("--model", default=None)
    lp.set_defaults(fn=cmd_loop)

    c = sub.add_parser("check", help="technical checks on a PNG")
    c.add_argument("png")
    c.add_argument("--scene", required=True)
    c.add_argument("--style", required=True)
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--fast", action="store_true")
    c.set_defaults(fn=cmd_check)

    e = sub.add_parser("eval", help="run the regression suite and write an HTML report")
    e.add_argument("--suite", action="store_true", help="run the fixed suite in evals/suite (+ holdout)")
    e.add_argument("--styles", default=None)
    e.add_argument("--judge", default="auto", choices=["auto", "claude", "mock", "none"])
    e.add_argument("--model", default=None)
    e.add_argument("--out", default="runs")
    e.add_argument("--label", default=None)
    e.add_argument("--no-holdout", action="store_true")
    e.add_argument("--no-pairwise", action="store_true")
    e.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    e.set_defaults(fn=cmd_eval)

    cp = sub.add_parser("compare", help="blind pairwise preference between two PNGs")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--style", default=None)
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--judge", default="auto", choices=["auto", "claude", "mock"])
    cp.add_argument("--model", default=None)
    cp.set_defaults(fn=cmd_compare)

    cal = sub.add_parser("calibrate", help="agreement between the judge and your own ratings")
    cal.add_argument("--ratings", default=str(ROOT / "evals" / "calibration" / "ratings.json"))
    cal.add_argument("--judge", default="auto", choices=["auto", "claude", "mock"])
    cal.add_argument("--model", default=None)
    cal.add_argument("--out", default=None)
    cal.add_argument("--init", action="store_true", help="render a ~20-image calibration set with empty ratings")
    cal.add_argument("-n", type=int, default=20)
    cal.set_defaults(fn=cmd_calibrate)

    pl = sub.add_parser("play", help="freestyle: describe a scene (or --surprise) and render it")
    pl.add_argument("prompt", nargs="*")
    pl.add_argument("--surprise", action="store_true")
    pl.add_argument("--setting", default=None, help="surprise setting: coast, countryside, mountains, night, still life, garden, rain")
    pl.add_argument("--styles", default="screenprint,sumie,impasto")
    pl.add_argument("--seed", type=int, default=None)
    pl.add_argument("--writer", default="auto", choices=["auto", "claude", "offline"], help="who turns the prompt into a scene")
    pl.add_argument("--model", default=None)
    pl.add_argument("--out", default="out/play")
    pl.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    pl.set_defaults(fn=cmd_play)

    sv = sub.add_parser("serve", help="local web studio")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
