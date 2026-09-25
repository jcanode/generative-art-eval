"""HTML report for an arena run: leaderboard, per-task gallery, every version with the model's own critique."""
from __future__ import annotations

import html
import os
from pathlib import Path

from ..evals.report import CSS


def _e(x):
    return html.escape(str(x))


def _f(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _pct(v):
    return "–" if v is None else f"{v:.0%}"


EXTRA_CSS = """
.lb td.player{font-weight:600}.strip{display:flex;gap:6px;overflow-x:auto;margin-top:6px}
.strip figure{margin:0;min-width:110px;max-width:130px}.strip img{width:100%;border-radius:3px;border:1px solid var(--line)}
.strip figcaption{font-size:11px;color:var(--muted)}.err{font-size:11px;color:var(--bad);white-space:pre-wrap;max-height:90px;overflow:auto}
pre.reply{white-space:pre-wrap;font-size:12px;max-height:280px;overflow:auto;background:var(--chip);padding:8px;border-radius:6px}
"""


def _versions(e, run_dir: Path) -> str:
    s = e.get("session")
    if not s:
        return ""
    cells = []
    for v in s["versions"]:
        vdir = Path(v["program"]).parent
        rel = lambda p: os.path.relpath(p, run_dir)  # noqa: E731
        head = f"v{v['n']}" + (" ★" if v["n"] == s.get("final") else "") + (f" · self {v['self_score']:.0f}/10" if v.get("self_score") else "")
        if v["ok"]:
            body = f"<a href='{_e(rel(v['image']))}'><img loading='lazy' src='{_e(rel(v['image']))}' alt='v{v['n']}'></a>"
        else:
            body = f"<div class='err'>{_e(v['stage'])}: {_e((v.get('error') or '')[:300])}</div>"
        links = f"<a href='{_e(rel(vdir / 'program.py'))}'>code</a> · <a href='{_e(rel(vdir / 'reply.md'))}'>reply</a>"
        cells.append(f"<figure>{body}<figcaption>{_e(head)}<br>{v['seconds']}s · ${v['cost']:.3f}<br>{links}</figcaption></figure>")
    return f"<div class='strip'>{''.join(cells)}</div>"


def _cell(e, run_dir) -> str:
    if not e:
        return "<td class='small'>–</td>"
    ev = e.get("eval", {})
    s = e.get("session", {})
    if not e.get("image"):
        msg = s.get("message") or "no version produced an image"
        return f"<td class='cell'><div class='bad'>no image</div><div class='small'>{_e(msg)}</div>{_versions(e, run_dir)}</td>"
    fid = ev.get("fidelity") or {}
    st = ev.get("style_score") or {}
    chips = [f"<span class='chip {'ok' if ev.get('technical', {}).get('passed') else 'bad'}'>tech</span>"]
    if fid.get("score") is not None:
        chips.append(f"<span class='chip'>subjects {_f(fid['score'])}</span>")
    if st:
        first = (e.get("first_style") or {}).get("mean")
        delta = f" ({'+' if st['mean'] - first >= 0 else ''}{st['mean'] - first:.1f} vs first)" if first is not None else ""
        chips.append(f"<span class='chip'>style {_f(st.get('mean'))}{_e(delta)}</span>")
    if s:
        chips.append(f"<span class='chip'>v{s.get('final')}/{len(s.get('versions', []))}</span>")
        if s.get("deterministic") is False:
            chips.append("<span class='chip bad'>not deterministic</span>")
    det = []
    if fid.get("judge_objects") is not None:
        det.append(f"<b>Judge saw:</b> {_e(', '.join(o['name'] for o in fid.get('judge_objects', [])) or '–')}<br>"
                   f"<b class='bad'>Missing:</b> {_e(', '.join(fid.get('missing', [])) or '–')}")
    if st:
        det.append("<ul>" + "".join(f"<li>{_e(x['criterion'])}: <b>{_e(x['score'])}</b> – {_e(x['reason'])}</li>" for x in st.get("scores", [])) + "</ul>")
    return (f"<td class='cell'><a href='{_e(e['image'])}'><img loading='lazy' src='{_e(e.get('thumb', e['image']))}' alt='{_e(e['player'])}'></a>"
            f"<div class='chips'>{''.join(chips)}</div>"
            + (f"<details><summary>scores</summary>{''.join(det)}</details>" if det else "")
            + (f"<details><summary>all versions</summary>{_versions(e, run_dir)}</details>" if s else "") + "</td>")


def write_arena_report(results: dict, run_dir: Path) -> Path:
    lb = results["leaderboard"]
    players = [r["player"] for r in lb]
    entries = results["entries"]
    by = {(e["task"], e["player"]): e for e in entries}
    j = results["judge"]
    banner = ("<div class='banner'><b>Mock judge.</b> No API key, so fidelity, style and pairwise numbers are placeholders "
              "from pixel statistics. Rendering, sandboxing, iteration counts and costs are real.</div>") if j["mock"] else ""
    lb_rows = "".join(
        f"<tr><td class='player'>{_e(r['player'])}</td><td class=num>{_f(r['elo'], 0)}</td><td class=num>{_pct(r['rendered'])}</td>"
        f"<td class=num>{_pct(r['first_try'])}</td><td class=num>{_f(r['attempts_to_render'], 1)}</td><td class=num>{_pct(r['deterministic'])}</td>"
        f"<td class=num>{_f(r['fidelity'])}</td><td class=num>{_f(r['style'])}</td><td class=num>{_f(r['style_first'])}</td>"
        f"<td class=num>{_f(r['overall'])}</td><td class=num>{'$' + _f(r['cost']) if r['cost'] is not None else '–'}</td></tr>" for r in lb)
    grid_head = "".join(f"<th>{_e(p)}</th>" for p in players)
    grid_rows = "".join(
        f"<tr><td><b>{_e(t.get('title', t['id']))}</b><div class='small'>{_e(t['id'])} · {_e(t.get('style', ''))}</div>"
        f"<div class='small'><a href='{_e(t['id'])}/brief.md'>brief</a></div></td>"
        + "".join(_cell(by.get((t["id"], p)), run_dir) for p in players) + "</tr>" for t in results["tasks"])
    pairs = results.get("pairs") or []
    n_cons = sum(1 for p in pairs if p.get("consistent"))
    pair_rows = "".join(
        f"<tr><td>{_e(p['task'])}</td><td>{_e(p.get('player_a'))}</td><td>{_e(p.get('player_b'))}</td><td><b>{_e(p.get('verdict', p.get('error')))}</b></td>"
        f"<td class='small'>{_e((p.get('runs') or [{}])[0].get('reason', ''))[:220]}</td></tr>" for p in pairs)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model arena {_e(results['label'])}</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Model arena: {_e(results['label'])}</h1>
<div class="meta">run {_e(results['run_id'])} · up to {results['iterations']} versions per model · sandbox {_e(results['sandbox'])} ·
judge {_e(j['name'])} {_e(j['model'] or '')}</div>
{banner}
<p class="small">Each model wrote a complete numpy/scipy/Pillow program for every task, saw its own render, critiqued it and revised it.
The last version that rendered is judged blind, exactly like the studio's own styles (<b>house</b>). Style (first) is the rubric score of each model's
first working version, so the gap to Style shows how much self-critique helped.</p>
<h2>Leaderboard</h2>
<table class="lb"><tr><th>Player</th><th>Elo</th><th>Rendered</th><th>First try</th><th>Attempts to render</th><th>Deterministic</th>
<th>Subjects</th><th>Style</th><th>Style (first)</th><th>Overall</th><th>Cost</th></tr>{lb_rows}</table>
<h2>Tasks</h2><div class="grid"><table><tr><th>Task</th>{grid_head}</tr>{grid_rows}</table></div>
<h2>Pairwise</h2><p class="small">{len(pairs)} pairs, {n_cons} consistent after swapping positions; only those move Elo.</p>
<table><tr><th>Task</th><th>A</th><th>B</th><th>Verdict</th><th>Reason</th></tr>{pair_rows}</table>
</main></body></html>"""
    path = run_dir / "report.html"
    path.write_text(doc)
    return path
