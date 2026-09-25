"""Static HTML report for a suite run: thumbnails, scores, diff against the previous run."""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root{--bg:#f7f5f0;--fg:#1d1c1a;--muted:#6b675f;--card:#fff;--line:#e3dfd6;--ok:#2f7d4f;--bad:#b3261e;--warn:#a15c00;--accent:#1f4fa3;--chip:#efece5}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161513;--fg:#ece9e2;--muted:#a09b91;--card:#211f1c;--line:#34312c;--ok:#6fcf97;--bad:#ff8a80;--warn:#ffb74d;--accent:#8ab4f8;--chip:#2b2925}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.meta{color:var(--muted);font-size:13px}.banner{padding:10px 14px;border-radius:8px;margin:14px 0;border:1px solid var(--warn);color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
td.num{font-variant-numeric:tabular-nums;text-align:right}.grid{overflow-x:auto}
.cell{min-width:260px}.cell img{width:100%;max-width:320px;border-radius:4px;border:1px solid var(--line);display:block;background:var(--chip)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.chip{background:var(--chip);border-radius:999px;padding:1px 8px;font-size:12px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.up{color:var(--ok)}.down{color:var(--bad)}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:12px}.small{font-size:12px;color:var(--muted)}
.pair{display:grid;grid-template-columns:1fr 1fr 2fr;gap:10px;align-items:start;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.pair img{width:100%;border-radius:4px;border:1px solid var(--line)}.win{outline:3px solid var(--ok);outline-offset:-3px}
.prev{opacity:.85;max-width:120px!important;margin-top:4px}
@media (max-width:700px){.pair{grid-template-columns:1fr 1fr}.pair>div:last-child{grid-column:1/-1}}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _fmt(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _delta(v, nd=2, invert=False):
    if v is None or v == 0:
        return ""
    good = (v > 0) != invert
    return f' <span class="{"up" if good else "down"}">{"+" if v > 0 else ""}{v:.{nd}f}</span>'


def _cell(it) -> str:
    if not it:
        return "<td class='small'>–</td>"
    if not it.get("png"):
        return f"<td class='cell'><div class='bad'>render failed</div><div class='small'>{_e((it.get('error') or '')[:300])}</div></td>"
    t = it.get("technical", {})
    fails = [c for c in t.get("checks", []) if not c["passed"]]
    d = it.get("diff", {})
    fid = it.get("fidelity") or {}
    st = it.get("style_score") or {}
    chips = [f"<span class='chip {'ok' if t.get('passed') else 'bad'}'>tech {'pass' if t.get('passed') else f'{len(fails)} fail'}</span>",
             f"<span class='chip'>{_fmt(it.get('seconds'), 1)}s</span>"]
    if fid:
        chips.append(f"<span class='chip'>subjects {_fmt(fid.get('score')) if fid.get('score') is not None else 'n/a'}{_delta(d.get('fidelity'))}</span>")
    if st:
        chips.append(f"<span class='chip'>style {_fmt(st.get('mean'))}{_delta(d.get('style'))}</span>")
    status = d.get("status")
    if status == "changed":
        chips.append(f"<span class='chip warn'>changed Δ{_fmt(d.get('mean_abs_diff'), 3)}</span>")
    elif status == "same":
        chips.append("<span class='chip'>unchanged</span>")
    elif status == "new":
        chips.append("<span class='chip'>new</span>")
    if d.get("technical"):
        chips.append(f"<span class='chip {'ok' if d['technical'] == 'fixed' else 'bad'}'>tech {d['technical']}</span>")
    parts = [f"<td class='cell'><a href='{_e(it['png'])}'><img loading='lazy' src='{_e(it.get('thumb', it['png']))}' alt='{_e(it['scene_title'])} in {_e(it['style'])}'></a>",
             f"<div class='chips'>{''.join(chips)}</div>"]
    det = []
    if fails:
        det.append("<b>Failed checks</b><ul>" + "".join(f"<li>{_e(c['name'])}: {_e(c['detail'])}</li>" for c in fails) + "</ul>")
    if fid and fid.get("note"):
        det.append(f"<span class='small'>{_e(fid['note'])}</span>")
    elif fid:
        det.append(f"<b>Blind judge saw:</b> {_e(', '.join(o['name'] for o in fid.get('judge_objects', [])) or '(nothing listed)')}<br>"
                   f"<b>Found:</b> {_e(', '.join(fid.get('found', {})) or '–')}<br>"
                   f"<b class='bad'>Missing:</b> {_e(', '.join(fid.get('missing', [])) or '–')}<br>"
                   f"<b>Unexpected:</b> {_e(', '.join(fid.get('unexpected', [])) or '–')}<br>"
                   f"<b>Medium guess:</b> {_e(fid.get('judge_medium', ''))}")
    if st:
        det.append("<b>Rubric</b><ul>" + "".join(
            f"<li>{_e(s['criterion'])}: <b>{_e(s['score'])}</b> – {_e(s['reason'])}</li>" for s in st.get("scores", [])) + "</ul>")
    m = t.get("metrics", {})
    if m:
        det.append("<span class='small'>" + " · ".join(f"{_e(k)} {_e(v)}" for k, v in m.items()) + "</span>")
    if it.get("judge_error"):
        det.append(f"<span class='bad'>judge error: {_e(it['judge_error'])}</span>")
    if d.get("prev_thumb"):
        det.append(f"<div class='small'>previous run:</div><img class='prev' src='{_e(d['prev_thumb'])}' alt='previous render'>")
    if det:
        parts.append("<details><summary>details</summary>" + "<br>".join(det) + "</details>")
    return "".join(parts) + "</td>"


def _grid(items, styles, split) -> str:
    rows = sorted({(it["scene_id"], it["scene_title"]) for it in items if it["split"] == split})
    if not rows:
        return ""
    by = {(it["scene_id"], it["style"]): it for it in items}
    head = "".join(f"<th>{_e(s)}</th>" for s in styles)
    body = "".join(
        f"<tr><td><b>{_e(title)}</b><div class='small'>{_e(sid)}</div><div class='small'>{_e(', '.join(next((it['kinds'] for it in items if it['scene_id'] == sid), [])))}</div></td>"
        + "".join(_cell(by.get((sid, s))) for s in styles) + "</tr>"
        for sid, title in rows)
    return f"<div class='grid'><table><tr><th>Scene</th>{head}</tr>{body}</table></div>"


def _summary(results) -> str:
    cur, prev = results.get("summary", {}), results.get("previous_summary") or {}
    rows = []
    for style, splits in cur.items():
        for split, s in splits.items():
            p = prev.get(style, {}).get(split, {})

            def dl(k, nd=2, inv=False):
                if s.get(k) is None or p.get(k) is None:
                    return ""
                return _delta(round(s[k] - p[k], 3), nd, inv)

            rows.append(f"<tr><td>{_e(style)}</td><td>{_e(split)}{' <span class=small>(never tuned on)</span>' if split == 'holdout' else ''}</td>"
                        f"<td class=num>{s['n']}</td><td class=num>{_fmt(s['technical_pass'])}{dl('technical_pass')}</td>"
                        f"<td class=num>{_fmt(s['fidelity'])}{dl('fidelity')}</td><td class=num>{_fmt(s['fidelity_subjects'])}{dl('fidelity_subjects')}</td>"
                        f"<td class=num>{_fmt(s['style'])}{dl('style')}</td><td class=num>{_fmt(s['seconds'], 1)}{dl('seconds', 1, True)}</td></tr>")
    return ("<table><tr><th>Style</th><th>Split</th><th>n</th><th>Tech pass</th><th>Subject fidelity</th>"
            "<th>Main subjects</th><th>Style (1-5)</th><th>Seconds</th></tr>" + "".join(rows) + "</table>")


def _pairs(results) -> str:
    pairs = results.get("pairs") or []
    if not pairs:
        return "<p class='small'>No pairwise comparisons in this run (needs a judge, and a previous run or several styles).</p>"
    out = []
    n_cons = sum(1 for p in pairs if p.get("consistent"))
    out.append(f"<p class='small'>{len(pairs)} pairs, {n_cons} consistent after swapping positions "
               f"({(n_cons / len(pairs)):.0%}). Only consistent verdicts move Elo.</p>")
    for p in pairs:
        if "error" in p:
            out.append(f"<div class='pair'><div class='bad'>{_e(p['error'])}</div></div>")
            continue
        v = p["verdict"]
        reasons = "".join(f"<li>order {'/'.join(r['order'])}: <b>{_e(r['winner'])}</b> – {_e(r['reason'])}</li>" for r in p["runs"])
        out.append(f"<div class='pair'><div><img class='{'win' if v == 'A' else ''}' src='{_e(p['a'])}' alt='A'><div class='small'>A: {_e(p['player_a'])}</div></div>"
                   f"<div><img class='{'win' if v == 'B' else ''}' src='{_e(p['b'])}' alt='B'><div class='small'>B: {_e(p['player_b'])}</div></div>"
                   f"<div><b>{_e(p['kind'])}</b> · {_e(p['scene_id'])} ({_e(p.get('split'))}) → <b>{_e(v)}</b><ul>{reasons}</ul></div></div>")
    return "".join(out)


def _elo(results) -> str:
    rows = results.get("elo") or []
    if not rows:
        return "<p class='small'>No ratings yet.</p>"
    return ("<table><tr><th>Version</th><th>Elo</th><th>W</th><th>L</th><th>T</th><th>Inconsistent</th></tr>"
            + "".join(f"<tr><td>{_e(r['player'])}</td><td class=num>{r['elo']:.0f}</td><td class=num>{r['w']}</td>"
                      f"<td class=num>{r['l']}</td><td class=num>{r['t']}</td><td class=num>{r['inconsistent']}</td></tr>" for r in rows)
            + "</table>")


def _calibration(results) -> str:
    a = results.get("calibration")
    if not a:
        return ("<p class='small'>No human calibration yet. Run <code>paint calibrate --init</code>, rate the ~20 images "
                "(in <code>paint serve</code> or ratings.json), then <code>paint calibrate</code>.</p>")
    keys = ["n", "spearman", "kendall_tau", "quadratic_kappa", "exact_agreement", "within_one", "mae", "judge_bias", "pair_concordance"]
    return ("<table><tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr><tr>"
            + "".join(f"<td class=num>{_e(a.get(k, '–'))}</td>" for k in keys) + "</tr></table>"
            + f"<p><b>{_e(a.get('verdict', ''))}</b></p>")


def write_report(results: dict, run_dir: Path) -> Path:
    styles = results["styles"]
    j = results["judge"]
    banner = ""
    if j["mock"]:
        banner = ("<div class='banner'><b>Mock judge.</b> No API key was available, so subject and style scores come from "
                  "pixel statistics, not a vision model. They exercise the pipeline and are <b>not</b> meaningful. "
                  "Set ANTHROPIC_API_KEY or use the web studio to run the real judge.</div>")
    elif j["name"] == "none":
        banner = "<div class='banner'>Judges were skipped (technical checks only).</div>"
    items = results["items"]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Suite report {_e(results['label'])}</title><style>{CSS}</style></head><body><main>
<h1>Suite report: {_e(results['label'])}</h1>
<div class="meta">run {_e(results['run_id'])} · code {_e(results['code'])} · judge {_e(j['name'])} {_e(j['model'] or '')} ·
rubrics {_e(json.dumps(results['rubrics']))} · {_e(results['seconds'])}s · compared with {_e(results.get('previous') or 'nothing (first run)')}</div>
{banner}
<h2>Summary</h2>{_summary(results)}
<p class="small">Subject fidelity = share of the scene's object kinds that a blind judge named from pixels alone (main subjects excludes setting like sea/sky/table).
Style = mean rubric score (1-5, evals/rubrics/*.md). Deltas are against the previous run. Holdout scenes are never used when iterating.</p>
<h2>Regression suite</h2>{_grid(items, styles, 'suite')}
<h2>Holdout scenes</h2>{_grid(items, styles, 'holdout') or "<p class=small>not included in this run</p>"}
<h2>Pairwise comparisons</h2>{_pairs(results)}
<h2>Elo across versions</h2>{_elo(results)}
<h2>Judge calibration against your ratings</h2>{_calibration(results)}
</main></body></html>"""
    path = run_dir / "report.html"
    path.write_text(doc)
    return path
