"""Subject fidelity: does a blind viewer name the things the scene asked for?

The judge lists what it sees from pixels alone. We match those names against
the scene's object kinds using a synonym table, so "sailboat" counts for
``boat`` and "ocean" for ``sea``. Score = fraction of distinct spec kinds that
were named. Names that match nothing in the spec are reported as ``unexpected``:
that is where "a helmet that reads as a phone" shows up.
"""
from __future__ import annotations

import re

from ..scene.shapes import SYNONYMS
from ..scene.spec import Scene

# Kinds that set the stage rather than being subjects; reported, but weighted lower.
SETTING = {"sea", "field", "table", "river", "hills", "stars", "rain", "cloud"}
SETTING_WEIGHT = 0.5


def _norm(word: str) -> str:
    w = word.lower().strip()
    w = re.sub(r"[^a-z ]", " ", w)
    return re.sub(r"\s+", " ", w).strip()


def _singular(w: str) -> str:
    for suf, rep in (("ies", "y"), ("ves", "f"), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            cand = w[: -len(suf)] + rep
            return cand
    return w


def _tokens(name: str) -> set[str]:
    n = _norm(name)
    toks = set(n.split()) | {n}
    return toks | {_singular(t) for t in toks}


def matches(kind: str, name: str) -> bool:
    syn = {_norm(s) for s in SYNONYMS.get(kind, [kind])} | {kind}
    syn |= {_singular(s) for s in syn}
    return bool(_tokens(name) & syn)


def score_subjects(scene: Scene, described: dict, min_confidence: str = "low") -> dict:
    order = {"low": 0, "medium": 1, "high": 2}
    objs = [o for o in described.get("objects", []) if order.get(o.get("confidence", "low"), 0) >= order[min_confidence]]
    names = [o["name"] for o in objs]
    kinds = sorted(set(scene.kinds))
    found, missing = {}, []
    for k in kinds:
        hits = [n for n in names if matches(k, n)]
        if hits:
            found[k] = hits
        else:
            missing.append(k)
    unexpected = [n for n in names if not any(matches(k, n) for k in kinds)]
    w = {k: (SETTING_WEIGHT if k in SETTING else 1.0) for k in kinds}
    total = sum(w.values()) or 1.0
    weighted = sum(w[k] for k in found) / total
    subjects = [k for k in kinds if k not in SETTING]
    subj_score = (sum(1 for k in subjects if k in found) / len(subjects)) if subjects else None
    return {
        "score": round(len(found) / len(kinds), 3) if kinds else None,
        "weighted": round(weighted, 3),
        "subjects_score": round(subj_score, 3) if subj_score is not None else None,
        "found": found,
        "missing": missing,
        "unexpected": unexpected,
        "judge_objects": objs,
        "judge_summary": described.get("summary", ""),
        "judge_medium": described.get("medium", ""),
    }
