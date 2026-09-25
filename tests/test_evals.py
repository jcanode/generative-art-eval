import json

import numpy as np
import pytest

from conftest import small_scene
from paint.core.io import save_png
from paint.evals.calibration import agreement, quadratic_kappa
from paint.evals.fidelity import matches, score_subjects
from paint.evals.judges import Judge, MockJudge, blind_png, load_rubric
from paint.evals.pairwise import EloTable, compare_pair
from paint.evals.technical import run_checks
from paint.render import render_to_file


def test_blank_image_fails_not_blank(tmp_path):
    sc = small_scene()
    p = save_png(np.full((120, 160, 3), 0.9, np.float32), tmp_path / "blank.png")
    rep = run_checks(p, sc, "screenprint", determinism=False)
    assert not next(c for c in rep["checks"] if c["name"] == "not_blank")["passed"]


def test_cropped_subject_fails_in_frame(tmp_path):
    sc = small_scene(objects=[{"kind": "sea"}, {"kind": "boat", "x": 0.99, "y": 0.75, "size": 0.4}])
    rec = render_to_file(sc, "screenprint", 3, tmp_path, name="crop")
    rep = run_checks(rec.png, sc, "screenprint", determinism=False)
    c = next(c for c in rep["checks"] if c["name"] == "in_frame")
    assert not c["passed"] and "boat" in c["detail"]


def test_real_render_passes_all_checks(tmp_path):
    sc = small_scene()
    rec = render_to_file(sc, "screenprint", 3, tmp_path, name="ok")
    rep = run_checks(rec.png, sc, "screenprint", seconds=rec.seconds)
    failed = [c for c in rep["checks"] if not c["passed"]]
    assert rep["passed"], failed


def test_blind_png_strips_metadata(tmp_path):
    p = save_png(np.zeros((10, 10, 3), np.float32), tmp_path / "x.png", {"paint:params": "secret-knobs"})
    data, _ = blind_png(p)
    assert b"secret-knobs" not in data and b"paint:" not in data


def test_fidelity_synonyms_and_unexpected():
    sc = small_scene()
    r = score_subjects(sc, {"objects": [{"name": "sailboat", "confidence": "high"},
                                        {"name": "ocean waves", "confidence": "high"},
                                        {"name": "telephone", "confidence": "medium"}]})
    assert set(r["found"]) == {"boat", "sea"}
    assert r["missing"] == ["sun"]
    assert r["unexpected"] == ["telephone"]
    assert matches("bird", "Seagulls") and not matches("boat", "goat")


class BiasedJudge(Judge):
    """Always prefers whatever is shown first: a pure position bias."""
    name = "biased"

    def _ask(self, images, prompt, schema):
        return {"reason": "first one", "winner": "1"}


class FairJudge(Judge):
    name = "fair"

    def __init__(self, prefer):
        super().__init__()
        self.prefer = prefer

    def compare(self, a, b, rubric=None):
        return {"reason": "", "winner": "1" if str(a) == self.prefer else "2"}


def test_position_bias_is_caught_by_swap(tmp_path):
    a = save_png(np.zeros((8, 8, 3), np.float32), tmp_path / "a.png")
    b = save_png(np.ones((8, 8, 3), np.float32), tmp_path / "b.png")
    assert compare_pair(BiasedJudge(), a, b)["verdict"] == "inconsistent"
    r = compare_pair(FairJudge(str(b)), a, b)
    assert r["verdict"] == "B" and r["consistent"]


def test_elo_only_moves_on_consistent_verdicts(tmp_path):
    elo = EloTable(tmp_path / "elo.json")
    elo.record("x", "y", "inconsistent")
    assert elo.ratings["x"] == elo.ratings["y"] == 1500
    elo.record("x", "y", "A")
    assert elo.ratings["x"] > 1500 > elo.ratings["y"]
    elo.save()
    assert json.loads((tmp_path / "elo.json").read_text())["games"]["x"]["w"] == 1


def test_calibration_agreement_math():
    perfect = agreement([1, 2, 3, 4, 5, 3], [1, 2, 3, 4, 5, 3])
    assert perfect["spearman"] == 1.0 and perfect["exact_agreement"] == 1.0 and perfect["pair_concordance"] == 1.0
    assert quadratic_kappa(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])) == pytest.approx(1.0)
    worse = agreement([1, 2, 3, 4, 5, 3], [5, 4, 3, 2, 1, 3])
    assert worse["spearman"] < 0 and "not trustworthy" in worse["verdict"]


def test_rubrics_parse():
    for style in ("screenprint", "sumie", "impasto"):
        r = load_rubric(style)
        assert len(r["criteria"]) >= 4 and all(c["definition"] for c in r["criteria"])


def test_mock_judge_is_labelled(tmp_path):
    p = save_png(np.random.default_rng(0).random((64, 64, 3)).astype(np.float32), tmp_path / "n.png")
    j = MockJudge()
    assert j.is_mock and j.describe(p)["objects"] == []
    assert j.compare(p, p)["winner"] == "tie"
