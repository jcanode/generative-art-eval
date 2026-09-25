from paint.evals.technical import in_frame_fractions
from paint.playground import from_prompt, parse_prompt, preview_scene, surprise
from paint.scene import load_scene


def test_surprise_scenes_are_valid_reproducible_and_uncropped():
    for seed in range(40):
        s = surprise(seed)
        assert s == surprise(seed)
        sc = load_scene(s)
        assert sc.objects
        assert all(d["fraction"] >= 0.85 for d in in_frame_fractions(sc, sc.seed)), (seed, s["title"])


def test_prompt_parser_maps_words_to_kinds():
    kinds = lambda text: {o["kind"] for o in parse_prompt(text, 1)["objects"]}  # noqa: E731
    assert {"lighthouse", "moon", "sea", "cliff"} <= kinds("a lighthouse at night")
    assert "house" not in kinds("a lighthouse at night")
    assert "tree" not in kinds("pine trees by a lake") and "pine" in kinds("pine trees by a lake")
    assert {"table", "teapot", "fruit"} <= kinds("a teapot and apples")
    s = parse_prompt("three apples on a table", 1)
    fruit = next(o for o in s["objects"] if o["kind"] == "fruit")
    assert fruit["count"] == 3 and fruit["variant"] == "apple"


def test_offline_writer_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scene, source = from_prompt("a windmill in a field")
    assert source == "offline" and any(o["kind"] == "windmill" for o in scene["objects"])


def test_preview_keeps_layout():
    s = surprise(5)
    p = preview_scene(s, 300)
    assert max(p["canvas"]["width"], p["canvas"]["height"]) <= 300
    assert p["objects"] == s["objects"]
