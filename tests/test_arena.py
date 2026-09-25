import json

from paint.arena.brief import describe_scene, load_tasks, task_message
from paint.arena.providers import ScriptedCoder, extract_code
from paint.arena.session import MAX_ITERS, run_session

OK = "import numpy as np\ndef paint(width, height, seed):\n    return np.full((height, width, 3), 0.5) + np.random.default_rng(seed).random((height, width, 3)) * 0.2\n"
BROKEN = "import numpy as np\ndef paint(width, height, seed):\n    return np.zeros((height, width))[0, 0, 0]\n"
ESCAPE = "import os\ndef paint(width, height, seed):\n    return os.listdir('/')\n"


def test_session_feeds_errors_back_and_recovers(tmp_path):
    coder = ScriptedCoder([ESCAPE, BROKEN, OK, OK])
    task = {"id": "t"}
    res = run_session(coder, task, "brief", tmp_path / "s", 64, 48, 3, iterations=4)
    stages = [v.stage for v in res.versions]
    assert stages[:2] == ["policy", "run"]
    assert res.first_ok == 3 and res.final == 4 and res.deterministic
    assert (tmp_path / "s" / "v1" / "error.txt").read_text().startswith("Your program failed at the policy stage")
    assert (tmp_path / "s" / "v4" / "image.png").exists()
    saved = json.loads((tmp_path / "s" / "session.json").read_text())
    assert saved["final"] == 4


def test_session_is_capped(tmp_path):
    res = run_session(ScriptedCoder([BROKEN]), {"id": "t"}, "brief", tmp_path / "s", 16, 16, 0, iterations=99)
    assert len(res.versions) == MAX_ITERS and res.final is None and res.status == "no_image"


def test_revision_messages_carry_the_image(tmp_path):
    seen = []

    class Spy(ScriptedCoder):
        def revise(self, blocks):
            seen.append(blocks)
            return super().revise(blocks)

    run_session(Spy([OK]), {"id": "t"}, "brief", tmp_path / "s", 32, 32, 0, iterations=2)
    assert seen and seen[0][0]["type"] == "image" and "Score: N/10" in seen[0][1]["text"]


def test_extract_code_prefers_block_with_paint():
    text = "plan\n```python\nx = 1\n```\nfinal:\n```python\ndef paint(w, h, s):\n    pass\n```"
    assert "def paint" in extract_code(text)


def test_briefs_are_identical_for_every_model_and_describe_the_scene():
    t = load_tasks()[0]
    import yaml
    from paint.cli import ROOT

    scene = yaml.safe_load((ROOT / t["scene"]).read_text())
    a = task_message(t, scene, 1024, 768, 7)
    assert a == task_message(t, scene, 1024, 768, 7)
    for o in scene["objects"]:
        assert o["kind"] in a
