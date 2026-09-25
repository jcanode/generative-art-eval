import json
import shutil

import pytest

from paint.loop.agent import MAX_ITERS, HoldoutError, run_loop
from paint.scene import dump_scene
from conftest import ROOT, small_scene


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "tiny.yaml"
    dump_scene(small_scene(), p)
    return p


def test_loop_is_capped_and_keeps_every_version(tmp_path, scene_file):
    res = run_loop(scene_file, "screenprint", out_dir=tmp_path / "loop", critic="heuristic", max_iters=99)
    assert res.status == "done"
    assert 1 <= len(res.versions) <= MAX_ITERS
    for v in res.versions:
        assert (res.run_dir / f"v{v['version']}" / "image.png").exists()
        assert (res.run_dir / f"v{v['version']}" / "critique.md").exists()
    assert (res.run_dir / "best.png").exists()


def test_manual_loop_waits_for_a_critique(tmp_path, scene_file):
    out = tmp_path / "loop"
    r1 = run_loop(scene_file, "screenprint", out_dir=out, critic="manual")
    assert r1.status == "awaiting_critique"
    v1 = r1.run_dir / "v1"
    assert (v1 / "critique.template.json").exists() and not (v1 / "critique.json").exists()
    (v1 / "critique.json").write_text(json.dumps({"summary": "too few halftone dots", "score": 4,
                                                  "adjustments": {"halftone_cell": 11.0}, "stop": False}))
    r2 = run_loop(scene_file, "screenprint", out_dir=out, critic="manual")
    assert r2.status == "awaiting_critique"
    v2 = json.loads((r2.run_dir / "loop.json").read_text())["versions"][1]
    assert v2["params"]["halftone_cell"] == 11.0


def test_loop_refuses_holdout(tmp_path):
    holdout = next((ROOT / "evals" / "holdout").glob("*.yaml"))
    with pytest.raises(HoldoutError):
        run_loop(holdout, "screenprint", out_dir=tmp_path, critic="heuristic")


def test_site_refuses_path_traversal():
    from paint.site.server import _local_path, _resolve_file

    assert _resolve_file("/files/runs/../pyproject.toml") is None
    assert _resolve_file("/files/runs/%2e%2e/pyproject.toml") is None
    assert _resolve_file("/files/etc/passwd") is None
    with pytest.raises(ValueError):
        _local_path("pyproject.toml")


def test_site_key_is_memory_only(tmp_path):
    from paint.site.server import KeyStore

    k = KeyStore()
    k.set("sk-ant-test-1234567890")
    assert k.masked().startswith("sk-ant-") and "1234567890" not in k.masked()
    k.set(None)
    assert k.get() is None
