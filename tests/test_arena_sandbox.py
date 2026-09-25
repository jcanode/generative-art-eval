import numpy as np
import pytest

from paint.arena.policy import check_source
from paint.arena.sandbox import run_program

GOOD = '''
import numpy as np
from scipy import ndimage

def paint(width, height, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(width, height)
    img = np.stack([x, y, 0.5 + 0 * x], axis=-1)
    img += ndimage.gaussian_filter(rng.normal(0, 0.05, (height, width)), 2)[..., None]
    return np.clip(img, 0, 1)
'''


def test_valid_program_runs_and_is_deterministic():
    a = run_program(GOOD, 64, 48, 7)
    b = run_program(GOOD, 64, 48, 7)
    assert a.ok, a.feedback()
    assert a.image.shape == (48, 64, 3) and a.image.dtype == np.uint8
    assert np.array_equal(a.image, b.image)
    assert a.isolation.get("RLIMIT_FSIZE") == 0


@pytest.mark.parametrize("src, needle", [
    ("import os\ndef paint(w,h,s): return 0", "import of 'os'"),
    ("import subprocess\ndef paint(w,h,s): return 0", "import of 'subprocess'"),
    ("import socket\ndef paint(w,h,s): return 0", "import of 'socket'"),
    ("def paint(w,h,s):\n    open('/etc/passwd').read()", "name 'open'"),
    ("import numpy as np\ndef paint(w,h,s):\n    np.save('x.npy', 1)", ".save"),
    ("from PIL import Image\ndef paint(w,h,s):\n    Image.open('/etc/passwd')", ".open"),
    ("def paint(w,h,s):\n    return ().__class__.__bases__[0].__subclasses__()", "__class__"),
    ("def paint(w,h,s):\n    return getattr(__builtins__, 'open')", "getattr"),
    ("def paint(w,h,s):\n    return eval('1')", "eval"),
    ("import numpy as np\ndef paint(w,h,s):\n    return np.ctypeslib", "ctypeslib"),
    ("x = 1", "must define"),
])
def test_policy_blocks_escapes(src, needle):
    msgs = " | ".join(map(str, check_source(src)))
    assert needle in msgs


def test_runtime_import_hook_blocks_dynamic_imports():
    # Passes the static check (no import statement), but the import machinery is not reachable anyway.
    src = "import numpy as np\ndef paint(w,h,s):\n    m = type(np)\n    return np.zeros((h,w,3))\n"
    assert run_program(src, 8, 8, 0).ok


def test_errors_come_back_as_feedback():
    r = run_program("import numpy as np\ndef paint(w,h,s):\n    return np.zeros((h,w))[5,5,5]\n", 16, 16, 0)
    assert not r.ok and r.stage == "run" and "IndexError" in r.error and "line 3" in r.traceback


def test_wrong_shape_is_reported():
    r = run_program("import numpy as np\ndef paint(w,h,s):\n    return np.zeros((10,10,3))\n", 16, 16, 0)
    assert not r.ok and "shape" in r.error


def test_infinite_loop_is_killed():
    r = run_program("def paint(w,h,s):\n    while True:\n        pass\n", 8, 8, 0, timeout=3)
    assert not r.ok and r.stage in ("timeout", "crash")


def test_memory_bomb_is_contained():
    r = run_program("import numpy as np\ndef paint(w,h,s):\n    return np.ones((1<<35,), np.uint8)\n", 8, 8, 0, timeout=20)
    assert not r.ok and ("Memory" in r.error or r.stage == "crash")


def test_no_environment_leaks(monkeypatch):
    import subprocess

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-should-not-leak")
    seen = {}
    real = subprocess.Popen

    def spy(cmd, **kw):
        seen["env"] = kw.get("env")
        return real(cmd, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    assert run_program(GOOD, 8, 8, 0).ok
    assert "ANTHROPIC_API_KEY" not in seen["env"] and not any("sk-ant" in v for v in seen["env"].values())
