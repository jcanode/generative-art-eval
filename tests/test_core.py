import numpy as np
import pytest

from paint.core import halftone
from paint.core.brush import Brush, render_stroke
from paint.core.flow import FlowBuilder
from paint.core.geom import Frame
from paint.core.guard import RenderFailure, check_canvas, run_guarded
from paint.core.masks import sdf_to_mask, supersample_mask
from paint.core.rng import Rng
from paint.core import sdf as S


def test_rng_is_reproducible_and_label_scoped():
    a, b = Rng(7).child("paper").random(5), Rng(7).child("paper").random(5)
    assert np.array_equal(a, b)
    # Creating another child first must not shift an existing stream.
    r = Rng(7)
    r.child("strokes").random(100)
    assert np.array_equal(r.child("paper").random(5), a)
    assert not np.array_equal(Rng(8).child("paper").random(5), a)


def test_sdf_mask_is_antialiased():
    f = Frame(64, 64)
    X, Y = f.grid
    m = sdf_to_mask(S.circle(X, Y, 32, 32, 10))
    assert m[32, 32] == 1.0 and m[0, 0] == 0.0
    assert ((m > 0) & (m < 1)).sum() > 20  # soft edge pixels exist
    ss = supersample_mask(lambda x, y: S.circle(x, y, 32, 32, 10), f, 4)
    assert abs(ss.sum() - np.pi * 100) / (np.pi * 100) < 0.03


def test_halftone_coverage_tracks_tone():
    f = Frame(128, 128)
    covs = [halftone.dots(np.full(f.shape, t, np.float32), f, 8, 45).mean() for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert covs[0] == 0 and covs[-1] == 1
    assert all(b > a for a, b in zip(covs, covs[1:]))
    assert abs(covs[2] - 0.5) < 0.08


def test_brush_stroke_tapers_and_is_deterministic():
    path = np.array([[20, 50], [60, 40], [110, 55]], np.float32)
    b = Brush(width=14, bristles=12, dry_rate=0.8)
    a1 = render_stroke(path, b, Rng(1), (100, 140))
    a2 = render_stroke(path, b, Rng(1), (100, 140))
    assert np.array_equal(a1.alpha, a2.alpha)
    assert a1.alpha.max() > 0.5
    # Ink runs dry: the end of the stroke deposits less than the start.
    t = a1.t
    assert a1.alpha[t < 0.3].mean() > a1.alpha[t > 0.8].mean()


def test_flow_field_traces_vectorised():
    f = Frame(96, 96)
    field = FlowBuilder(f, Rng(2)).uniform(0.0).vortex(48, 48, 20, 2.0).build()
    paths = field.trace(np.array([[10, 10], [80, 80]]), steps=8, step=2.0)
    assert paths.shape == (2, 9, 2)
    assert np.isfinite(paths).all()


def _sleep_forever():
    import time

    while True:
        time.sleep(0.1)


def _allocate_too_much():
    return np.ones((1 << 34,), dtype=np.uint8)  # 16 GiB


def _boom():
    raise ValueError("nope")


def test_guard_timeout_kills_runaway_render():
    with pytest.raises(RenderFailure, match="timed out"):
        run_guarded(_sleep_forever, timeout=1.0)


def test_guard_memory_ceiling_fails_loudly():
    with pytest.raises(RenderFailure, match="memory"):
        run_guarded(_allocate_too_much, timeout=20, mem_bytes=1 << 30)


def test_guard_forwards_errors():
    with pytest.raises(RenderFailure, match="ValueError: nope"):
        run_guarded(_boom, timeout=10)


def test_canvas_cap():
    with pytest.raises(RenderFailure):
        check_canvas(100_000, 100_000)
