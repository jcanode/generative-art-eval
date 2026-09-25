import numpy as np
import pytest

from conftest import small_scene
from paint.scene import KINDS, SceneError, build_shapes, load_scene
from paint.core.geom import Frame
from paint.core.rng import Rng
from paint.styles import available, get_style


def test_every_kind_builds_a_visible_shape():
    f = Frame(128, 128)
    for kind in sorted(KINDS):
        sc = load_scene({"canvas": {"width": 128, "height": 128}, "horizon": 0.8,
                         "objects": [{"kind": kind, "x": 0.5, "y": 0.7, "size": 0.5}]})
        shapes = build_shapes(sc, f, Rng(1))
        assert shapes and shapes[0].mask.sum() > 0, kind
        for p in shapes[0].parts:
            assert np.isfinite(p.sdf).all(), (kind, p.name)


def test_unknown_kind_is_rejected():
    with pytest.raises(SceneError):
        load_scene({"objects": [{"kind": "helicopter"}]})


@pytest.mark.parametrize("style", available())
def test_style_contract_and_determinism(style):
    mod = get_style(style)
    for attr in ("NAME", "DEFAULTS", "KNOBS", "RULES", "render", "render_array"):
        assert hasattr(mod, attr), attr
    assert set(mod.KNOBS) <= set(mod.DEFAULTS)
    sc = small_scene()
    a, b = mod.render(sc), mod.render(sc)
    assert a == b, "same seed must give identical bytes"
    img, _ = mod.render_array(sc)
    assert img.shape == (120, 160, 3) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    img2, _ = mod.render_array(sc, seed=4)
    assert not np.array_equal(img, img2), "different seeds should differ"
