import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def small_scene(**over):
    from paint.scene import load_scene

    d = {"title": "t", "canvas": {"width": 160, "height": 120}, "seed": 3, "horizon": 0.6,
         "palette": {"hints": ["#f2c14e", "#e8643c", "#2a8c88", "#1d2d4a"]},
         "objects": [{"kind": "sun", "x": 0.7, "y": 0.3, "size": 0.2}, {"kind": "sea"},
                     {"kind": "boat", "x": 0.4, "y": 0.75, "size": 0.3}]}
    d.update(over)
    return load_scene(d)
