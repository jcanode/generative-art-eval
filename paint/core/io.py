"""Image I/O. The only place float images become 8-bit."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


def to_uint8(img) -> np.ndarray:
    return (np.clip(np.nan_to_num(img, nan=0.0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def png_bytes(img, meta: dict | None = None) -> bytes:
    """Encode deterministically: fixed compression, only the metadata we pass."""
    info = PngImagePlugin.PngInfo()
    for k, v in (meta or {}).items():
        info.add_text(str(k), str(v))
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img)).save(buf, format="PNG", pnginfo=info, optimize=False, compress_level=6)
    return buf.getvalue()


def save_png(img, path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(img, meta))
    return path


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def thumbnail(path, out, size: int = 320) -> Path:
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="PNG")
    return out
