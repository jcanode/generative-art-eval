"""Contact sheets: many renders side by side, for looking at a whole suite at once."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def contact_sheet(paths, out, tile: int = 360, cols: int = 4, labels=None) -> Path:
    paths = [Path(p) for p in paths]
    rows = max(1, (len(paths) + cols - 1) // cols)
    label_h = 18 if labels else 0
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((tile - 8, tile - 8), Image.LANCZOS)
        x0 = (i % cols) * tile + (tile - im.width) // 2
        y0 = (i // cols) * (tile + label_h) + (tile - im.height) // 2
        sheet.paste(im, (x0, y0))
        if labels:
            draw.text(((i % cols) * tile + 6, (i // cols) * (tile + label_h) + tile), str(labels[i])[:48], fill=(40, 40, 40))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
