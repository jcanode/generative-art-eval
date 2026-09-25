"""Style registry. Each style is a module exposing NAME, DEFAULTS, KNOBS, RULES, render, render_array."""
from __future__ import annotations

import importlib

STYLES = ("screenprint", "sumie", "impasto", "pointillism", "woodblock", "linocut", "stainedglass")


def get_style(name: str):
    name = {"sumi-e": "sumie", "sumi_e": "sumie", "stained-glass": "stainedglass", "stained_glass": "stainedglass"}.get(name, name)
    if name not in STYLES:
        raise KeyError(f"unknown style {name!r}; available: {', '.join(STYLES)}")
    return importlib.import_module(f"{__name__}.{name}")


def available() -> list[str]:
    out = []
    for s in STYLES:
        try:
            get_style(s)
            out.append(s)
        except Exception:  # noqa: BLE001 - a style mid-development must not break the others
            pass
    return out
