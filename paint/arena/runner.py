"""Sandbox child: runs ONE model-written paint program and returns its pixels.

Started by sandbox.py as ``python -I -m paint.arena.runner`` in a fresh
interpreter with a scrubbed environment and an empty working directory. Protocol:
a JSON request on stdin; on stdout, one JSON status line followed (on success)
by the raw uint8 pixel bytes.

Order matters: pre-import the allowed libraries, then drop privileges (network
namespace, rlimits), then exec the program with restricted builtins and an
import hook that only returns pre-imported allowed modules.
"""
from __future__ import annotations

import builtins
import importlib
import io
import json
import sys
import time
import traceback


def _isolate(limits: dict) -> dict:
    applied = {}
    try:
        import resource

        mem = int(limits.get("memory_bytes", 2 * 1024**3))
        cpu = int(limits.get("cpu_seconds", 120))
        for name, val in (("RLIMIT_AS", mem), ("RLIMIT_CPU", cpu), ("RLIMIT_FSIZE", 0), ("RLIMIT_NPROC", 0)):
            lim = getattr(resource, name, None)
            if lim is None:
                continue
            try:
                resource.setrlimit(lim, (val, val))
                applied[name] = val
            except (ValueError, OSError):
                pass
    except ImportError:  # Windows: rely on the parent's timeout
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            applied["no_network"] = libc.unshare(0x40000000) == 0  # CLONE_NEWNET: an empty network namespace
            del ctypes, libc
        except Exception:  # noqa: BLE001
            applied["no_network"] = False
    return applied


def main():
    req = json.loads(sys.stdin.read())
    out = sys.stdout.buffer
    from paint.arena.policy import PREIMPORT, SAFE_BUILTINS, check_source

    mods = {}
    for name in PREIMPORT:
        try:
            mods[name] = importlib.import_module(name)
        except ImportError:
            pass
    import numpy as np

    violations = check_source(req["code"])
    if violations:
        out.write((json.dumps({"ok": False, "stage": "policy", "error": "\n".join(map(str, violations))}) + "\n").encode())
        return
    applied = _isolate(req.get("limits", {}))

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in mods:
            raise ImportError(f"import of '{name}' is not allowed in the sandbox")
        if fromlist:
            return mods[name]
        return mods[name.split(".")[0]] if name.split(".")[0] in mods else mods[name]

    safe = {k: getattr(builtins, k) for k in SAFE_BUILTINS if hasattr(builtins, k)}
    printed = io.StringIO()

    def sandbox_print(*a, **k):
        if printed.tell() < 20_000:
            k.pop("file", None)
            print(*a, file=printed, **k)

    safe["print"] = sandbox_print
    safe["__import__"] = guarded_import
    safe["__build_class__"] = builtins.__build_class__
    glb = {"__builtins__": safe, "__name__": "paint_program"}
    import linecache

    src_lines = req["code"].splitlines(keepends=True)
    linecache.cache["<program>"] = (len(req["code"]), None, src_lines, "<program>")  # so tracebacks show source
    t0 = time.perf_counter()
    try:
        exec(compile(req["code"], "<program>", "exec"), glb)  # noqa: S102 - this IS the sandbox
        paint = glb.get("paint")
        if not callable(paint):
            raise TypeError("paint(width, height, seed) is not defined")
        img = paint(int(req["width"]), int(req["height"]), int(req["seed"]))
        arr = np.asarray(img)
        H, W = int(req["height"]), int(req["width"])
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape != (H, W, 3):
            raise ValueError(f"paint() returned shape {arr.shape}, expected ({H}, {W}, 3)")
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError("paint() returned NaN or infinite values")
            if arr.max() > 1.0 + 1e-6:  # accept 0-255 floats too
                arr = arr / 255.0
            arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
        status = {"ok": True, "seconds": round(time.perf_counter() - t0, 3), "isolation": applied,
                  "stdout": printed.getvalue()[-4000:]}
        out.write((json.dumps(status) + "\n").encode())
        out.write(np.ascontiguousarray(arr).tobytes())
    except MemoryError:
        out.write((json.dumps({"ok": False, "stage": "run", "error": "MemoryError: the program exceeded its memory limit",
                               "isolation": applied}) + "\n").encode())
    except BaseException as exc:  # noqa: BLE001 - everything goes back to the model as feedback
        tb = traceback.extract_tb(exc.__traceback__)
        frames = [f for f in tb if f.filename == "<program>"] or tb[-3:]
        lines = [f'  line {f.lineno}, in {f.name}' + (f"\n    {f.line}" if f.line else "") for f in frames[-6:]]
        msg = f"{type(exc).__name__}: {exc}"
        out.write((json.dumps({"ok": False, "stage": "run", "error": msg[:2000], "traceback": "\n".join(lines)[:3000],
                               "stdout": printed.getvalue()[-2000:], "isolation": applied}) + "\n").encode())
    out.flush()


if __name__ == "__main__":
    main()
