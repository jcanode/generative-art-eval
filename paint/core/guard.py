"""Resource guards: every render has a wall-clock timeout and a memory ceiling.

Renders run in a forked child process. The parent waits with a deadline and
kills the child if it overruns; the child sets ``RLIMIT_AS`` so a runaway
allocation raises ``MemoryError`` instead of swapping the machine to death.
(macOS does not enforce ``RLIMIT_AS`` and Windows has no ``resource`` module, so
there the memory cap is best-effort; the timeout and pixel cap still apply.)
Both failure modes surface as ``RenderFailure`` with a loud, specific message.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_MEMORY_BYTES = 4 * 1024**3
MAX_PIXELS = 4096 * 4096


class RenderFailure(RuntimeError):
    """A render failed, timed out, or exceeded its memory budget."""


@dataclass
class GuardResult:
    value: object
    seconds: float
    peak_rss_mb: float


def check_canvas(width: int, height: int) -> None:
    """Refuse absurd canvases before allocating anything."""
    if width <= 0 or height <= 0:
        raise RenderFailure(f"canvas must be positive, got {width}x{height}")
    if width * height > MAX_PIXELS:
        raise RenderFailure(
            f"canvas {width}x{height} exceeds the {MAX_PIXELS} pixel cap; "
            "render smaller and upscale instead"
        )


def _peak_rss_mb() -> float:
    try:
        import resource

        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024.0**2 if sys.platform == "darwin" else rss / 1024.0  # bytes on macOS, KiB on Linux
    except Exception:  # pragma: no cover - non-POSIX
        return float("nan")


def _child(conn, fn, args, kwargs, mem_bytes):
    try:
        if mem_bytes:
            try:
                import resource

                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ImportError, ValueError, OSError):
                pass
        t0 = time.perf_counter()
        value = fn(*args, **kwargs)
        conn.send(("ok", value, time.perf_counter() - t0, _peak_rss_mb()))
    except MemoryError:
        conn.send(("mem", f"exceeded memory ceiling of {mem_bytes / 1024**3:.1f} GiB", 0.0, _peak_rss_mb()))
    except BaseException as exc:  # noqa: BLE001 - forward everything to the parent
        conn.send(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", 0.0, _peak_rss_mb()))
    finally:
        conn.close()


def _start_method() -> str:
    """fork on Linux (fast, no re-import); spawn on macOS/Windows, where fork is unsafe or missing.

    Override with PAINT_MP_START=fork|spawn|forkserver.
    """
    import sys

    forced = os.environ.get("PAINT_MP_START")
    if forced:
        return forced
    return "fork" if sys.platform.startswith("linux") else "spawn"


def run_guarded(fn, *args, timeout: float = DEFAULT_TIMEOUT_S,
                mem_bytes: int | None = DEFAULT_MEMORY_BYTES, **kwargs) -> GuardResult:
    """Run ``fn(*args, **kwargs)`` in a child process with a timeout and memory cap.

    ``fn`` and its return value must be picklable. Set ``PAINT_NO_FORK=1`` to run
    in-process (useful under debuggers); the timeout is then only checked after
    the fact.
    """
    if os.environ.get("PAINT_NO_FORK"):
        t0 = time.perf_counter()
        value = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt > timeout:
            raise RenderFailure(f"render took {dt:.1f}s, over the {timeout:.0f}s budget")
        return GuardResult(value, dt, _peak_rss_mb())

    ctx = mp.get_context(_start_method())
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, fn, args, kwargs, mem_bytes), daemon=True)
    t0 = time.perf_counter()
    proc.start()
    child.close()
    try:
        if not parent.poll(timeout):
            proc.kill()
            proc.join(5)
            raise RenderFailure(f"render timed out after {timeout:.0f}s and was killed")
        try:
            status, value, dt, rss = parent.recv()
        except EOFError:
            proc.join(5)
            raise RenderFailure(
                f"render process died without a result (exit code {proc.exitcode}); "
                "likely killed for memory"
            ) from None
    finally:
        parent.close()
    proc.join(5)
    if status == "mem":
        raise RenderFailure(f"render ran out of memory: {value}")
    if status == "err":
        raise RenderFailure(f"render raised {value}")
    return GuardResult(value, dt or (time.perf_counter() - t0), rss)
