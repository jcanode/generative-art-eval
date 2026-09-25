"""One contestant, one task: write -> run in the sandbox -> look -> critique -> revise (max 4).

The same loop the studio uses, applied to a model writing the whole program.
Every version is kept (program.py, image.png or error, the model's reply), and
the model sees its own render as an image at each step. Scored outputs:

* ``final``: the last version that rendered (the model's own answer after revising)
* ``first_ok``: the first version that rendered (one-shot quality, for measuring self-improvement)
"""
from __future__ import annotations

import io
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.io import save_png
from ..evals import metrics
from .brief import CRITIQUE_REQUEST, ERROR_REQUEST
from .providers import image_block
from .sandbox import run_program

MAX_ITERS = 4
SCORE_RE = re.compile(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", re.I)


@dataclass
class Version:
    n: int
    ok: bool
    seconds: float = 0.0
    error: str = ""
    stage: str = ""
    self_score: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    model_seconds: float = 0.0
    image: str | None = None
    program: str = ""


@dataclass
class SessionResult:
    contestant: str
    task_id: str
    dir: str
    versions: list[Version] = field(default_factory=list)
    final: int | None = None
    first_ok: int | None = None
    deterministic: bool | None = None
    status: str = "done"
    message: str = ""

    @property
    def cost(self):
        return round(sum(v.cost for v in self.versions), 4)

    def to_dict(self):
        d = asdict(self)
        d["cost"] = self.cost
        return d


def _png_for_model(img: np.ndarray, max_side: int = 1024) -> bytes:
    im = Image.fromarray(img)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _checks_text(img: np.ndarray, seconds: float, budget: float) -> tuple[str, dict]:
    m = metrics.summary(img.astype(np.float32) / 255.0)
    blank = m["lum_std"] < 0.03 or m["color_clusters"] < 3
    lines = [f"- rendered in {seconds:.1f}s (budget {budget:.0f}s){'  <- TOO SLOW' if seconds > budget else ''}",
             f"- {'BLANK / near-uniform image' if blank else 'not blank'} (luminance std {m['lum_std']:.3f})",
             f"- mean saturation {m['sat_mean']:.2f}, edge density {m['edge_density']:.2f}, {m['color_clusters']} colour clusters"]
    return "\n".join(lines), m


def run_session(coder, task: dict, task_text: str, out_dir: Path, width: int, height: int, seed: int,
                iterations: int = MAX_ITERS, timeout: float = 90.0, budget_s: float = 60.0,
                backend: str | None = None, deadline_s: float = 1800.0, max_cost: float | None = None,
                progress=None) -> SessionResult:
    say = progress or (lambda m: None)
    iterations = max(1, min(int(iterations), MAX_ITERS))
    out_dir.mkdir(parents=True, exist_ok=True)
    res = SessionResult(coder.name, task["id"], str(out_dir))
    t_start = time.time()
    feedback_blocks = None
    for n in range(1, iterations + 1):
        if time.time() - t_start > deadline_s:
            res.status, res.message = "deadline", f"session deadline {deadline_s:.0f}s reached"
            break
        if max_cost is not None and res.cost > max_cost:
            res.status, res.message = "budget", f"cost cap ${max_cost:.2f} reached"
            break
        vdir = out_dir / f"v{n}"
        vdir.mkdir(exist_ok=True)
        try:
            reply = coder.start(task_text) if n == 1 else coder.revise(feedback_blocks)
        except Exception as exc:  # noqa: BLE001 - API errors end the session, loudly
            res.status, res.message = "error", f"{type(exc).__name__}: {exc}"
            say(f"  {coder.name} {task['id']} v{n}: model call failed: {res.message[:160]}")
            break
        v = Version(n, False, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                    cost=round(coder.cost(reply), 5), model_seconds=round(reply.seconds, 1), program=str(vdir / "program.py"))
        (vdir / "reply.md").write_text(reply.text)
        (vdir / "program.py").write_text(reply.code or "# (no program found in the reply)\n")
        sm = SCORE_RE.findall(reply.text)
        if sm and n > 1:  # the score in reply n rates version n-1's image
            prev = res.versions[-1]
            prev.self_score = float(sm[-1])
        if not reply.code:
            v.stage, v.error = "format", "no ```python block with def paint(...) found in the reply"
            feedback = v.error
        else:
            r = run_program(reply.code, width, height, seed, timeout=timeout, backend=backend)
            v.seconds = round(r.seconds, 2)
            if r.ok:
                v.ok = True
                save_png(r.image.astype(np.float32) / 255.0, vdir / "image.png")
                v.image = str(vdir / "image.png")
                checks, m = _checks_text(r.image, r.seconds, budget_s)
                (vdir / "metrics.json").write_text(json.dumps(m))
                res.final = n
                res.first_ok = res.first_ok or n
                feedback = None
                if n < iterations:
                    feedback_blocks = [image_block(_png_for_model(r.image)),
                                       {"type": "text", "text": "Technical checks:\n" + checks + "\n\n" +
                                        CRITIQUE_REQUEST.format(n=n, seed=seed, width=width, height=height, max_iters=iterations)}]
            else:
                v.stage, v.error = r.stage, r.feedback()
                feedback = r.feedback()
        if not v.ok:
            (vdir / "error.txt").write_text(v.error)
            feedback_blocks = [{"type": "text", "text": ERROR_REQUEST.format(n=n, feedback=feedback[:6000], max_iters=iterations)}]
        res.versions.append(v)
        say(f"  {coder.name} {task['id']} v{n}: {'rendered in %.1fs' % v.seconds if v.ok else v.stage + ' error'}"
            f" · {v.output_tokens} out tokens · ${v.cost:.3f}")
        (out_dir / "session.json").write_text(json.dumps(res.to_dict(), indent=1, default=str))
    # Determinism of the judged version: run it again with the same seed.
    if res.final:
        code = (out_dir / f"v{res.final}" / "program.py").read_text()
        again = run_program(code, width, height, seed, timeout=timeout, backend=backend)
        first = np.asarray(Image.open(out_dir / f"v{res.final}" / "image.png").convert("RGB"))
        res.deterministic = bool(again.ok and np.array_equal(again.image, first))
    elif res.status == "done":
        res.status, res.message = "no_image", "no version produced an image"
    (out_dir / "session.json").write_text(json.dumps(res.to_dict(), indent=1, default=str))
    return res
