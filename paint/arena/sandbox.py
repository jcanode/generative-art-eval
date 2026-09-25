"""Run untrusted, model-written paint programs.

Layers (each one alone is not enough, together they are reasonable for this use):

1. Static policy (policy.py): allow-listed imports only; no file, process, network,
   or interpreter-introspection names; must define ``paint(width, height, seed)``.
2. A separate, fresh interpreter (``python -I``) started with an EMPTY environment
   (no API keys, no PYTHONPATH) in an empty temporary directory. Nothing from the
   studio process's memory is inherited (no fork).
3. Inside it: restricted builtins, an import hook that only hands out pre-imported
   allowed modules, and OS limits applied before the program runs: address-space
   and CPU-time caps, zero-byte file writes (RLIMIT_FSIZE), no new processes,
   and (Linux) an empty network namespace.
4. A wall-clock timeout in the parent that kills the whole process group.
5. Optional ``docker`` backend (``PAINT_SANDBOX=docker``): the same runner inside a
   container with ``--network none``, read-only root, dropped capabilities, memory,
   CPU and PID limits. Use it when running programs from models you don't trust at all.

The result is pixels only; the harness writes the PNG itself.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .policy import check_source

ROOT = Path(__file__).resolve().parents[2]
DOCKER_IMAGE = os.environ.get("PAINT_SANDBOX_IMAGE", "paint-arena-sandbox:latest")
BOOT = "import sys; sys.path.insert(0, {root!r}); from paint.arena.runner import main; main()"


@dataclass
class SandboxResult:
    ok: bool
    image: np.ndarray | None = None
    seconds: float = 0.0
    stage: str = ""            # policy | run | timeout | crash
    error: str = ""
    traceback: str = ""
    stdout: str = ""
    isolation: dict = field(default_factory=dict)
    backend: str = "process"

    def feedback(self) -> str:
        """What the model is told when its program fails."""
        if self.ok:
            return ""
        parts = [f"Your program failed at the {self.stage} stage.", self.error]
        if self.traceback:
            parts.append("Traceback (your code):\n" + self.traceback)
        if self.stdout:
            parts.append("Printed output:\n" + self.stdout[-1500:])
        return "\n".join(p for p in parts if p)


def _limits_preexec(mem_bytes: int, cpu_s: int):
    def fn():
        os.setsid()  # own process group, so a timeout can kill everything it started
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except Exception:  # noqa: BLE001
            pass
    return fn


def backend_name() -> str:
    b = os.environ.get("PAINT_SANDBOX", "process")
    return b if b in ("process", "docker") else "process"


def _command(backend: str, mem_bytes: int, cpus: float, name: str = ""):
    if backend == "docker":
        if not shutil.which("docker"):
            raise RuntimeError("PAINT_SANDBOX=docker but the docker CLI is not installed")
        mem_mb = max(256, mem_bytes // (1024 * 1024))
        return [
            "docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=16m",
            "--memory", f"{mem_mb}m", "--memory-swap", f"{mem_mb}m", "--cpus", str(cpus), "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "-v", f"{ROOT / 'paint'}:/app/paint:ro", "-w", "/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            DOCKER_IMAGE, "python", "-I", "-c", BOOT.format(root="/app"),
        ], None
    return [sys.executable, "-I", "-B", "-c", BOOT.format(root=str(ROOT))], _limits_preexec


def run_program(code: str, width: int, height: int, seed: int, timeout: float = 90.0,
                mem_bytes: int = 2 * 1024**3, cpu_seconds: int | None = None, backend: str | None = None) -> SandboxResult:
    backend = backend or backend_name()
    violations = check_source(code)
    if violations:
        return SandboxResult(False, stage="policy", error="\n".join(map(str, violations)), backend=backend)
    if width * height > 2048 * 2048:
        return SandboxResult(False, stage="policy", error="canvas too large", backend=backend)
    cpu_seconds = cpu_seconds or int(timeout) + 5
    req = json.dumps({"code": code, "width": width, "height": height, "seed": seed,
                      "limits": {"memory_bytes": mem_bytes, "cpu_seconds": cpu_seconds}}).encode()
    import uuid

    cname = f"paint-sandbox-{uuid.uuid4().hex[:12]}"
    cmd, preexec = _command(backend, mem_bytes, cpus=2, name=cname)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")} if backend == "docker" else \
          {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    if os.name == "nt":  # Windows needs these to start Python at all
        env.update({k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP") if k in os.environ})
    work = tempfile.mkdtemp(prefix="paint-sandbox-")
    t0 = time.perf_counter()
    kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work, env=env)
    if preexec and os.name == "posix":
        kwargs["preexec_fn"] = preexec(mem_bytes, cpu_seconds)
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        try:
            out, err = proc.communicate(req, timeout=timeout + (20 if backend == "docker" else 5))
        except subprocess.TimeoutExpired:
            _kill(proc)
            if backend == "docker":  # killing the CLI does not stop the container
                subprocess.run(["docker", "rm", "-f", cname], capture_output=True, timeout=30)
            proc.communicate()
            return SandboxResult(False, stage="timeout", error=f"the program ran longer than {timeout:.0f}s and was killed",
                                 seconds=time.perf_counter() - t0, backend=backend)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    secs = time.perf_counter() - t0
    head, _, body = out.partition(b"\n")
    try:
        status = json.loads(head or b"{}")
    except json.JSONDecodeError:
        status = {}
    if not status:
        why = "killed (CPU or memory limit)" if proc.returncode and proc.returncode < 0 else f"exit code {proc.returncode}"
        return SandboxResult(False, stage="crash", error=f"the sandbox process died: {why}. {err.decode(errors='replace')[-800:]}",
                             seconds=secs, backend=backend)
    if not status.get("ok"):
        return SandboxResult(False, stage=status.get("stage", "run"), error=status.get("error", ""),
                             traceback=status.get("traceback", ""), stdout=status.get("stdout", ""),
                             isolation=status.get("isolation", {}), seconds=secs, backend=backend)
    expected = width * height * 3
    if len(body) != expected:
        return SandboxResult(False, stage="crash", error=f"expected {expected} pixel bytes, got {len(body)}", seconds=secs, backend=backend)
    img = np.frombuffer(body, dtype=np.uint8).reshape(height, width, 3).copy()
    return SandboxResult(True, image=img, seconds=status.get("seconds", secs), stdout=status.get("stdout", ""),
                         isolation=status.get("isolation", {}), backend=backend)


def _kill(proc):
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def build_docker_image() -> str:
    """Build the sandbox image (numpy, scipy, pillow on python:3.11-slim)."""
    dockerfile = ROOT / "docker" / "arena-sandbox.Dockerfile"
    subprocess.run(["docker", "build", "-t", DOCKER_IMAGE, "-f", str(dockerfile), str(dockerfile.parent)], check=True)
    return DOCKER_IMAGE
