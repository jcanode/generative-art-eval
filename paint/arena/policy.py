"""What model-written paint programs may and may not do.

The contract: a program defines ``paint(width, height, seed)`` and returns an
(height, width, 3) RGB array (uint8 or float in [0, 1]). The harness saves the
PNG itself, so a program never needs files, network, processes, or
introspection, and all of those are refused.

This static check is the first of several layers (see sandbox.py). On its own
a Python-level check is not a security boundary, so it runs together with
process isolation and OS limits.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_MODULES = {
    "numpy", "math", "cmath", "random", "colorsys", "itertools", "functools", "collections", "dataclasses",
    "typing", "heapq", "bisect", "statistics", "operator", "enum", "string", "re", "fractions", "decimal",
    "scipy", "scipy.ndimage", "scipy.spatial", "scipy.interpolate", "scipy.signal", "scipy.special",
    "scipy.stats", "scipy.fft", "scipy.linalg", "scipy.sparse",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter", "PIL.ImageChops", "PIL.ImageOps",
    "PIL.ImageEnhance", "PIL.ImageColor", "PIL.ImageMath",
}
# Pre-imported in the sandbox before limits are applied (lazy imports would need file access later).
PREIMPORT = sorted(ALLOWED_MODULES)

# Attribute names that touch files, processes, the network, memory, or interpreter internals.
BLOCKED_ATTRS = {
    "open", "save", "load", "loads", "loadtxt", "savetxt", "savez", "savez_compressed", "genfromtxt", "fromfile",
    "tofile", "memmap", "fromregex", "DataSource", "ctypeslib", "ctypes", "system", "popen", "spawn", "fork",
    "remove", "unlink", "rmdir", "rmtree", "chmod", "chown", "rename", "replace", "truetype", "load_path",
    "frombuffer", "from_dlpack", "_io", "f_globals", "f_locals", "f_back", "gi_frame", "cr_frame", "tb_frame",
    "tb_next", "co_code", "func_globals", "show", "register_open", "register_save", "_getexif", "getexif",
    "set_printoptions", "seterrcall", "lib", "testing", "distutils", "f2py", "__loader__", "__spec__",
}
SAFE_DUNDERS = {"__init__", "__call__", "__len__", "__iter__", "__next__", "__getitem__", "__setitem__",
                "__repr__", "__str__", "__eq__", "__lt__", "__add__", "__mul__", "__post_init__", "__name__"}
BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "memoryview", "help", "exit", "quit", "__builtins__", "__loader__",
    "__spec__", "classmethod", "staticmethod",  # harmless, but keep the surface small
}
BLOCKED_NAMES -= {"classmethod", "staticmethod"}

MAX_SOURCE_BYTES = 200_000


@dataclass
class PolicyViolation:
    line: int
    message: str

    def __str__(self):
        return f"line {self.line}: {self.message}"


def check_source(src: str) -> list[PolicyViolation]:
    """Static policy check. Returns violations (empty list = allowed)."""
    if len(src.encode()) > MAX_SOURCE_BYTES:
        return [PolicyViolation(0, f"program is larger than {MAX_SOURCE_BYTES // 1000} kB")]
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [PolicyViolation(exc.lineno or 0, f"syntax error: {exc.msg}")]
    out: list[PolicyViolation] = []
    has_paint = False
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES:
                    out.append(PolicyViolation(ln, f"import of '{a.name}' is not allowed (numpy, scipy, Pillow and a few stdlib math modules only)"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod not in ALLOWED_MODULES:
                out.append(PolicyViolation(ln, f"import from '{mod}' is not allowed"))
            for a in node.names:
                if a.name == "*":
                    out.append(PolicyViolation(ln, "star imports are not allowed"))
                elif f"{mod}.{a.name}" not in ALLOWED_MODULES and a.name in BLOCKED_ATTRS:
                    out.append(PolicyViolation(ln, f"'{a.name}' is not allowed"))
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                out.append(PolicyViolation(ln, f"attribute '.{node.attr}' is not allowed (no file, process or interpreter access)"))
            elif node.attr.startswith("__") and node.attr not in SAFE_DUNDERS:
                out.append(PolicyViolation(ln, f"dunder attribute '.{node.attr}' is not allowed"))
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or (node.id.startswith("__") and node.id not in SAFE_DUNDERS):
                out.append(PolicyViolation(ln, f"name '{node.id}' is not allowed"))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
        elif isinstance(node, ast.FunctionDef) and node.name == "paint":
            has_paint = True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("__") and node.value.endswith("__"):
            out.append(PolicyViolation(ln, "dunder strings are not allowed"))
    if not has_paint:
        out.append(PolicyViolation(0, "the program must define a top-level function paint(width, height, seed)"))
    return out


SAFE_BUILTINS = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "super", "tuple", "type", "zip",
    "classmethod", "staticmethod", "hasattr",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError", "KeyError", "LookupError",
    "NotImplementedError", "OverflowError", "RuntimeError", "StopIteration", "TypeError", "ValueError",
    "ZeroDivisionError", "FloatingPointError",
]
