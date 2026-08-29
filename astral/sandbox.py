
"""Sandbox: двухуровневая исполнительная валидация сгенерированного кода.

Level 1 (static): Tree-Sitter — 0 ERROR-узлов.
Level 2 (dynamic): реальный запуск в субпроцессе с лимитом времени/RAM,
                   захват stdout/stderr.
Петля обратной связи: PASS -> «золотой якорь» в VSA-память;
                   FAIL -> traceback кодируется в HV_error.
"""

from __future__ import annotations

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
import re

import numpy as np


import fuga_core
from antitf.rust_bridge import packed_to_torch
from astral.code_synth import _get_parser


def level1_static(code: str, lang: str) -> dict:
    parser_mods = {"c": "tree_sitter_c", "rust": "tree_sitter_rust",
                   "python": "tree_sitter_python", "go": "tree_sitter_go"}
    parser = _get_parser(parser_mods[lang])
    tree = parser.parse(code.encode())
    errors = 0

    def walk(n):
        nonlocal errors
        if n.type == "ERROR" or n.is_missing:
            errors += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return {"errors": errors, "ok": errors == 0}


def _limit_memory(max_mb: int):
    # RLIMIT_AS: виртуальная память процесса
    resource.setrlimit(resource.RLIMIT_AS,
                       (max_mb * 1024 * 1024, max_mb * 1024 * 1024))


def run_python(code: str, timeout_s: float = 2.0, mem_mb: int = 256) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-u", path],
            capture_output=True, text=True, timeout=timeout_s,
            preexec_fn=lambda: _limit_memory(mem_mb))
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": "TIMEOUT", "stdout": "",
                "stderr": f"timeout after {timeout_s}s"}
    finally:
        os.unlink(path)


def run_rust(code: str, timeout_s: float = 6.0) -> dict:
    """rustc -> исполняемый -> запуск."""
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "prog.rs")
    with open(src, "w") as f:
        f.write(code)
    try:
        comp = subprocess.run(["rustc", "-O", src, "-o",
                               os.path.join(tmp, "prog")],
                              capture_output=True, text=True,
                              timeout=timeout_s)
        if comp.returncode != 0:
            return {"ok": False, "returncode": comp.returncode,
                    "stdout": "", "stderr": comp.stderr[-2000:]}
        run = subprocess.run([os.path.join(tmp, "prog")],
                             capture_output=True, text=True,
                             timeout=timeout_s)
        return {"ok": run.returncode == 0, "returncode": run.returncode,
                "stdout": run.stdout[-2000:], "stderr": run.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": "TIMEOUT", "stdout": "",
                "stderr": "rust timeout"}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def error_to_hv(binder, stderr: str) -> np.ndarray:
    """Traceback -> HV_error через кодирование значимых строк ошибки."""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    tokens = []
    for ln in lines[:8]:
        toks = re.findall(r"[a-z_]+", ln.lower())
        tokens += toks[:8]
    if not tokens:
        tokens = ["error"]
    pk = np.asarray(binder.bind_batch([tokens[:32]]))
    return packed_to_torch(pk)[0]


def validate(binder, code: str, lang: str) -> dict:
    l1 = level1_static(code, lang)
    result = {"level1": l1["ok"], "errors_l1": l1["errors"]}
    if not l1["ok"]:
        result["level2"] = False
        result["status"] = "REJECTED_L1"
        return result
    l2 = run_python(code) if lang == "python" else \
        (run_rust(code) if lang == "rust" else None)
    if l2 is None:
        result["level2"] = True
        result["status"] = "NO_EXEC"
        return result
    result["level2"] = l2["ok"]
    result["returncode"] = l2["returncode"]
    if l2["ok"]:
        result["status"] = "PASS"
        result["output"] = l2["stdout"]
    else:
        result["status"] = "FAIL"
        result["output"] = l2["stdout"]
        result["error_hv"] = error_to_hv(binder, l2["stderr"])
        result["error_text"] = l2["stderr"][:300]
    return result


def demo():
    binder = fuga_core.HybridBinder(2048)
    good = ("def parse_docs(data):\n"
            "    result = len(data)\n"
            "    return result\n\n"
            "print(parse_docs('hello'))\n")
    bad = ("def parse_docs(data):\n"
           "    return data[0] / 0   # ZeroDivision\n\n"
           "print(parse_docs([1]))\n")

    print("[python PASS]")
    r1 = validate(binder, good, "python")
    print("  ", r1["status"], "| level1:", r1["level1"], "| output:",
          r1.get("output", "").strip())

    print("[python FAIL -> HV_error]")
    r2 = validate(binder, bad, "python")
    print("  ", r2["status"], "| level1:", r2["level1"])
    if "error_hv" in r2:
        print("  error_text:", r2["error_text"].splitlines()[-1][:80])
        print("  HV_error нормирован:",
              round(float(r2["error_hv"].float().norm() / 2048), 3))


if __name__ == "__main__":
    demo()
