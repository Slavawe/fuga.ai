#!/usr/bin/env python3
"""sat_tokenizer_val: спутник-валидатор FugaTokenizer.

Автономно рождается материнской моделью, запускает полный набор
проверок токенизатора (L1 обратимость, AST-границы, L2 скорость),
сохраняет отчёт в BIM и возвращает метрики.
"""

from __future__ import annotations


import json
import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from astral.fuga_tokenizer import FugaTokenizer


VALIDATION_CODE = """#!/usr/bin/env python3
# sat_tokenizer_val: автономный валидатор FugaTokenizer (рождён {ts})
# L1 reversibility | L1 AST-boundaries | L2 speed -> JSON в stdout
import sys, os, json, time
sys.path.insert(0, {root!r})
from astral.fuga_tokenizer import FugaTokenizer
import fuga_core

binder = fuga_core.HybridBinder(2048)
tok = FugaTokenizer(binder)

NL = chr(10)
samples = [
    "def parse(data): return len(data)".encode(),
    "static inline void vmalloc_init(void){}".encode(),
    "hello mir  nihao".encode(),
    chr(0).encode("latin1") + chr(255).encode("latin1") + b" binary",
]
code = NL.join(["def parse_docs(data):", "    result = len(data)",
                "    return result"]) + NL
large = ("def f(x): return x*2" + NL) * 500

results = {{}}
results["reversibility"] = all(
    tok.decode(tok.encode_tokens(s)) == s for s in samples)
results["ast_boundary_score"] = tok.ast_boundary_score(code, "python")
results["no_oov"] = tok.no_oov(b"random " + chr(0).encode("latin1") + b" seq")
t0 = time.time()
tok.encode_tokens(large)
dt = time.time() - t0
lines = large.count(b"\n") + 1
results["speed_lines_per_sec"] = round(lines / max(dt, 1e-9), 0)
results["anchors"] = len(tok.anchors)
results["status"] = "PASS" if results["reversibility"] else "WARN"
print(json.dumps(results))
"""

def spawn_validator(binder, out_dir="satellites"):
    name = "sat_tokenizer_val"
    path = os.path.join(out_dir, f"{name}.py")
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code = (VALIDATION_CODE
            .replace("{ts}", str(int(time.time())))
            .replace("{root!r}", repr(root)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    # запуск (L2)
    import subprocess
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        report = json.loads(r.stdout)
    else:
        report = {"status": "FAIL", "stderr": r.stderr[:200]}
    report["path"] = path
    return report


def main():
    binder = fuga_core.HybridBinder(2048)
    print("[mother] спутник-валидатор токенизатора:")
    report = spawn_validator(binder)
    for k, v in report.items():
        if k in ("anchors", "speed_lines_per_sec", "ast_boundary_score", "reversibility", "no_oov", "status"):
            print(f"  {k}: {v}")
    print(f"  status: {report.get('status')}")
    # BIM-регистрация
    from astral.bim_engine import BIMEngine, BIMNode
    bim = BIMEngine(binder)
    bim.add_component(BIMNode(
        "sat_tokenizer_val", "validator",
        vram_gb=0.0, throughput=(
            f"rev={report.get('reversibility')}, ast={report.get('ast_boundary_score'):.2f}, "
            f"speed={report.get('speed_lines_per_sec',0):.0f} l/s")), [])
    print(f"[BIM] validator registered: {len(bim.nodes)} nodes")


if __name__ == "__main__":
    main()