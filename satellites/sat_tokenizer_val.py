#!/usr/bin/env python3
# sat_tokenizer_val: автономный валидатор FugaTokenizer (рождён 1787857079)
# L1 reversibility | L1 AST-boundaries | L2 speed -> JSON в stdout
import sys, os, json, time
sys.path.insert(0, '/home/slava/Anti-Tronsformers')
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
lines = large.count(b"
") + 1
results["speed_lines_per_sec"] = round(lines / max(dt, 1e-9), 0)
results["anchors"] = len(tok.anchors)
results["status"] = "PASS" if results["reversibility"] else "WARN"
print(json.dumps(results))
