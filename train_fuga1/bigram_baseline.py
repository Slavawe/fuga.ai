#!/usr/bin/env python3
# bigram_baseline.py — счетный bigram-baseline (этап 02, ревью 25.10).
# Использование: python bigram_baseline.py <corpus.jsonl> <val.bin> [val2.bin ...]
# Baseline: argmax следующего байта по предыдущему из счетов train-контента
# (извлечение зеркалит extract_bytes тренера: doc+\n+code).
import json
import sys
from collections import defaultdict

corpus, vals = sys.argv[1], sys.argv[2:]
hist = defaultdict(lambda: defaultdict(int))
for line in open(corpus, encoding="utf-8", errors="ignore"):
    try:
        v = json.loads(line)
        text = (v.get("doc", "") + ("\n" + v["code"] if "code" in v else "")) if "doc" in v else v.get("code", "")
    except Exception:
        text = line
    b = text.encode("utf-8", errors="ignore")
    for a, c in zip(b, b[1:]):
        hist[a][c] += 1
for name in vals:
    val = open(name, "rb").read()
    ok = tot = 0
    for a, c in zip(val, val[1:]):
        if a in hist:
            ok += max(hist[a].items(), key=lambda kv: kv[1])[0] == c
        tot += 1
    print(f"bigram({corpus}) на {name}: {ok/tot*100:.2f}% (n={tot})")
