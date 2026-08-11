#!/usr/bin/env python3
"""Метрика точности имён (Шаг 3): декодирует чекпоинт v6_validate-ом,
считает точные совпадения snake_case-имён корпуса (freq>=3, len>=6)
в потоках v2 и MegaByte_v2. Использование:
  python3 src/bin/name_metric.py fuga_v7_1_final.fuga [корпуса jsonl...]
"""
import re, subprocess, sys, os

ckpt = sys.argv[1] if len(sys.argv) > 1 else "fuga_v7_1_final.fuga"
corpora = sys.argv[2:] or [
    "fisig_corpus.jsonl", "corpus_doc_code_pairs.jsonl",
    "training_stack.jsonl", "corpus.jsonl",
]
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")

freq = {}
for corp in corpora:
    try:
        with open(corp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                for m in re.findall(r"\b[a-z_][a-z0-9_]{5,23}\b", line):
                    if "_" in m and not m.isupper():
                        freq[m] = freq.get(m, 0) + 1
    except FileNotFoundError:
        print(f"  (нет корпуса {corp})")
name_set = {n for n, c in freq.items() if c >= 3 and len(n) >= 6}
print(f"Имён в корпусе (snake_case, freq>=3, len>=6): {len(name_set)}")

r = subprocess.run(
    ["./target/release/v6_validate", ckpt, *corpora],
    capture_output=True, text=True, timeout=300, errors="replace",
)
v2_text, mb2_text = "", ""
for line in r.stdout.splitlines():
    if line.startswith("[MB2]"):
        m = re.search(r"\(\d+B\):\s*(.+)", line)
        if m:
            mb2_text += m.group(1)
    elif line.startswith("[V2") and "ph" not in line:
        m = re.search(r"\(\d+B\):\s*(.+)", line)
        if m:
            v2_text += m.group(1)

def hits(text, names):
    return {n: text.count(n) for n in names if text.count(n) > 0}

v2h, mb2h = hits(v2_text, name_set), hits(mb2_text, name_set)
print(f"v2  ({len(v2_text)}B): {len(v2h)} точных имён, {sum(v2h.values())} вхожд.: {sorted(list(v2h))[:6]}")
print(f"MB2 ({len(mb2_text)}B): {len(mb2h)} точных имён, {sum(mb2h.values())} вхожд.: {sorted(list(mb2h))[:6]}")
for name in ["codepoint_to_utf8", "buf_append", "RedisModule_Free", "code_buf"]:
    print(f"  '{name}': корпус={freq.get(name, 0)} v2={v2_text.count(name)} MB2={mb2_text.count(name)}")