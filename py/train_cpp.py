#!/usr/bin/env python3
# train_cpp.py — Python-оркестратор C++ ядра fuga (бета-тест C++).
#
# Роль Python: оркестрация (не compute). Собирает корпуса, запускает C++
# ядро обучения, генерирует entropy-vocab, суммирует результаты.
#
# Usage:
#   py/train_cpp.py train --jsonl fuga_unified_train.jsonl [--max-bytes 300000] [--out /tmp/w.bin]
#   py/train_cpp.py vocab --out /tmp/vocab.txt [--jsonl corpus.jsonl ...] [--limit 4000]
#   py/train_cpp.py decode --w /tmp/w.bin [--decoder recurrent] [--seed "fn main() {"]
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_BIN = os.path.join(ROOT, "cpp", "fuga_train")
DECODE_BIN = os.path.join(ROOT, "cpp", "fuga_decode")


def train(args):
    if not os.path.exists(CPP_BIN):
        sys.exit(f"нет {CPP_BIN} — собери: cd cpp && make")
    cmd = [CPP_BIN, "--max-bytes", str(args.max_bytes), "--out", args.out]
    for c in args.jsonl:
        cmd += ["--jsonl", c]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def vocab(args):
    """Собирает патч-грамматику (частотные 2-байтовые группы) из корпуса.
    Rust: patch_vocab собирается из byte-groups тренировочного корпуса; здесь —
    топ frequency-патчи размера psize, как строится словарь без токенов."""
    counter = Counter()
    for path in args.jsonl:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = ""
                for key in ("doc", "code"):
                    if isinstance(v.get(key), str):
                        text += v[key]
                if not text and "chapters" in v:
                    for ch in v["chapters"] or []:
                        for p in (ch.get("paragraphs") or []):
                            if isinstance(p, str):
                                text += p
                data = text.encode("utf-8")
                for i in range(0, len(data) - args.psize + 1):
                    counter[data[i:i + args.psize]] += 1
    top = counter.most_common(args.limit)
    with open(args.out, "w") as f:
        for patch, _ in top:
            f.write(patch.decode("utf-8", errors="replace") + "\n")
    print(f"vocab: {len(top)} патчей psize={args.psize} -> {args.out}")


def decode(args):
    if not os.path.exists(DECODE_BIN):
        _exit(f"нет {DECODE_BIN} — собери: 'cd cpp && make'")
    cmd = [DECODE_BIN, "--w", args.w, "--decoder", args.decoder,
           "--seed", args.seed, "--max", str(args.max)]
    if args.patch:
        cmd += ["--patch", args.patch]
    if args.vocab:
        cmd += ["--vocab", args.vocab]
    subprocess.run(cmd, check=True)


def _exit(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="fuga C++ оркестратор (бета)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--jsonl", nargs="+", required=True)
    p.add_argument("--max-bytes", type=int, default=300000)
    p.add_argument("--out", default="/tmp/fuga_cpp_w.bin")
    p.set_defaults(func=train)

    p = sub.add_parser("vocab")
    p.add_argument("--jsonl", nargs="+", required=True)
    p.add_argument("--psize", type=int, default=2)
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--out", default="/tmp/fuga_vocab.txt")
    p.set_defaults(func=vocab)

    p = sub.add_parser("decode")
    p.add_argument("--w", required=True)
    p.add_argument("--decoder", default="recurrent", choices=["naive", "recurrent", "entropy"])
    p.add_argument("--seed", default="fn main() {")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--patch", default="")
    p.add_argument("--vocab", default="")
    p.set_defaults(func=decode)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()