#!/usr/bin/env python3
"""Извлечение кода из клонированных репо в чистый JSONL-корпус.

Формат: {"code": "..."} — совместим с unified_gpu_train.
Каждая строка JSONL = фрагмент кода 100-400 байт (размер патча).

Фильтры (K3):
- только исходники: .c .h .rs .go .php (без тестов/генератов по пути)
- ASCII-процент > 0.98 (чистый код, без юникод-комментариев)
- длина 100-400 байт
- без типичных шумов (пустые файлы, однострочники мусора)
"""

import json
import os
import random
import sys

# Расширения и директории-исключения
CODE_EXT = {".c", ".h", ".rs", ".go", ".php"}
SKIP_DIRS = {
    "test", "tests", "testing", "node_modules", "vendor", "third_party",
    "third-party", "benchmark", "benchmarks", "fuzz", "fuzzing",
    "target", "build", "dist", ".git", "tools", "scripts", "utils",
    "libc-test", "gcc", "clang", "llvm",
}
MAX_BYTES = 400
MIN_BYTES = 100
ASCII_MIN = 0.98


def walk_files(repo_root: str):
    """Итерация по файлам кода (не рекурсивно в SKIP_DIRS)."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # фильтруем пропускные директории
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in CODE_EXT:
                yield os.path.join(dirpath, fn)


def chunk_lines(text: str, min_b: int, max_b: int):
    """Разбить файл на фрагменты 100-400 байт по строкам."""
    lines = text.splitlines()
    buf = []
    buf_len = 0
    for line in lines:
        b = line.encode("utf-8", errors="ignore")
        if buf and buf_len + len(b) > max_b:
            if buf_len >= min_b:
                yield "\n".join(buf)
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(b) + 1
    if buf and buf_len >= min_b:
        yield "\n".join(buf)


def is_ascii_clean(text: str) -> bool:
    if not text:
        return False
    ascii_cnt = sum(1 for c in text if ord(c) < 128)
    return ascii_cnt / len(text) >= ASCII_MIN


def main():
    repos = {
        "linux": "/tmp/fuga_corpus_src/linux",
        "rust": "/tmp/fuga_corpus_src/rust",
        "go": "/tmp/fuga_corpus_src/go",
        "php": "/tmp/fuga_corpus_src/php-src",
    }
    out_path = "/tmp/fuga_multicode.jsonl"
    lang_count = {k: 0 for k in repos}
    total = 0

    with open(out_path, "w") as out:
        for lang, repo in repos.items():
            if not os.path.isdir(repo):
                print(f"  [skip] {lang}: нет {repo}")
                continue
            n = 0
            for fpath in walk_files(repo):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                if not is_ascii_clean(text):
                    continue
                for chunk in chunk_lines(text, MIN_BYTES, MAX_BYTES):
                    out.write(json.dumps({"code": chunk}) + "\n")
                    n += 1
                    total += 1
            lang_count[lang] = n
            print(f"  {lang}: {n} фрагментов")
            sys.stdout.flush()

    print(f"\nИТОГО: {total} фрагментов → {out_path}")
    print(f"по языкам: {lang_count}")


if __name__ == "__main__":
    main()
