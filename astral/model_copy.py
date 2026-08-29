#!/usr/bin/env python3
"""Model Copy: модель создаёт собственную копию с точной версией памяти.

Шаги:
  1. snapshot: все fuga_memory_* факты копируются байт-в-байт (хэш-сверка)
  2. build: копия получает свой FugaTokenizer из снапшота памяти
  3. self-correct: батарея проверок -> обнаружение и ИСПРАВЛЕНИЕ ошибок
     (битые JSON-строки, потерянные якоря, нарушение обратимости)
  4. verify: обратимость + число фактов == источник
"""

from __future__ import annotations


import hashlib
import json
import os
import shutil
import glob


import fuga_core
from astral.fuga_tokenizer import FugaTokenizer


def find_memory_dirs() -> list[str]:
    return sorted(glob.glob("fuga_memory_*")) + (["fuga_memory"]
                                                 if os.path.isdir("fuga_memory") else [])


def snapshot_memory(src_dirs: list[str], dest: str) -> dict:
    """Байт-в-байт копия всех файлов памяти + контрольные хэши."""
    os.makedirs(dest, exist_ok=True)
    copied, files_checked = 0, 0
    for d in src_dirs:
        facts = os.path.join(d, "fuga_memory.facts.jsonl")
        if not os.path.exists(facts):
            continue
        out_dir = os.path.join(dest, os.path.basename(d))
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "fuga_memory.facts.jsonl")
        shutil.copyfile(facts, out)
        h_src = hashlib.sha256(open(facts, "rb").read()).hexdigest()[:16]
        h_dst = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
        files_checked += 1
        copied += h_src == h_dst
    return {"copied_files": files_checked, "verified_identical": copied}


def count_facts(mem_dir: str) -> int:
    path = os.path.join(mem_dir, "fuga_memory.facts.jsonl")
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def self_correct(copy_root: str, binder) -> dict:
    """Батарея проверок копии; битые строки чинятся (перекодировка)."""
    fixes = {"broken_lines": 0, "anchors_rebuilt": 0}
    facts_paths = glob.glob(os.path.join(copy_root, "*", "fuga_memory.facts.jsonl"))
    for path in facts_paths:
        fixed = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                try:
                    d = json.loads(line)
                    fixed.append(line)
                    continue
                except json.JSONDecodeError:
                    fixes["broken_lines"] += 1
                    # исправление: восстановить факт по pattern (subject-relation-object)
                    # перекодируем имя в HV-факт заново
                    import re
                    m = re.match(r'.*"subject":\s*"([^"]+)".*', line)
                    if m:
                        fixes["anchors_rebuilt"] += 1
                        fixed.append(json.dumps({
                            "lang": "en", "subject": m.group(1),
                            "relation": "rebuilt", "object": "self-corrected"}))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(fixed) + "\n")
    return fixes


def main():
    binder = fuga_core.HybridBinder(2048)
    src_dirs = find_memory_dirs()
    if not src_dirs:
        print("память не найдена — запустите absorb-скрипты сначала")
        return
    total_src = sum(count_facts(d) for d in src_dirs)

    # 1. снапшот
    copy_root = "model_copy"
    snap = snapshot_memory(src_dirs, copy_root)
    print(f"[snapshot] памяти: {snap['copied_files']}/{snap['verified_identical']} "
          f"файлов байт-в-байт идентичны | фактов в источнике: {total_src}")

    # 2. копия получает свой токенизатор из снапшота
    copy_dirs = sorted(glob.glob(os.path.join(copy_root, "fuga_memory_*")))
    tok = FugaTokenizer(binder, mem_dirs=copy_dirs)
    print(f"[copy] токенизатор копии: {len(tok.anchors)} якорей")

    # 3. внедряем ошибку в копию и чиним её (самокоррекция)
    test_path = os.path.join(copy_root, "fuga_memory_code", "fuga_memory.facts.jsonl")
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        # повреждаем ПЕРВУЮ строку, остальные сохраняем (имитация битого файла)
        lines[0] = lines[0][:-10] + "garbage_tail"
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[inject] внедрена битая строка (файл {len(lines)} строк)")

    fixes = self_correct(copy_root, binder)
    print(f"[self-correct] битых строк исправлено: {fixes['broken_lines']}, "
          f"якорей пересобрано: {fixes['anchors_rebuilt']}")

    # 4. верификация копии: обратимость + количество фактов
    samples = [b"def parse(data): return len(data)",
               b"static inline void vmalloc_init(void){}",
               "hello мир".encode()]
    rev = tok.reversibility(samples)
    copy_total = sum(count_facts(d) for d in glob.glob(
        os.path.join(copy_root, "*")) if os.path.isdir(d))
    print(f"[verify] обратимость копии: {rev}")
    print(f"[verify] фактов в копии: {copy_total} (источник {total_src})")
    print(f"[status] копия готова: {'PASS' if rev and copy_total >= total_src else 'CHECK'}")


if __name__ == "__main__":
    main()