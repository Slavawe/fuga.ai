
"""Satellite Engine: «нулевая стоимость» рождения микро-агента.

Материнская модель (BIM) профилирует задачу -> выбирает подмножество
VQ-якорей из кодовой VSA-памяти -> резонатор собирает новую комбинацию
-> спутник получает самоописание (BIM-карту) и Π^k-операторы.

Всё без обучения с нуля: экспорт якорей + операторов из общей памяти.
"""

from __future__ import annotations

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from antitf.rust_bridge import packed_to_torch
from astral.code_memory import CodeQueryEngine
from astral.resonator import VQResonator


TASK_KEYWORDS = {
    "kernel": ["mm", "sched", "vmalloc", "swap", "page", "hugetlb", "schedule"],
    "monitor": ["vmalloc", "schedule", "swap", "alloc", "free", "queue"],
    "json": ["parse", "from_json", "gson", "add"],
    "parser": ["parse", "token", "lexer", "json"],
    "agent": ["add", "parse", "read", "request"],
}


def profile_task(task: str, code_engine) -> list[str]:
    """Выборка релевантных VQ-якорей из кодовой памяти под задачу."""
    kw = set()
    for word in re.findall(r"[a-zа-яё]+", task.lower()):
        for cat, keys in TASK_KEYWORDS.items():
            if word in keys:
                kw.add(word)
    anchors = []
    for k in kw:
        hits = code_engine.query(k, 2)
        anchors += [nm for _, _, nm in hits]
    return anchors[:12]


def build_satellite(binder, code_engine, task: str) -> dict:
    """Профиль -> якоря -> резонатор -> спутник с BIM-самоописанием."""
    anchors = profile_task(task, code_engine)
    if not anchors:
        anchors = ["vmalloc_init", "schedule", "parse", "add"]
    print(f"[satellite] задача: '{task}'")
    print(f"[satellite] VQ-якоря (экспорт из общей памяти): {anchors}")

    # резонатор собирает комбинацию якорей = «мозг» спутника
    res = VQResonator(binder, anchors, dim=2048, iters=30)
    a = anchors[0]
    b = anchors[len(anchors) // 2] if len(anchors) > 1 else anchors[0]
    S = torch.sign(
        packed_to_torch(np.asarray(binder.bind_batch([[a]])))[0] *
        packed_to_torch(np.asarray(binder.bind_batch([[b]])))[0])
    x, y = res.recover_pair(S, n_restarts=8)
    print(f"[satellite] резонатор: S = {a} ⊗ {b} -> ({x}, {y})")

    # BIM-самоописание спутника (карта «как думает и что потребляет»)
    n_anchors = len(anchors)
    params_m = round(n_anchors * 2048 * 5 / 1e6 + 0.2, 2)   # ~якоря + 2 малых эксперта
    vram_mb = round(params_m * 4, 1)                         # FP32
    ops = [f"Π^{k}" for k in range(4)]                       # пермутационные операторы
    sat = {
        "name": f"satellite_{re.sub(r'\\W+', '_', task)[:24]}",
        "task": task,
        "anchors_exported": n_anchors,
        "params_M": params_m,
        "vram_MB": vram_mb,
        "operators": ops,
        "resonator_pair": (x, y),
        "bim_self_description": (
            f"Спутник использует {n_anchors} VQ-якорей кодовой памяти, "
            f"операторы {ops}, потребляет ~{vram_mb}МБ VRAM. Материнские "
            f"эмбеддинги/словари НЕ копируются — общая VSA-память 32768-bit."),
    }
    return sat


def main():
    binder = fuga_core.HybridBinder(2048)
    ce = CodeQueryEngine(binder, "fuga_memory_code")
    n = ce.load_index_from_disk()
    print(f"[mother] кодовая память: {n} символов (общая, не копируется)")

    for task in ("легкий агент мониторинга ядра Linux",
                 "парсер json документов"):
        sat = build_satellite(binder, ce, task)
        print("  ->", sat["name"], "|", sat["bim_self_description"], "\n")


if __name__ == "__main__":
    main()
