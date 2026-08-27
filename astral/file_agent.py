#!/usr/bin/env python3
"""FileAgent: автономное создание и работа с файлами (улучшение Астрала).

ИИ сам генерирует код модуля -> ПИШЕТ файл на диск -> валидирует
(L1 py_compile + L2 исполнение) -> регистрирует в BIM -> возвращает
импортируемый модуль. Файлы появляются без внешнего вмешательства.
"""

from __future__ import annotations


import importlib.util
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from astral.bim_engine import BIMEngine, BIMNode


class FileAgent:
    def __init__(self, binder, bim: BIMEngine | None = None,
                 work_dir: str = "ai_modules"):
        self.binder = binder
        self.bim = bim
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.created: list[str] = []

    def create_module(self, name: str, code: str, deps: list[str] | None = None,
                      validate_run=None) -> dict:
        """Полный цикл: генерация -> запись файла -> валидация -> BIM."""
        path = os.path.join(self.work_dir, f"{name}.py")
        # 1. пишем файл (ИИ создал код -> файл появляется на диске)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(code))
        # 2. L1: синтаксис
        l1_ok = True
        try:
            compile(open(path, encoding="utf-8").read(), path, "exec")
        except SyntaxError as e:
            l1_ok = False
        # 3. L2: исполнение (если задана функция проверки)
        run = None
        if l1_ok and validate_run is not None:
            run = validate_run(path)
        # 4. регистрация в BIM
        if self.bim is not None:
            self.bim.add_component(
                BIMNode(name, "ai_module", vram_gb=0.0,
                        throughput=f"l1={l1_ok}, run={bool(run and run.get('ok'))}"),
                deps=deps or [])
        rec = {"name": name, "path": path, "l1_ok": l1_ok, "run": run,
               "file_exists": os.path.exists(path),
               "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0}
        self.created.append(name)
        return rec

    def load_module(self, name: str):
        """Загрузить созданный модуль как импортируемый объект."""
        path = os.path.join(self.work_dir, f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# ---------- демо: ИИ создаёт модуль VSA-расстояния ----------
VSADIST_CODE = '''
"""vsa_distance: ИИ-созданный модуль (autonomous file generation).

Считает расстояние Хэмминга между двумя VSA-гипервекторами
на нативном Rust-ядре fuga_core. Создан ИИ через FileAgent.
"""
import sys, os
sys.path.insert(0, {root!r})
import numpy as np
import fuga_core
from antitf.rust_bridge import packed_to_torch


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Хэмминг между двумя packed u64 гипервекторами."""
    return int(np.unpackbits(np.bitwise_xor(
        a.view(np.uint8), b.view(np.uint8))).sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Косинус между двумя ±1 гипервекторами (через Rust-биндер)."""
    ta = packed_to_torch(a[None] if a.ndim == 1 else a).float().flatten()[:2048]
    tb = packed_to_torch(b[None] if b.ndim == 1 else b).float().flatten()[:2048]
    return float((ta * tb).mean())


def demo_distance(binder_name="anchor", other="anchor"):
    binder = fuga_core.HybridBinder(2048)
    a = np.asarray(binder.bind_batch([[binder_name]]))[0]
    b = np.asarray(binder.bind_batch([[other]]))[0]
    return {"hamming": hamming(a, b), "cosine": cosine(a, b)}


if __name__ == "__main__":
    r = demo_distance()
    print(r)
'''


def run_vsadist(path: str) -> dict:
    import subprocess
    r = subprocess.run([sys.executable, path], capture_output=True,
                       text=True, timeout=15)
    return {"ok": r.returncode == 0, "output": r.stdout.strip(),
            "stderr": r.stderr[-200:]}


def main():
    binder = fuga_core.HybridBinder(2048)
    bim = BIMEngine(binder)
    bim.add_component(BIMNode("vsa_memory", "memory", vram_gb=0.3,
                              throughput="30K facts"), [])
    bim.add_component(BIMNode("fast_vsa", "rust_core",
                              throughput="156K steps/s"), ["vsa_memory"])

    agent = FileAgent(binder, bim)
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    code = VSADIST_CODE.replace("{root!r}", repr(root))

    print("[file-agent] ИИ создаёт модуль vsa_distance.py ...")
    rec = agent.create_module("vsa_distance", code,
                              deps=["fast_vsa", "vsa_memory"],
                              validate_run=run_vsadist)
    print(f"  файл создан: {rec['file_exists']} ({rec['size_bytes']} байт)")
    print(f"  L1 (синтаксис): {rec['l1_ok']}")
    print(f"  L2 (исполнение): {rec['run'] and rec['run'].get('ok')} "
          f"-> {rec['run'] and rec['run'].get('output')}")

    # ИИ использует созданный модуль
    mod = agent.load_module("vsa_distance")
    out = mod.demo_distance("vmalloc_init", "schedule")
    print(f"  использование модуля: hamming={out['hamming']} "
          f"cosine={out['cosine']:.3f}")

    print(f"\n[BIM] узлов после создания модуля: {len(bim.nodes)}")
    print(f"[files] создано ИИ: {agent.created}")


if __name__ == "__main__":
    main()