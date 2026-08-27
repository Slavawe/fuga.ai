#!/usr/bin/env python3
"""make_duo_lib: ИИ создаёт Python-библиотеку, объединяющую PyTorch и JAX.

Через FileAgent: генерация кода duo_nn.py -> запись файла -> L1 компиляция
-> L2 исполнение (Linear на torch И на jax из одного кода -> сравнение
выходов). Ключевая метрика: max|output_torch - output_jax|.
"""

from __future__ import annotations


import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from astral.file_agent import FileAgent

DUO_NN_CODE = '''
"""duo_nn: единый API поверх PyTorch и JAX (создан ИИ через FileAgent).

Одна библиотека -> два бэкенда: torch / jax. Линейные слои и операции
диспетчеризуются по backend; массивы конвертируются автоматически.
"""
import numpy as np

_B = {"torch": None, "jax": None}

def _load():
    if _B["torch"] is None:
        try:
            import torch
            _B["torch"] = torch
        except Exception:
            pass
    if _B["jax"] is None:
        try:
            import jax.numpy as jnp
            _B["jax"] = jnp
        except Exception:
            pass

_load()

BACKENDS = [k for k, v in _B.items() if v is not None]


def detect_backend(x) -> str:
    if _B["torch"] is not None and isinstance(x, _B["torch"].Tensor):
        return "torch"
    if _B["jax"] is not None and isinstance(x, _B["jax"].ndarray):
        return "jax"
    return "numpy"


def to_torch(x):
    b = detect_backend(x)
    if b == "torch":
        return x
    return _B["torch"].from_numpy(np.asarray(x))


def to_jax(x):
    b = detect_backend(x)
    if b == "jax":
        return x
    return _B["jax"].array(np.asarray(x))


def to_backend(x, backend):
    return to_torch(x) if backend == "torch" else to_jax(x)


def _arr(x, backend):
    return to_backend(x, backend)


# ---------- операции ----------
def matmul(a, b, backend="torch"):
    a, b = _arr(a, backend), _arr(b, backend)
    return _B[backend].matmul(a, b) if backend == "torch" else _B[backend].matmul(a, b)


def relu(x, backend="torch"):
    x = _arr(x, backend)
    return _B[backend].relu(x) if backend == "torch" else _B[backend].maximum(x, 0)


def softmax(x, backend="torch"):
    x = _arr(x, backend)
    if backend == "torch":
        return _B[backend].softmax(x, dim=-1)
    e = _B[backend].exp(x - _B[backend].max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, backend="torch"):
    x = _arr(x, backend)
    if backend == "torch":
        return _B[backend].layer_norm(x, [x.shape[-1]])
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / _B[backend].sqrt(var + 1e-5)


# ---------- слои ----------
class Linear:
    def __init__(self, in_f: int, out_f: int, backend="torch", seed=0):
        self.backend = backend
        rng = np.random.RandomState(seed)
        self.weight = to_backend(rng.randn(out_f, in_f) * 0.02, backend)
        self.bias = to_backend(np.zeros(out_f), backend)

    def forward(self, x):
        w = _arr(self.weight, self.backend)
        b = _arr(self.bias, self.backend)
        x = _arr(x, self.backend)
        out = matmul(x, w.T, self.backend) + b
        return out


def demo(backend):
    layer = Linear(4, 3, backend=backend, seed=7)
    x = np.random.RandomState(1).randn(2, 4)
    out = layer.forward(x)
    return out


def main():
    torch_out = demo("torch")
    jax_out = demo("jax")
    diff = float(np.abs(np.asarray(torch_out) - np.asarray(jax_out)).max())
    print(f"backend={BACKENDS} max|torch-jax|={diff:.6f}")
    return diff
'''


def run_duo(path: str) -> dict:
    r = subprocess.run([sys.executable, path], capture_output=True,
                       text=True, timeout=30)
    return {"ok": r.returncode == 0, "output": r.stdout.strip(),
            "stderr": r.stderr[-300:]}


def main():
    binder = fuga_core.HybridBinder(2048)
    agent = FileAgent(binder)
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    print("[ИИ] создаю библиотеку duo_nn.py (PyTorch + JAX единый API)...")
    rec = agent.create_module(
        "duo_nn", DUO_NN_CODE.replace("{root!r}", repr(root)),
        deps=["fast_vsa"], validate_run=run_duo)
    print(f"  файл: {rec['path']} ({rec['size_bytes']} байт)")
    print(f"  L1 (компиляция): {rec['l1_ok']}")
    print(f"  L2 (исполнение): {rec['run'] and rec['run'].get('ok')}")
    out = rec["run"] and rec["run"].get("output") or ""
    print(f"  результат: {out}")

    if rec["l1_ok"]:
        mod = agent.load_module("duo_nn")
        print(f"  доступные бэкенды: {mod.BACKENDS}")
        print(f"  тип torch-выхода: {type(mod.demo('torch')).__module__}")
        print(f"  тип jax-выхода:   {type(mod.demo('jax')).__module__}")
    print(f"\n[BIM] модулей создано ИИ: {agent.created}")


if __name__ == "__main__":
    main()