
"""vsa_distance: ИИ-созданный модуль (autonomous file generation).

Считает расстояние Хэмминга между двумя VSA-гипервекторами
на нативном Rust-ядре fuga_core. Создан ИИ через FileAgent.
"""
import sys, os
sys.path.insert(0, '/home/slava/Anti-Tronsformers')
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
