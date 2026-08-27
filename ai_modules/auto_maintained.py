
"""auto_maintained: модуль, созданный Auto-Maintainer (Evolution V3).
Хэлпер для VSA-радиусов: нормировка гипервекторов в ±1.
"""
import sys, os
sys.path.insert(0, '/home/slava/Anti-Tronsformers')
import numpy as np
import fuga_core
from antitf.rust_bridge import packed_to_torch

def normalize_hv(packed) -> np.ndarray:
    """Нормирует packed u64 -> биполярный ±1 вектор через Rust-ядро."""
    return np.asarray(packed_to_torch(packed[None]))[0]

def demo():
    binder = fuga_core.HybridBinder(2048)
    a = np.asarray(binder.bind_batch([["anchor"]]))[0]
    b = normalize_hv(a)
    return float((b * b).mean())
