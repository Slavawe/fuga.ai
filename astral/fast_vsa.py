from __future__ import annotations

"""FastVSAEngine: битовые операции VSA на нативном Rust (PyO3).

Перенос горячего пути из Python-bigint: XOR-binding, битовая ротация,
мажоритарный бандлинг, конверсия в ±1 — всё на u64 словах без создания
промежуточных больших целых.
"""

from __future__ import annotations

import numpy as np
import torch

from fuga_core import FastVSA, PerModalFilterRust

__all__ = ["FastVSAEngine", "RustPerModalFilter"]


class FastVSAEngine:
    """Обёртка над Rust-реализацией. Совместим с конвенцией
    ProceduralWorldGen v3 (packed little-endian uint8)."""

    def __init__(self, dim_bits: int = 32768):
        self._engine = FastVSA(dim_bits)
        self.dim = dim_bits

    def random_state(self) -> np.ndarray:
        return np.asarray(self._engine.random_state())

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR-связывание двух packed состояний."""
        return np.asarray(self._engine.bind(a, b))

    def rot(self, x: np.ndarray, shift: int) -> np.ndarray:
        """Циклический битовый сдвиг."""
        return np.asarray(self._engine.rotate(x, shift))

    def bundle(self, states: list[np.ndarray]) -> np.ndarray:
        """Побитовое большинство по списку packed состояний."""
        return np.asarray(self._engine.bundle(states))

    @staticmethod
    def to_bipolar(packed: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(FastVSA.packed_to_f32(np.asarray(packed)))
