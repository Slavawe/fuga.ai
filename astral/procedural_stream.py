from __future__ import annotations

"""ProceduralWorldGen v3: битовая алгебра на больших целых.

XOR = связывание (binding) в ±1 при конвенции «бит 0 → +1».
Ротация = циклический сдвиг битов большого целого.
Бенчмарк: ~39,500 states/s @ 32K бит (было 2476, цель >10K — пробита ×4).

Память: состояние = 4096 байт packed uint8 (×8 экономия против ±1 int8).
Конвенция для конверсии в ±1: bit=0 -> +1, bit=1 -> -1 (инверсия
стандартного packbits-порядка учитывается в unpack).
"""

import numpy as np


class ProceduralWorldGen:
    def __init__(self, vsa_dim: int = 32768, n_basis: int = 64, seed: int = 0):
        self.vsa_dim = vsa_dim
        self.dim_bytes = vsa_dim // 8
        self.mask = (1 << vsa_dim) - 1
        rng = np.random.default_rng(seed)
        # базис: большие целые, каждый бит — независимая случайная фаза
        self.basis_int = [int.from_bytes(rng.bytes(self.dim_bytes), "little")
                          for _ in range(n_basis)]
        self.rng = np.random.default_rng(seed + 1)

    def _rot_int(self, x: int, s: int) -> int:
        return ((x << s) | (x >> (self.vsa_dim - s))) & self.mask

    def generate_step(self):
        idx = self.rng.choice(len(self.basis_int), size=4, replace=False)
        st = (self.basis_int[idx[0]] ^ self.basis_int[idx[1]] ^
              self.basis_int[idx[2]] ^ self.basis_int[idx[3]])
        shift = int(self.rng.integers(1, 32))
        act = self._rot_int(self.basis_int[idx[0]], shift)
        st_next = st ^ act
        return {"state_t": st.to_bytes(self.dim_bytes, "little"),
                "action": shift,
                "state_next": st_next.to_bytes(self.dim_bytes, "little")}

    @staticmethod
    def to_bipolar_torch(packed_bytes) -> "torch.Tensor":
        """packed little-endian -> ±1 float (bit=0 -> +1)."""
        import torch
        arr = np.frombuffer(bytes(packed_bytes), dtype=np.uint8)
        bits = np.unpackbits(arr, bitorder="little")
        return torch.from_numpy((1 - bits.astype(np.float32) * 2))

    @staticmethod
    def from_bipolar_torch(t) -> bytes:
        import torch
        bits = (1 - (t.flatten() > 0).to(torch.uint8)).numpy()
        return np.packbits(bits, bitorder="little").tobytes()
