from __future__ import annotations

"""ProceduralWorldGen: детерминированные пары S(t)->S(t+1) без диска."""

import numpy as np


class ProceduralWorldGen:
    def __init__(self, vsa_dim: int = 32768, n_basis: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vsa_dim = vsa_dim
        self.basis_vectors = rng.choice([-1, 1], size=(n_basis, vsa_dim)).astype(np.int8)
        self.rng = np.random.default_rng(seed + 1)

    def generate_step(self):
        idx = self.rng.choice(len(self.basis_vectors), size=4, replace=False)
        st = np.sign(np.sum(self.basis_vectors[idx], axis=0))
        st[st == 0] = 1
        shift = int(self.rng.integers(1, 32))
        action_op = np.roll(self.basis_vectors[idx[0]].astype(np.int16), shift)
        st_next = np.sign(st.astype(np.int16) * action_op)
        st_next[st_next == 0] = 1
        return {"state_t": st.astype(np.int8),
                "action": shift,
                "state_next": st_next.astype(np.int8)}
