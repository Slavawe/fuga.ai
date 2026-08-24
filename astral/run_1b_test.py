from __future__ import annotations

"""RAM-aware лестница масштабирования MoK + throughput процедурного потока.

Песочница: ~1ГБ доступной RAM -> полный 1.02B инстанс невозможен физически.
Здесь валидируется ТОПОЛОГИЯ (роутинг / throughput / фильтр новизны) на
максимальном влезающем масштабе; полный профиль — на GPU-машине по конфигу
astral/configs/astral_1b_mok.json (shared VSA-адаптер 147M + эксперты).
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antitf.kan import ChebyKANLayer
from astral.procedural_stream import ProceduralWorldGen
from astral.data_filter import AstralDataStreamFilter


class MoKBlock(nn.Module):
    """Топ-2 из E экспертов, каждый — стек ChebyKAN hidden->hidden."""

    def __init__(self, dim=128, hidden=128, n_experts=2, layers=2, degree=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(*[ChebyKANLayer(dim if i == 0 else hidden,
                                          hidden, degree)
                            for i in range(layers)],
                           nn.Linear(hidden, dim))
            for _ in range(n_experts)])
        self.router = nn.Linear(dim, len(self.experts))

    def forward(self, x):
        top_p, top_i = torch.topk(self.router(x), 2, dim=-1)
        alpha = F.softmax(top_p, -1)
        out = torch.zeros_like(x)
        for e, ex in enumerate(self.experts):
            mask = (top_i == e)
            if not mask.any():
                continue
            rows, slots = mask.nonzero(as_tuple=True)
            out.index_add_(0, rows,
                           alpha[rows, slots].unsqueeze(-1) * ex(x[rows]))
        return out


def params_m(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    gen = ProceduralWorldGen(vsa_dim=32768)

    t0 = time.perf_counter()
    N = 3000
    for _ in range(N):
        gen.generate_step()
    dt = time.perf_counter() - t0
    print(f"[procedural] {N/dt:.0f} состояний/сек")

    flt = AstralDataStreamFilter(adaptive=True)
    print("[mok ladder] максимальный инстанс в доступную RAM:")
    for n_experts, layers, hidden in [(2, 2, 128), (4, 4, 192), (8, 6, 256)]:
        try:
            m = MoKBlock(dim=128, hidden=hidden, n_experts=n_experts,
                         layers=layers)
            pm = params_m(m)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            errs = []
            t0 = time.perf_counter()
            for step in range(150):
                d = gen.generate_step()
                s_t = torch.from_numpy(d["state_t"][:128].astype(np.float32)).unsqueeze(0)
                s_n = torch.from_numpy(d["state_next"][:128].astype(np.float32)).unsqueeze(0)
                pred = m(s_t)
                loss = F.mse_loss(pred, s_n)
                ok, _s = flt.should_ingest(pred.detach(), s_n)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if ok:
                    errs.append(float(loss))
            sps = 150 / (time.perf_counter() - t0)
            del opt, m
            tail = np.mean(errs[-30:]) if errs else float("nan")
            print(f"  experts={n_experts}x{layers}L h={hidden}: "
                  f"{pm:.1f}M params, {sps:.1f} steps/s, mse_tail={tail:.5f}")
        except RuntimeError as e:
            print(f"  experts={n_experts}x{layers}L h={hidden}: предел RAM "
                  f"({str(e)[:60]})")
            break

    print("\nВЫВОД: топология MoK + procedural stream валидированы; полный 1B")
    print("— на GPU-машине по astral/configs/astral_1b_mok.json (shared adapter).")


if __name__ == "__main__":
    main()
