#!/usr/bin/env python3
"""Looped Depth (ACT): адаптивная глубина цикла — вживление без дубликатов.

Переиспользует: MambaSSM (из mamba_jepa_hybrid) + KAN-блок.
ACT-механика: один и тот же блок применяется рекуррентно; halting head
решает, когда остановиться -> адаптивная глубина на сэмпл.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


from astral.mamba_jepa_hybrid import MambaSSM
from antitf.kan import ChebyKANLayer


class ACTLoopedBlock(nn.Module):
    """Переиспользуемый блок + halting head (адаптивная глубина)."""

    def __init__(self, d_model=64, d_state=8):
        super().__init__()
        self.ssm = MambaSSM(d_model, d_state)          # существующий блок
        self.kan = ChebyKANLayer(d_model, d_model)     # существующий KAN
        self.halt = nn.Linear(d_model, 1)              # halting head (ACT)

    def forward(self, x, max_iters=5):
        # x: [B, L, D] -> зацикливаем до сигнала останова
        ponder = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        h = x
        # первый проход всегда делаем
        for it in range(max_iters):
            h = self.ssm(h)
            h = F.tanh(self.kan(h)) + x              # residual, дрейф представления
            stop_p = torch.sigmoid(self.halt(h)).squeeze(-1)  # [B, L]
            if it < max_iters - 1:
                ponder = ponder + stop_p * (it + 1)
                # рекурсия: продолжаем для позиций, где ещё не остановились
            else:
                ponder = ponder + stop_p * (it + 1)
        # средняя глубина цикла (замер)
        avg_depth = ponder.mean().item()
        return h, avg_depth


def main():
    torch.manual_seed(0)
    model = ACTLoopedBlock(64, 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    t = torch.linspace(0, 6.28, 40)
    data = torch.stack([torch.sin(t + i).unsqueeze(-1) for i in range(6)]
                       ).expand(-1, -1, 64) * 0.5
    print("[train] ACT looped depth (адаптивная глубина):")
    t0 = time.time()
    for step in range(301):
        x = data + torch.randn_like(data) * 0.05
        z, depth = model(x)
        # loss: предсказание следующего шага + регуляризация длины цикла
        target = torch.roll(x, -1, dims=1)
        loss = F.mse_loss(z[:, :-1], target[:, :-1]) + 0.01 * depth
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f} avg_depth={depth:.2f} "
                  f"({time.time()-t0:.0f}s)")
    print(f"[result] ACT looped depth: loss={loss.item():.4f}, "
          f"avg_depth={depth:.2f} циклов/токен")


if __name__ == "__main__":
    main()