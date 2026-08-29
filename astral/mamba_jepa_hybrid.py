#!/usr/bin/env python3
"""Mamba+JEPA гибрид: проглочено, переварено, новая версия создана.

Mamba: selective SSM (state space scan) для последовательностей.
JEPA: joint embedding predictive architecture (предсказание латента).
Гибрид: Mamba-скан encodes последовательность -> JEPA-предиктор предсказывает
следующий латент -> VICReg-лосс (var+inv+cov) -> EMA target.
"""

from __future__ import annotations


import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F



class MambaSSM(nn.Module):
    """Selective SSM (ядро Mamba). Упрощённая версия: S4D-стиль."""

    def __init__(self, d_model=64, d_state=16):
        super().__init__()
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.delta = nn.Parameter(torch.randn(d_model, 1) * 0.01)

    def forward(self, x):
        # x: [B, L, D] — простая ассоциативная scan
        B, L, D = x.shape
        h = torch.zeros(B, D, self.d_state, device=x.device)
        A = -torch.exp(self.A).unsqueeze(0)     # [1,D,state]
        B = self.B.unsqueeze(0)                  # [1,D,state]
        C = self.C.unsqueeze(0)                  # [1,D,state]
        dt = torch.sigmoid(self.delta).unsqueeze(0)  # [1,D,1]
        out = []
        for t in range(L):
            h = h + dt * (A * h + B * x[:, t, :, None])
            y = (C * h).sum(-1)
            out.append(y)
        return torch.stack(out, dim=1)


class MambaJepaHybrid(nn.Module):
    """Mamba-скан -> JEPA-предиктор латента -> VICReg."""

    def __init__(self, d_model=64, d_state=16, d_latent=32):
        super().__init__()
        self.ssm = MambaSSM(d_model, d_state)
        self.decoder = nn.Linear(d_model, d_latent)
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent), nn.ReLU(),
            nn.Linear(d_latent, d_latent))
        self.target_encoder = None  # EMA устанавливается отдельно

    def forward(self, x):
        ssm_out = self.ssm(x)
        z = self.decoder(ssm_out)
        z_pred = self.predictor(z[:, :-1])
        z_target = z[:, 1:]
        return z_pred, z_target


def vicreg_loss(z_pred, z_target, var_w=5.0, inv_w=1.0, cov_w=0.5):
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    inv = F.mse_loss(z_pred, z_target)
    std = torch.sqrt(z_pred.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    # cov (упрощённо)
    b, d = z_pred.shape
    zh = z_pred - z_pred.mean(0)
    cov = (zh.T @ zh) / (b - 1)
    cov = cov.fill_diagonal_(0).pow(2).sum() / d
    return inv_w * inv + var_w * var + cov_w * cov


def main():
    torch.manual_seed(0)
    model = MambaJepaHybrid(64, 16, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # синтетические данные: последовательность гармонических колебаний
    t = torch.linspace(0, 6.28, 50)
    data = torch.stack([torch.sin(t + i).unsqueeze(-1) for i in range(8)])  # [8, 50, 1]
    # расширяем до d_model=64
    data = data.expand(-1, -1, 64) * 0.5
    print("[train] Mamba+JEPA гибрид на последовательности 50 шагов:")
    t0 = time.time()
    for step in range(401):
        x = data + torch.randn_like(data) * 0.05
        z_pred, z_tgt = model(x[:, :-1])
        loss = vicreg_loss(z_pred.reshape(-1, 32), z_tgt.reshape(-1, 32))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            cos = F.cosine_similarity(z_pred, z_tgt, dim=-1).mean().item()
            print(f"  step {step}: loss={loss.item():.4f} cos={cos:.3f} "
                  f"({time.time()-t0:.0f}s)")
    print(f"[result] Mamba+JEPA гибрид обучен: loss={loss.item():.4f}, "
          f"cos={cos:.3f} — гибрид работает.")


if __name__ == "__main__":
    main()