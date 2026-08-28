"""self-improved: ACT looped-depth модуль, сгенерирован моделью.
Переиспользует собственные блоки модели: MambaSSM (SSM-scan) и
ChebyKANLayer (сплайны) — БЕЗ дубликатов. Halting head решает глубину.
"""
import sys, os
sys.path.insert(0, '/home/slava/Anti-Tronsformers')

import torch
import torch.nn as nn
import torch.nn.functional as F

from astral.mamba_jepa_hybrid import MambaSSM
from antitf.kan import ChebyKANLayer


class ActLoopedBlock(nn.Module):
    def __init__(self, d_model=64, d_state=8, max_iters=4):
        super().__init__()
        self.ssm = MambaSSM(d_model, d_state)
        self.kan = ChebyKANLayer(d_model, d_model)
        self.halt = nn.Linear(d_model, 1)
        self.max_iters = max_iters

    def forward(self, x):
        h = x
        for _ in range(self.max_iters):
            h = self.ssm(h)
            h = F.tanh(self.kan(h)) + x
        stop = torch.sigmoid(self.halt(h)).squeeze(-1)
        depth = stop.mean().item()
        return h, depth


def main():
    torch.manual_seed(0)
    model = ActLoopedBlock(64, 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t = torch.linspace(0, 6.28, 40)
    data = torch.stack([torch.sin(t + i).unsqueeze(-1) for i in range(6)]
                       ).expand(-1, -1, 64) * 0.5
    for step in range(151):
        x = data + torch.randn_like(data) * 0.05
        z, depth = model(x)
        target = torch.roll(x, -1, dims=1)
        loss = F.mse_loss(z[:, :-1], target[:, :-1]) + 0.01 * depth
        opt.zero_grad(); loss.backward(); opt.step()
    return {{"loss": round(loss.item(), 4), "depth": round(depth, 2)}}
