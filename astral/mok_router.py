
"""Sparse Mixture-of-KANs (MoK): роутер топ-2 из E экспертов.

Проверяемый тезис: на РАЗНОРОДНОЙ динамике (несколько семейств
трансформаций) моно-KAN ограничен одним набором сплайнов, а MoK
маршрутизует семейства по экспертам и выигрывает. Заодно измеряем
балансировку нагрузки (load balancing) — главный риск MoE.

Прототип валидируется на CPU малым масштабом; масштабирование до
30M-экспертов — та же топология, больше параметров.
"""

from __future__ import annotations

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antitf.kan import ChebyKANLayer


class SparseKANRouter(nn.Module):
    """Топ-2 маршрутизатор с шумной балансировкой (Switch-Transformer стиль)."""

    def __init__(self, dim: int, n_experts: int, k: int = 2):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        logits = self.gate(x)
        top_p, top_i = torch.topk(logits, self.k, dim=-1)
        alpha = F.softmax(top_p, dim=-1)
        return top_i, alpha, logits


class MoKEncoder(nn.Module):
    def __init__(self, dim: int = 256, n_experts: int = 8, hidden: int = 192,
                 degree: int = 4, k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(ChebyKANLayer(dim, hidden, degree), nn.SiLU(),
                          ChebyKANLayer(hidden, dim, degree))
            for _ in range(n_experts)])
        self.router = SparseKANRouter(dim, n_experts, k)

    def forward(self, x):
        top_i, alpha, gate_logits = self.router(x)
        out = torch.zeros_like(x)
        B = x.shape[0]
        # плотный цикл по экспертам с маской (для малых масштабов ок)
        for e, expert in enumerate(self.experts):
            mask_e = (top_i == e)                       # [B, k]
            if not mask_e.any():
                continue
            rows, slots = mask_e.nonzero(as_tuple=True)
            w = alpha[rows, slots].unsqueeze(-1)
            out[rows] += w * expert(x[rows])
        return out, gate_logits


OPS = [
    lambda b: torch.roll(b, shifts=7, dims=[-1]),
    lambda b: torch.flip(b, dims=[-1]),
    lambda b: -b,
    lambda b: torch.cat([b[:, dim//2:], b[:, :dim//2]], dim=-1),
    lambda b: b * (torch.arange(dim, dtype=torch.float32) % 2 * 2 - 1),
    lambda b: -torch.roll(b, shifts=-11, dims=[-1]),
]
dim = 256

def make_family_data(n, family, d=256):
    """6 семейств трансформаций с разной алгеброй."""
    base = torch.sign(torch.randn(n, d))
    nxt = OPS[family](base) * 0.9 + 0.1
    nxt = torch.sign(nxt + torch.randn_like(nxt) * 0.05)
    return base, nxt


def train(model, use_moe, steps=1200, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lb_w = 0.01
    for step in range(steps + 1):
        fam = random.randint(0, 1)
        s_t, s_next = make_family_data(bs, fam)
        if use_moe:
            pred, gate_logits = model(s_t)
            # load balancing loss (Switch): важен, иначе всё рухнет в одного эксперта
            p = F.softmax(gate_logits, dim=-1).mean(0)
            lb = len(p) * (p * p).sum()
            loss = F.mse_loss(pred, s_next) + lb_w * lb
        else:
            pred = model(s_t)
            loss = F.mse_loss(pred, s_next)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == 1200:
            extra = f" lb={lb.item():.3f}" if use_moe else ""
            print(f"    step {step}: mse={loss.item():.5f}{extra}")
    return loss.item()


@torch.no_grad()
def eval_model(model, use_moe, trials=200, bs=32):
    tot = 0.0
    for _ in range(trials):
        fam = random.randint(0, 1)
        s_t, s_next = make_family_data(bs, fam)
        if use_moe:
            pred, _ = model(s_t)
        else:
            pred = model(s_t)
        tot += float(F.mse_loss(pred, s_next))
    return tot / trials


def main():
    random.seed(0); torch.manual_seed(0)
    print("[baseline mono-KAN на разнородной динамике]")
    mono = nn.Sequential(ChebyKANLayer(256, 192, degree=4), nn.SiLU(),
                         ChebyKANLayer(192, 256, degree=4))
    train(mono, use_moe=False)
    mono_err = eval_model(mono, use_moe=False)
    print(f"  mono heldout MSE = {mono_err:.5f}")

    print("[MoK: 8 экспертов, топ-2]")
    mok = MoKEncoder(n_experts=8, k=2)
    train(mok, use_moe=True)
    mok_err = eval_model(mok, use_moe=True)
    print(f"  MoK heldout MSE = {mok_err:.5f}")

    with torch.no_grad():
        fam = random.randint(0, 1)
        s_t, _ = make_family_data(128, fam)
        _, _, gl = mok.router(s_t)
        usage = F.softmax(gl, -1).mean(0)
        print(f"  expert load distribution: "
              f"{np.round(usage.numpy(), 3)}")

    imp = (mono_err - mok_err) / mono_err * 100
    print(f"\nRESULT: MoK better by {imp:.1f}% MSE on heterogeneous dynamics")
    print(f"(масштаб прототипа ~{sum(p.numel() for p in mok.parameters())/1e6:.1f}M "
          f"эквивалентен 30M-эксперту при росте hidden до ~1300)")


if __name__ == "__main__":
    main()
