
"""VQ-Resonator Network: разложение суперпозиции HV на составляющие.

Классика (Frady/OSU): S = X ⊗ Y — факторизованная суперпозиция.
Резонанс: x̂ = sign(S * ŷ), cleanup; ŷ = sign(S * x̂), cleanup; повтор.
Если оба фактора неизвестны — резонанс восстанавливает пару (известный
принцип «комбинаторного изобретения»: новая комбинация из якорей памяти).
"""

from __future__ import annotations

from __future__ import annotations

import numpy as np
import torch

import sys, os

import fuga_core
from antitf.rust_bridge import packed_to_torch


class VQResonator:
    def __init__(self, binder, anchors: list[str], dim=2048, iters=40):
        self.binder = binder
        self.dim = dim
        self.iters = iters
        # кодбук якорей (детерминированные HV через binder)
        self.anchors = anchors
        self.codebook = torch.stack([
            packed_to_torch(np.asarray(binder.bind_batch([[a]])))[0]
            for a in anchors
        ]).float()   # [K, dim] bipolar

    def _cleanup(self, v: torch.Tensor) -> torch.Tensor:
        """Проекция на ближайший якорь (детерминированно)."""
        sims = self.codebook @ v
        best = int(sims.argmax())
        return self.codebook[best].clone()

    def _softclean(self, v: torch.Tensor, temp=3.0) -> torch.Tensor:
        """Мягкая версия: взвешенная суперпозиция якорей — «новое
        смешанное состояние», если резонанс не схлопнулся."""
        sims = self.codebook @ v
        w = torch.softmax(sims * temp, dim=0)
        return (w[:, None] * self.codebook).sum(0)

    def _resonate(self, S, seed):
        rng = torch.Generator().manual_seed(seed)
        x = torch.sign(torch.randn(self.dim, generator=rng))
        y = torch.sign(torch.randn(self.dim, generator=rng))
        for _ in range(self.iters):
            x = torch.sign(S * y)
            x = self._cleanup(x)
            y = torch.sign(S * x)
            y = self._cleanup(y)
        # энергия разложения: ||S - x⊗y|| мала при правильной паре
        err = float((S - x * y).norm())
        return x, y, err

    def recover_pair(self, S: torch.Tensor, n_restarts: int = 8,
                     n_iter: int | None = None):
        """S = X ⊗ Y с несколькими рестартами: выбираем разложение с
        минимальной энергией ||S - x⊗y|| (анти-аттрактор)."""
        self.iters = n_iter or self.iters
        best = None
        for r in range(n_restarts):
            x, y, err = self._resonate(S, seed=100 + r)
            if best is None or err < best[0]:
                best = (err, x, y)
        _, x, y = best
        return self._name_of(x), self._name_of(y)

    def _name_of(self, v) -> str:
        sims = self.codebook @ v
        return self.anchors[int(sims.argmax())]


def main():
    torch.manual_seed(0)
    binder = fuga_core.HybridBinder(2048)

    # якоря — реальные символы кода из нашей памяти
    anchors = [
        "vmalloc_init", "schedule", "gson.getAdapter", "from_json",
        "hugetlb_parse_params", "readRequestJSON", "swap_cluster_discard",
        "add", "parse", "main", "page", "struct", "push_back", "method",
        "loop", "alloc", "free", "hash", "lock", "queue",
    ]
    res = VQResonator(binder, anchors, dim=2048, iters=30)

    def hv(name):
        return packed_to_torch(np.asarray(binder.bind_batch([[name]])))[0]

    # ==== 1. Точность разложения пары ====
    rng = np.random.default_rng(1)
    trials = 50
    correct = 0
    for t in range(trials):
        a = anchors[int(rng.integers(len(anchors)))]
        b = anchors[int(rng.integers(len(anchors)))]
        S = torch.sign(hv(a) * hv(b))
        x, y = res.recover_pair(S)
        correct += int((x == a and y == b) or (x == b and y == a))
    print(f"[resonator] точность разложения пары: {correct}/{trials} "
          f"({correct/trials:.0%})")

    # ==== 2. Комбинаторное изобретение: НОВАЯ пара из якорей ====
    # «новая структура» = пара, закодированная связыванием, которую модель
    # не могла увидеть готовой — резонанс собирает её из двух якорей памяти.
    print("\n[invention] резонанс собирает новые пары из якорей:")
    for a, b in [("vmalloc_init", "schedule"),
                 ("gson.getAdapter", "from_json"),
                 ("hugetlb_parse_params", "readRequestJSON")]:
        S = torch.sign(hv(a) * hv(b))
        x, y = res.recover_pair(S)
        ok = (x == a and y == b) or (x == b and y == a)
        print(f"  S = {a} ⊗ {b} -> ({x}, {y})  {'OK' if ok else 'FAIL'}")

    # ==== 3. Промежуточная фаза (creativity claim) ====
    print("\n[phase] промежуточная фаза даёт смесь (новая структура):")
    S = torch.sign(hv("vmalloc_init") * hv("schedule"))
    x = torch.randn(2048, generator=torch.Generator().manual_seed(2))
    y = torch.randn(2048, generator=torch.Generator().manual_seed(3))
    for _ in range(3):   # недорезонировали — промежуточная фаза
        x = torch.sign(S * y)
        y = torch.sign(S * torch.sign(x))
    soft = res._softclean(x, temp=2.0)
    top_sims = (res.codebook @ soft).topk(3).values
    top_idx = (res.codebook @ soft).topk(3).indices
    print("  топ-3 якоря промежуточной фазы:",
          [res.anchors[int(i)] for i in top_idx],
          "— смешение между парами (новая композиция)")


if __name__ == "__main__":
    main()
