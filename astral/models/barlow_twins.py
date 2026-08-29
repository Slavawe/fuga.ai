"""Barlow Twins — анти-коллапс через кросс-корреляцию (бало-система).

Дополняет VICReg (mamba_jepa_hybrid.py): вместо трёх слагаемых
(inv+var+cov) — ОДНА кросс-корреляционная матрица между двумя
представлениями одного образца, стремящаяся к identity.

  C[i,j] = sum_b z_a[b,i] · z_b[b,j] / B  (нормированные латенты)
  L = -log(diag C) + λ·||off-diag C||²

Свойства:
- diag → 1: представления инвариантны к аугментации (информация сохранена)
- off-diag → 0: представления некоррелированы (нет коллапса/редундантности)
- Один гиперпараметр λ (не 3 как VICReg) — проще калибровать.

Использование: barlow_loss(z_a, z_b, lambda_=0.005) — напрямую в
JEPA-тренинге, где z_a = предсказанный латент, z_b = EMA-таргет.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def barlow_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    lambda_: float = 0.005,
) -> torch.Tensor:
    """Barlow Twins loss (Zbontar et al. 2021).

    Args:
        z_a: [B, D] представление 1 (напр. предсказанный латент).
        z_b: [B, D] представление 2 (напр. EMA-таргет).
        lambda_: вес off-diagonal членов.

    Returns:
        Скалярный лосс.
    """
    # Нормировка по батчу (центрирование + стандартизация признаков)
    def _norm(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, unbiased=False, keepdim=True) + 1e-6)
        return z

    za = _norm(z_a)
    zb = _norm(z_b)
    B, D = za.shape

    # Кросс-корреляционная матрица C [D, D]
    c = (za.T @ zb) / B

    # diag → 1 (инвариантность), off-diag → 0 (некоррелированность)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.fill_diagonal_(0).pow_(2).sum()
    loss = on_diag + lambda_ * off_diag
    return loss / D


class BarlowTwinsHead(nn.Module):
    """Проекционная голова для Barlow Twins.

    z_pred → projector → z_a;  z_target → EMA projector → z_b.
    Используется в JEPA-цикле: predict_next(z) кондиционирует
    предиктор, barlow_loss(z_a, z_b) — анти-коллапс.
    """

    def __init__(self, in_dim: int, hidden: int | None = None, out_dim: int | None = None):
        super().__init__()
        h = hidden or in_dim
        d = out_dim or in_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Linear(h, d),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


def smoke_test() -> dict:
    """Barlow: 1) сходится на парных латентах, 2) бьёт коллапс."""
    torch.manual_seed(0)
    B, D = 64, 32
    z = torch.randn(B, D)

    # 1) Идентичные представления → loss низкий
    l_same = barlow_loss(z, z)
    # 2) Случайные → высокий (корреляция есть из-за случайного совпадения)
    l_rand = barlow_loss(z, torch.randn(B, D))
    # 3) Коллапс-контроль: если заставить z=const → loss должен расти (наказание)
    z_const = torch.ones(B, D)
    l_collapse = barlow_loss(z_const, z_const)

    return {
        "same_views": round(float(l_same), 4),
        "random_views": round(float(l_rand), 4),
        "collapse_const": round(float(l_collapse), 4),
        "разделяет_коллапс": bool(l_collapse > l_same),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
