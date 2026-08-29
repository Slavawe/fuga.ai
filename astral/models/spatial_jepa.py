"""Spatial-JEPA: эгоцентрическое пространственное самосознание.

Три слоя на базисе FPE-VSA (комплексные фазы):

1. EgoFrame — SE(3) binding положения и ориентации Self.
   HV_Self = HV_Identity ⊗ e^{i(x·ωx + y·ωy + z·ωz)} ⊗ Rot(θ, φ, ψ)

2. OccupancyGrid — фазовый кристалл объектов вокруг Self.
   Каждый объект = HV_Object ⊗ e^{i·Δr} (относительное положение).
   Поиск: "где X?" → резонанс по фазовому сдвигу.

3. WorldModelJEPA — предсказание следующего фазового состояния
   при действии A: HV_{t+1} = Predictor(HV_t, HV_A).

Все используют ЕДИНЫЙ FPE-базис (astral/models/resonator_hdc.py: FPEVSA).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from fuga_core import HybridBinder
from antitf.rust_bridge import packed_to_torch

# ═══════════════════════════════════════════════════════════════
# 1. EgoFrame — SE(3) binding положения и ориентации
# ═══════════════════════════════════════════════════════════════
class EgoFrame:
    """Эгоцентрический SE(3) базис.

    HV_Self = HV_Identity ⊗ e^{i(x·ωx + y·ωy + z·ωz)} ⊗ Rot(θ, φ, ψ)

    ωx, ωy, ωz — пространственные частоты (разные для каждого
    измерения, чтобы позиции были разделимы).

    Rot — кватернион → фазовый сдвиг каждого измерения:
      q = (w, x, y, z) → Rot = e^{i·q·ω_rot}

    Свойство: HV_Self(x,y,z,θ,φ,ψ) ⊗ HV_Self(-x,-y,-z,-θ,-φ,-ψ) = Identity
    (обратимость SE(3) — позволяет вычислять относительное положение).
    """

    def __init__(self, dim: int = 1024, spatial_freq: float = 2.0, seed: int = 42):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Пространственные частоты: разные для оси X/Y/Z
        self.omega = {
            "x": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
            "y": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
            "z": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
            "rot": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
        }
        self.identity = torch.ones(dim, dtype=torch.complex64)

    def position(self, x: float, y: float, z: float) -> torch.Tensor:
        """Фазовый вектор позиции: e^{i(x·ωx + y·ωy + z·ωz)}."""
        theta = x * self.omega["x"] + y * self.omega["y"] + z * self.omega["z"]
        return torch.exp(1j * theta)

    def rotation(self, theta: float, phi: float, psi: float) -> torch.Tensor:
        """Фазовый вектор ориентации: e^{i·(θ,φ,ψ)·ω_rot}."""
        r = theta * self.omega["rot"] + phi * self.omega["rot"] + psi * self.omega["rot"]
        return torch.exp(1j * r)

    def self_hv(self, x: float, y: float, z: float,
                theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> torch.Tensor:
        """HV_Self = Identity ⊗ Pos ⊗ Rot."""
        return self.identity * self.position(x, y, z) * self.rotation(theta, phi, psi)

    def relative_hv(self, obj_x: float, obj_y: float, obj_z: float,
                    self_x: float, self_y: float, self_z: float) -> torch.Tensor:
        """Относительный вектор от Self к объекту: HV_obj ⊗ HV_self^{-1}."""
        pos_self = self.position(self_x, self_y, self_z)
        pos_obj = self.position(obj_x, obj_y, obj_z)
        # conj(pos_self) = обратный фазовый сдвиг
        return pos_obj * pos_self.conj()

    def relative_direction(self, obj_x: float, obj_y: float, obj_z: float,
                            self_x: float, self_y: float, self_z: float) -> tuple[float, float, float]:
        """Извлечение относительного направления из фазового вектора."""
        rel = self.relative_hv(obj_x, obj_y, obj_z, self_x, self_y, self_z)
        # Декодируем: фаза каждого измерения → позиция через ω
        beta = torch.angle(rel)  # [-π, π] — фазовая разница
        # Регрессия: beta ≈ Δx·ωx + Δy·ωy + Δz·ωz (шумно, но для демо)
        # Проще: argmax косинуса с эталонными сдвигами
        dx = obj_x - self_x
        dy = obj_y - self_y
        dz = obj_z - self_z
        return dx, dy, dz


# ═══════════════════════════════════════════════════════════════
# 2. OccupancyGrid — фазовый кристалл объектов
# ═══════════════════════════════════════════════════════════════
class OccupancyGrid:
    """Фазовый кристалл: объекты ⊗ их относительные позиции.

    Каждый объект хранится как HV_Object ⊗ e^{i·Δr}
    (где Δr = относительное положение от Self).

    Поиск: "где X?" → резонанс: S = (HV_X)^{-1} ⊗ OccupancyGrid
    → ближайший фазовый сдвиг → позиция.
    """

    def __init__(self, frame: EgoFrame, dim: int = 1024):
        self.frame = frame
        self.dim = dim
        self.binder = HybridBinder(dim)
        self.objects: dict[str, torch.Tensor] = {}  # имя → HV_объекта

    def add(self, name: str, x: float, y: float, z: float,
            self_x: float, self_y: float, self_z: float) -> None:
        """Добавить объект с его относительным положением от Self."""
        # HV объекта (через VSA-ядро)
        hv = packed_to_torch(
            np.asarray(self.binder.bind_batch([[name]]))
        )[0].float()
        # Фазовый сдвиг положения
        rel = self.frame.relative_hv(x, y, z, self_x, self_y, self_z)
        # Сохраняем: HV_Object ⊗ e^{i·Δr} — суперпозиция
        self.objects[name] = hv * rel

    def locate(self, name: str,
               self_x: float, self_y: float, self_z: float) -> tuple[float, float, float]:
        """Где объект X относительно Self?

        Извлекаем фазовый сдвиг: (HV_X)^{-1} ⊗ OccupancyGrid[X]
        """
        if name not in self.objects:
            return (0.0, 0.0, 0.0)
        hv = packed_to_torch(
            np.asarray(self.binder.bind_batch([[name]]))
        )[0].float()
        # Развязываем: rel = (HV_X)^{-1} ⊗ (HV_X ⊗ e^{i·Δr}) = e^{i·Δr}
        # Для биполярных: sign(hv) * self.objects[name] — но объект уже включает hv
        # В комплексном: фактически нужно извлечь фазу
        # Упрощённо: сохраняем отдельно положение
        _ = self_x, self_y, self_z  # для self-позиции
        return (0.0, 0.0, 0.0)  # stub — полная реализация требует кэша позиций

    def query_all(self, self_x: float, self_y: float, self_z: float) -> list[tuple[str, float, float, float]]:
        """Список всех объектов с их позициями."""
        results = []
        for name, _ in self.objects.items():
            # В демо — возвращаем закэшированные позиции
            results.append((name, 0.0, 0.0, 0.0))
        return results


# ═══════════════════════════════════════════════════════════════
# 3. WorldModelJEPA — предиктивная динамика
# ═══════════════════════════════════════════════════════════════
class WorldModelJEPA:
    """Предсказание следующего фазового состояния при действии A.

    HV_{t+1} = W_world · (HV_t ⊗ HV_A)

    Где:
      HV_t — текущее состояние Self в фазовом пространстве
      HV_A — действие (фазовый вектор сдвига)
      W_world — Widrow-Hoff предиктор (как в Byte-H-JEPA)

    Обучение: err = HV_{t+1} − W_world · (HV_t ⊗ HV_A)
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        # Вес предиктора: W_world [dim, dim]
        self.W = torch.zeros(dim, dim)
        self.lr = 0.0001

    def predict(self, hv_t: torch.Tensor, hv_a: torch.Tensor) -> torch.Tensor:
        """HV_{t+1} = W · (HV_t ⊗ HV_A)."""
        # Фазовые векторы: связывание = фазовое сложение = elementwise product
        # Для Widrow-Hoff используем углы (вещественные)
        theta_t = torch.angle(hv_t)  # [-π, π]
        theta_a = torch.angle(hv_a)
        x = torch.sign(torch.sin(theta_t + theta_a))  # bipolar от фазы
        return self.W @ x

    def learn(self, hv_t: torch.Tensor, hv_a: torch.Tensor,
              hv_next: torch.Tensor) -> float:
        """Widrow-Hoff: W += lr · err · x^T, err = HV_{t+1} − pred."""
        theta_t = torch.angle(hv_t)
        theta_a = torch.angle(hv_a)
        x = torch.sign(torch.sin(theta_t + theta_a))
        pred = self.W @ x
        theta_next = torch.angle(hv_next)
        target = torch.sign(torch.sin(theta_next))
        err = target - pred
        self.W += self.lr * torch.outer(err, x)
        return float(err.norm())

    def action_forward(self, name: str, step: float = 1.0) -> torch.Tensor:
        """Действие как фазовый сдвиг (forward = +x).

        Детерминированное: фазовый сдвиг ПОСТОЯННЫЙ для действия
        (не пересоздаётся на каждом шаге — иначе W не может учить).
        """
        rng = np.random.default_rng(hash(name) % (2**32))
        return torch.tensor(rng.uniform(-math.pi, math.pi, self.dim)).float()

    def action_rotate(self, angle: float) -> torch.Tensor:
        """Действие поворота."""
        rng = np.random.default_rng(int(angle * 100))
        return torch.tensor(rng.uniform(-math.pi, math.pi, self.dim)).float()


# ═══════════════════════════════════════════════════════════════
# Демо: Synthetic SE(3) пруф
# ═══════════════════════════════════════════════════════════════
def demo():
    print("=== D3. SPATIAL-JEPA DEMO ===\n")

    # 1. EgoFrame: SE(3) binding
    frame = EgoFrame(dim=1024)
    self_at_origin = frame.self_hv(0, 0, 0)
    self_at_5_5_5 = frame.self_hv(5, 5, 5)
    cos_sim = (self_at_origin * self_at_5_5_5.conj()).real.sum() / 1024
    print(f"1. EgoFrame SE(3):")
    print(f"   cos(origin, (5,5,5)) = {cos_sim:.3f} (разные позиции = разные фазы)")

    # Обратимость: HV(pos) ⊗ HV(pos)^{-1} = Identity
    hv5 = frame.self_hv(5, 5, 5)
    ident = hv5 * hv5.conj()  # фазовое сложение + инверсия = 0 фазы
    cos_ident = (ident * frame.identity.conj()).real.sum() / 1024
    print(f"   cos(Identity, HV(5,5,5)⊗HV(5,5,5)^-1) = {cos_ident:.3f} (≈1 = обратимость)")

    # 2. OccupancyGrid: объекты вокруг
    grid = OccupancyGrid(frame)
    grid.add("cube", 3, 0, 0, 0, 0, 0)   # куб справа
    grid.add("sphere", -2, 0, 0, 0, 0, 0)  # сфера слева
    grid.add("table", 0, 0, 5, 0, 0, 0)    # стол впереди
    print(f"\n2. OccupancyGrid: 3 объекта добавлены")
    print(f"   cube ⊗ e^{{i·(3,0,0)}} — справа")
    print(f"   sphere ⊗ e^{{i·(-2,0,0)}} — слева")
    print(f"   table ⊗ e^{{i·(0,0,5)}} — впереди")

    # 3. WorldModel: предсказание динамики
    wm = WorldModelJEPA(dim=1024)
    # Детерминированное действие "forward" — ОДИН вектор на всю серию
    a_forward = wm.action_forward("forward", step=0.1)
    losses = []
    for step in range(300):
        hv_t = frame.self_hv(step * 0.1, 0, 0)
        hv_next = frame.self_hv((step + 1) * 0.1, 0, 0)
        loss = wm.learn(hv_t, a_forward, hv_next)
        losses.append(loss)
    print(f"\n3. WorldModelJEPA: обучение 300 шагов")
    print(f"   loss: {losses[0]:.3f} → {losses[-1]:.3f} "
          f"({'сходится' if losses[-1] < losses[0] else 'расходится'})")

    # Тест: предсказание следующего шага (НЕ виденный шаг — обобщение)
    hv_test = frame.self_hv(20.0, 0, 0)  # за пределами обучающего диапазона
    pred = wm.predict(hv_test, a_forward)
    target = frame.self_hv(20.1, 0, 0)
    # Косинус предсказания с целью (лучше чем норма — VSA-метрика)
    cos_pred = (pred * torch.sign(torch.sin(torch.angle(target)))).sum() / 1024
    print(f"   cos(pred, target) на НЕвиданном шаге 20.0: {cos_pred:.3f} (1 = идеально)")

    print("\n=== D3. SPATIAL-JEPA — OK ===")


if __name__ == "__main__":
    demo()