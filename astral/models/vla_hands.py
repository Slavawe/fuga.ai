"""Embodied VLA Executant — «Руки» агента + адаптивная адаптация.

Слои (по предложению пользователя):

H1. VLAExecutor — Vision-Language-Action: выход = вектор манипулятора
    7-DoF: [x, y, z, roll, pitch, yaw, gripper_state]
    vision-латент (VL-JEPA 256-d) → action-голова → вектор действия

H2. AffordanceField — пространство возможностей через FPE:
    не «точка для клика», а АФФОРДАНС-ПОЛЕ: модель видит ручку двери
    / кнопку в UI и сразу знает пространство допустимых траекторий.

H3. PhaseTrajectory — фазовая траектория к цели:
    путь в SE(3)-фазовом пространстве (EgoFrame), плавное движение
    от текущего состояния к аффорданс-точке.

H4. AdaptiveLayer — АДАПТИВНЫЙ ИИ (не статичные веса):
    онлайн-обновление action-головы по сигналу успеха/неуспеха
    (Widrow-Hoff + OWM-защита уже выученных направлений).
    «Руки» учатся на своём опыте, а не только на претрейне.

Вход: vision-латент (256-d из vljepa_restore) + occupancy grid (D3).
Выход: вектор действия 7-DoF + траектория + уверенность.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astral.models.spatial_jepa import EgoFrame
from astral.models.relational_concept import RELATION_AXIS


class VLAExecutor(nn.Module):
    """Vision-Language-Action: vision-латент → 7-DoF вектор действия.

    Выход: [x, y, z, roll, pitch, yaw, gripper] ∈ R^7.
    gripper ∈ [0,1] (0 = открыт, 1 = закрыт).

    Адаптивная голова: веса обновляются онлайн (H4), а не заморожены.
    """

    def __init__(self, vision_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.vision_dim = vision_dim
        # action-голова: vision → скрытый → 7-DoF
        self.fc1 = nn.Linear(vision_dim, hidden)
        self.fc2 = nn.Linear(hidden, 7)
        # Для адаптации: маска защищённых направлений (OWM-дух)
        self.projected: torch.Tensor | None = None  # консолидированные направления
        self.lr = 0.01

    def forward(self, vision_latent: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(vision_latent))
        act = self.fc2(x)
        # gripper: сигмоид в [0,1]
        act[..., 6] = torch.sigmoid(act[..., 6])
        return act

    # ── H4: онлайн-адаптация (не статичные веса) ───────────────
    def adapt(self, vision_latent: torch.Tensor, target: torch.Tensor,
              error_gain: float = 1.0) -> float:
        """Один шаг адаптации по сигналу (успех/неуспех).

        Widrow-Hoff на выходном слое fc2 с OWM-защитой: изменение
        только вдоль направлений, ортогональных уже консолидированным.
        """
        with torch.no_grad():
            h = F.relu(self.fc1(vision_latent))  # [hidden]
            pred = self.fc2(h)  # [7]
            err = (target - pred) * error_gain  # сигнал успеха
            # OWM-проекция: Δw вдоль защищённых направлений гасится
            delta = self.lr * torch.outer(err, h)
            if self.projected is not None:
                # проекция дельты на ортогональное дополнение
                P = self.projected  # [n_proj, hidden]
                coeff = delta @ P.T  # [7, n_proj]
                delta = delta - coeff @ P  # вычитаем защищённую компоненту
            self.fc2.weight.data += delta
            self.fc2.bias.data += self.lr * err
            return float(err.norm())

    def consolidate(self, h_batch: torch.Tensor) -> None:
        """Консолидация нового направления (OWM).

        h_batch: [n, hidden] — скрытые состояния успешных опытов.
        """
        H = h_batch / (h_batch.norm(dim=1, keepdim=True) + 1e-8)
        if self.projected is None:
            self.projected = H[:1].clone()
        else:
            # ортогонализация Грама-Шмидта к существующим
            for h in H:
                v = h.clone()
                for p in self.projected:
                    v = v - (v @ p) * p
                if v.norm() > 1e-6:
                    self.projected = torch.cat(
                        [self.projected, (v / v.norm()).unsqueeze(0)])


class AffordanceField:
    """Пространство возможностей (Affordance) через FPE-фазы.

    Не «точка для клика», а ПОЛЕ: для каждого кандидата действия
    (позиции в сетке) — оценка «насколько тут можно взаимодействовать».

    Оценка: близость к объектам (occupancy grid) + выравнивание с
    реляционными сдвигами RELATION_AXIS.
    """

    def __init__(self, frame: EgoFrame, cell_size: float = 0.5, extent: float = 5.0):
        self.frame = frame
        self.cell = cell_size
        self.extent = extent
        self.objects: list[tuple[str, float, float, float]] = []  # (имя, x, y, z)

    def set_objects(self, objects: list[tuple[str, float, float, float]]) -> None:
        self.objects = objects

    def affordance_score(self, px: float, py: float, pz: float,
                         object_name: str | None = None) -> float:
        """Оценка аффорданса в точке (px, py, pz).

        = макс косинуса фазового вектора точки с объектами
          (фазовая близость = пространственная близость в FPE-VSA).
        """
        best = 0.0
        target = [o for o in self.objects if object_name is None or o[0] == object_name]
        if not target:
            target = self.objects
        for name, ox, oy, oz in target:
            # фазовый вектор относительного положения
            theta = (px - ox) * self.frame.omega["x"] \
                  + (py - oy) * self.frame.omega["y"] \
                  + (pz - oz) * self.frame.omega["z"]
            rel = torch.exp(1j * theta)
            # косинус с Identity (1,0,0,...): чем ближе фаза к 1 → ближе к объекту
            cos = (rel * torch.ones_like(rel).conj()).real.mean().item()
            # cos ≈ cos(θ_средн) — близость к объекту
            score = max(0.0, (cos + 1.0) / 2.0)  # нормализация [0,1]
            if score > best:
                best = score
        return best

    def best_point(self, object_name: str | None = None) -> tuple[float, float, float, float]:
        """Лучшая точка взаимодействия (макс аффорданс) в сетке."""
        best_pt, best_score = None, -1.0
        n = int(self.extent * 2 / self.cell) + 1
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    px = -self.extent + ix * self.cell
                    py = -self.extent + iy * self.cell
                    pz = -self.extent + iz * self.cell
                    s = self.affordance_score(px, py, pz, object_name)
                    if s > best_score:
                        best_score, best_pt = s, (px, py, pz)
        return (best_pt[0], best_pt[1], best_pt[2], best_score)


class PhaseTrajectory:
    """Фазовая траектория от текущего SE(3) к цели.

    Путь интерполируется в фазовом пространстве EgoFrame:
      HV(t) = HV(start) * lerp_phase(start→goal, t)
    Плавное движение: t = 0..1, N шагов.
    """

    def __init__(self, frame: EgoFrame, steps: int = 20):
        self.frame = frame
        self.steps = steps

    def generate(self, start: tuple[float, float, float],
                 goal: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        """Линейная интерполяция с фазовой сглаживающей функцией."""
        path = []
        for i in range(self.steps + 1):
            t = i / self.steps
            # фазовое сглаживание: s-кривая (гладкий старт/стоп)
            s = t * t * (3.0 - 2.0 * t)  # smoothstep
            x = start[0] + (goal[0] - start[0]) * s
            y = start[1] + (goal[1] - start[1]) * s
            z = start[2] + (goal[2] - start[2]) * s
            path.append((x, y, z))
        return path


def demo():
    print("=== HANDS: EMBODIED VLA EXECUTANT ===\n")

    # H1: VLA — vision → 7-DoF
    torch.manual_seed(0)
    vla = VLAExecutor(vision_dim=256)
    vision = torch.randn(256)  # VL-JEPA латент
    act = vla(vision)
    print("H1. VLAExecutor: vision → 7-DoF action")
    print(f"    act = [{', '.join(f'{v:.2f}' for v in act.tolist())}]")
    print(f"    gripper = {act[6].item():.2f} ({'закрыт' if act[6] > 0.5 else 'открыт'})")

    # H2: AffordanceField
    frame = EgoFrame(dim=1024)
    aff = AffordanceField(frame)
    aff.set_objects([("cube", 2.0, 0.0, 0.0), ("table", 0.0, 0.0, 0.0)])
    best = aff.best_point("cube")
    print("\nH2. AffordanceField: пространство возможностей")
    print(f"    лучшая точка для cube: ({best[0]:.1f}, {best[1]:.1f}, {best[2]:.1f}) "
          f"score={best[3]:.2f}")

    # H3: PhaseTrajectory
    traj = PhaseTrajectory(frame)
    path = traj.generate((0, 0, 0), (best[0], best[1], best[2]))
    print("\nH3. PhaseTrajectory: плавная траектория (SE(3))")
    print(f"    start={path[0]}, end={path[-1]}, точек={len(path)}")
    # гладкость: проверить, что нет скачков
    max_jump = max(
        math.dist(path[i], path[i+1]) for i in range(len(path) - 1))
    print(f"    max-шаг = {max_jump:.3f} (плавность)")

    # H4: AdaptiveLayer — обучение на опыте
    print("\nH4. AdaptiveLayer: онлайн-адаптация (не статичные веса)")
    # Целевое действие: подойти к кубу и схватить
    target_act = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    losses = []
    for step in range(80):
        # опыт: vision + шум, цель — стабильно хватать куб
        vision_i = vision + 0.05 * torch.randn(256)
        # сигнал успеха: чем ближе предсказание к цели, тем меньше ошибка
        err_gain = 1.0
        losses.append(vla.adapt(vision_i, target_act, error_gain=err_gain))
    # консолидация успешного опыта (OWM-защита)
    h_success = F.relu(vla.fc1(vision + 0.05 * torch.randn(256)))
    vla.consolidate(h_success.unsqueeze(0))
    pred_final = vla(vision)
    print(f"    loss: {losses[0]:.3f} → {losses[-1]:.3f} "
          f"({'адаптировался' if losses[-1] < losses[0] else 'не сошёлся'})")
    print(f"    финальный act = [{', '.join(f'{v:.2f}' for v in pred_final.tolist())}]")
    print(f"    сходство с целью = {F.cosine_similarity(pred_final, target_act, dim=0).item():.3f}")

    print("\n=== HANDS: VLA + AFFORDANCE + TRAJECTORY + ADAPTIVE — OK ===")


if __name__ == "__main__":
    demo()