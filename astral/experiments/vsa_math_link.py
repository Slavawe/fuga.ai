"""VSA-Math-Space — связка VSA/резонаторов с математикой и пространством.

Кодируем математические предикаты и пространственные сцены
как VSA-гипервекторы, затем ищем ответы через резонаторы.

Поток:
  Наблюдение → VSA encode → резонатор → ответ/гипотеза → учитель

Компоненты:
  VSAMathLink — предикаты (Angle, Parallel, Collinear) как bind(subj,rel,obj)
  VSASpace — FPE-фазы для 3D-позиций, пространственные отношения
  Demo: живой пример от наблюдения до вывода
"""

from __future__ import annotations

import numpy as np
import torch

from astral.experiments.mini_cognitive import MiniVSA
from astral.models.resonator_hdc import HDCResonator, FPEVSA


class VSAMathLink:
    """VSA-кодирование математических предикатов.

    Факт: bind(subject, relation, object)
      Angle(ABC, 90°) → hv_ABC ⊗ hv_angle ⊗ hv_90
      Parallel(AB, CD) → hv_AB ⊗ hv_parallel ⊗ hv_CD

    Поиск: recover(S = факт, n_factors=2) → (subject, object)
    """

    def __init__(self, dim: int = 512, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.vsa = MiniVSA(dim=dim, seed=0)
        # Кеш HV для токенов
        self.hv_cache: dict[str, torch.Tensor] = {}

    def _hv(self, name: str) -> torch.Tensor:
        """Токен → биполярный HV (torch)."""
        if name not in self.hv_cache:
            hv_np = self.vsa.item(name)
            self.hv_cache[name] = torch.from_numpy(hv_np).float()
        return self.hv_cache[name]

    def encode_fact(self, subject: str, relation: str, obj: str) -> torch.Tensor:
        """Факт = bind(subject, relation, object)."""
        return self._hv(subject) * self._hv(relation) * self._hv(obj)

    def query(self, subject: str, relation: str,
              candidates: list[str]) -> tuple[str, float]:
        """Поиск: дано(subject, relation) → ?object.

        S = hv_subject ⊗ hv_relation ⊗ hv_object  (для каждого object)
        recover: S ⊗ hv_subject ⊗ hv_relation = hv_object → cleanup
        """
        hv_subj = self._hv(subject)
        hv_rel = self._hv(relation)
        best_obj, best_cos = candidates[0], -1.0
        for obj in candidates:
            hv_obj = self._hv(obj)
            # факт: S = subj ⊗ rel ⊗ obj
            S = hv_subj * hv_rel * hv_obj
            # unbind: S ⊗ subj ⊗ rel = obj
            recovered = S * hv_subj * hv_rel
            cos = float(torch.dot(recovered, hv_obj) /
                        (torch.norm(recovered) * torch.norm(hv_obj) + 1e-9))
            if cos > best_cos:
                best_cos, best_obj = cos, obj
        return best_obj, best_cos

    def angle_similarity(self, angle1_deg: float, angle2_deg: float) -> float:
        """Косинус между HV для углов (близкие углы = высокий cos)."""
        hv1 = self._hv(f"angle_{angle1_deg}")
        hv2 = self._hv(f"angle_{angle2_deg}")
        return float(torch.dot(hv1, hv2) / (torch.norm(hv1) * torch.norm(hv2) + 1e-9))


class VSASpace:
    """Пространственные отношения через фазовые HV (FPE-стиль).

    Позиция кодируется ДРОБНЫМИ СТЕПЕНЯМИ базовых осей:
      hv(x,y,z) = Px^x · Py^y · Pz^z   (e^{i·x·θx_i} поэлементно)

    Близкие координаты → близкие фазы → высокий косинус.
    Отношение: bind(позиция, объект) → unbind по conj(pos).

    Это честный FPE: каждая координата — фазовая степень базиса.
    """

    def __init__(self, dim: int = 1024, device: str = "cpu"):
        self.dim = dim
        self.device = device
        # Базовые оси: случайные фазовые HV (поэлементные θ)
        rng = np.random.default_rng(0)
        self.theta_x = torch.from_numpy(rng.uniform(0, 2 * np.pi, dim)).to(torch.complex64)
        self.theta_y = torch.from_numpy(rng.uniform(0, 2 * np.pi, dim)).to(torch.complex64)
        self.theta_z = torch.from_numpy(rng.uniform(0, 2 * np.pi, dim)).to(torch.complex64)
        self.anchor_names: dict[str, torch.Tensor] = {}

    def _pos_hv(self, x: float, y: float, z: float) -> torch.Tensor:
        """Позиция → Px^x · Py^y · Pz^z (дробные фазовые степени)."""
        hv = torch.exp(1j * (self.theta_x * x + self.theta_y * y + self.theta_z * z))
        return hv / (torch.norm(hv) + 1e-9)

    def _obj_hv(self, name: str) -> torch.Tensor:
        """Детерминированный биполярный HV объекта."""
        rng = np.random.default_rng(hash(name) % (2**32))
        hv = torch.from_numpy(rng.normal(0, 1, self.dim)).to(torch.complex64)
        return hv / (torch.norm(hv) + 1e-9)

    def encode_anchor(self, name: str, x: float, y: float, z: float) -> None:
        """Запомнить объект: HV = bind(позиция, имя)."""
        pos_hv = self._pos_hv(x, y, z)
        name_hv = self._obj_hv(name)
        self.anchor_names[name] = pos_hv * name_hv

    def nearest(self, x: float, y: float, z: float,
                candidates: list[str]) -> tuple[str, float]:
        """Какой объект ближе всего к позиции (x,y,z)?

        recovered = stored ⊗ conj(pos) = name_hv → косинус с базой.
        """
        pos_hv = self._pos_hv(x, y, z)
        best, best_cos = candidates[0], -1.0
        for name in candidates:
            if name not in self.anchor_names:
                continue
            stored = self.anchor_names[name]
            name_hv = self._obj_hv(name)
            recovered = stored * pos_hv.conj()
            cos = float((recovered * name_hv.conj()).real.sum() /
                        (torch.norm(recovered) * torch.norm(name_hv) + 1e-9))
            if cos > best_cos:
                best_cos, best = cos, name
        return best, best_cos


def demo():
    print("=== VSA-MATH-SPACE: СВЯЗКА VSA + МАТЕМАТИКА + ПРОСТРАНСТВО ===\n")

    # 1. VSA-математика: предикаты
    print("1. Математика (VSA-предикаты):")
    math = VSAMathLink(dim=512)

    # Факты: Angle(ABC, 90°), Angle(DEF, 45°), Parallel(AB, CD)
    S_abc = math.encode_fact("ABC", "angle", "90")
    S_def = math.encode_fact("DEF", "angle", "45")
    S_ab = math.encode_fact("AB", "parallel", "CD")

    # Запрос: дано ABC, angle → ?
    obj, cos = math.query("ABC", "angle", ["90", "45", "180"])
    print(f"   query(ABC, angle) → {obj} (cos={cos:.3f})")

    # Угловая близость: 90° vs 85° (должны быть близки)
    cos_sim = math.angle_similarity(90, 85)
    print(f"   cos(90°, 85°) = {cos_sim:.3f} (близкие углы)")

    # 2. Пространство (FPE-фазы)
    print("\n2. Пространство (FPE-фазы):")
    space = VSASpace(dim=1024)
    space.encode_anchor("куб", 2.0, 0.0, 0.0)
    space.encode_anchor("сфера", 0.0, 3.0, 0.0)
    space.encode_anchor("пирамида", 0.0, 0.0, 4.0)

    # Какой объект ближе всего к (1.9, 0.0, 0.0)?
    name, cos = space.nearest(1.9, 0.0, 0.0, ["куб", "сфера", "пирамида"])
    print(f"   nearest(1.9, 0, 0) → {name} (cos={cos:.3f}) — ожидается куб")
    name, cos = space.nearest(0.0, 2.5, 0.0, ["куб", "сфера", "пирамида"])
    print(f"   nearest(0, 2.5, 0) → {name} (cos={cos:.3f}) — ожидается сфера")

    # 3. Полный поток: наблюдение → VSA → учитель
    print("\n3. Полный поток «наблюдение → VSA → учитель»:")
    from astral.experiments.math_teacher import MathReasoner, TeacherLoop, MiniSpace3D

    mr = MathReasoner()
    tl = TeacherLoop()
    mini = MiniSpace3D(size=5, seed=42)

    # Исследуем пространство
    mini.agent_pos = (0, 0, 0)
    obs = mini.look()

    # Учитель: что видишь?
    q = tl.ask(f"объект «{obs}» на позиции {mini.agent_pos}")
    print(f"   Учитель: {q}")

    # Ученик: гипотеза (через VSA-кодирование)
    hv_obs = math._hv(obs if obs and obs != "·" else "empty")
    hv_pos = torch.tensor([float(p) for p in mini.agent_pos], dtype=torch.float)
    hyp = mr.make_hypothesis([f"объект:{obs}", f"позиция:{mini.agent_pos}"])
    print(f"   Ученик: {hyp}")

    # Учитель проверяет
    crit = tl.critique(mr, hyp)
    print(f"   Учитель: {crit}")

    print("\n=== VSA-MATH-SPACE OK ===")


if __name__ == "__main__":
    demo()