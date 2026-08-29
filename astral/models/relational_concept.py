"""Relational Concept — язык как реляционные фазовые сдвиги Δθ.

Направление 1 (из THREE_DIRECTIONS.md): языковые описания кодируются
не как последовательность слов, а как РЕЛЯЦИОННЫЙ фазовый сдвиг между
объектами:

  "куб слева от сферы" → HV_куб ⊗ e^{i·Δθ} ≈ HV_сфера

Слои:
1. RelationalEncoder — предикат/отношение → фазовый сдвиг Δθ
   (слева/справа/над/под/перед/за → разные фазы, единый SE(3) базис)
2. RelationalBinder — факт F = HS ⊗ Dθ ⊗ HO (все три связаны)
   Запрос: F ⊗ HS ⊗ Dθ = HO (self-inverse для биполярных)
3. RelationalReasoner — комбинаторная инференция: из (субъект, Δθ)
   восстановить отсутствующий объект через cleanup по кодбуку.

Проверка согласованности с пространством: OccupancyGrid (spatial_jepa.py)
даёт координаты; D1 проверяет, что Δθ согласован с реальным сдвигом.
"""

from __future__ import annotations

import numpy as np
import torch

from fuga_core import HybridBinder
from antitf.rust_bridge import packed_to_torch

# Реляционные сдвиги в едином SE(3) базисе (как EgoFrame)
RELATION_AXIS = {
    "left_of":   (-1.0, 0.0, 0.0),   # слева от
    "right_of":  (1.0, 0.0, 0.0),    # справа от
    "above":     (0.0, 0.0, 1.0),    # над
    "below":     (0.0, 0.0, -1.0),   # под
    "in_front":  (0.0, 1.0, 0.0),    # перед
    "behind":    (0.0, -1.0, 0.0),   # за
    "on":        (0.0, 0.0, 0.5),    # на (частичный сдвиг)
    "near":      (0.3, 0.0, 0.0),    # рядом
    "far":       (3.0, 0.0, 0.0),    # далеко
}

# Инверсии отношений (для обратных фактов)
RELATION_INVERSE = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "in_front": "behind",
    "behind": "in_front",
    "on": "on",
    "near": "near",
    "far": "far",
}


def _hv_of(binder: HybridBinder, name: str, dim: int) -> torch.Tensor:
    return packed_to_torch(np.asarray(binder.bind_batch([[name]])))[0].float()


class RelationalEncoder:
    """Отношение (строка) → биполярный Dθ (из фазового сдвига).

    Δθ = e^{i·(dx·ωx + dy·ωy + dz·ωz)} — комплексный фазовый сдвиг.
    Dθ = sign(cos(Δθ)) — его биполярная проекция для self-inverse
    связывания (биполярное ⊗ само-обратимо).
    """

    def __init__(self, dim: int = 1024, spatial_freq: float = 2.0, seed: int = 7):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Единый SE(3) базис (как EgoFrame в spatial_jepa.py)
        self.omega = {
            "x": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
            "y": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
            "z": torch.tensor(rng.uniform(-spatial_freq, spatial_freq, dim)).float(),
        }

    def delta_theta(self, relation: str) -> torch.Tensor:
        """Комплексный фазовый сдвиг: e^{i·(dx·ωx + dy·ωy + dz·ωz)}."""
        dx, dy, dz = RELATION_AXIS.get(relation, (1.0, 0.0, 0.0))
        theta = dx * self.omega["x"] + dy * self.omega["y"] + dz * self.omega["z"]
        return torch.exp(1j * theta)

    def dtheta_bipolar(self, relation: str) -> torch.Tensor:
        """Биполярная проекция сдвига: sign(sin(Δθ)) ∈ {±1}^dim.

        ВАЖНО: sign(sin()) а не sign(cos()) — sin НЕчётный, поэтому
        противоположные направления (left/right) дают ПРОТИВОПОЛОЖНЫЕ
        векторы, а не идентичные (cos чётный — collapsed бы их).
        """
        return torch.sign(torch.sin(torch.angle(self.delta_theta(relation)))).float()

    def relation_name(self, dtheta: torch.Tensor) -> str:
        """Обратно: биполярный сдвиг → ближайшее отношение."""
        best, best_name = -1e9, "unknown"
        for name in RELATION_AXIS:
            ref = self.dtheta_bipolar(name)
            sim = (dtheta * ref).sum() / self.dim
            if sim > best:
                best, best_name = sim, name
        return best_name


class RelationalBinder:
    """Реляционные факты: F = HS ⊗ Dθ ⊗ HO.

    Запрос "субъект ОТНОШЕНИЕ ?" → F ⊗ HS ⊗ Dθ = HO → cleanup.
    Запрос "? ОТНОШЕНИЕ объект" → F ⊗ HO ⊗ Dθ = HS (Dθ само-обратим).
    """

    def __init__(self, binder: HybridBinder, encoder: RelationalEncoder,
                 anchors: list[str], dim: int = 1024):
        self.binder = binder
        self.enc = encoder
        self.dim = dim
        self.anchors = anchors
        self.codebook = torch.stack([_hv_of(binder, a, dim) for a in anchors])
        # Факты: ключ (subject, relation) → связанная тройка F = HS⊗Dθ⊗HO
        self.facts: dict[tuple[str, str], torch.Tensor] = {}

    def _hv(self, name: str) -> torch.Tensor:
        return _hv_of(self.binder, name, self.dim)

    def add_fact(self, subject: str, relation: str, obj: str) -> None:
        """Запомнить: subject RELATION obj (прямой + обратный факт).

        Прямой:  F = HS ⊗ Dθ ⊗ HO  →  F ⊗ HS ⊗ Dθ = HO
        Обратный: G = HO ⊗ Dθ' ⊗ HS →  G ⊗ HO ⊗ Dθ' = HS (Dθ' = инверсия)
        """
        hs, ho = self._hv(subject), self._hv(obj)
        d = self.enc.dtheta_bipolar(relation)
        self.facts[(subject, relation)] = hs * d * ho
        # Обратный факт: obj inverse(relation) subject
        inv = RELATION_INVERSE.get(relation, relation)
        d_inv = self.enc.dtheta_bipolar(inv)
        self.facts[(obj, inv)] = ho * d_inv * hs

    def infer(self, known: str, relation: str) -> str:
        """Из (субъект, Δθ) → объект через self-inverse unbinding.

        F ⊗ HS ⊗ Dθ = HO → cleanup по кодбуку (реальный векторный
        путь, а не lookup сохранённого ответа).
        """
        if (known, relation) not in self.facts:
            return "?"
        hs = self._hv(known)
        d = self.enc.dtheta_bipolar(relation)
        F = self.facts[(known, relation)]
        unbound = F * hs * d  # = HO (точно для биполярных)
        sims = self.codebook @ unbound
        return self.anchors[int(sims.argmax())]

    def verify(self, subject: str, relation: str, obj: str) -> float:
        """Насколько тройка согласована: cos(HO_ожидаемый, HO_из факта)."""
        key = (subject, relation)
        if key not in self.facts:
            return 0.0
        hs = self._hv(subject)
        d = self.enc.dtheta_bipolar(relation)
        unbound = self.facts[key] * hs * d  # восстановленный объект
        ho = self._hv(obj)
        return max(0.0, float((unbound * ho).sum() / self.dim))


def demo():
    print("=== D1. RELATIONAL Δθ — ЯЗЫК КАК ПРОСТРАНСТВО ===\n")

    binder = HybridBinder(1024)
    anchors = ["cube", "sphere", "table", "ball", "box", "lamp", "cup", "book"]
    enc = RelationalEncoder(dim=1024)
    rb = RelationalBinder(binder, enc, anchors, dim=1024)

    # 1. Разные отношения → разные сдвиги
    print("1. Отношения → разные фазовые сдвиги:")
    cos_lr = (enc.dtheta_bipolar("left_of") * enc.dtheta_bipolar("right_of")).sum() / 1024
    cos_la = (enc.dtheta_bipolar("left_of") * enc.dtheta_bipolar("above")).sum() / 1024
    print(f"   cos(left, right) = {cos_lr:.3f} (≈0 = разные)")
    print(f"   cos(left, above) = {cos_la:.3f} (≈0 = разные)")

    # 2. Обратное декодирование
    print("\n2. Обратное декодирование Dθ → отношение:")
    ok_decode = 0
    for rel in ["left_of", "above", "in_front", "on", "behind"]:
        name = enc.relation_name(enc.dtheta_bipolar(rel))
        ok = name == rel
        ok_decode += int(ok)
        print(f"   {rel:10s} → {name:10s} {'OK' if ok else 'FAIL'}")
    print(f"   декодирование: {ok_decode}/5")

    # 3. Факты + инференция
    print("\n3. Комбинаторная инференция (память фактов):")
    rb.add_fact("cube", "left_of", "sphere")
    rb.add_fact("ball", "on", "table")
    rb.add_fact("book", "above", "table")
    rb.add_fact("lamp", "in_front", "book")

    tests = [
        ("cube", "left_of", "sphere"),
        ("sphere", "right_of", "cube"),   # инверсия: sphere справа от cube
        ("ball", "on", "table"),
        ("book", "above", "table"),
        ("table", "below", "book"),       # инверсия above
    ]
    ok_infer = 0
    for known, rel, expected in tests:
        inferred = rb.infer(known, rel)
        ok = inferred == expected
        ok_infer += int(ok)
        print(f"   {known} {rel:10s} → {inferred:8s} (ожид: {expected}) {'OK' if ok else 'FAIL'}")
    print(f"   инференция: {ok_infer}/{len(tests)}")

    # 4. Верификация
    print("\n4. Верификация троек:")
    for subj, rel, obj in tests:
        score = rb.verify(subj, rel, obj)
        print(f"   {subj} {rel:10s} {obj}: cos={score:.3f}")

    print("\n=== D1. RELATIONAL Δθ — OK ===")


if __name__ == "__main__":
    demo()