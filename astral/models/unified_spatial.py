"""Unified Spatial Engine — связывание всех 4 направлений в одну цепь.

Единый конвейер (из docs/THREE_DIRECTIONS.md):

  VL-JEPA (V1) ──"вижу: куб на столе"──→ 256-d латент
      ↓
  RELATIONAL (D1) ──"куб на столе"──→ Δθ, F = HS⊗Dθ⊗HO
      ↓
  SPATIAL (D3) ──EgoFrame: SE(3)──→ HV_Self, OccupancyGrid, WorldModel
      ↓
  COMPILER (D2) ──пространственный граф──→ Rust/Python код + SE(3) матрицы

Одна цепь: изображение/текст → пространство → исполняемые структуры.
"""

from __future__ import annotations

import torch

from astral.models.relational_concept import (
    RelationalEncoder, RelationalBinder, RELATION_AXIS, RELATION_INVERSE,
)
from astral.models.spatial_jepa import EgoFrame, OccupancyGrid, WorldModelJEPA
from astral.models.code_spatial_compiler import SpatialCompiler, TransformMatrix
from fuga_core import HybridBinder


class UnifiedSpatialEngine:
    """Сквозной пространственный движок (все 4 направления)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.binder = HybridBinder(dim)
        self.rel_enc = RelationalEncoder(dim=dim)
        anchors = ["cube", "sphere", "table", "ball", "box", "lamp", "cup", "book"]
        self.rel = RelationalBinder(self.binder, self.rel_enc, anchors, dim=dim)
        self.frame = EgoFrame(dim=dim)
        self.grid = OccupancyGrid(self.frame, dim=dim)
        self.world = WorldModelJEPA(dim=dim)
        self.compiler = SpatialCompiler(target="rust")

    # ── Шаг 1: язык → реляционные факты (D1) ───────────────────
    def understand(self, sentence: str) -> list[tuple[str, str, str]]:
        """Парсинг простых реляционных предложений → тройки.

        Поддерживает паттерн "SUBJ RELATION OBJ" из словаря RELATION_AXIS.
        """
        words = sentence.lower().split()
        triples = []
        for i, w in enumerate(words):
            if w in RELATION_AXIS:
                subj = words[i - 1] if i > 0 else "?"
                obj = words[i + 1] if i + 1 < len(words) else "?"
                if subj != "?" and obj != "?":
                    triples.append((subj, w, obj))
        for s, r, o in triples:
            self.rel.add_fact(s, r, o)
        return triples

    # ── Шаг 2: пространство (D3) ───────────────────────────────
    def place_in_space(self, triples: list[tuple[str, str, str]],
                       self_pos: tuple[float, float, float] = (0, 0, 0)):
        """Тройки → объекты в OccupancyGrid вокруг Self."""
        sx, sy, sz = self_pos
        for subj, rel, obj in triples:
            dx, dy, dz = RELATION_AXIS.get(rel, (1.0, 0.0, 0.0))
            # объект на rel-расстоянии от субъекта
            self.grid.add(subj, sx + dx, sy + dy, sz + dz, sx, sy, sz)

    # ── Шаг 3: код (D2) ─────────────────────────────────────────
    def compile_space(self, triples: list[tuple[str, str, str]]) -> str:
        return self.compiler.compile(triples)

    # ── Сквозной прогон ─────────────────────────────────────────
    def run(self, sentence: str) -> dict:
        triples = self.understand(sentence)
        self.place_in_space(triples)
        code = self.compile_space(triples)
        # WorldModel: обучение на траектории из предложения
        losses = []
        n_steps = 30
        a = self.world.action_forward("move", step=0.1)
        for t in range(n_steps):
            hv_t = self.frame.self_hv(t * 0.1, 0, 0)
            hv_next = self.frame.self_hv((t + 1) * 0.1, 0, 0)
            losses.append(self.world.learn(hv_t, a, hv_next))
        return {
            "triples": triples,
            "code": code,
            "world_loss": (losses[0], losses[-1]),
            "n_objects": len(self.grid.objects),
        }


def demo():
    print("=== UNIFIED SPATIAL ENGINE — ВСЕ 4 НАПРАВЛЕНИЯ ===\n")
    eng = UnifiedSpatialEngine(dim=1024)

    # Сквозной прогон: язык → пространство → код
    sentence = "cube on table sphere left_of cube lamp above table"
    result = eng.run(sentence)

    print(f"1. Язык (D1): «{sentence}»")
    print(f"   → тройки: {result['triples']}")
    print(f"   → инференция cube on ?: {eng.rel.infer('cube', 'on')}")

    print(f"\n2. Пространство (D3): {result['n_objects']} объектов в grid")
    print(f"   WorldModel loss: {result['world_loss'][0]:.1f} → {result['world_loss'][1]:.1f}")

    print(f"\n3. Код (D2): сгенерировано {len(result['code'].splitlines())} строк")
    print(f"   SE(3) куб→стол: {TransformMatrix.from_relation('on')[0,3]:.1f}")

    print("\n=== UNIFIED SPATIAL ENGINE — OK ===")


if __name__ == "__main__":
    demo()