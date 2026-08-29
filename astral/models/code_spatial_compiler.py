"""Code Spatial Compiler — пространственные связи → исполняемые структуры.

Направление 2 (из THREE_DIRECTIONS.md): пространственные связи
компилируются непосредственно в исполняемые структуры данных:
AST-деревья, хитбоксы, матрицы трансформаций, сетевые пакеты.

  "куб на столе" → struct CubeOnTable { pos: Vec3, rot: Quat }

Слои:
1. SpatialCompiler — реляционный граф (D1) → Rust/Python AST-структура
2. TransformMatrix — фазовый сдвиг → матрица трансформации SE(3)
3. HitboxCompiler — координаты OccupancyGrid → AABB хитбоксы
4. NetworkPacket — относительные позиции → пакет (для мультиагентности)

Использует D1 (RELATION_AXIS) и D3 (EgoFrame/WorldModelJEPA).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from astral.models.relational_concept import RELATION_AXIS
from astral.models.spatial_jepa import EgoFrame


class TransformMatrix:
    """Фазовый сдвиг → матрица трансформации SE(3).

    Из (dx, dy, dz) отношения строим 4×4 однородную матрицу:
      [ R  t ]
      [ 0  1 ]
    Для демо: translation + поворот вокруг Z на угол из отношения.
    """

    @staticmethod
    def translation(dx: float, dy: float, dz: float) -> np.ndarray:
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = dx, dy, dz
        return T

    @staticmethod
    def rotation_z(angle_rad: float) -> np.ndarray:
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.eye(4)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
        return R

    @staticmethod
    def from_relation(relation: str) -> np.ndarray:
        """SE(3) матрица из реляционного сдвига RELATION_AXIS."""
        dx, dy, dz = RELATION_AXIS.get(relation, (1.0, 0.0, 0.0))
        T = TransformMatrix.translation(dx, dy, dz)
        # Угол поворота: норма сдвига * π/4 (условный)
        angle = math.hypot(dx, dy) * math.pi / 4
        R = TransformMatrix.rotation_z(angle)
        return R @ T


class SpatialCompiler:
    """Реляционный граф → Rust/Python AST-структура.

    Из списка троек (субъект, отношение, объект) строит код:
      struct CubeOnTable { pos: (f64, f64, f64), rot: (f64, f64, f64) }
      fn place() -> CubeOnTable { ... }
    """

    def __init__(self, target: str = "rust"):
        self.target = target

    def compile(self, triples: list[tuple[str, str, str]]) -> str:
        """Компиляция троек → исходный код структуры."""
        if self.target == "rust":
            return self._compile_rust(triples)
        return self._compile_python(triples)

    def _rust_struct_name(self, subj: str, obj: str) -> str:
        cap = lambda s: s[:1].upper() + s[1:]
        return f"{cap(subj)}{cap(obj)}"

    def _compile_rust(self, triples: list[tuple[str, str, str]]) -> str:
        lines = []
        for subj, rel, obj in triples:
            name = self._rust_struct_name(subj, obj)
            dx, dy, dz = RELATION_AXIS.get(rel, (1.0, 0.0, 0.0))
            lines.append(f"// {subj} {rel} {obj}  (Δr = ({dx}, {dy}, {dz}))")
            lines.append(f"struct {name} {{")
            lines.append(f"    pos: (f64, f64, f64), // объект относительно self")
            lines.append(f"    rot: (f64, f64, f64), // ориентация (SE(3))")
            lines.append(f"    relation: &'static str, // \"{rel}\"")
            lines.append(f"}}")
            lines.append(f"impl {name} {{")
            lines.append(f"    fn place() -> Self {{")
            lines.append(f"        Self {{ pos: ({dx}, {dy}, {dz}), rot: (0.0, 0.0, 0.0), relation: \"{rel}\" }}")
            lines.append(f"    }}")
            lines.append(f"}}")
        return "\n".join(lines)

    def _compile_python(self, triples: list[tuple[str, str, str]]) -> str:
        lines = ["from dataclasses import dataclass", ""]
        for subj, rel, obj in triples:
            name = self._rust_struct_name(subj, obj)
            dx, dy, dz = RELATION_AXIS.get(rel, (1.0, 0.0, 0.0))
            lines.append(f"@dataclass")
            lines.append(f"class {name}:")
            lines.append(f"    pos: tuple = ({dx}, {dy}, {dz})  # {rel}")
            lines.append(f"    rot: tuple = (0.0, 0.0, 0.0)")
            lines.append(f"    relation: str = '{rel}'")
            lines.append("")
        return "\n".join(lines)


class HitboxCompiler:
    """Координаты объектов → AABB хитбоксы (для столкновений)."""

    @staticmethod
    def aabb(cx: float, cy: float, cz: float,
             half: float = 0.5) -> dict[str, tuple]:
        """AABB: центр ± half по каждой оси."""
        return {
            "min": (cx - half, cy - half, cz - half),
            "max": (cx + half, cy + half, cz + half),
            "center": (cx, cy, cz),
        }


def demo():
    print("=== D2. CODE SPATIAL COMPILER ===\n")

    # 1. Компиляция пространственных троек → Rust-код
    compiler = SpatialCompiler(target="rust")
    triples = [
        ("cube", "on", "table"),
        ("sphere", "left_of", "cube"),
        ("lamp", "above", "table"),
    ]
    rust_code = compiler.compile(triples)
    print("1. Пространственный граф → Rust-структуры:")
    print(rust_code)
    print()

    # 2. Python-компиляция + выполнение (валидация AST)
    py_compiler = SpatialCompiler(target="python")
    py_code = py_compiler.compile(triples)
    print("2. Python-компиляция + исполнение:")
    ns = {}
    exec(py_code, ns)
    cube_on_table = ns["CubeTable"]()
    print(f"   CubeOnTable(): pos={cube_on_table.pos} relation='{cube_on_table.relation}'")
    print(f"   AABB куба: {HitboxCompiler.aabb(*cube_on_table.pos)}")

    # 3. TransformMatrix из отношений
    print("\n3. SE(3) матрицы трансформации:")
    for _, rel, _ in triples:
        M = TransformMatrix.from_relation(rel)
        print(f"   {rel:10s}: t=({M[0,3]:.1f}, {M[1,3]:.1f}, {M[2,3]:.1f})")

    # 4. Интеграция с EgoFrame (D3): self + объекты
    print("\n4. Интеграция с EgoFrame (D3):")
    frame = EgoFrame(dim=1024)
    self_hv = frame.self_hv(0, 0, 0)
    for subj, rel, obj in triples:
        dx, dy, dz = RELATION_AXIS.get(rel, (1.0, 0.0, 0.0))
        obj_hv = frame.self_hv(dx, dy, dz)
        # Код для размещения объекта
        print(f"   // {subj} {rel} {obj}: self⊗Δ({dx},{dy},{dz})")
    print("   cos(self(0,0,0), obj(dx,dy,dz)) = "
          f"{float((self_hv * frame.self_hv(1,0,0).conj()).real.sum() / 1024):.3f}")

    print("\n=== D2. CODE SPATIAL COMPILER — OK ===")


if __name__ == "__main__":
    demo()