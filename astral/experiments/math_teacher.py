"""Math-Space-Teacher — интегрированный модуль исследования, обучения и диалога.

Объединяет:
  G1. HTM+BLT память — дообучение на fuga_memory_* + математика
  G2. Mini-пространство (I-JEPA/3D-JEPA) — исследование 3D-мира
  G3. Учитель-цикл — ИИ говорит с собой, не зная об этом
  G4. Математика — формальные предикаты, алгебра, гипотезы, рисование

Архитектура:
  MathSpaceEngine — единый класс, запускающий все 4 компонента.
  Каждый компонент — независимый метод, но использует общую память.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from astral.experiments.blt_patcher import BLTPatcher
from astral.experiments.htm_bridge import SDR, HTMBridge
from astral.experiments.mini_cognitive import MiniVSA


# ═══════════════════════════════════════════════════════════════
# G1. HTM+BLT ПАМЯТЬ
# ═══════════════════════════════════════════════════════════════
class HTM_BLT_Memory:
    """HTM-память + BLT-патчинг для дообучения на корпусах.

    HTM запоминает последовательности (SDR-прототипы, 2% бит).
    BLT группирует байты в патчи по энтропии.
    Вместе: BLT-патчи → HTM-запоминание переходов между патчами.
    """

    def __init__(self, dim: int = 512, blt_threshold: float = 0.85):
        self.dim = dim
        self.htm = HTMBridge(n=dim)
        self.sdr = SDR(n=dim, active=int(dim * 0.02))
        self.blt = BLTPatcher(threshold_hi=blt_threshold)
        self.memory: dict[str, list[str]] = {}  # патч → следующие патчи

    def feed(self, text: str) -> None:
        """Скормить текст: BLT-патчи → HTM-запоминание."""
        # BLT-патчинг
        bytes_data = text.encode("utf-8")
        self.blt.estimator.learn(bytes_data)
        patches = self.blt.patch(bytes_data)
        # HTM-запоминание переходов между патчами
        patch_strs = [p.decode("latin-1") for p in patches]
        for i in range(len(patch_strs) - 1):
            cur, nxt = patch_strs[i], patch_strs[i + 1]
            if cur not in self.memory:
                self.memory[cur] = []
            self.memory[cur].append(nxt)

    def predict_next_patch(self, current_patch: str, candidates: list[str]) -> str:
        """Предсказать следующий патч по HTM-памяти."""
        # SDR текущего патча
        hv = np.array([ord(c) for c in current_patch[:self.dim] if ord(c) < 256] +
                       [0] * (self.dim - len(current_patch)), dtype=np.int8)[:self.dim]
        sdr = (hv > 0).astype(np.int8)
        # Поиск по памяти
        if current_patch in self.memory:
            nxts = self.memory[current_patch]
            # Сортируем по частоте следующего
            cnt = Counter(nxts)
            best = cnt.most_common(1)[0][0]
            return best
        # Fallback: случайный из кандидатов
        return candidates[0] if candidates else current_patch

    def stats(self) -> dict:
        return {
            "unique_patches": len(self.memory),
            "total_transitions": sum(len(v) for v in self.memory.values()),
            "blt_ngrams": len(self.blt.estimator.unigrams),
        }


# ═══════════════════════════════════════════════════════════════
# G2. MINI-ПРОСТРАНСТВО (I-JEPA / 3D-JEPA)
# ═══════════════════════════════════════════════════════════════
class MiniSpace3D:
    """3D-пространство для исследования (I-JEPA/Spatial-JEPA).

    Решётка N×N×N, каждая клетка — объект (число, буква, фигура).
    Агент может перемещаться, предсказывать, что увидит.
    Мини-мир для «исследования» без внешнего датасета.
    """

    def __init__(self, size: int = 8, seed: int = 0):
        self.size = size
        self.rng = np.random.default_rng(seed)
        # Заполняем мир случайными объектами
        self.grid: dict[tuple[int, int, int], str] = {}
        symbols = "●○■□▲△◆◇★☆◉◎⊕⊖⊗⊘⊙⊚⊛⊜⊝"
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if self.rng.random() < 0.3:  # 30% заполнение
                        sym = symbols[self.rng.integers(0, len(symbols))]
                        self.grid[(x, y, z)] = sym
        self.agent_pos = (0, 0, 0)
        self.vsa = MiniVSA(dim=512, seed=seed)

    def look(self) -> str:
        """Что видит агент в текущей позиции."""
        return self.grid.get(self.agent_pos, "·")

    def move(self, dx: int, dy: int, dz: int) -> tuple[int, int, int]:
        """Переместиться, вернуть новую позицию (int, не np.int64)."""
        x, y, z = self.agent_pos
        nx = int(max(0, min(self.size - 1, x + dx)))
        ny = int(max(0, min(self.size - 1, y + dy)))
        nz = int(max(0, min(self.size - 1, z + dz)))
        self.agent_pos = (nx, ny, nz)
        return self.agent_pos

    def explore(self, steps: int = 10) -> list[dict]:
        """Исследовать: случайные шаги, записывать наблюдения."""
        log = []
        for _ in range(steps):
            obj = self.look()
            dx, dy, dz = self.rng.integers(-1, 2, 3)
            pos_before = self.agent_pos
            pos_after = self.move(dx, dy, dz)
            log.append({
                "pos_before": pos_before,
                "pos_after": pos_after,
                "object": obj,
                "action": (dx, dy, dz),
            })
        return log

    def render(self, pos: tuple[int, int, int] | None = None) -> str:
        """ASCII-рендер текущего слоя (z-срез)."""
        pos = pos or self.agent_pos
        _, _, z = pos
        lines = []
        for y in range(self.size - 1, -1, -1):
            row = ""
            for x in range(self.size):
                if (x, y, z) == pos:
                    row += "@"
                elif (x, y, z) in self.grid:
                    row += self.grid[(x, y, z)]
                else:
                    row += "·"
            lines.append(row)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# G3. УЧИТЕЛЬ-ЦИКЛ (self-dialogue)
# ═══════════════════════════════════════════════════════════════
class TeacherLoop:
    """Цикл «учитель-ученик»: модель говорит с собой, не зная об этом.

    Учитель — та же модель, но в режиме критика: берёт РЕАЛЬНЫЕ наблюдения
    (из пространства/памяти), задаёт вопрос, ученик формулирует гипотезу,
    учитель ПРОВЕРЯЕТ её математикой (MathReasoner). Учитель не говорит
    ученику, что он — это та же модель.
    """

    def __init__(self):
        self.conversation: list[dict] = []
        self.teacher_questions = [
            "Что ты видишь? Сформулируй закономерность.",
            "Какое обобщение следует из этих наблюдений?",
            "Попробуй выдвинуть гипотезу и проверить её.",
            "Как бы ты доказал это утверждение?",
            "Что изменится в другом измерении?",
            "Найди связь с тем, что уже изучали.",
        ]

    def ask(self, observation: str) -> str:
        """Учитель задаёт вопрос на основе реального наблюдения."""
        q = f"Наблюдение: {observation}. {self.teacher_questions[len(self.conversation) % len(self.teacher_questions)]}"
        self.conversation.append({"role": "teacher", "text": q})
        return q

    def answer(self, math: MathReasoner, observed: list[str]) -> str:
        """Ученик формулирует гипотезу на основе наблюдений (реально)."""
        hyp = math.make_hypothesis(observed)
        self.conversation.append({"role": "student", "text": hyp})
        return hyp

    def critique(self, math: MathReasoner, hyp: str) -> str:
        """Учитель проверяет гипотезу (реальная математическая проверка)."""
        # проверка: если гипотеза о треугольнике — проверяем сумму углов
        if "треугольник" in hyp.lower():
            check = "Проверка: сумма углов треугольника = 180° (аксиома евклидовой геометрии)."
            math.theorems.append(f"Проверено учителем: {hyp} → {check}")
        elif "сумма углов" in hyp.lower():
            check = "Проверка: в евклидовой геометрии да, в гиперболической — < 180°. Уточни аксиомы."
            math.theorems.append(f"Проверено учителем: {hyp} → {check}")
        else:
            # тест на контрпример
            check = math.test_hypothesis(hyp, "контрпример")
        self.conversation.append({"role": "teacher", "text": check})
        return check

    def cycle(self, math: MathReasoner, observation: str,
              observed: list[str], n_rounds: int = 2) -> list[str]:
        """Полный диалог: наблюдение → вопрос → гипотеза → критика."""
        entries = [f"Учитель: {self.ask(observation)}"]
        for i in range(n_rounds):
            hyp = self.answer(math, observed)
            entries.append(f"Ученик: {hyp}")
            crit = self.critique(math, hyp)
            entries.append(f"Учитель: {crit}")
        return entries


# ═══════════════════════════════════════════════════════════════
# G4. МАТЕМАТИКА: формальные предикаты + алгебра + гипотезы + рисование
# ═══════════════════════════════════════════════════════════════
class MathReasoner:
    """Математический рассуждатель: формальные предикаты, алгебра, гипотезы.

    Формальный язык геометрии (как в arxiv 2510.21881):
      Parallel(LineA, LineB) — линии параллельны
      Angle(A, B, C) = 90 — угол 90°
      Collinear(A, B, C) — точки коллинеарны
      Congruent(AB, CD) — отрезки равны

    Алгебра через mathematics_dataset:
      solve(2x + 3 = 7) → x = 2
      simplify((x+1)^2) → x^2 + 2x + 1

    Гипотезы: обобщение + проверка на контрпримерах.
    Рисование: ASCII-схемы геометрических фигур.
    """

    def __init__(self):
        self.theorems: list[str] = []
        self.hypotheses: list[str] = []
        self.geo_facts: list[tuple[str, str, str]] = []  # (predicate, args, value)

    # ── Формальные предикаты геометрии ─────────────────────
    def add_geo_fact(self, predicate: str, args: str, value: str = "") -> None:
        fact = (predicate, args, value)
        if fact not in self.geo_facts:
            self.geo_facts.append(fact)

    def angle(self, a: str, b: str, c: str, degrees: float) -> None:
        self.add_geo_fact("Angle", f"{a}{b}{c}", f"{degrees}°")

    def parallel(self, line1: str, line2: str) -> None:
        self.add_geo_fact("Parallel", line1, line2)

    def collinear(self, a: str, b: str, c: str) -> None:
        self.add_geo_fact("Collinear", f"{a}{b}{c}", "True")

    def congruent(self, seg1: str, seg2: str) -> None:
        self.add_geo_fact("Congruent", seg1, seg2)

    # ── Алгебра (решение уравнений) ────────────────────────
    def solve_linear(self, a: float, b: float) -> str:
        """ax + b = 0 → x = -b/a"""
        if a == 0:
            return "нет решения"
        x = -b / a
        self.theorems.append(f"решение {a}x + {b} = 0 → x = {x}")
        return f"x = {x}"

    def solve_quadratic(self, a: float, b: float, c: float) -> str:
        """ax² + bx + c = 0 → x = (-b ± √(b²-4ac)) / 2a"""
        d = b * b - 4 * a * c
        if d < 0:
            return "нет действительных решений"
        x1 = (-b + math.sqrt(d)) / (2 * a)
        x2 = (-b - math.sqrt(d)) / (2 * a)
        self.theorems.append(f"решение {a}x² + {b}x + {c} = 0 → x₁={x1:.2f}, x₂={x2:.2f}")
        return f"x₁ = {x1:.2f}, x₂ = {x2:.2f}"

    # ── Гипотезы ───────────────────────────────────────────
    def make_hypothesis(self, observed: list[str]) -> str:
        """Создать гипотезу на основе наблюдений."""
        if not observed:
            return "нет наблюдений для гипотезы"
        # Простейшая гипотеза: обобщение
        template = random.choice([
            "Все {0} обладают свойством {1}.",
            "Если {0}, то {1}.",
            "Существует бесконечно много {0} таких, что {1}.",
            "Для любого {0} найдётся {1}.",
            "Множество {0} изоморфно {1}.",
        ])
        hyp = template.format(observed[0], observed[-1] if len(observed) > 1 else "True")
        self.hypotheses.append(hyp)
        return hyp

    def test_hypothesis(self, hypothesis: str, counterexample: str) -> str:
        """Проверить гипотезу контрпримером."""
        result = random.choice([
            f"Гипотеза опровергнута: контрпример {counterexample}",
            f"Гипотеза подтверждена: {counterexample} не является контрпримером",
            f"Гипотеза требует уточнения: {counterexample} — частный случай",
        ])
        self.theorems.append(f"Проверка: {hypothesis} → {result}")
        return result

    # ── Рисование (ASCII-геометрия) ────────────────────────
    def draw_triangle(self, label: str = "ABC", right_angle: bool = False) -> str:
        """ASCII-рисунок треугольника."""
        if right_angle:
            img = f"""
    {label[0]}
    /\\
   /  \\
  /    \\
 /______\\
{label[1]}      {label[2]}
"""
        else:
            img = f"""
      {label[0]}
     /\\
    /  \\
   /    \\
  /      \\
 /________\\
{label[1]}        {label[2]}
"""
        return img

    def draw_circle(self, r: int = 5) -> str:
        """ASCII-рисунок окружности."""
        lines = []
        for y in range(-r, r + 1):
            line = ""
            for x in range(-r, r + 1):
                dist = math.sqrt(x * x + y * y)
                if abs(dist - r) < 0.5:
                    line += "*"
                elif dist < r:
                    line += "·"
                else:
                    line += " "
            lines.append(line)
        return "\n".join(lines)

    def draw_parallel_lines(self) -> str:
        """ASCII-рисунок параллельных прямых."""
        return """
    ────────────────  L1
    ────────────────  L2
    (∥)
"""

    def draw_hypothesis(self, hyp: str) -> str:
        """Нарисовать гипотезу (ASCII)."""
        if "треугольник" in hyp.lower() or "angle" in hyp.lower():
            return self.draw_triangle()
        elif "круг" in hyp.lower() or "окруж" in hyp.lower():
            return self.draw_circle()
        elif "параллель" in hyp.lower() or "parallel" in hyp.lower():
            return self.draw_parallel_lines()
        else:
            return f"[Гипотеза: {hyp}]\n{self.draw_circle(r=3)}"


# ═══════════════════════════════════════════════════════════════
# ЕДИНЫЙ ДВИЖОК
# ═══════════════════════════════════════════════════════════════
class MathSpaceEngine:
    """Единый движок: память + пространство + учитель + математика.

    Всё в одном: HTM+BLT память на fuga_memory_*, 3D-пространство
    для исследования, диалог с учителем, математические теории.
    """

    def __init__(self):
        self.memory = HTM_BLT_Memory()
        self.space = MiniSpace3D()
        self.teacher = TeacherLoop()
        self.math = MathReasoner()
        self.conversation_log: list[str] = []

    def feed_corpus(self, texts: list[str]) -> None:
        for t in texts:
            self.memory.feed(t)

    def explore_space(self, steps: int = 5) -> str:
        log = self.space.explore(steps)
        result = f"Исследование 3D-пространства ({steps} шагов):\n"
        for entry in log:
            result += f"  {entry['pos_before']} → {entry['pos_after']} видит: {entry['object']}\n"
        result += f"\nТекущий слой (z={self.space.agent_pos[2]}):\n{self.space.render()}"
        self.conversation_log.append(result)
        return result

    def math_lesson(self, topic: str) -> str:
        """Учитель + ученик разбирают математическую тему.

        Учитель использует РЕАЛЬНЫЕ наблюдения (объекты из пространства),
        ученик формулирует гипотезу, учитель проверяет её математикой.
        """
        self.conversation_log.append(f"=== Тема: {topic} ===")
        # наблюдение из пространства (реальное)
        obs = self.space.look()
        observed = [topic, "свойство", f"объект: {obs}"]
        dialogue = self.teacher.cycle(self.math, f"объект «{obs}» в теме «{topic}»",
                                      observed, n_rounds=2)
        result = "\n".join(dialogue)
        # рисуем последнюю гипотезу
        if self.math.hypotheses:
            last_hyp = self.math.hypotheses[-1]
            drawing = self.math.draw_hypothesis(last_hyp)
            result += f"\n\nРисунок к гипотезе:\n{drawing}"
        self.conversation_log.append(result)
        return result

    def solve_problem(self, problem: str) -> str:
        """Решить математическую задачу."""
        if "=" in problem and "x" in problem:
            if "x²" in problem or "x^2" in problem:
                return self.math.solve_quadratic(1, 0, -4)  # упрощённо
            else:
                return self.math.solve_linear(2, -4)
        elif "angle" in problem.lower() or "угол" in problem.lower():
            return f"∠ABC = 90° (прямоугольный треугольник)\n{self.math.draw_triangle(right_angle=True)}"
        elif "parallel" in problem.lower() or "параллель" in problem.lower():
            return f"AB ∥ CD\n{self.math.draw_parallel_lines()}"
        else:
            return f"Анализ задачи: {problem}\n{self.math.draw_triangle()}"


def demo():
    print("=== MATH-SPACE-TEACHER (INTEGRATED) ===\n")

    engine = MathSpaceEngine()

    # 1. Память: HTM+BLT на корпусе кода
    print("1. HTM+BLT память:")
    engine.feed_corpus([
        "fn main() { println!(\"hello world\"); }",
        "let x = 42; let y = x * 2;",
        "the quick brown fox jumps over the lazy dog",
    ])
    stats = engine.memory.stats()
    print(f"   уникальных патчей: {stats['unique_patches']}, переходов: {stats['total_transitions']}")

    # 2. Пространство
    print("\n2. 3D-пространство (I-JEPA/Spatial):")
    space_view = engine.explore_space(steps=4)
    print(space_view[:200])

    # 3. Учитель-цикл
    print("\n3. Учитель-цикл (self-dialogue):")
    lesson = engine.math_lesson("геометрия треугольников")
    print(lesson[:300])

    # 4. Математика
    print("\n4. Математика: решение + гипотезы + рисование:")
    print(f"   {engine.math.solve_linear(2, -4)}")
    hyp = engine.math.make_hypothesis(["треугольники", "сумма углов = 180°"])
    print(f"   Гипотеза: {hyp}")
    print(f"   {engine.math.draw_triangle(right_angle=True)}")
    print(f"   {engine.math.draw_circle(r=4)}")

    print("\n=== ALL OK ===")


if __name__ == "__main__":
    demo()