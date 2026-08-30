"""Генератор синтетических математических задач (принцип mathematics_dataset).

Бесконечный Question → Answer по уровням (easy/medium/hard):
  - линейные уравнения:    2x + 3 = 7 → x = 2
  - квадратные:            x² - 5x + 6 = 0 → x=2,3
  - упрощение:             (x+1)² → x² + 2x + 1
  - последовательности:    арифметические/геометрические, следующий член
  - геометрия:             предикаты (Parallel, Angle, Collinear, Congruent)

Почему это важно: модели нужны ДАННЫЕ для HTM+BLT памяти — чтобы учить
паттерны математики так же, как учит паттерны кода. Это «еда» для памяти.

Использование:
  gen = MathProblemGenerator(seed=42)
  for problem, answer in gen.stream(n=10, level='easy'):
      ...
"""

from __future__ import annotations

import random


class MathProblemGenerator:
    """Генератор математических задач (без внешних данных)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # ── Линейные уравнения: ax + b = c → x = (c-b)/a ──────
    def linear(self) -> tuple[str, str]:
        a = self.rng.randint(1, 9)
        b = self.rng.randint(1, 9)
        c = self.rng.randint(1, 20)
        x = (c - b) / a
        q = f"Решите уравнение: {a}x + {b} = {c}"
        a_str = str(int(x)) if x == int(x) else f"{x:.2f}"
        return q, f"x = {a_str}"

    # ── Квадратные: ax² + bx + c = 0 ──────────────────────
    def quadratic(self) -> tuple[str, str]:
        # гарантируем целые корни: (x-p)(x-q)
        p = self.rng.randint(1, 6)
        q = self.rng.randint(1, 6)
        b = -(p + q)
        c = p * q
        q_str = f"Решите уравнение: x² + {b}x + {c} = 0"
        roots = sorted({p, q})
        if len(roots) == 1:
            return q_str, f"x = {roots[0]} (кратный корень)"
        return q_str, f"x = {roots[0]} или x = {roots[1]}"

    # ── Упрощение: (x+a)² ─────────────────────────────────
    def expand_square(self) -> tuple[str, str]:
        a = self.rng.randint(1, 5)
        q = f"Раскройте скобки: (x + {a})²"
        return q, f"x² + {2*a}x + {a*a}"

    # ── Арифметическая последовательность ─────────────────
    def arithmetic_seq(self) -> tuple[str, str]:
        a1 = self.rng.randint(1, 5)
        d = self.rng.randint(1, 5)
        n = self.rng.randint(3, 5)
        terms = [a1 + k * d for k in range(n)]
        next_term = a1 + n * d
        seq_str = ", ".join(str(t) for t in terms)
        q = f"Найдите следующий член последовательности: {seq_str}, ..."
        return q, f"{next_term}"

    # ── Геометрия: предикаты (Formal Geometry Language) ───
    def geometry(self) -> tuple[str, str]:
        kind = self.rng.randint(0, 3)
        if kind == 0:
            return ("Верно ли: Parallel(LineAB, LineCD) если они не пересекаются?",
                    "Да, Parallel(LineAB, LineCD) = True")
        if kind == 1:
            return ("Чему равен угол: Angle(A, B, C) если треугольник прямоугольный?",
                    "Angle(A, B, C) = 90°")
        if kind == 2:
            return ("Сколько градусов в сумме углов треугольника?",
                    "Angle(A,B,C) + Angle(B,C,A) + Angle(C,A,B) = 180°")
        return ("Верно ли: Collinear(A, B, C) если B — середина AC?",
                "Да, Collinear(A, B, C) = True")

    # ── Поток задач ───────────────────────────────────────
    def stream(self, n: int = 10, level: str = "easy") -> list[tuple[str, str]]:
        """Сгенерировать n задач заданного уровня."""
        items = []
        for _ in range(n):
            if level == "easy":
                fn = self.rng.choice([self.linear, self.arithmetic_seq, self.geometry])
            elif level == "medium":
                fn = self.rng.choice([self.quadratic, self.expand_square, self.arithmetic_seq])
            else:  # hard
                fn = self.rng.choice([self.quadratic, self.expand_square])
            items.append(fn())
        return items

    # ── Корпус для HTM+BLT памяти ─────────────────────────
    def to_corpus(self, n: int = 30) -> list[str]:
        """Задачи + ответы одним текстом (для feed в память)."""
        corpus = []
        for level in ["easy", "medium", "hard"]:
            for q, a in self.stream(n // 3, level):
                corpus.append(f"Q: {q}\nA: {a}")
        return corpus


def demo():
    gen = MathProblemGenerator(seed=42)
    print("=== МАТЕМАТИЧЕСКИЙ ГЕНЕРАТОР (mathematics_dataset принцип) ===\n")
    for level in ["easy", "medium", "hard"]:
        print(f"[{level.upper()}]")
        for q, a in gen.stream(3, level):
            print(f"  Q: {q}")
            print(f"  A: {a}")
    print("\nOK")


if __name__ == "__main__":
    demo()
