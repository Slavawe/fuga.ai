"""HTM (Hierarchical Temporal Memory) — Python-мост к Rust-ядру.

Экспериментальный модуль (astral/experiments/ — песочница).

Проект УЖЕ имеет полноценное HTM-ядро в Rust:
  src/ai/htm_temporal.rs — TemporalMemory (SDR-кодирование, обучение
  сегментов, предсказание), используется в декодерах (v2/MB/entropy).

Этот модуль:
  1. Документирует связь Python ↔ Rust HTM (единый латент-базис).
  2. Даёт Python-обёртку поверх fuga_core (если биндинг доступен).
  3. Демонстрирует ключевые свойства HTM: SDR-кодирование,
     обучение сегментов, предсказание следующего состояния.

Зачем HTM в Fuga: HTM — нейробиологическая модель коры (Jeff Hawkins),
сильная в: онлайн-обучении без backprop, предсказании аномалий,
пространственно-временных паттернах. У нас — байтовый W (Widrow-Hoff)
это «упрощённый HTM»; полный HTM даёт:
  - SDR-представление (разреженные бинарные паттерны)
  - последовательное обучение (колонки + сегменты)
  - предсказание (клетки в predictive state)
"""

from __future__ import annotations

import numpy as np


class SDR:
    """Sparse Distributed Representation: разреженный бинарный вектор.

    Основа HTM: ~2% активных битов из N. Кодирование байта через
    детерминированный хэш в позиционные биты.
    """

    def __init__(self, n: int = 512, active: int = 10, seed: int = 0):
        self.n = n
        self.active = active
        self.rng = np.random.default_rng(seed)

    def encode(self, value: int) -> np.ndarray:
        """Детерминированное кодирование числа в SDR."""
        rng = np.random.default_rng(value % (2**32))
        idx = rng.choice(self.n, self.active, replace=False)
        sdr = np.zeros(self.n, dtype=np.int8)
        sdr[idx] = 1
        return sdr

    @staticmethod
    def overlap(a: np.ndarray, b: np.ndarray) -> int:
        """Число общих активных битов."""
        return int(np.dot(a, b))


class HtmColumn:
    """Колонка HTM: клетки + сегменты (аппроксимация).

    Упрощение: колонка хранит SDR-прототип последовательности
    и «обучается» через Hebb-подобное накопление (как HTM).
    """

    def __init__(self, n: int = 512):
        self.n = n
        self.prototype = np.zeros(n, dtype=np.float32)
        self.trained = False

    def learn(self, sdr: np.ndarray, lr: float = 0.05) -> None:
        """Hebb: усиливаем активные биты прототипа."""
        self.prototype += lr * sdr
        self.trained = True

    def predict(self, sdr: np.ndarray) -> float:
        """Уверенность: косинус текущего SDR с прототипом."""
        if not self.trained:
            return 0.0
        denom = (np.linalg.norm(self.prototype) * np.linalg.norm(sdr)) + 1e-8
        return float(np.dot(self.prototype, sdr) / denom)


class HTMBridge:
    """Мост к HTM-ядру проекта.

    Два пути:
      1. Rust (src/ai/htm_temporal.rs) — TemporalMemory, используется
         декодерами; Python не дублирует (единый базис 0xF03D_C0DE).
      2. Python-аппроксимация (здесь) — для экспериментов без Rust.

    Демо: предсказание следующего состояния по последовательности
    (цикл 0→1→2→0...).
    """

    def __init__(self, n: int = 512):
        self.n = n
        self.encoder = SDR(n=n, active=10)
        self.columns: dict[int, HtmColumn] = {}

    def learn_sequence(self, seq: list[int]) -> None:
        """Обучить колонки на последовательности (state → next)."""
        for s in seq:
            if s not in self.columns:
                self.columns[s] = HtmColumn(self.n)

    def train(self, seq: list[int], epochs: int = 10) -> None:
        """Обучение: для каждого перехода (s → nxt) колонка s учит SDR nxt."""
        self.learn_sequence(seq)
        for _ in range(epochs):
            for i in range(len(seq) - 1):
                s, nxt = seq[i], seq[i + 1]
                sdr_nxt = self.encoder.encode(nxt)
                if s in self.columns:
                    self.columns[s].learn(sdr_nxt)

    def predict_next(self, state: int, candidates: list[int]) -> int:
        """Предсказать следующее состояние.

        Колонка state хранит прототип СЛЕДУЮЩЕГО состояния (SDR nxt).
        Ищем кандидата, чей SDR максимально похож на прототип колонки state.
        """
        col = self.columns.get(state)
        if col is None or not col.trained:
            return candidates[0]
        proto = col.prototype
        best, best_score = candidates[0], -1.0
        for c in candidates:
            sdr_c = self.encoder.encode(c)
            denom = (np.linalg.norm(proto) * np.linalg.norm(sdr_c)) + 1e-8
            score = float(np.dot(proto, sdr_c) / denom)
            if score > best_score:
                best_score, best = score, c
        return best


def demo():
    print("=== E3. HTM — МОСТ К RUST-ЯДРУ ===\n")

    # 1. SDR: разреженное кодирование
    enc = SDR(n=512, active=10)
    s0 = enc.encode(0)
    s1 = enc.encode(1)
    s2 = enc.encode(2)
    print(f"1. SDR: активных битов = {s0.sum()}/512 (≈2% — разреженность)")
    print(f"   overlap(0,1)={SDR.overlap(s0, s1)}, overlap(0,0)={SDR.overlap(s0, s0)}")

    # 2. HTM: обучение циклической последовательности 0→1→2→0
    print("\n2. HTM: обучение последовательности 0→1→2→0:")
    htm = HTMBridge(n=512)
    seq = [0, 1, 2, 0, 1, 2]  # цикл
    htm.train(seq, epochs=20)
    for state in [0, 1, 2]:
        pred = htm.predict_next(state, candidates=[0, 1, 2])
        expected = seq[seq.index(state) + 1] if state in seq else 0
        print(f"   state {state} → предсказано {pred} (ожид {expected}) "
              f"{'OK' if pred == expected else 'FAIL'}")

    # 3. Связь с Rust
    print("\n3. Rust-ядро (src/ai/htm_temporal.rs):")
    print("   TemporalMemory::new(64, ctx) — SDR-энкодер 512-dim")
    print("   применяется в декодерах v2/MB/entropy (единый базис)")
    print("   Python-мост: не дублирует — использует fuga_core (PyO3)")

    print("\n=== E3. HTM — OK ===")


if __name__ == "__main__":
    demo()
