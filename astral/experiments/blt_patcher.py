"""BLT (Byte Latent Transformer) — энтропийное патчевание байт.

Идея BLT (Meta AI, 2024): вместо фиксированных токенов — ПЕРЕМЕННЫЕ
патчи байт, границы которых определяются ЭНТРОПИЕЙ следующего байта.

  - Низкая энтропия (предсказуемый байт) → байт входит в текущий патч
  - Высокая энтропия (неожиданный байт) → начинается НОВЫЙ патч

Зачем: редкие/сложные байты (начала слов, код-символы) получают
ОТДЕЛЬНЫЕ патчи, а частые (пробелы, гласные) — сливаются в длинные.
Это концентрирует вычислительные ресурсы на «информативных» местах.

В Fuga: BLT-патчер — альтернатива фиксированному окну W_patch.
Патчи переменной длины → предсказание патча (W_patch) → байты внутри.

Сравнение:
  - Токенизация (BPE): фикс. словарь, подбор под корпус
  - MegaByte: фикс. длина патча (например, 4 байта)
  - BLT: ПЕРЕМЕННАЯ длина по энтропии — адаптивно к контенту
"""

from __future__ import annotations

import math
from collections import Counter


class EntropyEstimator:
    """Оценка энтропии следующего байта по частотной модели.

    P(next | history) ≈ bigram/unigram смесь (Witten-Bell сглаживание).
    """

    def __init__(self, vocab: int = 256):
        self.vocab = vocab
        self.unigrams: list[int] = [0] * vocab
        self.bigrams: list[list[int]] = [[0] * vocab for _ in range(vocab)]
        self.total_bytes = 0

    def learn(self, data: bytes) -> None:
        for i, b in enumerate(data):
            self.unigrams[b] += 1
            self.total_bytes += 1
            if i > 0:
                self.bigrams[data[i - 1]][b] += 1

    def surprise(self, history: bytes) -> float:
        """Неожиданность следующего байта: 1 − P(max|last).

        P(max|last) — вероятность самого вероятного продолжения
        (bigram+unigram смесь). 0 = предсказуемый байт, 1 = полная
        неожиданность. BLT режет патч при ВЫСОКОЙ неожиданности.
        """
        if self.total_bytes == 0:
            return 0.5
        last = history[-1] if history else 32  # space
        bigram_count = sum(self.bigrams[last])
        lam = bigram_count / (bigram_count + 5.0) if bigram_count > 0 else 0.0
        best_p = 0.0
        for b in range(self.vocab):
            pb = self.bigrams[last][b] / max(1, bigram_count) if bigram_count > 0 else 0.0
            pu = self.unigrams[b] / max(1, self.total_bytes)
            p = lam * pb + (1.0 - lam) * pu
            if p > best_p:
                best_p = p
        return float(1.0 - best_p)


class BLTPatcher:
    """Патчи байт переменной длины по энтропии (BLT-style).

    threshold_hi: если энтропия > порога → НОВЫЙ патч
    (информативный байт — отдельная группа).
    min_patch/max_patch: границы длины.
    """

    def __init__(self, threshold_hi: float = 0.85, min_patch: int = 1,
                 max_patch: int = 16):
        self.threshold_hi = threshold_hi
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.estimator = EntropyEstimator()

    def fit(self, corpus: list[bytes]) -> None:
        """Учим распределение байтов на корпусе."""
        for data in corpus:
            self.estimator.learn(data)

    def patch(self, data: bytes) -> list[bytes]:
        """Разбить байты на патчи по энтропии."""
        patches: list[bytes] = []
        current = bytearray()
        for i, b in enumerate(data):
            # энтропия следующего байта по последним 4 байтам истории
            history = data[max(0, i - 4) : i]
            if len(history) < 1:
                h = 0.0
            else:
                h = self.estimator.surprise(history)
            # начинаем новый патч если:
            #  1) энтропия высокая (информативный байт)
            #  2) текущий патч достиг max_patch
            if len(current) >= self.max_patch or (
                len(current) >= self.min_patch and h > self.threshold_hi
            ):
                patches.append(bytes(current))
                current = bytearray()
            current.append(b)
        if current:
            patches.append(bytes(current))
        return patches

    def patch_stats(self, data: bytes) -> dict:
        """Статистика патчевания."""
        patches = self.patch(data)
        lens = [len(p) for p in patches]
        return {
            "n_patches": len(patches),
            "mean_len": sum(lens) / max(1, len(lens)),
            "max_len": max(lens) if lens else 0,
            "min_len": min(lens) if lens else 0,
            "compression": len(data) / max(1, len(patches)),
        }


def demo():
    print("=== BLT (BYTE LATENT TRANSFORMER) ===\n")

    # Корпус: английский текст + код (как в Fuga)
    corpus = [
        b"the quick brown fox jumps over the lazy dog the quick brown fox",
        b"fn main() { println!(\"hello world\"); } fn main() { println!(\"rust\") }",
        b"let x = 42; let y = x * 2; let z = y + 1; return z;",
        b"import numpy as np\nx = np.array([1, 2, 3])\nprint(x.mean())",
    ]

    print("1. Обучение оценки энтропии на корпусе:")
    patcher = BLTPatcher(threshold_hi=3.5, min_patch=2, max_patch=16)
    patcher.fit(corpus)
    print(f"   bigram-модель обучена (256×256)")

    print("\n2. Патчевание по энтропии:")
    for data in corpus[:3]:
        stats = patcher.patch_stats(data)
        patches = patcher.patch(data)
        print(f"   '{data[:40].decode(errors='replace')}...'")
        print(f"     {len(data)}B → {stats['n_patches']} патчей, "
              f"средний {stats['mean_len']:.1f}B, сжатие ×{stats['compression']:.1f}")

    print("\n3. Сравнение с фиксированным окном (MegaByte, 4B):")
    for data in corpus[:2]:
        fixed = [data[i : i + 4] for i in range(0, len(data), 4)]
        blt = patcher.patch(data)
        print(f"   {len(data)}B: фикс-4B={len(fixed)} патчей, "
              f"BLT={len(blt)} патчей "
              f"({'BLT компактнее' if len(blt) < len(fixed) else 'фикс компактнее'})")

    print("\n=== BLT — OK ===")


if __name__ == "__main__":
    demo()
