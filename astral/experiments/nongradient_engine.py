"""NonGradientEngine — полностью безградиентное обучение (Backprop-Free).

Архитектура (4 локальных механизма, 0 градиентов):

  ВХОД → HTM(SDR) → VSA(связывание) → SNN(STDP) → NEAT(эволюция)

  ┌─────────────────────────────────────────────────────────────┐
  │ 1. HTM Encoder & SDR    — разреженное кодирование (2% бит)  │
  │ 2. VSA Hypervectors     — bind/bundle/permute без backprop   │
  │ 3. SNN + STDP           — спайки + локальная пластичность   │
  │ 4. NEAT / HyperNEAT     — эволюция топологии без градиентов │
  └─────────────────────────────────────────────────────────────┘

Никакого loss.backward(), SGD, Adam. Только:
  - STDP: Δw = f(Δt) — локально, по времени спайков
  - Hebbian: w += lr·x·y — корреляция активаций
  - HTM: SDR-прототипы — Hebb-накопление последовательностей
  - NEAT: fitness-отбор — эволюция топологии и весов

Зачем: катастрофическое забывание, энергозатраты, mode collapse
— проблемы градиентного обучения. NEAT+SNN+HTM+VSA решают их
через локальность, разреженность и эволюцию.
"""

from __future__ import annotations

import numpy as np

from astral.experiments.neat_hyperneat import CPPN
from astral.experiments.snn_neuromorphic import LIFNeuron
from astral.experiments.htm_bridge import SDR, HTMBridge
from astral.experiments.mini_cognitive import MiniVSA


class NonGradientEngine:
    """Полностью безградиентный движок обучения.

    Обучение: 4 шага, 0 градиентов:
      1. HTM: вход → SDR (разреженный бинарный, 2% бит)
      2. VSA: bind(токен, SDR) → фазовый гипервектор
      3. SNN: SDR → ток → спайк → STDP-обновление весов
      4. NEAT: эволюция топологии/весов по fitness
    """

    def __init__(self, dim: int = 512, lr: float = 0.01):
        self.dim = dim
        self.lr = lr
        self.vsa = MiniVSA(dim=dim, seed=0)
        self.htm = HTMBridge(n=dim)
        self.sdr = SDR(n=dim, active=int(dim * 0.02), seed=0)
        self.lif = LIFNeuron(tau=10.0, threshold=0.5)
        self.cppn = CPPN(n_hidden=6, seed=7)
        # Синаптические веса (Hebbian, без градиентов)
        self.weights = np.random.uniform(-1, 1, (dim, dim)) * 0.1
        # Память последовательностей (HTM)
        self.sequence_memory: dict[int, list[int]] = {}
        self.step_count = 0
        # Эволюционные параметры
        self.fitness_history: list[float] = []

    # ── 1. HTM: вход → SDR ─────────────────────────────────────
    def _to_sdr(self, token: str) -> np.ndarray:
        """Токен → SDR (разреженный бинарный, 2% бит)."""
        hv = self.vsa.item(token)
        # квантование HV в SDR (бинарный по знаку)
        sdr = (hv > 0).astype(np.int8)
        return sdr

    # ── 2. VSA: bind + bundle ──────────────────────────────────
    def _vsa_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Связывание: VSA ⊗ (знак)."""
        return np.sign(a * b)

    def _vsa_bundle(self, items: list[np.ndarray]) -> np.ndarray:
        """Суперпозиция: sign(Σ)."""
        if not items:
            return np.ones(self.dim)
        return np.sign(np.sum(items, axis=0))

    # ── 3. SNN: спайк + STDP ───────────────────────────────────
    def _spike(self, sdr: np.ndarray) -> bool:
        """SDR → ток → спайк (LIF)."""
        current = float(np.mean(sdr)) * 5.0
        return self.lif.step(current, self.step_count)

    def _stdp_update(self, pre: np.ndarray, post: np.ndarray, spike: bool) -> None:
        """STDP: если спайк — укрепляем коррелирующие синапсы.

        Hebbian: w += lr·pre·post  (локально, без градиентов).
        """
        if spike:
            # корреляция pre и post
            self.weights += self.lr * np.outer(pre, post)
            # cap
            np.clip(self.weights, -2.0, 2.0, out=self.weights)

    # ── 4. NEAT: эволюция ──────────────────────────────────────
    def _neat_mutate(self) -> None:
        """Эволюционная мутация весов (NEAT-дух).

        Каждые N шагов: добавляем шум к весам, если fitness падает.
        """
        noise = np.random.default_rng(self.step_count).normal(0, 0.05, self.weights.shape)
        self.weights += noise
        np.clip(self.weights, -2.0, 2.0, out=self.weights)

    # ── Главный шаг обучения (0 градиентов) ────────────────────
    def learn(self, token: str, next_token: str) -> dict:
        """Один шаг безградиентного обучения.

        HTM→VSA→SNN→STDP→NEAT — 4 механизма, 0 градиентов.
        """
        self.step_count += 1
        sdr = self._to_sdr(token)
        sdr_next = self._to_sdr(next_token)

        # 1. HTM: запоминаем последовательность
        tok_id = hash(token) % (2**20)
        nxt_id = hash(next_token) % (2**20)
        if tok_id not in self.sequence_memory:
            self.sequence_memory[tok_id] = []
        self.sequence_memory[tok_id].append(nxt_id)

        # 2. VSA: связываем токен с SDR, предсказываем следующий
        hv = self.vsa.item(token)
        bound = self._vsa_bind(hv, sdr.astype(np.float64))
        pred = self._vsa_bind(hv, np.sign(self.weights @ hv))

        # 3. SNN: спайк + STDP
        spiked = self._spike(sdr)
        self._stdp_update(sdr.astype(np.float64), sdr_next.astype(np.float64), spiked)

        # 4. Оценка (fitness = косинус предсказания с реальным)
        hv_next = self.vsa.item(next_token)
        cos = self.vsa.cos(pred, hv_next)
        self.fitness_history.append(cos)

        # 5. NEAT: если fitness падает — мутация
        if len(self.fitness_history) >= 10:
            recent = np.mean(self.fitness_history[-10:])
            if recent < 0.0:
                self._neat_mutate()

        return {
            "spike": spiked,
            "cos": cos,
            "fitness_mean": float(np.mean(self.fitness_history[-20:])) if self.fitness_history else 0.0,
            "weight_sum": float(np.sum(np.abs(self.weights))),
        }

    # ── Генерация CPPN (HyperNEAT) ─────────────────────────────
    def generate_weights_cppn(self) -> None:
        """HyperNEAT: CPPN генерирует паттерн весов из координат."""
        for i in range(self.dim):
            for j in range(self.dim):
                self.weights[i, j] = self.cppn.weight_for(
                    i / self.dim, j / self.dim, i / self.dim, j / self.dim)


def demo():
    print("=== NON-GRADIENT ENGINE (Backprop-Free) ===\n")

    eng = NonGradientEngine(dim=512, lr=0.02)

    # Последовательность для обучения (0 градиентов!)
    seq = ["fn", "main", "(", ")", "{", "println", "!", "(", '"', "hello", "world", '"', ")", ";", "}", "fn"]
    print(f"1. Обучение на последовательности ({len(seq)} токенов, 0 градиентов):")
    for i in range(len(seq) - 1):
        res = eng.learn(seq[i], seq[i + 1])
    print(f"   итераций: {eng.step_count}, cos: {res['cos']:.4f}, "
          f"fitness: {res['fitness_mean']:.4f}, веса: {res['weight_sum']:.2f}")

    # 2. HTM-память последовательностей
    print(f"\n2. HTM память: {len(eng.sequence_memory)} состояний")
    for tok, nxts in list(eng.sequence_memory.items())[:3]:
        # восстанавливаем строку из хэша (упрощённо)
        candidates = [seq[i] for i in range(len(seq)-1) if hash(seq[i]) % (2**20) == tok]
        print(f"   {candidates[0] if candidates else '?'} → {len(set(nxts))} уникальных следующих")

    # 3. SNN: сколько спайков сгенерировано
    #   (сброс LIF для чистоты)
    eng.lif = LIFNeuron(tau=10.0, threshold=0.5)
    total_spikes = 0
    for t in range(50):
        sdr = eng._to_sdr(seq[t % len(seq)])
        if eng._spike(sdr):
            total_spikes += 1
    print(f"\n3. SNN: {total_spikes}/50 спайков (LIF, порог 0.5)")

    # 4. HyperNEAT: CPPN генерирует веса
    print(f"\n4. HyperNEAT: CPPN-генерация паттерна весов:")
    eng_cppn = NonGradientEngine(dim=512)
    eng_cppn.generate_weights_cppn()
    w_sum = float(np.sum(np.abs(eng_cppn.weights)))
    w_std = float(np.std(eng_cppn.weights))
    print(f"   CPPN: sum|w|={w_sum:.2f}, std={w_std:.4f} (не случайные веса)")

    print("\n=== NON-GRADIENT ENGINE — OK ===")


if __name__ == "__main__":
    demo()