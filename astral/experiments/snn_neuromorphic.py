"""SNN (Spiking Neural Networks) + нейроморфные вычисления.

Экспериментальный модуль (astral/experiments/ — песочница).

E2a. LIFNeuron — Leaky Integrate-and-Fire нейрон:
  dv/dt = -v/τ + I_in; порог → спайк → reset.
  Аналог биологического нейрона: энергоэффективен (события, не потоки).

E2b. SpikingNetwork — слой LIF-нейронов:
  вход (скорость спайков) → мембранный потенциал → выходные спайки.
  Обучение: STDP (Spike-Timing-Dependent Plasticity):
    Δw = A+·exp(-Δt/τ+) если pre до post, A-·exp(-Δt/τ-) если наоборот.

E2c. NeuromorphicEncoder — кодирование в спайки (rate coding):
  значение x∈[0,1] → вероятность спайка в каждом временном окне.

Роль в Fuga: альтернативный «событийный» канал для пространственных
концептов (Spatial-JEPA) — спайки вместо непрерывных фаз; STDP вместо
Widrow-Hoff. Экспериментально: можно сравнить энергоэффективность и
обобщение на редких событиях.
"""

from __future__ import annotations

import numpy as np


class LIFNeuron:
    """Leaky Integrate-and-Fire нейрон (дискретное время)."""

    def __init__(self, tau: float = 20.0, threshold: float = 1.0,
                 reset: float = 0.0, dt: float = 1.0):
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.dt = dt
        self.v = 0.0  # мембранный потенциал
        self.spiked = False
        self.spike_times: list[int] = []  # времена спайков

    def step(self, current: float, t: int) -> bool:
        """Один шаг: утечка + вход → проверка порога."""
        # утечка: v *= exp(-dt/tau)
        self.v *= np.exp(-self.dt / self.tau)
        self.v += current
        self.spiked = self.v >= self.threshold
        if self.spiked:
            self.v = self.reset
            self.spike_times.append(t)
        return self.spiked

    def reset_state(self) -> None:
        self.v = 0.0
        self.spiked = False
        self.spike_times.clear()


class NeuromorphicEncoder:
    """Rate-coding: значение → вероятность спайка на каждом шаге.

    x∈[0,1] → P(spike) = x (каждый шаг Бернулли). Спайки по времени
    кодируют интенсивность.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def encode(self, x: float, n_steps: int) -> list[int]:
        """x ∈ [0,1] → времена спайков (n_steps окно)."""
        return [t for t in range(n_steps) if self.rng.random() < x]


class SpikingNetwork:
    """Слой LIF-нейронов + STDP-обучение.

    forward(inputs, n_steps):
      inputs: [n_in] скорости [0,1] → спайки входных нейронов
      выход: [n_out] число спайков каждого выходного нейрона

    learn_stdp(pre_spikes, post_spikes, lr):
      Δw = Σ lr·A·exp(-|Δt|/τ) по парам (pre_t, post_t)
    """

    def __init__(self, n_in: int, n_out: int, seed: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.uniform(-1.0, 1.0, (n_in, n_out))
        self.neurons = [LIFNeuron() for _ in range(n_out)]
        # STDP параметры
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.a_plus = 0.01
        self.a_minus = 0.012
        self.lr = 0.1

    def forward(self, inputs: list[float], n_steps: int = 50) -> list[int]:
        """Прогнать вход через слой, вернуть число спайков на выходе."""
        for n in self.neurons:
            n.reset_state()
        # входные спайки (rate coding)
        enc = NeuromorphicEncoder(seed=0)
        pre_spikes: list[list[int]] = [
            enc.encode(min(1.0, max(0.0, x)), n_steps) for x in inputs
        ]
        for t in range(n_steps):
            for j, neuron in enumerate(self.neurons):
                current = sum(
                    self.weights[i][j] * (1.0 if t in pre_spikes[i] else 0.0)
                    for i in range(self.n_in)
                )
                neuron.step(current, t)
        return [len(n.spike_times) for n in self.neurons]

    def stdp_update(self, pre_spikes: list[int], post_spike: int,
                    lr: float | None = None) -> None:
        """STDP по одной паре (вход спайк, выход спайк)."""
        lr = lr or self.lr
        dt = post_spike - pre_spikes[0]
        if dt > 0:  # pre до post → потенциация
            delta = lr * self.a_plus * np.exp(-dt / self.tau_plus)
        else:       # post до pre → депрессия
            delta = -lr * self.a_minus * np.exp(dt / self.tau_minus)
        # упрощённо: обновляем вес первой связи (для демо)
        self.weights[0][0] = np.clip(self.weights[0][0] + delta, -2, 2)


def demo():
    print("=== E2. SNN + НЕЙРОМОРФНЫЕ ===\n")

    # 1. LIF нейрон: спайки при входе
    print("1. LIFNeuron: ответ на постоянный ток:")
    n = LIFNeuron(tau=10.0, threshold=1.0)
    spikes = 0
    for t in range(100):
        if n.step(0.3, t):
            spikes += 1
    print(f"   вход 0.3 × 100 шагов → {spikes} спайков (порог 1.0, τ=10)")

    # 2. NeuromorphicEncoder: rate coding
    print("\n2. NeuromorphicEncoder: x → спайки:")
    enc = NeuromorphicEncoder(seed=1)
    for x in [0.1, 0.5, 0.9]:
        sp = enc.encode(x, 50)
        print(f"   x={x}: {len(sp)}/50 спайков (ожид ~{int(x*50)})")

    # 3. SpikingNetwork: обучение через STDP (ассоциация)
    print("\n3. SpikingNetwork: STDP ассоциация (вход A → выход спайк):")
    net = SpikingNetwork(n_in=2, n_out=1, seed=3)
    # позитивные веса: вход может возбуждать выход
    net.weights = np.array([[1.5, 0.0], [0.1, 0.0]], dtype=float)
    # вход A=0.9 (часто спайкует), B=0.1 (редко)
    out_before = net.forward([0.9, 0.1], n_steps=50)
    # STDP: усиливаем связь от входа 0 к выходу
    net.stdp_update(pre_spikes=list(range(0, 10)), post_spike=5)
    out_after = net.forward([0.9, 0.1], n_steps=50)
    print(f"   спайки выхода: до STDP={out_before[0]}, после={out_after[0]} "
          f"(w[0][0]={net.weights[0][0]:.3f})")

    print("\n=== E2. SNN + НЕЙРОМОРФНЫЕ — OK ===")


if __name__ == "__main__":
    demo()
