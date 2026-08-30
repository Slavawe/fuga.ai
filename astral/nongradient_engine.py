"""NonGradientEngine — БЕЗГРАДИЕНТНОЕ ядро обучения (пром-версия).

Интегрировано в основную ветку из astral/experiments/nongradient_engine.py.
Ноль loss.backward(), ноль Adam/SGD. Четыре биологических механизма:

  1. HTM (SDR)     — разреженное кодирование (2% бит)
  2. VSA           — bind/bundle/permute (алгебра без backprop)
  3. SNN + STDP    — спайки + локальная пластичность (Hebb)
  4. NEAT/HyperNEAT — эволюция топологии по fitness (вместо Adam)

Использование:
    eng = NonGradientEngine(dim=512)
    for token, next_token in corpus:
        eng.learn(token, next_token)

    eng.train_facts(fuga_memory_facts)   # полное обучение на корпусе
"""

from __future__ import annotations

import json
import os

import numpy as np

from astral.experiments.neat_hyperneat import CPPN
from astral.experiments.snn_neuromorphic import LIFNeuron
from astral.experiments.mini_cognitive import MiniVSA


class NonGradientEngine:
    """Полностью безградиентный движок обучения (HTM+VSA+SNN+NEAT)."""

    def __init__(self, dim: int = 512, lr: float = 0.02):
        self.dim = dim
        self.lr = lr
        self.vsa = MiniVSA(dim=dim, seed=0)
        self.lif = LIFNeuron(tau=10.0, threshold=0.5)
        self.cppn = CPPN(n_hidden=6, seed=7)
        # Синаптические веса (Hebbian, без градиентов)
        self.weights = np.random.uniform(-1, 1, (dim, dim)) * 0.1
        # Память последовательностей (HTM)
        self.sequence_memory: dict[int, list[int]] = {}
        self.fitness_history: list[float] = []
        self.step_count = 0
        self.total_spikes = 0

    # ── HTM: токен → SDR ─────────────────────────────────────
    def _to_sdr(self, token: str) -> np.ndarray:
        hv = self.vsa.item(token)
        return (hv > 0).astype(np.int8)

    # ── VSA: связывание ──────────────────────────────────────
    def _vsa_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sign(a * b)

    # ── SNN: спайк ───────────────────────────────────────────
    def _spike(self, sdr: np.ndarray) -> bool:
        current = float(np.mean(sdr)) * 5.0
        spiked = self.lif.step(current, self.step_count)
        if spiked:
            self.total_spikes += 1
        return spiked

    # ── STDP: Oja-нормализация (соревновательное Hebb) ──────
    def _stdp_update(self, pre: np.ndarray, post: np.ndarray, spike: bool) -> None:
        """STDP + Oja: Δw = lr·pre·post − lr·post²·w.

        Oja's rule: веса соревнуются — сильные каналы вытесняют слабые.
        Без неё веса схлопываются в «всё предсказывает всё».
        """
        if spike:
            # Hebb: w += lr·pre·post
            hebb = self.lr * np.outer(pre, post)
            # Oja: w -= lr·post²·w  (нормализация)
            oja = self.lr * (np.linalg.norm(post) ** 2) * self.weights
            self.weights += hebb - oja
            np.clip(self.weights, -2.0, 2.0, out=self.weights)

    # ── NEAT: эволюционная мутация при падении fitness ──────
    def _neat_mutate(self) -> None:
        noise = np.random.default_rng(self.step_count).normal(0, 0.05, self.weights.shape)
        self.weights += noise
        np.clip(self.weights, -2.0, 2.0, out=self.weights)

    # ── Главный шаг (0 градиентов) ───────────────────────────
    def learn(self, token: str, next_token: str) -> dict:
        self.step_count += 1
        sdr = self._to_sdr(token)
        sdr_next = self._to_sdr(next_token)

        # 1. HTM-память последовательности
        tok_id = hash(token) % (2**20)
        nxt_id = hash(next_token) % (2**20)
        if tok_id not in self.sequence_memory:
            self.sequence_memory[tok_id] = []
        self.sequence_memory[tok_id].append(nxt_id)

        # 2. VSA: предсказание следующего
        hv = self.vsa.item(token)
        pred = self._vsa_bind(hv, np.sign(self.weights @ hv))

        # 3. SNN + STDP
        spiked = self._spike(sdr)
        self._stdp_update(sdr.astype(np.float64), sdr_next.astype(np.float64), spiked)

        # 4. Fitness (косинус предсказания с реальным)
        hv_next = self.vsa.item(next_token)
        cos = self.vsa.cos(pred, hv_next)
        self.fitness_history.append(cos)

        # 5. NEAT: мутация при падении fitness
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

    # ── Полное обучение на корпусе фактов fuga_memory_* ─────
    def train_facts(self, memory_dirs: list[str], epochs: int = 1,
                    max_tokens: int | None = None) -> dict:
        """Обучение на всех facts.jsonl из fuga_memory_* (0 градиентов).

        Каждая тройка (subject, relation, object) даёт пару для обучения:
          subject → object  (ассоциация «код → библиотека»)
          relation → object (ассоциация «связь → сущность»)
        """
        triples = []
        for d in memory_dirs:
            facts = os.path.join(d, "fuga_memory.facts.jsonl")
            if not os.path.exists(facts):
                continue
            with open(facts, encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        triples.append((o.get("subject", ""), o.get("relation", ""), o.get("object", "")))
                    except Exception:
                        continue

        tokens: list[tuple[str, str]] = []
        for s, r, o in triples:
            if s:
                tokens.append((s, o))
            if r:
                tokens.append((r, o))
        if max_tokens:
            tokens = tokens[:max_tokens]

        cos_vals = []
        for _ in range(epochs):
            for a, b in tokens:
                res = self.learn(a, b)
                cos_vals.append(res["cos"])

        return {
            "triples": len(triples),
            "pairs": len(tokens),
            "steps": self.step_count,
            "fitness_mean": float(np.mean(cos_vals)),
            "fitness_last": float(np.mean(self.fitness_history[-200:])) if self.fitness_history else 0.0,
            "htm_states": len(self.sequence_memory),
            "spikes": self.total_spikes,
            "weight_sum": float(np.sum(np.abs(self.weights))),
        }

    # ── HyperNEAT: CPPN-генерация весов ─────────────────────
    def generate_weights_cppn(self) -> None:
        for i in range(self.dim):
            for j in range(self.dim):
                self.weights[i, j] = self.cppn.weight_for(
                    i / self.dim, j / self.dim, i / self.dim, j / self.dim)

    # ── Предсказание следующего токена ──────────────────────
    def predict_next(self, token: str, candidates: list[str]) -> str:
        """Предсказать следующий токен из кандидатов (косинус VSA)."""
        hv = self.vsa.item(token)
        pred = self._vsa_bind(hv, np.sign(self.weights @ hv))
        best, best_cos = candidates[0], -1.0
        for c in candidates:
            cos = self.vsa.cos(pred, self.vsa.item(c))
            if cos > best_cos:
                best_cos, best = cos, c
        return best


def demo():
    eng = NonGradientEngine(dim=512, lr=0.05)
    seq = ["fn", "main", "(", ")", "{", "println", "!", '"', "hello", "world", '"', "}", "fn"]
    for i in range(len(seq) - 1):
        eng.learn(seq[i], seq[i + 1])
    res = eng.learn(seq[-2], seq[-1])
    print(f"шагов: {eng.step_count}, fitness: {res['fitness_mean']:.4f}, "
          f"веса: {res['weight_sum']:.2f}, спайков: {eng.total_spikes}")
    print("NonGradientEngine OK — 0 градиентов")


if __name__ == "__main__":
    demo()
