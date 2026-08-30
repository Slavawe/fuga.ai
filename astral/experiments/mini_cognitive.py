"""Mini Cognitive Stack — мини-версии H-JEPA + VL-JEPA + VSA, связанные
с экспериментальными NEAT/HyperNEAT/SNN/HTM в ОДИН конвейер.

Экспериментальный модуль (astral/experiments/ — песочница).

Компоненты:
  M1. MiniVSA    — биполярные гипервекторы: bind(⊗)/bundle(+)/permute
  M2. MiniHJEPA  — Widrow-Hoff предиктор следующего латента
  M3. MiniVLJEPA — vision+text → общий VSA-латент (два проектора)

Связка с экспериментальными технологиями:
  NEAT       — эволюционирует ВЕСА MiniHJEPA (геном → веса W)
  HyperNEAT  — CPPN генерирует веса проектора из координат
  SNN        — спайковый энкодер входа (альтернатива непрерывному)
  HTM        — предсказание следующего состояния (для управления)

Поток: текст/vision → MiniVLJEPA → VSA-латент → MiniHJEPA (Widrow-Hoff
или NEAT-веса) → следующее состояние → HTM-проверка → SNN-эмиссия.

Всё на numpy (лёгкий, без torch) — self-contained для экспериментов.
"""

from __future__ import annotations

import math

import numpy as np


# ═══════════════════════════════════════════════════════════════
# M1. MiniVSA — биполярные гипервекторы
# ═══════════════════════════════════════════════════════════════
class MiniVSA:
    """Биполярные гипервекторы (±1)^dim.

    bind(x,y)  = x ⊗ y  (поэлементное умножение) — связывание
    bundle(...)= sign(Σ) — суперпозиция
    permute(x) = сдвиг — порядок/позиция
    """

    def __init__(self, dim: int = 512, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._item_cache: dict[str, np.ndarray] = {}

    def item(self, name: str) -> np.ndarray:
        """Детерминированный HV для токена."""
        if name not in self._item_cache:
            rng = np.random.default_rng(hash(name) % (2**32))
            self._item_cache[name] = np.sign(rng.uniform(-1, 1, self.dim))
        return self._item_cache[name]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sign(a * b)

    def bundle(self, items: list[np.ndarray]) -> np.ndarray:
        if not items:
            return np.ones(self.dim)
        return np.sign(np.sum(items, axis=0))

    def permute(self, a: np.ndarray, k: int = 1) -> np.ndarray:
        return np.roll(a, k)

    def cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def encode_sequence(self, tokens: list[str]) -> np.ndarray:
        """Связать токены с их позициями, затем сбандлить."""
        return self.bundle([
            self.bind(self.item(t), self.permute(np.ones(self.dim), i))
            for i, t in enumerate(tokens)
        ])


# ═══════════════════════════════════════════════════════════════
# M2. MiniHJEPA — Widrow-Hoff предиктор следующего латента
# ═══════════════════════════════════════════════════════════════
class MiniHJEPA:
    """Предсказание z_{t+1} = W·z_t (Widrow-Hoff, без backprop).

    W [dim, dim] обучается: W += lr·(z_next − W·z_t)⊗z_t.
    JEPA-идея: предсказываем в ЛАТЕНТНОМ пространстве, не в байтах.
    """

    def __init__(self, dim: int = 512, lr: float = 0.001):
        self.dim = dim
        self.lr = lr
        self.W = np.zeros((dim, dim))
        self.trained_steps = 0

    def predict(self, z: np.ndarray) -> np.ndarray:
        """z_next = W·z (нормализованный)."""
        pred = self.W @ z
        n = np.linalg.norm(pred) + 1e-9
        return pred / n

    def learn(self, z_t: np.ndarray, z_next: np.ndarray) -> float:
        """Один шаг Widrow-Hoff. Возвращает ошибку."""
        pred = self.W @ z_t
        err = z_next - pred
        self.W += self.lr * np.outer(err, z_t)
        self.trained_steps += 1
        return float(np.linalg.norm(err))

    def load_weights(self, W: np.ndarray) -> None:
        """Подставить эволюционированные веса (из NEAT)."""
        assert W.shape == self.W.shape, f"NEAT-веса {W.shape} != W {self.W.shape}"
        self.W = W.astype(np.float64)


# ═══════════════════════════════════════════════════════════════
# M3. MiniVLJEPA — vision + text → общий VSA-латент
# ═══════════════════════════════════════════════════════════════
class MiniVLJEPA:
    """Два проектора в общее VSA-пространство.

    text:  токены → VSA-бандл → Linear → латент [dim]
    vision: фича [v_dim] → Linear → латент [dim]
    Общий латент позволяет сравнить текст и изображение косинусом.
    """

    def __init__(self, dim: int = 512, v_dim: int = 64, seed: int = 0):
        self.dim = dim
        self.vsa = MiniVSA(dim=dim, seed=seed)
        rng = np.random.default_rng(seed)
        self.text_w = rng.uniform(-1, 1, (dim, dim)) / math.sqrt(dim)
        self.text_b = np.zeros(dim)
        self.img_w = rng.uniform(-1, 1, (dim, v_dim)) / math.sqrt(v_dim)
        self.img_b = np.zeros(dim)

    def encode_text(self, text: str) -> np.ndarray:
        """Текст → бандл токенов → Linear → нормированный латент."""
        tokens = text.lower().split()
        hv = self.vsa.encode_sequence(tokens[:8])  # окно 8 токенов
        z = self.text_w @ hv + self.text_b
        return z / (np.linalg.norm(z) + 1e-9)

    def encode_vision(self, feat: np.ndarray) -> np.ndarray:
        """Вектор фич изображения → латент."""
        z = self.img_w @ feat + self.img_b
        return z / (np.linalg.norm(z) + 1e-9)

    def cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ═══════════════════════════════════════════════════════════════
# M4. Связка с экспериментальными технологиями
# ═══════════════════════════════════════════════════════════════
def evolve_hjepa_weights_neat(hjepa: MiniHJEPA, trajectory: list[np.ndarray],
                              generations: int = 5) -> None:
    """NEAT-эволюция весов MiniHJEPA.

    Геном = список связей (как в neat_hyperneat), fitness = −средняя
    ошибка предсказания на траектории. Упрощённо: эволюционируем
    K случайных масок/весов, оставляем лучший.
    """
    best_err = float("inf")
    best_W = hjepa.W.copy()
    rng = np.random.default_rng(42)
    for gen in range(generations):
        # кандидат: W + шум (эволюционная мутация весов)
        candidate = hjepa.W + rng.normal(0, 0.1, hjepa.W.shape)
        err = 0.0
        for i in range(len(trajectory) - 1):
            pred = candidate @ trajectory[i]
            err += float(np.linalg.norm(trajectory[i + 1] - pred))
        err /= max(1, len(trajectory) - 1)
        if err < best_err:
            best_err, best_W = err, candidate
    hjepa.load_weights(best_W)
    print(f"  [NEAT] веса MiniHJEPA: ошибка {best_err:.4f} "
          f"(эволюционный поиск по {generations} поколениям)")


def generate_weights_hyperneat(vljepa: MiniVLJEPA, dim: int = 512) -> None:
    """HyperNEAT: CPPN генерирует веса проектора из координат.

    CPPN (neat_hyperneat.py) создаёт паттерн весов img_w из сетки
    координат — «генотип → фенотип» без ручной настройки.
    """
    from astral.experiments.neat_hyperneat import CPPN

    cppn = CPPN(n_hidden=6, seed=7)
    v_dim = vljepa.img_w.shape[1]
    new_w = np.zeros_like(vljepa.img_w)
    for i in range(new_w.shape[0]):
        for j in range(new_w.shape[1]):
            # координаты в [0,1] — CPPN даёт вес связи
            new_w[i, j] = cppn.weight_for(i / dim, j / v_dim, i / dim, j / v_dim)
    vljepa.img_w = new_w
    print(f"  [HyperNEAT] CPPN сгенерировал img_w "
          f"{new_w.shape[0]}×{new_w.shape[1]}")


def run_snn_emit(hjepa: MiniHJEPA, z: np.ndarray, n_steps: int = 20) -> int:
    """SNN: спайковый канал на выходе предсказания.

    Считаем предсказание как «ток» в LIF-нейрон — сколько спайков.
    Ток нормирован: чем «сильнее» предсказание, тем больше спайков.
    """
    from astral.experiments.snn_neuromorphic import LIFNeuron

    pred = hjepa.predict(z)
    neuron = LIFNeuron(tau=8.0, threshold=0.5)
    spikes = 0
    for t in range(n_steps):
        # ток = интенсивность предсказания (значимо > 0)
        current = float(np.mean(np.abs(pred))) * 3.0
        if neuron.step(current, t):
            spikes += 1
    return spikes


def run_htm_validate(states: list[int]) -> tuple[int, int]:
    """HTM: предсказать следующее состояние в циклической последовательности."""
    from astral.experiments.htm_bridge import HTMBridge

    htm = HTMBridge(n=512)
    htm.train(states, epochs=10)
    ok = 0
    for i in range(len(states) - 1):
        pred = htm.predict_next(states[i], candidates=list(set(states)))
        if pred == states[i + 1]:
            ok += 1
    return ok, len(states) - 1


def demo():
    print("=== MINI COGNITIVE STACK (H-JEPA + VL-JEPA + VSA) ===\n")

    # 1. MiniVSA: операции
    print("1. MiniVSA (dim=512):")
    vsa = MiniVSA(dim=512, seed=0)
    a = vsa.item("cube")
    b = vsa.item("sphere")
    ab = vsa.bind(a, b)
    print(f"   cos(cube, sphere) = {vsa.cos(a, b):.3f} (≈0 = разные)")
    print(f"   cos(cube⊗sphere, cube) = {vsa.cos(ab, a):.3f} (связка)")

    # 2. MiniHJEPA: обучение предсказанию
    print("\n2. MiniHJEPA (Widrow-Hoff):")
    hjepa = MiniHJEPA(dim=512, lr=0.005)
    rng = np.random.default_rng(1)
    trajectory = []
    for _ in range(200):
        trajectory.append(rng.uniform(-1, 1, 512) / math.sqrt(512))
    losses = []
    for i in range(len(trajectory) - 1):
        losses.append(hjepa.learn(trajectory[i], trajectory[i + 1]))
    print(f"   loss: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"({'сходится' if losses[-1] < losses[0] else 'расходится'})")

    # 3. MiniVLJEPA: текст и vision в общем пространстве
    print("\n3. MiniVLJEPA (text + vision → общий латент):")
    vljepa = MiniVLJEPA(dim=512, v_dim=64, seed=2)
    z_cube = vljepa.encode_text("a red cube on a table")
    z_sphere = vljepa.encode_text("a blue sphere in the sky")
    z_img = vljepa.encode_vision(rng.uniform(-1, 1, 64))
    print(f"   cos(text-cube, text-sphere) = {vljepa.cos(z_cube, z_sphere):.3f}")
    print(f"   cos(text-cube, vision)      = {vljepa.cos(z_cube, z_img):.3f}")

    # 4. Связка: NEAT эволюционирует веса, HyperNEAT генерирует проектор
    print("\n4. Связка с экспериментами:")
    evolve_hjepa_weights_neat(hjepa, trajectory, generations=5)
    generate_weights_hyperneat(vljepa, dim=512)

    # 5. SNN-эмиссия + HTM-валидация
    print("\n5. SNN + HTM на предсказании:")
    spikes = run_snn_emit(hjepa, trajectory[-1])
    print(f"   SNN: предсказание → {spikes} спайков (LIF-канал)")
    ok, total = run_htm_validate([0, 1, 2, 0, 1, 2])
    print(f"   HTM: предсказание цикла {ok}/{total}")

    print("\n=== MINI COGNITIVE STACK — OK ===")


if __name__ == "__main__":
    demo()
