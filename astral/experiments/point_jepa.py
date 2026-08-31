"""Point-JEPA: самопрогнозирующее обучение на облаках точек.

Идея (JEPA на облаке точек):
  Облако точек P = {p_1, ..., p_n} (3D) → латент z_P (VSA).
  Окно w_i = {p_{i-w}, ..., p_{i-1}} → предиктор W → z_{i+1}.
  Widrow-Hoff (без градиентов): err = z_{i+1} − W·z_ctx.

Зачем Point-JEPA:
  1. Инвариантность к числу точек (облако → одно латентное состояние).
  2. Разделимость форм: сфера / куб / плоскость → ортогональные латенты.
  3. Предсказание следующего "момента" сканирования (темпоральность).

Всё на VSA-базисе (FPE фазы), без backprop — философия проекта.
"""

from __future__ import annotations

import numpy as np

from astral.experiments.mini_cognitive import MiniVSA


class PointCloudEncoder:
    """Облако точек → VSA-латент (позиционная свёртка).

    Каждая точка p=(x,y,z) → фазовый HV e^{i(x·ωx + y·ωy + z·ωz)}.
    Облако → сумма (bundle) фаз всех точек, затем sign() → биполярный.

    Свойства:
      - Инвариантность к порядку точек (сумма коммутативна).
      - Опечаткоустойчивость: шум в координатах → малый сдвиг латента.
    """

    def __init__(self, dim: int = 512, spatial_freq: float = 2.0, seed: int = 7):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Пространственные частоты для X/Y/Z (разные, разделимость)
        self.omega = {
            "x": rng.uniform(-spatial_freq, spatial_freq, dim),
            "y": rng.uniform(-spatial_freq, spatial_freq, dim),
            "z": rng.uniform(-spatial_freq, spatial_freq, dim),
        }

    def _phase(self, x: float, y: float, z: float) -> np.ndarray:
        """Фазовый HV точки: e^{i(x·ωx + y·ωy + z·ωz)}."""
        angle = x * self.omega["x"] + y * self.omega["y"] + z * self.omega["z"]
        return np.exp(1j * angle)

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Облако (N×3) → биполярный латент (dim,), нормированный."""
        hvs = [self._phase(*p) for p in points]
        bundle = np.sum(hvs, axis=0)  # сумма фаз = bundle (коммутативно)
        lat = np.sign(bundle.real).astype(np.float32)
        # Единичная норма — стабильность Widrow-Hoff/Oja
        n = np.linalg.norm(lat)
        if n > 1e-9:
            lat = lat / n
        return lat

    def cos(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинус между латентами."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def make_shape(shape: str, n: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """Синтетическое облако точек для формы.

    shapes: 'sphere', 'cube', 'plane', 'line', 'cluster'
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if shape == "sphere":
        # Точки на сфере радиуса 1
        u = rng.uniform(0, 1, n)
        v = rng.uniform(0, 1, n)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        return np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ], axis=1)
    if shape == "cube":
        # Точки на гранях куба [-1, 1]³
        pts = []
        for _ in range(n):
            axis = rng.integers(0, 3)
            sign = rng.choice([-1.0, 1.0])
            p = rng.uniform(-1, 1, 3)
            p[axis] = sign
            pts.append(p)
        return np.array(pts)
    if shape == "plane":
        # Точки на плоскости z = 0
        return np.stack([
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            np.zeros(n),
        ], axis=1)
    if shape == "line":
        # Точки на линии x = y = z
        t = rng.uniform(-1, 1, n)
        return np.stack([t, t, t], axis=1)
    if shape == "cluster":
        # Кластер вокруг центра (1, 1, 1)
        return rng.normal(0, 0.15, (n, 3)) + np.array([1.0, 1.0, 1.0])
    raise ValueError(f"unknown shape: {shape}")


class PointJEPA:
    """Point-JEPA: предиктор следующего окна облака (Widrow-Hoff).

    Окно (w точек) → z_ctx. Цель: латент следующего окна.
    W: dim×dim, err = z_{i+1} − W·z_ctx.

    Вход: последовательность облаков P_0, P_1, ..., P_T (сцены во времени).
    Каждый P_t — форма из набора; предиктор учится предсказывать
    переход P_t → P_{t+1}.
    """

    def __init__(self, dim: int = 512, lr: float = 0.05):
        self.encoder = PointCloudEncoder(dim=dim)
        self.dim = dim
        self.lr = lr
        self.w = np.zeros((dim, dim), dtype=np.float32)
        self.losses: list[float] = []

    def train(self, clouds: list[np.ndarray], epochs: int = 30) -> float:
        """Обучение: переход облако_t → облако_{t+1}.

        Args:
            clouds: последовательность облаков точек (каждый N×3)
            epochs: число эпох

        Returns: средний loss
        """
        latents = [self.encoder.encode(c) for c in clouds]
        for _ in range(epochs):
            for i in range(len(latents) - 1):
                x = latents[i]
                target = latents[i + 1]
                pred = self.w @ x
                err = target - pred
                # Oja-правило: Δw = lr·err⊗x − lr·||pred||²·w
                # (самоподавление держит W ограниченным В КАЖДОМ шаге)
                oja = np.dot(pred, pred)
                self.w += self.lr * (np.outer(err, x) - oja * self.w)
                self.losses.append(float(np.linalg.norm(err)))
            # Страховка: мягкая норм ||W|| ≈ 30
            norm = np.linalg.norm(self.w)
            if norm > 1e-9 and norm > 30.0:
                self.w *= 30.0 / norm
        return float(np.mean(self.losses[-100:])) if self.losses else 0.0

    def predict_next(self, cloud: np.ndarray) -> np.ndarray:
        """Предсказание следующего облака (латент)."""
        return self.w @ self.encoder.encode(cloud)

    def predict_next_rust(self, rust: object, lat: np.ndarray) -> np.ndarray:
        """Предсказание через Rust-ядро (fuga_core.point_jepa_predict).

        Принимает УЖЕ закодированный латент (dim,).
        """
        return np.asarray(rust.point_jepa_predict(self.w, lat))

    def cos_pred(self, cloud: np.ndarray, actual_next: np.ndarray) -> float:
        """Косинус между предсказанием и фактическим следующим облаком."""
        pred = self.predict_next(cloud)
        actual = self.encoder.encode(actual_next)
        return self.encoder.cos(pred, actual)


def demo():
    print("═" * 60)
    print("POINT-JEPA: облака точек → латент, предиктор (Widrow-Hoff)")
    print("═" * 60)

    rng = np.random.default_rng(42)
    shapes = ["sphere", "cube", "plane", "line", "cluster"]
    clouds = [make_shape(s, n=64, rng=rng) for s in shapes]

    # 1. Разделимость: косинус между формами
    print("\n[1] РАЗДЕЛИМОСТЬ ФОРМ (косинусы латентов):")
    enc = PointCloudEncoder(dim=512)
    lats = [enc.encode(c) for c in clouds]
    print("        " + "  ".join(f"{s[:4]:>6}" for s in shapes))
    for i, s in enumerate(shapes):
        row = f"  {s[:4]:>6}"
        for j in range(len(shapes)):
            c = enc.cos(lats[i], lats[j])
            row += f"  {c:6.3f}"
        print(row)

    # 2. Терпкость к шуму
    print("\n[2] УСТОЙЧИВОСТЬ К ШУМУ (сфера + шум):")
    sphere = clouds[0]
    for noise in [0.0, 0.05, 0.15, 0.3]:
        noisy = sphere + rng.normal(0, noise, sphere.shape)
        c = enc.cos(enc.encode(sphere), enc.encode(noisy))
        print(f"    шум σ={noise:.2f}: cos={c:.4f}")

    # 3. Обучение предиктора: сцена_t → сцена_{t+1}
    print("\n[3] ОБУЧЕНИЕ ПРЕДИКТОРА (переходы между формами):")
    # Строим последовательность: sphere→cube→plane→line→sphere→...
    seq = [clouds[i % len(shapes)] for i in range(20)]
    pj = PointJEPA(dim=512, lr=0.05)
    loss = pj.train(seq, epochs=40)
    print(f"    средний loss (последние 100): {loss:.4f}")

    # 4. Предсказание: sphere → предсказание ≈ cube?
    print("\n[4] ПРЕДСКАЗАНИЕ СЛЕДУЮЩЕГО ОБЛАКА:")
    for i in range(min(4, len(seq) - 1)):
        c = pj.cos_pred(seq[i], seq[i + 1])
        print(f"    {shapes[i % len(shapes)]:8s} → {shapes[(i+1) % len(shapes)]:8s}: "
              f"cos(предсказание, факт) = {c:.4f}")

    print("\n=== POINT-JEPA OK ===")


if __name__ == "__main__":
    demo()