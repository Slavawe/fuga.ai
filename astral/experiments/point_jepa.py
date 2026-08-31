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


# ═══════════════════════════════════════════════════════════════
# V2: ФИКСЫ ПО ЗАКАЗУ
#   1. RFF Bank — QR-ортогональные частоты × 3 масштаба
#   2. Фазовая бинаризация — N-поливалентный Phase HV
#   3. Ortho-Oja — анти-коллапс в point_jepa_train
# ═══════════════════════════════════════════════════════════════

class PointCloudEncoderV2:
    """RFF Bank + Occupancy Grid энкодер.

    Фикс 1 (Multiscale Orthogonal Frequency Sampling):
      W_Ω = QR(randn(D×3)) — ортогональные частоты × 3 масштаба
      (low=объём, mid=изгибы, high=края/углы).

    Фикс 2 (Phase Normalization & Complex Binarization):
      Вместо sign(real(sum(exp(iθ)))) — гибрид:
        • OCCUPANCY GRID (главный разделитель): точки → бункеры 3D-сетки.
          Сфера занимает ОБЪЁМ, плоскость — слой z≈0 → разные бункеры.
          cos(сфера, плоскость) ≤ 0.20 при grid ≥ 7.
        • RFF-фазы (дополнение): QR-ортогональные частоты × 3 масштаба,
          one-hot по фазовым бункерам — сохраняет дельта-фазы.
      Компоненты конкатенируются → единый N-поливалентный Phase HV.
    """

    # Масштабы частот: low=объём, mid=изгибы, high=края/углы
    SCALES = {"low": 0.5, "mid": 2.0, "high": 8.0}

    def __init__(self, dim: int = 256, n_valent: int = 8, grid: int = 7,
                 seed: int = 7):
        self.dim = dim
        self.n_valent = n_valent  # поливалентность (фазовые бункеры)
        self.grid = grid          # разрешение occupancy-сетки
        self.grid_dim = grid ** 3
        self.phase_dim = dim * len(self.SCALES) * n_valent
        self.raw_dim = self.grid_dim + self.phase_dim
        rng = np.random.default_rng(seed)

        # RFF Bank: QR-ортогональные частоты для КАЖДОГО масштаба
        self.omega_bank: dict[str, dict[str, np.ndarray]] = {}
        for scale_name, scale in self.SCALES.items():
            # Случайная нормальная матрица D×3 → QR (ортогональные строки)
            mat = rng.normal(0, 1, (dim, 3))
            q, _ = np.linalg.qr(mat)
            self.omega_bank[scale_name] = {
                "x": scale * q[:, 0],
                "y": scale * q[:, 1],
                "z": scale * q[:, 2],
            }

    def _phase_angle(self, x: float, y: float, z: float,
                     omega: dict[str, np.ndarray]) -> np.ndarray:
        """Фазовый угол точки для заданных частот."""
        return x * omega["x"] + y * omega["y"] + z * omega["z"]

    def _occupancy(self, points: np.ndarray) -> np.ndarray:
        """Occupancy grid: какие ячейки 3D-сетки заняты точками."""
        p = (np.clip(points, -1, 1) + 1) / 2  # [0,1]
        idx = (p * self.grid).astype(int) % self.grid
        occ = np.zeros((self.grid,) * 3, dtype=np.float32)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        return occ.reshape(-1)

    def _phase_onehot(self, points: np.ndarray) -> np.ndarray:
        """RFF-фазы: 3 масштаба × dim компонент → N-поливалентный one-hot."""
        onehot_blocks = []
        for scale_name in self.SCALES:
            om = self.omega_bank[scale_name]
            # Сумма exp(iθ) по точкам — комплексный вектор (dim,)
            re = np.zeros(self.dim)
            im = np.zeros(self.dim)
            for p in points:
                ang = self._phase_angle(p[0], p[1], p[2], om)
                re += np.cos(ang)
                im += np.sin(ang)
            # Угол → N-поливалентный бункер
            angle = np.arctan2(im, re)  # [-π, π]
            bins = ((angle + np.pi) / (2 * np.pi) * self.n_valent).astype(int) % self.n_valent
            # one-hot: каждый бункер → N-мерный вектор (1 в активном)
            block = np.zeros((self.dim, self.n_valent), dtype=np.float32)
            block[np.arange(self.dim), bins] = 1.0
            onehot_blocks.append(block.reshape(-1))
        return np.concatenate(onehot_blocks)

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Облако → N-поливалентный HV (сетка + фазы, нормированный).

        Сетка — главный признак (вес 1.0), фазы — малая добавка
        (вес 0.05): сохраняют разделимость форм и добавляют фазу.
        """
        occ = self._occupancy(points)
        ph = self._phase_onehot(points)
        lat = np.concatenate([1.0 * occ, 0.05 * ph])
        n = np.linalg.norm(lat)
        if n > 1e-9:
            lat = lat / n
        return lat

    def cos(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class PointJEPAOrtho:
    """Point-JEPA с Ortho-Oja анти-коллапсом + RFF-редукцией.

    Фикс 3: в point_jepa_train добавляется штраф за коллинеарность
    весов W — Gram-Schmidt ортогонализация строк W каждые k шагов,
    пресекающая коллапс латентов к одной главной координате.

    RFF-редукция: raw HV (сетка+фазы) → случайная проекция → work_dim,
    чтобы W (work_dim×work_dim) был умеренным по размеру.
    """

    def __init__(self, dim: int = 256, n_valent: int = 8, grid: int = 7,
                 work_dim: int = 512, lr: float = 0.05,
                 ortho_every: int = 5, ortho_scale: float = 1.0,
                 seed: int = 7):
        self.encoder = PointCloudEncoderV2(dim=dim, n_valent=n_valent,
                                           grid=grid, seed=seed)
        self.raw_dim = self.encoder.raw_dim
        self.work_dim = work_dim
        self.dim = work_dim
        self.lr = lr
        self.ortho_every = ortho_every
        self.ortho_scale = ortho_scale
        # Случайная RFF-проекция: raw_dim → work_dim (фиксированная)
        rng = np.random.default_rng(seed + 1)
        self.proj = rng.normal(0, 1.0 / np.sqrt(self.raw_dim),
                               (work_dim, self.raw_dim)).astype(np.float32)
        self.w = np.zeros((work_dim, work_dim), dtype=np.float32)
        self.losses: list[float] = []

    def _project(self, lat: np.ndarray) -> np.ndarray:
        """raw HV → work_dim (RFF-редукция, нормировка)."""
        p = self.proj @ lat
        n = np.linalg.norm(p)
        return (p / n if n > 1e-9 else p).astype(np.float32)

    def encode(self, cloud: np.ndarray) -> np.ndarray:
        """Облако → work_dim латент (для предиктора)."""
        return self._project(self.encoder.encode(cloud))

    def _orthogonalize_rows(self) -> None:
        """Gram-Schmidt ортогонализация строк W (анти-коллапс)."""
        rows = self.w.copy()
        ortho = np.zeros_like(rows)
        for i in range(len(rows)):
            v = rows[i].copy()
            for j in range(i):
                v -= np.dot(ortho[j], v) * ortho[j]
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                ortho[i] = v / nv
        self.w = (self.ortho_scale * ortho).astype(np.float32)

    def train(self, clouds: list[np.ndarray], epochs: int = 30) -> float:
        """Обучение: переход облако_t → облако_{t+1} + Ortho-Oja."""
        latents = [self.encode(c) for c in clouds]
        for epoch in range(epochs):
            for i in range(len(latents) - 1):
                x = latents[i]
                target = latents[i + 1]
                pred = self.w @ x
                err = target - pred
                # Widrow-Hoff + Oja (анти-переполнение: предсказание нормировано)
                oja = np.dot(pred, pred)
                self.w += self.lr * (np.outer(err, x) - oja * self.w)
                self.losses.append(float(np.linalg.norm(err)))
            # Ortho-Oja: периодическая ортогонализация (анти-коллапс)
            if (epoch + 1) % self.ortho_every == 0:
                self._orthogonalize_rows()
        return float(np.mean(self.losses[-100:])) if self.losses else 0.0

    def predict_next(self, cloud: np.ndarray) -> np.ndarray:
        return self.w @ self.encode(cloud)

    def cos_pred(self, cloud: np.ndarray, actual_next: np.ndarray) -> float:
        pred = self.predict_next(cloud)
        actual = self.encode(actual_next)
        n1 = np.linalg.norm(pred)
        n2 = np.linalg.norm(actual)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        return float(np.dot(pred, actual) / (n1 * n2))


def demo_v2():
    """Демо фиксов: разделимость сфера/плоскость + предсказание."""
    print("═" * 60)
    print("POINT-JEPA V2: RFF Bank + Occupancy Grid + Ortho-Oja")
    print("═" * 60)

    rng = np.random.default_rng(42)
    shapes = ["sphere", "cube", "plane", "line", "cluster"]
    clouds = [make_shape(s, n=64, rng=rng) for s in shapes]

    # 1. Разделимость форм (V2 vs V1)
    print("\n[1] РАЗДЕЛИМОСТЬ ФОРМ (V2, grid=7, N-valent=8):")
    enc_v1 = PointCloudEncoder(dim=512)
    enc_v2 = PointCloudEncoderV2(dim=256, n_valent=8, grid=7)
    lats_v1 = [enc_v1.encode(c) for c in clouds]
    lats_v2 = [enc_v2.encode(c) for c in clouds]

    # Ключевая пара: сфера vs плоскость
    c_v1_sp = enc_v1.cos(lats_v1[0], lats_v1[2])
    c_v2_sp = enc_v2.cos(lats_v2[0], lats_v2[2])
    print(f"    сфера vs плоскость: V1 cos={c_v1_sp:.3f} → V2 cos={c_v2_sp:.3f} "
          f"({'✓ ≤ 0.20' if c_v2_sp <= 0.20 else '✗ > 0.20'})")

    # Полная матрица V2
    print("        " + "  ".join(f"{s[:4]:>6}" for s in shapes))
    for i, s in enumerate(shapes):
        row = f"  {s[:4]:>6}"
        for j in range(len(shapes)):
            row += f"  {enc_v2.cos(lats_v2[i], lats_v2[j]):6.3f}"
        print(row)

    # 2. Предсказание переходов (V2 + Ortho-Oja)
    print("\n[2] ПРЕДСКАЗАНИЕ ПЕРЕХОДОВ (V2 + Ortho-Oja, по ближайшей форме):")
    seq = [clouds[i % len(shapes)] for i in range(20)]
    pj2 = PointJEPAOrtho(dim=256, n_valent=8, grid=7, work_dim=512,
                         lr=0.05, ortho_every=5, ortho_scale=1.0)
    loss = pj2.train(seq, epochs=40)
    print(f"    loss: {loss:.4f}")
    # Латенты форм как «память» для классификации
    shape_lats = {s: pj2.encode(c) for s, c in zip(shapes, clouds)}
    correct = 0
    for i in range(min(8, len(seq) - 1)):
        pred = pj2.predict_next(seq[i])
        # классификация: ближайший латент формы
        best_name, best_c = None, -1.0
        for s, sl in shape_lats.items():
            c = float(np.dot(pred, sl) / (
                np.linalg.norm(pred) * np.linalg.norm(sl) + 1e-9))
            if c > best_c:
                best_name, best_c = s, c
        actual = shapes[(i + 1) % len(shapes)]
        ok = best_name == actual
        correct += ok
        print(f"    {shapes[i % len(shapes)]:8s} → {actual:8s} "
              f"(предсказано: {best_name:8s}, cos={best_c:.3f}) {'✓' if ok else '✗'}")
    print(f"    правильных: {correct}/{min(8, len(seq) - 1)}")

    # 3. Шум (V2)
    print("\n[3] УСТОЙЧИВОСТЬ К ШУМУ (V2):")
    sphere = clouds[0]
    for noise in [0.0, 0.05, 0.15, 0.3]:
        noisy = sphere + rng.normal(0, noise, sphere.shape)
        c = enc_v2.cos(enc_v2.encode(sphere), enc_v2.encode(noisy))
        print(f"    шум σ={noise:.2f}: cos={c:.4f}")

    print("\n=== POINT-JEPA V2 OK ===")


if __name__ == "__main__":
    demo()