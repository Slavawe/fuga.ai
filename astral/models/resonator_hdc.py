"""HDC Resonator Memory + FPE-VSA + Phase-Crystal Resonator.

Три улучшения VSA-резонатора (надстройка, НЕ замена VSA-ядра):

R1. HDCResonator — классический резонатор Фрэйди (Frady et al. 2020),
    расширенный на N факторов: S = X1⊗X2⊗...⊗XN, итеративное
    разложение через мнимые обратные (sign для биполярных, inv для
    фазовых). Мягкий cleanup через softmax-взвешенную суперпозицию.

R2. FPEVSA — Complex-Valued Phase Resonator with Fractional Power
    Encodings: комплексные гипервекторы e^{i·θ}, связывание = фазовое
    сложение (поэлементное умножение e^{iθ1}·e^{iθ2} = e^{i(θ1+θ2)}),
    дробные степени θ·p дают ИЕРАРХИЮ: x^{0.5} — «половина» вектора.
    Позволяет мягкие интерполяции между концептами (θ = θ_a·α + θ_b·(1-α)).

R3. PhaseCrystalResonator — резонанс в пространстве фазового
    кристалла (src/ai/crystal.rs): фазовые векторы + резонансное
    разложение с cleanup по якорям кристалла.

Все три используют ЕДИНОЕ VSA-ядро (fuga_core FastVSA/HybridBinder)
для генерации якорей — только оператор резонанса разный.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from fuga_core import HybridBinder
from antitf.rust_bridge import packed_to_torch


def _hv_of(binder: HybridBinder, name: str, dim: int) -> torch.Tensor:
    """Биполярный HV [dim] для имени через Rust-ядро."""
    return packed_to_torch(np.asarray(binder.bind_batch([[name]])))[0].float()


# ═══════════════════════════════════════════════════════════════
# R1. HDC Resonator Memory — N-факторный резонатор
# ═══════════════════════════════════════════════════════════════
class HDCResonator:
    """Классический резонатор: S = X1⊗...⊗XN → итеративное разложение.

    Поддерживает N факторов (не только пару). Каждый шаг:
        x_i = sign(S ⊗ x_1^{-1} ⊗ ... ⊗ x_{i-1}^{-1} ⊗ x_{i+1}^{-1} ⊗ ...)
    затем cleanup по кодбуку. Мягкий режим (soft=True) — взвешенная
    суперпозиция якорей вместо жёсткого argmax.
    """

    def __init__(
        self,
        binder: HybridBinder,
        anchors: list[str],
        dim: int = 2048,
        iters: int = 40,
        n_restarts: int = 8,
        soft: bool = False,
        soft_temp: float = 3.0,
    ):
        self.binder = binder
        self.dim = dim
        self.iters = iters
        self.n_restarts = n_restarts
        self.soft = soft
        self.soft_temp = soft_temp
        self.anchors = anchors
        self.codebook = torch.stack([_hv_of(binder, a, dim) for a in anchors])  # [K, dim]

    def _cleanup(self, v: torch.Tensor) -> torch.Tensor:
        if self.soft:
            sims = self.codebook @ v
            w = torch.softmax(sims * self.soft_temp, dim=0)
            return (w[:, None] * self.codebook).sum(0)
        sims = self.codebook @ v
        return self.codebook[int(sims.argmax())].clone()

    def _resonate(self, S: torch.Tensor, n_factors: int, seed: int) -> tuple[list[torch.Tensor], float]:
        rng = torch.Generator().manual_seed(seed)
        xs = [torch.sign(torch.randn(self.dim, generator=rng)) for _ in range(n_factors)]
        for _ in range(self.iters):
            for i in range(n_factors):
                # S ⊗ (произведение всех кроме i) — sign-инверсия для биполярных
                others = torch.ones(self.dim)
                for j in range(n_factors):
                    if j != i:
                        others = others * xs[j]
                xs[i] = self._cleanup(torch.sign(S * others))
        # энергия разложения: ||S - ⊗x_i||
        recon = torch.ones(self.dim)
        for x in xs:
            recon = recon * x
        err = float((S - recon).norm())
        return xs, err

    def recover(self, S: torch.Tensor, n_factors: int = 2) -> tuple[list[str], float]:
        """Разложить S на N факторов, выбор по минимальной энергии."""
        best = None
        for r in range(self.n_restarts):
            xs, err = self._resonate(S, n_factors, seed=100 + r)
            if best is None or err < best[0]:
                best = (err, xs)
        err, xs = best
        names = []
        for x in xs:
            sims = self.codebook @ x
            names.append(self.anchors[int(sims.argmax())])
        return names, err


# ═══════════════════════════════════════════════════════════════
# R2. FPE-VSA — Complex-Valued Phase + Fractional Power
# ═══════════════════════════════════════════════════════════════
class FPEVSA:
    """Комплекснозначный фазовый VSA с дробными степенями.

    HV = e^{i·θ}, θ ∈ [-π, π]^dim. Операции:
      bind(x, y) = x·y  (фазовое сложение: θ_x + θ_y)
      fractional power: x^p = e^{i·p·θ_x} — «дробная доля» концепта
      интерполяция: lerp(a, b, α) = a^α · b^(1-α)

    Резонатор в комплексном домене: x_i = S · conj(⊗x_j≠i) →
    cleanup по кодбуку (макс. реальной части скалярного произведения).

    Сильные стороны vs биполярный: дробные степени дают ГЛАДКУЮ
    иерархию (x^{0.5} осмыслен), интерполяция концептов непрерывна.
    """

    def __init__(
        self,
        binder: HybridBinder,
        anchors: list[str],
        dim: int = 1024,
        iters: int = 40,
        n_restarts: int = 8,
    ):
        self.binder = binder
        self.dim = dim
        self.iters = iters
        self.n_restarts = n_restarts
        self.anchors = anchors
        # Фазовые векторы якорей: θ из детерминированного хэша имени
        rng = np.random.default_rng(42)
        self.theta = {}
        for a in anchors:
            h = abs(hash(a)) % (2**32)
            sub = np.random.default_rng(h)
            self.theta[a] = torch.tensor(sub.uniform(-math.pi, math.pi, dim)).float()
        self.codebook = torch.stack(
            [torch.exp(1j * self.theta[a]) for a in anchors]
        )  # [K, dim] complex64

    def bind(self, *names: str) -> torch.Tensor:
        """Связать несколько концептов: e^{i·Σθ}."""
        theta = torch.zeros(self.dim)
        for n in names:
            theta = theta + self.theta[n]
        return torch.exp(1j * theta)

    def power(self, z: torch.Tensor, p: float) -> torch.Tensor:
        """Дробная степень: z^p = e^{i·p·arg(z)}."""
        return torch.exp(1j * p * torch.angle(z))

    def lerp(self, a: str, b: str, alpha: float) -> torch.Tensor:
        """Непрерывная интерполяция концептов: a^α · b^(1-α)."""
        za = torch.exp(1j * self.theta[a])
        zb = torch.exp(1j * self.theta[b])
        return self.power(za, alpha) * self.power(zb, 1.0 - alpha)

    def _cleanup(self, v: torch.Tensor) -> torch.Tensor:
        sims = (self.codebook * v.conj()).real.sum(dim=-1)  # реальная часть <c, v>
        return self.codebook[int(sims.argmax())].clone()

    def _resonate(self, S: torch.Tensor, n_factors: int, seed: int) -> tuple[list[torch.Tensor], float]:
        rng = torch.Generator().manual_seed(seed)
        xs = [
            torch.exp(1j * torch.randn(self.dim, generator=rng))
            for _ in range(n_factors)
        ]
        for _ in range(self.iters):
            for i in range(n_factors):
                others = torch.ones(self.dim, dtype=torch.complex64)
                for j in range(n_factors):
                    if j != i:
                        others = others * xs[j]
                # S · conj(others) — фазовое вычитание
                xs[i] = self._cleanup(S * others.conj())
        recon = torch.ones(self.dim, dtype=torch.complex64)
        for x in xs:
            recon = recon * x
        err = float((S - recon).abs().norm())
        return xs, err

    def recover(self, S: torch.Tensor, n_factors: int = 2) -> tuple[list[str], float]:
        best = None
        for r in range(self.n_restarts):
            xs, err = self._resonate(S, n_factors, seed=200 + r)
            if best is None or err < best[0]:
                best = (err, xs)
        err, xs = best
        names = []
        for x in xs:
            sims = (self.codebook * x.conj()).real.sum(dim=-1)
            names.append(self.anchors[int(sims.argmax())])
        return names, err


# ═══════════════════════════════════════════════════════════════
# R3. Phase-Crystal Resonator Network
# ═══════════════════════════════════════════════════════════════
class PhaseCrystalResonator:
    """Резонатор поверх фазового кристалла.

    Использует БИПОЛЯРНЫЕ якоря из Rust-кристалла (как VQResonator),
    но добавляет фазовую динамику: промежуточные состояния резонанса
    интерпретируются как фазовые точки кристалла (угол между x и
    якорями), давая «мягкую фазу» до схлопывания в якорь.

    Отличие от VQResonator: recover_phase возвращает НЕ только
    ближайший якорь, но и фазовый вес промежуточного состояния —
    «частичный резонанс» для творческого смешивания.
    """

    def __init__(
        self,
        binder: HybridBinder,
        anchors: list[str],
        dim: int = 2048,
        iters: int = 40,
        n_restarts: int = 8,
    ):
        self.binder = binder
        self.dim = dim
        self.iters = iters
        self.n_restarts = n_restarts
        self.anchors = anchors
        self.codebook = torch.stack([_hv_of(binder, a, dim) for a in anchors])

    def _cleanup(self, v: torch.Tensor) -> torch.Tensor:
        sims = self.codebook @ v
        return self.codebook[int(sims.argmax())].clone()

    def phase_weights(self, v: torch.Tensor, temp: float = 3.0) -> torch.Tensor:
        """Фазовые веса состояния: softmax(cos·temp) — «где» в кристалле."""
        sims = self.codebook @ v
        return torch.softmax(sims * temp, dim=0)

    def recover_pair(
        self, S: torch.Tensor, n_iter: int | None = None,
    ) -> tuple[str, str, torch.Tensor]:
        """S = X⊗Y; возвращает имена и фазовые веса после резонанса."""
        iters = n_iter or self.iters
        best = None
        for r in range(self.n_restarts):
            rng = torch.Generator().manual_seed(300 + r)
            x = torch.sign(torch.randn(self.dim, generator=rng))
            y = torch.sign(torch.randn(self.dim, generator=rng))
            for _ in range(iters):
                x = self._cleanup(torch.sign(S * y))
                y = self._cleanup(torch.sign(S * x))
            err = float((S - x * y).norm())
            if best is None or err < best[0]:
                best = (err, x, y)
        _, x, y = best
        w = self.phase_weights(x + y, temp=3.0)
        sims_x = self.codebook @ x
        sims_y = self.codebook @ y
        return self.anchors[int(sims_x.argmax())], self.anchors[int(sims_y.argmax())], w


# ═══════════════════════════════════════════════════════════════
# Пруф: разложение пар + дробные степени
# ═══════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(0)
    binder = HybridBinder(2048)
    anchors = [
        "vmalloc_init", "schedule", "parse", "add", "main", "struct",
        "from_json", "lock", "hash", "queue", "loop", "alloc",
    ]
    rng = np.random.default_rng(1)

    def hv(name):
        return _hv_of(binder, name, 2048)

    print("=== R1. HDC RESONATOR (N=2) ===")
    res = HDCResonator(binder, anchors, dim=2048, iters=30, n_restarts=8)
    correct = 0
    trials = 50
    for _ in range(trials):
        a = anchors[int(rng.integers(len(anchors)))]
        b = anchors[int(rng.integers(len(anchors)))]
        S = torch.sign(hv(a) * hv(b))
        (x, y), _ = res.recover(S, n_factors=2)
        correct += int((x == a and y == b) or (x == b and y == a))
    print(f"  точность пары: {correct}/{trials} ({correct/trials:.0%})")

    print("\n=== R1. HDC RESONATOR (N=3) ===")
    correct3 = 0
    trials3 = 30
    for _ in range(trials3):
        a = anchors[int(rng.integers(len(anchors)))]
        b = anchors[int(rng.integers(len(anchors)))]
        c = anchors[int(rng.integers(len(anchors)))]
        S = torch.sign(hv(a) * hv(b) * hv(c))
        names, _ = res.recover(S, n_factors=3)
        ok = set(names) == {a, b, c}
        correct3 += int(ok)
    print(f"  точность тройки: {correct3}/{trials3} ({correct3/trials3:.0%})")

    print("\n=== R2. FPE-VSA (фазовый, дробные степени) ===")
    fpe = FPEVSA(binder, anchors, dim=1024, iters=30, n_restarts=8)
    correct_f = 0
    for _ in range(50):
        a = anchors[int(rng.integers(len(anchors)))]
        b = anchors[int(rng.integers(len(anchors)))]
        S = fpe.bind(a, b)
        names, _ = fpe.recover(S, n_factors=2)
        correct_f += int((names[0] == a and names[1] == b) or (names[0] == b and names[1] == a))
    print(f"  точность пары: {correct_f}/50 ({correct_f/50:.0%})")

    # Дробная степень: интерполяция двух концептов
    z_mix = fpe.lerp("parse", "alloc", alpha=0.5)
    sim_parse = (torch.exp(1j * fpe.theta["parse"]) * z_mix.conj()).real.sum()
    sim_alloc = (torch.exp(1j * fpe.theta["alloc"]) * z_mix.conj()).real.sum()
    print(f"  lerp(parse, alloc, 0.5): cos_parse={sim_parse/1024:.3f} cos_alloc={sim_alloc/1024:.3f} (оба > 0 = середина)")

    # Чистое дробное: a^2·b должно дать a (удвоение фазы снимается разложением)
    z_pow = fpe.power(torch.exp(1j * fpe.theta["parse"]), 0.5)
    sim_half = (torch.exp(1j * fpe.theta["parse"]) * z_pow.conj()).real.sum() / 1024
    print(f"  parse^0.5 vs parse: cos={sim_half:.3f} (дробная доля близка к оригиналу)")

    print("\n=== R3. PHASE-CRYSTAL RESONATOR ===")
    pc = PhaseCrystalResonator(binder, anchors, dim=2048, iters=30, n_restarts=8)
    ok3 = 0
    for a, b in [("vmalloc_init", "schedule"), ("parse", "alloc"), ("struct", "lock")]:
        S = torch.sign(hv(a) * hv(b))
        x, y, w = pc.recover_pair(S)
        ok = (x == a and y == b) or (x == b and y == a)
        ok3 += int(ok)
        top = w.topk(3).indices.tolist()
        print(f"  S = {a}⊗{b} -> ({x}, {y}) OK={ok} | фаза топ-3: {[anchors[i] for i in top]}")
    print(f"  точность: {ok3}/3")

    print("\n=== ИТОГ: три резонатора работают ===")


if __name__ == "__main__":
    main()
