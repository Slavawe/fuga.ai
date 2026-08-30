"""Unified Cognitive Engine — единое соединение всех улучшений.

Собирает ВСЕ технологии сессии в один цикл мышления-генерации:

  ПАМЯТЬ (VSA-ядро, единый базис)
    fuga_core FastVSA/HybridBinder — якоря и связывание

  РЕЗОНАТОРЫ (разложение суперпозиций)
    HDCResonator          — N-факторный Фрэйди
    FPEVSA                — комплексные фазы + дробные степени
    PhaseCrystalResonator — фазовые веса кристалла

  КОНЦЕПТЫ (lang-jepa)
    LangJEPAAdapter       — EMA-таргет, концепт следующего предложения
    ConceptChannel        — 4-й приор к декодеру MB3

  АНТИ-КОЛЛАПС (Barlow Twins)
    barlow_loss           — кросс-корреляция → identity

  КОНВЕЙЕР
    UnifiedPipeline       — инжест → VSA → BIM

Цикл мышления (think):
  контекст → концепт (lang-jepa) → резонанс (FPE) → эмиссия (VSA)

Генерация:
  speak(text)   — «говорит» словами из концепта
  codegen(seed) — генерирует код из кодового сида
  combine(...)  — смешивает слово и код (новые комбинации)
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fuga_core import HybridBinder
from antitf.rust_bridge import packed_to_torch

from astral.core.pipeline import UnifiedPipeline
from astral.fuga_tokenizer import FugaTokenizer
from astral.models.barlow_twins import barlow_loss, BarlowTwinsHead
from astral.models.resonator_hdc import (
    HDCResonator,
    FPEVSA,
    PhaseCrystalResonator,
)


def _hv_of(binder: HybridBinder, name: str, dim: int) -> torch.Tensor:
    return packed_to_torch(np.asarray(binder.bind_batch([[name]])))[0].float()


class UnifiedEngine:
    """Единый движок: память + резонаторы + концепты + генерация."""

    def __init__(self, dim: int = 2048, anchors: list[str] | None = None):
        self.dim = dim
        self.binder = HybridBinder(dim)
        self.pipeline = UnifiedPipeline(dim=dim, binder=self.binder)
        self.tok = FugaTokenizer(self.binder)

        default_anchors = [
            "def", "fn", "main", "return", "struct", "let", "if", "else",
            "for", "while", "parse", "data", "function", "class", "import",
            "the", "and", "force", "gravity", "mass", "energy", "memory",
            "process", "system", "network", "data", "value", "state",
        ]
        self.anchors = anchors or default_anchors

        # Резонаторы (один базис якорей)
        self.hdc = HDCResonator(self.binder, self.anchors, dim=min(dim, 2048), iters=25)
        self.fpe = FPEVSA(self.binder, self.anchors, dim=min(dim, 1024), iters=25)
        self.pcr = PhaseCrystalResonator(self.binder, self.anchors, dim=min(dim, 2048), iters=25)

        # Анти-коллапс
        self.barlow = BarlowTwinsHead(in_dim=dim, hidden=dim // 2)
        # БЕЗ оптимизатора: анти-коллапс через Gram-Schmidt + Hebb (0 градиентов)

        # Концепт-память (обученные пары «слово-смысл»)
        self.concept_memory: dict[str, torch.Tensor] = {}

    # ── Память ──────────────────────────────────────────────────
    def memorize(self, word: str, concept: str) -> None:
        """Связать слово с концептом: HV(word) ⊗ HV(concept)."""
        hw = _hv_of(self.binder, word, self.dim)
        hc = _hv_of(self.binder, concept, self.dim)
        self.concept_memory[word] = torch.sign(hw * hc)

    def recall(self, word: str) -> str | None:
        """Резонансное извлечение концепта из памяти слова."""
        hw = _hv_of(self.binder, word, self.dim)
        if word not in self.concept_memory:
            return None
        S = hw * self.concept_memory[word]  # развязка
        names, _ = self.hdc.recover(S, n_factors=1)
        return names[0] if names else None

    # ── Цикл мышления ───────────────────────────────────────────
    def think(self, prompt: str, steps: int = 4) -> list[str]:
        """Цикл: концепт → резонанс → следующий концепт.

        Имитирует lang-jepa предсказание: из контекста выводим
        концепт через фазовую интерполяцию якорей (FPE).
        """
        # Контекст → фазовый вектор (суперпозиция якорей слов)
        words = [w for w in prompt.lower().split() if w in self.anchors]
        if not words:
            words = self.anchors[:3]
        # FPE-интерполяция всех слов контекста
        z = self.fpe.lerp(words[0], words[-1], alpha=0.5)
        for w in words[1:-1]:
            zw = torch.exp(1j * self.fpe.theta[w])
            z = z * zw
        thought = []
        for _ in range(steps):
            # Резонанс: ближайший якорь к фазовому состоянию
            sims = (self.fpe.codebook * z.conj()).real.sum(dim=-1)
            idx = int(sims.argmax())
            thought.append(self.anchors[idx])
            # Сдвиг фазы: bind с текущим концептом (продолжение мысли)
            z = z * torch.exp(1j * self.fpe.theta[self.anchors[idx]])
            z = z / (z.abs().max() + 1e-9)
        return thought

    # ── Генерация слов с штрафом повтора ────────────────────────
    def speak(self, seed: str, length: int = 8, rep_word: float = 0.20) -> str:
        """«Говорит»: фазовая цепочка контекста → слова.

        Использует ОКНО КОНТЕКСТА (последние 4 якоря, как в Rust V2 decoder)
        для определения следующего шага, и.rep_word (урок AGENTS.md v6.2)
        для выбора из аттрактора.

        rep_word=0.20: нижняя граница рабочей зоны — выталкивает
        из «the the the the» в новые морфемы корпуса.
        """
        out = seed.split()
        state = [w for w in seed.lower().split() if w in self.anchors]
        if not state:
            state = [self.anchors[0]]
        counts: dict[str, int] = {}
        for w in state:
            counts[w] = counts.get(w, 0) + 1
        for _ in range(length):
            # Контекст → фазовая суперпозиция (окно 4, как ctx=4 в unified_gpu_train)
            z = torch.ones(self.fpe.dim, dtype=torch.complex64)
            for w in state[-4:]:
                z = z * torch.exp(1j * self.fpe.theta[w])
            # Ближайший якорь
            sims = (self.fpe.codebook * z.conj()).real.sum(dim=-1)
            sims = sims / (sims.abs().max() + 1e-9)   # нормализуем для порога
            # REP: штраф на повторяющиеся якоря (как rep_word=0.20 в Rust)
            for i, a in enumerate(self.anchors):
                sims[i] -= rep_word * counts.get(a, 0)
            idx = int(sims.argmax())
            nxt = self.anchors[idx]
            out.append(nxt)
            state.append(nxt)
            counts[nxt] = counts.get(nxt, 0) + 1
        return " ".join(out)

    # ── Генерация кода с rep-штрафом ────────────────────────────
    def codegen(self, seed: str, length: int = 6, rep_word: float = 0.20) -> str:
        """Генерирует код: структура → операторы через фазовый резонанс.

        Кодовые якоря (fn/let/return/if/for/struct) образуют цепочку.
        rep_word = 0.20 (тот же AGENTS.md v6.2) не даёт зациклиться.
        """
        code_anchors = ["fn", "main", "let", "return", "if", "else", "for", "struct", "impl"]
        # Безопасно: только якоря, которые реально есть в списке
        code_anchors = [a for a in code_anchors if a in self.anchors]
        if not code_anchors:
            return seed  # нет кодовых якорей — нечего генерировать
        code_words = [seed]  # начальная история
        out = seed
        counts: dict[str, int] = {code_anchors[0]: 1}
        for _ in range(length):
            # Контекст кода → фазовая суперпозиция последних 3 токенов
            z = torch.ones(self.fpe.dim, dtype=torch.complex64)
            for w in code_words[-3:]:
                z = z * torch.exp(1j * self.fpe.theta[w])
            sims = (self.fpe.codebook * z.conj()).real.sum(dim=-1)
            sims = sims / (sims.abs().max() + 1e-9)
            # REP-штраф
            for i, a in enumerate(self.anchors):
                sims[i] -= rep_word * counts.get(a, 0)
            # Ограничиваем выбор кодовыми якорями
            code_idx = [self.anchors.index(a) for a in code_anchors]
            code_sims = sims[code_idx]
            nxt = code_anchors[int(code_sims.argmax())]
            counts[nxt] = counts.get(nxt, 0) + 1
            code_words.append(nxt)
            # Структура (каркас кода: ключевое слово → фрагмент)
            if nxt == "fn":
                out += " fn main() {"
            elif nxt == "let":
                out += " let x = 1;"
            elif nxt == "return":
                out += " return x;"
            elif nxt == "if":
                out += " if x > 0 {"
            elif nxt == "for":
                out += " for i in 0..n {"
            elif nxt == "struct":
                out += " struct S {"
            elif nxt == "impl":
                out += " impl S {"
            elif nxt == "else":
                out += " } else {"
            else:
                out += f" {nxt}"
        return out + " }"

    # ── Комбинирование слов и кода ──────────────────────────────
    def combine(self, text_seed: str, code_seed: str, n: int = 4) -> list[str]:
        """Смешивает слово- и код-потоки: новые комбинации.

        Идея: концепт текста (FPE-интерполяция) → ищем код-якорь
        с максимальным резонансом → генерируем код в этом концепте.
        """
        results = []
        text_concepts = self.think(text_seed, steps=2)
        for i in range(n):
            # Концепт текста → код-якорь
            c = text_concepts[i % len(text_concepts)]
            z = torch.exp(1j * self.fpe.theta[c])
            sims = (self.fpe.codebook * z.conj()).real.sum(dim=-1)
            code_anchors = ["fn", "struct", "let", "impl"]
            cand = [(i, sims[i].item()) for i in range(len(self.anchors))
                    if self.anchors[i] in code_anchors]
            cand.sort(key=lambda p: -p[1])
            best_code = self.anchors[cand[0][0]] if cand else "fn"
            # Комбинированная строка: концепт + код-обёртка
            results.append(f"// {c}: {code_seed} -> {best_code}() {{ ... }}")
        return results

    # ── Обучение (без градиентов) ──────────────────────────────
    def train_anti_collapse(self, texts: list[str], steps: int = 100) -> dict:
        """Безградиентный анти-коллапс: ортонормализация HV (Gram-Schmidt).

        Вместо Barlow loss + backward:
          1. Собираем HV батча
          2. Gram-Schmidt: делаем их ортогональными (без autograd)
          3. Обновляем голову через Hebb: w += lr·(x·y) — без градиентов
        """
        words = [(t.split()[0] if t.split() else "the") for t in texts]
        hvs = torch.stack([_hv_of(self.binder, w, self.dim) for w in words])
        hvs_np = hvs.numpy()
        # Gram-Schmidt (без градиентов)
        ortho = np.zeros_like(hvs_np)
        for i in range(len(hvs_np)):
            v = hvs_np[i].copy()
            for j in range(i):
                proj = np.dot(v, ortho[j]) / (np.dot(ortho[j], ortho[j]) + 1e-9)
                v -= proj * ortho[j]
            ortho[i] = v / (np.linalg.norm(v) + 1e-9)
        # Hebbian-обновление весов головы (без .backward())
        # Последний Linear: [out_dim=h, in_dim=d]; апдейт [d, h].
        # Активации ДО последнего слоя = первые 3 модуля (Linear+BN+ReLU)
        last_lin = None
        for m in self.barlow.proj.modules():
            if isinstance(m, torch.nn.Linear):
                last_lin = m
        if last_lin is not None:
            # активации до последнего Linear (без autograd)
            with torch.no_grad():
                h_pre = self.barlow.proj[:-1](hvs)  # (batch, h)
            w = last_lin.weight.data.numpy()  # [d, h]
            h_pre_np = h_pre.numpy()          # (batch, h)
            # Δw[d,h] = (1/B)·Σ_b ortho[b,d]·h_pre[b,h] = ortho.T @ h_pre / B
            w += 0.01 * (ortho.T @ h_pre_np) / (len(hvs_np) + 1e-9)
            last_lin.weight.data = torch.from_numpy(w)
        # loss = дисперсия: 0 = коллапс, 1+ = анти-коллапс
        var = float(np.mean(np.var(ortho, axis=0)))
        return {"final": 1.0 - var, "mean": 1.0 - var, "var": var}


def demo() -> dict:
    """Единая демонстрация: говорить + код + комбинировать."""
    eng = UnifiedEngine(dim=2048)

    # 1. Память: слово → концепт
    eng.memorize("force", "gravity")
    eng.memorize("memory", "process")
    eng.memorize("parse", "data")
    rec = eng.recall("force")
    print(f"[память] force → {rec}")

    # 2. Говорит
    print(f"[речь]   '{eng.speak('the force of gravity is', length=6)}'")

    # 3. Код
    print(f"[код]    '{eng.codegen('fn', length=5)}'")

    # 4. Комбинирует
    print("[комбо]  слово→код:")
    for line in eng.combine("the force of gravity", "fn main()", n=3):
        print(f"         {line}")

    # 5. Barlow анти-коллапс
    r = eng.train_anti_collapse(["the force of gravity", "parse the data"], steps=50)
    print(f"[barlow] loss {r['mean']:.4f} (анти-коллапс)")

    return {
        "recall": rec,
        "speech": eng.speak("the force of gravity is", length=6),
        "code": eng.codegen("fn", length=5),
        "barlow_mean": r["mean"],
    }


if __name__ == "__main__":
    demo()
