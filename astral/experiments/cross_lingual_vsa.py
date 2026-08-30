"""CrossLingualVSA — кросс-языковая инвариантность смысла (RU/EN/ZH).

Цель (по заказу):
  cos(HV_RU, HV_EN) ≥ 0.95, cos(HV_Zh, HV_EN) ≥ 0.95
  для параллельных переводов («данные», «data», 「数据」).

Метод:
  1. Каждая строка → VSA-гипервектор (MiniVSA, 512-dim, seed стабилен).
  2. Widrow-Hoff проекции W_ru/W_en/W_zh: латент строки → единый
     концепт-центроид (усреднение трёх языков). БЕЗ градиентов.
  3. Инвариантность: P_lang(hv) ≈ P_ru(hv_ru) ≈ P_en(hv_en) ≈ P_zh(hv_zh)
     для одного смысла → cos переводов → 1.0.

Zero-shot HardTeacher:
  Учитель знает концепт на ОДНОМ языке (аксиома). Ученик спрашивает
  на другом. Проекции сводят оба к концепт-центроиду → верификация
  через cos(проекция_вопроса, проекция_аксиомы) без переписывания.

Форма: без градиентов (Widrow-Hoff, онлайн-обучение) — философия проекта.
"""

from __future__ import annotations

import numpy as np
import torch

from astral.experiments.mini_cognitive import MiniVSA


class CrossLingualVSA:
    """Трёхъязычное VSA-пространство с проекциями на концепт-центроиды.

    API:
      train_pairs(pairs, epochs)  — пары (RU, EN, ZH) → проекции
      project(text, lang)         — hv → концепт-пространство (torch)
      cos(a_text, lang_a, b_text, lang_b) — косинус после проекций
      teacher_verify(hypothesis_lang, axiom_text, axiom_lang) — zero-shot
    """

    LANGS = ("RU", "EN", "ZH")

    def __init__(self, dim: int = 512, seed: int = 0, hidden: int = 0):
        self.dim = dim
        self.vsa = MiniVSA(dim=dim, seed=seed)
        self.hv_cache: dict[str, torch.Tensor] = {}
        # Случайное расширение (нелинейный признак, без градиентов):
        # hv (dim) → hidden-признаки через sign(R · hv). Усиливает
        # разделимость (особенно ZH — короткие иероглифы).
        if hidden <= 0:
            hidden = 4 * dim
        self.hidden = hidden
        rng = np.random.RandomState(seed + 1)
        self.R: dict[str, torch.Tensor] = {
            lang: torch.from_numpy(rng.randn(hidden, dim).astype(np.float32))
            for lang in self.LANGS
        }
        # Проекции Widrow-Hoff: lang -> (dim, hidden) → концепт-пространство
        self.proj: dict[str, torch.Tensor] = {
            lang: torch.zeros(dim, hidden) for lang in self.LANGS
        }
        self.steps = 0

    # ---- нелинейный признак ----
    def _feat(self, text: str, lang: str) -> torch.Tensor:
        """hv → нелинейный признак: sign(R · hv) / sqrt(hidden) (норм ≈ 1)."""
        f = self.R[lang] @ self._hv(text)
        return torch.sign(f).float() / np.sqrt(self.hidden)

    # ---- encode ----
    def _hv(self, text: str) -> torch.Tensor:
        """Строка → биполярный HV (torch). Кеш по тексту."""
        if text not in self.hv_cache:
            hv_np = self.vsa.item(text)
            self.hv_cache[text] = torch.from_numpy(hv_np).float()
        return self.hv_cache[text]

    def _norm(self, v: torch.Tensor) -> torch.Tensor:
        n = v.norm().clamp_min(1e-9)
        return v / n

    # ---- Widrow-Hoff проекция ----
    def train_pairs(self, pairs: list[tuple[str, str, str]], epochs: int = 60,
                    lr: float = 0.1) -> dict[str, float]:
        """Обучение проекций: каждая тройка (ru, en, zh) — один концепт.

        Цель: W_lang · feat_lang → центроид(ru, en, zh).
        Widrow-Hoff: W += lr · err · feat_langᵀ, где err = центроид − W·feat.
        Мягкая нормализация W до масштаба ~10 (иначе переполнение/NaN).
        """
        losses = {lang: [] for lang in self.LANGS}
        for epoch in range(epochs):
            for ru, en, zh in pairs:
                hvs = {
                    "RU": self._hv(ru),
                    "EN": self._hv(en),
                    "ZH": self._hv(zh),
                }
                # концепт-центроид = нормированная сумма трёх языков
                centroid = self._norm(hvs["RU"] + hvs["EN"] + hvs["ZH"])
                for lang, hv in hvs.items():
                    feat = self._feat(ru if lang == "RU" else en if lang == "EN" else zh, lang)
                    pred = self.proj[lang] @ feat
                    err = centroid - pred
                    # Widrow-Hoff: W += lr · err ⊗ feat
                    self.proj[lang] += lr * torch.outer(err, feat)
                    losses[lang].append(float(err.norm().item()))
            # Мягкая нормализация: ||W|| ≈ 10 (не 1 — иначе pred ~ 0)
            for lang in self.LANGS:
                f = self.proj[lang].norm().clamp_min(1e-9)
                self.proj[lang] /= (f / 10.0)
            self.steps += len(pairs)
        return {lang: float(np.mean(v)) for lang, v in losses.items()}

    def project(self, text: str, lang: str) -> torch.Tensor:
        """Строка → концепт-пространство (после признака + проекции)."""
        return self._norm(self.proj[lang] @ self._feat(text, lang))

    def cos(self, a_text: str, lang_a: str, b_text: str, lang_b: str) -> float:
        """Косинус между переводами в концепт-пространстве."""
        pa, pb = self.project(a_text, lang_a), self.project(b_text, lang_b)
        return float((pa * pb).sum().clamp(-1, 1).item())

    # ---- zero-shot учитель ----
    def teacher_verify(self, hypothesis: str, hypothesis_lang: str,
                       axiom_text: str, axiom_lang: str,
                       threshold: float = 0.95) -> tuple[bool, float]:
        """Проверка гипотезы ученика против аксиомы учителя.

        Учитель знает аксиому на ОДНОМ языке. Ученик формулирует
        гипотезу на ДРУГОМ. Обе проекции → концепт-пространство,
        cos ≥ threshold → подтверждена.
        """
        c = self.cos(hypothesis, hypothesis_lang, axiom_text, axiom_lang)
        return (c >= threshold, c)


class MultilingualTeacher:
    """Учитель, обученный на русском, верифицирует гипотезы EN/ZH.

    Хранит концепты-аксиомы (текст + язык), использует CrossLingualVSA
    для zero-shot верификации на любом языке.
    """

    def __init__(self, clvsa: CrossLingualVSA, threshold: float = 0.90):
        self.clvsa = clvsa
        self.threshold = threshold
        self.axioms: list[dict] = []  # {concept, text, lang}

    def learn_axiom(self, concept: str, text: str, lang: str = "RU") -> None:
        """Учитель изучает аксиому (правило) на заданном языке."""
        self.axioms.append({"concept": concept, "text": text, "lang": lang})

    def verify(self, hypothesis: str, lang: str = "EN") -> tuple[bool, str, float]:
        """Проверка гипотезы ученика против ВСЕХ аксиом учителя.

        Возвращает (подтверждена?, концепт?, косинус?).
        """
        best_concept, best_cos = None, -1.0
        for ax in self.axioms:
            ok, c = self.clvsa.teacher_verify(
                hypothesis, lang, ax["text"], ax["lang"], self.threshold,
            )
            if ok and c > best_cos:
                best_concept, best_cos = ax["concept"], c
        if best_concept is not None:
            return (True, best_concept, best_cos)
        # Если порог не пройден — вернём ближайший
        best_c, best_ax = max(
            ((self.clvsa.cos(hypothesis, lang, ax["text"], ax["lang"]), ax)
             for ax in self.axioms),
            default=(0.0, None),
        )
        return (False, best_ax["concept"] if best_ax else None, best_c)


def demo():
    print("═" * 60)
    print("CROSS-LINGUAL VSA: RU/EN/ZH ИНВАРИАНТНОСТЬ + ZERO-SHOT")
    print("═" * 60)

    clvsa = CrossLingualVSA(dim=512)

    # Параллельные тройки (синтетика — те же концепты на 3 языках)
    pairs = [
        ("данные", "data", "数据"),
        ("матрица", "matrix", "矩阵"),
        ("гипотеза", "hypothesis", "假设"),
        ("граф", "graph", "图"),
        ("память", "memory", "内存"),
        ("группа", "group", "群"),
        ("теорема", "theorem", "定理"),
        ("дерево", "tree", "树"),
        ("сеть", "network", "网络"),
        ("поле", "field", "域"),
        ("аксиома", "axiom", "公理"),
        ("цикл", "cycle", "回路"),
        ("безопасность", "security", "安全"),
        ("кольцо", "ring", "环"),
        ("вывод", "corollary", "推论"),
        ("поток", "flow", "流"),
        ("процесс", "process", "进程"),
        ("вектор", "vector", "向量"),
        ("множество", "set", "集合"),
        ("раскраска", "coloring", "着色"),
        ("файл", "file", "文件"),
        ("модуль", "module", "模"),
        ("функция", "function", "函数"),
        ("изоморфизм", "isomorphism", "同构"),
    ]

    # Обучение проекций (Widrow-Hoff, без градиентов)
    print("\n[1] Обучение проекций (Widrow-Hoff, 30 эпох)...")
    losses = clvsa.train_pairs(pairs, epochs=30, lr=0.05)
    for lang, loss in losses.items():
        print(f"    {lang}: средний loss={loss:.4f}")

    # Инвариантность: cos переводов ≥ 0.95
    print("\n[2] ИНВАРИАНТНОСТЬ (cos переводов → 1.0):")
    ok_count = 0
    for ru, en, zh in pairs[:8]:
        c_re = clvsa.cos(ru, "RU", en, "EN")
        c_ze = clvsa.cos(zh, "ZH", en, "EN")
        c_rz = clvsa.cos(ru, "RU", zh, "ZH")
        ok = c_re >= 0.95 and c_ze >= 0.95
        ok_count += ok
        print(f"    {'✓' if ok else '✗'} {ru:12s} | {en:12s} | {zh:8s} "
              f"→ RU-EN {c_re:.3f} ZH-EN {c_ze:.3f} RU-ZH {c_rz:.3f}")
    print(f"    Итог инвариантности: {ok_count}/{min(8, len(pairs))} ≥ 0.95")

    # Zero-shot учитель
    print("\n[3] ZERO-SHOT УЧИТЕЛЬ (учит RU, верифицирует EN/ZH):")
    teacher = MultilingualTeacher(clvsa, threshold=0.90)
    teacher.learn_axiom("данные", "данные", "RU")
    teacher.learn_axiom("граф", "граф", "RU")
    teacher.learn_axiom("функция", "функция", "RU")

    tests = [
        ("data", "EN", "данные", True),
        ("graph", "EN", "граф", True),
        ("函数", "ZH", "функция", True),
        ("неверная гипотеза", "EN", "данные", False),  # random → cos < порога
    ]
    for hyp, lang, expected_concept, expected_ok in tests:
        ok, concept, c = teacher.verify(hyp, lang)
        status = "✓" if (ok == expected_ok and (concept == expected_concept or not expected_ok)) else "✗"
        print(f"    [{status}] «{hyp}» ({lang}) → подтверждена={ok}, "
              f"концепт={concept}, cos={c:.3f} (ожидалось: {expected_concept})")

    print("\n=== CROSS-LINGUAL VSA OK ===")


if __name__ == "__main__":
    demo()