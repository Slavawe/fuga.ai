
"""Astral Data Filter: отбор сэмплов по новизне (Surprise Metric).

Модель обучается ТОЛЬКО на том, что ещё не предсказывает — экономия
GPU-циклов на рутине. Surprise = 1 - cos(HV_pred, HV_real).

Для биполярных HV косинус вырождается в dot/dim, поэтому считаем
через относительную L2-ошибку предсказания (эквивалент по смыслу,
устойчивее к масштабу).
"""

from __future__ import annotations

from __future__ import annotations

import numpy as np
import torch


class AstralDataStreamFilter:
    """Адаптивный режим: порог = EMA(surprise) * margin. Статичный порог
    бесполезен, когда шкала ошибки меняется по ходу обучения (проверено
    A/B 24.08: при fixed=0.35 pass_rate=100% — всё "удивительно")."""

    def __init__(self, novelty_threshold: float = 0.35, adaptive: bool = True,
                 margin: float = 0.25, ema_beta: float = 0.05):
        self.threshold = novelty_threshold
        self.adaptive = adaptive
        self.margin = margin
        self.beta = ema_beta
        self._ema: float | None = None
        self.stats = {"seen": 0, "passed": 0}

    def surprise(self, predicted_hv, target_hv) -> float:
        p = predicted_hv.flatten().detach().float()
        t = target_hv.flatten().float()
        rel_err = float((p - t).norm() / (t.norm() + 1e-9))
        return min(rel_err, 1.0)

    def should_ingest(self, predicted_hv, target_hv) -> tuple[bool, float]:
        s = self.surprise(predicted_hv, target_hv)
        if self.adaptive:
            thr = self.threshold if self._ema is None else                   min(self._ema * (1 + self.margin), 1.0)
            if self._ema is None:
                self._ema = s
            else:
                self._ema += self.beta * (s - self._ema)
        else:
            thr = self.threshold
        self.stats["seen"] += 1
        ok = s >= thr
        if ok:
            self.stats["passed"] += 1
        return ok, s

    def pass_rate(self) -> float:
        return self.stats["passed"] / max(self.stats["seen"], 1)
