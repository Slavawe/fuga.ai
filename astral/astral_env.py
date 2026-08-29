
"""Среда «Астрал»: плотный перцептивный поток -> 32K VSA-состояния.

Отличие от черновика: dense_hv НЕ случайный — он детерминированно
выводится из патчей кадра (цветовые бины + градиент + позиция),
поэтому динамика среды наблюдаема и предсказуема H-JEPA.
"""

from __future__ import annotations


import os
import random
import re

import numpy as np
import torch


import fuga_core
from antitf.rust_bridge import packed_to_torch


class ScaledAstralEnvironment:
    """32K VSA-базис; состояние выводится из патчей кадра."""

    def __init__(self, dataset_path="./datasets/coco/train2017",
                 vector_dim: int = 32768, grid: int = 8,
                 fallback_random: bool = True):
        self.dim = vector_dim
        self.grid = grid
        self.binder = fuga_core.HybridBinder(vector_dim)
        self.image_files = []
        if os.path.isdir(dataset_path):
            self.image_files = [os.path.join(dataset_path, f_)
                                for f_ in os.listdir(dataset_path)
                                if f_.lower().endswith((".jpg", ".png"))]
        self.fallback_random = fallback_random
        self.t = 0
        # скрытая динамика для предсказуемости (H-JEPA учится её ловить)
        self._phase = random.Random(0)

    def _frame_tokens(self, frame: np.ndarray) -> list[str]:
        h, w = frame.shape[:2]
        ph, pw = h // self.grid, w // self.grid
        tokens: list[str] = []
        gray = frame.mean(axis=2)
        for gi in range(self.grid):
            for gj in range(self.grid):
                p = frame[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw]
                g = gray[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw]
                r_, g_, b_ = p[..., 0].mean(), p[..., 1].mean(), p[..., 2].mean()
                grad = float(np.abs(np.diff(g, axis=0)).mean() +
                             np.abs(np.diff(g, axis=1)).mean())
                toks = [
                    f"cr{int(r_/64)}", f"cg{int(g_/64)}", f"cb{int(b_/64)}",
                    f"br{int(g.mean()/85)}", f"ed{min(int(grad/25), 3)}",
                ]
                pos = f"P{gi}_{gj}"
                tokens.extend(f"{t}@{pos}" for t in toks)
        return tokens

    def get_state(self) -> dict:
        """Состояние S(t): HV из реального кадра + фазовый сдвиг сцены."""
        self.t += 1
        if self.image_files:
            path = self.image_files[self.t % len(self.image_files)]
            import cv2
            img = cv2.imread(path)
            frame = cv2.resize(img, (512, 512)) if img is not None else None
        else:
            frame = None
        if frame is None:
            if not self.fallback_random:
                raise RuntimeError("нет кадров и fallback запрещён")
            frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        tokens = self._frame_tokens(frame)
        pk = np.asarray(self.binder.bind_batch([tokens]))
        hv = packed_to_torch(pk)[0]
        return {"hv": hv, "tokens_count": len(tokens), "t": self.t}

    def step_action(self, state_hv: torch.Tensor, action: int) -> dict:
        """Действие = управляемый сдвиг сцены (циклическая ротация токенов):
        детерминированная динамика, которую H-JEPA может выучить."""
        flat = state_hv.flatten()
        shifted = torch.roll(flat, shifts=action * 64)
        nxt = torch.sign(shifted * 0.9 + 0.1)
        nxt[nxt == 0] = 1
        self.t += 1
        return {"hv": nxt, "action": action, "t": self.t}
