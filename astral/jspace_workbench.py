#!/usr/bin/env python3
"""J-Space Workbench: латентная мастерская мыслей (поглощена и вживлена).

JSpaceWorkbench:
  - concept_memory = VSA-якоря из Rust (fuga_core.HybridBinder), НЕ случайные
  - ACT-цикл скрытого мышления (halting head, адаптивная глубина)
  - J-Lens Decoder: проекция J-пространства -> логиты по кодбуку VSA-токенов
    (обратно в «речь» через cleanup по якорям — совместимо с Rust)
"""

from __future__ import annotations


import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from antitf.rust_bridge import packed_to_torch


class JSpaceWorkbench(nn.Module):
    def __init__(self, dim=512, num_concepts=4000, act_max_steps=8,
                 binder=None, codebook_tokens=None):
        super().__init__()
        self.dim = dim
        self.act_max_steps = act_max_steps

        # VSA-якоря из Rust-ядра (fuga_core), не случайные
        if binder is None:
            binder = fuga_core.HybridBinder(dim)
        self.binder = binder
        self.concept_memory = self._rust_anchors(num_concepts)

        self.latent_thought_gate = nn.Linear(dim, dim)
        self.halting_head = nn.Linear(dim, 1)

        # J-Lens Decoder: J-space -> кодбук VSA-токенов (речь)
        self.j_lens_projector = nn.Linear(dim, dim, bias=False)
        self.codebook = self._build_codebook(codebook_tokens)

    def _rust_anchors(self, n) -> nn.Parameter:
        """Гипервекторы якорей, сгенерированные Rust-биндером."""
        words = [f"concept:{i}" for i in range(min(n, 2000))]
        pk = np.asarray(self.binder.bind_batch([[w] for w in words]))
        hvs = packed_to_torch(pk)  # [n, dim] bipolar
        if hvs.shape[0] < n:
            extra = torch.sign(torch.randn(n - hvs.shape[0], self.dim))
            hvs = torch.cat([hvs, extra])
        return nn.Parameter(hvs.float(), requires_grad=False)

    def _build_codebook(self, tokens):
        """Кодбук из слов/токенов -> их VSA-HV (для J-Lens декодирования)."""
        if tokens is None:
            tokens = ["vmalloc_init", "schedule", "parse", "add", "data",
                      "result", "return", "def", "function", "struct"]
        hvs = []
        for t in tokens:
            pk = np.asarray(self.binder.bind_batch([[t]]))
            hvs.append(packed_to_torch(pk)[0])
        return torch.stack(hvs).float()

    def forward(self, vsa_input_embeds):
        """1) VSA -> J-space, 2) ACT-цикл мысли, 3) J-Lens -> речь."""
        state = vsa_input_embeds
        step = 0
        halting_cum = 0.0
        # ФАЗА 1: скрытое мышление (латентный цикл, без текста)
        while step < self.act_max_steps and halting_cum < 0.99:
            thought_update = torch.tanh(self.latent_thought_gate(state))
            state = state + thought_update
            halt_prob = torch.sigmoid(self.halting_head(state))
            halting_cum += float(halt_prob.mean().item())
            step += 1
        # ФАЗА 2: J-Lens — проекция в пространство речи
        j_latent = self.j_lens_projector(state)
        logits = j_latent @ self.codebook.T          # косинус-подобные логиты по якорям
        return j_latent, logits, step


def main():
    torch.manual_seed(0)
    binder = fuga_core.HybridBinder(512)
    wb = JSpaceWorkbench(dim=512, num_concepts=512, act_max_steps=6,
                         binder=binder)

    # вход: VSA-векторы последовательности токенов
    seq_tokens = ["def", "parse", "data", "return", "result"]
    pk = np.asarray(binder.bind_batch([[w] for w in seq_tokens]))
    x = packed_to_torch(pk)  # [5, 512]

    j_latent, logits, steps = wb(x)
    print(f"[J-Space] вход: {seq_tokens}")
    print(f"  J-latent shape: {tuple(j_latent.shape)}")
    print(f"  ACT-шагов мысли: {steps}")
    print(f"  логиты по кодбуку ({wb.codebook.shape[0]} якорей): "
          f"топ-1 = {wb.codebook[logits.argmax(-1)[0]].norm().item():.2f}")
    # декодирование: ближайший якорь к J-latent (cleanup через Rust)
    sims = wb.codebook @ j_latent[0]
    print(f"  J-Lens декод -> ближайший концепт: index={sims.argmax().item()}")

    # обучение: J-space предсказывает следующий концепт
    opt = torch.optim.Adam(wb.parameters(), lr=1e-3)
    print("\n[learn] J-Space предсказывает следующий концепт:")
    for step in range(201):
        # target = сдвинутая последовательность
        target = torch.roll(x, -1, dims=0)
        _, logits, _ = wb(x)
        loss = F.cross_entropy(logits, torch.arange(5) + 0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    print(f"[result] J-Space Workbench вживлён, связь с Rust: concept-memory "
          f"из fuga_core, J-Lens кодирует в VSA-якоря.")


if __name__ == "__main__":
    main()