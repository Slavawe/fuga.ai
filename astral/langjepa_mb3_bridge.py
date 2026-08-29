#!/usr/bin/env python3
"""LJ-2: связка lang-jepa концепт-канала с декодером MB3.

Добавляет ЧЕТВЁРТЫЙ приор к патчевому коридору MB3:
  score = cos(W_patch·x_patch, lat)          # патчевый канал
        + beta_macro·cos(W_macro·x_ctx, lat) # макро-канал (есть в MB3)
        + beta_concept·cos(concept, lat)     # КОНЦЕПТ lang-jepa (новое)

Concept = предсказанный СЛЕДУЮЩИЙ концепт из контекста (EMA-таргет
lang-jepa). Веса читаются из FUGA1 (tag=8 CONCEPT_W) — единый файл.

Использование:
  python astral/langjepa_mb3_bridge.py <model.fuga> [--seed "text"]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astral.fuga1_concept import read_concept_section, flat_to_torch_state
from astral.fuga_tokenizer import FugaTokenizer
from fuga_core import HybridBinder


class ConceptChannel:
    """Концепт-приор для MB3: контекст → следующий концепт → score-бонус.

    Порт ConceptPredictor (attention + LayerNorm) на Python, веса из
    FUGA1 tag=8. Использует FugaTokenizer как концепт-энкодер.
    """

    def __init__(self, concept_flat: bytes, dim: int = 512):
        self.dim = dim
        state = flat_to_torch_state(concept_flat)
        self.query = state["query"].float()               # [1,1,dim]
        self.in_proj = state["context_attention.in_proj_weight"].float()
        self.in_bias = state["context_attention.in_proj_bias"].float()
        self.out_proj = state["context_attention.out_proj.weight"].float()
        self.out_bias = state["context_attention.out_proj.bias"].float()
        self.ln_weight = state["projection.weight"].float()
        self.ln_bias = state["projection.bias"].float()
        self.binder = HybridBinder(dim)
        self.tok = FugaTokenizer(self.binder)

    @torch.no_grad()
    def predict_next_concept(self, context_text: str) -> torch.Tensor:
        """Контекст (строка) → предсказанный концепт следующего [dim]."""
        hvs = self.tok.encode(context_text.encode("utf-8"))
        if not hvs:
            return torch.zeros(self.dim)
        ctx = torch.stack(hvs[-4:]).unsqueeze(0)  # [1, L, dim]
        L = ctx.size(1)
        # MultiheadAttention (1 голова, learnable query)
        q = self.query.expand(1, -1, -1)  # [1,1,dim]
        # in_proj: qkv [3*dim, dim] → [1, L+1, dim]
        qkv = F.linear(torch.cat([ctx, q], dim=1), self.in_proj, self.in_bias)
        q, k, v = qkv.split(self.dim, dim=-1)
        # ВАЖНО: query — ТОЛЬКО последняя строка (learnable), а не все L+1.
        # Контекстные строки в q не должны attend на себя.
        q = q[:, -1:, :]  # [1,1,dim] — предсказатель следующего концепта
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1)
        out = attn @ v  # [1, 1, dim]
        out = F.linear(out, self.out_proj, self.out_bias)
        # LayerNorm проекция (pred_dim == input_dim → LayerNorm only)
        mean = out.mean(-1, keepdim=True)
        var = out.var(-1, keepdim=True, unbiased=False)
        out = (out - mean) / torch.sqrt(var + 1e-5)
        out = out * self.ln_weight + self.ln_bias
        return F.normalize(out.squeeze(1)[0], dim=-1)


def main():
    ap = argparse.ArgumentParser(description="lang-jepa → MB3 bridge")
    ap.add_argument("fuga", help="путь к .fuga с секцией CONCEPT_W (tag=8)")
    ap.add_argument("--seed", default="the force of gravity is",
                    help="контекст для предсказания концепта")
    args = ap.parse_args()

    blob = read_concept_section(args.fuga)
    if blob is None:
        print(f"✗ CONCEPT_W (tag=8) не найден в {args.fuga}")
        print("  Сначала: python astral/train_langjepa.py + fuga1_concept.py")
        sys.exit(1)

    channel = ConceptChannel(blob)
    concept = channel.predict_next_concept(args.seed)
    print(f"✓ CONCEPT_W загружен из {args.fuga} (dim={channel.dim})")
    print(f"  сид: {args.seed[:60]}...")
    print(f"  концепт: норма={float(concept.norm()):.3f}")

    # Демо согласованности: сравнить концепт с HV якорей токенизатора
    anchors = list(channel.tok.anchors.items())[:200]
    if anchors:
        hvs = torch.stack([hv.float() for _, hv in anchors])
        sims = concept @ F.normalize(hvs, dim=-1).T
        top_vals, top_idx = torch.topk(sims, 5)
        print("\n  ближайшие якоря к концепту:")
        for i, idx in enumerate(top_idx.tolist()):
            name = anchors[idx][0]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            print(f"    {i+1}. {name!r}  cos={top_vals[i].item():.3f}")

    print("\n  MB3 score = patch + βm·macro + βc·concept  (4-й приор активен)")


if __name__ == "__main__":
    main()
