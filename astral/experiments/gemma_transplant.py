"""GemmaTransplant — РЕАЛЬНАЯ трансплантация трансформерной модели в VSA.

Работает с любой HuggingFace-моделью: TinyStories (3.7M, CPU) → Gemma 4 26B
(GPU 26GB+). Поток:

  1. Загружаем модель (AutoModel) + токенизатор
  2. EXTRACT: hidden states для каждого токена на корпусе
  3. ENCODE: hidden state → VSA-гипервектор (случайная проекция)
  4. TRANSPLANT: Widrow-Hoff (0 градиентов) учит динамику донора
  5. VERIFY: реципиент vs донор на невиденном тексте (косинус)

Принцип: НЕ копируем веса (26B = 52GB fp16), а трансплантируем
ПОВЕДЕНИЕ — траектории hidden states → VSA-стек (лёгкий, событийный).

Для Gemma 4 26B: нужен GPU с 26GB+ VRAM ИЛИ дамп hidden states
офлайн (один прогон модели на корпусе, сохранить .npz, затем
трансплантация без донора).
"""

from __future__ import annotations

import os

import numpy as np

from astral.experiments.brain_transplant import BrainTransplantProtocol


class HFDomainExtractor:
    """Экстрактор hidden states из HuggingFace-модели (реальный донор).

    model_name: 'roneneldan/TinyStories' (CPU, 3.7M) — демо
                'google/gemma-2-2b' (4GB) — средний
                'google/gemma-4-26b' (26B, GPU 26GB+) — целевой
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device
        self.hidden_dim = self.model.config.hidden_size
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def forward(self, text: str) -> list[np.ndarray]:
        """Токены текста → hidden states [n_tokens, hidden_dim]."""
        import torch
        enc = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        # последний hidden state: [1, n_tokens, hidden_dim]
        hs = out.last_hidden_state[0].cpu().numpy()
        return [hs[t] for t in range(hs.shape[0])]

    def corpus_states(self, corpus: list[str]) -> list[list[np.ndarray]]:
        return [self.forward(t) for t in corpus]


def real_transplant(model_name: str, corpus: list[str], seeds: list[str],
                    hv_dim: int = 512, epochs: int = 20) -> dict:
    """Полная реальная трансплантация: донор → VSA-реципиент."""
    print(f"\n=== РЕАЛЬНАЯ ТРАНСПЛАНТАЦИЯ: {model_name} ===")

    # 1. Донор
    print(f"1. Загрузка донора: {model_name}...")
    donor = HFDomainExtractor(model_name, device="cpu")
    print(f"   параметров: {donor.n_params/1e6:.1f}M, hidden={donor.hidden_dim}")

    # 2. EXTRACT
    print("\n2. EXTRACT — hidden states на корпусе:")
    trajs = donor.corpus_states(corpus)
    n_states = sum(len(t) for t in trajs)
    print(f"   траекторий: {len(trajs)}, hidden states: {n_states}, "
          f"dim={donor.hidden_dim}")

    # 3+4. Протокол (ENCODE + TRANSPLANT)
    print("\n3-4. ENCODE + TRANSPLANT (0 градиентов):")
    proto = BrainTransplantProtocol(hv_dim=hv_dim,
                                    donor_dim=donor.hidden_dim, seed=1)
    result = proto.transplant(trajs, epochs=epochs)
    print(f"   шагов: {result['steps']}")
    print(f"   cos обучения: {result['cos_mean']:.4f} (средний), "
          f"{result['cos_last']:.4f} (последние 200)")

    # 5. VERIFY
    print("\n5. VERIFY — реципиент vs донор на НЕвиденных текстах:")
    for seed in seeds:
        v = proto.verify(donor.forward, seed)
        print(f"   '{seed[:35]}...': cos={v['cos_mean']:.4f} "
              f"({v['n_donor_states']} состояний донора)")

    print("\n=== ТРАНСПЛАНТАЦИЯ ЗАВЕРШЕНА ===")
    return {
        "model": model_name,
        "n_params": donor.n_params,
        "hidden_dim": donor.hidden_dim,
        "train_cos": result["cos_mean"],
        "verify": {s: proto.verify(donor.forward, s)["cos_mean"] for s in seeds},
    }


def demo():
    # TinyStories — маленькая, работает на CPU (реальный трансформер)
    corpus = [
        "the sun is shining brightly in the sky today",
        "once upon a time there was a little bear",
        "she walked through the forest to find berries",
        "the cat sat on the mat and watched the birds",
    ]
    seeds = [
        "the moon rose over the quiet village",
        "a young fox explored the autumn woods",
    ]
    real_transplant("roneneldan/TinyStories", corpus, seeds,
                    hv_dim=512, epochs=20)


if __name__ == "__main__":
    demo()
