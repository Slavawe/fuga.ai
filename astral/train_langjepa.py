#!/usr/bin/env python3
"""Обучить lang-jepa адаптер на реальном корпусе dataset_vault.

Фаза 1: концепт-предиктор (EMA-таргет, smooth-L1).
Фаза 2: концепт-декодер (frozen encoder, GRU → кодбук).
Сохраняет веса в /tmp/langjepa_vault.pt для последующей
сериализации в FUGA1 (tag=8 CONCEPT_W).
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/slava/Anti-Tronsformers")

from astral.models.lang_jepa_adapter import LangJEPAAdapter


def load_vault_texts(path: str, limit: int) -> list[str]:
    """Читает vault_corpus.jsonl: каждая строка — текст."""
    texts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if len(line) >= 8:  # отбрасываем короткий мусор
                texts.append(line)
    return texts


def main():
    corpus = "/tmp/vault_corpus.jsonl"
    limit = int(os.environ.get("LJ_LIMIT", "5000"))
    steps_p = int(os.environ.get("LJ_STEPS_P", "400"))
    steps_d = int(os.environ.get("LJ_STEPS_D", "400"))
    dim = int(os.environ.get("LJ_DIM", "512"))

    t0 = time.time()
    texts = load_vault_texts(corpus, limit)
    print(f"[langjepa] корпус: {len(texts)} текстов (limit={limit}) за {time.time()-t0:.1f}s")

    adapter = LangJEPAAdapter(dim=dim)

    t0 = time.time()
    r1 = adapter.train_predictor(texts, steps=steps_p)
    print(f"[langjepa] Фаза 1 (предиктор, {steps_p} шагов): "
          f"loss {r1['mean_loss']:.4f} → {r1['final_loss']:.4f} за {time.time()-t0:.1f}s")

    t0 = time.time()
    r2 = adapter.train_decoder(texts, steps=steps_d)
    print(f"[langjepa] Фаза 2 (декодер, {steps_d} шагов): "
          f"loss {r2['mean_loss']:.4f} → {r2['final_loss']:.4f} за {time.time()-t0:.1f}s")

    # Генерация из концептов сидов корпуса
    print("\n[langjepa] генерация из концептов:")
    for seed in texts[:3]:
        gen = adapter.generate(seed, max_len=8)
        print(f"  сид: {seed[:50]}...")
        print(f"  gen: {gen}")

    # Сохраняем state_dict для сериализации в FUGA1
    out = "/tmp/langjepa_vault.pt"
    torch_state = {
        "dim": dim,
        "predictor": adapter.predictor.state_dict(),
        "decoder": adapter.decoder.state_dict() if adapter.decoder else None,
        "codebook": adapter._codebook_cache.cpu() if adapter._codebook_cache is not None else None,
    }
    import torch

    torch.save(torch_state, out)
    print(f"\n[langjepa] веса сохранены: {out} "
          f"({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
