#!/usr/bin/env python3
"""Честный A/B: NonGradient (HTM+VSA+SNN+NEAT) vs Gradient (Adam+backward).

Протокол (одинаковый для обоих):
  1. Один корпус: 15 fuga_memory_* → пары (subject → object)
  2. Train/test split 80/20 (test = НЕВИДЕННЫЕ пары)
  3. Оба учатся на train, предсказывают на test
  4. Метрики (одинаковые):
     - cos(pred, true) на тесте (обобщение, не запоминание)
     - top-1 accuracy: argmax по кандидатам теста
     - время обучения, число параметров
  5. Вывод: кто честно обобщает на невиденных парах
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/slava/Anti-Tronsformers")

from astral.nongradient_engine import NonGradientEngine


def load_pairs(memory_dirs, limit=20000):
    """Пары (subject → object) из facts.jsonl."""
    pairs = []
    for d in memory_dirs:
        facts = os.path.join(d, "fuga_memory.facts.jsonl")
        if not os.path.exists(facts):
            continue
        with open(facts, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    s, obj = o.get("subject", ""), o.get("object", "")
                    if s and obj:
                        pairs.append((s, obj))
                except Exception:
                    continue
    # дедуп + лимит
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
        if len(uniq) >= limit:
            break
    return uniq


def make_vsa(seed=0):
    from astral.experiments.mini_cognitive import MiniVSA
    return MiniVSA(dim=512, seed=seed)


def main():
    base = "/home/slava/Anti-Tronsformers"
    dirs = [os.path.join(base, f) for f in os.listdir(base)
            if f.startswith("fuga_memory_") and os.path.isdir(os.path.join(base, f))]
    pairs = load_pairs(dirs, limit=20000)
    print(f"пар (уникальных): {len(pairs)}")

    # Train/test split (80/20), детерминированный
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(pairs))
    n_train = int(len(pairs) * 0.8)
    train = [pairs[i] for i in idx[:n_train]]
    test = [pairs[i] for i in idx[n_train:]]
    print(f"train: {len(train)}, test: {len(test)} (НЕВИДЕННЫЕ)")

    # Кандидаты для top-1: все объекты теста
    test_objs = list({obj for _, obj in test})
    print(f"кандидатов в тесте: {len(test_objs)}")

    # ── МЕТРИКА ─────────────────────────────────────────────
    def evaluate(weights, vsa, pairs, candidates):
        cos_vals, hits = [], 0
        for s, obj in pairs:
            hv = vsa.item(s)
            pred = np.sign(hv * np.sign(weights @ hv))
            true_hv = vsa.item(obj)
            cos_vals.append(vsa.cos(pred, true_hv))
            # top-1 среди кандидатов
            best, best_c = candidates[0], -1.0
            for c in candidates:
                c_cos = vsa.cos(pred, vsa.item(c))
                if c_cos > best_c:
                    best_c, best = c_cos, c
            if best == obj:
                hits += 1
        return float(np.mean(cos_vals)), hits / max(1, len(pairs))

    def summarize(name, weights, vsa):
        cos_tr, acc_tr = evaluate(weights, vsa, train, test_objs)
        cos_te, acc_te = evaluate(weights, vsa, test, test_objs)
        print(f"  {name}: train cos={cos_tr:.4f} acc={acc_tr*100:.1f}% | "
              f"test cos={cos_te:.4f} acc={acc_te*100:.1f}%")
        return acc_tr, acc_te

    # ═══════════════════════════════════════════════════════
    # 1. NON-GRADIENT (HTM + VSA + SNN/STDP + NEAT)
    # ═══════════════════════════════════════════════════════
    print("\n=== 1. NON-GRADIENT (HTM+VSA+SNN+NEAT) ===")
    vsa = make_vsa()
    t0 = time.time()
    eng = NonGradientEngine(dim=512, lr=0.05)
    for a, b in train:
        eng.learn(a, b)
    t_ng = time.time() - t0
    acc_ng_tr, acc_ng_te = summarize("NonGrad", eng.weights, vsa)
    print(f"  время: {t_ng:.1f}s, веса: {eng.weights.size} параметров, "
          f"HTM-состояний: {len(eng.sequence_memory)}")

    # ═══════════════════════════════════════════════════════
    # 2. GRADIENT (Adam + loss.backward) — эталон
    # ═══════════════════════════════════════════════════════
    print("\n=== 2. GRADIENT (Adam + backward) — эталон ===")
    import torch
    import torch.nn as nn

    dim = 512
    model = nn.Linear(dim, dim, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # HV для train-пар
    hv_train_a = torch.stack([torch.from_numpy(vsa.item(a)) for a, _ in train]).float()
    hv_train_b = torch.stack([torch.from_numpy(vsa.item(b)) for _, b in train]).float()

    t0 = time.time()
    epochs = 5
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(hv_train_a)
        loss = loss_fn(pred, hv_train_b)
        loss.backward()
        opt.step()
    t_grad = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  время: {t_grad:.1f}s, параметров: {n_params}")

    # evaluate в numpy
    w_grad = model.weight.detach().numpy()
    acc_g_tr, acc_g_te = summarize("Grad", w_grad, vsa)

    # ═══════════════════════════════════════════════════════
    # 3. СРАВНЕНИЕ
    # ═══════════════════════════════════════════════════════
    print("\n=== 3. ЧЕСТНОЕ СРАВНЕНИЕ ===")
    print(f"  {'':18}{'NonGrad':>10}{'Grad':>10}")
    print(f"  {'train acc (запоминание):':18}{acc_ng_tr*100:>9.1f}%{acc_g_tr*100:>9.1f}%")
    print(f"  {'test acc (обобщение):':18}{acc_ng_te*100:>9.1f}%{acc_g_te*100:>9.1f}%")
    print(f"  {'время:':18}{t_ng:>9.1f}s{t_grad:>9.1f}s")
    print(f"  {'параметры:':18}{eng.weights.size:>10}{n_params:>10}")

    if acc_ng_te > acc_g_te + 0.01:
        verdict = "NonGradient обобщает ЛУЧШЕ (и без backward!)"
    elif acc_g_te > acc_ng_te + 0.01:
        verdict = "Gradient обобщает лучше"
    elif acc_ng_tr > acc_g_tr + 0.01:
        verdict = "Ничья на обобщении, но NonGrad ЛУЧШЕ запоминает"
    else:
        verdict = "Полный паритет — NonGrad без градиентов не уступает"
    print(f"\nВЕРДИКТ: {verdict}")


if __name__ == "__main__":
    main()
