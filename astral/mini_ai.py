#!/usr/bin/env python3
"""Mini-AI: собственная модель ИИ, обученная на FugaTokenizer.

Вход — НЕ int-индексы и НЕ nn.Embedding: готовые биполярные Phase Crystal
гипервекторы из VSA-памяти (2048-d). Мини-GRU предсказывает следующий
токен-гипервектор в VSA-пространстве; декодирование — cleanup по якорям.
"""

from __future__ import annotations


import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from astral.fuga_tokenizer import FugaTokenizer


class HVGRU(nn.Module):
    """GRU над гипервекторами: вход HV(2048) -> скрытое -> прогноз HV(2048)."""

    def __init__(self, dim=2048, hidden=256):
        super().__init__()
        self.dim = dim
        self.gru = nn.GRUCell(dim, hidden)
        self.proj = nn.Linear(hidden, dim)

    def forward(self, hv, h=None):
        if h is None:
            h = torch.zeros(hv.shape[0], self.gru.hidden_size)
        h = self.gru(hv, h)
        return torch.tanh(self.proj(h)), h


def load_corpus(path: str, limit: int = 600) -> list[str]:
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = d.get("response") or d.get("ru") or d.get("en") or ""
                if len(t) > 10:
                    out.append(t)
                if len(out) >= limit:
                    break
    if not out:
        out = ["def f(x): return x*2", "the cat sleeps quietly",
               "данные обрабатываются быстро", "hello world 你好"]
    return out


def main():
    torch.manual_seed(0)
    binder = fuga_core.HybridBinder(2048)
    tok = FugaTokenizer(binder)
    print(f"[tokenizer] якорей: {len(tok.anchors)}")

    corpus = load_corpus("dataset_vault/04_pragmatic/pragmatic_triads.jsonl")
    # кодируем все предложения в последовательности HV
    seqs = [tok.encode(t.encode("utf-8", "ignore")) for t in corpus]
    seqs = [s for s in seqs if len(s) >= 2]
    # макс длина для батча
    max_len = min(max(len(s) for s in seqs), 24)
    print(f"[corpus] предложений: {len(seqs)}, max_len={max_len}")

    # кодбук якорей для cleanup (декодирование прогноза -> токен)
    anchor_items = list(tok.anchors.items())
    codebook = torch.stack([hv for _, hv in anchor_items[:4000]])  # [K, 2048]
    print(f"[codebook] {codebook.shape[0]} якорей")

    model = HVGRU(dim=2048, hidden=256)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cut = int(len(seqs) * 0.85)
    tr, te = seqs[:cut], seqs[cut:]

    # обучение: teacher forcing, предсказание следующего HV
    t0 = time.time()
    for step in range(500):
        idx = np.random.randint(0, len(tr), 8)
        lens = [len(tr[i]) for i in idx]
        B = 8
        # батчим по 3 случайных позиции
        loss_acc = 0.0
        n = 0
        h = torch.zeros(B, 256)
        ctx = torch.zeros(B, 2048)
        for tpos in range(max_len):
            cur = torch.stack([tr[idx[b]][tpos] if tpos < len(tr[idx[b]])
                               else torch.zeros(2048) for b in range(B)])
            pred, h = model(cur, h)
            nxt = torch.stack([tr[idx[b]][tpos + 1] if tpos + 1 < len(tr[idx[b]])
                               else torch.zeros(2048) for b in range(B)])
            mask = torch.tensor([tpos + 1 < len(tr[idx[b]]) for b in range(B)])
            if mask.any():
                loss_acc += (1 - F.cosine_similarity(
                    pred[mask], nxt[mask], dim=-1)).sum()
                n += int(mask.sum())
            # teacher forcing: следующее состояние = следующий HV
        loss = loss_acc / max(n, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

    # оценка held-out: cosine предсказанного следующего HV vs истинный
    @torch.no_grad()
    def eval_cos(seqs_):
        cos_true, cos_rand = [], []
        for s in seqs_:
            h = torch.zeros(1, 256)
            for i in range(min(len(s) - 1, max_len - 1)):
                pred, h = model(s[i].unsqueeze(0), h)
                cos_true.append(F.cosine_similarity(pred, s[i + 1].unsqueeze(0),
                                                    dim=-1).item())
                rnd = torch.sign(torch.randn(1, 2048))
                cos_rand.append(F.cosine_similarity(pred, rnd, dim=-1).item())
        return np.mean(cos_true), np.mean(cos_rand)

    ct, cr = eval_cos(te)
    print(f"\n[held-out] cosine(pred, true_next) = {ct:.4f}")
    print(f"           cosine(pred, random)     = {cr:.4f}")
    print(f"           сигнал над шансом:       {ct - cr:+.4f}")

    # токен-точность: декодирование прогноза -> ближайший якорь
    @torch.no_grad()
    def token_acc(seqs_):
        hits = tot = 0
        for s in seqs_[:100]:
            h = torch.zeros(1, 256)
            for i in range(min(len(s) - 1, max_len - 1)):
                pred, h = model(s[i].unsqueeze(0), h)
                best = int((codebook @ pred[0]).argmax())
                _, true_tok = anchor_items[best] if best < len(anchor_items) else (b"", None)
                # истинный токен: ищем в кодбуке по соответствию
                true_hv = s[i + 1]
                true_idx = int((codebook @ true_hv).argmax())
                hits += int(best == true_idx)
                tot += 1
        return hits / max(tot, 1)

    print(f"[token-accuracy] топ-1 токен в кодбуке: {token_acc(te):.3f}")


if __name__ == "__main__":
    main()
