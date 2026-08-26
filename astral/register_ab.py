
"""A/B: добавляет ли регистр-кондиционирование сигнал при предсказании
мешка слов ответа по HV контекста (прагматические триады OASST).
"""

from __future__ import annotations

from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

random.seed(0)
torch.manual_seed(0)

import fuga_core
from antitf.rust_bridge import packed_to_torch
from astral.cliche_filter import detect_cliche

REG = {"formal": 0, "casual": 1, "neutral": 2}
binder = fuga_core.HybridBinder(2048)


def load_data(limit=12000):
    data = []
    with open("dataset_vault/04_pragmatic/pragmatic_triads.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ctx_w = [w.lower() for w in
                     re.findall(r"[a-zа-яё]+", d["context"].lower())][:16]
            rsp = set(w.lower() for w in
                      re.findall(r"[a-zа-яё]+", d["response"].lower()))
            if len(ctx_w) < 3 or len(rsp) < 2:
                continue
            data.append((d["lang"], REG[d["register"]], ctx_w, rsp))
            if len(data) >= limit:
                break
    random.shuffle(data)
    return data


class CondModel(nn.Module):
    def __init__(self, use_register: bool, dim=2048, n_reg=3, hidden=512):
        super().__init__()
        in_dim = dim + (n_reg if use_register else 0)
        self.use_register = use_register
        self.reg_emb = nn.Embedding(n_reg, n_reg) if use_register else None
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 4096)

    def forward(self, hv, reg=None):
        if self.use_register:
            onehot = F.one_hot(reg, 3).float()
            x = torch.cat([hv, self.reg_emb(onehot @ torch.eye(3))], dim=-1) \
                if False else torch.cat([hv, onehot.float()], dim=-1)
        else:
            x = hv
        return self.head(self.net(x))


def main():
    data = load_data()
    cut = int(len(data) * 0.85)
    tr, te = data[:cut], data[cut:]

    from collections import Counter
    cnt = Counter()
    for _, _, _, rsp in tr:
        cnt.update(rsp)
    vocab = [w for w, _ in cnt.most_common(4096)]
    w2i = {w: i for i, w in enumerate(vocab)}
    print(f"train={len(tr)} heldout={len(te)} vocab={len(vocab)}")

    def make(rows):
        words_rows = [rsp for (_, _, _, rsp) in rows]  # для gold-метрик
        hvs = []
        for lang, reg, ctx_w, _rsp in rows:
            pk = np.asarray(binder.bind_batch([ctx_w]))
            hvs.append(packed_to_torch(pk)[0])
        hv = torch.stack(hvs)
        regs = torch.tensor([r for _, r, _, _ in rows])
        tgt = torch.zeros(len(rows), len(vocab))
        for i, (_, _, _, rsp) in enumerate(rows):
            for w in rsp:
                j = w2i.get(w)
                if j:
                    tgt[i, j] = 1.0
        return hv, regs, tgt, words_rows

    tr_hv, tr_r, tr_t, _ = make(tr)
    te_hv, te_r, te_t, te_gold = make(te)

    results = {}
    for use_reg in (False, True):
        tag = "+reg" if use_reg else "-reg"
        model = CondModel(use_register=use_reg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        B = 96
        for step in range(601):
            idx = torch.randint(0, len(tr_hv), (B,))
            reg_idx = tr_r[idx] if use_reg else None
            logits = model(tr_hv[idx], reg_idx) if use_reg \
                else model(tr_hv[idx])
            loss = F.binary_cross_entropy_with_logits(logits, tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 600:
                with torch.no_grad():
                    logits = model(te_hv, te_r if use_reg else None) \
                        if use_reg else model(te_hv)
                    probs = torch.sigmoid(logits)
                    top10 = probs.topk(10, dim=1).indices
                    hits = 0.0
                    for i in range(len(te)):
                        gold_ids = {w2i[w] for w in te_gold[i] if w in w2i}
                        pred_ids = set(top10[i].tolist())
                        hits += len(pred_ids & gold_ids) / max(len(gold_ids), 1)
                    p10 = hits / len(te)
                results[tag] = {"bce": float(loss), "word_hit@10": round(p10, 4)}
                print(f"[{tag}] bce={loss.item():.4f} word_hit@10={p10:.4f}")

    d = (results["+reg"]["word_hit@10"] -
         results["-reg"]["word_hit@10"])
    print(f"\nDELTA от регистр-кондиционирования: {d:+.4f}")


if __name__ == "__main__":
    main()
