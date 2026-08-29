
"""Form-Aware Surface Decoder: регистр управляет ПОВЕРХНОСТЬЮ речи.

Авторегрессия по байтам ответа; условие [HV контекста ⊗ регистр].
Метрика эффекта: доля «вы/ваш»-форм в генерациях при formal-теге
против «ты/тво»-форм при casual-теге на held-out контекстах.
"""

from __future__ import annotations

from __future__ import annotations

import json
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


random.seed(0)
torch.manual_seed(0)

import fuga_core
from antitf.rust_bridge import packed_to_torch

REG = {"formal": 0, "casual": 1, "neutral": 2}
CTX_BYTES = 96
MAXLEN = 110
EOS = 0


def load_data(limit=12000):
    data = []
    with open("dataset_vault/04_pragmatic/pragmatic_triads.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["lang"] != "ru":
                continue
            resp = d["response"].strip()
            b = list(resp.encode("utf-8")[:CTX_BYTES]) + [EOS]
            ctx_w = [w.lower() for w in
                     re.findall(r"[a-zа-яё]+", d["context"].lower())][:16]
            if len(ctx_w) < 2 or len(b) < 4:
                continue
            data.append((ctx_w, b, REG[d["register"]]))
            if len(data) >= limit:
                break
    random.shuffle(data)
    return data


class SurfaceDecoder(nn.Module):
    def __init__(self, dim=2048, hidden=384, use_register=True):
        super().__init__()
        self.use_register = use_register
        self.byte_emb = nn.Embedding(256, 32)
        reg_in = 16 if use_register else 0
        self.reg_emb = nn.Embedding(3, 16) if use_register else None
        self.ctx_proj = nn.Linear(dim, 64)
        self.gru = nn.GRUCell(32 + 64 + reg_in, hidden)
        self.head = nn.Linear(hidden, 256)

    def condition(self, hv, reg_id=None):
        """hv: [D] или [B,D]; reg_id: скаляр или [B]. Возвращает [B, 64+16]."""
        if hv.dim() == 1:
            hv = hv.unsqueeze(0)
        c = self.ctx_proj(hv)
        parts = [c]
        if self.use_register:
            b = c.shape[0]
            if reg_id is None:
                r = torch.full((b,), 2, dtype=torch.long, device=hv.device)
            elif isinstance(reg_id, int):
                r = torch.full((b,), reg_id, dtype=torch.long,
                               device=hv.device)
            else:
                r = reg_id.to(torch.long).reshape(-1)
                if r.shape[0] == 1 and b > 1:
                    r = r.expand(b)
            parts.append(self.reg_emb(r))
        return torch.cat(parts, dim=-1)

    def init_state(self, cond):
        w = self.gru.hidden_size if hasattr(self.gru, "hidden_size") else 384
        h = cond[:, :w] if cond.shape[1] >= w else \
            F.pad(cond, (0, w - cond.shape[1]))
        return h * 0.5

    def step(self, byte_idx, h, cond):
        if byte_idx.dim() == 0:
            byte_idx = byte_idx.view(1)
        x = torch.cat([self.byte_emb(byte_idx), cond], dim=-1)
        h = self.gru(x, h)
        return self.head(h), h


FORMAL_MARKS = ("вы", "ваш", "ваше", "вами")
CASUAL_MARKS = ("ты", "тво", "тебя", "тебе")


@torch.no_grad()
def generate(model, hv, reg_id, temp=0.7, max_len=MAXLEN):
    cond = model.condition(hv, reg_id)
    h = model.init_state(cond)
    ctx = torch.zeros(1, dtype=torch.long)
    out = bytearray()
    for _ in range(max_len):
        logits, h = model.step(ctx, h, cond)
        probs = F.softmax(logits[0] / temp, dim=-1).numpy()
        nb = int(np.random.choice(256, p=probs))
        if nb == EOS:
            break
        out.append(nb)
        ctx = torch.tensor([nb])
    return out.decode("utf-8", "ignore").lower()


def main():
    binder = fuga_core.HybridBinder(2048)
    data = load_data()
    cut = int(len(data) * 0.85)
    tr, te = data[:cut], data[cut:]
    print(f"RU triads: train={len(tr)} heldout={len(te)}")

    def enc_ctx(ctx_w):
        pk = np.asarray(binder.bind_batch([ctx_w]))
        return packed_to_torch(pk)[0]

    print("encoding contexts ...")
    tr_hv = torch.stack([enc_ctx(c) for c, _, _ in tr])
    te_hv = torch.stack([enc_ctx(c) for c, _, _ in te])

    for use_reg in (False, True):
        tag = "+reg" if use_reg else "-reg"
        model = SurfaceDecoder(use_register=use_reg)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        B = 64
        n = len(tr)
        t0 = time.time()

        def batch_regs(idx_list):
            return torch.tensor([tr[i][2] for i in idx_list])

        for step in range(1201):
            idx = torch.randint(0, n, (B,))
            regs = batch_regs(idx.tolist())
            cond = model.condition(tr_hv[idx],
                                   regs if use_reg else torch.full((B,), 2))
            h = model.init_state(cond)
            seqs = [tr[i][1] for i in idx.tolist()]
            maxL = min(max(len(s_) for s_ in seqs), MAXLEN)
            ctx_b = torch.zeros(B, dtype=torch.long)
            ce_sum = 0.0
            n_tok = 0
            for t in range(maxL):
                logits, h = model.step(ctx_b, h, cond)
                tgt = torch.tensor(
                    [seqs_i[t] if t < len(seqs_i) else EOS for seqs_i in seqs])
                mask = torch.tensor([t < len(seqs_i) for seqs_i in seqs])
                if mask.any():
                    ce_sum += F.cross_entropy(logits[mask], tgt[mask],
                                              reduction="sum")
                    n_tok += int(mask.sum())
                ctx_b = tgt.clamp(min=0)
            loss = ce_sum / max(n_tok, 1)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 400 == 0 or step == 1200:
                print(f"[{tag}] step {step}: ce/byte={loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)")

        # ===== генерация под разными регистрами на held-out =====
        f_cnt = c_cnt = gen_n = 0
        f_hit = c_hit = 0
        formal_hits_gold = casual_hits_gold = 0
        with torch.no_grad():
            for i in range(min(len(te), 120)):
                ctx_w, gold_bytes, gold_reg = te[i]
                outs = {}
                for reg_name, reg_id in (("formal", REG["formal"]),
                                         ("casual", REG["casual"])):
                    outs[reg_name] = generate(model, te_hv[i], reg_id)
                gold_text = bytes(gold_bytes).decode("utf-8", "ignore").lower()
                # золото: есть ли в исходном ответе вежливые формы
                gold_formal = any(m in gold_text for m in FORMAL_MARKS)
                gold_casual = any(m in gold_text for m in CASUAL_MARKS)
                if not (gold_formal or gold_casual):
                    continue
                gen_n += 1
                f_has = any(m in outs["formal"] for m in FORMAL_MARKS)
                c_has = any(m in outs["casual"] for m in CASUAL_MARKS)
                if gold_formal:
                    f_hit += int(f_has); formal_hits_gold += 1
                if gold_casual:
                    c_hit += int(c_has); casual_hits_gold += 1
                f_cnt += int(f_has); c_cnt += int(c_has)
        print(f"[{tag}] GEN: вы-form@formal-gen={f_cnt}/{gen_n}, "
              f"ты-form@casual-gen={c_cnt}/{gen_n} | "
              f"gold-aligned: formal {f_hit}/{formal_hits_gold}, "
              f"casual {c_hit}/{casual_hits_gold}")


if __name__ == "__main__":
    main()
