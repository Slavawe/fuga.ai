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

sys.path.insert(0, ".")

import fuga_core
from antitf.kan import ChebyKANLayer
from antitf.rust_bridge import packed_to_torch

DIM = 2048
CTX_BYTES = 10          # байтовое окно истории при авторегрессии
BYTE_EMB = 24
EOS = 0                 # нулевой байт = конец


class ByteKANDecoder(nn.Module):
    """P(next_byte | HV_контекста, предыдущие CTX_BYTES байт).

    Условие: непрерывный вектор содержания (факт/тема/аккумулятор диалога).
    Поверхность: чистая авторегрессия по байтам — флексии и связки берутся
    из статистики реального корпуса, а не из шаблонов.
    """

    def __init__(self, dim=DIM, hidden=768, byte_emb=BYTE_EMB, degree=4):
        super().__init__()
        self.byte_emb = nn.Embedding(256, byte_emb)
        self.adapter = nn.Sequential(
            nn.Linear(dim + CTX_BYTES * byte_emb, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.kan = ChebyKANLayer(hidden, hidden, degree=degree)
        self.head = nn.Linear(hidden, 256)

    def forward(self, cond_hv, ctx_bytes):
        # cond_hv: [B, DIM]; ctx_bytes: [B, CTX_BYTES] long
        e = self.byte_emb(ctx_bytes).flatten(1)          # [B, CTX*emb]
        h = self.adapter(torch.cat([cond_hv, e], dim=-1))
        h = F.silu(self.kan(h))
        return self.head(h)                               # logits [B, 256]


def build_corpus(n_sentences=25000):
    rows = []
    with open("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d["ru"].lower().strip()
            t = re.sub(r"\s+", " ", t)
            if 12 <= len(t) <= 90 and re.search(r"[а-яё]", t):
                rows.append(t)
            if len(rows) >= n_sentences:
                break
    return rows


def main():
    random.seed(0)
    torch.manual_seed(0)
    device = "cpu"

    binder = fuga_core.HybridBinder(DIM)
    corpus = build_corpus()
    print(f"corpus sentences: {len(corpus)}")

    # --- условные HV содержания: бандл слов предложения (packed -> ±1) ---
    print("encoding content HVs ...")
    t0 = time.perf_counter()
    conds = []
    for s in range(0, len(corpus), 2000):
        chunk = [re.findall(r"\w+", c)[:16] for c in corpus[s:s + 2000]]
        pk = np.asarray(binder.bind_batch(chunk))
        conds.append(packed_to_torch(pk))
    cond_all = torch.cat(conds).to(device)
    print(f"  done in {time.perf_counter()-t0:.1f}s, shape={tuple(cond_all.shape)}")

    # байтовые представления предложений (+EOS)
    byte_seqs = []
    for c in corpus:
        b = list(c.encode("utf-8")[:120]) + [EOS]
        byte_seqs.append(b)

    model = ByteKANDecoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    def sample_batch(bs=96):
        idx = np.random.randint(0, len(corpus), bs)
        cond_b, ctx_b, tgt_b = [], [], []
        for i in idx:
            seq = byte_seqs[i]
            pos = random.randint(0, len(seq) - 1)
            ctx = seq[max(0, pos - CTX_BYTES):pos]
            ctx = [0] * (CTX_BYTES - len(ctx)) + ctx      # паддинг слева нулями
            cond_b.append(cond_all[i])
            ctx_b.append(ctx)
            tgt_b.append(seq[pos])
        return (torch.stack(cond_b),
                torch.tensor(ctx_b, dtype=torch.long),
                torch.tensor(tgt_b, dtype=torch.long))

    STEPS = 3500
    t0 = time.perf_counter()
    for step in range(STEPS + 1):
        cb, xb, tb = sample_batch()
        logits = model(cb, xb)
        loss = F.cross_entropy(logits, tb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            acc = (logits.argmax(1) == tb).float().mean().item()
            print(f"step {step}: ce={loss.item():.4f} top1={acc:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)")

    # --- генерация ---
    vocab_itos = {}
    @torch.no_grad()
    def generate(cond_hv, max_len=90, temp=0.7):
        ctx = [0] * CTX_BYTES
        out = bytearray()
        for _ in range(max_len):
            logits = model(cond_hv.unsqueeze(0),
                           torch.tensor([ctx], dtype=torch.long))[0]
            probs = F.softmax(logits / temp, dim=-1).numpy()
            nb = int(np.random.choice(256, p=probs))
            if nb == EOS:
                break
            out.append(nb)
            ctx = ctx[1:] + [nb]
        return out.decode("utf-8", errors="ignore").strip()

    print("\n===== ГЕНЕРАЦИЯ ПОВЕРХНОСТИ ИЗ ВЕКТОРА СОДЕРЖАНИЯ =====")
    tests = ["кошка", "мама мыла раму", "погода сегодня хорошая",
             "я люблю читать книги вечером"]
    for t in tests:
        hv = packed_to_torch(np.asarray(binder.bind_batch(
            [re.findall(r"\w+", t)[:16]])))[0]
        gens = {generate(hv) for _ in range(3)}
        print(f"  cond='{t}'")
        for g in list(gens)[:3]:
            print(f"    -> \"{g}\"")

    # --- метрика беглости: доля биграмм, существующих в корпусе ---
    corp_bigrams = set()
    for c in corpus[:20000]:
        b = ("^" + c + "$").encode("utf-8")
        corp_bigrams.update(zip(b, b[1:]))

    def fluency(text):
        b = ("^" + text + "$").encode("utf-8", errors="ignore")
        bg = list(zip(b, b[1:]))
        if not bg:
            return 0.0
        known = sum(1 for p in bg if p in corp_bigrams)
        return known / len(bg)

    scores = []
    for t in ["кот", "погода за окном", "мы говорили об этом вчера",
              "он живёт в иокогаме", "завтра можно не приходить"]:
        hv = packed_to_torch(np.asarray(binder.bind_batch(
            [re.findall(r"\w+", t)[:16]])))[0]
        g = generate(hv)
        if g:
            scores.append(fluency(g))
    print(f"\nfluency (bigrams-in-corpus): {np.mean(scores):.3f} по {len(scores)} генераций")

    model_path = "byte_kan_decoder.pt"
    torch.save(model.state_dict(), model_path)
    print(f"weights saved: {model_path}")


if __name__ == "__main__":
    main()
