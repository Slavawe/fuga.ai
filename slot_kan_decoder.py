"""Slot-Conditioned Byte-KAN Decoder (v0.2, Этап 1-2).

Урок контрольного эксперимента: одиночный бандл-HV условия декодер
игнорирует. Здесь условие адресуемое — ПОСЛОВНЫЕ VSA-слоты, и на каждом
байтовом шаге attention решает, какой слот сейчас выражается:

    h_t     = GRUCell(byte_emb(b_{t-1}) ++ ctx_{t-1}, h_{t-1})
    alpha_t = softmax(keys @ query(h_t))            # по слотам предложения
    ctx_t   = sum_i alpha_i * value_i
    logits  = head([h_t ; ctx_t])

Семантическое управление проверяется метрикой content coverage:
доля слов-слотов, реально появившихся в сгенерированной строке.
"""

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

DIM = 2048
MAX_SLOTS = 12
KV_DIM = 64
CTX_BYTES = 10
BYTE_EMB = 24
EOS = 0


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text.lower())][:MAX_SLOTS]


def load_corpus(n=25000):
    rows = []
    with open("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d["ru"].lower().strip()
            t = re.sub(r"\s+", " ", t)
            if 12 <= len(t) <= 90 and re.search(r"[а-яё]", t):
                rows.append(t)
            if len(rows) >= n:
                break
    return rows


def unpack_slots(pk: np.ndarray) -> torch.Tensor:
    """u64 [B, N, W] -> bipolar [B, N, D]."""
    w = torch.from_numpy(np.ascontiguousarray(pk)).long()
    b, n, W = w.shape
    bits = ((w.unsqueeze(-1) >> torch.arange(64)) & 1).reshape(b, n, W * 64)
    return bits.float() * 2 - 1


class SlotAttentionDecoder(nn.Module):
    def __init__(self, kv_dim=KV_DIM, hidden=512, byte_emb=BYTE_EMB):
        super().__init__()
        self.kv_dim = kv_dim
        self.byte_emb = nn.Embedding(256, byte_emb)
        self.gru = nn.GRUCell(byte_emb + kv_dim, hidden)
        self.query = nn.Linear(hidden, kv_dim)
        self.kan = ChebyKANLayer(hidden + kv_dim, hidden, degree=4)
        self.head = nn.Linear(hidden, 256)

    def forward(self, ctx_bytes, h, keys, values):
        """Один шаг. ctx_bytes: [B], h: [B,H], keys/values: [B,N,KV]."""
        # attention: query из состояния декодера, ключи — проекции слотов
        q = self.query(h)                                   # [B,KV]
        scores = (keys @ q.unsqueeze(-1)).squeeze(-1) / np.sqrt(self.kv_dim)
        alpha = F.softmax(scores, dim=-1)                   # [B,N]
        ctx_vec = (alpha.unsqueeze(-1) * values).sum(dim=1)  # [B,KV]
        x = torch.cat([self.byte_emb(ctx_bytes), ctx_vec], dim=-1)
        h = self.gru(x, h)
        feat = self.kan(torch.cat([h, ctx_vec], dim=-1))
        logits = self.head(F.silu(feat))
        return logits, h, alpha


class FixedProjection:
    """Замороженная случайная проекция VSA->KV (random features):
    градиент не нужен — внимание обучается через query."""

    def __init__(self, seed=1234):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(DIM, KV_DIM, generator=g) / np.sqrt(DIM)

    def project(self, slots_bipolar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = slots_bipolar @ self.W
        v = torch.tanh(slots_bipolar @ self.W * 1.7)
        return k, v


def main():
    random.seed(0)
    torch.manual_seed(0)

    binder = fuga_core.HybridBinder(DIM)
    corpus = load_corpus()
    rows_words = [tokenize(c) for c in corpus]
    keep = [i for i, r in enumerate(rows_words) if len(r) >= 2]
    corpus = [corpus[i] for i in keep]
    rows_words = [rows_words[i] for i in keep]
    print(f"corpus={len(corpus)}")

    print("encoding word slots (Rust) ...")
    t0 = time.perf_counter()
    pk_all = []
    for s in range(0, len(rows_words), 2000):
        pk_all.append(np.asarray(binder.extract_word_hvs_batch(
            rows_words[s:s + 2000], MAX_SLOTS)))
    pk_all = np.concatenate(pk_all)                    # [B, N, W] u64
    lens = torch.tensor([min(len(r), MAX_SLOTS) for r in rows_words])
    print(f"  {pk_all.shape} за {time.perf_counter()-t0:.0f}s")

    proj = FixedProjection()

    def get_kv(idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        slots = unpack_slots(pk_all[idx])               # [B,N,D]
        return proj.project(slots)

    model = SlotAttentionDecoder()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    byte_seqs = []
    for c in corpus:
        b = list(c.encode("utf-8")[:110]) + [EOS]
        byte_seqs.append(b)

    STEPS = 2200
    B = 64
    t0 = time.perf_counter()
    for step in range(STEPS + 1):
        idx = np.random.randint(0, len(corpus), B)
        K, V = get_kv(list(idx))
        h = torch.zeros(B, 512)
        loss = 0.0
        # teacher forcing по всей длине самой длинной последовательности батча
        max_len = min(max(len(byte_seqs[i]) for i in idx), 100)
        ctx_bytes = torch.zeros(B, dtype=torch.long)
        ce_sum = 0.0
        for t in range(max_len):
            logits, h, _ = model(ctx_bytes, h, K, V)
            tgt = torch.tensor([byte_seqs[i][t] if t < len(byte_seqs[i]) else EOS
                                for i in idx])
            mask = torch.tensor([t < len(byte_seqs[i]) for i in idx])
            if mask.any():
                ce = F.cross_entropy(logits[mask], tgt[mask], reduction="sum")
                ce_sum = ce_sum + ce
            nb = tgt.clamp(min=0)
            ctx_bytes = nb
        loss = ce_sum / sum(len(byte_seqs[i]) for i in idx)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == STEPS:
            with torch.no_grad():
                acc_mask, correct = 0, 0
            print(f"step {step}: ce/byte={loss.item():.4f} ({time.perf_counter()-t0:.0f}s)")

    # ===== ГЕНЕРАЦИЯ С ATTENTION-ПЕРЕКЛЮЧЕНИЕМ СЛОТОВ =====
    vocab = sorted({w for r in rows_words[:5000] for w in r})

    @torch.no_grad()
    def generate(words: list[str], temp=0.65, max_len=90):
        words = words[:MAX_SLOTS]
        pk = np.asarray(binder.extract_word_hvs_batch([words], MAX_SLOTS))
        K, V = proj.project(unpack_slots(pk))
        n_slots = len(words)
        ctx = [0] * CTX_BYTES
        h = torch.zeros(1, 512)
        out = bytearray()
        alphas_log = []
        for _ in range(max_len):
            logits, h, alpha = model(torch.tensor([ctx[-1]], dtype=torch.long),
                                     h, K, V)
            alphas_log.append(alpha[0].numpy())
            probs = F.softmax(logits[0] / temp, dim=-1).numpy()
            nb = int(np.random.choice(256, p=probs))
            if nb == EOS:
                break
            out.append(nb)
            ctx = ctx[1:] + [nb]
        att_usage = np.stack(alphas_log).mean(axis=0)[:n_slots]
        return out.decode("utf-8", "ignore").strip(), att_usage

    print("\n===== SLOT-CONDITIONED GENERATION =====")
    tests = [
        ["мама", "мыла", "раму"],
        ["погода", "сегодня", "хорошая"],
        ["кот", "спит", "на", "диване"],
        ["я", "люблю", "читать", "книги", "вечером"],
    ]
    cov_all = []
    for words in tests:
        g, att = generate(words)
        # content coverage: какие слова-слота просели в текст
        covered = sum(1 for w in words if w[:5] in g or any(
            w[:4] in g for _ in [0]))
        cov_all.append(covered / len(words))
        att_str = " ".join(f"{a:.2f}" for a in att)
        print(f"  slots={words}")
        print(f"    -> \"{g}\"")
        print(f"    coverage={covered}/{len(words)}  attention={att_str}")

    # held-out замер coverage на реальных корпусных предложениях
    hcov = []
    test_idx = random.sample(range(len(corpus)), 60)
    for i in test_idx:
        g, _ = generate(rows_words[i])
        if not g:
            continue
        covered = sum(1 for w in rows_words[i][:6] if w[:5] in g)
        hcov.append(covered / min(len(rows_words[i]), 6))
    print(f"\nheld-out slot coverage@{len(test_idx)} предложений: "
          f"{np.mean(hcov):.3f}")

    torch.save(model.state_dict(), "slot_kan_decoder.pt")
    print("weights saved: slot_kan_decoder.pt")


if __name__ == "__main__":
    main()
