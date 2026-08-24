from __future__ import annotations

import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

import fuga_core

from antitf.data_i18n import load_tatoeba_pairs
from antitf.linguistic_data import load_rucola, tokenize as ling_tokenize
from antitf.rust_bridge import packed_to_torch


def tokenize(text: str, max_tokens: int = 16) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text.lower())][:max_tokens] if False else \
        [w.lower() for w in __import__("re").findall(r"\w+", text.lower())][:max_tokens]


import re  # noqa: E402


def bipolar_to_packed(arr: np.ndarray) -> np.ndarray:
    b = (np.asarray(arr) > 0)
    words = np.zeros((b.shape[0], b.shape[1] // 64), dtype=np.uint64)
    for i in range(64):
        words |= b[:, i::64].astype(np.uint64) << np.uint64(i)
    return words


class FugaCrossAligner(torch.nn.Module):
    def __init__(self, dim=2048, hidden=1024):
        super().__init__()
        self.kan_bridge = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, dim),
        )

    def forward(self, hv):
        return F.normalize(self.kan_bridge(hv.float()), dim=-1)


def hybrid_loss(z_pred, hv_en_true, hv_neg, var_w=5.0, neg_w=0.5):
    inv = 1.0 - F.cosine_similarity(z_pred, hv_en_true, dim=-1).mean()
    repulsion = (torch.relu(z_pred @ hv_neg.T).mean()
                 if hv_neg is not None and hv_neg.shape[0] > 0 else torch.zeros(()))
    std = torch.sqrt(z_pred.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    return inv + neg_w * repulsion + var_w * var_loss


@torch.no_grad()
def unbind_acc(binder, z_pred, en_rows, vocab_words, positions=(0, 1, 2, 3)):
    hv_packed = bipolar_to_packed(torch.sign(z_pred).cpu().numpy())
    hits = total = 0
    for pos in positions:
        unbound = np.asarray(binder.unbind_batch(hv_packed, pos + 1))
        scores = np.asarray(binder.score_items(unbound, vocab_words))
        pred_idx = scores.argmax(axis=1)
        for row, pi in enumerate(pred_idx):
            gold = en_rows[row][pos] if pos < len(en_rows[row]) else ""
            if not gold:
                continue
            total += 1
            hits += int(vocab_words[pi] == gold)
    return hits / max(total, 1)


@torch.no_grad()
def evaluate(binder, aligner, hv_ru, hv_en, en_rows, vocab_words):
    z = aligner(F.normalize(hv_ru, dim=-1))
    cos = F.cosine_similarity(z, F.normalize(hv_en, dim=-1), dim=-1).mean().item()
    perm = torch.randperm(z.shape[0])
    cos_shuf = F.cosine_similarity(z, F.normalize(hv_en, dim=-1)[perm], dim=-1).mean().item()
    acc = unbind_acc(binder, z[:500], en_rows[:500], vocab_words)
    return {"cos": cos, "cos_shuf": cos_shuf, "acc@1": acc}


def main():
    random.seed(0)
    torch.manual_seed(0)

    flt = fuga_core.RustLinguisticFilter()
    flt.load_wiktionary_pos_jsonl("datasets/wiktionary/kaikki_ru.jsonl")
    transitions = []
    for s_, a in load_rucola("in_domain_train"):
        if a == 1:
            tt = ling_tokenize(s_); transitions += list(zip(tt, tt[1:]))
    all_pairs = load_tatoeba_pairs(max_pairs=12000)
    for a, b in all_pairs:
        tt = ling_tokenize(a); transitions += list(zip(tt, tt[1:]))
    flt.load_rucola_transitions(transitions)

    # --- фильтрация ДО сплита (тест тоже должен быть валидным RU) ---
    ok_mask = [flt.is_acceptable(p[0], 0.6, 0.3) for p in all_pairs]
    valid_pairs = [p for p, m in zip(all_pairs, ok_mask) if m]
    rejected_ru = [p[0] for p, m in zip(all_pairs, ok_mask) if not m]

    # --- СПЛИТ 80/20 ---
    random.shuffle(valid_pairs)
    cut = int(len(valid_pairs) * 0.8)
    train_pairs, test_pairs = valid_pairs[:cut], valid_pairs[cut:]
    print(f"valid={len(valid_pairs)}  train={len(train_pairs)}  heldout={len(test_pairs)}")

    binder = fuga_core.HybridBinder(2048)

    def encode(rows):
        return packed_to_torch(np.asarray(binder.bind_batch(rows)))

    tr_rows = [(tokenize(a), tokenize(b)) for a, b in train_pairs]
    te_rows = [(tokenize(a), tokenize(b)) for a, b in test_pairs]
    tr_rows = [(a, b) for a, b in tr_rows if a and b]
    te_rows = [(a, b) for a, b in te_rows if a and b]

    hv_ru_tr = encode([a for a, _ in tr_rows])
    hv_en_tr = encode([b for _, b in tr_rows])
    hv_ru_te = encode([a for a, _ in te_rows])
    hv_en_te = encode([b for _, b in te_rows])

    train_ru_texts = [a for a, _ in train_pairs[:400]]
    neg_rows = [tokenize(t) for t in rejected_ru if tokenize(t)]
    salads = flt.make_word_salad_negatives(train_ru_texts, n_shuffles=2)
    neg_rows += [tokenize(s) for s in salads if tokenize(s)]
    hv_neg = F.normalize(encode(neg_rows), dim=-1)

    # словарь кандидатов — ТОЛЬКО из train EN (строгий вариант)
    vocab_words = sorted({w for _, b in tr_rows for w in b})
    en_train = [b for _, b in tr_rows]
    en_test = [b for _, b in te_rows]
    print(f"encoded: train={hv_ru_tr.shape[0]} heldout={hv_ru_te.shape[0]} "
          f"neg={hv_neg.shape[0]} vocab(train-only)={len(vocab_words)}")

    aligner = FugaCrossAligner()
    opt = torch.optim.Adam(aligner.parameters(), lr=1e-3)
    owm = None
    WARMUP = 1200
    TOTAL = 2200
    n = hv_ru_tr.shape[0]
    log_every = 150

    for step in range(TOTAL + 1):
        idx = torch.randint(0, n, (128,))
        nidx = torch.randint(0, hv_neg.shape[0], (min(256, hv_neg.shape[0]),))
        z_pred = aligner(F.normalize(hv_ru_tr[idx], dim=-1))
        loss = hybrid_loss(z_pred, F.normalize(hv_en_tr[idx], dim=-1), hv_neg[nidx])

        if step < WARMUP or owm is None:
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            if step == WARMUP:
                from antitf.owm import WoodburyOWMExecutor
                owm = WoodburyOWMExecutor(aligner, lr=2e-4)
                owm.update_space("kan_bridge.0.weight", F.normalize(hv_ru_tr[:512], dim=-1))
                print(f"  -- phase 2: OWM fixation @step {WARMUP} --")
            owm.zero_grad(); loss.backward(); owm.apply_gradients(lr=2e-4)

        if step % log_every == 0:
            tr = evaluate(binder, aligner, hv_ru_tr[:1000], hv_en_tr[:1000], en_train[:1000], vocab_words)
            ho = evaluate(binder, aligner, hv_ru_te, hv_en_te, en_test, vocab_words)
            print(f"step {step}: TRAIN acc@1={tr['acc@1']:.3f} cos={tr['cos']:.3f} | "
                  f"HELDOUT acc@1={ho['acc@1']:.3f} cos={ho['cos']:.3f} "
                  f"gap={(tr['acc@1']-ho['acc@1']):+.3f}")

    tr = evaluate(binder, aligner, hv_ru_tr[:2000], hv_en_tr[:2000], en_train[:2000], vocab_words)
    ho = evaluate(binder, aligner, hv_ru_te, hv_en_te, en_test, vocab_words)
    print("\n===== FINAL =====")
    print(f"TRAIN   acc@1={tr['acc@1']:.4f}  cos={tr['cos']:.3f}  cos_shuf={tr['cos_shuf']:.3f}")
    print(f"HELDOUT acc@1={ho['acc@1']:.4f}  cos={ho['cos']:.3f}  cos_shuf={ho['cos_shuf']:.3f}")
    print(f"generalization gap (train-heldout): {tr['acc@1']-ho['acc@1']:+.4f}")


if __name__ == "__main__":
    main()
