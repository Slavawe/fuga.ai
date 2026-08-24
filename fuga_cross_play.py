from __future__ import annotations

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

from antitf.data_i18n import load_tatoeba_pairs
from antitf.item_memory import SimpleWordVocab
from antitf.linguistic_data import load_rucola, tokenize
from antitf.rust_bridge import packed_to_torch


def tokenize(text: str, max_tokens: int = 16) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text.lower())][:max_tokens]


def bipolar_to_packed(arr: np.ndarray) -> np.ndarray:
    """±1 float [B, bits] -> packed uint64 [B, bits//64] (bit i слова = позиция j*64+i)."""
    b = (np.asarray(arr) > 0)
    words = np.zeros((b.shape[0], b.shape[1] // 64), dtype=np.uint64)
    for i in range(64):
        words |= b[:, i::64].astype(np.uint64) << np.uint64(i)
    return words


class FugaCrossAligner(nn.Module):
    """HV_ru -> непрерывный латент в пространстве HV_en (без sign на выходе)."""

    def __init__(self, dim: int = 2048, hidden: int = 1024):
        super().__init__()
        self.kan_bridge = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, hv_ru: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.kan_bridge(hv_ru.float()), dim=-1)


def hybrid_loss(z_pred, hv_en_true, hv_neg, var_w=5.0, neg_w=0.5):
    inv = 1.0 - F.cosine_similarity(z_pred, hv_en_true, dim=-1).mean()
    if hv_neg.shape[0] > 0:
        neg_sim = z_pred @ hv_neg.T                      # [B, Neg]
        repulsion = torch.relu(neg_sim).mean()
    else:
        repulsion = torch.zeros((), )
    std = torch.sqrt(z_pred.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    return inv + neg_w * repulsion + var_w * var_loss, {
        "inv": float(inv), "rep": float(repulsion), "var": float(var_loss)}


@torch.no_grad()
def verify_unbinding(binder, z_pred: torch.Tensor, en_token_rows: list[list[str]],
                     vocab_words: list[str], positions=range(4)) -> float:
    """sign -> packed u64 -> Rust unbind -> score_items по словарю."""
    hv_packed = bipolar_to_packed(torch.sign(z_pred).cpu().numpy())
    hits = total = 0
    for pos in positions:
        unbound = np.asarray(binder.unbind_batch(hv_packed, pos + 1))
        scores = np.asarray(binder.score_items(unbound, vocab_words))
        pred_idx = scores.argmax(axis=1)
        for row, pi in enumerate(pred_idx):
            gold = en_token_rows[row][pos] if pos < len(en_token_rows[row]) else ""
            if not gold:
                continue
            total += 1
            hits += int(vocab_words[pi] == gold)
    return hits / max(total, 1)


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    device = "cpu"

    flt = fuga_core.RustLinguisticFilter()
    flt.load_wiktionary_pos_jsonl("datasets/wiktionary/kaikki_ru.jsonl")
    # переходы: RuCoLA ok + Tatoeba (без них trans_cov=0 -> фильтр режет всё)
    transitions = []
    for s_, a in load_rucola("in_domain_train"):
        if a == 1:
            tt = tokenize(s_); transitions += list(zip(tt, tt[1:]))
    pairs_pre = load_tatoeba_pairs(max_pairs=8000)
    for a, b in pairs_pre:
        tt = tokenize(a); transitions += list(zip(tt, tt[1:]))
    flt.load_rucola_transitions(transitions)
    print(f"lexicon={flt.vocab_size()} transitions={flt.transitions_size()}")

    pairs = pairs_pre
    # фильтрация RU-стороны до кодирования
    ru_texts = [p[0] for p in pairs]
    ok_mask = [flt.is_acceptable(t, 0.6, 0.3) for t in ru_texts]
    valid = [(p[0], p[1]) for p, m in zip(pairs, ok_mask) if m]
    rejected_ru = [p[0] for p, m in zip(pairs, ok_mask) if not m]
    salads = flt.make_word_salad_negatives([v[0] for v in valid[:400]], n_shuffles=2)
    rejected_ru += salads
    random.shuffle(rejected_ru)
    rejected_ru = rejected_ru[: len(valid) // 2]
    print(f"pairs={len(pairs)} valid={len(valid)} negatives={len(rejected_ru)}")

    binder = fuga_core.HybridBinder(2048)
    tok_valid = [(tokenize(a), tokenize(b)) for a, b in valid]
    keep = [(a, b) for a, b in tok_valid if a and b]

    def encode(rows):
        return packed_to_torch(np.asarray(binder.bind_batch(rows)))

    hv_ru = encode([a for a, _ in keep])
    hv_en = encode([b for _, b in keep])
    hv_neg = F.normalize(encode([tokenize(t) for t in rejected_ru if tokenize(t)]), dim=-1)

    en_rows = [b for _, b in keep]
    vocab_words = sorted({w for r in en_rows for w in r})
    print(f"encoded: hv_ru={tuple(hv_ru.shape)} hv_en={tuple(hv_en.shape)} "
          f"neg={tuple(hv_neg.shape)} vocab={len(vocab_words)}")

    # sanity: развязка на ИСТИННЫХ EN-HV должна быть ~1.0
    sanity = verify_unbinding(binder, encode(en_rows[:200]), en_rows[:200], vocab_words)
    print(f"sanity unbinding on true HV_en: acc@1={sanity:.4f}")

    aligner = FugaCrossAligner()
    # Фаза 1 (Adam): OWM с пустой памятью задач заморозил бы ВСЕ градиенты.
    # Фаза 2 (после warmup): фиксация через Woodbury-проекцию.
    opt = torch.optim.Adam(aligner.parameters(), lr=1e-3)
    owm = None
    WARMUP = 250
    print("optimizer: Adam(warmup) -> WoodburyOWM(phase2)")

    hv_ru_t = F.normalize(hv_ru, dim=-1)
    hv_en_t = F.normalize(hv_en, dim=-1)

    n = hv_ru_t.shape[0]
    history = []
    for step in range(401):
        idx = torch.randint(0, n, (128,))
        nidx = torch.randint(0, hv_neg.shape[0],
                             (min(256, hv_neg.shape[0]),)) if hv_neg.shape[0] else None
        z_pred = aligner(hv_ru_t[idx])
        loss, parts = hybrid_loss(z_pred, hv_en_t[idx],
                                  hv_neg[nidx] if nidx is not None else torch.zeros(0, 2048))
        if step < WARMUP or owm is None:
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            if step == WARMUP:
                from antitf.owm import WoodburyOWMExecutor
                owm = WoodburyOWMExecutor(aligner, lr=1e-4)
                print(f"  -- phase 2: OWM fixation on (Adam -> projected) --")
                # консолидация: пространство активаций фазы 1
                with torch.no_grad():
                    sample_idx = torch.randint(0, n, (512,))
                    A = hv_ru_t[sample_idx]
                owm.update_space("kan_bridge.0.weight", A)
            owm.zero_grad(); loss.backward()
            owm.apply_gradients(lr=1e-4)

        if step % 50 == 0 or step == 400:
            with torch.no_grad():
                zp = aligner(hv_ru_t[:512])
                cos_en = F.cosine_similarity(zp, hv_en_t[:512], dim=-1).mean().item()
                acc1 = verify_unbinding(binder, zp[:300], en_rows[:300], vocab_words)
                perm = torch.randperm(512)
                cos_shuf = F.cosine_similarity(zp, hv_en_t[perm], dim=-1).mean().item()
            history.append((step, cos_en, cos_shuf, acc1))
            print(f"step {step}: loss={loss.item():.4f} inv={parts['inv']:.3f} "
                  f"cos(pred,en)={cos_en:.3f} cos(shuffled)={cos_shuf:.3f} "
                  f"acc@1_unbind={acc1:.4f}")

    print("\ndynamics:")
    for st, ce, cs, a1 in history:
        print(f"  step {st}: acc@1={a1:.4f} gap={ce-cs:+.3f}")


if __name__ == "__main__":
    main()
