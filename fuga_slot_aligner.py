from __future__ import annotations

import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

import fuga_core

from antitf.data_i18n import load_tatoeba_pairs
from antitf.linguistic_data import load_rucola, tokenize as ling_tokenize


def tokenize(text: str, max_tokens: int = 12) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text.lower())][:max_tokens]


def unpack3(packed: np.ndarray) -> torch.Tensor:
    """u64 [B, N, W] -> bipolar float [B, N, W*64]."""
    words = torch.from_numpy(np.ascontiguousarray(packed)).long()
    b, n, w = words.shape
    bits = ((words.unsqueeze(-1) >> torch.arange(64)) & 1).reshape(b, n, w * 64)
    return bits.float() * 2 - 1


class SlotTranslator(nn.Module):
    """Слово -> слово: единая функция перевода для ЛЮБОГО предложения."""

    def __init__(self, dim=2048, hidden=1024):
        super().__init__()
        self.word_translator = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, word_hvs):          # [B, N, D]
        return F.normalize(self.word_translator(word_hvs), dim=-1)


def compose(pred_words: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """z_sent = normalize(sum_i slot_i); слоты Rust УЖЕ повёрнуты на позицию."""
    b, n, d = pred_words.shape
    acc = torch.zeros(b, d)
    for i in range(n):
        acc += pred_words[:, i, :] * (seq_lens > i).float().unsqueeze(1)
    return F.normalize(acc, dim=-1)


def slot_loss(pred_w, true_w, seq_lens, sent_w=0.5):
    n_slots = pred_w.shape[1]
    ar = torch.arange(n_slots)
    mask = (ar[None, :] < seq_lens[:, None])               # [B, N]
    cos_w = F.cosine_similarity(pred_w, true_w, dim=-1)    # [B, N]
    word_loss = (1.0 - cos_w)[mask].mean()
    sent_loss = 1.0 - F.cosine_similarity(
        compose(pred_w, seq_lens), compose(true_w, seq_lens), dim=-1).mean()
    return word_loss + sent_w * sent_loss


@torch.no_grad()
def unbind_eval(binder, z_sent, en_rows, vocab_words):
    hv = ((torch.sign(z_sent).cpu().numpy() > 0)).astype(np.uint64)
    packed = np.zeros((hv.shape[0], hv.shape[1] // 64), dtype=np.uint64)
    for i in range(64):
        packed |= hv[:, i::64].astype(np.uint64) << np.uint64(i)
    hits = total = 0
    for pos in (0, 1, 2, 3):
        unbound = np.asarray(binder.unbind_batch(packed, pos + 1))
        scores = np.asarray(binder.score_items(unbound, vocab_words))
        pred_idx = scores.argmax(axis=1)
        for r, pi in enumerate(pred_idx):
            gold = en_rows[r][pos] if pos < len(en_rows[r]) else ""
            if not gold:
                continue
            total += 1
            hits += int(vocab_words[pi] == gold)
    return hits / max(total, 1)


def main():
    random.seed(0)
    torch.manual_seed(0)

    flt = fuga_core.RustLinguisticFilter()
    flt.load_wiktionary_pos_jsonl("datasets/wiktionary/kaikki_ru.jsonl")
    transitions = []
    for s_, a in load_rucola("in_domain_train"):
        if a == 1:
            tt = ling_tokenize(s_, ); transitions += list(zip(tt, tt[1:]))
    all_pairs = load_tatoeba_pairs(max_pairs=12000)
    for a, b in all_pairs:
        tt = ling_tokenize(a); transitions += list(zip(tt, tt[1:]))
    flt.load_rucola_transitions(transitions)

    ok_mask = [flt.is_acceptable(p[0], 0.6, 0.3) for p in all_pairs]
    valid_pairs = [p for p, m in zip(all_pairs, ok_mask) if m]
    random.shuffle(valid_pairs)
    cut = int(len(valid_pairs) * 0.8)
    train_pairs, test_pairs = valid_pairs[:cut], valid_pairs[cut:]
    print(f"valid={len(valid_pairs)} train={len(train_pairs)} heldout={len(test_pairs)}")

    binder = fuga_core.HybridBinder(2048)
    MAXLEN = 12
    tr_ru = [tokenize(a) for a, _ in train_pairs]
    tr_en = [tokenize(b) for _, b in train_pairs]
    te_ru = [tokenize(a) for a, _ in test_pairs]
    te_en = [tokenize(b) for _, b in test_pairs]

    # Храним packed u64 и распаковываем ЛЕНИВО по батчу (иначе ~5ГБ float).
    pk_tr_ru = np.asarray(binder.extract_word_hvs_batch(tr_ru, MAXLEN))
    pk_tr_en = np.asarray(binder.extract_word_hvs_batch(tr_en, MAXLEN))
    pk_te_ru = np.asarray(binder.extract_word_hvs_batch(te_ru, MAXLEN))
    pk_te_en = np.asarray(binder.extract_word_hvs_batch(te_en, MAXLEN))

    def unpack_rows(pk, idx=None):
        sub = pk if idx is None else pk[idx]
        return unpack3(sub)
    len_tr = torch.tensor([min(len(r), MAXLEN) for r in tr_ru])
    len_te = torch.tensor([min(len(r), MAXLEN) for r in te_ru])
    vocab_words = sorted({w for r in tr_en for w in r})
    print(f"vocab(train-only)={len(vocab_words)}")

    # sanity: развязка из ИСТИННОЙ EN-композиции
    z_true_te = compose(unpack_rows(pk_te_en, slice(0, 500)), len_te[:500])
    print(f"sanity true-compose acc@1={unbind_eval(binder, z_true_te, te_en[:500], vocab_words):.4f}")

    model = SlotTranslator()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = pk_tr_ru.shape[0]
    B = 96

    def ev(set_name, pk_ru, pk_en, rows, lens):
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, min(pk_ru.shape[0], 1000), 256):
                preds.append(model(unpack_rows(pk_ru, slice(s, s + 256))))
            zp = torch.cat(preds)
            lens_sub = lens[:zp.shape[0]]
            en_sub = unpack_rows(pk_en, slice(0, zp.shape[0]))
            wl = slot_loss(zp, en_sub, lens_sub).item()
            acc = unbind_eval(binder, compose(zp, lens_sub), rows[:zp.shape[0]], vocab_words)
        print(f"  {set_name}: word_cos_loss={wl:.4f} acc@1_unbind={acc:.4f}")

    for step in range(1501):
        idx = torch.randint(0, n, (B,))
        pw = model(unpack_rows(pk_tr_ru, idx))
        loss = slot_loss(pw, unpack_rows(pk_tr_en, idx), len_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            print(f"step {step}: loss={loss.item():.4f}")
            ev("TRAIN", pk_tr_ru, pk_tr_en, tr_en, len_tr)
            ev("HELDOUT", pk_te_ru, pk_te_en, te_en, len_te)


if __name__ == "__main__":
    main()
