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
from antitf.linguistic_data import load_rucola, tokenize as ling_tokenize


def tokenize(text: str, max_tokens: int = 12) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text.lower())][:max_tokens]


def unpack3(pk):
    w = torch.from_numpy(np.ascontiguousarray(pk)).long()
    b, n, W = w.shape
    return (((w.unsqueeze(-1) >> torch.arange(64)) & 1).reshape(b, n, W * 64)).float() * 2 - 1


class SlotTranslator(nn.Module):
    def __init__(self, dim=2048, hidden=1024):
        super().__init__()
        self.word_translator = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, word_hvs):
        return F.normalize(self.word_translator(word_hvs), dim=-1)


def compose(slots, lens):
    acc = torch.zeros(slots.shape[0], slots.shape[2])
    for i in range(slots.shape[1]):
        acc += slots[:, i] * (lens > i).float().unsqueeze(1)
    return F.normalize(acc, dim=-1)


@torch.no_grad()
def unbind_eval(binder, z_sent, en_rows, vocab_words):
    sgn = torch.sign(z_sent).cpu().numpy() > 0
    packed = np.zeros((sgn.shape[0], sgn.shape[1] // 64), dtype=np.uint64)
    for i in range(64):
        packed |= sgn[:, i::64].astype(np.uint64) << np.uint64(i)
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
            tt = ling_tokenize(s_); transitions += list(zip(tt, tt[1:]))
    all_pairs = load_tatoeba_pairs(max_pairs=12000)
    for a, b in all_pairs:
        tt = ling_tokenize(a); transitions += list(zip(tt, tt[1:]))
    flt.load_rucola_transitions(transitions)

    ok_mask = [flt.is_acceptable(p[0], 0.6, 0.3) for p in all_pairs]
    valid_pairs = [p for p, m in zip(all_pairs, ok_mask) if m]
    random.shuffle(valid_pairs)
    cut = int(len(valid_pairs) * 0.8)
    train_pairs, test_pairs = valid_pairs[:cut], valid_pairs[cut:]

    tr_ru = [tokenize(a) for a, _ in train_pairs]
    tr_en = [tokenize(b) for _, b in train_pairs]
    te_ru = [tokenize(a) for a, _ in test_pairs]
    te_en = [tokenize(b) for _, b in test_pairs]
    vocab_words = sorted({w for r in tr_en for w in r})
    print(f"train={len(train_pairs)} heldout={len(test_pairs)} vocab={len(vocab_words)}")

    # --- 1. IBM Model-1 EM на Rust ---
    ibm = fuga_core.IbmModel1()
    t0 = time.perf_counter()
    n_params = ibm.train([(a, b) for a, b in zip(tr_ru, tr_en)], epochs=4)
    print(f"IBM-1 trained: {n_params} пар (ru,en) за {time.perf_counter()-t0:.1f}s "
          f"vocabs={ibm.vocab_sizes()}")
    for probe in ("дом", "кошка", "книга", "идти"):
        top = ibm.translate_topk(probe, 3)
        print(f"  {probe!r} -> {top}")

    binder = fuga_core.HybridBinder(2048)
    MAXLEN = 12

    def pack_slots(rows):
        out = []
        for s in range(0, len(rows), 2000):
            out.append(np.asarray(binder.extract_word_hvs_batch(rows[s:s+2000], MAXLEN)))
        return np.concatenate(out)

    pk_tr_ru, pk_tr_en = pack_slots(tr_ru), pack_slots(tr_en)
    pk_te_ru, pk_te_en = pack_slots(te_ru), pack_slots(te_en)
    len_tr = torch.tensor([min(len(r), MAXLEN) for r in tr_ru])
    len_te = torch.tensor([min(len(r), MAXLEN) for r in te_ru])

    model = SlotTranslator()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 2. Супервизия: soft-target через alignment ---
    # target_ru_i = sum_j A[i,j] * true_en_slot_j   (в HV-пространстве ±1,
    # нормировка строк A делает выпуклую комбинацию; знак сохраняет биполярность
    # после маскирования паддинга и повторной нормализации косинусом).
    def ibm_soft_targets(batch_idx):
        tgts = torch.zeros(len(batch_idx), MAXLEN, 2048)
        for bi, gi in enumerate(batch_idx):
            A = np.asarray(ibm.align_pair(tr_ru[gi], tr_en[gi]))[:MAXLEN]
            if A.size == 0:
                continue
            E = unpack3(pk_tr_en[gi:gi+1])[0]           # [n_en, D]
            tgt = torch.from_numpy(A).float() @ E[: A.shape[1]]
            tgts[bi, : tgt.shape[0]] = F.normalize(tgt, dim=-1)
        return tgts

    def eval_set(name, pk_ru, rows_en, lens, use_ibm_targets):
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, min(pk_ru.shape[0], 1000), 256):
                sl = unpack3(pk_ru[s:s+256])
                preds.append(model(sl))
            zp = torch.cat(preds)
            acc = unbind_eval(binder, compose(zp, lens[:zp.shape[0]]),
                              rows_en[:zp.shape[0]], vocab_words)
        print(f"  {name}: acc@1_unbind={acc:.4f}")

    print("[baseline: чистый словарь IBM без KAN]")
    # жадный перевод каждого RU слова -> compose из предсказанных EN слов
    greedy_hits = greedy_tot = 0
    for gi in range(300):
        pred_words = []
        for rw in te_ru[gi]:
            top = ibm.translate_topk(rw, 1)
            if top and top[0][1] > 0.01:
                pred_words.append(top[0][0])
        if not pred_words or not te_en[gi]:
            continue
        z = compose(unpack3(np.asarray(
            binder.extract_word_hvs_batch([pred_words], MAXLEN))), torch.tensor([len(pred_words)]))
        acc_row = unbind_eval(binder, z, [te_en[gi]], vocab_words)
        greedy_hits += acc_row
        greedy_tot += 1
    print(f"  GREEDY-DICT heldout acc@1={greedy_hits/max(greedy_tot,1):.4f} ({greedy_tot} предложений)")

    print("[KAN slot translator: IBM-supervised loss]")
    n = pk_tr_ru.shape[0]
    B = 96
    for step in range(1001):
        idx = torch.randint(0, n, (B,))
        idx_list = idx.tolist()
        ru_slots = unpack3(pk_tr_ru[idx])
        pw = model(ru_slots)
        soft_t = ibm_soft_targets(idx_list).to(pw.device)
        mask = (torch.arange(MAXLEN)[None, :] < len_tr[idx][:, None])
        cos = F.cosine_similarity(pw, soft_t, dim=-1)
        word_loss = (1.0 - cos)[mask].mean()
        loss = word_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1000:
            print(f"step {step}: ibm_word_loss={loss.item():.4f}")
            eval_set("TRAIN", pk_tr_ru, tr_en, len_tr, True)
            eval_set("HELDOUT", pk_te_ru, te_en, len_te, True)


if __name__ == "__main__":
    main()
