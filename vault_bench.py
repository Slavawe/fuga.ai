from __future__ import annotations

import json
import random
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

import fuga_core
from antitf.rust_bridge import packed_to_torch


def tokenize(text, n=12):
    return [w.lower() for w in re.findall(r"\w+", text.lower())][:n]


def main():
    random.seed(0)

    # ===== 1. IBM Model-1 на 150K реальных пар =====
    pairs = []
    with open("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ru, en = tokenize(d["ru"]), tokenize(d["en"])
            if ru and en:
                pairs.append((ru, en))
    random.shuffle(pairs)
    cut = int(len(pairs) * 0.95)
    train, test = pairs[:cut], pairs[cut:]
    print(f"[tatoeba-real] total={len(pairs)} train={len(train)} heldout={len(test)}")

    ibm = fuga_core.IbmModel1()
    t0 = time.perf_counter()
    npairs = ibm.train(train, epochs=4)
    print(f"IBM-1: {npairs} словарных пар за {time.perf_counter()-t0:.1f}s "
          f"(на {len(train)} предложениях)")
    for probe in ("дом", "кошка", "книга", "город"):
        print(f"  {probe!r} -> {ibm.translate_topk(probe, 3)}")

    # жадный словарный перевод held-out: топ-1 слово RU -> топ-1 EN,
    # проверка попадания в ЗОЛОТОЙ набор слов предложения (recall)
    hits = tot = 0
    for ru, en in test[:1000]:
        gold = set(en)
        for w in set(ru):
            top = ibm.translate_topk(w, 1)
            if not top or top[0][1] < 0.05:
                continue
            tot += 1
            hits += int(top[0][0] in gold)
    print(f"greedy dict word-hit@heldout: {hits/max(tot,1):.4f} ({tot} прогнозов)")

    # ===== 2. ConceptNet per-subject память =====
    facts = []
    seen_subjects = {}
    with open("dataset_vault/02_world_concepts/conceptnet_sro_real.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["lang"] != "en":
                continue
            s = "_".join(d["subject"].lower().split())
            r = d["relation"]
            o = "_".join(d["object"].lower().split())
            if not s or not o or s == o or len(s) > 24 or len(o) > 24:
                continue
            facts.append((s, r, o))
    # берём субъектов с >=2 фактами для честного дискриминационного теста
    by_subj = {}
    for s, r, o in facts:
        by_subj.setdefault(s, []).append((r, o))
    multi = [(s, ro) for s, ro in by_subj.items() if len(ro) >= 2]
    random.shuffle(multi)
    sample = multi[:300]
    print(f"\n[conceptnet] subjects with >=2 facts tested: {len(sample)}")

    binder = fuga_core.HybridBinder(2048)

    def hv(name):
        return packed_to_torch(np.asarray(binder.bind_batch([[name]])))[0]

    def rot(v, k):
        return torch.roll(v, shifts=k)

    def make_fact(s_, r_, o_):
        return torch.sign(rot(hv(f"S:{s_}"), 1) * rot(hv(f"R:{r_}"), 2) *
                          rot(hv(f"O:{o_}"), 3) + 1e-5)

    ok = tot2 = 0
    t0 = time.perf_counter()
    for subj, ro_list in sample:
        mem = torch.sign(sum(make_fact(subj, r_, o_) for r_, o_ in ro_list) + 1e-5)
        mem[mem == 0] = 1
        r_q, o_gold = random.choice(ro_list)
        q = torch.sign(rot(hv(f"S:{subj}"), 1) * rot(hv(f"R:{r_q}"), 2) + 1e-5)
        residual = torch.sign(mem * q + 1e-5)
        candidates = sorted({o_ for _, o_ in ro_list})
        sims = [(float((residual * rot(hv(f"O:{o_}"), 3)).mean()), o_)
                for o_ in candidates]
        pred = max(sims)[1]
        tot2 += 1
        ok += int(pred == o_gold)
    dt = time.perf_counter() - t0
    print(f"concept-memory acc@1: {ok/max(tot2,1):.4f} ({ok}/{tot2}) за {dt:.1f}s")


if __name__ == "__main__":
    main()
