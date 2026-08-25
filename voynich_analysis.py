
"""Структурный анализ Манускрипта Войнича через VSA-стек.

Три измеримых вопроса:
 1. Отличим ли EVA-текст от естественного языка по поведению
    novelty-фильтра (surprise-динамика)?
 2. Есть ли у строк Войнича кластерная структура в VSA
    (повторы = реальная структура, не шум)?
 3. Как выглядит граф переходов EVA против естественного языка?
"""

from __future__ import annotations

from __future__ import annotations

import random
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

import fuga_core
from antitf.rust_bridge import packed_to_torch
from astral.data_filter import AstralDataStreamFilter
from astral.voynich_main import VoynichMainManuscript


def main():
    random.seed(0)
    ms = VoynichMainManuscript()
    print("[voynich f1r stats]", ms.stats())

    binder = fuga_core.HybridBinder(2048)
    lines = ms.lines()

    # ===== 1. Surprise-профиль: Войнич vs русский Tatoeba vs shuffle =====
    flt_v = AstralDataStreamFilter(adaptive=True, margin=0.25)
    flt_r = AstralDataStreamFilter(adaptive=True, margin=0.25)
    flt_s = AstralDataStreamFilter(adaptive=True, margin=0.25)

    from datasets import load_dataset
    ru_pairs = []
    try:
        with open("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ru_pairs.append(d["ru"])
                if len(ru_pairs) >= 200:
                    break
    except Exception:
        pass

    def surprise_profile(flt, texts):
        surps = []
        prev_hv = None
        for t in texts:
            words = [w for w in re.findall(r"[a-zа-яё]+", t.lower())][:16]
            if len(words) < 2:
                continue
            pk = np.asarray(binder.bind_batch([words]))
            hv = packed_to_torch(pk)[0]
            if prev_hv is None:
                prev_hv = hv
                continue
            pred = torch.roll(prev_hv, shifts=5)   # тривиальный предиктор
            s = flt.surprise(pred, hv)
            surps.append(s)
            prev_hv = hv
        return surps

    v_lines = [" ".join(l) for l in lines]
    shuffled = [" ".join(random.sample(l, len(l))) for l in lines]

    sv = surprise_profile(flt_v, v_lines)
    sr = surprise_profile(flt_r, ru_texts := [p for p in ru_pairs][:len(v_lines)])
    ss = surprise_profile(flt_s, shuffled)

    import statistics as st
    print("\n[surprise profile, тривиальный предиктор]")
    if sv: print(f"  voynich lines : mean={st.mean(sv):.3f}")
    if sr: print(f"  russian text  : mean={st.mean(sr):.3f}")
    if ss: print(f"  shuffled lines: mean={st.mean(ss):.3f}")

    # ===== 2. Кластеризация строк в VSA =====
    hvs = torch.stack([ms.line_hv(binder, l) for l in lines]).float()
    sims = (hvs @ hvs.T).numpy()
    np.fill_diagonal(sims, 0)
    nn_sim = sims.max(axis=1)
    print("\n[line structure] mean nearest-line similarity:",
          round(float(nn_sim.mean()), 4))
    # контроль: случайные HV
    rnd = torch.sign(torch.randn(len(lines), 2048)).float()
    rs = (rnd @ rnd.T).numpy(); np.fill_diagonal(rs, 0)
    print("  random baseline:", round(float(rs.max(axis=1).mean()), 4))

    # ===== 3. Граф переходов EVA vs естественный язык =====
    def transition_stats(lines_):
        trans = {}
        for ln in lines_:
            for a, b in zip(ln, ln[1:]):
                trans.setdefault(a, set()).add(b)
        fanout = [len(v) for v in trans.values()]
        return len(trans), sum(fanout)/max(len(fanout),1)

    kt, kf = transition_stats(lines)
    rt, rf = transition_stats([p.lower().split()[:10] for p in ru_pairs[:100]])
    print(f"\n[transition graph] eva: {kt} узлов, fanout={kf:.2f} | "
          f"natural-ru: {rt} узлов, fanout={rf:.2f}")


if __name__ == "__main__":
    import json, re
    main()
