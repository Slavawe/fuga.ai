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


def load_pairs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ru, en = tokenize(d["ru"]), tokenize(d["en"])
            if ru and en:
                out.append((ru, en))
    return out


def main():
    random.seed(0)
    torch.manual_seed(0)

    pairs = load_pairs("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl")
    random.shuffle(pairs)
    cut = int(len(pairs) * 0.95)
    train, heldout = pairs[:cut], pairs[cut:]
    print(f"corpus: train={len(train)} heldout={len(heldout)}")

    # ===== БАЗОВЫЙ IBM =====
    ibm = fuga_core.IbmModel1()
    ibm.train(train, epochs=4)

    def word_hit(model, test, n=800):
        hits = tot = 0
        for ru, en in test[:n]:
            gold = set(en)
            for w in set(ru):
                top = model.translate_topk(w, 1)
                if not top or top[0][1] < 0.05:
                    continue
                tot += 1
                hits += int(top[0][0] in gold)
        return hits / max(tot, 1)

    base_hit = word_hit(ibm, heldout)
    print(f"[before self-play] word-hit@heldout = {base_hit:.4f}")

    # ===== РЕВЕРС-МОДЕЛЬ для round-trip проверки =====
    ibm_rev = fuga_core.IbmModel1()
    ibm_rev.train([(b, a) for a, b in train], epochs=3)

    # ===== Фильтр грамматичности EN-стороны =====
    flt = fuga_core.RustLinguisticFilter()
    lex_words, trans = [], []
    for _, en in train[:60000]:
        tt = tokenize(" ".join(en)); trans += list(zip(tt, tt[1:])); lex_words += tt
    flt.load_wiktionary_vocab(lex_words)
    flt.load_rucola_transitions(trans)

    binder = fuga_core.HybridBinder(2048)

    def structure_integrity(words):
        """VSA самопроверка: компонуем предложение -> развязываем по позициям ->
        доля слов, восстановившихся из собственного HV."""
        pk = np.asarray(binder.bind_batch([words]))
        hv = packed_to_torch(pk)
        sgn = torch.sign(hv).numpy() > 0
        packed_back = np.zeros((1, 32), dtype=np.uint64)
        for i in range(64):
            packed_back |= sgn[:, i::64].astype(np.uint64) << np.uint64(i)
        hits = tot = 0
        for pos in range(min(len(words), 6)):
            unb = np.asarray(binder.unbind_batch(packed_back, pos + 1))
            scores = np.asarray(binder.score_items(unb, words))[0]
            order = np.argsort(-scores)
            hit = any(words[order[k]] == words[pos] for k in range(min(2, len(order))))
            hits += int(hit); tot += 1
        return hits / max(tot, 1)

    # ===== САМОГЕНЕРАЦИЯ: собственные переводы модели на heldout-RU =====
    t0 = time.perf_counter()
    proposals = []
    stats = {"tried": 0, "filter_reject": 0, "integrity_reject": 0, "rt_reject": 0}
    for ru, _ in heldout[:3000]:
        if len(ru) < 2:
            continue
        gen_words = []
        for w in ru:
            cands = ibm.translate_topk(w, 5)
            cands = [(x, p) for x, p in cands if p > 0.02]
            if not cands:
                continue
            ps = np.array([p for _, p in cands]); ps /= ps.sum()
            idx = np.random.choice(len(cands), p=ps)
            gen_words.append(cands[idx][0])
        if len(gen_words) < 2:
            continue
        stats["tried"] += 1
        sent = " ".join(gen_words)
        if not flt.is_acceptable(sent, 0.7, 0.25):
            stats["filter_reject"] += 1
            continue
        if structure_integrity(gen_words) < 0.6:
            stats["integrity_reject"] += 1
            continue
        # round-trip: EN -> RU должно возвращаться к исходному смыслу
        back = []
        for w in gen_words:
            rc = ibm_rev.translate_topk(w, 1)
            if rc:
                back.append(rc[0][0])
        if not back:
            stats["rt_reject"] += 1
            continue
        hv_orig = packed_to_torch(np.asarray(binder.bind_batch([ru])))[0]
        hv_back = packed_to_torch(np.asarray(binder.bind_batch([back])))[0]
        rt_cos = float((hv_orig * hv_back).mean())
        if rt_cos < 0.15:
            stats["rt_reject"] += 1
            continue
        proposals.append({"ru": ru, "en_generated": gen_words,
                          "rt_consistency": rt_cos})
    dt = time.perf_counter() - t0
    acc_rate = len(proposals) / max(stats["tried"], 1)
    print(f"\n[self-generation] {dt:.1f}s: tried={stats['tried']} "
          f"accepted={len(proposals)} ({acc_rate*100:.1f}%)")
    print(f"  rejects: filter={stats['filter_reject']} "
          f"integrity={stats['integrity_reject']} roundtrip={stats['rt_reject']}")
    for pr in proposals[:5]:
        print(f"  пример: {' '.join(pr['ru'])} => {' '.join(pr['en_generated'])}"
              f"  [rt={pr['rt_consistency']:.2f}]")

    # ===== САМООБУЧЕНИЕ (coverage-targeted) =====
    # 1) строгий round-trip; 2) только пары с НОВЫМИ ru-словами (не подкреплять
    #    известное — иначе шум самоподкрепляется); 3) длина >= 3.
    known_ru = {w for ru, _ in train for w in ru}
    novel = []
    for p in proposals:
        if p["rt_consistency"] < 0.35 or len(p["ru"]) < 3:
            continue
        if any(w not in known_ru for w in p["ru"]):
            novel.append((p["ru"], p["en_generated"]))
    print(f"\n[coverage-targeted] отобрано {len(novel)} новых гипотез")
    aug = train + novel
    ibm2 = fuga_core.IbmModel1()
    ibm2.train(aug, epochs=4)
    new_hit = word_hit(ibm2, heldout)
    print(f"[after targeted self-play] word-hit@heldout = {new_hit:.4f} "
          f"(delta {new_hit-base_hit:+.4f}, +{len(novel)} псевдопар)")

    # ===== ВЕРБАЛИЗАЦИЯ СОБСТВЕННОЙ ПАМЯТИ (ConceptNet факты) =====
    print("\n[fact verbalization из собственной памяти]")
    facts_by_subj = {}
    with open("dataset_vault/02_world_concepts/conceptnet_sro_real.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["lang"] != "en" or d["relation"] not in ("usedfor", "capableof", "isa"):
                continue
            s = "_".join(d["subject"].lower().split())
            o = "_".join(d["object"].lower().split())
            if s and o and len(s.split()) <= 2 and len(o.split()) <= 2:
                facts_by_subj.setdefault((s, d["relation"]), o)
    templates = {
        "usedfor": lambda s, o: f"you can use {s} for {o.replace('_',' ')}",
        "capableof": lambda s, o: f"a {s.replace('_',' ')} can {o.replace('_',' ')}",
        "isa": lambda s, o: f"{s.replace('_',' ')} is a {o.replace('_',' ')}",
    }
    ART = {"a", "an", "the"}
    def clean_surface(words):
        out = []
        for w in words:
            if out and w in ART and out[-1] in ART | {"can", "is", "for"}:
                continue
            out.append(w)
        return out
    generated_facts = []
    items = list(facts_by_subj.items())
    random.shuffle(items)
    for (s, rel), o in items[:20000]:
        sent = templates[rel](s, o)
        sent = " ".join(clean_surface(sent.split()))
        if flt.is_acceptable(sent, 0.9, 0.35):
            generated_facts.append(sent)
        if len(generated_facts) >= 500:
            break
    print(f"  вербализовано и принято фильтром: {len(generated_facts)} предложений")
    for g in generated_facts[:5]:
        print(f"    \"{g}\"")


if __name__ == "__main__":
    main()
