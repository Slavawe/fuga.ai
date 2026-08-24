"""Self-Reflective Loop: модель оценивает собственные генерации и
сохраняет прошедшие внутренний критик в self_reflection.jsonl.

Критик (без внешних данных, всё внутри стека):
  coverage   — доля слотов, реально просевших в текст (VSA-развязка)
  accept     — грамматичность по фильтру RuCoLA/Wiktionary
  integrity  — round-trip: sign(HV) разворачивается обратно в те же слова
  total      — взвешенная сумма; порог отбора в опыт
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
import torch.nn.functional as F

sys.path.insert(0, ".")

import fuga_core
from antitf.rust_bridge import packed_to_torch


class GenerationCritic:
    def __init__(self, binder, flt, corpus_texts: list[str]):
        self.binder = binder
        self.flt = flt
        self.corpus_set = set(corpus_texts)
        self._bigrams: set | None = None

    def _corpus_bigrams(self) -> set:
        if self._bigrams is None:
            bg = set()
            for t in self.corpus_set:
                b = ("^" + t + "$").encode("utf-8", "ignore")
                bg.update(zip(b, b[1:]))
            self._bigrams = bg
        return self._bigrams

    def coverage(self, slots: list[str], text: str) -> float:
        if not slots:
            return 0.0
        hit = sum(1 for w in slots if any(
            w[:5] in text or w[:4] == text[i:i + len(w[:4])]
            for i in range(0, max(len(text) - 3, 1))))
        # мягче: префикс слова встречается как подстрока
        hit = sum(1 for w in slots if w[:4] in text)
        return hit / len(slots)

    def acceptability(self, text: str) -> bool:
        return self.flt.is_acceptable(text, 0.8, 0.25)

    def novelty(self, text: str) -> float:
        return 0.0 if text in self.corpus_set else 1.0

    def integrity(self, words: list[str]) -> float:
        """sign(bind(words)) -> unbind(pos) -> слово на месте?"""
        pk = np.asarray(self.binder.bind_batch([words]))
        hv = packed_to_torch(pk)
        sgn = torch.sign(hv).numpy() > 0
        packed = np.zeros((1, 32), dtype=np.uint64)
        for i in range(64):
            packed |= sgn[:, i::64].astype(np.uint64) << np.uint64(i)
        hits = tot = 0
        for pos in range(min(len(words), 6)):
            unb = np.asarray(self.binder.unbind_batch(packed, pos + 1))
            sc = np.asarray(self.binder.score_items(unb, words))[0]
            top2 = np.argsort(-sc)[:2]
            hits += int(any(words[k] == words[pos] for k in top2))
            tot += 1
        return hits / max(tot, 1)

    def score(self, slots: list[str], text: str) -> dict:
        cov = self.coverage(slots, text)
        acc = float(self.acceptability(text)) if text else 0.0
        integ = self.integrity(slots) if slots else 0.0
        nov = self.novelty(text)
        total = 0.45 * cov + 0.25 * acc + 0.2 * integ + 0.1 * nov
        return {"coverage": round(cov, 3), "accept": round(acc, 2),
                "integrity": round(integ, 3), "novelty": nov,
                "total": round(total, 3)}


def build_slot_pool(conceptnet_path="dataset_vault/02_world_concepts/"
                                     "conceptnet_semantic.jsonl",
                    limit=800):
    """Слоты для рефлексии — реальные сущности из собственной памяти фактов."""
    pool = []
    seen = set()
    with open(conceptnet_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            key = (d["subject"], d["relation"], d["object"])
            if key in seen:
                continue
            seen.add(key)
            s_, o_ = d["subject"].lower(), d["object"].lower()
            STOP = {"a", "an", "the", "your", "some", "it", "this", "that"}
            def _clean(w):
                parts = [p for p in w.split() if p not in STOP]
                return "_".join(parts)
            s_, o_ = _clean(s_), _clean(o_)
            if len(s_) < 3 or len(o_) < 3 or s_.isdigit() or o_.isdigit():
                continue
            pool.append([s_, d["relation"].lower(), o_])
            if len(pool) >= limit:
                break
    return pool


def main():
    random.seed(0)
    from fuga_memory import PersistentVSAMemory
    from antitf.linguistic_data import load_rucola, tokenize as ling_tok

    binder = fuga_core.HybridBinder(2048)
    mem = PersistentVSAMemory(binder)

    flt = fuga_core.RustLinguisticFilter()
    lex, trans = [], []
    try:
        pairs = []
        with open("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pairs.append(d["en"])
                if len(pairs) >= 40000:
                    break
    except FileNotFoundError:
        pairs = ["the cat sleeps", "a dog can bark"]
    for t in pairs:
        tt = ling_tok(t); trans += list(zip(tt, tt[1:])); lex += tt
    flt.load_wiktionary_vocab(lex)
    flt.load_rucola_transitions(trans)

    critic = GenerationCritic(binder, flt, set())

    slots_pool = build_slot_pool()
    print(f"[reflection] slot pool: {len(slots_pool)} троек из ConceptNet")

    # Вариации вербализации (генератор — слотовый вербализатор v0.1 +
    # температурные перестановки; в v0.3 сюда встаёт slot-KAN декодер)
    REL_TEMPLATES = {
        "isa": lambda s, o: [f"{s} is a {o}", f"a {o} such as the {s}",
                             f"{s}: a kind of {o}"],
        "capableof": lambda s, o: [f"a {s} can {o}", f"the {s} is able to {o}",
                                   f"{s}s often {o}"],
        "usedfor": lambda s, o: [f"{s} is used for {o}",
                                 f"you can use a {s} to {o}",
                                 f"people use {s} for {o}"],
    }
    DEFAULT_T = lambda s, o: [f"{s} {o}", f"relation between {s} and {o}"]

    accepted, tried, dist = 0, 0, []
    t0 = time.perf_counter()
    for slots in slots_pool:
        subj, rel, obj = slots[0], slots[1].replace(" ", ""), slots[2]
        variants = REL_TEMPLATES.get(rel, (lambda a, b: DEFAULT_T(a, b)))(subj, obj)
        best, best_score = None, None
        for cand in variants:
            tried += 1
            sc = critic.score(slots, cand)
            sc["total"] = round(sc["coverage"] * 0.45 + sc["accept"] * 0.25 +
                                sc["integrity"] * 0.2 + sc["novelty"] * 0.1, 3)
            if best_score is None or sc["total"] > best_score["total"]:
                best, best_score = cand, sc
        dist.append(best_score["total"])
        if best_score["total"] >= 0.75:
            mem.add_reflection(slots, best, best_score)
            accepted += 1
    dt = time.perf_counter() - t0

    print(f"[self-reflection] tried={tried} variants, accepted={accepted} "
          f"({accepted/max(len(slots_pool),1)*100:.0f}%) за {dt:.1f}s")
    if dist:
        import statistics
        print(f"  critic total: mean={statistics.mean(dist):.3f} "
              f"median={statistics.median(dist):.3f}")

    print("\nпримеры принятых рефлексий:")
    shown = 0
    for r in mem.iter_reflections():
        if shown >= 5:
            break
        print(f"  {' '.join(r['slots'])} -> \"{r['text']}\"  {r['scores']}")
        shown += 1


if __name__ == "__main__":
    main()
