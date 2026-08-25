
"""Трёхъязычный потоковый инжест (EN/ZH/RU): HF streaming -> Rust VSA core.

Каналы проверены на доступность:
  EN: FineWeb-Edu sample-10BT | ZH: wikimedia/wikipedia zh | RU: Gazeta
(The Stack v2 и COIG закрыты/мертвы в datasets 5.x — см. SESSION_LOG.)

Словесные HV — через HybridBinder (детерминированные атомы), горячие
XOR/rot — через нативный FastVSA.
"""

from __future__ import annotations

from __future__ import annotations

import itertools
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from antitf.rust_bridge import packed_to_torch
from astral.unified_front import PerModalNoveltyFilter
from fuga_core import FastVSA


def words_of(text: str, lang: str, n=32) -> list[str]:
    if lang == "zh":
        return [c for c in text if "\u4e00" <= c <= "\u9fff"][:n]
    return [w.lower() for w in re.findall(r"[a-zа-яё]+", text.lower())][:n]


class TrilingualStreamer:
    FIELDS = {"en": ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, "text"),
              "zh": ("wikimedia/wikipedia", {"name": "20231101.zh"}, "text"),
              "ru": ("IlyaGusev/gazeta", {}, "text")}

    def __init__(self):
        from datasets import load_dataset
        self.iters = {}
        for lang, (repo, kwargs, field) in self.FIELDS.items():
            ds = load_dataset(repo, split="train", streaming=True, **kwargs)
            self.iters[lang] = iter(ds)
            self.field = field

    def next_text(self, lang):
        row = next(self.iters[lang], None)
        if row is None:
            return None
        return row.get(self.field, "")


def main(max_per_lang: int = 40):
    random.seed(0)
    binder = fuga_core.HybridBinder(32768)
    vsa = FastVSA(32768)
    flt = PerModalNoveltyFilter(alpha=0.05, threshold=0.85)

    streamer = TrilingualStreamer()
    langs = ["en", "zh", "ru"]
    stats = {k: [0, 0] for k in langs}
    n = 0
    t0 = time.time()
    rr = itertools.cycle(langs)
    while n < max_per_lang * len(langs):
        lang = next(rr)
        txt = streamer.next_text(lang)
        if not txt:
            continue
        toks = words_of(txt, lang)
        if len(toks) < 4:
            continue

        # строка -> HV: XOR-цепочка слов на нативном Rust-ядре (packed u64)
        packed_words = [np.asarray(binder.bind_batch([[t]]))[0] for t in toks]
        acc = np.asarray(vsa.bind(packed_words[0], packed_words[1]))
        for i in range(2, len(packed_words)):
            acc = np.asarray(vsa.bind(acc, packed_words[i]))
        # предиктор-заглушка для фильтра: ротация состояния
        pred = np.asarray(vsa.rotate(acc, 5))
        diff_bits = np.unpackbits(np.bitwise_xor(
            pred.view(np.uint8), acc.view(np.uint8)))
        surprise_rel = float(diff_bits.mean())
        ok = flt.filter(f"lang_{lang}", surprise_rel)
        stats[lang][0] += 1
        stats[lang][1] += int(ok)
        n += 1

    dt = time.time() - t0
    print(f"[trilingual] {n} документов за {dt:.1f}s ({n/dt:.1f} docs/s)")
    for k in sorted(stats):
        seen, passed = stats[k]
        print(f"  {k}: seen={seen} pass={passed} ({passed/max(seen,1):.0%})")


def dim_of(x):
    return x.shape[-1] * 64


if __name__ == "__main__":
    main()
