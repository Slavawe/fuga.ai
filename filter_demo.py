from __future__ import annotations

import random
import sys
import time

sys.path.insert(0, ".")

import fuga_core

from antitf.data_i18n import load_tatoeba_pairs
from antitf.item_memory import SimpleWordVocab, VSAItemMemory
from antitf.linguistic_data import build_filter, load_rucola


def main() -> None:
    random.seed(0)
    pairs = load_tatoeba_pairs(max_pairs=20000)
    ru_texts = [p[0] for p in pairs]
    flt = build_filter(fuga_core, ru_texts)
    print(f"lexicon={flt.vocab_size()}  transitions={flt.transitions_size()}")

    # Тестовые наборы: реальные ok / реальные неликвидные / word-salad.
    dev = load_rucola("in_domain_dev") + load_rucola("out_of_domain_dev")
    valid = [s for s, a in dev if a == 1]
    invalid = [s for s, a in dev if a == 0]
    salads = flt.make_word_salad_negatives(valid[: len(invalid)], n_shuffles=1)

    print(f"\nsets: valid={len(valid)} invalid(rucola)={len(invalid)} salad={len(salads)}")

    for name, vocab_cov, trans_cov in (("strict", 1.0, 1.0), ("soft", 0.85, 0.45)):
        t0 = time.perf_counter()
        ok_v, _ = flt.filter_batch(valid, vocab_cov, trans_cov)
        ok_i, _ = flt.filter_batch(invalid, vocab_cov, trans_cov)
        ok_s, _ = flt.filter_batch(salads, vocab_cov, trans_cov)
        dt = (time.perf_counter() - t0) * 1000

        acc_v = len(ok_v) / max(len(valid), 1)
        rej_i = 1.0 - len(ok_i) / max(len(invalid), 1)
        rej_s = 1.0 - len(ok_s) / max(len(salads), 1)
        # Сбалансированная точность бинарной классификации ok vs not-ok.
        bal = 0.5 * (acc_v + (rej_i + rej_s) / 2.0)
        print(f"[{name}] accept_valid={acc_v:.3f} reject_rucola_bad={rej_i:.3f} "
              f"reject_salad={rej_s:.3f} | balanced_acc={bal:.3f} | {dt:.1f}ms "
              f"({len(valid)+len(invalid)+len(salads)} текстов)")

    # Интеграция с пайплайном: filter -> bind_batch только валидных.
    binder = fuga_core.HybridBinder(2048)
    batch = valid[:2000] + salads[:500]
    ok_batch, bad_batch = flt.filter_batch(batch, 0.85, 0.45)
    hv = binder.bind_batch([w.split() for w in ok_batch])
    print(f"\npipeline: {len(batch)} -> bind {len(ok_batch)} valid "
          f"(rejected {len(bad_batch)} как negatives), hv={hv.shape}")

    mem = VSAItemMemory(vocab_size=flt.vocab_size() + 10, hyper_dim=2048)
    print(f"python item-memory готов к приёму тех же атомов")


if __name__ == "__main__":
    main()
