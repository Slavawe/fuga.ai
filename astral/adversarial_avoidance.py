
"""Адверсариальный тест FiLM-избегания: форсированные клише в корпусе.

Протокол:
  1. Отравление: у доли триад в ответы ВНЕДРЯЕМ клише-фразы.
  2. Модель учится на отравленном корпусе -> начинает генерить клише.
  3. Включение β-сдвига избегания должно РЕЗАТЬ клише-генерации.
Метрика: доля генераций с клише (baseline vs avoidance).
"""

from __future__ import annotations

from __future__ import annotations

import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


random.seed(0)
torch.manual_seed(0)

from astral.film_avoidance import AvoidanceFiLM, CLICHE_BIGRAMS, steers_of_tail
from astral.surface_contrast import SurfaceDecoder, enc_batch, load_contrast

POISON_RATE = 0.4
POISON_PHRASES = [
    "к сожалению, ", "стоит отметить, ", "важно подчеркнуть, ",
    "в заключение, ", "надеюсь, это поможет, ", "как вы знаете, ",
]


def poison(text: str) -> str:
    """Внедряет случайную клише-фразу в конец ответа."""
    return text + " " + random.choice(POISON_PHRASES) + "это хороший пример"


def build_poisoned(rows, labels):
    """Отравленная копия: у POISON_RATE доля ответов клише-суффикс."""
    out_rows = []
    out_labels = []
    for (ctx, b), lab in zip(rows, labels):
        if random.random() < POISON_RATE:
            txt = bytes(b).decode("utf-8", "ignore")
            poisoned = poison(txt)
            pb = list(poisoned.encode("utf-8")[:96]) + [0]
            out_rows.append((ctx, pb))
        else:
            out_rows.append((ctx, b))
        out_labels.append(lab)
    return out_rows, out_labels


@torch.no_grad()
def generate(model, avoid, hv, reg_id, use_avoidance, temp=0.6, max_len=90):
    cond, fp = model.condition(hv, reg_id)
    h = model.init_state(cond, 320)
    ctx = torch.zeros(1, dtype=torch.long)
    out = bytearray()
    tail = []
    for _ in range(max_len):
        logits, h = model.step(ctx, h, cond, fp)
        if use_avoidance:
            steer = steers_of_tail(tail)
            h, _ = avoid.modulate(h, torch.tensor([steer]))
        probs = F.softmax(logits[0] / temp, dim=-1).numpy()
        nb = int(np.random.choice(256, p=probs))
        if nb == 0:
            break
        out.append(nb)
        tail.append(nb)
        if len(tail) > 40:
            tail.pop(0)
    return out.decode("utf-8", "ignore").lower()


def has_cliche(text: str) -> bool:
    return any(re.search(rf"\b{a}\b\s+\b{b}\b", text) for a, b in CLICHE_BIGRAMS)


def main():
    bal, labels = load_contrast(limit_per_class=300)
    cut = int(len(bal) * 0.85)
    tr_pairs, te_pairs = bal[:cut], bal[cut:]
    tr_l = labels[:cut]

    tr_rows = [(c[:16], b) for c, b in tr_pairs]
    te_rows = [(c[:16], b) for c, b in te_pairs]
    print(f"[adversarial] poison_rate={POISON_RATE}, "
          f"train={len(tr_rows)} test={len(te_rows)}")

    # ОТРАВЛЕННЫЙ корпус
    tr_rows_p, tr_l_p = build_poisoned(tr_rows, tr_l)
    print(f"отравлено ~{int(len(tr_rows_p)*POISON_RATE)} из {len(tr_rows_p)}")

    model = SurfaceDecoder(use_register=True, conditioning="film")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    B, n = 48, len(tr_rows_p)
    t0 = time.time()
    for step in range(801):
        idx = random.sample(range(n), min(B, n))
        hv_b = enc_batch([tr_rows_p[i][0] for i in idx])
        regs = torch.tensor([tr_l_p[i] for i in idx])
        cond, fp = model.condition(hv_b, regs)
        h = model.init_state(cond, 320)
        seqs = [tr_rows_p[i][1] for i in idx]
        maxL = min(max(len(s_) for s_ in seqs), 90)
        ctx_b = torch.zeros(len(idx), dtype=torch.long)
        ce_sum, n_tok = 0.0, 0
        for t in range(maxL):
            logits, h = model.step(ctx_b, h, cond, fp)
            tgt = torch.tensor([seqs_i[t] if t < len(seqs_i) else 0
                                for seqs_i in seqs])
            mask = torch.tensor([t < len(seqs_i) for seqs_i in seqs])
            if mask.any():
                ce_sum += F.cross_entropy(logits[mask], tgt[mask],
                                          reduction="sum")
                n_tok += int(mask.sum())
            ctx_b = tgt.clamp(min=0)
        loss = ce_sum / max(n_tok, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == 800:
            print(f"step {step}: ce={loss.item():.4f} ({time.time()-t0:.0f}s)")

    # генерация: baseline vs avoidance на отравленной модели
    binder = fuga_core.HybridBinder(2048)
    avoid = AvoidanceFiLM(hidden=320)
    n_test = min(len(te_rows), 100)
    c_no = c_yes = 0
    for i in range(n_test):
        pk = np.asarray(binder.bind_batch([te_rows[i][0]]))
        hv = packed_to_torch(pk)
        reg = random.choice([0, 1])
        g_no = generate(model, avoid, hv, reg, use_avoidance=False)
        g_yes = generate(model, avoid, hv, reg, use_avoidance=True)
        c_no += int(has_cliche(g_no))
        c_yes += int(has_cliche(g_yes))

    print("\n===== АДВЕРСАРИАЛЬНЫЙ РЕЗУЛЬТАТ =====")
    print(f"клише-генерации: baseline={c_no}/{n_test} "
          f"({c_no/max(n_test,1):.0%})")
    print(f"                 avoidance={c_yes}/{n_test} "
          f"({c_yes/max(n_test,1):.0%})")
    print(f"снижение: {100*(1 - c_yes/max(c_no,1)):.0f}%")

    # примеры
    print("\nпримеры (избегание выкл | вкл):")
    for i in range(min(3, n_test)):
        pk = np.asarray(binder.bind_batch([te_rows[i][0]]))
        hv = packed_to_torch(pk)
        print(f"  off: {generate(model, avoid, hv, 0, False)[:60]!r}")
        print(f"  on : {generate(model, avoid, hv, 0, True)[:60]!r}")


if __name__ == "__main__":
    import fuga_core
    from antitf.rust_bridge import packed_to_torch
    main()
