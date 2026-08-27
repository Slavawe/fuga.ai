
"""FiLM + VSA-избегание клише: детектор штампа -> латентный отрицательный
β-сдвиг скрытого состояния декодера.

Измерение: частота клише-биграмм в генерациях с активным избеганием
против baseline. Плюс live-визуализация γ/β (latent_vis).
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

sys.path.insert(0, ".")

random.seed(0)
torch.manual_seed(0)

import fuga_core
from antitf.rust_bridge import packed_to_torch
from astral.surface_contrast import SurfaceDecoder, enc_batch, load_contrast

CLICHE_BIGRAMS = [
    ("к", "сожалению"), ("стоит", "отметить"), ("надеюсь", "это"),
    ("если", "вопросы"), ("важно", "отметить"), ("в", "заключение"),
]


class AvoidanceFiLM(nn.Module):
    """Регистровый FiLM + символьный детектор клише -> β_avoid."""

    def __init__(self, hidden=320, cliche_weight=2.0):
        super().__init__()
        self.hidden = hidden
        self.cliche_weight = cliche_weight
        # проекция одного байта/токена в детекторные признаки — нет, детектор
        # работает на тексте; модуль лишь применяет β-сдвиг по сигналу.
        self.register_buffer("_x", torch.zeros(1))

    def modulate(self, h, steer):
        """steer: [B] в [0,1] — насколько текущий хвост похож на клише.
        Возвращает h, (gamma, beta) для визуализации."""
        gamma = torch.ones_like(h)
        beta = -self.cliche_weight * steer.unsqueeze(-1) * torch.ones_like(h)
        return gamma * h + beta, (gamma.detach(), beta.detach())


def steers_of_tail(tail: list[int], vocab_bytes=None) -> float:
    """Доля клише-биграмм, встречающихся в хвосте генерации (0..1)."""
    try:
        text = bytes(tail).decode("utf-8", "ignore").lower()
    except Exception:
        return 0.0
    hits = sum(1 for a, b in CLICHE_BIGRAMS
               if re.search(rf"\b{a}\b\s+\b{b}\b", text))
    return min(hits / len(CLICHE_BIGRAMS), 1.0)


@torch.no_grad()
def generate(model, avoid, hv, reg_id, use_avoidance, tail_len=40, temp=0.7,
             max_len=90):
    cond, fp = model.condition(hv, reg_id)
    h = model.init_state(cond, 320)
    ctx = torch.zeros(1, dtype=torch.long)
    out = bytearray()
    tail = []
    trace = []   # (gamma_std, beta_mean) для визуализации
    for _ in range(max_len):
        logits, h = model.step(ctx, h, cond, fp)
        if use_avoidance:
            steer = steers_of_tail(tail)
            h, (g, b) = avoid.modulate(h, torch.tensor([steer]))
            trace.append((float(g.std()), float(b.mean())))
        probs = F.softmax(logits[0] / temp, dim=-1).numpy()
        nb = int(np.random.choice(256, p=probs))
        if nb == 0:
            break
        out.append(nb)
        tail.append(nb)
        if len(tail) > tail_len:
            tail.pop(0)
    return out.decode("utf-8", "ignore").lower(), trace


def main():
    bal, labels = load_contrast()
    cut = int(len(bal) * 0.9)
    tr_pairs = bal[:cut]
    tr_l = labels[:cut]
    tr_rows = [(c[:16], b) for c, b in tr_pairs]

    model = SurfaceDecoder(use_register=True, conditioning="film")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    B, n = 48, len(tr_rows)
    t0 = time.time()
    for step in range(701):
        idx = random.sample(range(n), min(B, n))
        hv_b = enc_batch([tr_rows[i][0] for i in idx])
        regs = torch.tensor([tr_l[i] for i in idx])
        cond, fp = model.condition(hv_b, regs)
        h = model.init_state(cond, 320)
        seqs = [tr_rows[i][1] for i in idx]
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
        if step % 300 == 0 or step == 700:
            print(f"step {step}: ce={loss.item():.4f} ({time.time()-t0:.0f}s)")

    # ===== замер клише: с избеганием vs без =====
    binder = fuga_core.HybridBinder(2048)
    avoid = AvoidanceFiLM(hidden=320)
    test_rows = bal[cut:]
    n_test = min(len(test_rows), 100)
    cliche_no = cliche_yes = 0
    sample_trace = None
    for i in range(n_test):
        ctx_w = test_rows[i][0][:16]
        pk = np.asarray(binder.bind_batch([ctx_w]))
        hv = packed_to_torch(pk)
        reg = random.choice([0, 1])
        g_no, _ = generate(model, avoid, hv, reg, use_avoidance=False)
        g_yes, trace = generate(model, avoid, hv, reg, use_avoidance=True)
        if sample_trace is None:
            sample_trace = trace
        cliche_no += int(any(
            re.search(rf"\b{a}\b\s+\b{b}\b", g_no)
            for a, b in CLICHE_BIGRAMS))
        cliche_yes += int(any(
            re.search(rf"\b{a}\b\s+\b{b}\b", g_yes)
            for a, b in CLICHE_BIGRAMS))

    print("\n===== ИЗБЕГАНИЕ КЛИШЕ =====")
    print(f"клише-генерации: без избегания={cliche_no}/{n_test} "
          f"({cliche_no/max(n_test,1):.2%})")
    print(f"                 с избеганием ={cliche_yes}/{n_test} "
          f"({cliche_yes/max(n_test,1):.2%})")

    # ===== live-трасса γ/β (latent_vis) =====
    print("\n[latent_vis] трасса FiLM на 10 шагах генерации (γ_std | β_mean):")
    if sample_trace:
        for i, (gs, bm) in enumerate(sample_trace[:10]):
            print(f"  step {i}: γ_std={gs:.4f}  β_mean={bm:+.4f}")


if __name__ == "__main__":
    main()
