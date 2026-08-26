
"""Register-Contrast A/B: baseline vs concat vs FiLM на строгом подкорпусе.

Данные: только триады с ЯВНЫМИ формами вежливости в ответе
(formal: вы/ваш/вам без сленга; casual: ты/твой/тебе или сленг),
сбалансированно. Метрика руля: P(вы-формы | formal-генерация) −
P(вы-формы | casual-генерация) и симметрично для ты-форм.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

import fuga_core as _fc_core

random.seed(0)
torch.manual_seed(0)

import fuga_core
from antitf.rust_bridge import packed_to_torch

REG = {"formal": 0, "casual": 1}
FORMAL = ("вы", "ваш", "вам", "ваше", "вами", "вашу")
CASUAL = ("ты", "твой", "твоя", "тебе", "тебя", "твоё")
SLANG = ("чё", "щас", "короче", "блин", "лол", "ваще", "норм", "круто",
         "жесть", "офигенн")


def load_contrast(limit_per_class=450):
    data_f, data_c = [], []
    with open("dataset_vault/04_pragmatic/pragmatic_triads.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["lang"] != "ru":
                continue
            t = (d["context"] + " " + d["response"]).lower()
            hf = any(re.search(rf"\b{w}\b", t) for w in FORMAL)
            hc = (any(re.search(rf"\b{w}\b", t) for w in CASUAL)
                  or any(s in t for s in SLANG))
            resp = d["response"].strip()
            b = list(resp.encode("utf-8")[:96]) + [0]
            ctx_w = [w.lower() for w in re.findall(r"[a-zа-яё]+",
                                                   d["context"].lower())][:16]
            if len(ctx_w) < 2 or len(b) < 6:
                continue
            if hf and not hc:
                data_f.append((ctx_w, b))
            elif hc and not hf:
                data_c.append((ctx_w, b))
    cap = min(len(data_f), len(data_c), limit_per_class)
    random.shuffle(data_f); random.shuffle(data_c)
    bal = data_f[:cap] + data_c[:cap]
    labels = [0] * cap + [1] * cap   # 0=formal-class, 1=casual-class
    print(f"contrast corpus: formal={len(data_f)} casual={len(data_c)} "
          f"-> balanced {len(bal)}")
    return bal, labels


class SurfaceDecoder(nn.Module):
    def __init__(self, dim=2048, hidden=320, use_register=True,
                 conditioning="concat"):
        super().__init__()
        self.use_register = use_register
        self.cond_type = conditioning if use_register else "none"
        self.byte_emb = nn.Embedding(256, 32)
        self.ctx_proj = nn.Linear(dim, 64)
        self.reg_emb = nn.Embedding(2, 16) if use_register else None
        reg_in = 16 if (use_register and conditioning == "concat") else 0
        self.gru = nn.GRUCell(32 + 64 + reg_in, hidden)
        if use_register and conditioning == "film":
            self.film = nn.Linear(16, hidden * 2)
        else:
            self.film = None
        self.head = nn.Linear(hidden, 256)

    def condition(self, hv, reg_id=None):
        if hv.dim() == 1:
            hv = hv.unsqueeze(0)
        c = self.ctx_proj(hv)
        parts = [c]
        b = c.shape[0]
        r = None
        if self.use_register:
            if reg_id is None:
                r = torch.zeros(b, dtype=torch.long, device=hv.device)
            elif isinstance(reg_id, int):
                r = torch.full((b,), reg_id, dtype=torch.long,
                               device=hv.device)
            else:
                r = reg_id.to(torch.long).reshape(-1)
                if r.shape[0] == 1 and b > 1:
                    r = r.expand(b)
            if self.cond_type == "concat":
                parts.append(self.reg_emb(r))
        cond = torch.cat(parts, dim=-1)
        film_params = None
        if self.film is not None and r is not None:
            film_params = self.film(self.reg_emb(r))
        return cond, film_params

    def init_state(self, cond, hidden):
        return torch.tanh(cond[:, :hidden]) * 0.5 \
            if cond.shape[1] >= hidden else \
            torch.tanh(F.pad(cond, (0, hidden - cond.shape[1]))) * 0.5

    def step(self, byte_idx, h, cond, film_params=None):
        if byte_idx.dim() == 0:
            byte_idx = byte_idx.view(1)
        x = torch.cat([self.byte_emb(byte_idx), cond], dim=-1)
        h = self.gru(x, h)
        if film_params is not None:
            gamma, beta = film_params.chunk(2, dim=-1)
            h = torch.sigmoid(gamma) * h + beta
        return self.head(h), h


@torch.no_grad()
def generate(model, hv, reg_id, temp=0.7, max_len=90):
    cond, fp = model.condition(hv, reg_id)
    h = model.init_state(cond, 320)
    ctx = torch.zeros(1, dtype=torch.long)
    out = bytearray()
    for _ in range(max_len):
        logits, h = model.step(ctx, h, cond, fp)
        probs = F.softmax(logits[0] / temp, dim=-1).numpy()
        nb = int(np.random.choice(256, p=probs))
        if nb == 0:
            break
        out.append(nb)
        ctx = torch.tensor([nb])
    return out.decode("utf-8", "ignore").lower()


def run_model(tag, use_register, conditioning, tr_rows, tr_labels,
              steps=900):
    model = SurfaceDecoder(use_register=use_register,
                           conditioning=conditioning)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    def enc(rows):
        return torch.stack([
            packed_to_torch(np.asarray(
                fuga_core.HybridBinder(2048).bind_batch([w])))[0]
            for w, _, _ in [(r[0], None, None) for r in rows]])
    # кодируем один раз глобальным байндером снаружи — здесь заглушка не нужна

    B = 48
    n = len(tr_rows)
    t0 = time.time()
    for step in range(steps + 1):
        idx = random.sample(range(n), min(B, n))
        hv_b = enc_batch([tr_rows[i][0] for i in idx])
        regs = torch.tensor([tr_labels[i] for i in idx])
        cond, fp = (model.condition(hv_b, regs)
                    if model.use_register else model.condition(hv_b))
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
        if step % 300 == 0 or step == steps:
            print(f"  [{tag}] step {step}: ce={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)")
    return model


_ENC_BINDER = None


def enc_batch(ctx_word_rows):
    global _ENC_BINDER
    if _ENC_BINDER is None:
        import fuga_core
        _ENC_BINDER = fuga_core.HybridBinder(2048)
    pk = np.asarray(_ENC_BINDER.bind_batch(ctx_word_rows))
    return packed_to_torch(pk)


def main():
    bal, labels = load_contrast()
    cut = int(len(bal) * 0.85)
    tr_pairs, te_pairs = bal[:cut], bal[cut:]
    tr_l, te_l = labels[:cut], labels[cut:]
    # слова контекста для энкодера
    tr_rows = [(c[:16], b) for c, b in tr_pairs]
    te_rows = [(c[:16], b) for c, b in te_pairs]

    import time as _t

    results = {}
    for tag, ureg, cond_t in (("baseline", False, "concat"),
                              ("concat", True, "concat"),
                              ("film", True, "film")):
        print(f"\n=== model: {tag} ===")
        model = run_model(tag, ureg, cond_t, tr_rows, tr_l)
        # генерации под двумя регистрами на одних и тех же контекстах
        n_test = min(len(te_rows), 120)
        f_vy = f_ty = c_vy = c_ty = 0
        with torch.no_grad():
            for i in range(n_test):
                hv = packed_to_torch(np.asarray(_ENC_BINDER_HV(te_rows[i][0])))
                gf = generate(model, hv, REG["formal"])
                gc = generate(model, REG["casual"] and hv, REG["casual"])
                f_vy += int(any(w in gf for w in ("вы", "ваш", "вам")))
                c_vy += int(any(w in gc for w in ("вы", "ваш", "вам")))
                f_ty += int(any(w in gf for w in ("ты", "тебе", "тво")))
                c_ty += int(any(w in gc for w in ("ты", "тебе", "тво")))
        steer_vy = (f_vy - c_vy) / n_test
        steer_ty = (c_ty - f_ty) / n_test
        results[tag] = {"vy@formal": f_vy/n_test, "vy@casual": c_vy/n_test,
                        "steer_vy": steer_vy, "steer_ty": steer_ty}
        print(f"  [{tag}] vy@formal={f_vy/n_test:.2f} vy@casual={c_vy/n_test:.2f}"
              f" -> steering(vy)={steer_vy:+.2f}")

    print("\n===== ИТОГ =====")
    for tag, r in results.items():
        print(f"{tag:9} steering(vy)={r['steer_vy']:+.3f}")


def _ENC_BINDER_HV(ctx_words):
    global _ENC_BINDER
    if _ENC_BINDER is None:
        import fuga_core
        _ENC_BINDER = fuga_core.HybridBinder(2048)
    return np.asarray(_ENC_BINDER.bind_batch([ctx_words]))   # [1, W]


if __name__ == "__main__":
    main()
