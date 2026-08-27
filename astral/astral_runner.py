
"""Astral Runner: H-JEPA предиктор динамики 32K VSA-состояний среды.

Проверяемое утверждение: предиктор снижает ошибку предсказания S(t+1)
относительно бейзлайна (prev-state copy) на управляемой динамике.
"""

from __future__ import annotations


import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astral.astral_env import ScaledAstralEnvironment


class JepaPredictor(nn.Module):
    def __init__(self, dim=32768, hidden=1024, action_emb=8):
        super().__init__()
        self.action_emb = nn.Embedding(16, action_emb)
        self.net = nn.Sequential(
            nn.Linear(dim + action_emb, hidden), nn.LayerNorm(hidden),
            nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, state_hv, action):
        if state_hv.dim() == 1:
            state_hv = state_hv.unsqueeze(0)
        if action.dim() == 0:
            action = action.view(1)
        a = self.action_emb(action)
        return torch.tanh(self.net(torch.cat([state_hv, a], dim=-1)))


class MoKPredictor(nn.Module):
    """MoK-топология из конфига: адаптер 32768->896 -> эксперты -> выход 896->32768."""

    def __init__(self, cfg, dim, action_emb=8):
        super().__init__()
        try:
            a = cfg["architecture"]
            n_e = a["num_experts"]
            hidden = a["expert_config"]["hidden_dim"]
            layers = a["expert_config"]["num_layers"]
        except (KeyError, TypeError):
            raise ValueError("config needs architecture.num_experts")
        self.action_emb = nn.Embedding(16, action_emb)
        # общий VSA-адаптер
        from antitf.kan import ChebyKANLayer
        self.adapter = ChebyKANLayer(dim + action_emb, hidden, degree=4)
        # эксперты (только внутренние hidden->hidden)
        self.experts = nn.ModuleList([
            nn.Sequential(*[ChebyKANLayer(hidden, hidden, degree=4)
                            for _ in range(layers)])
            for _ in range(n_e)])
        # общий выходной слой
        self.head = nn.Linear(hidden, dim)
        self.router = nn.Linear(hidden, n_e)
        self._n_e = n_e

    def forward(self, state_hv, action):
        if state_hv.dim() == 1:
            state_hv = state_hv.unsqueeze(0)
        x = self.adapter(torch.cat([state_hv, self.action_emb(action)], dim=-1))
        # топ-2
        top_p, top_i = torch.topk(self.router(x), 2, dim=-1)
        alpha = torch.softmax(top_p, -1)
        out = torch.zeros(x.shape[0], self.head.out_features)
        for e, expert in enumerate(self.experts):
            mask = (top_i == e)
            if not mask.any():
                continue
            rows, slots = mask.nonzero(as_tuple=True)
            out.index_add_(0, rows,
                           alpha[rows, slots].unsqueeze(-1) * expert(x[rows]))
        return torch.tanh(self.head(out))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="astral/configs/astral_scaled.json")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--profile-vram", action="store_true",
                    help="печатать потребление VRAM на каждом логе (CUDA)")
    ap.add_argument("--fp16", action="store_true",
                    help="веса в FP16 (половина VRAM); Adam остаётся FP32")
    ap.add_argument("--novelty-filter", action="store_true",
                    help="адаптивный surprise-фильтр: шаг оптимизатора только "
                         "на непредсказуемых состояниях")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    dim = cfg["vsa_dimension"]
    device = args.device
    print(f"[astral] {cfg['environment_name']}  VSA={dim} bit  device={device}")

    env = ScaledAstralEnvironment(vector_dim=dim)

    use_mok = device == "cuda" and "num_experts" in cfg.get("architecture", {})
    if use_mok:
        predictor = MoKPredictor(cfg, dim).to(device)
        pm = sum(p.numel() for p in predictor.parameters())
        print(f"  MoK-профиль: {pm/1e6:.0f}M параметров, "
              f"экспертов={predictor._n_e}")
        est_vram = pm * 4 * 3 / 2**30    # веса FP32 + Adam x2
        if args.fp16:
            predictor = predictor.half()
            est_vram = pm * 2 * 3 / 2**30  # FP16 веса + Adam
        print(f"  оценочная VRAM: {est_vram:.1f}GB (всего 6GB)")
        if est_vram > 5.5:
            print("  ⚠️  риск OOM: уменьшите экспертов/hidden или включите "
                  "bitsandbytes AdamW8bit")
    else:
        predictor = JepaPredictor(dim).to(device)
        pm = sum(p.numel() for p in predictor.parameters())
        if device == "cpu" and "num_experts" in cfg.get("architecture", {}):
            arch = cfg.get("architecture", {})
            target = arch.get("total_parameters_approx") or \
                f"{arch.get('total_parameters_M', '?')}M"
            print(f"  CPU-фолбэк: JepaPredictor ({pm/1e6:.0f}M парам.) — "
                  f"MoK-профиль {target} требует --device cuda")
        else:
            print(f"  параметры: {pm/1e6:.0f}M")

    flt = None
    if args.novelty_filter:
        from astral.data_filter import AstralDataStreamFilter
        flt = AstralDataStreamFilter(novelty_threshold=0.35, adaptive=True)
        print("[filter] adaptive novelty: optimizer.step() только при surprise "
              ">= EMA*1.25")
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # старт: реальный кадр из COCO (если докачан) или синтетика
    state = env.get_state()

    base_errs, pred_errs = [], []
    state['hv'] = state['hv'].flatten()
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        action = random.randint(0, 7)
        nxt_real = env.step_action(state["hv"], action)

        h_prev = state["hv"].detach()
        pred = predictor(h_prev, torch.tensor([action]))
        real_cpu = nxt_real["hv"].cpu()
        h_prev_cpu = h_prev.cpu()
        err_pred = float((pred.detach().cpu() - real_cpu).norm() / (real_cpu.norm() + 1e-9))
        err_base = float((h_prev_cpu - real_cpu).norm() / (real_cpu.norm() + 1e-9))
        pred_errs.append(err_pred)
        base_errs.append(err_base)

        loss = err_pred + 0.05 * torch.relu(1.0 - pred.std()).mean() \
            if isinstance(err_pred, float) else None
        # тензорный путь:
        pred_t = predictor(h_prev.view(1, -1).to(device),
                           torch.tensor([action], device=device))
        do_update = True
        if flt is not None:
            ok_f, _s = flt.should_ingest(pred_t.detach().cpu(), nxt_real["hv"].cpu())
            do_update = ok_f
        loss = ((pred_t - nxt_real["hv"].to(device)) ** 2).mean()
        if do_update:
            opt.zero_grad(); loss.backward(); opt.step()
        state = nxt_real

        if step % 100 == 0 or step == args.steps:
            if device == "cuda" and args.profile_vram:
                print(f"  VRAM allocated: "
                      f"{torch.cuda.memory_allocated()/2**20:.0f} MB")
            w = 50
            bp = np.mean(pred_errs[-w:]) if pred_errs else 0
            bb = np.mean(base_errs[-w:]) if base_errs else 0
            print(f"step {step}: pred_rel_err={bp:.4f} baseline={bb:.4f} "
                  f"improvement={(bb-bp)/max(bb,1e-9)*100:.1f}%")

    print("\n===== ASTRAL VALIDATION =====")
    tail_p = np.mean(pred_errs[-100:])
    tail_b = np.mean(base_errs[-100:])
    print(f"pred rel err: {tail_p:.4f} | copy-baseline: {tail_b:.4f} | "
          f"better by {(tail_b-tail_p)/max(tail_b,1e-9)*100:.1f}%")
    print(f"throughput: {args.steps/(time.perf_counter()-t0):.1f} steps/s @ {dim} bit")


if __name__ == "__main__":
    main()
