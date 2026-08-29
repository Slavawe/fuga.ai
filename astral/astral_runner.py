
"""Astral Runner: H-JEPA предиктор динамики 32K VSA-состояний среды.

Проверяемое утверждение: предиктор снижает ошибку предсказания S(t+1)
относительно бейзлайна (prev-state copy) на управляемой динамике.
"""

from __future__ import annotations


import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from astral.astral_env import ScaledAstralEnvironment


class JepaPredictor(nn.Module):
    def __init__(self, dim=32768, hidden=1024, action_emb=8):
        super().__init__()
        self.action_emb = nn.Embedding(16, action_emb)
        self.net = nn.Sequential(
            nn.Linear(dim + action_emb, hidden), nn.LayerNorm(hidden),
            nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, state_hv, action):
        dev = self.action_emb.weight.device
        state_hv = state_hv.to(dev)
        action = action.to(dev)
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
        self._hidden = hidden

    def forward(self, state_hv, action):
        dev = self.action_emb.weight.device      # устройство МОДЕЛИ (cuda/cpu)
        state_hv = state_hv.to(dev)              # среда отдаёт CPU-тензор
        action = action.to(dev)
        if state_hv.dim() == 1:
            state_hv = state_hv.unsqueeze(0)
        if action.dim() == 0:
            action = action.view(1)
        x = self.adapter(torch.cat([state_hv, self.action_emb(action)], dim=-1))
        # топ-2
        top_p, top_i = torch.topk(self.router(x), 2, dim=-1)
        alpha = torch.softmax(top_p, -1)
        out = torch.zeros(x.shape[0], self._hidden, device=dev)
        for e, expert in enumerate(self.experts):
            mask = (top_i == e)
            if not mask.any():
                continue
            rows, slots = mask.nonzero(as_tuple=True)
            out.index_add_(0, rows,
                           alpha[rows, slots].unsqueeze(-1) * expert(x[rows]))
        return torch.tanh(self.head(out))


def device_flag_has_cuda(args):
    return getattr(args, "device", "cpu") == "cuda"


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
    ap.add_argument("--acc-steps", type=int, default=1,
                    help="накопление градиентов (эффективный батч) — важно "
                         "для MoK при последовательной среде")
    ap.add_argument("--novelty-filter", action="store_true",
                    help="адаптивный surprise-фильтр: шаг оптимизатора только "
                         "на непредсказуемых состояниях")
    args = ap.parse_args()
    import random as _r
    _r.seed(0)
    torch.manual_seed(0)
    if device_flag_has_cuda(args):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    cfg = json.load(open(args.config))
    dim = cfg["vsa_dimension"]
    device = args.device
    print(f"[astral] {cfg['environment_name']}  VSA={dim} bit  device={device}")

    env = ScaledAstralEnvironment(vector_dim=dim)

    use_mok = device == "cuda" and "num_experts" in cfg.get("architecture", {})
    if use_mok:
        # ПРЕДЗАПУСКОВЫЙ КОНТРОЛЬ VRAM: до инстанцирования (иначе OOM-kill
        # убьёт процесс до печати предупреждения — так и случилось с 500M).
        n_experts = cfg["architecture"]["num_experts"]
        hidden = cfg["architecture"]["expert_config"]["hidden_dim"]
        n_layers = cfg["architecture"]["expert_config"]["num_layers"]
        adapter_p = dim * hidden * 5
        expert_p = n_experts * n_layers * hidden * hidden * 5
        head_p = hidden * dim
        pm = adapter_p + expert_p + head_p
        bytes_per = (2 if args.fp16 else 4) + 8   # веса + Adam(2x FP32)
        est = pm * bytes_per / 2**30
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_vram / 2**30
        except Exception:
            total_gb = 6.0
        print(f"  MoK-профиль: {pm/1e6:.0f}M параметров, "
              f"экспертов={n_experts}")
        print(f"  оценочная VRAM: {est:.1f}GB (доступно {total_gb:.1f}GB)")
        if est > total_gb * 0.85:
            print("  ❌ OOM невозможен: профиль не влезает в VRAM с Adam FP32.")
            print(f"     Решения: (1) astral/configs/astral_6gb_221m.json "
                  f"({int(pm/1e6*0.42)}M), (2) pip install bitsandbytes + "
                  f"AdamW8bit, (3) --fp16 (веса, но Adam остаётся FP32).")
            raise SystemExit(1)
        predictor = MoKPredictor(cfg, dim).to(device)
        pm = sum(p.numel() for p in predictor.parameters())
        bytes_per = (2 if args.fp16 else 4) + 8
        est_vram = pm * bytes_per / 2**30
        if args.fp16:
            predictor = predictor.half()
        print(f"  реальные параметры: {pm/1e6:.0f}M | "
              f"оценочная VRAM: {est_vram:.1f}GB")
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
    acc_steps = max(getattr(args, "acc_steps", 1), 1)
    opt.zero_grad()
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

        do_update = True
        if flt is not None:
            ok_f, _s = flt.should_ingest(pred.detach().cpu(), nxt_real["hv"].cpu())
            do_update = ok_f
        # накопление градиентов: эффективный батч acc_steps при последовательной
        # среде (один сэмпл/шаг) — иначе 222M не сойдутся (проверено: sqrt(2)-коллапс)
        if do_update:
            loss = ((pred - nxt_real["hv"].to(pred.device)) ** 2).mean() / acc_steps
            loss.backward()
            if (step + 1) % acc_steps == 0:
                opt.step()
                opt.zero_grad()
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
