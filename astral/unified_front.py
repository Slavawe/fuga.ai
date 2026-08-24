
"""Unified Training Front: все каналы данных -> один MoK-контур обучения.

Микшер модальностей 40% физика / 40% язык+код / 20% видео-динамика.
Фильтр новизны решает, тратить ли шаг оптимизатора.
Профили: 'sandbox' (CPU, здесь) и конфиг astral_1b_mok.json (GPU).
"""

import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fuga_core
from antitf.rust_bridge import packed_to_torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antitf.kan import ChebyKANLayer
from astral.procedural_stream import ProceduralWorldGen
from astral.data_filter import AstralDataStreamFilter


class StreamMixer:
    """Интерливинг каналов с заданными пропорциями."""

    def __init__(self, binder, text_limit: int | None = None):
        self.binder = binder
        self.proc = ProceduralWorldGen(vsa_dim=2048, n_basis=64)
        self.rng = random.Random(7)

        from astral.mega_streamer import MegaDataStreamer
        self.mega = MegaDataStreamer(max_text=text_limit)
        self.text_iter = iter(self.mega._text_stream())
        self.code_iter = self.mega.code_bytes()

    def next_sample(self) -> tuple[str, torch.Tensor]:
        roll = self.rng.random()
        if roll < 0.40:                                   # физика/алгебра
            d = self.proc.generate_step()
            return "physics", self.proc.to_bipolar_torch(d["state_next"])
        if roll < 0.80:                                   # язык
            try:
                item = next(self.text_iter)
            except StopIteration:
                return "physics", self.proc.generate_step()["state_next"]
            words = [w for w in re.findall(r"[a-z]+",
                    item["bytes"].decode("latin-1", "ignore").lower())][:32]
            if len(words) < 4:
                return "lang", torch.ones(128)
            pk = np.asarray(self.binder.bind_batch([words]))
            return "lang", packed_to_torch(pk)[0]
        # код
        try:
            item = next(self.code_iter)
        except StopIteration:
            return "physics", self.proc.generate_step()["state_next"]
        words = [w for w in re.findall(r"[A-Za-z]+",
                item["bytes"].decode("latin-1", "ignore").lower())][:32]
        pk = np.asarray(self.binder.bind_batch([words]))
        return "code", packed_to_torch(pk)[0]


class UnifiedMoK(nn.Module):
    """Общий адаптер -> топ-2 из E экспертов -> выход."""

    def __init__(self, in_dim=128, hidden=192, out_dim=128,
                 n_experts=4, layers=2):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(in_dim, hidden),
                                     nn.LayerNorm(hidden), nn.SiLU())
        self.experts = nn.ModuleList([
            nn.Sequential(*[ChebyKANLayer(hidden, hidden) for _ in range(layers)],
                          nn.Linear(hidden, out_dim))
            for _ in range(n_experts)])
        self.router = nn.Linear(hidden, n_experts)

    def forward(self, x):
        h = self.adapter(x)
        top_p, top_i = torch.topk(self.router(h), 2, dim=-1)
        alpha = F.softmax(top_p, -1)
        out = torch.zeros(len(h), self.experts[0][-1].out_features)
        for e, ex in enumerate(self.experts):
            mask = (top_i == e)
            if not mask.any():
                continue
            rows, slots = mask.nonzero(as_tuple=True)
            out.index_add_(0, rows, alpha[rows, slots].unsqueeze(-1) * ex(h[rows]))
        return out


def main():
    random.seed(0); torch.manual_seed(0)
    print("[UNIFIED FRONT] sandbox profile")

    binder = fuga_core.HybridBinder(2048)
    mixer = StreamMixer(binder)
    flt = AstralDataStreamFilter(adaptive=True, margin=0.25)

    model = UnifiedMoK(in_dim=2048, hidden=192, out_dim=2048, n_experts=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    stats = {"physics": [0, 0.0], "lang": [0, 0.0], "code": [0, 0.0]}
    updates = skipped = 0
    t0 = time.perf_counter()

    def target_for(kind: str, hv) -> torch.Tensor:
        """Таргет = трансформация состояния (детерминированная на канал).
        physics приходит packed bytes из v3-генератора."""
        if isinstance(hv, (bytes, bytearray)):
            arr = np.frombuffer(hv, dtype=np.uint8)
            rolled = np.roll(arr, 5)
            bits = ((rolled > 0).astype(np.float32) * 2 - 1)
            return torch.from_numpy(bits)
        if kind == "physics":
            base = np.asarray(hv).astype(np.int8)
            nxt = np.roll(base, 5)
            return torch.from_numpy((nxt > 0).astype(np.float32) * 2 - 1)
        return torch.roll(hv, shifts=-3)

    STEPS = 800
    for step in range(STEPS + 1):
        kind, x = mixer.next_sample()
        tgt = target_for(kind, x)
        pred = model(x.unsqueeze(0))[0]

        ok, s = flt.should_ingest(pred.detach(), tgt)
        loss_kind = F.mse_loss(pred, tgt)
        if ok:
            loss = loss_kind + F.cross_entropy(
                model.router(model.adapter(x.unsqueeze(0))),
                torch.tensor([random.Random(kind).randint(0,3)])) * 0.01
            opt.zero_grad(); loss.backward(); opt.step()
            updates += 1
        else:
            skipped += 1

        st = stats.setdefault(kind, [0, 0.0])
        st[0] += 1; st[1] += float(loss_kind)

        if step % 200 == 0:
            print(f"step {step}: updates={updates} skipped={skipped} "
                  f"pass={updates/max(step+1,1):.2f}")

    dt = time.perf_counter() - t0
    print(f"\n[unified front] {STEPS} шагов за {dt:.0f}s "
          f"({STEPS/dt:.1f} steps/s), обновлено {updates}, пропущено фильтром {skipped}")
    for kind, (cnt, lsum) in stats.items():
        if cnt:
            print(f"  {kind}: samples={cnt}")

    sys.stdout.flush()
    os._exit(0)   # HF streaming threads роняют интерпретатор на teardown


if __name__ == "__main__":
    main()
