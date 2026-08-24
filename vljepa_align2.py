"""VL-JEPA v1: ResNet18 (предобученное восприятие) <-> caption bundle.

Урок vljepa_align: handcrafted цветовые хеши не несут семантики объектов —
выравнивать нечего (held-out на шансе при train 1.0). Здесь визуальная
сторона получает РЕАЛЬНЫЕ признаки ImageNet-обучения, контрастивный мост
обучает соответствие объект-в-кадре <-> слова-в-подписи.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, ".")

import fuga_core
from antitf.rust_bridge import packed_to_torch
from vljepa_dataset_loader import load_captions


class VisionEncoder(nn.Module):
    """Замороженный ResNet18 -> обучаемая проекция в общее пространство."""

    def __init__(self, out=256):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2],
                                      nn.AdaptiveAvgPool2d(1))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512),
                                  nn.SiLU(), nn.Linear(512, out))

    @torch.no_grad()
    def features(self, img_bgr: np.ndarray) -> torch.Tensor:
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear")
        return self.backbone(x).flatten(1)              # [1, 512]

    def forward(self, feats):
        return F.normalize(self.proj(feats), dim=-1)


class CaptionProjector(nn.Module):
    def __init__(self, dim=2048, out=256):
        super().__init__()
        # ВАЖНО: биполярный ±1 вход 2048 измерений даёт |x|~45 — без
        # нормализации SiLU насыщается и градиент умирает (loss на шансе).
        self.net = nn.Sequential(nn.Linear(dim, 768), nn.LayerNorm(768),
                                 nn.SiLU(), nn.Linear(768, out))

    def forward(self, hv):
        return F.normalize(self.net(hv), dim=-1)


def infonce(z1, z2, tau=0.07, var_w=8.0):
    """InfoNCE + Variance-барьер: без него модель минимизирует лосс
    коллапсом всех эмбеддингов в точку (logit_std -> 0, loss -> ln B).
    Проверено экспериментом 24.08: чистый InfoNCE замерзал на 4.85=ln128."""
    logits = z1 @ z2.T / tau
    labels = torch.arange(len(z1))
    ce = (F.cross_entropy(logits, labels) +
          F.cross_entropy(logits.T, labels)) / 2
    std1 = torch.sqrt(z1.var(0) + 1e-4)
    std2 = torch.sqrt(z2.var(0) + 1e-4)
    var = torch.relu(1.0 - std1).mean() + torch.relu(1.0 - std2).mean()
    return ce + var_w * var


@torch.no_grad()
def retrieval(venc, cproj, f_img, c_hvs, cap_texts=None, query_words=None):
    zi = venc(f_img)
    zc = cproj(c_hvs)
    ranks = (-(zi @ zc.T)).argsort(dim=1)
    gold = torch.arange(len(zi))
    a1 = (ranks[:, 0] == gold).float().mean().item()
    a10 = (ranks[:, :10] == gold.unsqueeze(1)).any(1).float().mean().item()
    return a1, a10


def main():
    random.seed(0); torch.manual_seed(0)

    binder = fuga_core.HybridBinder(DIM := 2048)
    venc = VisionEncoder()
    cproj = CaptionProjector(DIM)
    caps_all = load_captions("datasets/coco/annotations_trainval2017.zip")

    val_dir = "datasets/coco/val2017"
    imgs = sorted(os.listdir(val_dir))
    random.shuffle(imgs)

    print("[encode] resnet18 features + caption HVs ...")
    t0 = time.perf_counter()
    feats, cap_hvs = [], []
    texts = []
    for name in imgs:
        img = cv2.imread(os.path.join(val_dir, name))
        if img is None:
            continue
        caps = caps_all.get(name)
        if not caps:
            continue
        feats.append(venc.features(img)[0])
        words = [w.lower() for w in re.findall(r"[a-z]+", caps[0].lower())][:16]
        if not words:
            continue
        cap_hvs.append(packed_to_torch(np.asarray(binder.bind_batch([words])))[0])
        texts.append(words)
        if len(feats) >= 3000:
            break
        if len(feats) % 500 == 0:
            print(f"  {len(feats)} ({time.perf_counter()-t0:.0f}s)")
    feats = torch.stack(feats)
    cap_hvs = torch.stack(cap_hvs)
    n_train = 2400
    tr_f, te_f = feats[:n_train], feats[n_train:]
    tr_c, te_c = cap_hvs[:n_train], cap_hvs[n_train:]
    tr_words, te_words = texts[:n_train], texts[n_train:]
    print(f"train={n_train} heldout={len(te_f)}")

    opt = torch.optim.Adam(
        list(venc.proj.parameters()) + list(cproj.parameters()), lr=1e-3)
    n = n_train
    for step in range(2001):
        idx = torch.randint(0, n, (128,))
        zi = venc(tr_f[idx])
        zc = cproj(tr_c[idx])
        loss = infonce(zi, zc)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0 or step == 2000:
            with torch.no_grad():
                a1, a10 = retrieval(venc, cproj, tr_f[:600], tr_c[:600])
                h1, h10 = retrieval(venc, cproj, te_f, te_c)
            chance = 1.0 / len(te_f)
            print(f"step {step}: loss={loss.item():.3f} | TRAIN {a1:.3f}/{a10:.3f} | "
                  f"HELDOUT acc@1={h1:.3f} @10={h10:.3f} "
                  f"(chance {chance:.4f})")

    torch.save({"vision_proj": venc.proj.state_dict(),
                "caption_proj": cproj.state_dict()}, "vljepa_bridge_v1.pt")

    # сохранение выровненных vision-латентов в персистентную память
    from fuga_memory import PersistentVSAMemory
    mem = PersistentVSAMemory(binder, directory="fuga_memory_vision")
    with torch.no_grad():
        for i in range(min(50, len(te_f))):
            mem.add_fact("en", "vl_latent:" + str(i),
                         "caption", " ".join(te_words[i])[:80],
                         dedupe_key=("vl", i))
    print("bridge saved: vljepa_bridge_v1.pt; 50 aligned latents в памяти")


if __name__ == "__main__":
    main()
