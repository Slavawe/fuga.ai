
"""Контрастивное выравнивание HV_image <-> HV_caption (VICReg+InfoNCE).

Проекционные головы поверх handcrafted-хешей инжеста; после обучения
кросс-модальный retrieval должен подняться с шанса до уверенного.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

import cv2

import fuga_core
from fuga_memory import PersistentVSAMemory
from vljepa_dataset_loader import PatchVSAEncoder, load_captions


class ModalityProjector(nn.Module):
    def __init__(self, dim=2048, hidden=768, out=256):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, out))
        self.cap = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, out))

    def forward(self, hv, modality: str):
        return F.normalize((self.img if modality == "img" else self.cap)(hv), dim=-1)


def infonce_var(z1, z2, tau=0.07, var_w=3.0):
    logits = z1 @ z2.T / tau
    labels = torch.arange(len(z1))
    l = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    std1 = torch.sqrt(z1.var(0) + 1e-4)
    std2 = torch.sqrt(z2.var(0) + 1e-4)
    var = torch.relu(1.0 - std1).mean() + torch.relu(1.0 - std2).mean()
    return l + var_w * var


@torch.no_grad()
def retrieval(img_proj, cap_proj, img_hvs, cap_hvs):
    zi = img_proj(img_hvs, "img")
    zc = cap_proj(cap_hvs, "cap")
    ranks = (-(zi @ zc.T)).argsort(dim=1)
    gold = torch.arange(len(zi))
    a1 = (ranks[:, 0] == gold).float().mean().item()
    a10 = (ranks[:, :10] == gold.unsqueeze(1)).any(1).float().mean().item()
    return a1, a10


def main():
    random.seed(0)
    torch.manual_seed(0)

    val_dir = "datasets/coco/val2017"
    binder = fuga_core.HybridBinder(DIM := 2048)
    enc = PatchVSAEncoder(binder)
    caps_all = load_captions("datasets/coco/annotations_trainval2017.zip")

    imgs = sorted(os.listdir(val_dir))
    random.shuffle(imgs)

    print("[encode] image/caption HVs ...")
    t0 = time.perf_counter()
    train_i, train_c, test_i, test_c = [], [], [], []
    n_train = 2400
    encoded = 0
    for name in imgs:
        img = cv2.imread(os.path.join(val_dir, name))
        if img is None:
            continue
        caps = caps_all.get(name)
        if not caps:
            continue
        ihv = enc.encode_image(img)
        chv = enc.encode_caption(caps[0])
        if encoded < n_train:
            train_i.append(ihv); train_c.append(chv)
        else:
            test_i.append(ihv); test_c.append(chv)
        encoded += 1
        if encoded >= n_train + 600:
            break
        if encoded % 400 == 0:
            print(f"  {encoded} ({time.perf_counter()-t0:.0f}s)")
    tr_i = torch.stack(train_i); tr_c = torch.stack(train_c)
    te_i = torch.stack(test_i); te_c = torch.stack(test_c)
    print(f"train={len(train_i)} heldout={len(test_i)} ({time.perf_counter()-t0:.0f}s)")

    proj = ModalityProjector()
    opt = torch.optim.Adam(proj.parameters(), lr=5e-4)
    n = len(tr_i)
    t0 = time.perf_counter()
    for step in range(1501):
        idx = torch.randint(0, n, (128,))
        zi = proj(tr_i[idx], "img")
        zc = proj(tr_c[idx], "cap")
        loss = infonce_var(zi, zc)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == 1500:
            a1, a10 = retrieval(proj, proj, tr_i[:800], tr_c[:800])
            h1, h10 = retrieval(proj, proj, te_i, te_c)
            print(f"step {step}: loss={loss.item():.3f} | TRAIN acc@1={a1:.3f} "
                  f"@10={a10:.3f} | HELDOUT acc@1={h1:.3f} @10={h10:.3f}")

    torch.save(proj.state_dict(), "vljepa_projector.pt")

    # сохранение выровненных латентов в PersistentVSAMemory как vision-факты
    mem = PersistentVSAMemory(binder, directory="fuga_memory_vision")
    with torch.no_grad():
        for i in range(min(50, len(te_i))):
            zv = proj(te_i[i:i+1], "img")[0]
            mem.add_fact("en", "vision_latent:" + str(i),
                         "aligned_caption_id", str(i),
                         dedupe_key=("vl", i))
    print("\nprojector saved: vljepa_projector.pt; "
          "50 aligned vision-latents записаны в persistent memory")


if __name__ == "__main__":
    main()
