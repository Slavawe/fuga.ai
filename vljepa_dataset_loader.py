
"""VL-JEPA Ingest v0: кадры/патчи -> VSA-кристаллы -> PersistentVSAMemory.

Честная стадия v0: визуальный энкодер — детерминированное хеширование
рукотворных признаков патча (цветовые бины, градиент/край, яркость).
Обучаемый JEPA-предиктор поверх этих слотов — следующий шаг;
модальностно-агностичная память принимает их без изменений ядра.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
import time

import numpy as np

sys.path.insert(0, ".")


import torch

import fuga_core
from fuga_memory import PersistentVSAMemory
from antitf.rust_bridge import packed_to_torch


def iter_coco_images(val_dir: str, limit: int | None = None):
    names = sorted(os.listdir(val_dir))
    if limit:
        names = names[:limit]
    for name in names:
        yield os.path.join(val_dir, name)


def load_captions(ann_zip: str, limit_ids: set[str] | None = None):
    """{image_file_name: [captions]} из annotations/captions_val2017.json."""
    out = {}
    with zipfile.ZipFile(ann_zip) as zf:
        inner = [x for x in zf.namelist() if x.endswith("captions_val2017.json")][0]
        with zf.open(inner) as f:
            data = json.load(f)
    for ann in data["annotations"]:
        fn = str(ann["image_id"]).zfill(12) + ".jpg"
        if limit_ids is not None and fn not in limit_ids:
            continue
        out.setdefault(fn, []).append(ann["caption"].strip().lower())
    return out


class PatchVSAEncoder:
    """v0: патч 32x32 -> дискретные признаки -> VSA-бандл."""

    COLOR_BINS = 4   # на канал
    EDGE_BINS = 3

    def __init__(self, binder: fuga_core.HybridBinder, patch_size: int = 32,
                 grid: int = 4):
        self.binder = binder
        self.ps = patch_size
        self.grid = grid

    def _patch_tokens(self, patch: np.ndarray) -> list[str]:
        r, g, b = patch[..., 0].mean(), patch[..., 1].mean(), patch[..., 2].mean()
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        grad = np.abs(np.diff(patch.mean(axis=2), axis=0)).mean() + \
               np.abs(np.diff(patch.mean(axis=2), axis=1)).mean()
        cb = int(r / 256 * self.COLOR_BINS)
        cg = int(g / 256 * self.COLOR_BINS)
        cbb = int(b / 256 * self.COLOR_BINS)
        eb = min(int(grad / 40 * self.EDGE_BINS), self.EDGE_BINS - 1)
        br = int(gray / 256 * 3)
        return [f"VIS:cr{cb}", f"VIS:cg{cg}", f"VIS:cb{cbb}",
                f"VIS:edge{eb}", f"VIS:bright{br}"]

    @torch.no_grad()
    def encode_image(self, img: np.ndarray) -> torch.Tensor:
        """Изображение -> один HV (бандл всех патчевых токенов)."""
        h, w = img.shape[:2]
        ph, pw = h // self.grid, w // self.grid
        all_tokens: list[str] = []
        for gi in range(self.grid):
            for gj in range(self.grid):
                patch = img[gi * ph:(gi + 1) * ph, gj * pw:(gj + 1) * pw]
                toks = self._patch_tokens(patch)
                # позиция патча входит в имя атома -> пространственное связывание
                pos = f"P:{gi}_{gj}"
                all_tokens.extend(f"{t}@{pos}" for t in toks)
        pk = np.asarray(self.binder.bind_batch([all_tokens]))
        return packed_to_torch(pk)[0]

    @torch.no_grad()
    def encode_caption(self, caption: str) -> torch.Tensor:
        words = re.findall(r"[a-z]+", caption.lower())[:16]
        pk = np.asarray(self.binder.bind_batch([words]))
        return packed_to_torch(pk)[0]


import re  # noqa: E402


def main():
    import cv2

    val_dir = "datasets/coco/val2017"
    ann_zip = "datasets/coco/annotations_trainval2017.zip"
    if not os.path.isdir(val_dir):
        print("val2017 ещё не распакован — распакуйте zip из datasets/coco")
        sys.exit(1)

    imgs = sorted(os.listdir(val_dir))[:400]
    id_set = {n for n in imgs}
    caps = load_captions(ann_zip, id_set)
    print(f"images={len(imgs)}, captions available={len(caps)}")

    binder = fuga_core.HybridBinder(DIM := 2048)
    mem = PersistentVSAMemory(binder, directory="fuga_memory_vision")

    enc = PatchVSAEncoder(binder)

    t0 = time.perf_counter()
    pairs = []          # (img_hv, cap_hv, caption)
    for i, path in enumerate(iter_coco_images(val_dir)):
        img = cv2.imread(path)
        if img is None:
            continue
        img_hv = enc.encode_image(img)
        caps_for_img = caps.get(os.path.basename(path))
        if not caps_for_img:
            continue
        cap_hv = enc.encode_caption(caps_for_img[0])
        pairs.append((img_hv, cap_hv, caps_for_img[0]))
        if (i + 1) % 100 == 0:
            print(f"  encoded {i+1} ({time.perf_counter()-t0:.0f}s)")
        if len(pairs) >= 300:
            break

    print(f"\n[vsa-ingest] пар закодировано: {len(pairs)} "
          f"за {time.perf_counter()-t0:.1f}s")

    # --- кросс-модальная проверка v0: изображение -> ближайшая подпись ---
    img_mat = torch.stack([p[0] for p in pairs])
    cap_mat = torch.stack([p[1] for p in pairs])
    sims = img_mat @ cap_mat.T
    ranks = (-sims).argsort(dim=1)
    gold = torch.arange(len(pairs))
    acc1 = (ranks[:, 0] == gold).float().mean().item()
    acc10 = (ranks[:, :10] == gold.unsqueeze(1)).any(1).float().mean().item()
    print(f"image->caption retrieval v0 (без контрастивного обучения): "
          f"acc@1={acc1:.4f} acc@10={acc10:.4f} (шанс 1/{len(pairs)}={1/len(pairs):.4f})")

    # сохраняем пару образцов в персистентную память как vision-факты
    for img_hv, cap_hv, cap in pairs[:20]:
        mem.add_fact("en", "vision:" + re.sub(r"\W+", "_", cap)[:48],
                     "encoded_from", "coco_val2017",
                     dedupe_key=("vision", cap))
    print("20 vision-фактов записано в PersistentVSAMemory")


if __name__ == "__main__":
    main()
