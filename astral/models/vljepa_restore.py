"""VL-JEPA Restore — восстановление vision-language JEPA из сиротских весов.

Веса есть (vljepa_projector.pt 14MB, vljepa_bridge_v1.pt 8.7MB), кода нет.
Восстанавливаем полный конвейер:

  img_feat 2048 ──projector.img──→ 256-d
  text_hv  2048 ──caption_proj───→ 256-d  (из FugaTokenizer)
  vsa      512  ──vision_proj────→ 256-d  (из fuga_core)

Все три модальности → единое 256-d пространство → косинусная близость.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fuga_core import HybridBinder
from astral.fuga_tokenizer import FugaTokenizer


class VLJEPAEncoder(nn.Module):
    """Изображение/текст → 256-d латент (единое пространство)."""

    def __init__(self, projector_path: str, bridge_path: str):
        super().__init__()
        self.dim = 2048
        self.latent_dim = 256

        # 1. Projector: img 2048→768→256, cap 2048→768→256
        proj = torch.load(projector_path, map_location="cpu")
        self.img_proj = nn.Sequential(
            nn.Linear(2048, 768), nn.ReLU(), nn.Linear(768, 256),
        )
        self.img_proj[0].weight.data.copy_(proj["img.0.weight"])
        self.img_proj[0].bias.data.copy_(proj["img.0.bias"])
        self.img_proj[2].weight.data.copy_(proj["img.2.weight"])
        self.img_proj[2].bias.data.copy_(proj["img.2.bias"])
        self.cap_proj = nn.Sequential(
            nn.Linear(2048, 768), nn.ReLU(), nn.Linear(768, 256),
        )
        self.cap_proj[0].weight.data.copy_(proj["cap.0.weight"])
        self.cap_proj[0].bias.data.copy_(proj["cap.0.bias"])
        self.cap_proj[2].weight.data.copy_(proj["cap.2.weight"])
        self.cap_proj[2].bias.data.copy_(proj["cap.2.bias"])

        # 2. Bridge: vsa 512→512→256 (vision_proj), hv 2048→768→256 (caption_proj)
        bridge = torch.load(bridge_path, map_location="cpu")
        vp = bridge["vision_proj"]
        self.vsa_proj = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 512), nn.Linear(512, 256),
        )
        self.vsa_proj[1].weight.data.copy_(vp["0.weight"])
        self.vsa_proj[1].bias.data.copy_(vp["0.bias"])
        self.vsa_proj[2].weight.data.copy_(vp["3.weight"])
        self.vsa_proj[2].bias.data.copy_(vp["3.bias"])

        cp = bridge["caption_proj"]
        self.hv_proj = nn.Sequential(
            nn.Linear(2048, 768), nn.LayerNorm(768), nn.Linear(768, 256),
        )
        self.hv_proj[0].weight.data.copy_(cp["net.0.weight"])
        self.hv_proj[0].bias.data.copy_(cp["net.0.bias"])
        self.hv_proj[2].weight.data.copy_(cp["net.3.weight"])
        self.hv_proj[2].bias.data.copy_(cp["net.3.bias"])

        for p in self.parameters():
            p.requires_grad = False

    def encode_image(self, feat: torch.Tensor) -> torch.Tensor:
        """2048-d фича изображения → 256-d латент (L2-norm)."""
        return F.normalize(self.img_proj(feat), dim=-1)

    def encode_caption(self, feat: torch.Tensor) -> torch.Tensor:
        """2048-d фича текста → 256-d латент (L2-norm)."""
        return F.normalize(self.cap_proj(feat), dim=-1)

    def encode_hv(self, hv: torch.Tensor) -> torch.Tensor:
        """2048-d HV (из FugaTokenizer) → 256-d латент."""
        return F.normalize(self.hv_proj(hv), dim=-1)

    def encode_vsa(self, vsa: torch.Tensor) -> torch.Tensor:
        """512-d VSA-вектор → 256-d латент."""
        return F.normalize(self.vsa_proj(vsa), dim=-1)


def demo():
    torch.manual_seed(0)
    enc = VLJEPAEncoder("vljepa_projector.pt", "vljepa_bridge_v1.pt")
    n = sum(p.numel() for p in enc.parameters())
    print(f"=== V1. VL-JEPA — {n:,} параметров, веса загружены ===\n")

    # 1. Image → 256-d
    img = torch.randn(2048)
    il = enc.encode_image(img)
    print(f"1. img → 256-d: norm={il.norm():.3f}")

    # 2. Caption → 256-d
    cap = torch.randn(2048)
    cl = enc.encode_caption(cap)
    print(f"2. cap → 256-d: norm={cl.norm():.3f}, cos(img,cap)={(il*cl).sum():.3f}")

    # 3. HV (FugaTokenizer) → 256-d
    binder = HybridBinder(2048)
    tok = FugaTokenizer(binder)
    hvs = tok.encode(b"a red cube on a table")
    if hvs:
        hv = torch.stack(hvs).float().sum(0)
        hl = enc.encode_hv(hv)
        hi = enc.encode_hv(torch.stack(tok.encode(b"sunny weather today")).float().sum(0))
        hd = enc.encode_hv(torch.stack(tok.encode(b"a red cube on a table")).float().sum(0))
        print(f"3. HV → 256-d: norm={hl.norm():.3f}, "
              f"cos(same,similar)={(hl*hl).sum():.3f}, "
              f"cos(same,diff)={(hl*hi).sum():.3f}")
        print(f"   cos(same,repeat)={(hl*hd).sum():.3f} (≈1 если детерминированно)")

    # 4. VSA → 256-d
    vsa = torch.randn(512)
    vl = enc.encode_vsa(vsa)
    print(f"4. VSA → 256-d: norm={vl.norm():.3f}")

    # 5. Cross-modal matching
    print(f"\n5. Кросс-модальная близость (все в 256-d):")
    print(f"   cos(img, cap)={(il*cl).sum():.3f}")
    print(f"   cos(img, vsa)={(il*enc.encode_vsa(vsa+0.01)).sum():.3f}")
    print(f"   cos(cap, hv) ={(cl*hl).sum():.3f}")

    print("\n=== V1. VL-JEPA — OK ===")


if __name__ == "__main__":
    demo()