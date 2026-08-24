from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import SequencePooler, VectorAdapter
from .kan import ChebyKANLayer, ema_update


class HJEPAPredictor(nn.Module):
    """Adapter (HV -> latent) + two Fast-KAN layers.

    forward(hv_context) returns (z_context, z_hat_next): the latent of the
    current fragment and the predicted latent of the next fragment.
    """

    def __init__(self, hyper_dim: int = 2048, latent_dim: int = 512, degree: int = 4):
        super().__init__()
        self.adapter = VectorAdapter(hyper_dim, latent_dim)
        self.kan1 = ChebyKANLayer(latent_dim, latent_dim, degree=degree)
        self.kan2 = ChebyKANLayer(latent_dim, latent_dim, degree=degree)

    def encode(self, hv: torch.Tensor) -> torch.Tensor:
        return self.adapter(hv)

    def forward(self, hv_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.adapter(hv_context)
        h1 = F.gelu(self.kan1(z))
        z_hat = self.kan2(h1)
        return z, z_hat


class HJEPA(nn.Module):
    """Hierarchical wrapper: level 0 = next window, level 1 = next block mean."""

    def __init__(self, hyper_dim: int = 2048, latent_dim: int = 512,
                 degree: int = 4, ema_momentum: float = 0.99):
        super().__init__()
        self.online = HJEPAPredictor(hyper_dim, latent_dim, degree)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.block_head = ChebyKANLayer(latent_dim, latent_dim, degree=degree)
        self.ema_momentum = ema_momentum

    def forward(self, hv_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """hv_seq: [B, T, D] windows -> prediction targets for VICReg."""
        b, t, _ = hv_seq.shape
        flat = hv_seq.reshape(b * t, -1)
        z_all, pred_all = self.online(flat)
        z_all = z_all.reshape(b, t, -1)
        pred_all = pred_all.reshape(b, t, -1)

        with torch.no_grad():
            _, tgt_all = self.target(flat)
            tgt_all = tgt_all.reshape(b, t, -1)

        out = {
            "pred_l0": pred_all[:, :-1],
            "target_l0": tgt_all[:, 1:],
        }
        if t >= 3:
            blocks_pred = SequencePooler.chunk_pool(pred_all, 2)[:, :-1]
            blocks_tgt = SequencePooler.chunk_pool(tgt_all, 2)[:, 1:]
            out["pred_l1"] = self.block_head(blocks_pred)
            out["target_l1"] = blocks_tgt
        return out

    def update_target(self) -> None:
        ema_update(self.target, self.online, self.ema_momentum)
