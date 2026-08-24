from __future__ import annotations

import torch
import torch.nn as nn


class VectorAdapter(nn.Module):
    def __init__(self, hv_dim: int = 2048, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hv_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, hv: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(hv.float()), p=2, dim=-1)


class SequencePooler(nn.Module):
    """Hierarchical level: mean-pool consecutive chunk latents into block latents."""

    @staticmethod
    def chunk_pool(z: torch.Tensor, chunk_size: int = 2) -> torch.Tensor:
        b, t, d = z.shape
        t2 = (t // chunk_size) * chunk_size
        return z[:, :t2].reshape(b, -1, chunk_size, d).mean(dim=2)
