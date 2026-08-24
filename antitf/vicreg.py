from __future__ import annotations

import torch
import torch.nn as nn


class VICRegLoss(nn.Module):
    def __init__(self, inv_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def forward(self, z_hat: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        z_hat = z_hat.flatten(0, -2)
        z_target = z_target.flatten(0, -2)

        sim_loss = torch.mean((z_hat - z_target) ** 2)

        std_hat = torch.sqrt(torch.var(z_hat, dim=0) + self.eps)
        std_tgt = torch.sqrt(torch.var(z_target, dim=0) + self.eps)
        var_loss = torch.mean(torch.relu(1.0 - std_hat)) + torch.mean(
            torch.relu(1.0 - std_tgt)
        )

        batch_size, d = z_hat.shape
        zh = z_hat - z_hat.mean(dim=0)
        zt = z_target - z_target.mean(dim=0)
        cov_hat = (zh.T @ zh) / (batch_size - 1)
        cov_tgt = (zt.T @ zt) / (batch_size - 1)
        cov_hat.fill_diagonal_(0.0)
        cov_tgt.fill_diagonal_(0.0)
        cov_loss = cov_hat.pow(2).sum() / d + cov_tgt.pow(2).sum() / d

        return self.inv_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
