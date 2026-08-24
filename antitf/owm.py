from __future__ import annotations

import torch
import torch.nn as nn


class OWMExecutor:
    """OWM via SVD row-space projector. Kept as reference implementation."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, eps: float = 1e-3,
                 max_stored: int = 4096):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.max_stored = max_stored
        self.projections: dict[str, torch.Tensor] = {}
        self.stored: dict[str, list[torch.Tensor]] = {}

    @torch.no_grad()
    def update_space(self, param_name: str, activations: torch.Tensor) -> None:
        A = activations.detach().reshape(-1, activations.shape[-1])
        bucket = self.stored.setdefault(param_name, [])
        bucket.append(A.cpu())
        A = torch.cat(bucket, dim=0)
        if A.shape[0] > self.max_stored:
            A = A[torch.randperm(A.shape[0])[: self.max_stored]]
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        keep = S > self.eps * max(S.max().item(), 1.0)
        V = Vh.T[:, keep]
        proj = torch.eye(A.shape[1], device=A.device) - V @ V.T
        self.projections[param_name] = proj

    @torch.no_grad()
    def apply_gradients(self, lr: float | None = None) -> None:
        step_lr = self.lr if lr is None else lr
        for name, p in self.model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            g = p.grad
            if name in self.projections:
                g = (g.reshape(p.shape[0], -1) @ self.projections[name]).view_as(p)
            p.sub_(step_lr * g)

    @torch.no_grad()
    def zero_grad(self) -> None:
        for _, p in self.model.named_parameters():
            p.grad = None


class WoodburyOWMExecutor(OWMExecutor):
    """OWM via Woodbury identity: cost O(B^3) per update, B << D.

    Ridge row-space projector applied in factorized form without ever
    materializing a [D, D] matrix:

        P = I - (1/eps) * A^T (I_B + A A^T / eps)^{-1} A

    Gradient update: g_proj = g - (1/eps) * ((g @ A^T) inv_inner @ A).
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, eps: float = 1e-3,
                 max_stored: int = 2048):
        super().__init__(model, lr, eps, max_stored)
        self.factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def update_space(self, param_name: str, activations: torch.Tensor) -> None:
        A_new = activations.detach().reshape(-1, activations.shape[-1]).cpu()
        bucket = self.stored.setdefault(param_name, [])
        bucket.append(A_new)
        A = torch.cat(bucket, dim=0)
        if A.shape[0] > self.max_stored:
            keep_idx = torch.randperm(A.shape[0])[: self.max_stored]
            A = A[keep_idx]
            bucket.clear()
            bucket.append(A)
        n = A.shape[0]
        inner = (A @ A.T) / self.eps + torch.eye(n)
        inv_inner = torch.linalg.inv(inner)
        self.factors[param_name] = (A, inv_inner)

    @torch.no_grad()
    def apply_gradients(self, lr: float | None = None) -> None:
        step_lr = self.lr if lr is None else lr
        for name, p in self.model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            g2d = p.grad.reshape(p.shape[0], -1)
            if name in self.factors:
                A, inv_inner = self.factors[name]
                corr = (g2d.cpu() @ A.T) @ inv_inner @ A
                g2d = (g2d.cpu() - corr / self.eps).to(p.device)
            p.sub_(step_lr * g2d.view_as(p))
