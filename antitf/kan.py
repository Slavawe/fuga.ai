from __future__ import annotations

import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1) * 0.1
        )
        self.last_basis: torch.Tensor | None = None

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        xc = torch.tanh(x)
        basis = [torch.ones_like(xc), xc]
        for k in range(2, self.degree + 1):
            basis.append(2.0 * xc * basis[-1] - basis[-2])
        return torch.stack(basis, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._chebyshev_basis(x)
        self.last_basis = basis.detach()
        return torch.einsum("...id,oid->...o", basis, self.coeffs)


class ChebyMLP(nn.Module):
    def __init__(self, dims: list[int], degree: int = 4):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(ChebyKANLayer(a, b, degree))
        self.layers = nn.ModuleList(layers)
        self.activations: dict[int, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            self._capture(x)
            x = layer(x)
            if layer is not self.layers[-1]:
                x = torch.tanh(x)
        return x

    def _capture(self, x: torch.Tensor) -> None:
        self.activations[len(self.activations)] = x.detach()


def ema_update(target: nn.Module, online: nn.Module, momentum: float = 0.99) -> None:
    with torch.no_grad():
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.mul_(momentum).add_(po, alpha=1.0 - momentum)
