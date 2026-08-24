from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

import fuga_core

from antitf.jepa import HJEPAPredictor
from antitf.owm import WoodburyOWMExecutor
from antitf.vicreg import VICRegLoss

LANG_PYTHON = 0
LANG_C = 1


def run_fuga_hybrid_pipeline(steps: int = 30) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    rust_encoder = fuga_core.RustVSAEncoder()
    predictor = HJEPAPredictor(hyper_dim=2048, latent_dim=512).to(device)
    criterion = VICRegLoss()
    owm = WoodburyOWMExecutor(predictor, lr=1e-3)

    code_samples_ctx = [
        "def process_data(x): return [i * 2 for i in x]",
        "class Node:\n    def __init__(self, val):\n        self.val = val",
        "def add(a, b): return a + b",
        "for i in range(10):\n    print(i)",
    ]
    code_samples_tgt = [
        "def process_data_next(y): return sum(y)",
        "class NodeNext:\n    def set_val(self, v):\n        self.val = v",
        "def sub(a, b): return a - b",
        "while x < 10:\n    x += 1",
    ]

    hv_ctx_np: np.ndarray = rust_encoder.encode_batch_py(code_samples_ctx, lang=LANG_PYTHON)
    hv_tgt_np: np.ndarray = rust_encoder.encode_batch_py(code_samples_tgt, lang=LANG_PYTHON)
    hv_ctx = torch.from_numpy(hv_ctx_np).to(device)
    hv_tgt = torch.from_numpy(hv_tgt_np).to(device)

    packed_check = np.asarray(rust_encoder.encode_batch_packed(code_samples_ctx[:1], lang=LANG_PYTHON))
    assert packed_check.shape == (1, 32), packed_check.shape

    history = []
    for step in range(steps):
        z_ctx, z_hat = predictor(hv_ctx)
        with torch.no_grad():
            _, z_tgt = predictor(hv_tgt)

        loss = criterion(z_hat, z_tgt)

        owm.zero_grad()
        loss.backward()

        owm.update_space("adapter.net.0.weight", hv_ctx)
        b1 = predictor.kan1.last_basis
        b2 = predictor.kan2.last_basis
        if b1 is not None:
            feat = predictor.kan1.in_features * (predictor.kan1.degree + 1)
            owm.update_space("kan1.coeffs", b1.reshape(-1, feat))
        if b2 is not None:
            feat = predictor.kan2.in_features * (predictor.kan2.degree + 1)
            owm.update_space("kan2.coeffs", b2.reshape(-1, feat))
        predictor.kan1.last_basis = None
        predictor.kan2.last_basis = None

        owm.apply_gradients(lr=1e-3)
        history.append(loss.item())

    print("[Fuga Core Integrated] steps:", steps)
    print(f"  loss[0]={history[0]:.4f}  loss[mid]={history[len(history)//2]:.4f}  loss[-1]={history[-1]:.4f}")
    print(f"  packed storage: {packed_check.nbytes}B vs float {hv_ctx_np[:1].nbytes}B per sample")


if __name__ == "__main__":
    run_fuga_hybrid_pipeline()
