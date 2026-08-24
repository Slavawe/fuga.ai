from __future__ import annotations

import argparse

import torch

from .ast_vsa import TreeSitterVSAEncoder
from .data import CORPUS_TASK_0, CORPUS_TASK_1, make_windows
from .jepa import HJEPA
from .owm import OWMExecutor, WoodburyOWMExecutor
from .vicreg import VICRegLoss
from .vsa import VSAEncoder


def train_task(model: HJEPA, owm, loss_fn: VICRegLoss,
               vsa: VSAEncoder, ast: TreeSitterVSAEncoder,
               corpus: list[str], steps: int = 30,
               batch: int = 8, seq_len: int = 6, device: str = "cpu") -> float:
    windows, raws = make_windows(corpus)
    windows = windows.to(device)
    n = windows.shape[0]
    hv_bytes = vsa(windows)
    hv_syntax = vsa.encode_syntax(raws)
    snippets = [bytes(w) for w in windows.cpu().tolist()]
    try:
        hv_ast = ast.encode_batch(snippets)
    except Exception:
        hv_ast = torch.zeros_like(hv_bytes)
    hv = vsa.mix(vsa.mix(hv_bytes, hv_syntax), hv_ast)

    total = 0.0
    for _ in range(steps):
        start = torch.randint(0, max(n - seq_len, 1), (batch,))
        idx = (start[:, None] + torch.arange(seq_len)) % n
        seq = hv[idx]

        out = model(seq)
        loss = loss_fn(out["pred_l0"], out["target_l0"])
        if "pred_l1" in out:
            loss = loss + loss_fn(out["pred_l1"], out["target_l1"])

        owm.zero_grad()
        loss.backward()

        with torch.no_grad():
            flat = seq.reshape(-1, seq.shape[-1]).float()
            z = model.online.adapter(flat)
        basis1 = model.online.kan1.last_basis
        basis2 = model.online.kan2.last_basis
        if basis1 is not None:
            owm.update_space("online.kan1.coeffs",
                             basis1.reshape(-1, basis1.shape[-2] * basis1.shape[-1]))
        if basis2 is not None:
            feat_in = model.online.kan1.out_features
            deg = model.online.kan1.degree
            owm.update_space("online.kan2.coeffs", basis2.reshape(-1, feat_in * (deg + 1)))
        model.online.kan1.last_basis = None
        model.online.kan2.last_basis = None

        owm.apply_gradients(lr=3e-4)
        model.update_target()
        total += loss.item()
    return total / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--hv-dim", type=int, default=2048)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(0)
    vsa = VSAEncoder(vocab_size=256, hyper_dim=args.hv_dim).to(device)
    ast = TreeSitterVSAEncoder(hyper_dim=args.hv_dim, device=device)
    model = HJEPA(hyper_dim=args.hv_dim, latent_dim=args.latent_dim).to(device)
    criterion = VICRegLoss().to(device)
    owm = WoodburyOWMExecutor(model, lr=3e-4)

    l0 = train_task(model, owm, criterion, vsa, ast, CORPUS_TASK_0, args.steps, device=device)
    print(f"task 0 (C basics):      avg VICReg loss = {l0:.4f}")
    l1 = train_task(model, owm, criterion, vsa, ast, CORPUS_TASK_1, args.steps, device=device)
    print(f"task 1 (C control/IO):  avg VICReg loss = {l1:.4f}")

    memory, _ = make_windows(CORPUS_TASK_0 + CORPUS_TASK_1)
    memory_hv = vsa(memory.to(device))
    query_hv = vsa(make_windows(["while (*p != 0) { p++; }"])[0].to(device))
    ranking = vsa.similarity_search(query_hv, memory_hv)[0][:3]
    print("associative search top-3 window ids:", ranking.tolist())


if __name__ == "__main__":
    main()
