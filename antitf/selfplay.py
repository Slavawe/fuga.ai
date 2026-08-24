from __future__ import annotations

import torch

from .data_i18n import make_byte_windows
from .vsa import VSAEncoder


def encode_sentences(vsa: VSAEncoder, texts: list[str], device: str,
                     batch_size: int = 512) -> torch.Tensor:
    rows = [
        torch.frombuffer(bytearray(make_byte_windows(t)), dtype=torch.uint8).clone()
        for t in texts
    ]
    out = []
    for s in range(0, len(rows), batch_size):
        windows = torch.stack(rows[s : s + batch_size]).to(device)
        out.append(vsa(windows))
    return torch.cat(out, dim=0)


@torch.no_grad()
def estimate_role_vector(hv_ru: torch.Tensor, hv_en: torch.Tensor) -> torch.Tensor:
    """R = majority-bundle(bind(H_ru_i, H_en_i)): the shared RU->EN operator."""
    bound = hv_ru * hv_en
    return torch.sign(bound.sum(dim=0) + 1e-5).masked_fill_(
        torch.sign(bound.sum(dim=0) + 1e-5) == 0, 1.0)


@torch.no_grad()
def hamming_topk_retrieval(query: torch.Tensor, memory: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Bipolar trick: hamming(q, m) = (D - q·m) / 2, computed via matmul."""
    sim = query.float() @ memory.float().T
    return (-sim).argsort(dim=1)[:, :k]


@torch.no_grad()
def selfplay_vsa_metrics(vsa: VSAEncoder, pairs_train: list[tuple[str, str]],
                         pairs_test: list[tuple[str, str]], device: str = "cpu",
                         k_values: tuple[int, ...] = (1, 5, 10)) -> dict:
    hv_ru_tr = encode_sentences(vsa, [p[0] for p in pairs_train], device)
    hv_en_tr = encode_sentences(vsa, [p[1] for p in pairs_train], device)
    role = estimate_role_vector(hv_ru_tr, hv_en_tr)

    hv_ru_te = encode_sentences(vsa, [p[0] for p in pairs_test], device)
    hv_en_te = encode_sentences(vsa, [p[1] for p in pairs_test], device)

    translated = hv_ru_te * role
    ranks_gold = hamming_topk_retrieval(translated, hv_en_te, k=max(k_values))

    base_ranks = hamming_topk_retrieval(hv_ru_te, hv_en_te, k=max(k_values))

    gold = torch.arange(len(pairs_test), device=device).unsqueeze(1)
    out: dict[str, float] = {"role_norm": float(role.float().mean().abs())}
    for k in k_values:
        out[f"acc@{k}_translated"] = (ranks_gold == gold).any(dim=1).float().mean().item()
        out[f"acc@{k}_baseline"] = (base_ranks == gold).any(dim=1).float().mean().item()
    return out


@torch.no_grad()
def _pair_cosine(predictor, hv_a: torch.Tensor, hv_b: torch.Tensor) -> float:
    za = predictor.encode(hv_a)
    zb = predictor.encode(hv_b)
    return torch.nn.functional.cosine_similarity(za, zb, dim=-1).mean().item()


def neural_alignment(vsa: VSAEncoder, pairs_train: list[tuple[str, str]],
                     pairs_test: list[tuple[str, str]], device: str = "cpu",
                     steps: int = 60, batch: int = 32,
                     hyper_dim: int = 2048, latent_dim: int = 256) -> dict:
    """Cross-lingual latent alignment: predictor learns z(H_ru) ~ z(H_en)."""
    from .jepa import HJEPAPredictor
    from .owm import WoodburyOWMExecutor
    from .vicreg import VICRegLoss

    torch.manual_seed(0)
    hv_ru_tr = encode_sentences(vsa, [p[0] for p in pairs_train], device)
    hv_en_tr = encode_sentences(vsa, [p[1] for p in pairs_train], device)
    hv_ru_te = encode_sentences(vsa, [p[0] for p in pairs_test], device)
    hv_en_te = encode_sentences(vsa, [p[1] for p in pairs_test], device)

    model = HJEPAPredictor(hyper_dim=hyper_dim, latent_dim=latent_dim).to(device)
    owm = WoodburyOWMExecutor(model, lr=3e-4)
    loss_fn = VICRegLoss()

    cos_before = _pair_cosine(model, hv_ru_te, hv_en_te)
    perm0 = torch.randperm(len(pairs_test))
    cos_shuffled_init = _pair_cosine(model, hv_ru_te, hv_en_te[perm0])

    n = hv_ru_tr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,))
        z_ctx, z_hat = model(hv_ru_tr[idx])
        with torch.no_grad():
            _, z_tgt = model(hv_en_tr[idx])
        loss = loss_fn(z_hat, z_tgt)

        owm.zero_grad()
        loss.backward()
        b1 = model.kan1.last_basis
        if b1 is not None:
            feat = model.kan1.in_features * (model.kan1.degree + 1)
            owm.update_space("kan1.coeffs", b1.reshape(-1, feat))
            model.kan1.last_basis = None
        owm.apply_gradients(lr=3e-4)

    perm = torch.randperm(len(pairs_test))
    return {
        "cos_before": cos_before,
        "cos_shuffled_init": cos_shuffled_init,
        "cos_after_aligned": _pair_cosine(model, hv_ru_te, hv_en_te),
        "cos_after_shuffled": _pair_cosine(model, hv_ru_te, hv_en_te[perm]),
    }


@torch.no_grad()
def item_memory_unbinding_check(mem, vocab, texts: list[str], seq_len: int = 32,
                                max_positions: int = 16) -> dict:
    """Self-play check: unbind word at each position, compare argmax with truth."""
    ids = torch.tensor([vocab.encode(t, seq_len) for t in texts])
    hv = mem.encode_structured_sequence(ids)
    correct = total = 0
    per_pos = [0] * max_positions
    per_pos_total = [0] * max_positions
    for pos in range(max_positions):
        logits = mem.query_word_at_position(hv, pos)
        pred = logits.argmax(dim=1)
        gold = ids[:, pos]
        mask = gold != 0
        hit = (pred[mask] == gold[mask])
        correct += int(hit.sum())
        total += int(mask.sum())
        per_pos[pos] = float(hit.float().mean()) if mask.any() else 0.0
        per_pos_total[pos] = int(mask.sum())
    return {
        "unbinding_acc@1": correct / max(total, 1),
        "positions_evaluated": total,
        "per_position_acc_first8": [round(p, 4) for p in per_pos[:8]],
    }


@torch.no_grad()
def structured_retrieval(mem, vocab, pairs_test: list[tuple[str, str]],
                         ks: tuple[int, ...] = (1, 5, 10), seq_len: int = 32) -> dict:
    """RU->EN retrieval with compositional structural binding (vs bag-of-bytes)."""
    ru_ids = torch.tensor([vocab.encode(p[0], seq_len) for p in pairs_test])
    en_ids = torch.tensor([vocab.encode(p[1], seq_len) for p in pairs_test])
    hv_ru = mem.encode_structured_sequence(ru_ids)
    hv_en = mem.encode_structured_sequence(en_ids)
    sim = hv_ru.float() @ hv_en.float().T
    ranks = sim.argsort(dim=1, descending=True)
    gold = torch.arange(len(pairs_test)).unsqueeze(1)
    out = {}
    for k in ks:
        out[f"struct_acc@{k}"] = (ranks[:, :k] == gold).any(1).float().mean().item()
    return out
