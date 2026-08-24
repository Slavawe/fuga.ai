from __future__ import annotations

import numpy as np
import torch

try:
    import fuga_core
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def get_binder(bits: int = 2048):
    if not HAS_RUST:
        return None
    return fuga_core.HybridBinder(bits)


def tokens_to_items(token_ids: list[int]) -> list[str]:
    """Стабильное имя атома памяти для id токена (единый маппинг RU/EN)."""
    return [f"tok:{i}" for i in token_ids]


@torch.no_grad()
def rust_unbinding_check(binder, vocab, texts: list[str], seq_len: int = 32,
                         max_positions: int = 16) -> dict:
    """Развязка через Rust: unbind_batch + score_items по всему словарю."""
    ids_rows = [vocab.encode(t, seq_len)[:max_positions] for t in texts]
    sentences = [tokens_to_items(r) for r in ids_rows]
    hv = binder.bind_batch(sentences)
    vocab_items = [f"tok:{i}" for i in range(len(vocab))]
    correct = total = 0
    per_pos = []
    for pos in range(max_positions):
        unbound = np.asarray(binder.unbind_batch(hv, pos + 1))
        scores = np.asarray(binder.score_items(unbound, vocab_items))
        pred = scores.argmax(axis=1)
        gold = np.array([r[pos] if pos < len(r) else 0 for r in ids_rows])
        mask = gold != 0
        hit = pred[mask] == gold[mask]
        correct += int(hit.sum())
        total += int(mask.sum())
        per_pos.append(float(hit.mean()) if mask.any() else 0.0)
    return {
        "unbinding_acc@1": correct / max(total, 1),
        "positions_evaluated": total,
        "per_position_first4": [round(p, 4) for p in per_pos[:4]],
    }


@torch.no_grad()
def packed_to_torch(hv_packed: np.ndarray) -> torch.Tensor:
    """packed uint64 [B, W] -> bipolar float32 [B, bits] для PyTorch-канала."""
    words = torch.from_numpy(np.ascontiguousarray(hv_packed)).long()
    bits = words.shape[1] * 64
    bit_planes = ((words.unsqueeze(-1) >> torch.arange(64)) & 1).reshape(words.shape[0], bits)
    return bit_planes.float() * 2.0 - 1.0
