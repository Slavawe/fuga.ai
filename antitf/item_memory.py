from __future__ import annotations

import re
from collections import Counter

import torch
import torch.nn as nn

UNK_ID = 0


class SimpleWordVocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi

    @classmethod
    def build(cls, texts: list[str], max_size: int = 50000) -> "SimpleWordVocab":
        counter = Counter()
        for t in texts:
            counter.update(w.lower() for w in re.findall(r"\w+", t.lower()))
        stoi = {"<unk>": UNK_ID}
        for word, _ in counter.most_common(max_size - 1):
            if word not in stoi:
                stoi[word] = len(stoi)
        return cls(stoi)

    def encode(self, text: str, seq_len: int = 32) -> list[int]:
        ids = [self.stoi.get(w.lower(), UNK_ID) for w in re.findall(r"\w+", text.lower())]
        ids = ids[:seq_len]
        return ids + [UNK_ID] * (seq_len - len(ids))

    def __len__(self) -> int:
        return len(self.stoi)


class VSAItemMemory(nn.Module):
    """Orthogonal bipolar item memory with positional binding and O(1) unbinding."""

    def __init__(self, vocab_size: int = 50000, hyper_dim: int = 2048,
                 max_positions: int = 128, seed: int = 0):
        super().__init__()
        self.hyper_dim = hyper_dim
        g = torch.Generator().manual_seed(seed)
        words = torch.sign(torch.randn(vocab_size, hyper_dim, generator=g))
        words[UNK_ID].fill_(1.0)
        positions = torch.sign(torch.randn(max_positions, hyper_dim, generator=g))
        self.register_buffer("memory", words)
        self.register_buffer("pos_memory", positions)

    @torch.no_grad()
    def encode_structured_sequence(self, token_ids: torch.Tensor,
                                   batch_size: int = 512) -> torch.Tensor:
        """token_ids: [B, N] -> compositional HV [B, D]: bundle_i(word_i * pos_i)."""
        out = []
        for s in range(0, token_ids.shape[0], batch_size):
            chunk = token_ids[s : s + batch_size].long()
            n = chunk.shape[1]
            words = self.memory[chunk]
            bound = words * self.pos_memory[:n].unsqueeze(0)
            summed = bound.sum(dim=1)
            hv = torch.sign(summed + 1e-5)
            out.append(hv.masked_fill_(hv == 0, 1.0))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def query_word_at_position(self, hv_sequence: torch.Tensor,
                               pos_idx: int) -> torch.Tensor:
        """Unbinding at a position, then cosine logits over the item memory."""
        pos_vec = self.pos_memory[pos_idx].unsqueeze(0)
        unbound = hv_sequence * pos_vec
        return torch.matmul(unbound, self.memory.T) / self.hyper_dim
