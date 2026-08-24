from __future__ import annotations

import torch
import torch.nn as nn

KEYWORDS = (
    "if", "else", "for", "while", "return", "struct", "typedef",
    "switch", "case", "break", "continue", "sizeof", "static",
    "const", "void", "int", "char", "float", "double",
)

OPERATORS = tuple("+-*/%=&|^~!<>(){}[];:,.?#")


class VSAEncoder(nn.Module):
    """Binary hypervector encoder: multiply-binding + majority bundling."""

    def __init__(self, vocab_size: int = 256, hyper_dim: int = 2048,
                 n_positions: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        base = torch.randint(0, 2, (vocab_size, hyper_dim), generator=g).float() * 2 - 1
        pos = torch.randint(0, 2, (n_positions, hyper_dim), generator=g).float() * 2 - 1
        vocab = list(KEYWORDS) + list(OPERATORS)
        tok = torch.randint(0, 2, (len(vocab), hyper_dim), generator=g).float() * 2 - 1
        self.register_buffer("base_vectors", base)
        self.register_buffer("pos_vectors", pos)
        self.register_buffer("token_vectors", tok)
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.hyper_dim = hyper_dim
        self.n_positions = n_positions

    def bind(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return v1 * v2

    def bundle(self, vectors: torch.Tensor, dim: int = 1) -> torch.Tensor:
        s = torch.sign(vectors.sum(dim=dim) + 1e-5)
        return s.masked_fill_(s == 0, 1.0)

    def forward(self, byte_seq: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """byte_seq: [B, W] uint8 -> bipolar hypervectors [B, hyper_dim].

        Positions are processed in chunks so the intermediate tensor is
        [B, chunk_size, D], not [B, W, D].
        """
        b, w = byte_seq.shape
        acc = torch.zeros(b, self.hyper_dim, device=byte_seq.device)
        for s in range(0, w, chunk_size):
            seg = byte_seq[:, s : s + chunk_size].long()
            c = seg.shape[1]
            embedded = self.base_vectors[seg]
            pos_ids = (torch.arange(s, s + c, device=byte_seq.device)) % self.n_positions
            bound = embedded * self.pos_vectors[pos_ids].unsqueeze(0)
            acc += bound.sum(dim=1)
        return self.bundle(acc.unsqueeze(1), dim=1).squeeze(1)

    @torch.no_grad()
    def encode_syntax(self, raw_windows: list[bytes]) -> torch.Tensor:
        out = torch.zeros(len(raw_windows), self.hyper_dim, device=self.base_vectors.device)
        for row, raw in enumerate(raw_windows):
            text = raw.decode("latin-1")
            hits = []
            for tok, ti in self.vocab.items():
                start = 0
                while True:
                    i = text.find(tok, start)
                    if i < 0:
                        break
                    hits.append(ti)
                    start = i + max(len(tok), 1)
            if hits:
                out[row] = self.token_vectors[hits].sum(dim=0)
        out[out == 0] = 1.0
        return torch.sign(out)

    @torch.no_grad()
    def mix(self, hv_bytes: torch.Tensor, hv_syntax: torch.Tensor) -> torch.Tensor:
        return self.bundle(torch.stack([hv_bytes, hv_syntax], dim=1), dim=1)

    @torch.no_grad()
    def similarity_search(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Hamming-based associative lookup via bipolar dot-product trick."""
        sim = query.float() @ memory.float().T
        return (-sim).argsort(dim=1)
