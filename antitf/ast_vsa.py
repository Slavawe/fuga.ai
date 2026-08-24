from __future__ import annotations

import torch

from tree_sitter import Language, Parser
import tree_sitter_c


class TreeSitterVSAEncoder:
    """Encodes a C AST into a single bipolar hypervector.

    H_node = sign( H_kind * bundle_i( permute(H_child_i, i+1) ) )

    Leaf values are bound with a value-hash vector so distinct identifiers
    map to distinct (quasi-orthogonal) components.
    """

    def __init__(self, hyper_dim: int = 2048, device: str = "cpu", seed: int = 0):
        self.hyper_dim = hyper_dim
        self.device = device
        g = torch.Generator().manual_seed(seed)
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser(self.language)
        self.node_memory: dict[str, torch.Tensor] = {}
        self.value_memory: dict[bytes, torch.Tensor] = {}
        self._g = g

    @property
    def kind_vectors(self) -> dict[str, torch.Tensor]:
        return self.node_memory

    def _kind_vector(self, node_type: str) -> torch.Tensor:
        if node_type not in self.node_memory:
            v = torch.randint(0, 2, (self.hyper_dim,), generator=self._g).float() * 2 - 1
            self.node_memory[node_type] = v.to(self.device)
        return self.node_memory[node_type]

    def _value_vector(self, text: bytes) -> torch.Tensor:
        if text not in self.value_memory:
            v = torch.randint(0, 2, (self.hyper_dim,), generator=self._g).float() * 2 - 1
            self.value_memory[text] = v.to(self.device)
        return self.value_memory[text]

    @staticmethod
    def _permute(v: torch.Tensor, shift: int) -> torch.Tensor:
        return torch.roll(v, shifts=shift, dims=-1)

    def _encode_node(self, node) -> torch.Tensor:
        kind = self._kind_vector(node.type)
        if node.child_count == 0:
            text = node.text if node.text is not None else b""
            if text.strip():
                return self.bind(kind, self._value_vector(text))
            return kind

        children = [self._encode_node(c) for c in node.children]
        permuted = [self._permute(c, i + 1) for i, c in enumerate(children)]
        bundled = torch.stack(permuted, dim=0).sum(dim=0)
        out = kind * bundled
        return torch.sign(out + 1e-5).masked_fill_(torch.sign(out + 1e-5) == 0, 1.0)

    @staticmethod
    def bind(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return torch.sign(v1 * v2 + 1e-5)

    def parse_and_encode(self, code_bytes: bytes) -> torch.Tensor:
        tree = self.parser.parse(code_bytes)
        return self._encode_node(tree.root_node)

    def encode_batch(self, code_snippets: list[bytes]) -> torch.Tensor:
        return torch.stack([self.parse_and_encode(s) for s in code_snippets]).to(self.device)
