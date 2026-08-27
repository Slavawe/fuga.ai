#!/usr/bin/env python3
"""Fuga Tokenizer: бестокеновый VSA-токенизатор.

Принципы (ТЗ):
  - Алфавит: байты 0-255 как базисные гипервекторы + композитные якоря
    из VSA-памяти (символы кода minGPT/JAX и т.д.).
  - Сегментация: trie longest-match по якорям; выход каждого токена =
    биполярный гипервектор (Phase Crystal) — ГОТОВ к H-JEPA/KAN без
    nn.Embedding.
  - Обратимость: Decode(Encode(Bytes)) == Bytes (бит-в-бит).
  - Нет OOV: неизвестные байты раскладываются до атомарных HV.
  - Проверка AST-границ: токены не режут синтаксические узлы кода.
"""

from __future__ import annotations


import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from antitf.rust_bridge import packed_to_torch
from fuga_memory import PersistentVSAMemory


class FugaTokenizer:
    def __init__(self, binder, mem_dirs=("fuga_memory_code", "fuga_memory_jax"),
                 max_anchors: int = 4000):
        self.binder = binder
        self.dim = 2048
        # байтовый базис 0-255
        self.anchors: dict[bytes, torch.Tensor] = {}
        for b in range(256):
            self.anchors[bytes([b])] = packed_to_torch(np.asarray(
                binder.bind_batch([[f"BYTE:{b}"]])))[0]
        # композитные якоря из VSA-памяти (символы кода)
        for mem_dir in mem_dirs:
            mem = PersistentVSAMemory(binder, directory=mem_dir)
            path = os.path.join(mem.dir, "fuga_memory.facts.jsonl")
            if not os.path.exists(path):
                continue
            import json
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = d["subject"].replace("code:", "", 1)
                    if len(self.anchors) >= max_anchors:
                        break
                    if 1 <= len(name.encode()) <= 24:
                        self.anchors[name.encode()] = packed_to_torch(
                            np.asarray(binder.bind_batch([[name]])))[0]
        # trie для longest-match
        self.trie = {}
        for token in self.anchors:
            node = self.trie
            for b in token:
                node = node.setdefault(b, {})
            node["$"] = token

    # ---------- сегментация ----------
    def encode(self, data: bytes) -> list[torch.Tensor]:
        """Жадный longest-match по trie якорей -> список биполярных HV."""
        tokens_hv = []
        i = 0
        n = len(data)
        while i < n:
            node = self.trie
            best = None
            j = i
            while j < n and data[j] in node:
                node = node[data[j]]
                j += 1
                if "$" in node:
                    best = node["$"]
            if best is None:          # OOV: атомарный байт
                best = data[i:i + 1]
                i += 1
            else:
                i += len(best)
            tokens_hv.append(self.anchors[best])
        return tokens_hv

    def encode_tokens(self, data: bytes) -> list[bytes]:
        """Список строк токенов (для decode/проверок)."""
        toks = []
        i = 0
        n = len(data)
        while i < n:
            node = self.trie
            best = None
            j = i
            while j < n and data[j] in node:
                node = node[data[j]]
                j += 1
                if "$" in node:
                    best = node["$"]
            if best is None:
                best = data[i:i + 1]
                i += 1
            else:
                i += len(best)
            toks.append(best)
        return toks

    @staticmethod
    def decode(tokens: list[bytes]) -> bytes:
        return b"".join(tokens)

    # ---------- проверки ----------
    def reversibility(self, samples: list[bytes]) -> bool:
        for s in samples:
            if self.decode(self.encode_tokens(s)) != s:
                return False
        return True

    def no_oov(self, data: bytes) -> bool:
        """Любой байт покрывается (нет неразложимых последовательностей)."""
        toks = self.encode_tokens(data)
        return all(t in self.anchors for t in toks)

    def ast_boundary_score(self, code: bytes, lang: str) -> float:
        """Доля границ токенов, НЕ разрезающих AST-узлы (идентификаторы)."""
        import re
        from astral.code_synth import _get_parser
        parser = _get_parser({"python": "tree_sitter_python",
                              "c": "tree_sitter_c",
                              "rust": "tree_sitter_rust"}[lang])
        tree = parser.parse(code)
        # границы идентификаторов из AST
        ident_boundaries = set()

        def walk(node):
            if node.type in ("identifier", "type_identifier"):
                r = node.byte_range
                ident_boundaries.add((r[0], r[1]))
            for c in node.children:
                walk(c)

        walk(tree.root_node)
        toks = self.encode_tokens(code)
        tok_boundaries = set()
        pos = 0
        for t in toks:
            tok_boundaries.add(pos)
            pos += len(t)
        # внутренние позиции идентификаторов, попавшие внутрь токена
        violated = 0
        total = 0
        for a, b in ident_boundaries:
            total += 1
            # если внутри (a,b) есть граница токена -> разрез идентификатора
            for tb in tok_boundaries:
                if a < tb < b:
                    violated += 1
                    break
        return 1.0 - violated / max(total, 1)

    def speed_lines_per_sec(self, text: bytes) -> float:
        lines = text.count(b"\n") + 1
        t0 = time.time()
        self.encode_tokens(text)
        dt = time.time() - t0
        return lines / max(dt, 1e-9)


def main():
    binder = fuga_core.HybridBinder(2048)
    tok = FugaTokenizer(binder)
    print(f"[tokenizer] якорей: {len(tok.anchors)} "
          f"(байт 256 + композитных {len(tok.anchors)-256})")

    # 1. обратимость (бит-в-бит)
    samples = [b"def parse(data): return len(data)",
               b"static inline unsigned long vmalloc_init(void){}",
               "hello мир 你好".encode(),
               b"\x00\xff\x01\xfe arbitrary binary"]
    print(f"[reversibility] {tok.reversibility(samples)}")

    # 2. нет OOV
    print(f"[no_oov] {tok.no_oov(b'random \x00\xff unknown sequence 12345')}")

    # 3. Phase Crystal: выход = биполярные HV
    hv = tok.encode(samples[0])
    print(f"[phase-crystal] {len(hv)} токенов -> {len(hv)} биполярных HV "
          f"({hv[0].shape[0]}-d), ready for H-JEPA")

    # 4. AST-границы (Python код не режется по идентификаторам)
    code = b"def parse_docs(data):\n    result = len(data)\n    return result\n"
    score = tok.ast_boundary_score(code, "python")
    print(f"[ast-boundaries] сохранено: {score:.2f}")

    # 5. скорость
    big = (b"def f(x): return x*2\n" * 500)
    sps = tok.speed_lines_per_sec(big)
    print(f"[speed] {sps:.0f} lines/sec (байт {len(big)})")


if __name__ == "__main__":
    main()
