
"""gpu_engine: устройство-агностичный VSA-поиск по фактам.

Факты -> биполярные HV [N, D] float32 в torch-тензоре. Поиск = matmul
(cosine), работает на CPU здесь и на CUDA на вашей машине через тот же
код (`--device cuda`). Метрика: latency поиска по всей базе + throughput.
"""

from __future__ import annotations

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch


import fuga_core
from antitf.rust_bridge import packed_to_torch


class VSAMemoryIndexer:
    def __init__(self, facts_path: str, dim: int = 2048,
                 device: str | None = None):
        self.binder = fuga_core.HybridBinder(dim)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.names: list[str] = []
        self.matrix: torch.Tensor | None = None
        self.load_facts(facts_path)

    def load_facts(self, facts_path: str):
        names = []
        with open(facts_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                names.append(d["subject"].replace("code:", "", 1))
        self.names = names
        # кодируем батчами
        hvs = []
        for s in range(0, len(names), 1000):
            chunk = [[n] for n in names[s:s + 1000]]
            pk = np.asarray(self.binder.bind_batch(chunk))
            hvs.append(packed_to_torch(pk))
        self.matrix = torch.cat(hvs).float().to(self.device)
        print(f"[gpu-engine] фактов={len(names)} устройство={self.device} "
              f"матрица={tuple(self.matrix.shape)}")

    @torch.no_grad()
    def search(self, query: str, topk: int = 5, chunk: int = 4096):
        """Префикс-поиск по имени + cosine по HV. Возвращает (names, sims)."""
        # ТОЧНЫЕ префиксные совпадения приоритетны (детерминировано),
        # векторный косинус — только для добивки top-k.
        q = query.lower()
        exact_idx = [i for i, n in enumerate(self.names) if q in n.lower()]
        qhv = packed_to_torch(np.asarray(
            self.binder.bind_batch([[query]])))[0].to(self.device)
        best = []
        for s in range(0, len(self.names), chunk):
            block = self.matrix[s:s + chunk]
            sims = block @ qhv
            for j in range(len(sims)):
                best.append((float(sims[j]), s + j))
        best.sort(reverse=True)
        ranked = list(dict.fromkeys(exact_idx + [i for _, i in best]))
        out_names = [self.names[i] for i in ranked[:topk]]
        out_sims = [best[0][0]] * 0 or [float(
            (self.matrix[i] @ qhv).item()) for i in ranked[:topk]]
        return out_names, out_sims


def benchmark(idx, queries, repeats=3):
    print("\n[bench] поиск по базе:")
    t0 = time.time()
    for _ in range(repeats):
        for q in queries:
            idx.search(q)
    dt = (time.time() - t0) / (repeats * len(queries))
    print(f"  средняя latency запроса: {dt*1000:.2f} ms "
          f"({len(idx.names)} фактов)")
    t0 = time.time()
    for _ in range(50):
        idx.search(queries[0])
    dt = (time.time() - t0) / 50
    print(f"  latency (прогрев): {dt*1000:.2f} ms")


if __name__ == "__main__":
    facts = sys.argv[1] if len(sys.argv) > 1 else \
        "fuga_memory_code/fuga_memory.facts.jsonl"
    idx = VSAMemoryIndexer(facts)
    for q in ("vmalloc", "schedule", "Gson", "parse"):
        names, sims = idx.search(q, 3)
        print(f"  '{q}': {list(zip(names, [f'{s:.3f}' for s in sims]))}")
    benchmark(idx, ["vmalloc", "schedule", "Gson", "parse", "add"])
