"""Unified Cognitive Pipeline: данные → VSA → BIM → GPU.

Единый входной конвейер (Single Pipeline Input):
  Data (bytes/text/code)
    → Tree-Sitter / Byte Stream
      → fuga_core (PyO3, Rust FastVSA)
        → Phase Crystal VSA-связывание
          → BIM (bimbuf_v2.bin) регистрация
            → GPU-обучение (sync_channel backpressure)

Условие стабильности: порог резонатора cos(θ) >= 0.75 — исключает
смешивание шума при связывании несвязанных понятий.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from fuga_core import FastVSA, HybridBinder

# Порог резонатора: ниже — фазовый кристалл сливает несвязанные
# понятия в один шум (см. docs/decisions/003-byte-level-main-path.md).
RESONANCE_THRESHOLD = 0.75

# Путь BIM-индекса (bimbuf_v2.bin)
BIM_BUFFER_PATH = "bimbuf_v2.bin"


@dataclass
class PipelineSample:
    """Один обработанный образец конвейера."""

    source: str                 # исходный текст/код
    kind: str = "text"          # text | code
    hv: np.ndarray | None = None  # биполярный гипервектор [-1, +1]
    resonance: float = 0.0      # cos(θ) стабильности
    stable: bool = False        # resonance >= RESONANCE_THRESHOLD
    meta: dict = field(default_factory=dict)


class UnifiedPipeline:
    """Единый когнитивный конвейер: инжест → VSA → BIM.

    Все входные данные проходят ОДИН путь через Rust-ядро
    (fuga_core.FastVSA), никаких параллельных VSA-реализаций
    из Python (см. docs/architecture.md, Source of Truth).
    """

    def __init__(
        self,
        dim: int = 2048,
        binder: HybridBinder | None = None,
        bim_path: str = BIM_BUFFER_PATH,
        resonance_threshold: float = RESONANCE_THRESHOLD,
    ):
        self.dim = dim
        self.binder = binder or HybridBinder(dim)
        self.bim_path = bim_path
        self.threshold = resonance_threshold
        self._samples: list[PipelineSample] = []
        self._bim_entries: list[dict] = []

    # ── Инжест ──────────────────────────────────────────────────
    def ingest_text(self, text: str, kind: str = "text") -> PipelineSample:
        """Байтовый инжест текста через Rust-ядро."""
        sample = self._encode(text, kind="text")
        return self._register(sample)

    def ingest_code(self, path: str) -> PipelineSample:
        """Инжест кода: читает файл, кодирует как байтовый поток."""
        with open(path, encoding="utf-8", errors="ignore") as f:
            code = f.read()
        sample = self._encode(code, kind="code", meta={"path": path})
        return self._register(sample)

    def ingest_bytes(self, data: bytes, kind: str = "bytes") -> PipelineSample:
        """Инжест сырых байтов (произвольный бинарный поток)."""
        sample = self._encode(data.decode("utf-8", errors="replace"), kind=kind)
        return self._register(sample)

    # ── Ядро ────────────────────────────────────────────────────
    def _encode(self, text: str, kind: str, meta: dict | None = None) -> PipelineSample:
        """Кодирование текста в биполярный HV через Rust-ядро.

        Путь: text → байтовые чанки → bind_batch (Rust) → bundle (Rust)
        → packed_to_torch (±1 bipolar). Никаких Python-реализаций VSA.
        """
        chunks = [text[i : i + 8] for i in range(0, max(len(text), 1), 8)]
        hv = self._bundle_chunks(chunks)
        sample = PipelineSample(
            source=text,
            kind=kind,
            hv=hv,
            meta=meta or {},
        )
        return sample

    def _bundle_chunks(self, chunks: list[str]) -> np.ndarray:
        """Бандлинг чанков через Rust FastVSA (packed u64, мажоритарно).

        Возвращает биполярный HV [-1, +1] размерности self.dim.
        """
        if not chunks:
            return np.ones(self.dim, dtype=np.float32)
        # Позиционная свёртка: связываем каждый чанк с его позицией
        words: list[list[str]] = []
        for i, ch in enumerate(chunks):
            words.append([f"{ch}#{i}"])
        packed = self.binder.bind_batch(words)  # [N, dim/64] u64
        # Бандлинг: XOR-сумма с накоплением через FastVSA.bundle
        v = FastVSA(self.dim)
        states = [packed[i] for i in range(packed.shape[0])]
        bundled = v.bundle(states) if len(states) > 1 else states[0]
        # Конверсия packed u64 → биполярный ±1
        from antitf.rust_bridge import packed_to_torch

        hv = packed_to_torch(bundled[None])[0]  # [dim] ±1
        return np.asarray(hv, dtype=np.float32)

    def _register(self, sample: PipelineSample) -> PipelineSample:
        """Проверка стабильности (резонанс) и запись в BIM.

        Резонанс = косинус HV к памяти, НОРМИРОВАННЫЙ на базлайн
        случайного шума: resonance = (cos_hv_mem - cos_noise) /
        (1 - cos_noise). Так порог 0.75 имеет смысл независимо от
        размерности и схожести корпуса.
        """
        # Базлайн: косинус к случайному биполярному HV
        rng = np.random.default_rng(0)
        noise = rng.choice([-1.0, 1.0], size=self.dim)
        cos_noise = float(
            np.dot(sample.hv, noise) / (np.linalg.norm(sample.hv) * np.linalg.norm(noise))
        )

        if self._samples:
            memory_avg = np.mean(
                np.stack([s.hv for s in self._samples[-64:]]), axis=0
            )
            denom = np.linalg.norm(memory_avg) * np.linalg.norm(sample.hv)
            cos_mem = (
                float(np.dot(sample.hv, memory_avg) / denom) if denom > 0 else 0.0
            )
        else:
            cos_mem = 1.0  # первый образец — всегда стабилен

        # Нормализация на базлайн шума
        sample.resonance = (cos_mem - cos_noise) / max(1.0 - cos_noise, 1e-9)
        sample.resonance = max(-1.0, min(1.0, sample.resonance))

        sample.stable = sample.resonance >= self.threshold
        self._samples.append(sample)

        if sample.stable:
            self._bim_entries.append(
                {
                    "kind": sample.kind,
                    "resonance": round(sample.resonance, 4),
                    "cos_mem": round(cos_mem, 4),
                    "cos_noise": round(cos_noise, 4),
                }
            )
            self._flush_bim()

        return sample

    # ── BIM ─────────────────────────────────────────────────────
    def _flush_bim(self) -> None:
        """Запись BIM-буфера на диск (bimbuf_v2.bin)."""
        try:
            os.makedirs(os.path.dirname(self.bim_path) or ".", exist_ok=True)
            with open(self.bim_path, "w") as f:
                for entry in self._bim_entries[-1000:]:
                    f.write(f"{entry}\n")
        except OSError as e:
            print(f"[pipeline] BIM flush failed: {e}")

    def bim_stats(self) -> dict:
        """Сводка BIM-буфера."""
        stable = sum(1 for s in self._samples if s.stable)
        rejected = len(self._samples) - stable
        return {
            "total": len(self._samples),
            "stable": stable,
            "rejected": rejected,
            "threshold": self.threshold,
            "bim_entries": len(self._bim_entries),
        }

    # ── GPU-мост ────────────────────────────────────────────────
    def to_torch(self) -> torch.Tensor:
        """HV-матрица [N, dim] для GPU-конвейера (batch_delta)."""
        hvs = [s.hv for s in self._samples if s.hv is not None]
        if not hvs:
            return torch.zeros(0, self.dim)
        return torch.from_numpy(np.stack(hvs)).float()

    def report(self) -> str:
        """Человекочитаемый отчёт о состоянии конвейера."""
        stats = self.bim_stats()
        return (
            f"[UnifiedPipeline] dim={self.dim} threshold={self.threshold}\n"
            f"  samples: {stats['total']} | stable: {stats['stable']} "
            f"| rejected(шум): {stats['rejected']}\n"
            f"  bim_entries: {stats['bim_entries']} -> {self.bim_path}"
        )


def smoke_test() -> dict:
    """E2E smoke: 3 образца через единый конвейер."""
    pipe = UnifiedPipeline(dim=1024, bim_path="/tmp/bimbuf_smoke.bin")
    results = {}
    for name, text in [
        ("text1", "the force of gravity is proportional to mass"),
        ("code1", "fn main() { let x = 4; }"),
        ("noise", "zxqjwv kqplmn rtyuio asdfgh 1234567890 qwertyuiop"),
    ]:
        s = pipe.ingest_text(text)
        results[name] = {
            "stable": s.stable,
            "resonance": round(s.resonance, 3),
        }
    results["stats"] = pipe.bim_stats()
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
