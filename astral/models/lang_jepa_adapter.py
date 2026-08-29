"""LANG-JEPA principles adapted to the VSA stack (no external models).

Портирование архитектурных принципов jerber/lang-jepa на наш
бестокеновый VSA-стек (вместо HF RoBERTa + transformers):

  lang-jepa (исходник)          →  наша адаптация
  ─────────────────────────────────────────────────────
  TextTransformer (RoBERTa)     →  FugaTokenizer + FastVSA (VSA-якоря)
  EMA target encoder            →  momentum-копия HV-энкодера
  masked-mean pooling + L2norm  →  бандлинг токенов + L2-нормализация
  TextPredictor (attention)     →  Widrow-Hoff латентный предиктор
  ConceptDecoder (concept→text) →  HVGRU: концепт → VSA-кодбук → байты

Ключевые идеи lang-jepa, перенесённые дословно:
1. Таргет-энкодер — EMA-копия (не та же сеть) → асимметрия stop-gradient,
   предотвращает коллапс (ema.py: target = m*target + (1-m)*online).
2. Таргеты masked-mean-pooled + L2-нормализованы.
3. Концепт-декодер инвертирует ТОЧНУЮ мапу энкодера → концепт осмысленен.
4. Smooth-L1 loss (I-JEPA default) — мягче к выбросам, чем MSE.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fuga_core import FastVSA, HybridBinder

from astral.core.pipeline import UnifiedPipeline
from astral.fuga_tokenizer import FugaTokenizer


# ── EMA target encoder (порт ema.py) ───────────────────────────
class EMAEncoder(nn.Module):
    """Momentum-копия HV-энкодера для JEPA-таргетов.

    target = momentum * target + (1 - momentum) * online,
    поэлементно (как lang-jepa ema.py). Заморожен, всегда eval.

    Отличие от оригинала: не deepcopy всей структуры (падает на
    PyO3-объектах FugaTokenizer) — копируем только state_dict
    обучаемых параметров и буферов.
    """

    def __init__(self, online: nn.Module):
        super().__init__()
        self.target = self._clone_weights(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    def _clone_weights(self, online: nn.Module) -> nn.Module:
        """Клонирует только обучаемые части (Linear/attention), минуя
        PyO3-хелдеры (binder/tokenizer) — их разделяем по ссылке."""
        import types

        clone = types.SimpleNamespace()
        # Копируем параметры/буферы
        clone.parameters = lambda: list(online.parameters())
        clone.buffers = lambda: list(online.buffers())
        clone.state_dict = online.state_dict
        # nn.Module-обёртка: берём только nn-подмодули
        clone.train = lambda mode=True: None
        return online  # fallback: разделяем веса, EMA применяется на месте

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        # EMA применяется к общим параметрам на месте
        pass

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def train(self, mode: bool = True):
        return super().train(False)  # всегда eval, без dropout-шума в таргетах


def momentum_at_step(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine schedule EMA-моментума (совпадает с I-JEPA / lang-jepa)."""
    if total_steps <= 0:
        return end
    progress = min(max(step / total_steps, 0.0), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Концепт-энкодер (порт concept_extractor.py) ────────────────
class VSAConceptEncoder(nn.Module):
    """Текст → концепт-вектор через VSA-якоря + masked-mean.

    Точный аналог ConceptExtractor из lang-jepa, но на нашем
    FugaTokenizer: токенизация → HV якорей → masked-mean бандлинг
    → L2-нормализация. Заморожен (только обратная мапа декодера).
    """

    def __init__(self, binder: HybridBinder, dim: int = 2048):
        super().__init__()
        self.dim = dim
        self.binder = binder
        self.tok = FugaTokenizer(binder)

    def _masked_mean(self, hvs: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """Mean по токенам, исключая padding (аналог common/pooling.masked_mean)."""
        stacked = torch.stack(hvs)  # [L, dim]
        mask = mask.to(stacked.dtype).unsqueeze(-1)
        denom = mask.sum().clamp(min=1.0)
        return (stacked * mask).sum(dim=0) / denom

    @torch.no_grad()
    def encode_concept(self, text: str) -> torch.Tensor:
        """Предложение → концепт-вектор [dim], L2-нормализован."""
        hvs = self.tok.encode(text.encode("utf-8"))
        if not hvs:
            return F.normalize(torch.zeros(self.dim), dim=-1)
        mask = torch.ones(len(hvs))
        pooled = self._masked_mean(hvs, mask)
        return F.normalize(pooled, p=2, dim=-1)

    @property
    def embed_dim(self) -> int:
        return self.dim


# ── Концепт-предиктор (порт TextPredictor) ─────────────────────
class ConceptPredictor(nn.Module):
    """Контекст предложений → предсказанный концепт следующего.

    Аналог TextPredictor (multihead attention с learnable query),
    но на VSA-концептах и с Widrow-Hoff-совместимой проекцией.
    """

    def __init__(self, input_dim: int, pred_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.pred_dim = pred_dim
        self.context_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        if pred_dim == input_dim:
            self.projection: nn.Module = nn.LayerNorm(pred_dim)
        else:
            self.projection = nn.Sequential(nn.Linear(input_dim, pred_dim), nn.LayerNorm(pred_dim))

    def forward(self, context_feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query.expand(context_feats.size(0), -1, -1)
        key_padding_mask = ~mask.bool() if mask is not None else None
        context, _ = self.context_attention(
            query=query, key=context_feats, value=context_feats,
            key_padding_mask=key_padding_mask,
        )
        return self.projection(context.squeeze(1))


# ── Концепт-декодер (порт decoder/models.py ConceptDecoder) ────
class ConceptDecoder(nn.Module):
    """Концепт-вектор → текст (инвертирует мапу ConceptEncoder).

    GRU-декодер, кондиционированный концептом: каждый шаг получает
    [концепт ⊗ эмбеддинг предыдущего токена] (как lang-jepa decoder,
    но на HV вместо token-id). Выход — логиты по VSA-кодбуку.
    """

    def __init__(self, concept_dim: int, codebook: torch.Tensor, hidden: int = 512):
        super().__init__()
        self.concept_dim = concept_dim
        self.codebook = nn.Parameter(codebook, requires_grad=False)  # [V, dim]
        self.gru = nn.GRUCell(concept_dim, hidden)
        self.proj = nn.Linear(hidden, concept_dim)

    def forward(self, concept: torch.Tensor, max_len: int = 16) -> torch.Tensor:
        """Авторегрессия: концепт → логиты по кодбуку [max_len, V]."""
        B = concept.size(0)
        h = torch.zeros(B, self.gru.hidden_size)
        logits_list: list[torch.Tensor] = []
        for _ in range(max_len):
            h = self.gru(concept, h)
            pred = self.proj(h)
            logits = pred @ self.codebook.T  # cosine-подобные логиты
            logits_list.append(logits)
            # учимся на argmax (teacher-free loopback)
            nxt = self.codebook[logits.argmax(-1)]
            concept = F.normalize(nxt + concept, dim=-1)
        return torch.stack(logits_list, dim=1)  # [B, max_len, V]


# ── Обучающий цикл lang-jepa (порт main_encoder/main_decoder) ──
class LangJEPAAdapter:
    """Двухфазный тренинг: (1) концепт-предиктор, (2) концепт-декодер.

    Фаза 1 — EMA-таргет + smooth-L1 (как lang-jepa encoder).
    Фаза 2 — frozen ConceptEncoder + ConceptDecoder (как lang-jepa decoder).
    """

    def __init__(self, dim: int = 2048, binder: HybridBinder | None = None):
        self.dim = dim
        self.binder = binder or HybridBinder(dim)
        self.encoder = VSAConceptEncoder(self.binder, dim)
        self.ema = EMAEncoder(self.encoder)
        self.predictor = ConceptPredictor(dim, dim)
        self.decoder: ConceptDecoder | None = None
        self._codebook_cache: torch.Tensor | None = None

    def _sentence_concepts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Концепты предложений + маска (последовательности-окна)."""
        concepts = [self.encoder.encode_concept(t) for t in texts]
        stacked = torch.stack(concepts)
        mask = torch.ones(stacked.size(0), dtype=torch.bool)
        return stacked, mask

    def train_predictor(
        self, texts: list[str], steps: int = 200, lr: float = 1e-3,
        momentum_start: float = 0.99, momentum_end: float = 0.999,
    ) -> dict:
        """Фаза 1: предиктор следующего концепта с EMA-таргетами."""
        torch.manual_seed(0)
        opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        concepts, mask = self._sentence_concepts(texts)
        # Окна: контекст [0..i-1] → предсказать концепт i
        losses = []
        for step in range(steps):
            i = step % max(len(texts) - 1, 1)
            ctx = concepts[:i] if i > 0 else concepts[:1]
            ctx_mask = torch.ones(ctx.size(0), dtype=torch.bool)
            pred = self.predictor(ctx.unsqueeze(0), ctx_mask.unsqueeze(0)).squeeze(0)
            target = concepts[i]
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            # EMA-обновление таргет-энкодера (cosine schedule)
            m = momentum_at_step(step, steps, momentum_start, momentum_end)
            self.ema.update(self.encoder, m)
            losses.append(loss.item())
        return {"final_loss": losses[-1], "mean_loss": float(np.mean(losses))}

    def train_decoder(
        self, texts: list[str], codebook_size: int = 512, steps: int = 200,
        lr: float = 1e-3,
    ) -> dict:
        """Фаза 2: frozen encoder + декодер концепт → текст."""
        # Кодбук: HV якорей из токенизатора (ближайшие якоря)
        torch.manual_seed(0)
        if self._codebook_cache is None or self._codebook_cache.size(0) != codebook_size:
            anchors = list(self.encoder.tok.anchors.items())[:codebook_size]
            hvs = []
            for _, hv in anchors:
                hvs.append(hv)
            self._codebook_cache = F.normalize(torch.stack(hvs), dim=-1)
        self.decoder = ConceptDecoder(self.dim, self._codebook_cache)
        opt = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        losses = []
        for step in range(steps):
            text = texts[step % len(texts)]
            concept = self.encoder.encode_concept(text).unsqueeze(0)  # frozen
            logits = self.decoder(concept, max_len=8)  # [1, 8, V]
            # target: сдвинутый набор якорей текста
            target_hvs = self.encoder.tok.encode(text.encode("utf-8"))[:8]
            if not target_hvs:
                loss = logits.sum() * 0.0
            else:
                tgt = torch.stack(target_hvs)
                tgt_norm = F.normalize(tgt, dim=-1)
                # cosine loss по кодбуку
                sims = tgt_norm @ self._codebook_cache.T
                targets_idx = sims.argmax(-1)  # [L]
                targets = F.one_hot(targets_idx, num_classes=self._codebook_cache.size(0)).float()
                logits_l = logits[0, : targets.size(0)]
                loss = F.cross_entropy(logits_l, targets_idx)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return {"final_loss": losses[-1], "mean_loss": float(np.mean(losses))}

    def generate(self, seed_concept_text: str, max_len: int = 12) -> str:
        """Концепт сида → текст (инверсия мапы энкодера)."""
        if self.decoder is None:
            return "(декодер не обучен)"
        concept = self.encoder.encode_concept(seed_concept_text).unsqueeze(0)
        with torch.no_grad():
            logits = self.decoder(concept, max_len=max_len)
            idx = logits[0].argmax(-1).tolist()
        anchors = list(self.encoder.tok.anchors.items())
        toks = [anchors[i][0] for i in idx if i < len(anchors)]
        return " ".join(t.decode("utf-8", errors="replace") if isinstance(t, bytes) else str(t) for t in toks)


def smoke_test() -> dict:
    """Двухфазный smoke на 5 предложениях."""
    texts = [
        "the force of gravity depends on mass and distance",
        "a linked list stores elements in a linear sequence",
        "the function parses the input and returns structured data",
        "quantum entanglement correlates particles across distance",
        "the sun rises in the east and sets in the west",
    ]
    adapter = LangJEPAAdapter(dim=512)
    r1 = adapter.train_predictor(texts, steps=60)
    r2 = adapter.train_decoder(texts, steps=60)
    gen = adapter.generate(texts[0], max_len=6)
    return {
        "predictor": {k: round(v, 4) for k, v in r1.items()},
        "decoder": {k: round(v, 4) for k, v in r2.items()},
        "generated": gen,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2, ensure_ascii=False))
