
"""Voynich MS 408 ingest: EVA-транскрипция -> VSA-слоты Единого Фронта.

Важно: сырые UTF-8 байты текста, нарезанные по 4096, — НЕ гипервекторы.
EVA-токены кодируются байндером (bind+rotate+bundle) как все каналы стека.
Корпус f1r здесь — валидация пайплайна; для исследования нужен полный
интерлинейный архив EVA (voynich.nu / IVTTF).
"""

from __future__ import annotations

from __future__ import annotations

import re

import numpy as np


class VoynichMainManuscript:
    FOLIO_F1R = """
    fachys ykal ar atey chodaiin pchey qokedy qokain dal chedy
    bshey qokain shol or shor cphy sholdy qokychy qokain or chodaiin
    ychey qokaiin or chedy qokain sholdy qokaiin dal chedy qokain dal
    dair chesh qokain or chedy qokain sholdy qokaiin dal chedy
    pol chedy qokain sholdy qokychy qokain or chodaiin dal
    qokain or chedy qokain sholdy qokaiin dal chedy qokain
    ychey qokaiin or chedy qokain sholdy qokaiin dal chedy
    """

    def __init__(self, corpus_text: str | None = None):
        self.text = corpus_text if corpus_text is not None else self.FOLIO_F1R

    def lines(self) -> list[list[str]]:
        out = []
        for ln in self.text.strip().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = re.findall(r"[a-z]+", ln.lower())
            if toks:
                out.append(toks)
        return out

    def flat_tokens(self) -> list[str]:
        return [t for ln in self.lines() for t in ln]

    # ---------- VSA ----------
    def line_hv(self, binder, line_tokens):
        pk = np.asarray(binder.bind_batch([line_tokens[:32]]))
        from antitf.rust_bridge import packed_to_torch
        return packed_to_torch(pk)[0]

    def slot_hvs(self, binder, line_tokens):
        """Пословные слоты строки (уже повёрнутые по позиции)."""
        pk = np.asarray(binder.extract_word_hvs_batch([line_tokens], 12))
        from antitf.rust_bridge import packed_to_torch
        return packed_to_torch(pk)[0]

    # ---------- структурная статистика ----------
    def stats(self) -> dict:
        toks = self.flat_tokens()
        vocab = sorted(set(toks))
        import math
        freq: dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        # энтропия Шеннона по токенам
        H = -sum((c / len(toks)) * math.log2(c / len(toks)) for c in freq.values())
        # биграммные переходы
        bigrams = set()
        for ln in self.lines():
            bigrams.update(zip(ln, ln[1:]))
        return {
            "tokens": len(toks),
            "vocab": len(vocab),
            "token_entropy_bits": round(H, 3),
            "unique_bigrams": len(bigrams),
            "top5": sorted(freq.items(), key=lambda kv: -kv[1])[:5],
        }
