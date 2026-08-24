
"""MegaDataStreamer: интерливинг текстовых и кодовых потоков в VSA 32K.

Каналы:
  text — FineWeb-Edu (HF streaming, работает без токена)
  code — локальные исходники репозитория (.py/.rs): The Stack v2 gated,
         а строгие символьные структуры уже лежат на диске
Видео/аудио каналы — интерфейс ready, инжест через astral_env/UCF101.
"""

from __future__ import annotations
import glob
import itertools
import os
import random
import re

import numpy as np


class MegaDataStreamer:
    TEXT_REPO = "HuggingFaceFW/fineweb-edu"
    CODE_GLOBS = ["fuga-core/src/**/*.rs", "antitf/*.py", "src/ai/*.rs",
                  "astral/*.py"]

    def __init__(self, max_text: int | None = None):
        self.max_text = max_text
        self._text_iter = self._text_stream()
        self._code_pool = self._load_local_code()
        self._code_idx = 0

    # ---------- каналы ----------
    def _text_stream(self):
        from datasets import load_dataset
        ds = load_dataset(self.TEXT_REPO, name="sample-10BT", split="train",
                          streaming=True)
        count = 0
        for row in ds:
            txt = row["text"][:600]
            if len(txt) > 80:
                yield {"type": "text", "bytes": txt.encode("utf-8", "ignore")}
                count += 1
                if self.max_text and count >= self.max_text:
                    return

    def _load_local_code(self) -> list[bytes]:
        out = []
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for pattern in self.CODE_GLOBS:
            for path in glob.glob(os.path.join(root, pattern), recursive=True):
                try:
                    src = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                # режем на блоки по функциям/структурам (~строки между пустыми)
                blocks = re.split(r"\n\s*\n", src)
                for b in blocks:
                    b = b.strip()
                    if 60 <= len(b) <= 1200:
                        out.append(b.encode("utf-8", "ignore"))
        random.shuffle(out)
        return out

    def code_bytes(self):
        while True:
            if not self._code_pool:
                return
            yield {"type": "code", "bytes": self._code_pool[self._code_idx % len(self._code_pool)]}
            self._code_idx += 1

    def interleaved(self):
        """Чередование текст/код; текстовый поток конечен по max_text."""
        text_gen = self._text_iter
        code_gen = self.code_bytes()
        toggle = random.Random(0).random() < 0.5
        for t, c in zip(text_gen, itertools.cycle(code_gen)):
            yield t if toggle else c
            yield c if toggle else t
            toggle = not toggle


import itertools  # noqa: E402
