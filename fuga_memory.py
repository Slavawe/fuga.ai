
"""Persistent VSA Memory: эпизодическая и семантическая память между сессиями.

Формат хранения:
  fuga_memory.facts.jsonl  — факты {lang, subject, relation, object}
  fuga_memory.episodes.jsonl — диалоговые эпизоды {user, response, ts}
  fuga_memory.context.npy  — затухающий HV-аккумулятор глобальной истории

Контекстный регистр по формуле карты v2.0:
  HV_ctx^(t) = Decay * HV_ctx^(t-1) + Bind(HV_входа)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch


class PersistentVSAMemory:
    def __init__(self, binder, directory: str = "fuga_memory", decay: float = 0.7):
        self.binder = binder
        self.dir = directory
        self.decay = decay
        os.makedirs(directory, exist_ok=True)
        self.facts_path = os.path.join(directory, "fuga_memory.facts.jsonl")
        self.episodes_path = os.path.join(directory, "fuga_memory.episodes.jsonl")
        self.ctx_path = os.path.join(directory, "fuga_memory.context.npy")
        self._seen_facts: set[tuple] = set()

    # ---------- факты ----------
    def add_fact(self, lang: str, subject: str, relation: str, obj: str,
                 dedupe_key: tuple | None = None) -> bool:
        key = dedupe_key or (lang, subject, relation, obj)
        if key in self._seen_facts:
            return False
        with open(self.facts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"lang": lang, "subject": subject,
                                "relation": relation, "object": obj,
                                "ts": time.time()}, ensure_ascii=False) + "\n")
        self._seen_facts.add(key)
        return True

    def load_facts_into(self, engine_memory: dict) -> int:
        """Прогрузка сохранённых фактов в живую память чат-движка."""
        n = 0
        if not os.path.exists(self.facts_path):
            return 0
        with open(self.facts_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                s = "_".join(d["subject"].lower().split())
                o = "_".join(d["object"].lower().split())
                if not s or not o:
                    continue
                engine_memory.setdefault(d["lang"], {}).setdefault(s, []).append(
                    (d["relation"], o))
                self._seen_facts.add((d["lang"], s, d["relation"], o))
                n += 1
        return n

    # ---------- эпизоды ----------
    def add_episode(self, user_text: str, response_text: str) -> None:
        with open(self.episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"user": user_text, "response": response_text,
                                "ts": time.time()}, ensure_ascii=False) + "\n")

    def iter_episodes(self):
        if not os.path.exists(self.episodes_path):
            return
        with open(self.episodes_path, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # ---------- контекстный аккумулятор ----------
    @torch.no_grad()
    def update_context(self, hv_bipolar: torch.Tensor) -> torch.Tensor:
        ctx = self.get_context()
        new = self.decay * ctx + hv_bipolar.float()
        new = torch.sign(new + 1e-5)
        new[new == 0] = 1
        np.save(self.ctx_path, new.numpy())
        return new

    def get_context(self) -> torch.Tensor:
        if os.path.exists(self.ctx_path):
            return torch.from_numpy(np.load(self.ctx_path)).float()
        return torch.ones(2048)

    # ---------- рефлексии (для Self-Reflective Loop) ----------
    def add_reflection(self, slots: list[str], text: str, scores: dict) -> bool:
        path = os.path.join(self.dir, "self_reflection.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"slots": slots, "text": text,
                                "scores": scores, "ts": time.time()},
                               ensure_ascii=False) + "\n")
        return True

    def iter_reflections(self, min_score: float | None = None):
        path = os.path.join(self.dir, "self_reflection.jsonl")
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if min_score is None or d.get("scores", {}).get("total", 0) >= min_score:
                    yield d
