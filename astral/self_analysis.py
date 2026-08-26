
"""Self-Analysis Engine: рефлексия над собственными генерациями.

Цикл: кандидаты -> критик (coverage/accept/integrity/novelty) ->
диагноз слабого измерения -> правило избежания -> self_analysis_rules.jsonl.
Правила накапливаются между сессиями и подмешиваются в вербализатор.
"""

from __future__ import annotations

from __future__ import annotations

import json
import os
import time
from collections import Counter


class SelfAnalysisEngine:
    def __init__(self, critic, memory_dir: str = "fuga_memory"):
        self.critic = critic
        self.rules_path = os.path.join(memory_dir,
                                       "self_analysis_rules.jsonl")
        self.rules: list[dict] = []
        self._load_rules()

    def _load_rules(self):
        if not os.path.exists(self.rules_path):
            return
        with open(self.rules_path, encoding="utf-8") as f:
            for line in f:
                try:
                    self.rules.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def diagnose(self, slots: list[str], text: str, scores: dict) -> dict | None:
        """Слабейшее измерение -> правило избежания. None если всё в норме."""
        fails = []
        if scores.get("accept", 1) < 1.0:
            fails.append(("acceptability",
                          "шаблон дал неестественную поверхность"))
        if scores.get("coverage", 1) < 0.5:
            fails.append(("coverage", "слоты не просели в текст"))
        if scores.get("integrity", 1) < 0.5:
            fails.append(("integrity", "round-trip развязки потерял слова"))
        if not fails or not text.strip():
            return None
        dim, reason = min(fails, key=lambda f: {
            "acceptability": scores.get("accept", 1),
            "coverage": scores.get("coverage", 1),
            "integrity": scores.get("integrity", 1)}[f[0]])
        return {"dimension": dim, "reason": reason,
                "slots": slots, "text": text}

    def reflect_batch(self, slot_sets: list[list[str]],
                      candidate_sets: list[list[str]]) -> dict:
        """Для каждого набора слотов выбирает лучшего кандидата и извлекает
        правила из провалов остальных."""
        best_list, rules_new = [], []
        dim_fail_counter = Counter()
        for slots, cands in zip(slot_sets, candidate_sets):
            scored = []
            for cand in cands:
                sc = self.critic.score(slots, cand)
                scored.append((sc["total"], cand, sc))
            scored.sort(key=lambda t: -t[0])
            best_total, best_text, best_sc = scored[0]
            best_list.append((best_text, best_sc))

            for total, cand, sc in scored[1:]:
                diag = self.diagnose(slots, cand, sc)
                if diag is not None and sc["total"] < best_total:
                    dim_fail_counter[diag["dimension"]] += 1
                    rules_new.append({
                        "rule": f"избегать паттерна '{cand[:60]}' "
                                f"({diag['dimension']}: {diag['reason']})",
                        "better_example": best_text[:80],
                        "ts": time.time(),
                    })
        self.rules.extend(rules_new)
        self._save_rules(rules_new)
        return {"best": best_list,
                "new_rules": len(rules_new),
                "fail_dims": dict(dim_fail_counter)}

    def applicable_rules(self, limit: int = 10) -> list[str]:
        return [r["rule"] for r in self.rules[-limit:]]

    def _save_rules(self, new_rules: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.rules_path), exist_ok=True)
        with open(self.rules_path, "a", encoding="utf-8") as f:
            for r in new_rules:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


import time  # noqa: E402
