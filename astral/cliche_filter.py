
"""Фильтр «нейросетевого мусора»: клише LLM в RU/EN.

Интегрируется в критик Self-Analysis Engine как жёсткий отказ
(acceptability=0 при обнаружении штампа).
"""

from __future__ import annotations

from __future__ import annotations

import re

CLICHES_RU = [
    r"я\s+(?:—\s*)?(?:ии|искусственный интеллект|языковая модель)",
    r"как\s+(?:ии|языкова[яя] модель)",
    r"я не могу", r"к сожалению,", r"однако стоит отметить",
    r"важно отметить", r"стоит подчеркнуть", r"в заключение",
    r"надеюсь,? это помогло", r"если у вас есть (?:ещё |дополнительные )?вопросы",
    r"с удовольствием помогу", r"рад(?:а)? помочь",
    r"давайте разберёмся", r"погрузимся в",
]
CLICHES_EN = [
    r"i'?m (?:an )?(?:ai|language model)", r"as an ai",
    r"i cannot|i can't assist", r"it'?s worth noting",
    r"in conclusion", r"i hope this helps",
    r"let me know if you have any", r"certainly! here'?s",
    r"here'?s what i found", r"dive into", r"delve into",
]

_COMPILED = [(lang, re.compile(p, re.IGNORECASE))
             for lang, pats in (("ru", CLICHES_RU), ("en", CLICHES_EN))
             for p in pats]


def detect_cliche(text: str) -> list[str]:
    hits = []
    for lang, _ in _COMPILED[:0]:
        pass
    for src, pat in _COMPILED:
        m = pat.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def cliche_free(text: str) -> bool:
    return not detect_cliche(text)


class ClicheFilter:
    """Жёсткое измерение для критика: штамп = acceptability 0."""

    name = "cliche"

    def score(self, slots, text) -> float:
        return 1.0 if cliche_free(text) else 0.0
