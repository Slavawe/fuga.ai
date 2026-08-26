
"""Прагматический датасет из реальных диалогов OASST1 (dataset_vault).

Триада: {context, response, lang, register, tone_markers}
Регистр определяется эвристиками формальности (RU/EN) — v0 без модели.
"""

from __future__ import annotations

from __future__ import annotations

import json
import os
import re

VAULT = os.path.join("dataset_vault", "01_everyday_dialogues",
                     "open_assistant_real.jsonl")

# --- эвристики регистра ---
FORMAL_RU = {"вы", "ваш", "ваше", "пожалуйста", "будьте", "добрый день",
             "уважаемый", "благодарю", "признателен"}
CASUAL_RU = {"ты", "твоё", "твой", "чё", "щас", "короче", "блин", "кстати",
             "ваще", "норм", "ага", "ну", "хах", "лол"}
FORMAL_EN = {"please", "kindly", "would you", "dear", "sincerely",
             "i would like", "thank you"}
CASUAL_EN = {"gonna", "wanna", "yeah", "lol", "dude", "hey", "gotta",
             "kinda", "stuff", "meh"}


def detect_register(text: str, lang: str) -> tuple[str, list[str]]:
    t = text.lower()
    markers = []
    src_f = FORMAL_RU if lang == "ru" else FORMAL_EN
    src_c = CASUAL_RU if lang == "ru" else CASUAL_EN
    formal_hits = [w for w in src_f if w in t]
    casual_hits = [w for w in src_c if w in t]
    if lang == "ru":
        if re.search(r"\bвы\b|\bваш", t):
            formal_hits.append("вы-форма")
        if re.search(r"\bты\b|\bтво", t):
            casual_hits.append("ты-форма")
    exclam = text.count("!") >= 2
    if formal_hits:
        markers += formal_hits[:2]
    if casual_hits:
        markers += casual_hits[:2]
    if not formal_hits and not casual_hits:
        return "neutral", []
    if len(casual_hits) > len(formal_hits) or exclam and casual_hits:
        return "casual", casual_hits[:3]
    return "formal", formal_hits[:3]


def build_pragmatic_dataset(out_path="dataset_vault/04_pragmatic/"
                                        "pragmatic_triads.jsonl",
                            limit=20000):
    src = os.path.join("dataset_vault", "01_everyday_dialogues",
                       "open_assistant_real.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = written = 0
    reg_count = {"formal": 0, "casual": 0, "neutral": 0}
    with open(src, encoding="utf-8") as f, open(out_path, "w",
                                                encoding="utf-8") as out:
        for line in f:
            d = json.loads(line)
            n += 1
            ctx, resp = d["context"].strip(), d["response"].strip()
            if len(ctx) < 10 or len(resp) < 5:
                continue
            if len(resp) > 600 or len(ctx) > 800:
                continue
            lang = d["lang"]
            reg, markers = detect_register(ctx + " " + resp, lang)
            reg_count[reg] += 1
            out.write(json.dumps({
                "lang": lang,
                "context": ctx[:800],
                "response": resp[:600],
                "register": reg,
                "tone_markers": markers,
            }, ensure_ascii=False) + "\n")
            written += 1
            if written >= limit:
                break
    return written, reg_count


if __name__ == "__main__":
    w, rc = build_pragmatic_dataset()
    print(f"[pragmatic] строк исходника={w}, записано триад={w}")
    print(f"  регистры: {rc}")
