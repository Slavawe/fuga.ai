#!/usr/bin/env python3
# fetch_wiki.py — RU-Википедия для talk-домена Fuga-1.0.
# Фикс против gen_wiki.py: redirects=1 + описательный User-Agent
# (gen_wiki блокировался — все ошибки глотались except: pass).
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "raw" / "wiki_ru"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://ru.wikipedia.org/w/api.php"
UA = "FugaCorpusBuilder/1.0 (byte-level LM research; local script)"

TOPICS = [
    "программирование", "искусственный интеллект", "физика", "математика",
    "философия", "история России", "литература", "музыка", "космос",
    "биология", "химия", "экономика", "психология", "спорт",
    "робототехника", "машинное обучение", "вселенная", "человек",
    "наука", "технология", "искусство", "культура", "образование",
    "компьютер", "интернет", "архитектура", "медицина", "география",
    "языкознание", "электричество",
]

def fetch(title: str) -> str | None:
    url = (
        f"{API}?action=query&prop=extracts&exintro=1&explaintext=1"
        f"&redirects=1&format=json&titles={urllib.parse.quote(title)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                ext = page.get("extract", "")
                if len(ext) > 200:
                    return f"=== {page.get('title', title)} ===\n{ext}\n"
            return None
        except Exception as e:
            if attempt == 2:
                print(f"  ✗ {title}: {e}")
                return None
            time.sleep(2)

ok = 0
total_bytes = 0
for t in TOPICS:
    name = t.replace(" ", "_") + ".txt"
    if (OUT / name).exists():
        ok += 1
        total_bytes += (OUT / name).stat().st_size
        continue
    text = fetch(t)
    if text:
        (OUT / name).write_text(text, encoding="utf-8")
        ok += 1
        total_bytes += len(text.encode("utf-8"))
    time.sleep(2.0)  # вежливый темп: Wikipedia режет 429 при частых запросах
print(f"wiki_ru: {ok}/{len(TOPICS)} статей, {total_bytes} байт")
