from __future__ import annotations

import csv
import os
import re
import urllib.request

RUCOLA_DIR = "datasets/rucola"
RUCOLA_BASE = "https://raw.githubusercontent.com/RussianNLP/RuCoLA/main/data"
RUCOLA_FILES = {
    "in_domain_train.csv": f"{RUCOLA_BASE}/in_domain_train.csv",
    "in_domain_dev.csv": f"{RUCOLA_BASE}/in_domain_dev.csv",
    "out_of_domain_dev.csv": f"{RUCOLA_BASE}/out_of_domain_dev.csv",
}


def ensure_rucola(cache_dir: str = RUCOLA_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for name, url in RUCOLA_FILES.items():
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)


def load_rucola(split: str = "in_domain_train", cache_dir: str = RUCOLA_DIR):
    """Returns [(sentence, acceptable 0/1), ...]."""
    ensure_rucola(cache_dir)
    rows = []
    with open(os.path.join(cache_dir, f"{split}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["sentence"], int(r["acceptable"])))
    return rows


def build_filter(fuga_core, tatoeba_texts: list[str], max_lexicon: int = 100000,
                 max_transitions: int = 400000, use_tatoeba_bigrams: bool = True):
    """Лексикон: Tatoeba RU (прокси Wiktionary) + слова RuCoLA ok.
    Переходы: биграммы из acceptable=1 RuCoLA + (опц.) реальные фразы Tatoeba."""
    from antitf.item_memory import SimpleWordVocab

    lex = SimpleWordVocab.build(tatoeba_texts, max_size=max_lexicon)
    words = list(lex.stoi.keys())

    flt = fuga_core.RustLinguisticFilter()
    transitions = []
    ok_rows = load_rucola("in_domain_train")
    for sent, acc in ok_rows:
        if acc != 1:
            continue
        toks = tokenize(sent)
        for a, b in zip(toks, toks[1:]):
            if len(transitions) < max_transitions:
                transitions.append((a, b))
        words.extend(toks)

    if use_tatoeba_bigrams:
        # Реальные фразы корпуса: тот же канал допустимых связей,
        # но на порядок богаче RuCoLA (7.7K vs 200K предложений).
        for sent in tatoeba_texts:
            toks = tokenize(sent)
            for a, b in zip(toks, toks[1:]):
                if len(transitions) < max_transitions:
                    transitions.append((a, b))
            words.extend(toks)

    # дедуп лексикона на стороне Rust через HashSet
    flt.load_wiktionary_vocab(words)
    flt.load_rucola_transitions(transitions)
    return flt


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import fuga_core
    from antitf.data_i18n import load_tatoeba_pairs

    pairs = load_tatoeba_pairs(max_pairs=20000)
    ru_texts = [p[0] for p in pairs]
    flt = build_filter(fuga_core, ru_texts)
    print(f"lexicon={flt.vocab_size()}  transitions={flt.transitions_size()}")
