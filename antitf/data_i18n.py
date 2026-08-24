from __future__ import annotations

import bz2
import os
import urllib.request

TATOEBAA_DIR = "datasets/tatoeba"
LINKS_URL = "https://downloads.tatoeba.org/exports/per_language/rus/rus-eng_links.tsv.bz2"
LINKS_FILE = "rus-eng_links.tsv.bz2"
SENTS_URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
SENTS_TAR = "sentences.tar.bz2"
SENTS_CSV = "sentences.csv"
SENTS_CC0_TAR = "sentences_CC0.tar.bz2"
SENTS_CC0_CSV = "sentences_CC0.csv"


def ensure_tatoeba(cache_dir: str = TATOEBAA_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    links_path = os.path.join(cache_dir, LINKS_FILE)
    sents_path = os.path.join(cache_dir, SENTS_CSV)
    cc0_path = os.path.join(cache_dir, SENTS_CC0_CSV)
    if not os.path.exists(links_path):
        print(f"downloading {LINKS_URL} ...")
        urllib.request.urlretrieve(LINKS_URL, links_path)
    if not os.path.exists(sents_path):
        tar_path = os.path.join(cache_dir, SENTS_TAR)
        if not os.path.exists(tar_path):
            print(f"downloading {SENTS_URL} ...")
            urllib.request.urlretrieve(SENTS_URL, tar_path)
        import tarfile
        with tarfile.open(tar_path) as tf:
            tf.extractall(cache_dir)
    elif not os.path.exists(cc0_path) and os.path.exists(os.path.join(cache_dir, SENTS_CC0_TAR)):
        import tarfile
        with tarfile.open(os.path.join(cache_dir, SENTS_CC0_TAR)) as tf:
            tf.extractall(cache_dir)


def load_tatoeba_pairs(max_pairs: int = 20000,
                       cache_dir: str = TATOEBAA_DIR) -> list[tuple[str, str]]:
    """Returns [(ru_sentence, en_sentence), ...] joined from links + CC0 texts."""
    ensure_tatoeba(cache_dir)
    sents: dict[int, tuple[str, str]] = {}
    csv_path = os.path.join(cache_dir, SENTS_CSV)
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] in ("rus", "eng"):
                try:
                    sents[int(parts[0])] = (parts[1], parts[2])
                except ValueError:
                    continue

    links_path = os.path.join(cache_dir, LINKS_FILE)
    pairs: list[tuple[str, str]] = []
    with bz2.open(links_path, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            try:
                a, b = int(cols[0]), int(cols[1])
            except ValueError:
                continue
            sa, sb = sents.get(a), sents.get(b)
            if not sa or not sb or sa[0] == sb[0]:
                continue
            ru, en = (sa[1], sb[1]) if sa[0] == "rus" else (sb[1], sa[1])
            if 0 < len(ru.encode("utf-8")) <= 256 and 0 < len(en.encode("utf-8")) <= 256:
                pairs.append((ru, en))
            if len(pairs) >= max_pairs:
                break
    return pairs


def make_byte_windows(text: str, window: int = 128) -> bytes:
    raw = text.encode("utf-8")[:window]
    return raw.ljust(window, b"\x00")
