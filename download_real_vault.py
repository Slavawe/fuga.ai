from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import os
import re
import urllib.request
import zipfile

VAULT = "dataset_vault"
TATOEBA_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-ru.txt.zip"
CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
OASST_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_all.messages.jsonl.gz"


def _get(url: str, timeout: int = 600):
    return urllib.request.urlopen(urllib.request.Request(url), timeout=timeout)


def download_tatoeba_ru_en(limit: int = 150000) -> str:
    out = os.path.join(VAULT, "03_core_dictionary", "tatoeba_real_ru_en.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[*] OPUS Tatoeba en-ru moses ...")
    raw = _get(TATOEBA_URL).read()
    n = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        en_name = [x for x in names if x.endswith(".en")][0]
        ru_name = [x for x in names if x.endswith(".ru")][0]
        with zf.open(en_name) as fen, zf.open(ru_name) as fru, open(out, "w", encoding="utf-8") as f:
            for en_line, ru_line in zip(io.TextIOWrapper(fen, encoding="utf-8"),
                                        io.TextIOWrapper(fru, encoding="utf-8")):
                en, ru = en_line.strip(), ru_line.strip()
                if len(en) > 3 and len(ru) > 3 and len(en) < 300 and len(ru) < 300:
                    f.write(json.dumps({"en": en, "ru": ru}, ensure_ascii=False) + "\n")
                    n += 1
                    if n >= limit:
                        break
    print(f"[+] tatoeba_real_ru_en.jsonl: {n} пар")
    return out


def _conceptnet_label(uri: str) -> str:
    # /c/en/dog/n -> "dog"; берём каноничный термин без POS-суффикса
    parts = uri.split("/")
    term = parts[3].replace("_", " ") if len(parts) > 3 else ""
    return term


def download_conceptnet_triplets(limit: int = 120000) -> str:
    out = os.path.join(VAULT, "02_world_concepts", "conceptnet_sro_real.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("[*] ConceptNet assertions (stream, ~500MB gz) ...")
    n = 0
    resp = _get(CONCEPTNET_URL, timeout=1800)
    gz = gzip.GzipFile(fileobj=resp)
    with open(out, "w", encoding="utf-8") as f:
        for raw_line in io.TextIOWrapper(gz, encoding="utf-8"):
            cols = raw_line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            _, rel_uri, start_uri, end_uri, meta = cols
            langs = re.findall(r"/c/([a-z]{2})/", start_uri + " " + end_uri)
            ls = start_uri.split("/")[2] if start_uri.startswith("/c/") else ""
            le = end_uri.split("/")[2] if end_uri.startswith("/c/") else ""
            if ls != le or ls not in ("en", "ru"):
                continue
            try:
                mj = json.loads(meta)
            except json.JSONDecodeError:
                continue
            s_txt = mj.get("surfaceStart") or _conceptnet_label(start_uri)
            o_txt = mj.get("surfaceEnd") or _conceptnet_label(end_uri)
            rel = rel_uri.split("/r/")[-1].lower()
            if not s_txt or not o_txt or s_txt.lower() == o_txt.lower():
                continue
            f.write(json.dumps({
                "lang": ls, "subject": s_txt, "relation": rel, "object": o_txt,
                "weight": float(mj.get("weight", 1.0)),
            }, ensure_ascii=False) + "\n")
            n += 1
            if n >= limit:
                break
    resp.close()
    print(f"[+] conceptnet_sro_real.jsonl: {n} троек")
    return out


def download_oasst_dialogues(limit: int = 40000) -> str:
    out = os.path.join(VAULT, "01_everyday_dialogues", "open_assistant_real.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("[*] OpenAssistant messages (stream) ...")
    by_id: dict[str, dict] = {}
    pairs = []
    resp = _get(OASST_URL, timeout=1800)
    gz = gzip.GzipFile(fileobj=resp)
    for raw_line in io.TextIOWrapper(gz, encoding="utf-8"):
        try:
            m = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        mid = m.get("message_id")
        if mid:
            by_id[mid] = m
        parent_id = m.get("parent_id")
        lang = m.get("lang", "")
        if not parent_id or lang not in ("en", "ru"):
            continue
        parent = by_id.get(parent_id)
        if not parent or not parent.get("text") or not m.get("text"):
            continue
        pairs.append({"lang": lang,
                      "context_role": parent.get("role", ""),
                      "context": parent["text"],
                      "response": m["text"]})
        del by_id[parent_id]
        if len(pairs) >= limit:
            break
    resp.close()
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[+] open_assistant_real.jsonl: {len(pairs)} пар контекст-ответ")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tatoeba", type=int, default=150000)
    ap.add_argument("--conceptnet", type=int, default=120000)
    ap.add_argument("--oasst", type=int, default=40000)
    args = ap.parse_args()
    download_tatoeba_ru_en(args.tatoeba)
    download_conceptnet_triplets(args.conceptnet)
    download_oasst_dialogues(args.oasst)
    print("\n[SUCCESS] dataset_vault готов")
