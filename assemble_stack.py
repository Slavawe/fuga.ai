#!/usr/bin/env python3
"""Assemble the unified Fuga 2.0 training stack.

Brings every technology — old, current, and future — into ONE training corpus
(training_stack.jsonl) in the shared CorpusDoc schema:

  code   : balanced subsample of all 21 repos across 5 languages (from
           omni_corpus_repos.jsonl)
  text   : dialogue + literature (text_corpus/)
  physics: aether/tesla/alubierre/mach-effect/qft/reactor/architecture docs
           (omni_corpus.jsonl)
  future : Fuga's own src/ (the system teaching itself)
"""
import json
import random
from pathlib import Path

random.seed(42)

OUT = "training_stack.jsonl"
REPO_DOCS = "omni_corpus_repos.jsonl"
PHYSICS = "omni_corpus.jsonl"
TEXT_DIR = Path("text_corpus")
SRC_DIR = Path("src")
CODE_TARGET = 3500
TEXT_CHUNK = 600  # chars per paragraph for text docs


def balanced_code_sample(docs, target):
    by_repo = {}
    for d in docs:
        t = d.get("title", "") or ""
        repo = t.split("/")[0] if "/" in t else t.split(" ")[0]
        by_repo.setdefault(repo, []).append(d)
    repos = list(by_repo)
    floor = min(20, target // (2 * len(repos)))
    quota = {r: floor for r in repos}
    remaining = max(target - floor * len(repos), 0)
    sizes = {r: len(by_repo[r]) for r in repos}
    total = sum(max(s - floor, 0) for s in sizes.values()) or 1
    for r in repos:
        extra = max(sizes[r] - floor, 0) * remaining // max(total, 1)
        quota[r] = min(sizes[r], quota[r] + extra)
    diff = sum(quota.values()) - target
    i = 0
    while diff > 0:
        r = sorted(repos, key=lambda x: (quota[x] < sizes[x], sizes[x]), reverse=True)[i % len(repos)]
        if quota[r] < sizes[r]:
            quota[r] += 1
            diff -= 1
        i += 1
    while diff < 0:
        r = min(repos, key=lambda x: sizes[x] - quota[x])
        quota[r] -= 1
        diff += 1
    picked = []
    for r in repos:
        picked += random.sample(by_repo[r], quota[r])
    random.shuffle(picked)
    info = {r: quota[r] for r in repos}
    return picked, info


def text_doc(path, tag):
    text = path.read_text(encoding="utf-8", errors="replace")
    paras = [text[i:i + TEXT_CHUNK] for i in range(0, len(text), TEXT_CHUNK)]
    paras = [p for p in paras if p.strip()]
    if not paras:
        return None
    return {
        "source_url": str(path),
        "title": str(path),
        "author": "text_" + tag,
        "language": "text",
        "chapters": [{"heading": f"Text: {tag}", "paragraphs": paras}],
    }


def code_path_doc(path, tag):
    code = path.read_text(encoding="utf-8", errors="replace")
    if len(code) < 100:
        return None
    paras = [code[0:2000], code[2000:4000], code[4000:6000]]
    paras = [p for p in paras if p]
    return {
        "source_url": str(path),
        "title": str(path),
        "author": tag,
        "language": "rust",
        "chapters": [{"heading": f"Code: {tag}", "paragraphs": paras}],
    }


def main():
    with open(REPO_DOCS) as f:
        repo_docs = [json.loads(l) for l in f if l.strip()]
    code_docs, info = balanced_code_sample(repo_docs, CODE_TARGET)
    print(f"code sample: {len(code_docs)} docs")
    for r in sorted(info, key=lambda x: -info[x]):
        print(f"  {r:14} {info[r]}")

    all_docs = code_docs

    # physics / architecture / reactor
    with open(PHYSICS) as f:
        for line in f:
            if line.strip():
                all_docs.append(json.loads(line))
    print(f"physics/arch docs: {len(all_docs) - len(code_docs)}")

    # text corpus
    n_text = 0
    for sub in ("dialogue", "literature"):
        d = TEXT_DIR / sub
        if d.is_dir():
            for p in sorted(d.rglob("*.txt")):
                doc = text_doc(p, sub)
                if doc:
                    all_docs.append(doc)
                    n_text += 1
    print(f"text docs: {n_text}")

    # fuga's own source (future: the system teaching itself)
    n_src = 0
    for p in sorted(SRC_DIR.rglob("*.rs")):
        if "target" in p.parts:
            continue
        if p.stat().st_size > 400_000:
            continue
        doc = code_path_doc(p, "fuga_src")
        if doc:
            all_docs.append(doc)
            n_src += 1
    print(f"fuga src docs: {n_src}")

    random.shuffle(all_docs)
    with open(OUT, "w", encoding="utf-8") as out:
        for d in all_docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n unified training stack: {len(all_docs)} docs -> {OUT}")


if __name__ == "__main__":
    main()