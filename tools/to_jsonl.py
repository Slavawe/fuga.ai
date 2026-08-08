#!/usr/bin/env python3
"""to_jsonl.py — перевод ВСЕХ датасетов (от текста до кода) в единый JSONL.

Гарантирует, что каждый датасет репозитория лежит построленный JSONL в общей
схеме Fuga 2.0 (source_url/title/author/language/chapters[].paragraphs),
той же, что assemble_stack.py использует для training_stack.jsonl.

Источники (файловые, ещё НЕ JSONL):
  - код:   corpus_sources/* (C/Go/Rust/C++ репозитории), corpus_ext/*,
            /tmp/fuga_corpus_rs/* (7218 .rs сниппетов)
  - текст: text_corpus_processed/*.txt, /tmp/fuga_corpus_text/*.txt

Уже-готовые *.jsonl стеки (omni_corpus*, corpus*.jsonl, training_stack.jsonl)
не трогаются — они уже JSONL в этой же схеме.

Использование:  python3 tools/to_jsonl.py [--out fuga_unified_train.jsonl]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 600          # chars per paragraph for text
CODE_CHUNK = 2000    # chars per paragraph for code
MIN_CODE = 100       # skip tiny files

# language tag by extension
EXT_LANG = {
    ".rs": "rust", ".go": "go", ".c": "c", ".h": "c", ".cc": "cpp",
    ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".py": "python",
    ".java": "java", ".js": "js", ".ts": "ts", ".rb": "ruby", ".sh": "sh",
    ".toml": "toml", ".md": "markdown",
}
CODE_EXTS = set(EXT_LANG)

CODE_DIRS = [
    ROOT / "corpus_sources",
    ROOT / "corpus_ext",
    Path("/tmp/fuga_corpus_rs"),
]
TEXT_DIRS = [
    ROOT / "text_corpus_processed",
    ROOT / "text_corpus",          # dialogue/ + literature/ (полные книги)
    Path("/tmp/fuga_corpus_text"),
]
SKIP_PARTS = ("target", "node_modules", ".git", "thirdparty", "build", "out")


def text_doc(text, tag, src):
    paras = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    paras = [p for p in paras if p.strip()]
    if not paras:
        return None
    return {
        "source_url": relpath(src),
        "title": relpath(src),
        "author": "text_" + tag,
        "language": "text",
        "chapters": [{"heading": "Text: " + tag, "paragraphs": paras}],
    }


def code_doc(code, lang, src):
    paras = [code[i:i + CODE_CHUNK] for i in range(0, len(code), CODE_CHUNK)]
    paras = [p for p in paras if p.strip()]
    if not paras:
        return None
    return {
        "source_url": relpath(src),
        "title": relpath(src),
        "author": "code_" + lang,
        "language": lang,
        "chapters": [{"heading": "Code: " + lang, "paragraphs": paras}],
    }


def relpath(p):
    """Absolute path -> portable relative (under ROOT, or /tmp/fuga_corpus_*)."""
    try:
        rel = os.path.relpath(str(p), str(ROOT))
        if not rel.startswith(".."):
            return rel
    except Exception:
        pass
    for sig in ("/tmp/fuga_corpus_rs", "/tmp/fuga_corpus_text"):
        if sig in str(p):
            return str(p).replace("/tmp/fuga_corpus_rs/", "corpus_rs/").replace("/tmp/fuga_corpus_text/", "corpus_text/")
    return str(p)


def strip_gutenberg(text):
    """Drop the Project Gutenberg license header/footer, keep the actual book body."""
    lines = text.splitlines()
    start, end = 0, len(lines)
    for i, l in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG" in l or "START OF THIS PROJECT GUTENBERG" in l:
            start = i + 1
        if "END OF THE PROJECT GUTENBERG" in l or "END OF THIS PROJECT GUTENBERG" in l:
            end = i
            break
    return "\n".join(lines[start:end])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "fuga_unified_train.jsonl"))
    args = ap.parse_args()

    docs = []
    seen = set()
    code_n = text_n = 0

    # --- CODE sources ---
    for base in CODE_DIRS:
        if not base.exists():
            continue
        for root, dirs_, files in os.walk(base):
            dirs_[:] = [d for d in dirs_ if d not in SKIP_PARTS and not d.startswith(".")]
            for fn in files:
                p = Path(root) / fn
                ext = p.suffix.lower()
                if ext not in CODE_EXTS:
                    continue
                real = str(p)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    code = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if len(code) < MIN_CODE:
                    continue
                doc = code_doc(code, EXT_LANG[ext], p)
                if doc:
                    docs.append(doc)
                    code_n += 1

    # --- text sources ---
    for base in TEXT_DIRS:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.txt")):
            real = str(p)
            if real in seen:
                continue
            seen.add(real)
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not text.strip():
                continue
            body = strip_gutenberg(text)
            tag = p.stem if not base.name.startswith("fuga_corpus_text") else "lit"
            doc = text_doc(body, tag, p)
            if doc:
                docs.append(doc)
                text_n += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"code docs: {code_n}")
    print(f"text docs: {text_n}")
    print(f"total: {len(docs)} docs -> {args.out}")


if __name__ == "__main__":
    main()