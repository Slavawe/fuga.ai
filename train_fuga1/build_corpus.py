#!/usr/bin/env python3
# build_corpus.py — сборка корпуса Fuga-1.0: «говорит (EN/RU) + пишет код».
#
# Домены и бюджеты (байты контента):
#   code   ~50%: локальный Rust/Python (src/, *.py) + решения russian-code (HF parquet)
#   en     ~25%: *.md доки репозитория (README, docs/, references/)
#   ru     ~25%: RU-википедия (raw/wiki_ru) + AGENTS.md + диалоги/проза gen_synthetic
#
# Формат: JSONL {"doc": текст} | {"code": код} — читается extract_bytes тренера.
# Instruction-пары russian-code сериализуются: doc="Задача: <problem>", code=<solution>
# (ответ на вопрос карты датасетов: байтовые потоки, не token-level; FastKAN не instruction-tuner).
#
# Val: последние 10% КАЖДОГО домена → val_code.bin / val_en.bin / val_ru.bin
# (сырые байты; контроль «размытия» RU/EN по доменам на чекпоинтах).
# Train-документы перемешиваются round-robin по доменам (детерминированно).
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"
OUT = HERE / "corpus"
OUT.mkdir(exist_ok=True)

MAX_DOC = 60_000   # лимит строки-монстра тренера 64KB — с запасом
VAL_SHARE = 0.10
BUDGET = {"code_local": 700_000, "code_rc": 1_200_000, "en": 600_000, "ru": 700_000}

def clip(b: bytes, limit: int = MAX_DOC) -> bytes:
    return b[:limit]

# ---------- 1. CODE: локальный Rust/Python ----------
def collect_code_local():
    files = []
    for pat in ("src/ai/*.rs", "src/bin/*.rs", "src/bin/*.py", "*.py", "py/*.py", "tools/*.py"):
        files += [p for p in ROOT.glob(pat) if p.is_file()]
    files = sorted(set(files), key=lambda p: str(p))
    out, total = [], 0
    for p in files:
        try:
            b = p.read_bytes()
        except Exception:
            continue
        if not (200 < len(b) <= MAX_DOC):
            continue
        if total + len(b) > BUDGET["code_local"]:
            break
        out.append(b"\n// ==== FILE: " + str(p.relative_to(ROOT)).encode() + b" ====\n" + b)
        total += len(b)
    return out

# ---------- 2. CODE: russian-code (instruction → doc/code пара) ----------
def collect_code_rc():
    pq = RAW / "russian_code.parquet"
    if not pq.exists():
        print("  ! russian_code.parquet нет — пропускаю домен code_rc")
        return []
    import pyarrow.parquet as pp
    t = pp.read_table(str(pq))
    cols = set(t.column_names)
    print(f"  parquet: {t.num_rows} строк, колонки {t.column_names}")
    prob_col = next((c for c in ("problem", "query", "ru_query", "instruction") if c in cols), None)
    sol_col = next((c for c in ("solution", "document", "response", "output", "content") if c in cols), None)
    if not (prob_col and sol_col):
        print(f"  ! не нашёл пары колонок в {cols} — пропускаю")
        return []
    problems = t.column(prob_col).to_pylist()
    solutions = t.column(sol_col).to_pylist()
    rng = random.Random(42)
    idx = list(range(len(problems)))
    rng.shuffle(idx)  # детерминированная выборка по всему файлу
    out, total, i = [], 0, 0
    while total < BUDGET["code_rc"] and i < len(idx):
        j = idx[i]
        i += 1
        prob, sol = problems[j], solutions[j]
        if not (isinstance(prob, str) and isinstance(sol, str)):
            continue
        if not (30 < len(sol) < 6000):
            continue
        doc = ("Задача: " + prob.strip()[:2000]).encode("utf-8")
        code = sol.encode("utf-8")
        piece = b'{"doc": ' + json.dumps(doc.decode("utf-8"), ensure_ascii=False).encode("utf-8") \
            + b', "code": ' + json.dumps(code.decode("utf-8"), ensure_ascii=False).encode("utf-8") + b"}\n"
        if len(piece) > MAX_DOC:
            continue
        out.append(piece)
        total += len(code) + len(doc)
    print(f"  russian-code: взято {len(out)} пар ({total} байт)")
    return out

# ---------- 3. EN TEXT: доки репозитория ----------
def collect_en():
    files = sorted(
        list(ROOT.glob("*.md")) + list(ROOT.glob("docs/*.md")) + list(ROOT.glob("references/*.md")),
        key=lambda p: str(p),
    )
    out, total = [], 0
    for p in files:
        try:
            b = p.read_bytes()
        except Exception:
            continue
        for k in range(0, len(b), 30_000):
            chunk = clip(b[k:k + 30_000])
            if len(chunk) > 500 and total < BUDGET["en"]:
                out.append(chunk)
                total += len(chunk)
    return out

# ---------- 4. RU TEXT: wiki + AGENTS.md + диалоги/проза ----------
def collect_ru():
    out, total = [], 0
    for pat in ("raw/wiki_ru/*.txt", "text_corpus/dialogue/*.txt",
                "text_corpus/literature/*.txt"):
        for p in sorted(HERE.glob(pat), key=lambda x: str(x)):
            try:
                b = p.read_bytes()
            except Exception:
                continue
            if total >= BUDGET["ru"]:
                break
            out.append(clip(b))
            total += len(b)
    ag = ROOT / "AGENTS.md"
    if ag.exists() and total < BUDGET["ru"]:
        b = ag.read_bytes()
        for k in range(0, len(b), 30_000):
            chunk = clip(b[k:k + 30_000])
            if len(chunk) > 500 and total < BUDGET["ru"]:
                out.append(chunk)
                total += len(chunk)
    return out

# ---------- сборка ----------
def split_domain(pieces, name):
    """Последние VAL_SHARE байт домена → val_{name}.bin, остальное → train.
    Кусок идёт в val, если его СЕРЕДИНА пересекает границу (иначе последний
    крупный кусок целиком уходил в train и val получался пустым)."""
    total = sum(len(p) for p in pieces)
    boundary = int(total * (1.0 - VAL_SHARE))
    train, val, acc = [], [], 0
    for p in pieces:
        if acc + len(p) // 2 >= boundary:
            val.append(p)
        else:
            train.append(p)
        acc += len(p)
    (OUT / f"val_{name}.bin").write_bytes(b"\n".join(val))
    print(f"  [{name}] train={sum(len(p) for p in train)} B, val={sum(len(p) for p in val)} B")
    return train

def jsonl_line(kind: str, piece: bytes) -> bytes:
    text = piece.decode("utf-8", errors="replace")
    return (json.dumps({kind: text}, ensure_ascii=False) + "\n").encode("utf-8")

print("== Сбор корпуса Fuga-1.0 ==")
code_local = collect_code_local()
print(f"  code_local: {len(code_local)} файлов, {sum(map(len, code_local))} B")
code_rc = collect_code_rc()
en = collect_en()
ru = collect_ru()

tr_code = split_domain(code_local + code_rc, "code")
tr_en = split_domain(en, "en")
tr_ru = split_domain(ru, "ru")

# Round-robin доменов: code, en, ru, code, en, ru... (mixing против доменных блоков)
pools = {"code": tr_code, "en": tr_en, "ru": tr_ru}
for v in pools.values():
    random.Random(7).shuffle(v)
order = ["code", "en", "ru"]
lines, k = [], 0
while any(pools.values()):
    d = order[k % 3]
    k += 1
    if pools[d]:
        piece = pools[d].pop(0)
        kind = "code" if d.startswith("code") else "doc"
        lines.append(jsonl_line(kind, piece))

corpus = OUT / "fuga1_train.jsonl"
with corpus.open("wb") as f:
    f.writelines(lines)
total_b = sum(len(l) for l in lines)
print(f"== corpus: {corpus.name} — {len(lines)} доков, {total_b} B (~{total_b/1e6:.2f} MB) ==")
print("== val: val_code.bin / val_en.bin / val_ru.bin ==")
