#!/usr/bin/env python3
# build_corpus_v11.py — корпус Fuga-1.1: МАРКЕРЫ ДОМЕНОВ как control-input.
#
# Единственная переменная относительно v1.0 (build_corpus.py): каждый train-док
# получает префикс домена  [CODE] / [EN] / [RU]  ВНУТРИ JSON-значения
# (формат JSONL {"doc":...}/{"code":...} не нарушается). На декоде маркер
# задаётся сидом «[RU] текст» / «[CODE] def ...» — байтовый control-input,
# сдвигающий распределение домена (рычаг «фазовый curriculum + маркеры»,
# предрегистрированный критерий v1.1: cyr≤30% при код-сиде И code%≥5%).
#
# Вал-файлы — БЕЗ маркеров (чистое измерение языкового моделирования),
# пересобираются детерминированно из тех же raw → val_all_v11.bin для тренера.
#
# Скалярный бюджет доменов тот же (code~50% / en~25% / ru~25%), round-robin
# чередование то же. Отладочная печать: доля строк с маркерами.
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"
OUT = HERE / "corpus"
OUT.mkdir(exist_ok=True)

MAX_DOC = 60_000
VAL_SHARE = 0.10
BUDGET = {"code_local": 700_000, "code_rc": 1_200_000, "en": 600_000, "ru": 700_000}

MARKER = {"code": b"[CODE] ", "en": b"[EN] ", "ru": b"[RU] "}


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
    rng.shuffle(idx)
    out, total, i = [], 0, 0
    while total < BUDGET["code_rc"] and i < len(idx):
        j = idx[i]
        i += 1
        prob, sol = problems[j], solutions[j]
        if not (isinstance(prob, str) and isinstance(sol, str)):
            continue
        if not (30 < len(sol) < 6000):
            continue
        # маркер [CODE] ВНУТРИ doc-поля (формат JSONL не нарушается)
        doc = ("[CODE] Задача: " + prob.strip()[:2000]).encode("utf-8")
        code = sol.encode("utf-8")
        piece = (
            b'{"doc": '
            + json.dumps(doc.decode("utf-8"), ensure_ascii=False).encode("utf-8")
            + b', "code": '
            + json.dumps(code.decode("utf-8"), ensure_ascii=False).encode("utf-8")
            + b"}\n"
        )
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
            chunk = clip(b[k : k + 30_000])
            if len(chunk) > 500 and total < BUDGET["en"]:
                out.append(chunk)
                total += len(chunk)
    return out


# ---------- 4. RU TEXT: wiki + AGENTS.md + диалоги/проза ----------
def collect_ru():
    out, total = [], 0
    for pat in ("raw/wiki_ru/*.txt", "text_corpus/dialogue/*.txt", "text_corpus/literature/*.txt"):
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
            chunk = clip(b[k : k + 30_000])
            if len(chunk) > 500 and total < BUDGET["ru"]:
                out.append(chunk)
                total += len(chunk)
    return out


# ---------- сборка ----------
def split_domain(pieces, name):
    """Последние VAL_SHARE байт домена → val_{name}_v11.bin, остальное → train.
    Логика границы — как в v1.0 (кусок идёт в val, если его середина
    пересекает границу). Вал-файлы БЕЗ маркеров."""
    total = sum(len(p) for p in pieces)
    boundary = int(total * (1.0 - VAL_SHARE))
    train, val, acc = [], [], 0
    for p in pieces:
        if acc + len(p) // 2 >= boundary:
            val.append(p)
        else:
            train.append(p)
        acc += len(p)
    (OUT / f"val_{name}_v11.bin").write_bytes(b"\n".join(val))
    print(f"  [{name}] train={sum(len(p) for p in train)} B, val={sum(len(p) for p in val)} B")
    return train


def jsonl_line(kind: str, piece: bytes) -> bytes:
    text = piece.decode("utf-8", errors="replace")
    return (json.dumps({kind: text}, ensure_ascii=False) + "\n").encode("utf-8")


print("== Сбор корпуса Fuga-1.1 (маркеры доменов) ==")
code_local = collect_code_local()
print(f"  code_local: {len(code_local)} файлов, {sum(map(len, code_local))} B")
code_rc = collect_code_rc()
en = collect_en()
ru = collect_ru()

tr_code = split_domain(code_local + code_rc, "code")
tr_en = split_domain(en, "en")
tr_ru = split_domain(ru, "ru")

# Round-robin доменов с маркером на каждом доке (code, en, ru, ...)
pools = {"code": tr_code, "en": tr_en, "ru": tr_ru}
for v in pools.values():
    random.Random(7).shuffle(v)
order = ["code", "en", "ru"]
lines, k, marked = [], 0, 0
while any(pools.values()):
    d = order[k % 3]
    k += 1
    if pools[d]:
        piece = pools[d].pop(0)
        kind = "code" if d.startswith("code") else "doc"
        if d == "code":
            # rc-пары уже JSONL с маркером внутри doc; файловые куски —
            # plain bytes → маркер префиксом содержимого
            if not piece.startswith(b'{"doc"'):
                piece = MARKER["code"] + piece
        else:
            piece = MARKER[d] + piece
        lines.append(jsonl_line(kind, piece))
        marked += 1

corpus = OUT / "fuga11_train.jsonl"
with corpus.open("wb") as f:
    f.writelines(lines)
total_b = sum(len(l) for l in lines)
n_code = sum(1 for l in lines if b"[CODE] " in l[:60])
n_en = sum(1 for l in lines if b"[EN] " in l[:60])
n_ru = sum(1 for l in lines if b"[RU] " in l[:60])
print(f"== corpus: {corpus.name} — {len(lines)} доков (маркеров {marked}), {total_b} B (~{total_b/1e6:.2f} MB) ==")
print(f"   [CODE]={n_code} [EN]={n_en} [RU]={n_ru} (контроль покрытия)")
print("== val (без маркеров): val_code_v11 / val_en_v11 / val_ru_v11 ==")

# val_all для встроенного мониторинга тренера
va = b"".join(
    (OUT / f"val_{n}_v11.bin").read_bytes() for n in ("code", "en", "ru")
)
(OUT / "val_all_v11.bin").write_bytes(va)
print(f"== val_all_v11.bin: {len(va)} B ==")
