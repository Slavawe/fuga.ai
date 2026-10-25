#!/usr/bin/env python3
# build_corpus_v12.py — корпус Fuga-1.2: FineWeb-Edu (лучшее из Common Crawl)
# как главный EN-домен + code/ru домены v1.1.
#
# «Самое лучшее» из перечисленных пользователем корпусов (FineWeb/FineWeb-Edu,
# The Pile, Common Crawl, Dolma) на бюджете этой машины:
#   - FineWeb-Edu = Common Crawl, отфильтрованный по образовательной ценности
#     (LLM-классификатор, score 1–5); берём score ≥ 3 (educational) — это
#     выжимка качества Common Crawl без петабайтов мусора.
#   - The Pile / Dolma — шарды 2–3GB+ с худшей плотностью качества на байт;
#     на бюджете ~100MB FineWeb-Edu даёт больше сигнала (решение 15.10:
#     «отбор датасетов по бюджету машины»). Приоритет — качество/байт.
#
# Структура и маркеры — как v1.1 (build_corpus_v11.py): [CODE]/[EN]/[RU]
# ВНУТРИ JSON-значений. Вал — без маркеров.
#
# Перемешивание доменов: ПРОПОРЦИОНАЛЬНЫЙ планировщик (домен с минимальной
# выполненной долей бюджета берёт следующий док) — round-robin v1.1 вымывал
# бы маленькие домены при en_web ~80MB.
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"
OUT = HERE / "corpus"
OUT.mkdir(exist_ok=True)

MAX_DOC = 60_000
MIN_DOC = 300
VAL_SHARE = 0.10
BUDGET = {"code_local": 700_000, "code_rc": 1_200_000, "en": 600_000,
          "ru": 700_000, "en_web": 80_000_000}
EN_WEB_VAL = 400_000  # отдельный вал FineWeb-Edu (байты)

MARKER = {"code": b"[CODE] ", "en": b"[EN] ", "ru": b"[RU] ", "en_web": b"[EN] "}


def clip(b: bytes, limit: int = MAX_DOC) -> bytes:
    return b[:limit]


def is_englishish(b: bytes) -> bool:
    """Английский ли текст: кириллицы нет, доля ASCII-букв высока."""
    if b"\xd0" in b or b"\xd1" in b:  # кириллические lead-байты
        return False
    letters = sum(1 for c in b if 65 <= c <= 90 or 97 <= c <= 122)
    return letters / max(1, len(b)) > 0.55


# ---------- 1. FineWeb-Edu: главный EN-домен ----------
def collect_en_web():
    import pyarrow.parquet as pp

    shards = sorted(RAW.glob("fineweb_edu/*.parquet"))
    if not shards:
        print("  ! fineweb_edu/*.parquet нет — пропускаю en_web")
        return [], []
    rng = random.Random(42)
    train, val = [], []
    vtotal = 0
    total = 0
    n_scanned = 0
    for shard in shards:
        # Потоковое чтение батчами: таблица 1.4M строк не влезает в
        # to_pylist() целиком (MemoryError 24.10) — обрабатываем по 20K.
        pf = pp.ParquetFile(str(shard))
        cols = ["text"] + (["score"] if "score" in pf.schema_arrow.names else [])
        for batch in pf.iter_batches(batch_size=20_000, columns=cols):
            texts = batch.column("text").to_pylist()
            scores = batch.column("score").to_pylist() if "score" in batch.schema.names else [3] * len(texts)
            for j in range(len(texts)):
                if total >= BUDGET["en_web"] and vtotal >= EN_WEB_VAL:
                    break
                s = texts[j]
                n_scanned += 1
                if not isinstance(s, str):
                    continue
                if scores[j] is not None and scores[j] < 3:
                    continue
                b = s.encode("utf-8", errors="ignore")
                if not (MIN_DOC < len(b)):
                    continue
                b = clip(b)
                if not is_englishish(b[:4000]):
                    continue
                if vtotal < EN_WEB_VAL and rng.random() < 0.02:
                    val.append(b)
                    vtotal += len(b)
                    continue
                if total < BUDGET["en_web"]:
                    train.append(b)
                    total += len(b)
            del texts, scores, batch
        print(f"  {shard.name}: сканировано {n_scanned}, train {total} B, val {vtotal} B")
        if total >= BUDGET["en_web"] and vtotal >= EN_WEB_VAL:
            break
    print(f"  en_web: {len(train)} доков ({total} B) + val {len(val)} ({vtotal} B)")
    return train, val


# ---------- 2-4: домены v1.1 (код/ru/доки) ----------
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


def collect_code_rc():
    pq = RAW / "russian_code.parquet"
    if not pq.exists():
        print("  ! russian_code.parquet нет — пропускаю code_rc")
        return []
    import pyarrow.parquet as pp

    t = pp.read_table(str(pq))
    cols = set(t.column_names)
    prob_col = next((c for c in ("problem", "query", "ru_query", "instruction") if c in cols), None)
    sol_col = next((c for c in ("solution", "document", "response", "output", "content") if c in cols), None)
    if not (prob_col and sol_col):
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
    print(f"  russian-code: {len(out)} пар ({total} B)")
    return out


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


def split_domain(pieces, name, val_share=VAL_SHARE):
    total = sum(len(p) for p in pieces)
    boundary = int(total * (1.0 - val_share))
    train, val, acc = [], [], 0
    for p in pieces:
        if acc + len(p) // 2 >= boundary:
            val.append(p)
        else:
            train.append(p)
        acc += len(p)
    (OUT / f"val_{name}_v12.bin").write_bytes(b"\n".join(val))
    print(f"  [{name}] train={sum(len(p) for p in train)} B, val={sum(len(p) for p in val)} B")
    return train


def jsonl_line(kind: str, piece: bytes) -> bytes:
    text = piece.decode("utf-8", errors="replace")
    return (json.dumps({kind: text}, ensure_ascii=False) + "\n").encode("utf-8")


print("== Сбор корпуса Fuga-1.2 (FineWeb-Edu + code/ru) ==")
code_local = collect_code_local()
print(f"  code_local: {len(code_local)} файлов, {sum(map(len, code_local))} B")
code_rc = collect_code_rc()
en_docs = collect_en()
ru = collect_ru()
en_web, en_web_val = collect_en_web()
(OUT / "val_en_web_v12.bin").write_bytes(b"\n".join(en_web_val))

tr_code = split_domain(code_local + code_rc, "code")
tr_en = split_domain(en_docs, "en")
tr_ru = split_domain(ru, "ru")

# ПРОПОРЦИОНАЛЬНОЕ перемешивание: на каждом шаге берём док из домена с
# минимальной выполненной долей бюджета (детерминированно, seed 11).
# ОВЕРСЭМПЛИНГ малых доменов (защита bilingual от вымывания при EN-web 86%):
# первый вариант (ru ×4, code ×2) дал на 1M-чекпоинте code cos 0.116 / ru 0.112
# против 0.486/0.407 у v1.1@1.5M — доля малых доменов за 5M шагов дала бы
# ru ~55K пар против 375K у v1.1. Усилено: ru ×40 (~10% рациона, RU-raw всего
# 240KB — повторение = единственный путь говорить по-русски, так же учились
# v1.0/v1.1: agent_loop ×22), code ×6 (~11%), en ×4. Целевые доли рациона
# на 5M: en_web ~4.3M пар, code ~560K, ru ~510K.
pools = {"code": tr_code * 6, "en": tr_en * 4, "ru": tr_ru * 40, "en_web": en_web}
budgets = {"code": (BUDGET["code_local"] + BUDGET["code_rc"]) * 6, "en": BUDGET["en"] * 4,
           "ru": BUDGET["ru"] * 40, "en_web": BUDGET["en_web"]}
for v in pools.values():
    random.Random(11).shuffle(v)
idx = {d: 0 for d in pools}
consumed = {d: 0.0 for d in pools}
lines, marked = [], 0
while True:
    cands = [(consumed[d] / budgets[d], d) for d in pools if idx[d] < len(pools[d])]
    if not cands:
        break
    _, d = min(cands)
    piece = pools[d][idx[d]]
    idx[d] += 1
    consumed[d] += len(piece)
    kind = "code" if d.startswith("code") else "doc"
    if d == "code":
        if not piece.startswith(b'{"doc"'):
            piece = MARKER["code"] + piece
    else:
        piece = MARKER[d] + piece
    lines.append(jsonl_line(kind, piece))
    marked += 1

corpus = OUT / "fuga12_train.jsonl"
with corpus.open("wb") as f:
    f.writelines(lines)
total_b = sum(len(l) for l in lines)
print(f"== corpus: {corpus.name} — {len(lines)} доков, {total_b} B (~{total_b/1e6:.1f} MB) ==")
for d in pools:
    print(f"   {d}: {idx[d]} доков, {int(consumed[d])} B ({100*consumed[d]/total_b:.1f}%)")

# val_all для тренера: code + en + ru + en_web (все без маркеров)
va = b"".join((OUT / f"val_{n}_v12.bin").read_bytes()
              for n in ("code", "en", "ru", "en_web"))
(OUT / "val_all_v12.bin").write_bytes(va)
print(f"== val: val_code/en/ru/en_web_v12.bin, val_all_v12.bin {len(va)} B ==")
