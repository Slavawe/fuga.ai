#!/usr/bin/env python3
"""Проверка способностей модели + текстовое объяснение + анализ данных.

Что умеет модель (проверяется):
  1. Генерация кода (BLT-декодер, Beam Search)
  2. Генерация языка (RU/EN/ZH — трёхъязычный чекпоинт)
  3. Решение гипотез (VSA-учитель, геометрия, эллиптические кривые)
  4. Кросс-языковая инвариантность (cos переводов → 1.0)
  5. Облака точек (Point-JEPA — разделимость форм)
  6. Анализ входных данных (входной текст → факты → вывод)

Режимы:
  --check     — полная проверка способностей (таблица)
  --explain   — модель ОБЪЯСНЯЕТ текстом (не голосом)
  --analyze   — анализ входных данных (аргумент или stdin)
"""

import sys
import os
import subprocess
import json

sys.path.insert(0, "/home/slava/Anti-Tronsformers")

BLT = "/home/slava/Anti-Tronsformers/target/release/blt_decode"
CKPT_TRI = "/tmp/blt_trilingual_tatoeba_500k.fuga"
CORPUS_TRI = "/tmp/trilingual/corpus_merged.jsonl"
CKPT_CODE = "/tmp/blt_multicode_500k.fuga"
CORPUS_CODE = "/tmp/fuga_multicode.jsonl"


# ═══════════════════════════════════════════════════════════════
# 1. ГЕНЕРАЦИЯ (BLT-декодер)
# ═══════════════════════════════════════════════════════════════
def blt_decode(ckpt: str, corpus: str, seed: str, beam: int = 20) -> tuple[int, str]:
    if not os.path.exists(ckpt):
        return 0, "[нет чекпоинта]"
    try:
        r = subprocess.run(
            [BLT, ckpt, corpus, "0.85", str(beam), seed],
            capture_output=True, text=True, timeout=90,
            cwd="/home/slava/Anti-Tronsformers",
        )
        for line in r.stdout.split("\n"):
            if "BLT:" in line:
                n = int(line.split("BLT:")[1].split("B")[0].strip())
                return n, line.split("→")[-1].strip()
        return 0, "[нет вывода]"
    except Exception:
        return 0, "[ошибка]"


def check_generation() -> dict:
    """Проверка генерации кода и языка."""
    results = {}

    # Код
    code_len, code_out = blt_decode(CKPT_CODE, CORPUS_CODE, "fn main() {")
    results["код"] = {
        "ok": code_len > 50,
        "детали": f"{code_len}B: {code_out[:60]}...",
    }

    # Язык (EN)
    en_len, en_out = blt_decode(CKPT_TRI, CORPUS_TRI, "The data pointer is null.")
    results["язык_EN"] = {
        "ok": en_len > 50,
        "детали": f"{en_len}B: {en_out[:60]}...",
    }

    # Язык (RU)
    ru_len, ru_out = blt_decode(CKPT_TRI, CORPUS_TRI, "Указатель на данные равен нулю.")
    results["язык_RU"] = {
        "ok": ru_len > 50,
        "детали": f"{ru_len}B: {ru_out[:60]}...",
    }

    # Язык (ZH)
    zh_len, zh_out = blt_decode(CKPT_TRI, CORPUS_TRI, "数据指针为空。")
    results["язык_ZH"] = {
        "ok": zh_len > 50,
        "детали": f"{zh_len}B: {zh_out[:60]}...",
    }

    return results


# ═══════════════════════════════════════════════════════════════
# 2. ГИПОТЕЗЫ (учитель-ученик)
# ═══════════════════════════════════════════════════════════════
def check_hypotheses() -> dict:
    """Проверка решения гипотез."""
    results = {}

    # Геометрический медиан
    from astral.experiments.beam_teacher import LandscapeAnalyzer
    la = LandscapeAnalyzer()
    res = la.global_min([(0, 4), (4, 4), (4, 0), (1, 1)], grid=60)
    ok = abs(res["refined"][0] - 2.0) < 0.1 and abs(res["refined"][1] - 2.0) < 0.1
    results["геометрия"] = {
        "ok": ok,
        "детали": f"P({res['refined'][0]:.3f},{res['refined'][1]:.3f})",
    }

    # Эллиптическая кривая
    from astral.experiments.beam_teacher import AlgebraicFalsifier
    af = AlgebraicFalsifier()
    sols = [(2, 1), (2, -1), (32, 181), (32, -181)]
    all_ok = all(af.verify_diophantine(x, y) for x, y in sols)
    results["эллиптическая_кривая"] = {
        "ok": all_ok,
        "детали": f"4 решения: {sols[:2]}...",
    }

    # Кросс-языковая верификация (zero-shot)
    from astral.experiments.cross_lingual_vsa import CrossLingualVSA, MultilingualTeacher
    clvsa = CrossLingualVSA(dim=512)
    pairs = [
        ("данные", "data", "数据"),
        ("матрица", "matrix", "矩阵"),
        ("гипотеза", "hypothesis", "假设"),
        ("граф", "graph", "图"),
    ]
    clvsa.train_pairs(pairs, epochs=30, lr=0.1)
    teacher = MultilingualTeacher(clvsa, threshold=0.90)
    teacher.learn_axiom("данные", "данные", "RU")
    ok, concept, c = teacher.verify("data", "EN")
    results["zero-shot_учитель"] = {
        "ok": ok and concept == "данные",
        "детали": f"«data»(EN) → {concept}, cos={c:.3f}",
    }

    # Инвариантность
    c_inv = clvsa.cos("данные", "RU", "data", "EN")
    results["инвариантность"] = {
        "ok": c_inv >= 0.95,
        "детали": f"cos(RU,EN) = {c_inv:.4f}",
    }

    return results


# ═══════════════════════════════════════════════════════════════
# 3. ОБЛАКА ТОЧЕК (Point-JEPA)
# ═══════════════════════════════════════════════════════════════
def check_point_jepa() -> dict:
    """Проверка Point-JEPA: разделимость форм и предсказание."""
    from astral.experiments.point_jepa import PointCloudEncoder, PointJEPA, make_shape
    import numpy as np

    rng = np.random.default_rng(42)
    shapes = ["sphere", "cube", "plane", "line", "cluster"]
    clouds = [make_shape(s, n=64, rng=rng) for s in shapes]

    # Разделимость cluster от остальных
    enc = PointCloudEncoder(dim=512)
    lats = [enc.encode(c) for c in clouds]
    cluster_cos = [enc.cos(lats[4], lats[i]) for i in range(4)]
    sep_ok = max(cluster_cos) < 0.5

    # Предсказание перехода
    seq = [clouds[i % len(shapes)] for i in range(20)]
    pj = PointJEPA(dim=512, lr=0.05)
    loss = pj.train(seq, epochs=30)
    pred_cos = [pj.cos_pred(seq[i], seq[i + 1]) for i in range(min(4, len(seq) - 1))]

    return {
        "разделимость_форм": {
            "ok": sep_ok,
            "детали": f"cluster vs [sph,cube,pln,ln]: {[f'{c:.2f}' for c in cluster_cos]}",
        },
        "предсказание_переходов": {
            "ok": np.mean(pred_cos) > 0.6,
            "детали": f"cos(предск,факт): {[f'{c:.2f}' for c in pred_cos]}, loss={loss:.3f}",
        },
    }


# ═══════════════════════════════════════════════════════════════
# 4. ТЕКСТОВОЕ ОБЪЯСНЕНИЕ (не голосом)
# ═══════════════════════════════════════════════════════════════
def explain_text(results: dict) -> None:
    """Модель объясняет свои способности ТЕКСТОМ."""
    print("\n" + "═" * 64)
    print("  ОБЪЯСНЕНИЕ МОДЕЛИ (текст, без голоса)")
    print("═" * 64)
    print("""
  Я — Fuga Cognitive Engine, безградиентный стек на VSA/JEPA/KAN.
  Вот что я умею и как это работает:

  [Генерация кода и языка]
  Я читаю корпус байт-в-байт, дроблю его на BLT-патчи по энтропии
  (где байты непредсказуемы — там граница патча), а W_patch учит
  переходы между патчами. При генерации Beam Search держит 20
  параллельных гипотез и выбирает лучшую по косинусу направления.
""")
    for name, r in results.items():
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} {name}: {r['детали']}")
    print("""
  [Гипотезы и учитель]
  Учитель кодирует правильный ответ как VSA-факт. Ученик выдвигает
  гипотезы, учитель считает косинус с фактом: cos=1.000 — верно,
  cos≈0 — ортогонально. Учитель знает ответ на русском, а ученик
  спрашивает на английском или китайском — проекции сводят оба
  к одному концепт-центроиду.

  [Точки и пространство]
  Point-JEPA превращает облако точек в один фазовый гипервектор:
  каждая точка даёт фазу e^{i(x·ωx+y·ωy+z·ωz)}, сумма фаз — форма.
  Предиктор (Widrow-Hoff) учит переходы между формами во времени.

  Вся архитектура — БЕЗ градиентов: Widrow-Hoff, Oja, Hebb, STDP.
""")
    print("═" * 64)


# ═══════════════════════════════════════════════════════════════
# 5. АНАЛИЗ ВХОДНЫХ ДАННЫХ
# ═══════════════════════════════════════════════════════════════
def analyze_data(text: str) -> dict:
    """Анализ входных данных: текст → статистика → факты → вывод.

    Анализирует: язык(и), длину, слова, структуру, "смысловые"
    признаки (числа, скобки, ключевые слова кода).
    """
    import re

    analysis = {}

    # Языки
    langs = []
    if re.search(r"[а-яёА-ЯЁ]", text):
        langs.append("RU")
    if re.search(r"[a-zA-Z]", text):
        langs.append("EN")
    if re.search(r"[\u4e00-\u9fff]", text):
        langs.append("ZH")
    analysis["языки"] = langs if langs else ["неизвестно"]

    # Длина и слова
    analysis["длина"] = len(text)
    words = re.findall(r"[a-zA-Zа-яёА-ЯЁ]+", text)
    analysis["слов"] = len(words)

    # Структура (код?)
    code_markers = sum(1 for ch in "{}();=<>+-*/\"'\\n" if ch in text)
    is_code = code_markers > len(text) * 0.15
    analysis["это_код"] = bool(is_code)

    # Числа
    nums = re.findall(r"\d+", text)
    analysis["чисел"] = len(nums)
    if nums:
        analysis["числа"] = [int(n) for n in nums[:10]]

    # Ключевые слова
    keywords = ["fn", "def", "class", "struct", "return", "if", "for",
                "while", "const", "let", "import", "данные", "data", "数据"]
    found = [k for k in keywords if k in text.lower()]
    analysis["ключевые_слова"] = found

    # Вывод (текстовое объяснение анализа)
    analysis["объяснение"] = (
        f"Я проанализировал входные данные: {analysis['длина']} символов, "
        f"{analysis['слов']} слов, языки: {', '.join(analysis['языки'])}. "
        + ("Похоже на КОД (структурные символы). " if is_code
           else "Похоже на ЕСТЕСТВЕННЫЙ ТЕКСТ. ")
        + (f"Найдено чисел: {analysis['чисел']}. " if nums else "")
        + (f"Ключевые слова: {', '.join(found[:5])}." if found else "Ключевых слов не найдено.")
    )

    return analysis


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    print("═" * 64)
    print("  ПРОВЕРКА СПОСОБНОСТЕЙ МОДЕЛИ")
    print("═" * 64)

    if mode == "--check":
        print("\n[1] Генерация (BLT-декодер)...")
        gen = check_generation()
        for name, r in gen.items():
            print(f"  {'✓' if r['ok'] else '✗'} {name}: {r['детали']}")

        print("\n[2] Гипотезы (учитель-ученик)...")
        hyp = check_hypotheses()
        for name, r in hyp.items():
            print(f"  {'✓' if r['ok'] else '✗'} {name}: {r['детали']}")

        print("\n[3] Облака точек (Point-JEPA)...")
        pt = check_point_jepa()
        for name, r in pt.items():
            print(f"  {'✓' if r['ok'] else '✗'} {name}: {r['детали']}")

        # Сводка
        all_results = {**gen, **hyp, **pt}
        total = len(all_results)
        passed = sum(1 for r in all_results.values() if r["ok"])
        print(f"\n  ВСЕГО: {passed}/{total} способностей работают")
        print(f"  {'ВСЁ РАБОТАЕТ ✓' if passed == total else 'ЕСТЬ ОГРАНИЧЕНИЯ'}")

        explain_text(all_results)

    elif mode == "--explain":
        # Только текстовое объяснение
        from astral.experiments.point_jepa import PointCloudEncoder, make_shape
        import numpy as np
        rng = np.random.default_rng(42)
        shapes = ["sphere", "cube", "plane", "line", "cluster"]
        clouds = [make_shape(s, n=64, rng=rng) for s in shapes]
        enc = PointCloudEncoder(dim=512)
        lats = [enc.encode(c) for c in clouds]
        demo_results = {
            "пример_разделимость": {"ok": True,
                                    "детали": f"cluster vs sphere cos={enc.cos(lats[4], lats[0]):.3f}"},
        }
        explain_text(demo_results)

    elif mode == "--analyze":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read()
        if not text.strip():
            text = "fn main() { let x = 4; return x + 42; }"
        a = analyze_data(text)
        print("\n  АНАЛИЗ ВХОДНЫХ ДАННЫХ:")
        for k, v in a.items():
            if k != "объяснение":
                print(f"  {k}: {v}")
        print(f"\n  {a['объяснение']}")

    else:
        print(f"  Неизвестный режим: {mode}")
        print("  Использование: capability_check.py [--check|--explain|--analyze <текст>]")


if __name__ == "__main__":
    main()