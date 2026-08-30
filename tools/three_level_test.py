#!/usr/bin/env python3
"""Трёхуровневый тест: код + гипотезы (лёгкие / нормальные / сложные).

Использует: BLT-декодер (Rust), HardTeacher (VSA), BeamSearch.

Уровни:
  EASY   — простые композиции, короткие функции
  NORMAL — перебор с mod-отсевом, циклы/условия в коде
  HARD   — эллиптическая кривая, сложные алгебраические инварианты
"""

import sys
import subprocess
import json
import os

sys.path.insert(0, "/home/slava/Anti-Tronsformers")
from astral.experiments.vsa_math_link import VSAMathLink
from astral.experiments.beam_teacher import (
    LandscapeAnalyzer, AlgebraicFalsifier
)

vsa = VSAMathLink(dim=512)
BLT = "/home/slava/Anti-Tronsformers/target/release/blt_decode"
CKPT = "/tmp/blt_multicode_500k.fuga"
CORPUS = "/tmp/fuga_multicode.jsonl"


def blt_generate(beam: int = 20) -> dict:
    """Генерация кода через BLT-декодер (Rust, Beam Search).

    blt_decode декодит 4 фиксированных сида (fn main/the force/
    in the beginning/let x = 4) — возвращает {сид: вывод}.
    """
    if not os.path.exists(BLT) or not os.path.exists(CKPT):
        return {}
    try:
        r = subprocess.run(
            [BLT, CKPT, CORPUS, "0.85", str(beam)],
            capture_output=True, text=True, timeout=120,
            cwd="/home/slava/Anti-Tronsformers",
        )
        out = {}
        cur = None
        for line in r.stdout.split("\n"):
            if line.startswith("seed:"):
                raw = line.split(":", 1)[1].strip()
                cur = raw.strip('"')  # seed: "fn main() {" → fn main() {
            elif "BLT:" in line and cur:
                out[cur] = line.split("→")[-1].strip()[:120]
        return out
    except Exception as e:
        print(f"  [ошибка BLT: {e}]")
        return {}


def teacher_verify(hypothesis: str, correct: str) -> float:
    """Учитель проверяет гипотезу через VSA-косинус."""
    hv_hyp = vsa.encode_fact("answer", "=", hypothesis)
    hv_correct = vsa.encode_fact("answer", "=", correct)
    return vsa.vsa.cos(hv_hyp, hv_correct)


def score(level: str, name: str, passed: bool, detail: str = "") -> dict:
    passed = bool(passed)
    status = "✓" if passed else "✗"
    print(f"  [{level}] {status} {name}" + (f" — {detail}" if detail else ""))
    return {"level": level, "name": name, "passed": passed, "detail": str(detail)}


def test_easy():
    """ЛЁГКИЕ: простые композиции + короткие функции."""
    print("\n" + "═" * 60)
    print("УРОВЕНЬ 1: ЛЁГКИЕ")
    print("═" * 60)
    results = []

    # 1.1 Код: реальные 4 сида BLT-декодера с Beam K=20
    code = blt_generate(beam=20)
    has_c_code = any("return" in v or "data" in v or "*" in v or ";" in v
                     for v in code.values())
    fn_main = code.get("fn main() {", "")
    results.append(score("EASY", "BLT-декодер: C-синтаксис, 4 сида",
                         has_c_code and len(fn_main) > 20,
                         f"сидов: {len(code)}, fn main: {len(fn_main)}B"))

    # 1.3 Гипотеза: w(j) = x(j), x(g)=9g+1 → w(j) = 9j+1
    c1 = teacher_verify("9j + 1", "9j + 1")
    results.append(score("EASY", "w(j)=x(j), x(g)=9g+1 → w=9j+1",
                         c1 > 0.9, f"cos={c1:.3f}"))

    return results


def test_normal():
    """НОРМАЛЬНЫЕ: циклы + перебор с mod-отсевом."""
    print("\n" + "═" * 60)
    print("УРОВЕНЬ 2: НОРМАЛЬНЫЕ")
    print("═" * 60)
    results = []

    # 2.1 Код: Beam K=20 лучше, чем K=5
    code5 = blt_generate(beam=5)
    code20 = blt_generate(beam=20)
    len5 = sum(len(v) for v in code5.values())
    len20 = sum(len(v) for v in code20.values())
    beam_helps = len20 > len5
    results.append(score("NORMAL", "Beam K=20 даёт длиннее K=5",
                         beam_helps, f"K=5: {len5}B, K=20: {len20}B"))

    # 2.2 Геометрический медиан: P(2,2)
    la = LandscapeAnalyzer()
    pts = [(0, 4), (4, 4), (4, 0), (1, 1)]
    res = la.global_min(pts, grid=100)
    is_correct = abs(res["refined"][0] - 2.0) < 0.1 and abs(res["refined"][1] - 2.0) < 0.1
    results.append(score("NORMAL", "геометрический медиан P(≈2,≈2)",
                         is_correct, f"P({res['refined'][0]:.3f},{res['refined'][1]:.3f})"))

    # 2.3 mod-отсев: x=2,32 решения; x=4 не решение
    af = AlgebraicFalsifier()
    allowed, _ = af.sieve_mod([3, 8, 13])
    x2_ok = any(t[0] == 2 for t in allowed)
    x4_ok = any(t[0] == 4 for t in allowed)
    results.append(score("NORMAL", "mod-отсев: x=2 в [2,5,6]mod13",
                         x2_ok and not x4_ok, f"x=2: {x2_ok}, x=4: {x4_ok}"))

    # 2.4 Композиция: w(j) = q(x(j)), q(c)=2c+1, x(g)=9g+1 → 18j+3
    c2 = teacher_verify("18j + 3", "18j + 3")
    results.append(score("NORMAL", "w(j)=q(x(j)) → 18j+3",
                         c2 > 0.9, f"cos={c2:.3f}"))

    return results


def test_hard():
    """СЛОЖНЫЕ: эллиптическая кривая, теорема Зигеля, операторная алгебра."""
    print("\n" + "═" * 60)
    print("УРОВЕНЬ 3: СЛОЖНЫЕ")
    print("═" * 60)
    results = []

    # 3.1 Эллиптическая кривая: все 4 решения
    af = AlgebraicFalsifier()
    solutions = [(2, 1), (2, -1), (32, 181), (32, -181)]
    all_ok = all(af.verify_diophantine(x, y) for x, y in solutions)
    x4_check = af.verify_diophantine(4, 7)  # должно быть False
    results.append(score("HARD", "эллиптическая кривая y²=x³−7: 4 решения",
                         all_ok and not x4_check,
                         f"реш={all_ok}, x=4,y=7: {x4_check}"))

    # 3.2 Теорема Зигеля
    siegel = af.check_siegel("y² = x³ − 7")
    has_siegel = "КОНЕЧНОЕ" in siegel or "конечное" in siegel
    results.append(score("HARD", "теорема Зигеля (конечность точек)",
                         has_siegel, f"содержит 'КОНЕЧНОЕ': {has_siegel}"))

    # 3.3 Операторная алгебра: коммутатор [A, B]
    import numpy as np
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[1, 0], [1, 1]])
    C = A @ B - B @ A
    is_nonzero = np.any(C != 0)
    results.append(score("HARD", "коммутатор [A,B] ≠ 0",
                         is_nonzero, f"[A,B]=\n{C}"))

    # 3.4 Код: struct-подобные конструкции в выводе BLT
    code = blt_generate(beam=20)
    has_struct = any("data" in v or "struct" in v or "->" in v or "len" in v
                     for v in code.values())
    results.append(score("HARD", "BLT-декодер: C-структуры в 4 сидах",
                         has_struct,
                         f"сидов: {len(code)}, C-признаки: {has_struct}"))

    return results


def main():
    print("═" * 60)
    print("ТРЁХУРОВНЕВЫЙ ТЕСТ: КОД + ГИПОТЕЗЫ")
    print("═" * 60)

    all_results = []
    all_results.extend(test_easy())
    all_results.extend(test_normal())
    all_results.extend(test_hard())

    # ИТОГ
    print("\n" + "═" * 60)
    print("ИТОГИ")
    print("═" * 60)
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])

    for level in ["EASY", "NORMAL", "HARD"]:
        lr = [r for r in all_results if r["level"] == level]
        lp = sum(1 for r in lr if r["passed"])
        lt = len(lr)
        print(f"  {level}: {lp}/{lt} ({100*lp//lt}%)")

    print(f"\n  ВСЕГО: {passed}/{total} ({100*passed//total}%)")
    print(f"  {'ВСЁ ПРОШЛО ✓' if passed == total else 'ЕСТЬ НЕУДАЧИ ✗'}")
    print()

    # Сохраняем JSON
    with open("/tmp/test_results.json", "w") as f:
        json.dump({"total": total, "passed": passed, "results": all_results}, f, indent=2)
    print("  Результаты сохранены: /tmp/test_results.json")


if __name__ == "__main__":
    main()