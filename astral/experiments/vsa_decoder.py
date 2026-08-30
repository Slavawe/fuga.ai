"""VSA-DECODER — генерация кода/языка через VSA+BLT память + HardTeacher.

Связка пункта 1: «HardTeacher с реальным декодером».

Как работает генерация:
  1. Контекст (байты) → BLT-патчи → VSA-гипервекторы
  2. VSA+BLT память предсказывает следующий патч (по косинусу)
  3. HardTeacher ОЦЕНИВАЕТ кандидатов против VSA-фактов (аксиом)
  4. Выбираем байт: argmax по (память · учитель) с rep-штрафами

Это «teacher-guided decoding»: память предлагает, учитель фильтрует.
В отличие от чистого argmax — генерация направлена к ОСМЫСЛЕННОМУ
(код/математика), а не к случайной статистике байт.
"""

from __future__ import annotations

import numpy as np

from astral.experiments.hard_teacher_vsa_blt import VSA_BLT_Memory, HardTeacher

# Rust-ядро для batch-косинуса (fuga_core.vsa_cos_batch), фолбэк numpy
try:
    import fuga_core
    _RUST = fuga_core if hasattr(fuga_core, "vsa_cos_batch") else None
except Exception:
    _RUST = None


class VSADecoder:
    """Генерация через VSA+BLT память + учитель.

    window: сколько байт истории смотреть для предсказания
    rep_word: штраф за повторные патчи (как rep_word в Rust V2)
    teacher_weight: насколько сильно учитель влияет на выбор
    """

    def __init__(self, memory: VSA_BLT_Memory, teacher: HardTeacher,
                 window: int = 4, rep_word: float = 0.20,
                 teacher_weight: float = 0.30):
        self.memory = memory
        self.teacher = teacher
        self.window = window
        self.rep_word = rep_word
        self.teacher_weight = teacher_weight
        self.counter: dict[str, int] = {}

    def _context_patch(self, history: bytes) -> bytes:
        """Последний байт истории как «текущий патч» (упрощённо)."""
        # берём последние window байт как патч-контекст
        return history[-self.window:] if history else b" "

    def _score_candidates(self, history: bytes, top_k: int = 3) -> dict:
        """Оценить топ-K кандидатов: память + учитель + rep-штраф.

        Память даёт топ-K следующих патчей (по частоте/косинусу),
        учитель штрафует «неосмысленные» (не код/не математика).
        score = mem_cos + teacher_weight·teacher_cos − rep_word·count
        """
        ctx = self._context_patch(history)
        # 1. Память: топ-K кандидатов
        candidates = self._top_k_candidates(ctx, top_k)
        scored = []
        for pid, mem_cos in candidates:
            # 2. Учитель: осмысленность патча
            teacher_cos = self._teacher_consistency(self.memory.bytes_by_id.get(pid, b" "))
            count = self.counter.get(str(pid), 0)
            score = mem_cos + self.teacher_weight * teacher_cos - self.rep_word * count
            scored.append({"pid": pid, "score": score, "mem_cos": mem_cos,
                           "teacher_cos": teacher_cos, "count": count})
        scored.sort(key=lambda s: -s["score"])
        return scored

    def _top_k_candidates(self, ctx: bytes, k: int) -> list[tuple[int, float]]:
        """Топ-K следующих патчей из памяти (частота + косинус)."""
        pid, _ = self.memory._patch_hv(ctx)
        # 1. Из переходов: частые следующие
        freq: dict[int, int] = {}
        if pid in self.memory.transitions:
            for nxt in self.memory.transitions[pid]:
                freq[nxt] = freq.get(nxt, 0) + 1
        if freq:
            total = sum(freq.values())
            cands = [(p, c / total) for p, c in
                     sorted(freq.items(), key=lambda kv: -kv[1])[:k]]
            return cands
        # 2. Поиск по косинусу (если патч не встречался)
        hv = self.memory.hv_by_id.get(pid)
        if hv is None:
            return [(pid, 0.0)]
        cos_map = [(known_pid, self.memory.vsa.cos(hv, known_hv))
                   for known_pid, known_hv in self.memory.hv_by_id.items()
                   if known_pid != pid]
        cos_map.sort(key=lambda kv: -kv[1])
        return cos_map[:k]

    def _teacher_consistency(self, patch: bytes) -> float:
        """Осмысленность патча через учителя.

        +1 за «кодовые» байты (ASCII, структура кода),
        −1 за мусор (бинарные, нечитаемые).
        Плюс близость к VSA-фактам (математика).
        """
        if not patch:
            return 0.0
        # читаемость: доля печатных ASCII
        printable = sum(1 for b in patch if 32 <= b < 127)
        readability = printable / len(patch)
        # структура кода: скобки/операторы
        code_chars = b"{}();=<>+-*/\"'\\n\t "
        structure = sum(1 for b in patch if b in code_chars) / len(patch)
        base = readability * 0.6 + structure * 0.4
        # +VSA-близость к фактам учителя (математика) — batch через Rust
        hv = self.memory.vsa.item(patch.decode("latin-1"))
        # собираем HV всех фактов в матрицу [B, dim]
        fact_hvs = []
        for fact_str, _ in self.teacher.fact_hvs:
            parts = fact_str.lower().split()
            if len(parts) >= 3:
                fact_hv = self.teacher.vsa_math.encode_fact(
                    parts[0], parts[1], " ".join(parts[2:]))
                fact_hvs.append(fact_hv.numpy().astype(np.float32))
        if not fact_hvs:
            return base
        fact_mat = np.stack(fact_hvs)
        hv_f = hv.astype(np.float32)
        if _RUST is not None:
            cosines = np.asarray(_RUST.vsa_cos_batch(fact_mat, hv_f))
        else:
            norms = np.linalg.norm(fact_mat, axis=1) * np.linalg.norm(hv_f) + 1e-9
            cosines = (fact_mat @ hv_f) / norms
        base += 0.4 * float(np.mean(cosines))
        return base

    def generate(self, seed: str, max_bytes: int = 200) -> str:
        """Сгенерировать продолжение от seed (teacher-guided, топ-K)."""
        out = seed.encode("utf-8")
        self.counter.clear()
        for _ in range(max_bytes):
            history = out[-self.window:]
            scored = self._score_candidates(history, top_k=3)
            if not scored:
                break
            best = scored[0]
            next_patch = self.memory.bytes_by_id.get(best["pid"], b" ")
            if not next_patch:
                break
            out += next_patch
            self.counter[str(best["pid"])] = best.get("count", 0) + 1
            if len(out) > 60:
                tail = out[-6:]
                if out.count(tail) > 3:
                    break
        return out.decode("utf-8", errors="replace")


def demo():
    print("=== VSA-DECODER: генерация через память + учитель ===\n")

    # Обучаем память на коде и математике + добавляем «мусорные» бинарные патчи
    memory = VSA_BLT_Memory(dim=512)
    teacher = HardTeacher()
    corpus = [
        "fn main() { println!(\"hello world\"); }",
        "fn process(x) { return x * 2; }",
        "fn compute() { let y = x + 1; }",
        "fn add(a, b) { return a + b; }",
        "sum of angles in a triangle equals 180 degrees",
        "the quick brown fox jumps over the lazy dog",
    ]
    for t in corpus:
        memory.feed(t)
    # Мусорные бинарные патчи (для теста учителя)
    garbage = [
        "\x00\x01\x02\x03\xff\xfe",
        "\x80\x81\x82\x83\x84\x85",
        "\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff",
        "\x01\x02\x03\x04\x05\x06\x07\x08",
    ]
    for g in garbage:
        memory.feed(g)

    print(f"1. Память обучена: {memory.stats()['unique_patches']} патчей, "
          f"{memory.stats()['transitions']} переходов (вкл. мусорные)")

    # Декодер с учителем и без
    dec = VSADecoder(memory, teacher, window=4, teacher_weight=0.3)
    dec_plain = VSADecoder(memory, teacher, window=4, teacher_weight=0.0)

    print("\n2. Генерация: с учителем vs без (teacher отсекает мусор):")
    for seed in ["fn main() {", "the force of", "let x = 4"]:
        out_t = dec.generate(seed, max_bytes=40)
        out_p = dec_plain.generate(seed, max_bytes=40)
        # доля печатных ASCII (чем выше, тем осмысленнее)
        pt = sum(1 for c in out_t if 32 <= ord(c) < 127) / len(out_t)
        pp = sum(1 for c in out_p if 32 <= ord(c) < 127) / len(out_p)
        print(f"   '{seed}':")
        print(f"     с учителем:  {out_t!r}  (печать {pt:.0%})")
        print(f"     без учителя: {out_p!r}  (печать {pp:.0%})")

    print("\n=== VSA-DECODER OK ===")


if __name__ == "__main__":
    demo()