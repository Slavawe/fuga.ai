from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

import fuga_core
from antitf.rust_bridge import packed_to_torch


from fuga_persona import PersonaSynthesizer


class FugaChatEngine:
    def __init__(self, style="adaptive"):
        self.persona = PersonaSynthesizer(style=style)
        self.state = {"last_intent": None, "facts_in_row": 0}
        self.sym = fuga_core.SymbolicExecutor()
        self.ibm = fuga_core.IbmModel1()
        self.binder = fuga_core.HybridBinder(2048)
        self.flt = fuga_core.RustLinguisticFilter()
        self.last_subject = None
        # память фактов: lang -> subject -> [(relation, object)]
        self.memory = {"ru": {}, "en": {}}
        self._load_conceptnet()
        self._load_ibm()

    # ---------- загрузка ----------
    JUNK_REL = {"antonym", "distinctfrom", "relatedrelated", "synonym",
                "formof", "etymologicallyrelatedto", "derivedfrom",
                "etymologicallyderivedfrom"}

    def _load_conceptnet(self):
        n = 0
        paths = ["dataset_vault/02_world_concepts/conceptnet_semantic.jsonl",
                 "dataset_vault/02_world_concepts/conceptnet_sro_real.jsonl",
                 "dataset_vault/02_world_concepts/conceptnet_sro_ru_meaningful.jsonl"]
        for path in paths:
            try:
                fh = open(path, encoding="utf-8")
            except FileNotFoundError:
                continue
            with fh as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("relation") in self.JUNK_REL:
                        continue
                    s = "_".join(d["subject"].lower().split())
                    o = "_".join(d["object"].lower().split())
                    r = d["relation"]
                    if not s or not o or s == o or len(s) < 3:
                        continue
                    self.memory[d["lang"]].setdefault(s, []).append((r, o))
                    n += 1
        print(f"[memory] загружено {n} фактов "
              f"(ru-subjects={len(self.memory['ru'])}, en-subjects={len(self.memory['en'])})")

    def _load_ibm(self):
        pairs = []
        with open("dataset_vault/03_core_dictionary/tatoeba_real_ru_en.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ru = self._tok(d["ru"]); en = self._tok(d["en"])
                if ru and en:
                    pairs.append((ru, en))
        random.seed(0)
        random.shuffle(pairs)
        self.ibm.train(pairs[:100000], epochs=3)

    @staticmethod
    def _tok(text, n=12):
        return [w.lower() for w in re.findall(r"\w+", text.lower())][:n]

    # ---------- интенты ----------
    GREETINGS = {"привет", "здравствуй", "hello", "hi", "хай", "добрый"}
    BYE = {"пока", "bye", "до свидания", "прощай"}

    def detect_intent(self, text):
        t = text.lower().strip()
        words = set(re.findall(r"\w+", t))
        if words & self.GREETINGS and len(t.split()) <= 4:
            return "greeting"
        if any(w in words for w in self.BYE):
            return "bye"
        if re.search(r"\d\s*[+\-*/(]\s*\d|\d\s*[+\-*/]", t):
            return "calc"
        m = re.match(r"(?:запомни|remember)[:\s]+(.+)", t)
        if m:
            return ("learn", m.group(1))
        if "?" in text or any(w in words for w in
                              ("что", "кто", "какой", "расскажи", "знаешь",
                               "what", "who", "tell")):
            return "query"
        if len(t.split()) <= 6:
            return "query"
        return "chat"

    # ---------- обработка ----------
    RU_ENDINGS = ("ами", "ями", "иями", "ов", "ев", "ей", "ах", "ях", "ую", "юю",
                  "ом", "ем", "ых", "их", "ая", "яя", "ое", "ее", "ий", "ый", "ой",
                  "у", "ю", "е", "и", "ы", "а", "я", "ь")

    def stem_ru(self, w):
        for cut in range(len(w), 2, -1):
            if w[:cut] in self.memory.get("ru", {}):
                return w[:cut]
        w2 = w
        for _ in range(3):
            for e in self.RU_ENDINGS:
                if w2.endswith(e) and len(w2) - len(e) >= 3:
                    w2 = w2[:-len(e)]
                    break
            else:
                break
            if w2 in self.memory.get("ru", {}):
                return w2
        return None

    def _prefix_match(self, stem, lang):
        """Ключи памяти, начинающиеся со стема ('кошк' -> 'кошка')."""
        if not stem or len(stem) < 3:
            return None
        for key in self.memory.get(lang, {}):
            if key.startswith(stem):
                return key
        return None

    def find_subject(self, text, lang="ru"):
        """Точное совпадение -> стем/префикс -> VSA-nearest по последнему слову."""
        toks = self._tok(text)
        best_exact = None
        for w in reversed(toks):                    # существительное чаще в конце
            if w in self.memory[lang]:
                return w
            st = self.stem_ru(w) if lang == "ru" else None
            if st is None:
                # грубый стем: срезаем флексию и ищем префикс среди ключей
                w2 = w
                for e in self.RU_ENDINGS:
                    if w2.endswith(e) and len(w2) - len(e) >= 3:
                        w2 = w2[:-len(e)]
                        break
                if w2 != w:
                    st = self._prefix_match(w2, lang) or None
            if st:
                return st
            if best_exact is None and len(w) > 3:
                best_exact = w
        if not self.memory[lang] or not toks:
            return best_exact
        # VSA-nearest по разнородным ключам здесь шумит (морфология RU),
        # поэтому честно возвращаем None — сработает кросс-языковой фолбэк.
        return None

    def query_facts(self, subject, lang="ru", limit=3):
        facts = self.memory[lang].get(subject, [])
        out, used = [], set()
        for r, o in facts:
            if r in used:
                continue
            used.add(r)
            out.append((r, o))
            if len(out) >= limit:
                break
        return out

    REL_RU = {"isa": "это", "usedfor": "используется для",
              "capableof": "умеет", "causes": "приводит к",
              "hasproperty": "обладает свойством", "partof": "часть",
              "atlocation": "бывает в", "receivesaction": "подвергается"}

    def verbalize(self, subject, rel, obj, lang="ru"):
        obj_h = obj.replace("_", " ")
        if lang == "ru":
            if rel == "isa":
                return f"{subject.replace('_', ' ')} — это {obj_h}"
            word = self.REL_RU.get(rel, rel)
            return f"{subject.replace('_', ' ')} {word} {obj_h}".strip()
        tmpl = {"isa": "{} is a {}", "capableof": "a {} can {}",
                "usedfor": "{} is used for {}", "causes": "{} causes {}"}
        return tmpl.get(rel, "{} {} {}").format(
            subject.replace("_", " "), rel, obj_h)

    def respond(self, user_input: str) -> str:
        text = user_input.strip()
        intent = self.detect_intent(text)

        if intent == "greeting":
            return (self.persona.greeting() +
                    " Факты, вычисления ('12*7+3') или обучение: 'запомни: X — это Y'.")
        if intent == "bye":
            return "Память сохранена. До связи."
        if intent == "calc":
            expr = re.sub(r"[^\d+\-*/(). ]", "", text)
            try:
                val = self.sym.eval_expression(expr)
                self.state["last_intent"] = "calc"
                return self.persona.format_math(expr, val)
            except Exception as e:
                return f"Символьное ядро не разобрало '{expr}': {e}"
        if isinstance(intent, tuple) and intent[0] == "learn":
            body = intent[1]
            parts = re.split(r"\s*[—\-:]?\s*—\s*|\s+это\s+|\s+is\s+", body, maxsplit=1)
            if len(parts) == 2:
                s_ = "_".join(self._tok(parts[0], 3))
                rest = re.sub(r"^(это|is)\s+", "", parts[1].strip())
                if re.search(r"умеет|can|могут", body):
                    rel_guess = "capableof"
                elif re.search(r"свойств", body):
                    rel_guess = "hasproperty"
                else:
                    rel_guess = "isa"
                o_ = "_".join(self._tok(rest, 6)) or rest
                self.memory.setdefault("ru", {}).setdefault(s_, []).append((rel_guess, o_))
                self.state["last_intent"] = "learn"
                return self.persona.format_ingest(s_, rel_guess, o_)
            return "Формат обучения: 'запомни: X — это Y'."
        if intent == "query":
            subj = self.find_subject(text, "ru")
            # валидность субъекта: он должен быть префиксно связан с текстом,
            # иначе стем-матчинг утащил нас на чужое слово
            if subj is not None:
                toks_list = [t for t in self._tok(text) if len(t) >= 3]
                if toks_list and not any(
                        subj.startswith(t[:4]) or t.startswith(subj[:4])
                        for t in toks_list):
                    subj = None

            if subj is not None:
                if self.state.get("last_intent") != "query":
                    self.state["facts_in_row"] = 0
                facts = self.query_facts(subj, "ru")
                if facts:
                    self.last_subject = subj
                    self.state["facts_in_row"] += 1
                    body = ". ".join(self.verbalize(subj, r, o) for r, o in facts)
                    return self.persona.format_fact(subj, body, self.state)

            # кросс-языковой фолбэк: RU-слово -> IBM -> EN-память
            en_subj = None
            for w in reversed(self._tok(text)):
                for cand, _p in self.ibm.translate_topk(w, 5):
                    if cand in self.memory["en"]:
                        en_subj = cand
                        break
                    pm = self._prefix_match(cand, "en")
                    if pm:
                        en_subj = pm
                        break
                if en_subj:
                    break
            if en_subj:
                facts_en = self.query_facts(en_subj, "en", limit=3)
                if facts_en:
                    self.last_subject = en_subj
                    self.state["facts_in_row"] += 1
                    body = ". ".join(self.verbalize(en_subj, r, o) for r, o in facts_en)
                    return self.persona.cross_language_note(en_subj, body)
            elif self.last_subject and len(text.split()) <= 5 and not [
                    t for t in self._tok(text) if len(t) >= 4]:
                subj = self.last_subject                      # контекстный стек
                facts = self.query_facts(subj, "ru")
                if facts:
                    intros = ["Возвращаясь к теме:", "Продолжим про это:"]
                    body = ". ".join(self.verbalize(subj, r, o) for r, o in facts)
                    return f"{random.choice(intros)} {body}."
            self.state["last_intent"] = "query"
            return self.persona.fallback_unknown_subject(subj)
        # chat fallback
        subj = self.find_subject(text, "ru")
        if subj:
            self.last_subject = subj
            return f"Услышал про '{subj}'. Спроси конкретно — расскажу факты."
        return "Я тебя услышал. Назови объект или дай вычисление."


def demo():
    style = sys.argv[2] if len(sys.argv) > 2 else "adaptive"
    eng = FugaChatEngine(style=style)
    print(f"[persona style: {style}]")
    script = [
        "Привет!",
        "Расскажи про кошку",
        "А что такое собака?",
        "вода?",
        "сколько будет 12*7+3?",
        "запомни: фуга — это бестокеновый интеллект",
        "что такое фуга?",
        "Пока!",
    ]
    print("\n===== ДИАЛОГ =====")
    for turn in script:
        print(f"Ты:   {turn}")
        print(f"Fuga: {eng.respond(turn)}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        eng = FugaChatEngine()
        print("Fuga chat (exit/пока — выход)")
        while True:
            try:
                inp = input("Ты:   ")
            except EOFError:
                break
            if inp.lower() in ("exit", "quit"):
                break
            print(f"Fuga: {eng.respond(inp)}")
