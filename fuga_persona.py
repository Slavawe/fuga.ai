from __future__ import annotations

import random


class PersonaSynthesizer:
    """Сборщик отклика из слотов данных + стиля + состояния диалога.

    style: adaptive | informative | dry | ironic | laconic
    В adaptive выбор пула зависит от истории (серия фактов, математика,
    только что обучили) — клише «Я знаю: ... О чём ещё спросишь?» исчезают.
    """

    def __init__(self, style: str = "adaptive", seed: int = 7):
        self.style = style
        self.rng = random.Random(seed)

        self.greetings = {
            "informative": ["На связи. Память VSA загружена, слушаю.",
                            "Контур Fuga активен. Готов разбирать контекст."],
            "dry": ["Система готова. Вводите запрос.",
                    "Слушаю. Формат любой: факт, вычисление, обучение."],
            "ironic": ["Опять ты. Ну давай, чем на этот раз займёмся?",
                       "Я тут, векторы скучали. Работаем."],
            "laconic": ["Слушаю.", "Fuga на связи."],
        }

        self.fact_openers = {
            "informative": ["Фиксирую в памяти:", "Извлечено из графа знаний:",
                            "Смысловой вектор указывает:"],
            "dry": ["Данные:", "Результат запроса:"],
            "ironic": ["Ну ладно, держи:", "Это даже интересно. Смотри:",
                       "Подобрал из недр памяти:"],
            "laconic": ["Кратко:", ""],
        }

        self.closers = {
            "fact": {
                "informative": ["Есть ещё вопросы по теме?", "Разворачиваем граф дальше?"],
                "dry": ["Следующий запрос.", ""],
                "ironic": ["Что ещё покопаем?", "Продолжим экскурсию по моей памяти?"],
                "laconic": ["", "Дальше?"],
            },
            "math": {
                "informative": ["Вычислено символьным ядром — без галлюцинаций.",
                                "Точный результат, Rust не ошибается."],
                "dry": ["Посчитано.", "Точно."],
                "ironic": ["Калькулятор нервно курит.", "Арифметика — мой комфортный жанр."],
                "laconic": ["Точно.", ""],
            },
            "ingest": {
                "informative": ["Встроено в биполярный бандл памяти.",
                                "Связка зафиксирована в VSA-графе."],
                "dry": ["Записано.", "Зафиксировано."],
                "ironic": ["Теперь я это знаю. Спасибо, что лишаешь неведения.",
                           "Записал. Не подведи, спрашивая позже."],
                "laconic": ["Запомнил.", "Есть."],
            },
        }

        # контекстные прологи (зависят от серии ходов)
        self.streak_fact_open = [
            "Продолжаем погружение:", "Углубляемся:",
            "Тема развивается — вот ещё:"]
        self.after_math_open = ["Rust-ядро готово к новым расчетам.",
                                "Из цифр обратно в смыслы:"]
        self.after_learn_open = ["Свежая запись в бандле проверяется..."]

        # «чертики личности» — стилевые смещения выбираются детерминированно
        # от VSA-хеша субъекта: одно и то же слово -> один и тот же стиль подачи
        self._style_cache = {}

    def _effective_style(self, subject: str | None) -> str:
        if self.style != "adaptive":
            return self.style
        if subject is None:
            return self.rng.choice(["informative", "dry"])
        if subject not in self._style_cache:
            self._style_cache[subject] = self.rng.choice(
                ["informative", "dry", "ironic", "laconic"])
        return self._style_cache[subject]

    def greeting(self) -> str:
        return self.rng.choice(self.greetings[self._effective_style(None)])

    def opening(self, context_state: dict, subject: str | None) -> str:
        """Пролог по состоянию диалога, а не случайный."""
        st = context_state
        if st.get("last_intent") == "learn":
            return self.rng.choice(self.after_learn_open)
        if st.get("last_intent") == "calc":
            return self.rng.choice(self.after_math_open)
        if st.get("facts_in_row", 0) >= 2:
            return self.rng.choice(self.streak_fact_open)
        return ""

    def format_fact(self, subject: str | None, fact_text: str,
                    context_state: dict) -> str:
        style = self._effective_style(subject)
        op = self.opening(context_state, subject)
        body_opener = self.rng.choice(self.fact_openers[style])
        closer = self.rng.choice(self.closers["fact"][style])
        fact = fact_text.strip()
        if fact and not fact.endswith((".", "!", "?")):
            fact += "."
        parts = [p.strip() for p in (op, f"{body_opener} {fact}".strip(), closer) if p]
        return " ".join(parts)

    def format_math(self, expr: str, value: float) -> str:
        style = self._effective_style(None)
        closer = self.rng.choice(self.closers["math"][style])
        core = f"[symbolic] {expr} = {value:g}"
        return f"{core}. {closer}" if closer else core

    def format_ingest(self, subject: str, rel: str, obj: str) -> str:
        style = self._effective_style(subject)
        closer = self.rng.choice(self.closers["ingest"][style])
        core = f"Зафиксировал связку ({subject} ──[{rel}]──► {obj})."
        return f"{core} {closer}".strip()

    def fallback_unknown_subject(self, subject: str | None) -> str:
        if subject:
            return (f"Про '{subject}' в графе пусто. Научи: "
                    f"'запомни: {subject} — это ...'")
        return "Назови объект или дай выражение — разберу."

    def cross_language_note(self, en_subject: str, fact_text: str) -> str:
        style = self._effective_style(en_subject)
        pre = {"informative": "Русской карточки нет, достаю из английского графа:",
               "dry": "Источник: EN-граф.",
               "ironic": "По-русски не записано, но англоязычная память помнит:",
               "laconic": "EN-граф:"}[style]
        return f"{pre} {fact_text}"
