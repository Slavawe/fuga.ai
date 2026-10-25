# Fuga AI — Telegram Bot

> **LEGACY (omni-стек до байтового стандарта)** — документ описывает tgbot эпохи VSA-куба;
> факты выверены по коду 15.10.2026 (см. audit/docs_config.md). Актуальный стек — AGENTS.md.

Telegram-бот для семантического поиска по обученной модели Fuga AI (VSA + плавающий куб).

## Быстрый старт

### 1. Тренировка на корпусе

```bash
cargo run --release -- train corpus_rus_eng.jsonl --dim 256 --save fuga_cube.bin
```

Создаст:
- `fuga_cube.bin` — 4×4×4×256 биполярный куб + memory store
- `fuga_cube_mem.bin` — текстовые индексы для retrieval

### 2. Запуск бота

```bash
./start-tgbot.sh
```

Или напрямую:

```bash
cargo run --release --bin tgbot
```

Токен берётся из (по приоритету):
1. `FUGA_TG_TOKEN` env var
2. `fuga.token` файл
3. Fallback НЕТ: без токена бот печатает ошибку и завершается с exit(1)
   (tgbot.rs:954-960; хардкод-токена в исходнике нет — проверено аудитом 15.10)

## Команды в Telegram

| Команда | Описание |
|---|---|
| `/start`, `/help` | Показать список команд |
| `/ask <вопрос>` | Ответ по обученной memory |
| `/solve <задача>` | Декомпозиция задачи + multi-step answer |
| `/think <текст>` | Запустить Weaver Engine на тексте |
| `/stats` | Энтропия куба, coherence, размер memory |
| `/train <corpus.jsonl>` | Обучить на корпусе прямо в чате |

## Примеры

```
/ask What is gravity?
/ask Что такое гравитация?
/solve How does refraction work and what causes color?
/think def hello(): print("world")
/stats
/train corpus_rus_eng.jsonl
```

## Архитектура

```
Telegram API (long-poll)
    ↓
tgbot.rs (long-poll, dispatch)
    ↓
fuga AI Core
    ├─ weaver/ — tokenization, bundle/unbundle
    ├─ ai/core — think/answer/solve
    ├─ ai/memory_store — text→vector retrieval
    └─ core/wave_cube — associative storage
```

## Разработка

Создание своего бота:
1. Открыть [@BotFather](https://t.me/BotFather) в Telegram
2. `/newbot` → получить токен
3. Задать токен через `FUGA_TG_TOKEN` или файл `fuga.token` (в исходнике токена НЕТ)
4. Размерность: `FUGA_DIM` — по умолчанию 8192 (tgbot.rs:963-965)

## Ограничения

- Текущая размерность VSA = 8192 (по умолчанию, через `FUGA_DIM`)
- Cube side = 4×4×4 (64 ячейки). Увеличение side через `FugaAI::new(dim, side, window)`
- Для продакшн нужен более умный ответ-генератор (LLM-style natural language из retrieved facts)
