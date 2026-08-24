# Fuga AI — Telegram Bot

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
cargo run --release --bin fuga-tgbot
```

Токен берётся из (по приоритету):
1. `FUGA_TG_TOKEN` env var
2. `fuga.token` файл
3. Хардкоженный fallback в `src/bin/tgbot.rs`

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
3. Заменить токен в `src/bin/tgbot.rs` или `FUGA_TG_TOKEN`
4. Установить размерность: `FUGA_DIM=256` (default) или 512/8192

## Ограничения

- Текущая размерность VSA = 256 (по умолчанию), 512/1024/8192 — через `FUGA_DIM`
- Cube side = 4×4×4 (64 ячейки). Увеличение side через `FugaAI::new(dim, side, window)`
- Для продакшн нужен более умный ответ-генератор (LLM-style natural language из retrieved facts)
