# Fuga Cognitive Engine (Anti-Tronsformers)

Собственный «агентский стек» без внешних LLM: VSA + JEPA + KAN + резонаторы.
Байтовое двухскоростное латентное обучение → генерация кода и текста.

## Быстрый старт

```bash
# Rust-ядро
cargo build --release
cargo test --release --lib          # 137/137 тестов

# Python-слой (astral)
cd astral && pip install -e .

# PyO3-мост (fuga_core)
cd fuga-core && PATH="$HOME/.cargo/bin:$PATH" ../.venv/bin/maturin develop --release
```

## Ключевые модули

| Модуль | Назначение |
|--------|-----------|
| `src/` | Rust-ядро: TM, JEPA, H-JEPA, OWM, KAN, GPU, декодеры |
| `src/cli/` | 13 модулей CLI (main.rs = 1,085 строк диспетчер) |
| `src/bin/` | 42 стенда (production/tools/experiments) |
| `astral/` | Python-слой: pipeline, memory, модели, агенты |
| `astral/models/unified_engine.py` | Единый движок: speak/codegen/combine |
| `astral/models/resonator_hdc.py` | HDC + FPE-VSA + PhaseCrystal резонаторы |
| `astral/models/lang_jepa_adapter.py` | Концепт-пространство (EMA-таргет) |
| `astral/models/barlow_twins.py` | Анти-коллапс (кросс-корреляция) |
| `cpp/` | C++-порт FUGA1 (бинарно-совместим) |

## Формат чекпоинтов: FUGA1

Теги: 1=LOCAL_W, 2=PATCH_W, 3=OWM_P, 4=META, 5=HJEPA, 6=KAN_C,
7=MACRO_W, 8=CONCEPT_W. C++/Rust бинарно-совместимы (проверено).

## Документация

- `docs/architecture.md` — слои и источник истины
- `docs/ROADMAP.md` — **единая дорога обучения** (актуальный план)
- `docs/refactor-plan.md` — план «монолит → конструктор»
- `docs/decisions/` — ADR (001-003)
- `docs/archive/` — исторические отчёты (v10 HybridCore, 14.08)
- `AGENTS.md` — состояние обучения, калибровки, известные грабли
- `SESSION_LOG.md` — журнал изменений с 24.08

## Лучший декодер (AGENTS.md v6.2)

```
V2: α=0, τ=0.01, corridor=0, min_cos=0.001, β=0,
    rep_word=0.20, rep_phrase=0.8, window=9 (ctx=8), PHR_LEN=12
MB3: + βm·cos(W_macro·x, lat) + βc·cos(concept, lat)
```
