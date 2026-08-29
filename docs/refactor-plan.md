# Фазированный план: Монолит → Конструктор

## Цель
Разбить монолиты (main.rs 10.7K строк, tm_generate.rs 2.3K,
self_mirror.rs 2.2K, htm_temporal.rs 1.6K) на модульный
«конструктор» — независимые компоненты, читаемые и модернизируемые.

## Принцип
Каждая группа функций = отдельный модуль со своей ответственностью.
main.rs остаётся тонким диспетчером (match command → модуль).

## Фаза 1 — src/cli/ (декомпозиция main.rs, 141 fn → модули)
| Модуль | Содержимое | Риск |
|---|---|---|
| cli/args.rs | parse_* хелперы (1039-1135) | SAFE |
| cli/analyze.rs | run_analyze/fix/translate/scan/quality | CAREFUL |
| cli/train.rs | run_fisig_train/train_unified/train_stack/*_source | CAREFUL |
| cli/tm.rs | run_htm_*/run_tm_gen/lex_rust/ts_* | CAREFUL |
| cli/crystal.rs | run_crystal_* (7851-9109) | CAREFUL |
| cli/agent.rs | run_agent/agent_loop/generate/merge | CAREFUL |
| cli/query.rs | run_query/solve/codegen/ask/think/readout | CAREFUL |
| cli/reflect.rs | run_reflect/self_query/refactor/docs | CAREFUL |
| cli/sim.rs | run_sim/perceive/reactor/3d | CAREFUL |
| cli/print.rs | print_rust_code/print_cpp_code/print_usage | SAFE |

## Фаза 2 — src/ai/ декомпозиция
| Файл | LOC | Действие |
|---|---|---|
| tm_generate.rs | 2325 | decoders/ → по одному декодеру на файл |
| self_mirror.rs | 2242 | index/ + train/ + predict/ |
| htm_temporal.rs | 1617 | tm/ + serialize/ + predict/ |

## Фаза 3 — Rust bin-стенды
42 стенда → production/tools/experiments (feature-gated)

## Фаза 4 — Python astral/ (уже начато)
core/models/ingest/agents/experiments + удаление sys.path.insert

## Верификация каждой фазы
```bash
cargo build --release --bin fuga        # после каждой выгрузки модуля
cargo test --release --lib              # lib не ломать
python -m pytest astral/tests/          # Python не ломать
```
