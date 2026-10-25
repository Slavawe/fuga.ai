# Fuga Agent Goals

Актуально на 15.10.2026 (переписано в ходе Windows-аудита; прежняя цель
«train H-JEPA until L0 < 0.8» закрыта как недостижимая в архитектуре —
XOR-плато loss ~0.93, см. jepa_analysis.md и docs/archive/).

## Активные цели (Byte-H-JEPA, карта references/byte-h-jepa.md)

- [x] 1. Синтетический пруф разделимости латентов (macro_ab.rs, 16/16)
- [x] 2. Третий канал W_macro в unified_gpu_train (tag=7 MACRO_W)
- [x] 3. Декодер MB3 (z_{t+1} кондиционирует патчевый коридор)
- [ ] 4. Полный прогон с обученным W_macro + честный A/B v7.1 (имена) / v8.1 (структура) / MB3 (обе цели) на name_metric + C-AST errors
- [ ] 5. AST-маска через tree-sitter (будущий узел как цель макро-канала)

## Инфраструктура (сессия 15.10.2026)

- [x] Windows-порт сборки (morris_sandbox cfg + cuda_stubs через cc)
- [x] Полный аудит проекта (audit/*.md, 4 отчёта)
- [x] Рефакторинг: warnings 41→3, кроссплатформенные пути, консолидация доков
- [ ] Обучение на одном Python-файле (agent_loop.py → train_one_py/fuga_py_v1.fuga)
- [ ] A/B чекпоинтов 100K/200K/300K/400K на декодах v2/MB2

## Journal

- journal: 2026-10-15 аудит 4 отчётами, фиксы сборки Windows, рефакторинг, старт обучения на 1 .py
- journal: 2026-07-29T15:35:40+03:00 eval: L0=1.9680 L1=1.9753 L2=0.5120 samples=1000
