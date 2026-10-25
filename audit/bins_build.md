# Аудит бинарей и сборки fuga — 15.10.2026 (составлено AutoClaw после фиксов)
Метод: инвентаризация src/bin (42 файла), Cargo.toml/lock, build.rs, tests/, cpp/, целевые grep. Замеры: cargo check --lib, cargo build --release (2 бинаря), см. журнал AGENTS.md.

## Категории бинарей (42)
- **MAIN (главный путь обучения/валидации)**: unified_gpu_train (49KB, трёхканальный CPU/GPU), hybrid_train, hybrid_train_v10 (HybridCore v10), talk_model, v5_validate, v6_validate, unified_e2e, unified_roundtrip_cpp, macro_ab, macro_validate, name_metric.py, w_diag, dbg_cos, checkpoint_check, combined_decode, byte_baseline, train_byte_w, data_prep, kan_calib, full_byte_train.
- **PROD**: omni-web (64.6KB, маск), tgbot (38KB), fuga (src/main.rs CLI).
- **RESEARCH/A-B (разовые стенды, живы)**: gpu_bench, two_speed_test, rec_test, byte_gen_test, cosine_gate_test, ab_window, l0_diag, l2_ab, reallatent_test, audit_l2, ast_mem_probe, megabyte_check, txt_search_debug, dialogue_test, ingest-knowledge, cube-viz, fuga-wave-broadcast/recv.
- **LEGACY**: gpu_train (Hebb-остаток — подтверждено AGENTS: не использовать для качества), unified_train (ранний CPU-путь).

## Проблемы
1. **Захардкоженные Linux-пути (/tmp, /home/slava) в 9 бинарях**: unified_gpu_train, unified_train, hybrid_train, hybrid_train_v10, w_diag, gpu_train, unified_e2e, unified_roundtrip_cpp, omni-web. На Windows дефолты (`--out /tmp/*.fuga`, дефолтный корпус `corpus_doc_code_pairs.jsonl`) не пишутся/не находятся. Лечится передачей флагов; правильный фикс — относительные дефолты/`std::env::temp_dir()`.
2. **Линковка cuda.lib** (была): все бины падали LNK1181 на Windows — исправлено 15.10 (cfg_attr + cuda_stubs.c через cc).
3. **Дефолтный корпус не существует** в воркспейсе ни в одном бине — все MAIN-прогоны требуют явных --jsonl (см. обучение на одном .py).
4. **tests/**: 6 .rs (hierarchical_jepa_test, jepa_test, moe_routing_test, test_anomaly_detection, test_self_assembly, wave_mesh_test) + test_smoke.py. Полный baseline: см. cargo_test_baseline.log (запущен 15.10).
5. **cpp/**: fuga_core.h (27KB), decode.cpp, train.cpp, hjepa.h, Makefile + ЛИНУКС-бинарь `unify_check` (26.6KB, скомпилирован ранее) в дереве. На Windows make нет — C++-ядро не собираем (не блокирует Rust-путь); `unify_check` — артефакт, кандидат в .gitignore/очистку.
6. **examples/**: только mem2crystal.rs (README_v9 ссылается на несуществующий minimal_hybrid_train.rs). tools/: build_pairs.py, build_pair_dataset.rs, to_jsonl.py (/tmp-путь).

## Рекомендации (по приоритету)
- P1: заменить Linux-дефолты путей на кроссплатформенные (std::env::temp_dir()/относительные) — 9 бинарей, механическая правка.
- P1: таблица «бин → обязательные аргументы» в AGENTS.md (устранить класс «дефолт не существует»).
- P2: cpp/unify_check — убрать из дерева (артефакт), добавить правило в .gitignore.
- P2: пометить LEGACY-бины шапкой в файле (1 строка), без удаления.
