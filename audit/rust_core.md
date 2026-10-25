# Аудит Rust-ядра (src/ai, lib.rs, main.rs) — AutoClaw, 15.10.2026
Метод: инвентаризация каталога + целевые grep (импорты, dead_code, cfg). Полный текст: cargo check --lib (0 ошибок, 41 warning после фикса morris_sandbox).

## Инвентаризация src/ai (36 модулей)
- **Байтовый/латентный стандарт (live, главный путь)**: sdr.rs, latent_jepa.rs, htm_temporal.rs (FUGA1 save/load), tm_generate.rs (декодеры v1/v2/MB2/MB3), kan.rs, hybrid.rs (HybridTransition W+KAN), gpu_ops.rs (wgpu-шейдеры: batch_delta/residual/kan), hopfield.rs, byte_lstm.rs, soft_sdr.rs.
- **HybridCore (новая эпоха, v2.1.0)**: vsa_jepa_kan.rs — импортируют hybrid.rs, mod.rs, bin/hybrid_train_v10.rs. Интеграция в unified_gpu_train ещё НЕ сделана (план Фазы 5).
- **Сервис-слой (прод-память omni-web/tgbot)**: core.rs, memory_store.rs, moe.rs, hnsw.rs, answer_engine.rs, router.rs, resonance_attention.rs, codegen.rs, prompts.rs, crystal.rs, sdr_store.rs, jepa.rs, world.rs, anomaly.rs, decoder.rs, personas/.
- **Легаси-токенные (вне байтового стандарта, но ПОДКЛЮЧЕНЫ)**: self_mirror.rs, temporal_predictor.rs, hierarchical_jepa.rs — импортируются из crystal.rs, assembly/legion.rs, assembly/unified_pipeline.rs, bin/l2_ab.rs, lib.rs, main.rs. Простое удаление сломает сборку — нужен отдельный этап с решением пользователя.

## Проблемы
1. `#[allow(dead_code)]` — всего 1 (soft_sdr.rs) — мёртвого кода в ядре почти нет; 41 warning — это unused variable/doc comment (мелочь), `cargo fix` покрывает 12.
2. Легаси-модули token-эпохи всё ещё в дереве зависимостей main.rs/assembly — кандидат на архивацию отдельным решением (не трогать без подтверждения).
3 Windows-совместимость ядра: единственные unix-only места были в src/assembly/morris_sandbox.rs (исправлено 15.10) и src/gpu.rs (#[link cuda], исправлено 15.10).
4. Рассинхрон версии: Cargo.toml 2.0.0 vs README «v2.1.0».

## Рекомендации (по приоритету)
- P1: прогнать `cargo fix --lib` (12 авто-правок) + руками unused variables — 41 warning → ~10.
- P2: решить судьбу legacy-токенных модулей (self_mirror/temporal_predictor/hierarchical_jepa + assembly/{legion,unified_pipeline} + bin/l2_ab) — либо пометить `#[deprecated]`/doc-шапкой LEGACY, либо вынести из сборки фичей.
- P3: консолидация декодеров tm_generate.rs (v1/v2/MB2/MB3 частично дублируют скоринг) — аккуратный рефактор с тестами, средний риск.
