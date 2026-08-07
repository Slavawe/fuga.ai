# AGENTS.md — состояние и контекст для агентов

## Проект
Собственный «агентский стек» без внешних LLM: `necli/` (локальный каталог/CLI) → OpenAI-совместимый маск `fuga-web` → генерация кода → гейты (compile + relevance) → рекурсивное дообучение на уроках, прошедших гейты.
Двухскоростная генерация: **H-JEPA task-коридор (`eligible`) держит содержание, TM-авторегрессор держит порядок**. Мост в CLI (`tm-gen`) и в маск (`handle_code_generate`).
**Непрерывный (tokenless) декод**: `tm_generate_latent` (tm_generate.rs) — ранжирует кандидатов по cosine в латентном пространстве (predict_latent → LatentVector), словарь остаётся только gate/коридор. Подключён в `omni-web.rs:486` (`handle_code_generate`). Старый `tm_generate`/`decode_weighted` (веса по битам) остаётся для CLI `tm-gen`.
**Byte-level (ByT5/MegaByte) путь (07.08)**: отказ от токенного словаря как зависимости. Фиксированный алфавит **256 сырых UTF-8 байтов** (`byte_basis`, sdr.rs), позиционная свертка `encode_bytes_sdr`, байтовые переходы TM (`learn_bytes`/`predict_bytes_latent`, htm_temporal.rs) и непрерывный байт-декод `tm_generate_latent_bytes` (tm_generate.rs) — ранжирует 256 байт-кандидатов по cosine в латентном, гейт = LATENT_MIN_COSINE + коридор байт. Никакого словаря — любой язык и код, опечаткоустойчивость (один байт смещает малую часть свертки). Тесты: `byte_basis_is_fixed_and_position_sensitive`, `tm_generate_latent_bytes_reproduces_text_without_dictionary`. lib-набор теперь **123/123**.
   **Проверено на деле (стенд `src/bin/byte_gen_test.rs`)** — реальный Rust-корпус corpus_doc_code_pairs.jsonl, сырые UTF-8 байты:
   - 600 сниппетов: 108K байт-шагов, max byte-cosine 0.337 (116/256 за порогом), выход 3 байта.
   - 3000 сниппетов: 665K байт-шагов (969 b/s), **max byte-cosine 0.543**, 54/256 различимых, декод 1460 b/s, выход 200 байт.
   - **ЧЕСТНОЕ ОГРАНИЧЕНИЕ**: инфраструктура работает и W сходится (cosine растёт острее), но выдача — повторяющийся локальный мусор (`ele ocrane)e}e...`): наивный вывод по одному байту из 256 без ГЛОБАЛЬНОГО уровня деградирует в местные биграммы. Для осмысленного кода нужна two-speed как в MegaByte (глобальные патчи → байты внизу) — это задача `A` (следующая итерация). Итерация `B` (данная): инфраструктура подтверждена, ограничение задокументировано.

## Команды
- Сборка: `cargo build --release --bin fuga-web` (или `--bin fuga`)
- Тесты circuit_breaker: `cargo test --release --package fuga --lib circuit_breaker`
- Прогон аудита L2: `./target/release/audit_l2 agent_lessons.jsonl /tmp/cs.jsonl`

## Архитектура JEPA/breaker (после сессии калибровки)
- Контексты `DEFAULT_L0/L1/L2_CTX = 4/3/2`, strides 1/3/5 (hierarchical_jepa.rs:9-14).
- `learn()` (hierarchical_jepa.rs:856) шлёт `actual[li]` сырым таргетом в градиент (`delta = baseline.bind(actual)`, в `train_step` :307).
- **Circuit breaker в `inspect()`** (circuit_breaker.rs:24): `warning_boundary = max_allowed * 0.75`; loss < warning → Nominal, ≤ max_allowed → Warning (lr*=0.5), > max → Critical (reset rand±0.005). Критический порог прод-мест = **1.15** (был 0.57 — «убивал» здоровый L2 мгновенным reset'ом).
- **L2-таргет = `phase_smooth(ls_bind(l1_pred, actual), 2)`** — коррекция ошибки L1 (AUDIT-CALIB), согласована с predict_side (`predict_refined`, err_traj). Замена вместо старого `l0_pred`. Применён во **всех 4** feed_learn-путях (см. ниже).

## Калибровочные данные (7.4K строк, `/tmp/cs.jsonl`)
- Старый таргет `l0_pred` без breaker: L2 loss пик 1.33-1.38 (анти-обучение).
- Новый таргет `corr(l1,actual)` без breaker: L2 0.78-1.03, EMA ~0.93 (здоровый, как L1).
- L0 без breaker: ~1.40-1.48 (нормальный масштаб метрики).
- Итог с breaker 1.15: L2 EMA ~0.93, resets 9/7466 = **0.12%** (практически здоров).
- Повторный прогон (06.08, без breaker, feed_learn corr): L2 EMA ~0.91 стабильно, resets 12/7466 = 0.16% — тот же порядок, разброс из-за недетерминизма (TM random).

## Незавершённые задачи (next steps)
7. **Токенный словарь хардкодит пути (07.08, диагностировано на полном обучении)**: `build_token_vocab_from_files` (self_mirror.rs:710) использует `allowed_dirs = ["/home/slava/fuga/src/", "/home/slava/neural-engine/"]` — словарь строится ТОЛЬКО из этих путей, а не из индексированных файлов корпуса. При `generate-code "fn main" --tokens` на обученном корпусе: `Token vocab: 0 entries` → паника в htm_temporal.rs:249 (index out of bounds: len 1096, index 3699), EXIT=101. Фикс: строить словарь из `self.nodes[].path` (индексированные файлы) без фильтра allowed_dirs, либо принимать каталог-аргумент. PhaseNode-путь (`generate-code --gen` без --tokens) работает нормально.
8. **Полное обучение 2.1 (07.08)**: конвейер на corpus_doc_code_pairs.jsonl (7218 .rs) + corpus.jsonl (7 книг: Newton, Descartes, Euler): mirror-index 7218 файлов → train-predictor 20 эпох (avg_loss 1.0953, fuga_mirror_{jepa,tm,nodes}.bin) → crystal-learn-dir код (9346 чанков, 137299→146645) → тексты (16763 чанка, →163408 записей, fuga_code_crystal_full.bin 228MB). Верификация: 121/121 lib-тестов; crystal-query длинным запросом RESONANCE=0.382 (L1 route #9, `handle.0`); короткие запросы (2 токена) дают deterministic silence — порог 0.25 рассчитан на чанки по 30 слов (свойство VSA-блиндинга, не баг). Скрипт `full_train.sh` в корне (временный), логи в `train_logs/`, чекпоинты в `train_checkpoints/` (30GB — включая 21GB fuga_code_cube_mem.bin, копировать аккуратно).
1. ~~**Закоммитить рабочие фиксы**~~ **СДЕЛАНО (06.08, коммит `7d4c4eb`)**: temporal_predictor.rs (corr ×4 + смоук), hierarchical_jepa.rs (1.15 ×2), circuit_breaker.rs (inspect+тесты), sdr.rs (stride-гвард), audit_l2.rs, AGENTS.md. В корне остался `fuga_new_files.zip` (неизвестного происхождения, не код — решить судьбу).
2. ~~Осталные 3 места L2-таргета~~ **СДЕЛАНО (06.08)**: corr-фикс применён во всех 4 путях — `feed_learn` (170-173), `feed_learn_no_tm` (225-230), `feed_learn_ff` (271-276), `feed_learn_hv_only` (409-414). Смоук-тест `corr_target_smoke_all_feed_paths` покрывает все 4 (NaN/Inf + len=3).
3. ~~**Находка 3**: `STRUCTURE_STRIDE=977` (sdr.rs:17) — «мина»~~ **СДЕЛАНО (06.08)**: `structure_shift()` с debug_assert `gcd(SDR_DIM, STRIDE)==1` (+`euclid_gcd`), документация мины на константе, тест `structure_stride_is_coprime_and_mixes`.
4. ~~**Находка 2**: identity-канал есть только в `/tmp/oc_loop.rs`~~ **БЛОКИРОВАНО, ПОИСК ПРОВЕДЁН (06.08)**: `/tmp/oc_loop.rs` не находится — bash/zsh history, /tmp, /home/slava, история сессий: 0 совпадений (только упоминание о существовании файла в отчёте прошлой сессии, кода нет). Воссоздать из истории команд нельзя; нужно реализовывать заново. По сути identity/task-канал живёт в CLI: `src/main.rs:1767` (corridor→tm_generate) и `src/main.rs:6811` (task-mask + H-JEPA trajectory), не вынесен в src/ai/*.
5. ~~**L0-сходимость**~~ **ДИАГНОЗ ЗАКРЫТ (06.08, стенд `l0_diag` v2-v4, 3 прогона)**: «L0 не сходится» — **ложная тревога, метрика врёт**. loss train_step(mode 2) = `1 - cosine(raw, actual)`, а raw аппроксимирует δ=baseline⊗actual; для разреженного bipolar-таргета (2% единиц) cosine(δ,actual)≈Σbaseline/dim≈-0.96 → loss стабильно ~1.45-1.49 даже при отличной модели. Реальные качества: **L0 pred~actual = +0.87** (отлично), **L1 pred~own_target = +0.46** (умеренно, растёт), L2 = 0.93. Все 3 уровня учатся; breaker 1.15 корректен (L2-метрика честная). Не менять loss без смены всей калибровки. Стенд: `src/bin/l0_diag.rs` (не закоммичен).
   ⚠ **РЕВЬЮ-ПЕТЛЯ РАЗОРВАНА (06.08)**: диагноз Задачи 5 строился на пост-фиксном состоянии (corr уже во всех путях). Проведено честное A/B (`src/bin/l2_ab.rs`): один код, один корпус, 7466 шагов, старо=`l0_pred` vs нов=`corr(l1,actual)`, остальное идентично:
   - **OLD (l0_pred)**: L2 loss=1.0000, **resets=7466/7466 (100%)** — L2 не обучаем, breaker сбрасывает на каждом шаге (анти-обучение подтверждено).
   - **NEW (corr)**: L2=0.9232, resets=18/7466 (0.24%) — L2 оживает и сходится.
   - Задачи 2 и 5 независимо подтверждены: Задача 2 этим A/B; Задача 5 (метрика L0 врёт) инвариантна таргету — simL0≈0.87 в обоих режимах при loss 1.48-1.49.
6. **Corridor мост** (уточнено 06.08): в `necli/src/system_prompt.py:270` corridor **отсутствует** — там блоки environment/skills/MCP/memory, слово corridor в файле не встречается. Сам мост живёт в `src/main.rs:1767` и `src/ai/tm_generate.rs` (eligible). Замер размера промпта заблокирован (не запускался); проверка «~9K симв./690 токенов мусора» требует аудита на живой системе.

## GIT, релевантные коммиты
- `064b4fc` fix(mask): `find_lesson` считает **уникальные** совпадения (не повторные — «via»×2 крал factorial-урок для reverse_string). Порог `hits>=2 && ratio>=0.5`.
- `d9db37e` feat(mask): two-speed bridge в `handle_code_generate`.
- `87d8657` feat(agent): two-speed bridge в probe — lessons vocab + task corridor.

## Ключевые файлы
- `src/ai/temporal_predictor.rs`: `feed_learn` (128, corr-таргет 170-173), `feed_learn_no_tm` (180, corr 225-230), `feed_learn_ff` (226, corr 271-276), `feed_learn_hv_only` (376, corr 409-414); смоук-тест `corr_target_smoke_all_feed_paths` в конце файла.
- `src/ai/hierarchical_jepa.rs`: `learn()` (856), `train_on_directory` (1028), `predict_refined` (783), контексты (9-14), `delta = baseline.bind(actual)` в `train_step` (307).
- `src/safety/circuit_breaker.rs`: `inspect` (24), тесты.
- `src/vsa/topology.rs`: `ls_bind` (3), `phase_smooth` (87).
- `src/bin/audit_l2.rs` (не закоммичен, временный): EMA+гистограмма+счётчик resets.
- `src/bin/omni-web.rs`: `find_lesson` (324), мост C2 (`handle_code_generate` 386), `meaningful_tokens` (306).
- `src/ai/sdr.rs`: `STRUCTURE_STRIDE=977` (17), `structure_shift` + `euclid_gcd` (20-42), `encode_text` (253), `byte_basis`/`encode_bytes_sdr` (байтовый алфавит, ~426+), тест `structure_stride_is_coprime_and_mixes`, `byte_basis_is_fixed_and_position_sensitive`.
- `src/ai/tm_generate.rs`: `tm_generate` (весовой, CLI), `tm_generate_latent` (непрерывный через латент, прод-мост), `tm_generate_latent_bytes` (byte-level декол по 256 байтам, без словаря), `decode_weighted` hard gate, `MIN_AVG_WEIGHT=6`, `LATENT_MIN_COSINE=0.05`; тесты `tm_generate_latent_uses_continuous_decode_and_respects_gate`, `tm_generate_latent_bytes_reproduces_text_without_dictionary`, `byte_basis_is_fixed_and_position_sensitive`.
- `necli/src/system_prompt.py` (270).
- Данные: `fuga_stack_tm.bin` (394MB), `fuga_hjepa.bin`, `agent_lessons.jsonl` (1 урок factorial), `corpus*.jsonl`, `/tmp/cs.jsonl`.

## Конвенции
- Недетерминированные прогоны: L1 сходится 1.20→0.88 без breaker, L0 стабильно ~1.4 — не считать это баг breaker, это масштаб метрики.
- Аудит-стенды (как audit_l2.rs) — временные, обычно не коммитятся, но при передаче фиксов их стоит сохранять для воспроизводимости.
- **Известный флаки (06.08)**: `latent_jepa::tests::encoder_is_deterministic_and_512_dimensional` падает ~1/3 полных lib-прогонов (в параллели, гонка на TOKEN_SDR_CACHE/LATENT_ENC_CACHE), изолированно — стабильно ok. Pre-existing, к правкам сессии отношения не имеет.