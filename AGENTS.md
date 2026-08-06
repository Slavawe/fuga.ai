# AGENTS.md — состояние и контекст для агентов

## Проект
Собственный «агентский стек» без внешних LLM: `necli/` (локальный каталог/CLI) → OpenAI-совместимый маск `fuga-web` → генерация кода → гейты (compile + relevance) → рекурсивное дообучение на уроках, прошедших гейты.
Двухскоростная генерация: **H-JEPA task-коридор (`eligible`) держит содержание, TM-авторегрессор держит порядок**. Мост в CLI (`tm-gen`) и в маск (`handle_code_generate`).

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
1. **Закоммитить рабочие фиксы** (дерево грязное): temporal_predictor.rs (corr-таргет ×4 + смоук-тест), hierarchical_jepa.rs (порог 1.15 ×2), circuit_breaker.rs (inspect + тесты). Решить судьбу `src/bin/audit_l2.rs` (временный стенд) и `fuga_new_files.zip` (кто-то бросил в корень).
2. ~~Осталные 3 места L2-таргета~~ **СДЕЛАНО (06.08)**: corr-фикс применён во всех 4 путях — `feed_learn` (170-173), `feed_learn_no_tm` (225-230), `feed_learn_ff` (271-276), `feed_learn_hv_only` (409-414). Смоук-тест `corr_target_smoke_all_feed_paths` покрывает все 4 (NaN/Inf + len=3).
3. ~~**Находка 3**: `STRUCTURE_STRIDE=977` (sdr.rs:17) — «мина»~~ **СДЕЛАНО (06.08)**: `structure_shift()` с debug_assert `gcd(SDR_DIM, STRIDE)==1` (+`euclid_gcd`), документация мины на константе, тест `structure_stride_is_coprime_and_mixes`.
4. **Находка 2**: identity-канал есть только в `/tmp/oc_loop.rs`, не перенесён в src/ai/*.
5. **L0-сходимость**: L0 loss ~1.4-1.48 не сходится (стабильно плох), отдельная проблема — breaker L0 нет.
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
- `src/ai/sdr.rs`: `STRUCTURE_STRIDE=977` (17), `structure_shift` + `euclid_gcd` (20-42), `encode_text` (253), тест `structure_stride_is_coprime_and_mixes`.
- `src/ai/tm_generate.rs`: `tm_generate(..., eligible)`, `decode_weighted` hard gate, `MIN_AVG_WEIGHT=6`.
- `necli/src/system_prompt.py` (270).
- Данные: `fuga_stack_tm.bin` (394MB), `fuga_hjepa.bin`, `agent_lessons.jsonl` (1 урок factorial), `corpus*.jsonl`, `/tmp/cs.jsonl`.

## Конвенции
- Недетерминированные прогоны: L1 сходится 1.20→0.88 без breaker, L0 стабильно ~1.4 — не считать это баг breaker, это масштаб метрики.
- Аудит-стенды (как audit_l2.rs) — временные, обычно не коммитятся, но при передаче фиксов их стоит сохранять для воспроизводимости.