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
   **Задача A — two-speed декодер (07.08)**: глобальный патчевый transition-оператор `patch_predictor` в TemporalMemory (`learn_patch`/`predict_patch_latent`, htm_temporal.rs) + декодер `tm_generate_two_speed` (tm_generate.rs): глобальный уровень предсказывает ЦЕЛЫЙ байт-патч за шаг по cosine к патч-vocab, локальный байтовый W остаётся точностью внутри. Реально проверено (стенд `src/bin/two_speed_test.rs`, 800 .rs сниппетов, patch=4): naive byte даёт 2 байта, **two-speed даёт 52 байта (×26)** и узнаваемые морфемы (`dendrite`, `ent`) — локальный мусор преодолён. **КАЛИБРОВКА (07.08, honest A/B)** выявила: (a) порог LATENT_MIN_COSINE 0.05→0.25 НЕ влияет — топ-1 патч почти всегда ≥0.25, привод invariant; (b) vocab 4000→512 усиливает W_patch (max-cosine 0.316→0.548); (c) реальная проблема — топ-1 аргмакс зацикливается на частотных би-паттернах (`er er er`), анти-повтора идентичности недостаточно (цикл период-2).
   **Задача 1 — энтропийный патчер BLT (07.08)**: `tm_generate_two_speed_entropy` (tm_generate.rs) — динамические границы патча по предсказуемости вместо фиксированного размера: локальный байтовый W даёт cosine-распределение по 256 байтам, **gap топ-1−топ-2** решает скорость (большой gap = предсказуемый участок → локальная эмиссия, патч растёт; малый gap = сложный участок → глобальный W выбирает патч). Не raw-softmax-энтропия: она сатурирует ~1.0 (256 кандидатов с cosine-базлайном ~0.12, измерено 0.999 даже на чистом a→b), gap устойчив. A/B на том же 800 .rs (vocab=512/patch=2): fixed=50 байт, **entropy=200 байт (полный бюджет, 891 B/s)** — BLT-механика подтверждена; контент всё ещё `er er...` — предел не декодер, а качество локального байтового W (частотная пара e→r выучена увереннее всего).
   **Задача 2 — speculative (draft→verify, 07.08)**: `tm_generate_speculative` (tm_generate.rs) — глобальный W набрасывает патч-черновик, локальный байтовый W верифицирует каждый байт (перезаписывает если gap порога подтверждает другой). Реальный A/B (тот же 800 .rs): **10 байт, 11 B/s** — НЕ выигрывает против entropy (200б, 891 B/s). Честный отрицательный результат: в нашей последовательной VSA-схеме verify не бесплатен (256-кандидатный per-byte верификатор дороже преимущества draft); output ` in se` (похоже на код, не `er er` мусор), но мал и медлен. entropy/BLT остаётся лучшим декодером. lib-набор **126/126**.
   **Задача 3 — beam search декодер (07.08)**: `tm_generate_beam` (tm_generate.rs) — классический beam-N (beam=3, top_m=5) держит K гипотез с накопленным log-score вместо greedy top-1, призван лечить автогрессионные зацикливания. Реальный A/B (800 .rs): **13 байт, 491 B/s** — это seed-эхо (12 б) + 1 байт: при softmax-uniform по 256 байтам все log-prob малые, луч падает после первого шага, контента почти не добавляет. Не конкурирует с entropy (200б/874B/s). Ряд из 4 декодеров подтверждает: **предел генерации — не декодер, а качество локального байтового W** (частотная биграмма e→r гасит все декодеры). entropy/BLT остаётся лучшим. lib-набор **127/127**.
   **Итерация 2 векторы (07.08)**:
   (a) Контекстная глубина W: окно 4→8 (both learn_bytes+learn_patch, 5-й арг стенда) — **честный отрицательный результат**: max-cosine 0.547→0.483, кандид 277→235, p/s 340→291. Больше окно разбавляет W_patch, а не даёт структурные зависимости (learn_patch складывает окно-патчи, разрежение контекста влияние). Окно 4 остаётся оптимумом.
   (b) Acceptance rate (задача 3 spec): `tm_generate_speculative_stats` — счётчик подтверждённых draft-байт. На реальном корпусе **acceptance=1.000** (bytes=10): локальный verifier не уверен ни на одном байте (gap<0.60 везде), draft никогда не перезаписывается — телеметрия не помогает, но диагностирует, почему speculative даёт 7-10б: локальный W не уверен → verifier бездействует → стоп на seed.
   (c) Gap-порог BLT (задача 2, sweep на одном обучении window=4, 800 .rs): **0.30→6б/72B/s, 0.45→6б/83B/s, 0.60→200б/1004B/s, 0.75→200б/1004B/s**. Точка перегиба: **между 0.45 и 0.60**. Ниже порога каждый шаг мнимо-предсказуем → локальная эмиссия застревает на самоповторе (6б); выше — сложные байты уходят глобальному W-патчу и движение продолжается (200б/1004B/s). **Оптимальный gap ≈ 0.55–0.65** (чуть выше cosine-базлайна ~0.12). Параметры `gap_thresh` (6-й арг стенда) и `ctx_window` (5-й) введены.
   Тесты: **127/127** (no new test, stats-функция без нового теста).
   **Итерация 3 — SSM-lite рекуррентная память h(t) (07.08)**: `LatentPredictor::predict_next_rnn`/`learn_transition_rnn`/`advance_h` + `learn_bytes_rnn` (htm_temporal) + декодер `tm_generate_recurrent` (tm_generate). Скрытое состояние h (unit latent) — Mamba-стиль leaky-интеграция `h'=φh+(1-φ)enc(byte)`, смешивается с локальным окном ``local ⊕ mix·h`` до W. **Честный A/B** (стенд `src/bin/rec_test.rs`, 800 .rs, единственный параметр — stateful vs stateless обучение W): stateless-декод даёт **3-6 байт** (`yete`, `y  `), **stateful-trained W при mix=0 — 17 байт** (`yy_rnesl utiel`, морфемы) — подтверждает диагноз: state в обучении даёт W способность помнить >окна, ломает чистый локальный аттрактор. Но **mix>0 на декоде даёт меньше (5-7)** — train/spec exposure-разрыв (h в декоде строится от self-generated, а не от реального корпуса). Рекуррентная память — реальный кандидат на удержание длинного контекста, требует teacher-forcing loopback (итерация 3.1). lib **128/128** (+recurrent).
   **Итерация 3.1 — exposure-bias закрытие (07.08, честный отрицательный)**: два рычага из плана:
   (a) **Scheduled Sampling** (rec_test ε-арг, ε=0.15: с вероятностью ε в advance_h подмешивается предсказанный байт вместо честного) — **НЕ помог**: stateful mix=0 дал **4 б** против **17 б** при ε=0. Причина: предсказанный argmax почти всегда = частотный e/r, поэтому дрейф-шум в h УСИЛИВАЕТ аттрактор, а не учит устойчивость (в LLM SS работает из-за редкости draft-ошибок; здесь 15% шума в каждом окне размазывает W).
   (b) **φ(gap) адаптация** (tm_generate_recurrent: при малом gap уверенности φ→0.05, «забыть» зашумленное h) — внедрена, не изменила итог (выходы 3-4 б).
   Итог: ε=0 остаётся оптимумом; учить устойчивость к дрейфу надо НА СГЕНЕРИРОВАННОМ выходе (teacher-forcing loopback декод-гипотезами), а не подмешиванием шума в h. lib **128/128**.
   **Итерация 3.2 — Non-argmax State Advance (07.08, честный отрицательный)**: `tm_generate_recurrent_nucleus` (tm_generate.rs) — эмиссия остаётся argmax, но в advance_h идёт Nucleus-сэмплированный байт (softmax(cos/T), T=1.2, top_p=0.9), чтобы h не заполнялся доминантным e/r. A/B (800 .rs, stateful W ε=0, 2 rng-сид): **mix=0 → идентичные 17 б** (nucleus при mix=0 неактивен), **mix>0 → 3-6 б** (как argmax). **Гипотеза «e-заполнение h — причина» НЕ подтвердилась изолированно**: разнообразив h, декодер на mix>0 по-прежнему останавливается. Причина глубже: W не обучен ИЗВЛЕКАТЬ пользу из state при смешивании (train: h от честных байт корпуса, декод: h от self-эмиссий — дрейф на уровне распределения, не байт). Остаются: Hopfield read (пункт 4) и self-correction loopback (обучение на сгенерированных). lib **128/128**.
   **Итерация 4 — Modern Hopfield, аналог FSQ/VQ состояния (07.08, честный отрицательный)**: `src/ai/hopfield.rs` (HopfieldMemory: `h_clean = M_vals·softmax(β·M_keysᵀ·h)`, банк Rust-структурных шаблонов `fn`/`pub`/`{`/`}`…) + декодер `tm_generate_hop_reader`. Ассоциативный read притягивает дрейфующий h к ближайшей структурной ячейке — по механике это мягкая версия пункта 1 новой карты (FSQ/VQ). **Честный A/B** (800 .rs, stateful W ε=0): **mix=0 → 17 б (идентично raw), mix>0 → 3-4 б** — очистка состояния не устраняет остановку. Тест hopfield (притяжение при дрейфе) PASS, lib **130/130** (+hop_reader test, +hopfield drift test) и **консенсус 4 рычагов** (scheduled-sampling, φ-gate, nucleus, hopfield): все дают mix=0 → 17б, mix>0 → 3-6б. Предел — не h, а сам локальный байтовый W (линейный ландшафт не разделяет e→r vs структурные аттракторы). Указывает на пункт 5 карты (KAN: сплайновые φ(x) вместо линейной W·x) — единственный рычаг, меняющий сам оператор.
   **Итерация 5 — KAN-lite сплайновый оператор (07.08, смешанный результат)**: `src/ai/kan.rs` (KanTransition: per-edge B-spline φ_{o,i}(x)=Σ_k c·B_k(x), K=6 узлов на [-1,1], Widrow-Hoff на узлах) + декодер `tm_generate_kan`/`learn_byte_kan` (tm_generate.rs). **Синтетический пруф PASS**: два линейно-неразделимых аттрактора (x_B=−x_A, линейный W обязан зеркалить) — KAN разделяет оба (A→α, B→β, B-обучение не стирает A). **Реальный A/B (800 .rs): 1 байт `}` за 431s** — Widrow-Hoff на сплайнах НЕ сошёлся на плотных латентах: cap_outputs (NORM_CAP=4 на строку 3072 коэфф, каждые 50 обновлений) сжимает обучение; lr=0.05/stride=4 малы. Диагноз: оператор архитектурно способен (пруф), требуется калибровка (мягкий per-node cap, выше lr, без stride). lib **131/131** (+kan_splits test).
   **Итерация 6 — внешний сверстник-эталон: байтовый LSTM (07.08)**: `src/ai/byte_lstm.rs` (vanilla 1-слойный LSTM, H=128 ≈ **230K params**, тот же 256-байтный алфавит, BPTT-8, Adam, детерминированный splitmix64, без внешних зависимостей) + стенд `src/bin/byte_baseline.rs` (тот же корпус, тот же seed `fn `, та же метрика "bytes-until-stall"). **Почему это честный сверстник**: у него НЕТ никакого VSA — чистая классическая рекуррентность, тот же порядок параметров, как у нашего байтового стека, поэтому любой разрыв = оператор, не словарь. **Результаты**: синтетика (ab-чередование) — LSTM выучивает период-2 и удерживает его (BPTT корректен, PASS). Малый слайс 40 (18K байт): **LSTM 200 б** (полный бюджет!) vs naive 6 б — классическая рекуррентность держит цикл; но контент — автоповтор `histogram histogram…` (то же притяжение, длиннее период). Полный 800-слайс (382K байт): **LSTM 1 б** (CE 1.617) vs fuga stateful 17 б / entropy BLT 200 б. **Честный вывод**: предел генерации — не VSA, а общий автогрессионный аттрактор маломасштабной байтовой генерации; классический рекуррентный сверстник в том же масштабе не обходит fuga (и уступает entropy BLT на полном корпусе). Публикуемый контраст-стенд. lib **132/132** (+lstm test).
   **Итерация 7 — соединение всех технологий + проверка на сохранённых чекпоинтах (07.08)**:
   (a) **Проверка формата** (`src/bin/checkpoint_check.rs`): load читает все секции нового формата (cells+window+W+OWM-P+W_patch), обратная совместимость по остаточной длине. `fuga_stack_tm.bin`: 35397 клеток, **W_updates=224478 (W обучен!)** — но это ТОКЕННЫЙ корпус-стек (encode_text), НЕ байтовый. `fuga_mirror_tm{,_500,_1000,_2500}`.bin (полное 2.1): W и W_patch **тождественные** (updates=0, trained=false) — обучение 2.1 сохраняло только клетки TM, байтовый W не тронут.
   (b) **Объединённый конвейер** (`src/bin/combined_decode.rs`): один процесс — загрузка чекпоинта → патч-словарь из того же корпуса → обучение KAN + LSTM на том же корпусе → Hopfield-банк по encoder → прогон **10 декодеров на одинаковом seed** ("fn main() {"): naive W, two-speed, entropy BLT, recurrent h(t) (mix 0/0.4), recurrent+nucleus, Hopfield-read, beam-3, KAN, LSTM-peer. На обученном токенном чекпоинте: entropy BLT 200 б / recurrent 200 б / naive 161 б, но контент — бинарный шум (W токенный, байтовое не совпадает); beam=12 (seed-эхо), KAN=1 б, LSTM-peer=3 б. **Честный вывод**: конвейер технически соединяет все технологии, но **обученного байтового W в чекпоинтах нет** — он тренировался сессионно (800-слайс на лету, не сохранялся); нужен этап, который сохраняет обучение байтового W в чекпоинт. lib **132/132**.
   **Итерация 8 — save_byte_w/load_byte_w + прямое обучение W (07.08)**: восполнено недостающее звено из итерации 7:
   (a) **API** (htm_temporal.rs): `save_byte_w(path)`/`load_byte_w(path)`/`apply_byte_w(w)` — sidecar-формат magic "FBW1" + u32 len + f32[LATENT_DIM²=262144], дефензивный load (bad magic/length → None, никогда не паникует). Тест `byte_w_sidecar_roundtrip` — бит-в-бит roundtrip + порча magic → None, lib **133/133**.
   (b) **Стенд `src/bin/train_byte_w.rs`**: обучает байтовый W НАПРЯМУЮ через `LatentPredictor::learn_transition` (окно байт → байт), НЕ через learn_bytes (который растет клетки TM и медленный) — 800 .rs / 380980 байт-шагов / 224с, W_updates=380K, non-trivially trained=true, roundtrip OK → `fuga_byte_w_800.bin` (1MB).
   (c) **combined_decode** получил 4-й арг (sidecar path): `apply_byte_w` перед прогоном 10 декодеров. **Честный обзор после присоединения**: naive byte W даёт **2-3 б**, recurrent mix=0.4 **17 б**, entropy BLT **200 б** (полный бюджет), KAN 1 б, LSTM-peer 3 б. Разница с итерацией 7 (161 б токенного шума): теперь оператор честный байтовый, и видно реальное обучение на 800-слайсе — короткие runs до локального аттрактора. Это подтверждает: персистентность W работает, но качество ограничено обучением (800 .rs), не форматом. lib **133/133**.
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
7. ~~**Токенный словарь хардкодит пути**~~ **ИСПРАВЛЕНО (08.08)**: `build_token_vocab_from_files` (self_mirror.rs:710) больше не использует `allowed_dirs` ограничение, токенизирует любые индексированные исходники и имеет фолбэк на `node.name` при отсутствии файлов на диске.
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
## Сессия 08.08 (вечер): баг-чек + двухсторонний CPU/GPU конвейер
### Баг-чек (приоритет №1, найдено и исправлено)
- **БАГ A — hopfield/kan выпали из сборки** (08.08, при восстановлении из git после случайного удаления): `src/ai/mod.rs` потерял `pub mod hopfield;`/`pub mod kan;` и `pub use kan::*` — тесты молча 128 вместо 133, функции недоступны. Исправлено: модули восстановлены, lib-тесты 128→130 (в AGENTS.md ранее было 133 — часть тестов не в текущем дереве, документированных ключевых НЕТ пропажи: byte_w_sidecar_roundtrip, byte_basis, bytes_reproduces проходят).
- **Баг B (OOM-класс)**: 10M-прогон full_byte_train убит OOM killer (kernel log, 7.5GB RAM). Две причины класса «неограниченный рост в памяти»:
  (1) `dirs.push(LatentVector::zero())` каждые 64 шага без лимита;
  (2) воспроизведено в gpu_train: **неограниченный mpsc::channel** — CPU записал 1.5M пар × 4KB = **6GB в память**, GPU потом переваривал (CPU%=0, RAM free 862MB). Фикс: `mpsc::sync_channel(batch*4)` = backpressure/двойная буферизация, RAM free после прогона 4M пар = 1921MB. Урок: ВСЕ конвейеры — только ограниченные каналы.

### Двухсторонний конвейер CPU/GPU (`src/bin/gpu_train.rs`)
- Разделение: **CPU-поток** читает JSONL → structure-fold окна → SdrEncoder (x=encode(window), err≈target, W=0 init) → шлёт пары в ограниченный канал; **GPU** применяет Widrow-Hoff пачками `batch_delta` (W живёт в VRAM) + **cap-стадия** (новый WGSL шейдер `cap_w`, ROW_NORM_CAP²=4, каждые 50 пачек — как CAP_EVERY в Rust).
- `GpuOps` расширен: `cap_pipeline`/`cap_buf`/`cap_w(cap_sq)` — без cap веса взрывались: 1M пар → max|W|=101; с cap 4M пар → max=0.95, stdev=0.149.
- **Честный A/B** (один корпус fuga_unified_train.jsonl, одинаковые пары):
  - CPU-only (--no-gpu): **10 240 пар/с**
  - GPU-конвейер: **16 869 пар/с** (4M пар, 237s) — **×1.65**
  - CPU загрузка в обоих режимах ≈145% (1.4 ядра из 8) — bottleneck остаётся
    SdrEncoder.encode (164 set-бита × 512), GPU util ~30-35% (дельта почти бесплатна).
  - «Снятие нагрузки с CPU 10-70%»: в текущем виде GPU забирает только Widrow-Hoff x2
    (32K флоп/шаг из ~84K×2 ответов), CPU остаётся основным. Реальное снятие —
    когда на GPU уйдёт encode (первый шаг к этому — gpu_ops уже покрывает delta).
- Скорость конвейера стабильна: 12.5-16.9K пар/с (batch=512).
- W сохраняется FBW1 (bin-совместим с Rust save_byte_w).

### Файлы сессии
- `src/bin/gpu_train.rs` (новый) — двухсторонний конвейер
- `src/ai/gpu_ops.rs` — +cap-шейдер/метод; `src/ai/mod.rs` — hopfield/kan восстановлены
- `cpp/*` — C++ ядро (бета-тест C++, выложено на GitHub в fuga.ai, description=«бета тест C++»)
- `py/train_cpp.py` — Python-оркестратор (train/vocab/decode)

## Сессия 08.08 (ночь): настоящая проверка обучения + C++ beam/H-JEPA/OWM
### Настоящая проверка обучения (диалоговый e2e, `src/bin/dialogue_test.rs`)
- Стенд грузит реальные кубы (peek_cube_header → динамический N,S), прогоняет
  omni.ai.answer() (FugaAI::answer, core.rs:425: think → route → memory.search + 
  retrieve_context + text-fallback) на 6 запросах (привет/вопрос/код/разговор).
- Результаты (ЧЕСТНО):
  - omni_cube.bin (N3S4, память 3932 записи — физика/propulsion): retrieval РАБОТАЕТ,
    все ответы содержательные (sim=0.51-0.52), НО релевантность НЕ РАЗЛИЧАЕТ ТЕМЫ:
    «привет» → Mach Effect Thruster; «сортировка» → Quantum Field Theory; корпус
    однородный (вся физика) → всё резонирует ~равно.
  - omni_cube_idf.bin (N4S8, 604 869 записей, 1.2GB — redis/etcd/gin/jemalloc):
    sim поднялся до 0.74-0.75, но ответы — СЫРЫЕ дампы исходников, нерелевантные
    («временная память» → mutex.h 0.751, «сортировка» → gin/context.go).
  - **ГЛАВНЫЙ ДИАГНОЗ**: инфраструктура (VSA-роутинг + retrieval + домены) жива,
    но релевантность ранжирования низкая — VSA-эмбеддинг кода схлопывается
    (весь код похож, sim-полотность 0.52+), верх корпуса забит одинаковыми
    файлами. Нужно (next): домен-first классификатор + IDF/лексика-буст +
    порог отсечения очень высоких шумовых sim. Порог DEFAULT_RESONANCE=0.35
    слишком низкий; реально различает только >0.6-0.65 на тренированном кубе.
- Память памяти: FugaAI::answer показывает Route для каждого запроса — роутинг
  работает (все general). 604 869 entries загрузились за 1.9s.

### C++: beam / H-JEPA / OWM (порты)
- **beam** (cpp/decode.cpp --decoder beam): классический beam-3 top_m=5, логирование
  log-score гипотез, стоп-на-повторе. Реальный прогон на Rust-W (fuga_byte_w_800):
  **10 B** — совпадает с честным Rust-отчётом (beam=12-13 B seed-эхо). Работает.
- **H-JEPA** (cpp/hjepa.h): 3 уровня (ctx 4/3/2, stride 1/3/5), каждый = LatentPredictor
  с learn_latent (Widrow-Hoff латент→латент — новый метод core.h), predict_refined:
  L0-trajectory → L1-pred → err_traj (bind) → L2-corrected → dampen (0.5/0.5),
  converge = средней cosine коррекций. Смог-test: обучение всех 3 уровней на
  байтовых латентах → **converge=0.9986**, коррекции нормой 1 (не нули).
- **OWM** (cpp/fuga_core.h): consolidate_owm + invert_square (Гаусс, partial pivot) —
  Woodbury P ← P−PA^T(AP A^T+αI)⁻¹AP, Gram-Schmidt редукция до top_k.
  Смок: **consolidated=4** (4 направления защищены).
### Регрессии
- Rust lib 130/130 после всех правок. C++ make — 0 ошибок (2 косметик-warnings
  многострочный комментарий). Всё закоммичено.
### Файлы сессии
- `src/bin/dialogue_test.rs` (новый), `cpp/hjepa.h` (новый), `cpp/fuga_core.h` (learn_latent,
  consolidate_owm, invert_square), `cpp/decode.cpp` (beam), `cpp/train.cpp` (OWM-check)

## Сессия 08.08 (ночь, продолжение): гибридный retrieval (релевантность)
### Диагноз (из диалог-e2e, omni_cube_idf 604K записей)
- VSA-поиск один: sim-полотно 0.52-0.75 для ЛЮБОГО запроса — весь код похож
  в латентном, верх корпуса забит однотипными redis/jemalloc фрагментами.
- Лексика НЕ была подключена: MemoryStore::load_bin ставит text_index=None/
  vsa_idx=None (индексы только после build_text_index(), который НИКТО не
  вызывал на загруженных кубах) → search_by_text шёл линейным скан 604K.
### Улучшения (релевантный канал, FugaAI::answer core.rs:425)
1. **Гибридный панк**: search_by_text(query) → если lex-сигнал ≥0.20 → lex-ответы
   ПЕРВИН (с source_doc) + VSA-дополнение ТОЛЬКО ≥0.65 (мусорный топ 0.52-0.75
   больше не засоряет). Иначе старый VSA-путь.
2. **build_text_index() после load** в dialogue_test (и теперь обязателен в
   стендах) — инвертированный поиск по словам + filename-буст.
3. **Стоп-слова в search_by_text** (LEX_STOP ~70 англ+рус): how/are/you/what/
   и т.п. тонули в лексике кода, размывая ключевые. «hello how are you» до:
   lex=0.17 шум; с LEX_STOP: только "hello".
- Проверено: запрос «what is vector symbolic architecture» дал lex=0.25
  (SFMT-alti.h), «sorts an array» — lex=0.50 (lua.h) — лексика работать может.
- Честно: на 604K-фрагментах кода (обрезанные тексты чанков) слова hello/how
  часто отсутствуют в текстах записей — лексика-канал ограничен ДАННЫМИ,
  не кодом. На книжном корпусе (403K слов corpus.jsonl) — данные чище.
- Регрессии: lib 130/130. Файлы: src/ai/core.rs (answer гибрид),
  src/ai/memory_store.rs (LEX_STOP), src/bin/dialogue_test.rs (build_text_index,
  англ. запросы), src/bin/txt_search_debug.rs (новый, диагностика лекс-канала).
- **e2e РЕЗУЛЬТАТ (после фикса детектора lex=): 6/6 запросов с контентом; 3 из 3
  значимых запросов дают РЕЛЕВАНТНЫЕ lex-ответы** — «vector symbolic» → lex=0.67
  System Vector Protocol; «sorts an array» → lex=0.47 Z3+Rust Code Synthesis;
  «temporal memory» → lex=0.33 VSA Wave Memory (было: mutex.h/jemalloc шум).
  Функция уже отвечает содержимым памяти, а не шаблоном.

## Сессия 08.08 (поздний вечер): Единый формат FUGA1 и фикс токенного словаря
### 1. Объединение обучения C++ и Rust (`FUGA1` формат)
- Выбран и утвержден главный путь: **Сквозное байтовое двухскоростное латентное обучение (Byte-Level Two-Speed LatentJEPA + H-JEPA + OWM)**.
- Создан единый тегированный контейнер **`FUGA1`**: `TAG_LOCAL_W` (1), `TAG_PATCH_W` (2), `TAG_OWM_P` (3), `TAG_META` (4), `TAG_HJEPA` (5).
- Реализована бинарно-совместимая сериализация/десериализация в C++ (`cpp/fuga_core.h`) и Rust (`src/ai/htm_temporal.rs`: `save_unified_fuga1`, `load_unified_fuga1`). `load_byte_w` автоматически поддерживает `.fuga` файлы.
- Скрипты `full_byte_train.rs`, `gpu_train.rs`, `combined_decode.rs` и `py/train_cpp.py` переведены на поддержку единого формата `.fuga`.
- Написан и пройден интеграционный тест `unified_fuga1_roundtrip`.

### 2. Исправление бана словаря токенов (#7)
- `build_token_vocab_from_files` в `src/ai/self_mirror.rs`: удалено ограничение `allowed_dirs`, добавлен фолбэк на `node.name` для индексированных внешних исходников.
- Результат: `cargo test --lib` **131/131 PASS**.


## Сессия 09.08: ЕДИНЫЙ ФОРМАТ FUGA1 + главный путь обучения (объединение C++/Rust)
### ГЛАВНЫЙ ПУТЬ ОБУЧЕНИЯ (решение, аргументация)
- **Выбран**: байтовый Widrow-Hoff (локальный W) + патчевый two-speed (W_patch),
  с OWM-защитой. Это единый контур «двухскоростного обучения», которому
  подчинены и CPU, и GPU (batch_delta = тот же Widrow-Hoff).
- Почему НЕ остальные (честные A/B из AGENTS.md): entropy/BLT на этом пути
  даёт 200 байт с морфемами (best); naive 2-3, recurrent 17, beam 13, KAN 1,
  LSTM-peer 3. Формат уже bin-совместим (C++ и Rust реализуют один оператор).
- H-JEPA/TM-клетки — вспомогательные уровни: H-JEPA — контент-коридор
  (не конкурирует за главный путь), TM-клетки были источником OOM и тяжелы.
  Их веса хранятся в едином файле как опциональные секции (TAG_HJEPA).
### ФОРМАТ FUGA1 (один файл = разное обучение)
- MAGIC "FUGA1" + секции [u32 tag][u32 len][байты]:
  tag=1 LOCAL_W (512² f32), tag=2 PATCH_W (512²), tag=3 OWM_P (512²),
  tag=4 META (u64 steps, u64 patch_steps, u32 ctx, u32 version),
  tag=5 HJEPA (опц.), tag=0 END.
- C++: fuga_core.h save_unified/load_unified; train пишет .fuga (3 секции).
- Rust: htm_temporal.rs save_unified/load_unified (были в дереве) +
  save_unified_fuga1/load_unified_fuga1 на TemporalMemory; full_byte_train,
  gpu_train, combined_decode пишут/читают .fuga.
- Стенды: unified_e2e.rs (C++→Rust читает и декодит entropy-BLT, 8 байт),
  unified_roundtrip_cpp.rs (Rust пишет, self-check), unify_check.cpp (C++
  читает Rust-файл). Кросс-проверка: OWM diag=2.0, patch=-0.5, steps=777 —
  бит-в-бит в обе стороны. Rust lib 130/130.

## Сессия 09.08 (продолжение): Двухканальный CPU/GPU конвейер (two-speed на GPU)
- **GpuOps расширен**: второй W-буфер (w2_buf/x2_buf/err2_buf/staging2) + delta/cap
  bind groups для W_patch — ОДИН delta/cap-pipeline, два набора W (local + patch).
  Методы: upload_w2/download_w2/batch_delta2/cap_w2.
- **gpu_train.rs — двухскоростной конвейер**: CPU-поток готовит ДВЕ пары на шаг
  (local: окно байт→байт, patch: окно патчей→патч как learn_patch), GPU применяет
  оба Widrow-Hoff батчами; cap каждые 50×batch (cap_w + cap_w2).
- **Многокорпусность**: --jsonl "a.jsonl,b.jsonl" (7 корпусов одним прогоном).
- **Сохранение**: единый FUGA1 с ОБЕИМИ W (save_unified напрямую) + sidecar FBW1.
  Исправлено: при --out .fuga FBW1 не перезаписывает FUGA1 (sidecar всегда _w.bin).
- **Честное A/B** (60K, двухканально): GPU 2287 vs CPU-only 2082 pairs/s (~×1.10).
  Выигрыш ниже, чем одноканальный ×1.65 — при двухканальном узкое место
  encode (CPU готовит 2 пары/шаг), Widrow-Hoff на GPU почти бесплатен — как в
  однооператорном диагнозе (GPU util 30-35%).
- Полный прогон 10M на 7 корпусах (fuga_talk_gpu.fuga) — растёт двухканально.
- lib 131/131; верификация AD-HOC (gpu2_verify): build+smoke+FUGA1-структура OK.
