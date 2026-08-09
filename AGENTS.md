# AGENTS.md — состояние и контекст для агентов

## Проект (коротко)
Собственный «агентский стек» без внешних LLM: `necli/` (CLI) → маск `fuga-web` → генерация кода → гейты → рекурсивное дообучение.
**Главный путь обучения (09.08)**: сквозное байтовое двухскоростное латентное обучение — **локальный байтовый W (Widrow-Hoff, частотные биграммы) + патчевый W_patch (two-speed) + KAN-сплайны на остатке + OWM-защита**. Гибридный оператор `HybridTransition` (src/ai/hybrid.rs): `pred = W·x + α·KAN(x)` — W держит частотные пары, KAN учит нелинейные структурные аттракторы (единственный разделил линейно-неразделимые аттракторы в синтетическом пруфе).
Всё хранится в **едином формате FUGA1** (magic `"FUGA1"`, секции `[u32 tag][u32 len][bytes]`): tag=1 LOCAL_W (512²), tag=2 PATCH_W, tag=3 OWM_P, tag=4 META (steps/patch_steps/ctx/version), tag=5 HJEPA (опц.), **tag=6 KAN_C (512²×6 сплайн-коэфф.)**, tag=0 END. C++ (`cpp/fuga_core.h`) и Rust (`htm_temporal.rs`) бинарно-совместимы — проверено бит-в-бит в обе стороны (unified_e2e / unified_roundtrip_cpp).

## Статус обучения (09.08, ночь — ЗАФИКСИРОВАНО)
**Действующий прогон**: `unified_gpu_train` (двухскоростной CPU/GPU + KAN) на 4 корпусах → `fuga_unified_gpu_lms.fuga` — 2'000'000 пар за **971.5s (~2059 pairs/s, GTX 1660 Ti)**, чекпоинты каждые 500K (3 шт + финал), дефолтный `hybrid_train` (CPU) остановлен по решению пользователя (вариант 2: GPU).

### КРИТИЧЕСКИЙ ФИКС (LMS вместо Hebb) — коммит 463cfcc
**Диагноз**: предыдущий GPU-конвейер (gpu_train/kan_calib/unified_gpu_train v1) учил `err ≈ target` (W=0 init → pred=0). Это HEBB-накопление (`ΔW = η·t·xᵀ`), а НЕ Widrow-Hoff: без вычитания W·x нет отрицательной обратной связи → частотный шум накапливается, KAN получает сырой таргет вместо остатка → зацикливание декодера («ddoladopostiddys»).
**Решение (вариант b, GPU-native)**: добавлен `RESIDUAL_SHADER` в gpu_ops.rs — `err = t − W·x` считает GPU (W в VRAM, без PCIe-синхронизации на шаг). `hybrid_step()` = residual → W-delta → KAN-delta на ТОМ ЖЕ остатке (гибридная декомпозиция W+KAN восстановлена).
**A/B (честный, один корпус fisig, 50K пар)**:
| | Hebb (err≈target) | LMS (e=t−W·x) |
|---|---|---|
| naive TEXT | `nh\u{1} \u{12}n` (бинарный мусор) | **`ncget oht f`** (буквы+пробелы) |
| naive CODE | ` tnt \u{12}t` (мусор) | **`ataththtnvoegu`** (ASCII) |
| скорость | 2273 pairs/s | 2059 pairs/s (−9%) |
- Полный 2M LMS: naive даёт осмысленные токены (`\ne tr `), но связной генерации по-прежнему нет (naive 4-6B, entropy-BLT мусор) — следующий рычаг НЕ GPU-исполнение (оно честно), а архитектура декодера.
- НЕ подтверждено: гипотеза «LMS даст связность на 200K» — на 2M связности нет; нужен MegaByte-порядок (патч ДО байт) или косинус-выбор по KAN вместо argmax.

### Файлы-артефакты
- `fuga_unified_gpu_lms.fuga` (2M, LMS, все 5 секций) + `.ckpt.fuga` ×3 — полный прогон 16 мин
- `fuga_unified_gpu.fuga` (2M, Hebb-версия — исторический, не использовать)
- `fuga_unified_v2.fuga` (CPU 400K чекпоинт, прерван) — исторический

- Уроки скорости: KAN — самый тяжёлый оператор (1.5M коэффициентов); GPU-конвейер = **2059 pairs/s vs CPU hybrid 152 steps/s (×15)**, полный прогон 2M на GPU — 16 мин вместо 3.7 ч CPU.

## Реестр инструментов (src/bin/*.rs) — что реально полезно и когда
| Инструмент | Что делает | Когда использовать |
|---|---|---|
| `unified_gpu_train.rs` | **Трёхканальный CPU/GPU: local W + patch W + KAN на GPU, честный LMS остаток (RESIDUAL_SHADER), OWM на CPU, единый FUGA1+KAN** | **ГЛАВНЫЙ прогон обучения** (2059 pairs/s, 2M пар = 16 мин; `--ckpt-every` для ребута) |
| `hybrid_train.rs` | Гибрид W+KAN обучение (CPU, честный Widrow-Hoff) → FUGA1+KAN_C | CPU-эталон / малые A/B (152 steps/s) |
| `gpu_train.rs` | Двухскоростной CPU/GPU конвейер (local+patch W на GPU, cap-шейдеры) | Легаси: Hebb-остаток (err≈target), НЕ использовать для качества — только для замера конвейера |
| `kan_calib.rs` | Калибровка KAN, включая GPU-канал (kan_batch_delta) | Проверка KAN-оператора отдельно |
| `talk_model.rs` | Декод FUGA1-чекпоинта всеми декодерами (naive/two-speed/entropy BLT/…), seed | Быстрая проверка чекпоинта |
| `unified_e2e.rs` / `unified_roundtrip_cpp.rs` | Кросс-проверка FUGA1 между C++ и Rust | После любого изменения формата/сериализации |
| `combined_decode.rs` | Чекпоинт → patch-vocab → KAN+LSTM → 10 декодеров | Сравнение технологий на сохранённых чекпоинтах |
| `train_byte_w.rs` | Обучение локального W напрямую (learn_transition) + sidecar FBW1 | Когда нужен только байтовый W без TM-клеток |
| `checkpoint_check.rs` | Интроспекция чекпоинта (клетки, W_updates, trained?) | Проверка, что чекпоинт реально обучен |
| `byte_baseline.rs` | Честный сверстник LSTM на том же корпусе и метрике | Контраст «VSA vs классика» в публикациях |
| `dialogue_test.rs` | e2e FugaAI.answer() на реальных кубах (роутинг+retrieval) | Диалоговые проверки памяти |
| `gpu_bench.rs` | CPU vs GPU A/B для Widrow-Hoff дельты | Замер узких мест |
| `two_speed_test.rs`, `rec_test.rs`, `byte_gen_test.rs` | A/B для two-speed / рекуррентных / байтовых декодеров | Эксперименты декодеров, не для прод-прогонов |
| `txt_search_debug.rs` | Диагностика лексического канала на кубе | Разборы retrieval |
| `omni-web.rs` (fuga-web), `tgbot.rs`, `fuga` (main.rs) | Продовские маск/CLI | Продакшен |

## Лучшие решения (проверенные честным A/B, 08.08–09.08)
1. **entropy/BLT патчер — лучший декодер**: 200 байт из бюджета (vs naive 2–3, recurrent 17, beam 13, KAN 1, speculative 10). Механика: gap топ-1−топ-2 решает скорость эмиссии; **gap-порог 0.55–0.65** — точка перегиба (ниже — самоповтор 6б, выше — 200б/1004 B/s).
2. **Two-speed W_patch — ×26 генерации** (2 → 52 байта, узнаваемые морфемы `dendrite`).
3. **Гибрид W+KAN+OWM**: W → частотные пары, KAN → остаток; единственный оператор, разделивший линейно-неразделимые аттракторы; FUGA1 расширен tag=6.
4. **OWM (Woodbury)**: защита ранее выученных направлений (consolidated=4 в C++).
5. **Byte-level без словаря**: 256 сырых UTF-8 байтов, позиционная свёртка — опечаткоустойчивость, любой язык и код.
6. **GPU-конвейер**: двухканальный (local+patch) 2287 pairs/s vs 2082 CPU; одноканальный ×1.65. Бутылочное горло — encode на CPU.
7. **Честный LSTM-сверстник** (byte_lstm.rs, 230K params): классическая рекуррентность в том же масштабе не обходит fuga (микро-слайс даёт 200б, но полный 800: 1б vs entropy 200б) — публикуемый контраст.

## Известные флаки и грабли
- **OOM-класс**: неограниченный mpsc::channel (1.5M пар → 6GB RAM). Правило: ВСЕ конвейеры — только `sync_channel` (bounded).
- Неограниченный `dirs.push` каждые 64 шага — источник OOM.
- Флаки-тест `latent_jepa::encoder_is_deterministic_and_512_dimensional` падает ~1/3 в параллели (гонка TOKEN_SDR_CACHE), изолированно ok. Pre-existing.
- Недетерминизм: loss L0/L1 ~1.4–1.5 не считать баг (масштаб метрики VSA-блinding); смотреть cosine pred~actual.
- TOP-1 argmax залипает на частотную бигра площадка `er er` — предел не декодер, а качество локального байтового W.
- **Честные отрицательные итерации 3–6**: scheduled-sampling, nucleus, hopfield-read, beam, speculative и регулярный KAN (без калибровки: 1 байт `}` за 431s) не обгоняют entropy/BLT в малых масштабах.

## Формат чекпоинтов и API
- `save_unified_with_kan` / `load_unified` (htm_temporal.rs) — FUGA1+KAN; `save_byte_w`/`load_byte_w` — sidecar FBW1 (magic `"FBW1"` + len + f32[262144]); дефензивный load → None, никогда не паникует.
- `TemporalMemory::new(64, ctx)` — TM-обёртка-энкодер; `apply_byte_w(w)` прикрепление W.
- Декодеры в `tm_generate.rs`: `tm_generate_latent_bytes`, `tm_generate_two_speed[_entropy]`, `tm_generate_recurrent[_nucleus]`, `tm_generate_hop_reader`, `tm_generate_beam`, `tm_generate_hybrid`, `tm_generate_speculative`.

## Команды
- Сборка: `cargo build --release` (все bin-стенды в src/bin). Lib-тесты: `cargo test --release --lib` — **132/132**.
- Обучение (главный путь): `./target/release/hybrid_train --jsonl "a.jsonl,b.jsonl" --max-steps N --lr-w 0.05 --lr-kan 0.3 --out model.fuga`
- GPU: `./target/release/gpu_train --jsonl ... --out model.fuga` (для переживаемости ребута добавлять `--ckpt-every`).
- Проверка формата после правок: `./target/release/unified_e2e && ./target/release/unified_roundtrip_cpp`.
- C++: make в cpp/ (fuga_core.h — save/load_unified; decode.cpp).

## Незавершённые задачи (next steps)
1. **MegaByte-порядок декодера** (ВЫБРАН пользователем 09.08 ночь): патч-оператор решает ДО байтов — сначала предсказание патча (W_patch глобальный), затем байты ВНУТРИ выбранного патча (локальный W). Сейчас two-speed обратный: локальный байт + патчевая эмиссия. Цель: концентрация решения (патч = направление, байты = детализация). Кандидаты: перестроить tm_generate_two_speed → сначала W_patch прогноз, потом внутри; или косинус-выбор кандидата по KAN-прогнозу вместо argmax.
2. **Оптимизация GPU-KAN**: батчан-диспатч (цикл по парам внутри шейдера, один poll) — устранить launch-overhead; крупные storage-буферы.
3. **Полный прогон 20M+ с гибридом** (W+KAN+OWM) на GPU — проверка генерации на большом бюджете (~2.5 ч).
4. Калибровка KAN (мягкий per-node cap, выше lr) на реальном корпусе.
5. `soft_sdr` — редкое использование (1 место, htm_temporal): либо интегрировать, либо убрать из mod.rs.
6. Ревизия легаси-модулей src/ (spatial/render/sim/physics; зависимости rapier3d/minifb/hound) — используются ли fuga-web/tgbot-ом.
7. Судьба больших корпусов и чекпоинтов: держать в корне только то, что читает код (см. ниже).

## Данные после чистки (09.08)
При старте проекта 88G в корне → после чистки **39G**.
- **Оставлено в корне** (читается кодом — проверено grep): `fuga_code_cube_mem.bin` (21G, main/tgbot/ingest), `fuga_knowledge_mem.bin` (3.4G), `fuga_code_cube_code_mem.bin` (2.4G), `fuga_stack_tm.bin`, `fuga_crystal.bin`, `fuga_htm.bin`, `fuga_mirror_{jepa,nodes,tm}.bin` (только базовые), `omni_cube.{bin,mem,idf,idf_mem}`, `fuga_talk_gpu.fuga`, `fuga_full_byte_tm.*`.
- **Удалено**: старые серии mirror (500–2500/phase/v1/trainonly/hybrid), `crystal_old/merged/v4`, `htm_src/htm_src_w/htm_tokio`, omni-варианты (1024/3d8/best/repos/v2/v3), sweep-кубы, `w_base.bin`, `fuga_5.gguf`, sidecar-бэкапы; **`train_checkpoints/` целиком (30G** — подтверждено бит-в-бит дубль корня, 16MB-хэши), **`target/debug` (13G)**, `test.1.er` (188M), `fuga_new_files.zip`, `src/bin/{chat.html,fuga_nano_mem.bin}`, `cube-3d.html`.
- Безопасно не тронуто: `target/release` (сборка для обучения), корпуса `.jsonl`, `workspace/`, `temp_repos/`, `necli/` (нужны кодам).
- Манифесты чистки: `/home/slava/.hermes/tmp/fuga_cleanup/` (manifest_phase1.txt, manifest_train_checkpoints.txt, AGENTS.md.bak).

## Ревизия стека под байтовый стандарт (09.08, вечер)
- **Удалена мёртвая ветка** (0 внешних использований, только токен-модели вне байтового стандарта): `src/ai/{agent,autonomous_mind,unity_mind,unified_mind,state_loader,mentalese,predictive_coder}.rs`, `src/bin/autonomous_cycle.rs` (+ их pub-use из mod.rs/lib.rs). Полная пересборка `cargo check --release --lib --bins` — прошла (только warnings).
- **Оставшееся деление**: байтовый/латентный стандарт (live): `sdr.rs` (byte_basis), `latent_jepa.rs`, `htm_temporal.rs`, `tm_generate.rs`, `kan.rs`, `hybrid.rs`, `hopfield.rs`, `byte_lstm.rs`, `gpu_ops.rs`, `soft_sdr.rs`; сервис-слой (core/memory_store/moe/…, используется omni-web/tgbot) — остаётся, т.к. подключает прод-память; легаси-токенные (self_mirror/temporal_predictor/hierarchical_jepa, text-dir) — вне стандарта, используются старыми стендами (omni-web fallback), требует отдельного решения — см. next steps.### Диагноз убийств прогонов (exit=143) — ОКОНЧАТЕЛЬНЫЙ (09.08)
- НЕ OOM, НЕ ребут: journalctl показывает `PrepareForSleep(true)` от
  org_kde_powerdevil — ноутбук уходит в suspend (крышка/таймаут), процесс
  получает SIGTERM. RAM была в норме (2722MB free), dmesg без OOM-kill.
- ЛЕЧЕНИЕ: длинные прогоны запускать через
  `systemd-inhibit --what=sleep ./target/release/hybrid_train ...` —
  держит inhibitor, PowerDevil не усыпит машину. Плюс --ckpt-every.
