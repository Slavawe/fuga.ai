<p align="center">
  <img src="photo_2026-08-01_11-32-28.jpg" alt="Fuga 2.0 logo" width="400">
</p>

<h1 align="center">Fuga 2.0 — VSA Иерархическая Предсказательная Память и Генерация Кода</h1>

**Ядро AGI без трансформеров**, построенное на Vector Symbolic Architectures (VSA), Иерархическом JEPA, Временной Памяти (Temporal Memory) и локально-чувствительном связывании. Заменяет обратное распространение ошибки локальными обновлениями по Delta-правилу и генерирует Rust-код через чистую VSA/TM-авторегрессию — без зависимости от LLM.

---

## Ключевые концепции

| Технология | Файл | Что делает |
|---|---|---|
| **Hypervector** | `src/core/hypervector.rs` | 8192-битные, ~2% плотности (~164 активных бита), XOR bind / sum bundle / permute |
| **Hierarchical JEPA** | `src/ai/hierarchical_jepa.rs` | L0 (статическая), L1 (макрo), L2 (метапознание) с `ls_bind` фазовым сдвигом |
| **Temporal Memory** | `src/ai/htm_temporal.rs` | Ячейки с DendriteSegments, learn_segment / reinforce / prune / predict_next |
| **SDR** | `src/ai/sdr.rs` | `encode_text()` → детерминированный хеш-основанный разрежённый бинарный вектор |
| **PhaseCrystal** | `src/ai/crystal.rs` | Ассоциативная VSA-память (фазы L0/L1/L2), `learn()` / `query()` / порог резонанса, FNV-1a индекс ключей, `crystal-reencode` (позиционно-инвариантная перекодировка) |
| **Tokenizer Bridge** | `src/core/tokenizer_bridge.rs` | Выводит детерминированные VSA-гипервекторы на токен через n-граммный permute-bind энкодер — без 128K embed-векторов, всего несколько МБ словаря |
| **Weight Transpile** | `src/ai/transpile.rs` | Стримит HuggingFace safetensors шарды → VSA-кристалл; параллельный фетчер чанков, режим `--raw`, извлечение MoE-маршрутов |
| **MoE Router** | `src/ai/moe.rs` | Mixture-of-Experts с обучаемым роутером, команды `set-router` / `set-topk` |
| **Resonance Attention** | `src/ai/resonance_attention.rs` | VSA-основанное внимание над фазовыми векторами |
| **SDR Store / HNSW** | `src/ai/sdr_store.rs`, `src/ai/hnsw.rs` | Постоянное SDR-хранилище + приближённый индекс ближайших соседей |
| **Self Mirror** | `src/ai/self_mirror.rs` | Индексирует `.rs` файлы в PhaseNodes (SDR) + состояние TM + HJEPA |
| **Multi-layer Weaver** | `src/multi/` | syntax_layer / semantic_layer / chaos_layer / language / translate / patterns |
| **Fuga Synthesizer** | `src/core/fuga_synthesizer.rs` | Синтез голоса/речи |
| **GGUF snapshot** | `src/gguf.rs` | `snapshot-gguf` — чтение GGUF файлов моделей |
| **GPU** | `src/gpu.rs` | CUDA (GTX 1660 Ti) хуки ускорения |

---

## Конвейер обучения

### 1. Индексация исходников в фазовый граф

```bash
# Индексация директории — читает .rs файлы, создаёт PhaseNodes с SDR-кодированием
cargo run --release -- mirror-index src/ai

# Загрузка существующего зеркала и индексация другой директории
cargo run --release -- mirror-index src/core
```

Создаёт `fuga_mirror_nodes.bin` (фазовые узлы), `fuga_mirror_tm.bin` (TM), `fuga_mirror_jepa.bin` (HJEPA).

### 2. Обучение предсказателя (HJEPA + TM)

```bash
# Обучение на существующих узлах зеркала (5 эпох, chunk=1)
cargo run --release -- train-predictor 5

# С большими чанками для паттернов последовательностей
cargo run --release -- train-predictor 10 --chunk 3
```

### 3. Обучение токенного словаря (встроено в генерацию)

```bash
# Строит топ-4000 посмвольный токенный словарь из проиндексированных .rs файлов
# затем обучает TM на 20000+ шагах биграмм токенов
# затем генерирует токены
cargo run --release -- generate-code "fn new" --tokens
```

Токенный тренер:
- Посимвольный токенизатор: разделяет идентификаторы от операторов, распознаёт `->` `::` `=>` `!=` `==` `>=` `<=` `+=` `-=` `&&` `||`
- Инъекция синтаксических паттернов: 14 захардкоженных Rust-паттернов × 5 повторов
- WTA (Winner-Take-All) предсказание с Inhibition of Return
- Анти-повторное окно (16 токенов)

---

## Фазовый кристалл (VSA-память)

```bash
# Запись пары ключ/текст в кристалл
cargo run --release -- crystal-learn <key> <text>

# Обучение на целой директории текстовых файлов (нарезка на сниппеты по 20 слов)
cargo run --release -- crystal-learn-dir corpus/ --from fuga_crystal.bin --threshold 0.28 --chunk 20

# Запрос к кристаллу
cargo run --release -- crystal-query "ваш текст" --from fuga_crystal.bin

# Тест резонанса (точные ключи, шум, скан матрицы)
cargo run --release -- crystal-test --from fuga_crystal.bin

# Перекодировка всех записей позиционно-инвариантным энкодером
cargo run --release -- crystal-reencode --from fuga_crystal.bin

# Статистика / popcount скан / забывание
cargo run --release -- crystal-stats --from fuga_crystal.bin
cargo run --release -- crystal-popcount "текст"
cargo run --release -- crystal-forget <key>
```

Ключевые свойства:
- **Позиционно-инвариантный n-граммный энкодер** (`encode_bytes_nopos`): сниппеты из середины чанков резонируют — светятся и середина/конец, а не только начало.
- **Dedup грамм**: частые слова занимают один слот, поэтому точность top-1 остаётся 99–100%.
- **Мягкий overlap-скоринг** (`intersection / max(popcount(q), popcount(e))`) с порогом резонанса — шум → тишина в CLI.

---

## Перенос весов (HF safetensors → кристалл)

Стримит модель HuggingFace прямо в кристалл без хранения сырых весов:

```bash
# Полный raw-дамп: каждый не-MoE тензор становится отдельной фазовой записью
cargo run --release -- transpile deepseek-ai/DeepSeek-V4-Flash-0731 --raw --whole --concurrency 8 \
  --finalize /tmp/raw_v4.bin --state /tmp/raw_v4.state

# Возобновляемо — передайте тот же --state для продолжения после сбоя
```

- Параллельный фетчер чанков (чанки 16 МБ, общий keep-alive агент) — насыщает предел пропускной способности HF.
- `--whole` стриминговое окно держит RAM ограниченной (переживает OOM-уязвимые хосты с малым объёмом памяти).
- `--raw` сохраняет каждый плотный тензор (embed / attn / mlp / norm) как фазу; MoE роутеры/gates/эксперты исключены — они уже в кристалле как маршруты экспертов.
- `--dry-run` показывает тензоры без сохранения.

---

## Генерация

### Уровень токенов (синтаксический)

```bash
cargo run --release -- generate-code "fn new" --tokens
```

Выдаёт настоящие Rust-токены: `( ) { } [ ] , :: . ' -> ` + идентификаторы, числа

### Уровень PhaseNode (семантический)

```bash
# Beam search по графу PhaseNode
cargo run --release -- generate-code "struct Foo"

# Авторегрессивный режим (генерирует полные сниппеты)
cargo run --release -- generate-code "fn new" --gen

# С шириной beam и температурой
cargo run --release -- generate-code "async fn" --beam 3 --temp 1.2
```

---

## Запросы и оценка

```bash
# Self-query — поиск подходящих фазовых узлов
cargo run --release -- self-query "async fn handle"

# Запрос к кристаллу
cargo run --release -- crystal-query "текст" --from fuga_crystal.bin

# Оценка качества зеркала
cargo run --release -- eval

# Инспекция текста или файла
cargo run --release -- inspect "fn new() -> Self"
cargo run --release -- inspect src/main.rs
```

---

## Тесты

```bash
# Все библиотечные тесты
cargo test --lib

# Детекция аномалий (Inhibition of Return, overshoot)
cargo test --test test_anomaly_detection -- --nocapture

# JEPA / TM / MoE тесты
cargo test --test jepa_test
cargo test --test hierarchical_jepa_test
cargo test --test moe_routing_test
```

---

## Архитектура

| Компонент | Описание |
|---|---|
| Hypervector | 8192-битный, ~2% плотности (~164 активных бита), XOR bind / sum bundle / permute |
| Hierarchical JEPA | L0 (статическая), L1 (макро), L2 (метапознание) с ls_bind фазовым сдвигом |
| Temporal Memory | Ячейки с DendriteSegments, learn_segment / reinforce / prune / predict_next |
| SDR (Sparse Distributed Representation) | `encode_text()` → детерминированный хеш-основанный разрежённый бинарный вектор |
| PhaseCrystal | VSA-ассоциативная память; FNV-1a L0 индекс, мягкий overlap-скоринг, порог резонанса → тишина на шуме |
| Tokenizer Bridge | N-граммный permute-bind энкодер (`encode_bytes_nopos`), dedup грамм, MAX_GRAMS=48, позиционно-инвариантный |
| Weight Transpile | Стриминг safetensors → кристалл; параллельный фетчер чанков по 16 МБ; `--raw` / `--whole` / возобновляемое состояние |
| Tokenizer | Посимвольный: разделяет идентификаторы от операторов, распознавание многосимвольных операторов |
| WTA | Winner-Take-All с Inhibition of Return (усталость = победы × 10, затухание каждые 10 шагов) |
| AnomalyEvent | Детекция перегрузки фазы — `pred_count > 100` или `power_mw > 500` запускает overshoot |

## Лицензия

Apache-2.0
