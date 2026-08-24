# SESSION LOG — Fuga Cognitive Engine

> Журнал фиксируется с 24.08. Каждое изменение архитектуры, метрика и решение
> записываются сюда. Коммитится в git на каждой итерации.

---

## Стек (итоговое состояние)

| Слой | Технология | Файлы |
|---|---|---|
| Data streaming | Go (goroutines, gzip, dedupe) | `fuga-downloader/fuga_downloader.go` |
| Bit core / фильтры / IBM-1 / symbolic | Rust + PyO3 (`fuga-core`, maturin) | `fuga-core/src/{lib,filter,ibm_model,symbolic_eval}.rs` |
| Обучение | PyTorch CPU (venv `.venv`) | `antitf/*` |
| Legacy engine | Rust `fuga` crate | `src/` |

Сборка PyO3: `cd fuga-core && PATH="$HOME/.cargo/bin:$PATH" ../.venv/bin/maturin develop --release`

---

## Хронология результатов

### Этап A. Ядро VSA+KAN+VICReg+OWM
- VSA Encoder (nn.Module, packed u64, chunked forward), HybridBinder
  (XOR-bind, bit-rotate permute, per-bit majority).
- ChebyKAN (Чебышев T_k), VICReg, WoodburyOWMExecutor (SVD + Woodbury,
  O(B^3), факторизованная проекция без [D,D] матриц).
- Тесты: `tests/test_smoke.py` — все зелёные.

### Этап B. Кросс-язык (Tatoeba → 150K реальных пар)
- IBM Model-1 в Rust (интернинг u32): **1.5M словарных пар за 10.9s**.
- word-hit@heldout = **0.67**; дом→house 0.77, кошка→cat 0.59.
- Cross-Aligner эволюция (held-out acc@1 unbinding):
  | Архитектура | train | heldout | gap |
  |---|---|---|---|
  | Whole-Sentence | 0.689 | 0.250 | +0.44 |
  | Slot-Positional | 0.645 | 0.276 | +0.37 |
  | **Slot+IBM-supervision** | 0.456 | **0.268** | **+0.19 ✅** |

### Этап C. Фильтр грамматичности
- RuCoLA (real CSV) + Wiktionary-дампы (kaikki 89MB gz → **477K словоформ**,
  стриминг Go за 19.5s; Rust JSONL 47K w/s).
- POS-биграммы отвергнуты измерением (0.650 vs 0.650) — нет дискриминации.
- Рабочая точка пре-фильтра: пороги 0.6/0.3, reject_salad до **0.92**, 7ms/batch.
- Вывод: потолок 0.61 balanced без dependency-парсера; tree-sitter-russian
  не существует на crates.io (проверено API).

### Этап D. Символьный гибрид
- SymbolicExecutor: рекурсивный парсер +-*/(), скобки.
- GSM8K end-to-end: **452/494 = 91.5% точных ответов**, нуль галлюцинаций чисел
  (против ~2% у чисто латентной петли мыслей).

### Этап E. Память фактов
- SPO ConceptNet пер-субъектные бандлы: **4/4 запросов** после перехода на
  точную биполярную алгебру F = S@1 ⊗ R@2 ⊗ O@3 (без мажоритарных потерь).
- Уроки: циркулярная верификация ловится только прогоном; naive cleanup
  «ближайший атом» теряет роль.

### Этап F. Slot-KAN декодер (ветка feature/slot-kan-decoder)
- Контрольный эксперимент: бандл-HV условие игнорируется декодером
  (ce 1.40 ≈ безусловный 1.37) — vanity-метрика byte-bigram fluency вскрыта.
- Slot-Attention decoder: ce/**байт 0.61** vs 1.40 — сигнал течёт ✅.
- Free-running coverage пока 0.23 (нужны длинный прогон, sharpened attention,
  scheduled sampling).

### Этап G. v2.0 Persistent Memory + Reflection (эта ветка)
- `fuga_memory.py`: PersistentVSAMemory (facts/episodes JSONL +
  затухающий HV-аккумулятор контекста Decay·ctx + Bind(input)).
- **Кросс-сессия доказана**: факт выучен в диалоге → восстановлен в новом процессе.
- `fuga_reflection.py`: критик (coverage/accept/integrity/novelty),
  строгий отбор **10%** рефлексий в self_reflection.jsonl.
- `fuga_chat.py`: диалоговый контур (интенты greeting/calc/query/learn/chat),
  кросс-языковой фолбэк через IBM, стем-матчинг RU, persona synthesizer
  (5 стилей, контекстные прологи, детерминированные от VSA-хеша субъекта).

### Этап H. VL-JEPA ingest (в работе)
- COCO val2017 + annotations качаются (resume-режим curl -C -).
- `vljepa_dataset_loader.py`: патчи 32×32 → handcrafted признаки
  (цветовые бины/градиент/яркость) → VSA-бандл с пространственными
  позициями P:i_j → PersistentVSAMemory ("vision:" факты).
- MSR-VTT недоступен (gated зеркала); видео-альтернатива — UCF101 (открытый).
- Обучаемый JEPA visual encoder — следующий шаг после smoke-теста инжеста.

---

## Известные баги/уроки (не повторять)

1. Ротация НЕ самоинверсна — развязка требует bits − pos.
2. OWM с пустой памятью задач замораживает все градиенты → схема
   Adam-warmup → OWM fixation (фаза 2).
3. pyo3: второй #[pymethods]-блок не экспортируется; Result<_,String> не
   конвертится в PyErr; класс надо add_class в pymodule.
4. bind_batch возвращает packed u64 [B,32], не float — всегда unpack.
5. Фильтр без загруженных переходов отбраковывает всё (trans_cov=0).
6. sign() в выходном слое убивает градиент — бинаризовать только no_grad.
7. Self-play на переводе самопределен при слабом генераторе (−0.007);
   продуктивен только coverage-targeted или с сильным верификатором.
8. Латентная петля мыслей над хеш-сегментами не переносится на новые задачи
   (cos gap +0.022); арифметика — зона символьного ядра.

## Открытые пункты

- [ ] Push веток (token revoked): slot-kan-decoder, v2.0-vljepa-memory
- [ ] VL-JEPA: обучаемый визуальный энкодер (после smoke инжеста)
- [ ] OWM fine-tune slot translator на self_reflection.jsonl
- [ ] Длинный прогон slot-decoder (scheduled sampling, attention sharpening)
- [ ] UCF101 для motion-VSA (замена gated MSR-VTT)

### Этап H — продолжение (24.08, инжест проверен)
- COCO val2017 (5000 img) + annotations загружены и распакованы;
  train2017 докачивается фоном (resume через curl -C -).
- `vljepa_dataset_loader.py` smoke: **300 изображений → VSA-кристаллы за 5.1s**
  (~60 img/s), 20 vision-фактов записано в PersistentVSAMemory.
- Кросс-модальный retrieval БЕЗ контрастивного обучения: acc@1=0.000
  (шанс 0.0033), acc@10=0.027 — ожидаемо: handcrafted цвет/граничные хеши
  не выровнены с текстовыми HV. Следующий шаг — контрастивный JEPA-тренинг
  (image-HV ↔ caption-HV) поверх готового инжеста.
- Урок: `from __future__ import annotations` должен стоять сразу после
  docstring; импорты torch — до использования в декораторах класса.

### Этап I. Контрастивный мост VL-JEPA — НЕГАТИВНЫЙ РЕЗУЛЬТАТ (24.08)

Три варианта выравнивания HV_image <-> HV_caption на COCO val (2400 train /
600 held-out), все завершились held-out acc@1 ≈ шанс (0.0017):

| Вариант | TRAIN acc@1 | HELDOUT | Диагноз |
|---|---|---|---|
| handcrafted хеши + InfoNCE | 1.00 (меморизация) | 0.003 | признакам нечем
пересечься |
| ResNet18-frozen + InfoNCE | — | — | **коллапс в точку**: logit_std 0.83→0.007,
loss→ln(B)=4.85 |
| ResNet18 + InfoNCE+Var-барьер | 0.007 | 0.002 | коллапс предотвращён, но парной
структуры не найдено |

**Корневая причина (количественная):** обе стороны моста семантически пусты —
визуальная (замороженный ResNet без общего обучения) и текстовая (случайный
VSA-базис слов) не имеют пересекающейся статистики, из которой малые MLP за
2K шагов могли бы извлечь соответствие. CLIP-класс выравнивания требует
миллионы пар и крупные энкодеры с обеих сторон.

**Урок:** контрастивный тренинг механически исправен (изолированный тест
на меморизацию сходится за 50 шагов; Var-барьер устраняет коллапс) —
узкое место в ДАННЫХ/ЭНКОДЕРАХ, не в коде.

**Путь вперед (за пределами текущего песочницы-CPU):**
1. Полный COCO train2017 (18ГБ, докачивается) + GPU + десятки эпох.
2. Либо предобученный CLIP/ViT-B энкодер для обеих модальностей — тогда
   VSA-память принимает уже выровненные латенты без собственного тренинга.
3. Vision-факты в PersistentVSAMemory остаются полезными для
   ассоциативного поиска по изображениям внутри одной модальности.

Коммиты: vljepa_align.py (v0), vljepa_align2.py (v1 + Var-барьер),
vljepa_bridge_v1.pt.
