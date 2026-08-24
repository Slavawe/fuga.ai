# Fuga / Astral Engine

Бестокеновый когнитивный движок: **VSA + H-JEPA + ChebyKAN + OWM** без
токенизаторов, attention-трансформеров и KV-кэша.

## Ядро архитектуры

```
Сырые байты / пиксели
   → VSA гипервекторы (детерминированное связывание, 2048–32768 бит)
   → Vector Adapter (chunk pooling + L2)
   → H-JEPA предиктор латента следующего шага (ChebyKAN)
   → VICReg Loss (антиколлапс) + Woodbury OWM (анти-забывание)
   → Slot-Conditioned Byte-KAN Decoder (поверхность речи)
```

Каждый слой делает то, в чём силён:
- **Rust core** (`fuga-core`, PyO3): байтовые VSA-операции на rayon,
  AST-энкодер tree-sitter, лингвистический фильтр, IBM Model-1 EM за секунды,
  символьный исполнитель точной арифметики.
- **Python/PyTorch** (`antitf`): обучение латентных предикторов.
- **Go** (`fuga-downloader`): стриминг датасетов.

## Валидированные результаты (всё воспроизводимо)

| Задача | Метрика | Модуль |
|---|---|---|
| Развязка слова из бандла 12K кандидатов | acc@1 = **1.00** | `HybridBinder` |
| Словарь RU→EN (150K пар, 10.9s EM) | word-hit = **0.67** | `IbmModel1` |
| Выравнивание KAN-моста | held-out 0.27, gap **+0.19** | VICReg+OWM |
| Арифметика GSM8K (гибрид) | **91.5% точно**, 0 галлюцинаций | `SymbolicExecutor` |
| Ёмкость 32K бандла | 256 слотов @ acc 1.00 | Astral |
| Language→World управление | **200/200** бит-в-бит | Action-Grounding |

## Честные негативные результаты (тоже зафиксированы)

- Бандл-HV целого предложения как условие декодера — игнорируется
  (доказано контрольным экспериментом); работает только адресуемый слот.
- Контрастивный мост ResNet↔word-hash без CLIP-класса энкодеров — шанс.
- Чисто латентная петля «мыслей» над хешами не решает арифметику (2%).
- Самопоучение на собственных переводах нейтрально при слабом генераторе.

Полный журнал: [SESSION_LOG.md](SESSION_LOG.md).

## Быстрый старт

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy pyarrow opencv-python-headless

# Rust-ядро -> Python модуль
cd fuga-core && PATH="$HOME/.cargo/bin:$PATH" ../.venv/bin/maturin develop --release && cd ..

# диалог на реальных фактах ConceptNet
.venv/bin/python fuga_chat.py --demo

# мультимодальный инжест COCO -> VSA
.venv/bin/python vljepa_dataset_loader.py
```

## Структура

```
antitf/            PyTorch ядро (KAN, JEPA, VICReg, OWM, память)
fuga-core/         Rust/PyO3 (VSA, фильтры, IBM, symbolic)
astral/            автономная среда 32K VSA (env, runner, ingest_yt, MoK)
fuga-downloader/   Go-стример дампов
tests/             smoke-тесты
```

## Статус

`v0.2-alpha` — исследовательский движок. Диалог, факты, вычисления и
мультимодальная память работают; свободная связная генерация речи —
активное направление (slot-conditioned декодер, ce 0.61/байт).

⚠️ Токены доступа никогда не коммитятся. Утечки — отзыв немедленно.
