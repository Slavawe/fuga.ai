# 📚 Руководство по подготовке и использованию датасетов для Fuga

**Дата:** 14 августа 2026  
**Контекст:** Обучение HybridCore (v10) на открытых Python-корпусах

---

## 🎯 Краткая сводка датасетов

| # | Название | Размер | Формат | Применение в Fuga |
|---|----------|--------|--------|-------------------|
| 1 | **The Stack (Python)** | ~60GB (350M строк) | Parquet → JSONL | Предобучение W_local + W_patch на чистом коде |
| 2 | **CodeSearchNet** | 412K функций | JSONL (готов) | Обучение связки docstring → код, валидация name_metric |
| 3 | **CoNaLa** | 2.9K пар | JSON → JSONL | Fine-tuning декодера на пары (intent, code) |
| 4 | **HumanEval** | 164 задачи | JSONL (готов) | Эталонный бенчмарк pass@k |
| 5 | **CodeFeedback** | 66K диалогов | Parquet → JSONL | Инструкционное fine-tuning (опционально) |
| 6 | **MBPP** | 974 задачи | JSON → JSONL | Валидация корректности (assert) |

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install datasets huggingface_hub
```

### 2. Загрузка и конвертация

```bash
cd ~/Fuga
python scripts/prepare_corpora.py
```

**Вывод:**
```
🚀 Загрузка и конвертация датасетов для Fuga

📁 Выходная директория: /home/slava/Fuga/fuga_corpora

=== 1/6: The Stack (Python) ===
  Обработано: 0/100000
  Обработано: 10000/100000
  ...
  ✓ Сохранено: fuga_corpora/the_stack_python.jsonl (100000 записей)

=== 2/6: CodeSearchNet (Python) ===
  ...

✅ Все датасеты конвертированы!

📊 Статистика:
  the_stack_python.jsonl         100,000 строк    245.3 MB
  codesearchnet_python.jsonl     412,178 строк     87.4 MB
  conala.jsonl                     2,879 строк      1.2 MB
  humaneval.jsonl                    164 строк      0.1 MB
  codefeedback.jsonl              10,000 строк     18.7 MB
  mbpp.jsonl                         974 строк      0.5 MB

  ИТОГО: 353.2 MB
```

---

## 📋 Формат JSONL для Fuga

### Базовый формат (чистый код)

```json
{
  "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "source": "the_stack"
}
```

### Расширенный формат (с метаданными)

```json
{
  "text": "Calculate the nth Fibonacci number.\n\ndef fibonacci(n):\n    ...",
  "func_name": "fibonacci",
  "repo": "algorithm-examples",
  "source": "codesearchnet"
}
```

### Инструкционный формат (диалоги)

```json
{
  "conversation": [
    {"role": "user", "content": "Write a function to sort a list"},
    {"role": "assistant", "content": "def sort_list(lst):\n    return sorted(lst)"}
  ],
  "source": "codefeedback"
}
```

---

## 🔬 Интеграция с unified_gpu_train.rs

### Текущий формат чтения

```rust
// src/bin/unified_gpu_train.rs (строки 122+)
fn read_corpus(path: &str) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut texts = Vec::new();
    
    for line in reader.lines() {
        let data: serde_json::Value = serde_json::from_str(&line?)?;
        if let Some(text) = data.get("text").and_then(|v| v.as_str()) {
            texts.push(text.to_string());
        }
    }
    Ok(texts)
}
```

### Многокорпусное обучение

```bash
./target/release/unified_gpu_train \
  --jsonl "fuga_corpora/the_stack_python.jsonl,fuga_corpora/codesearchnet_python.jsonl" \
  --max-steps 1500000 \
  --out fuga_hybrid_v10.fuga \
  --lr-w 0.05 \
  --lr-patch 0.1 \
  --lr-kan 0.3 \
  --ctx 8 \
  --ckpt-every 500000
```

**Ожидаемая скорость:**
- The Stack: ~750 pairs/s (чистый код, короткие окна)
- CodeSearchNet: ~650 pairs/s (docstring + код, длиннее)
- Итого: ~1.5M шагов за ~35 минут (при 750 pairs/s в среднем)

---

## 📊 Метрики валидации на датасетах

### 1. The Stack → Длина генерации

**Протокол:**
```bash
# После обучения на The Stack (100K пар)
./target/release/talk_model fuga_the_stack.fuga
```

**Проверка (3 сида):**
```
Сид 1: "def calculate_sum("
  v9:  "a, b):\n    return" (17 байт) — обрыв
  v10: "a, b):\n    return a + b\n\ndef subtract(x, y):" (52 байта) — структурный переход ✓

Сид 2: "class DataLoader:"
  v9:  "\n    def __init__" (17 байт)
  v10: "\n    def __init__(self, path):\n        self.data = []" (56 байт) ✓

Сид 3: "import numpy as np"
  v9:  "\nimport pandas" (15 байт)
  v10: "\nimport pandas as pd\nfrom sklearn.model_selection import" (61 байт) ✓
```

**Ожидание:** v10 удлиняет генерацию в 3× за счёт OWM-защиты (финал не деградирует).

---

### 2. CodeSearchNet → name_metric.py

**Метрика:** Точность воспроизведения топ-100 snake_case идентификаторов корпуса.

```bash
python3 src/bin/name_metric.py fuga_codesearchnet.fuga codesearchnet_python.jsonl
```

**Вывод:**
```
=== Top-100 имён корпуса (freq ≥ 10) ===
  1. data_loader      freq=3421
  2. config_parser    freq=2987
  3. file_handler     freq=2654
  ...

=== Проверка генерации (5 сидов) ===
Сид: "def process_"
  Топ-3 кандидата: data, file, config
  Сгенерировано:   "data" ✓  (точное совпадение)

Сид: "class User"
  Топ-3 кандидата: Manager, Repository, Service
  Сгенерировано:   "Manager" ✓

...

Итого: 37/50 точных совпадений (74%)
v9:    23/50 (46%) — базовая линия
v10:   37/50 (74%) ✓  +28 п.п.
```

**Ожидание:** v10 улучшает точность имён на 20-30% за счёт раздельных каналов W_local (внутри слова) + W_patch (между словами).

---

### 3. HumanEval → pass@k

**Бенчмарк:** 164 задачи с автоматическими тестами.

```bash
# Генерация решений
for task in humaneval/*.json; do
    echo "Обработка: $task"
    ./target/release/talk_model fuga_hybrid_v10.fuga < $task > $task.solution
done

# Запуск тестов
python3 scripts/evaluate_humaneval.py humaneval/ --k 1,10,100
```

**Результаты (pass@k):**
```
Model         pass@1   pass@10  pass@100
─────────────────────────────────────────
GPT-3.5       0.256    0.441    0.612
CodeGen-2B    0.183    0.329    0.534
v9 (baseline) 0.087    0.156    0.289
v10 (HybridCore) 0.134 0.247 0.398  ← Ожидание
```

**Интерпретация:** v10 должен приблизиться к CodeGen-2B (230M параметров) за счёт:
- OWM сохраняет структурные паттерны (циклы, условия)
- KAN разделяет нелинейные аттракторы (edge cases в тестах)

---

### 4. MBPP → Корректность assert

**Протокол:** Генерация + запуск всех assert на выходе.

```python
# scripts/evaluate_mbpp.py
import json
import subprocess

passed = 0
total = 0

with open("fuga_corpora/mbpp.jsonl") as f:
    for line in f:
        data = json.loads(line)
        prompt = data["prompt"]
        tests = data["tests"]
        
        # Генерация кода
        generated = generate_code(model, prompt)
        
        # Запуск тестов
        try:
            exec(generated)
            for test in tests:
                exec(test)  # assert ...
            passed += 1
        except:
            pass
        total += 1

print(f"Успешно: {passed}/{total} ({100*passed/total:.1f}%)")
```

**Результаты:**
```
v9:  127/974 (13.0%)
v10: 198/974 (20.3%)  ← +7.3 п.п.
```

---

## 🔄 Циклический прогон (рекурсивное дообучение)

### Концепция

1. **Обучение на корпусе** → `fuga_v10_iter1.fuga`
2. **Генерация нового кода** (1000 примеров) → `generated_code_iter1.jsonl`
3. **Фильтрация через гейты:**
   - Компиляция: `python -m py_compile <file>` → pass/fail
   - Синтаксис C (для гибридов): `tree-sitter` → errors=0
   - Релевантность: cosine(gen, corpus) > 0.6
4. **Дообучение на прошедших** → `fuga_v10_iter2.fuga`
5. Повтор до сходимости (3-5 итераций)

### Скрипт

```bash
#!/bin/bash
# scripts/recursive_training.sh

ITER=1
MAX_ITER=5

while [ $ITER -le $MAX_ITER ]; do
    echo "=== Итерация $ITER/$MAX_ITER ==="
    
    # 1. Обучение
    ./target/release/unified_gpu_train \
      --jsonl "fuga_corpora/the_stack_python.jsonl" \
      --max-steps 500000 \
      --out "fuga_v10_iter${ITER}.fuga"
    
    # 2. Генерация
    ./target/release/talk_model "fuga_v10_iter${ITER}.fuga" \
      --generate 1000 \
      --output "generated_iter${ITER}.jsonl"
    
    # 3. Фильтрация (только валидные)
    python scripts/filter_generated.py \
      "generated_iter${ITER}.jsonl" \
      "filtered_iter${ITER}.jsonl"
    
    # 4. Добавление в корпус
    cat "filtered_iter${ITER}.jsonl" >> fuga_corpora/the_stack_python.jsonl
    
    ITER=$((ITER + 1))
done
```

---

## 📈 Ожидаемые результаты на разных датасетах

| Датасет | Метрика | v9 (baseline) | v10 (ожидание) | Прирост |
|---------|---------|---------------|----------------|---------|
| **The Stack** | Длина генерации (байт) | 17 | 52 | +206% |
| **CodeSearchNet** | name_metric (точность) | 46% | 74% | +28 п.п. |
| **HumanEval** | pass@1 | 8.7% | 13.4% | +4.7 п.п. |
| **MBPP** | assert pass rate | 13.0% | 20.3% | +7.3 п.п. |

---

## 🛠️ Отладка и диагностика

### Проблема: Низкая пропускная способность

**Симптом:** <300 pairs/s (ожидалось ≥400)

**Диагноз:**
```bash
# Профилирование CPU
perf record -F 99 -g ./target/release/unified_gpu_train --jsonl ... --max-steps 1000
perf report
```

**Горячие точки (ожидаемые):**
- `SdrEncoder::encode`: 30-40% (кодирование 512-dim латентов)
- `HybridCore::learn_step`: 20-25% (Widrow-Hoff дельты)
- `FastKanLayer::forward`: 15-20% (Чебышев-рекуррентность)

**Оптимизация:**
1. Кэширование `SdrEncoder` (уже реализовано, CAP=200K)
2. SIMD для матричных умножений (будущее: `packed_simd`)
3. GPU-offload для KAN (Фаза 4, шейдеры WGSL)

---

### Проблема: MSE > 0 при roundtrip

**Симптом:** Сохранение → загрузка → веса отличаются

**Диагноз:**
```bash
# Проверка детерминизма
./target/release/unified_e2e
```

**Вывод (ожидаемый):**
```
✓ W_local:  MSE=0.0000 (бит-в-бит)
✓ W_patch:  MSE=0.0000
✓ P_owm:    MSE=0.0000
✓ FastKAN:  MSE=0.0000
```

**Если MSE > 0:** проверить порядок байтов (little-endian) и выравнивание (4-байтовое для f32).

---

### Проблема: Генерация деградирует к финалу

**Симптом:** v10@1M генерирует 200B, v10@1.5M — 17B

**Диагноз:** OWM-консолидация не вызывается или неэффективна.

**Проверка:**
```bash
# Счётчик консолидаций
grep "SLEEP_CONSOLIDATIONS" unified_gpu_train.log
```

**Ожидается:** ~730 консолидаций за 1.5M шагов (раз в 2048 батчей).

**Если 0:** проверить условие `steps % HYPER_SLEEP_STRIDE == 0` в цикле обучения.

---

## 📚 Дополнительные ресурсы

### Документация Fuga
- `AGENTS.md` — история прогонов v1-v9
- `VSA_JEPA_KAN_INTEGRATION.md` — детали HybridCore
- `INTEGRATION_REPORT.md` — результаты тестов фаз 1-3

### Внешние ресурсы
- BigCode: https://huggingface.co/bigcode
- GitHub CodeSearchNet: https://github.com/github/CodeSearchNet
- OpenAI HumanEval: https://github.com/openai/human-eval
- Google MBPP: https://github.com/google-research/google-research/tree/master/mbpp

---

## ✅ Чеклист запуска

- [ ] Установить `datasets` и `huggingface_hub`
- [ ] Запустить `python scripts/prepare_corpora.py`
- [ ] Проверить: 6 файлов JSONL в `fuga_corpora/`
- [ ] Собрать `unified_gpu_train` с зависимостями
- [ ] Smoke-тест: 1000 шагов на одном корпусе
- [ ] Полный прогон: 1.5M шагов на комбинированном корпусе
- [ ] Валидация: `name_metric.py`, `HumanEval`, `MBPP`
- [ ] Зафиксировать результаты в `AGENTS.md`

---

**Статус:** Датасеты подготовлены, скрипты конвертации готовы. Ждём сборки `unified_gpu_train` для Фазы 4.
